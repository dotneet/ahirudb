//! 書き出し（`export` フィーチャ）。
//!
//! 読み取り経路（`Source` / `TableFormat`）には一切触れない。`Session` の
//! 既存の公開 API（`prepare` / `step`）を外側から叩くだけで組み立てている
//! ので、読み取り側の不変条件（バイト範囲は一度入ったら書き換わらない、
//! 分割境界でしか I/O を待たない）を壊しようがない。これが「オプトアウト
//! 可能」の中身: `export` フィーチャを外せばこのモジュールごと消え、
//! 他のどこにも影響が残らない（DESIGN.md §15）。
//!
//! ## v1 の制限: 非再開設計
//!
//! 読み取りエンジンの中核は「バイトが足りなければ止めて要求を返す」設計
//! （DESIGN.md §6）だが、この書き出しドライバはその型を露出していない。
//! `export_all` はクエリの実行中に `NEED_IO` / `NEED_CODEC` が発生したら
//! `Err(IoFailed)` で失敗する。全データがメモリ上にある場合（CLI での
//! 利用、または JS 側が事前にテーブルを完全に取得している場合）にしか
//! 使えない。ホストの fetch ループと協調する再開可能な書き出しは、
//! `ahiru_query_step` と同じ形の ABI を書き出し版にも用意する必要がある
//! ため、v1 では見送っている。

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "jsonl")]
pub mod jsonl;
#[cfg(feature = "export-parquet")]
pub mod parquet;

use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::session::{Prepared, Query, QueryStep, Session};
use crate::sql::ast::{ExprArena, QueryStmt};
use crate::vector::{Batch, Field, Value};

/// 出力フォーマット。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum ExportFormat {
    #[cfg(feature = "csv")]
    Csv,
    #[cfg(feature = "jsonl")]
    Jsonl,
    #[cfg(feature = "export-parquet")]
    Parquet,
}

/// 書き出し先。`Batch` を受け取ってバイト列に変換するだけの薄い抽象。
///
/// 読み取り側の `TableFormat` と対称の設計（`Batch` を produce する側と
/// consume する側）。コア型（`Batch`/`Vector`/`Field`）は共有するが、
/// 読み取り側の型には一切依存しない。
pub trait TableSink {
    /// ヘッダ相当の情報。実装は必要なら最初のバイト列をここで書いてよい。
    fn begin(&mut self, schema: &[Field]) -> Result<()>;
    /// 1 バッチぶんの行を書く。selection は解決済み（`materialize` 後）を渡す。
    fn write_batch(&mut self, schema: &[Field], batch: &Batch) -> Result<()>;
    /// 末尾処理（フッタや閉じ括弧）をして完成したバイト列を返す。
    fn finish(&mut self) -> Result<Vec<u8>>;
}

/// クエリを実行し、結果を丸ごと `sink` に書き出す。
///
/// **非再開**: 実行中に `NEED_IO` / `NEED_CODEC` が起きたら `IoFailed` で
/// 失敗する（モジュール doc 参照）。
pub fn export_all(
    session: &mut Session,
    sql: &str,
    params: &[Value],
    sink: &mut dyn TableSink,
) -> Result<Vec<u8>> {
    let mut q = match session.prepare(sql, params)? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => err!(IoFailed),
    };
    sink.begin(&q.schema)?;
    loop {
        match session.step(&mut q)? {
            QueryStep::Batch(mut b) => {
                // selection を解決してから渡す。sink 側は密な列だけを見ればよい。
                b.materialize();
                sink.write_batch(&q.schema, &b)?;
            }
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => err!(IoFailed),
            QueryStep::Done => break,
        }
    }
    sink.finish()
}

/// `export_all` の「既にパース済みの `QueryStmt` から」版。
///
/// `Stmt::Copy` は `Session::prepare` の中で既に `ExprArena`/`QueryStmt` を
/// 持っている。`export_all` に渡すために木をわざわざ SQL テキストへ戻す
/// 必要が出ないよう、入力だけ AST 版にした双子の関数を用意する。`Session`
/// の `prepare`/`step` という公開 API 経由で組み立てる点は `export_all` と
/// 同じ（モジュール doc の「独立性」参照。`plan::bind`/`exec::build` には
/// 一切触れない）。
///
/// **非再開**: `export_all` と同じ制約（モジュール doc 参照）。
fn export_query(
    session: &mut Session,
    arena: &ExprArena,
    query: &QueryStmt,
    params: &[Value],
    sink: &mut dyn TableSink,
) -> Result<Vec<u8>> {
    let mut q = match session.prepare_query(arena, query, params)? {
        Prepared::Ready(q) => q,
        Prepared::NeedIo(_) => err!(IoFailed),
    };
    sink.begin(&q.schema)?;
    loop {
        match session.step(&mut q)? {
            QueryStep::Batch(mut b) => {
                b.materialize();
                sink.write_batch(&q.schema, &b)?;
            }
            QueryStep::NeedIo(_) | QueryStep::NeedCodec(_) => err!(IoFailed),
            QueryStep::Done => break,
        }
    }
    sink.finish()
}

/// `Stmt::Copy` の実行本体。`Session::prepare` から呼ばれる。
///
/// The format comes from an explicit `FORMAT csv|jsonl|json|parquet` when
/// given, and otherwise from the extension of `path` via
/// `format::FormatKind::detect`. Which formats can actually be written
/// depends on the enabled features (`csv`, `jsonl`, `export-parquet`);
/// anything else is `UnsupportedFeature`.
///
/// **ファイルには書かない**: 結果は `Query::copy_result` に包んで返す。
/// 実際に `path` へ書き込むのは呼び出し側（ネイティブなら `ahiru-cli`）の
/// 役目（モジュール doc、DESIGN.md §15 参照）。
pub(crate) fn copy(
    session: &mut Session,
    arena: &ExprArena,
    query: &QueryStmt,
    path: &str,
    format: Option<&str>,
    params: &[Value],
) -> Result<Prepared> {
    let fmt = resolve_format(path, format)?;
    let data = match fmt {
        #[cfg(feature = "csv")]
        ExportFormat::Csv => {
            let mut sink = csv::CsvSink::new();
            export_query(session, arena, query, params, &mut sink)?
        }
        #[cfg(feature = "jsonl")]
        ExportFormat::Jsonl => {
            let mut sink = jsonl::JsonlSink::new();
            export_query(session, arena, query, params, &mut sink)?
        }
        #[cfg(feature = "export-parquet")]
        ExportFormat::Parquet => {
            let mut sink = parquet::ParquetSink::new();
            export_query(session, arena, query, params, &mut sink)?
        }
    };
    Ok(Prepared::Ready(Query::copy_result(path.to_owned(), data)))
}

/// `format` があればそれを解決し、無ければ `path` の拡張子から推定する。
fn resolve_format(path: &str, format: Option<&str>) -> Result<ExportFormat> {
    match format {
        Some(f) => format_by_name(f),
        // `FormatKind::Csv`/`Tsv` は CSV シンクへ、`Jsonl`/`Json` は JSONL
        // シンクへ寄せる。DuckDB の `COPY ... (FORMAT JSON)` も配列では
        // なく改行区切りの JSON を書く（実測して確認済み）ので、読み取り側の
        // 「1 ファイル 1 JSON 値」の `Json` とは書き出し側で意味が分かれる
        // ことになるが、他に書き出し用の JSON 配列シンクを持たない v1 では
        // これが妥当な対応先。
        None => match crate::format::FormatKind::detect(path) {
            #[cfg(feature = "csv")]
            crate::format::FormatKind::Csv | crate::format::FormatKind::Tsv => {
                Ok(ExportFormat::Csv)
            }
            #[cfg(feature = "jsonl")]
            crate::format::FormatKind::Jsonl | crate::format::FormatKind::Json => {
                Ok(ExportFormat::Jsonl)
            }
            // `detect` resolves both `.parquet` and any unknown extension
            // to `Parquet`, mirroring the read side. Without
            // `export-parquet` there is no sink for it, so say so rather
            // than quietly picking some other format.
            #[cfg(feature = "export-parquet")]
            crate::format::FormatKind::Parquet => Ok(ExportFormat::Parquet),
            _ => err!(UnsupportedFeature),
        },
    }
}

/// `(FORMAT <name>)` の値を `ExportFormat` に対応付ける。
fn format_by_name(name: &str) -> Result<ExportFormat> {
    #[cfg(feature = "csv")]
    if eq_ascii_ci(name.as_bytes(), b"csv") {
        return Ok(ExportFormat::Csv);
    }
    #[cfg(feature = "jsonl")]
    if eq_ascii_ci(name.as_bytes(), b"jsonl") || eq_ascii_ci(name.as_bytes(), b"json") {
        return Ok(ExportFormat::Jsonl);
    }
    #[cfg(feature = "export-parquet")]
    if eq_ascii_ci(name.as_bytes(), b"parquet") {
        return Ok(ExportFormat::Parquet);
    }
    err!(UnsupportedFeature)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用の記録シンク。書き込まれた行を `Value` の行列として溜める。
    struct RecordSink {
        rows: Vec<Vec<Value>>,
        began: bool,
        finished: bool,
    }

    impl RecordSink {
        fn new() -> Self {
            RecordSink { rows: Vec::new(), began: false, finished: false }
        }
    }

    impl TableSink for RecordSink {
        fn begin(&mut self, _schema: &[Field]) -> Result<()> {
            self.began = true;
            Ok(())
        }
        fn write_batch(&mut self, schema: &[Field], batch: &Batch) -> Result<()> {
            for r in 0..batch.num_rows() {
                let mut row = Vec::with_capacity(schema.len());
                for c in &batch.cols {
                    row.push(c.value_at(r));
                }
                self.rows.push(row);
            }
            Ok(())
        }
        fn finish(&mut self) -> Result<Vec<u8>> {
            self.finished = true;
            Ok(Vec::new())
        }
    }

    #[test]
    #[cfg(feature = "csv")]
    fn drives_begin_write_finish_in_order() {
        // CSV は手で組み立てられるので、他のフォーマットのテストデータや
        // 別の実装に依存せずにこのモジュール単体でテストできる。
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n2\n3\n".to_vec(), crate::format::FormatKind::Csv)
            .unwrap();
        let mut sink = RecordSink::new();
        export_all(&mut s, "SELECT id FROM t ORDER BY id", &[], &mut sink).unwrap();
        assert!(sink.began);
        assert!(sink.finished);
        assert_eq!(sink.rows.len(), 3);
        assert_eq!(sink.rows[0][0], Value::I64(1));
        assert_eq!(sink.rows[2][0], Value::I64(3));
    }

    #[test]
    fn need_io_fails_clearly_for_unresolved_remote_table() {
        let mut s = Session::new();
        s.register_remote("t", 100).unwrap();
        let mut sink = RecordSink::new();
        let r = export_all(&mut s, "SELECT * FROM t", &[], &mut sink);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::IoFailed));
    }

    // --- COPY（`Session::prepare` 経由の統合テスト）--------------------------
    // パーサから `write::copy` まで通しで検証する。`ahiru-core` はファイルへは
    // 書かないので、ここで見るのは「正しいパスと正しいバイト列が `Query` に
    // 載って返るか」まで（実ファイルへの書き込み検証は `ahiru-cli` 側）。

    #[cfg(feature = "csv")]
    fn copy_ready(sql: &str) -> crate::session::CopyResult {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id,name\n2,b\n1,a\n".to_vec(), crate::format::FormatKind::Csv)
            .unwrap();
        match s.prepare(sql, &[]).unwrap() {
            Prepared::Ready(q) => q.copy.expect("COPY の結果が無い"),
            Prepared::NeedIo(_) => panic!("unexpected NeedIo"),
        }
    }

    #[test]
    #[cfg(feature = "csv")]
    fn copy_subquery_infers_csv_from_extension() {
        let r = copy_ready("COPY (SELECT id, name FROM t ORDER BY id) TO 'out.csv'");
        assert_eq!(r.path, "out.csv");
        assert_eq!(String::from_utf8(r.data).unwrap(), "id,name\n1,a\n2,b\n");
    }

    #[test]
    #[cfg(feature = "jsonl")]
    fn copy_explicit_format_overrides_extension() {
        let r = copy_ready("COPY (SELECT id, name FROM t ORDER BY id) TO 'out.dat' (FORMAT jsonl)");
        assert_eq!(r.path, "out.dat");
        let text = String::from_utf8(r.data).unwrap();
        assert_eq!(text, "{\"id\":1,\"name\":\"a\"}\n{\"id\":2,\"name\":\"b\"}\n");
    }

    #[test]
    #[cfg(feature = "jsonl")]
    fn copy_format_json_writes_ndjson_like_duckdb() {
        // DuckDB の `COPY ... (FORMAT JSON)` は JSON 配列ではなく改行区切り
        // (NDJSON) を書く（手元の duckdb CLI で実測して確認済み）。書き出し
        // 側の JSON 配列シンクを別途持たない v1 では、JSONL シンクへ寄せる
        // のがこの挙動に一致する。
        let r = copy_ready("COPY (SELECT id FROM t ORDER BY id) TO 'out.json'");
        assert_eq!(String::from_utf8(r.data).unwrap(), "{\"id\":1}\n{\"id\":2}\n");
    }

    #[test]
    #[cfg(feature = "csv")]
    fn copy_table_form_is_select_star_from_table() {
        let r = copy_ready("COPY t TO 'out.csv'");
        assert_eq!(String::from_utf8(r.data).unwrap(), "id,name\n2,b\n1,a\n");
    }

    /// Without `export-parquet` there is no Parquet sink, so both the
    /// extension route and the explicit `FORMAT parquet` route have to say
    /// so instead of quietly falling back to some other format.
    #[test]
    #[cfg(all(feature = "csv", not(feature = "export-parquet")))]
    fn copy_unsupported_format_is_rejected() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        let r = s.prepare("COPY (SELECT id FROM t) TO 'out.parquet'", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::UnsupportedFeature));

        let r = s.prepare("COPY (SELECT id FROM t) TO 'out.csv' (FORMAT parquet)", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::UnsupportedFeature));
    }

    /// A format name no build ever supports is rejected whatever features
    /// are on (the `export-parquet` counterpart of the test above, which
    /// can no longer use `parquet` as its example).
    #[test]
    #[cfg(feature = "csv")]
    fn copy_unknown_format_name_is_rejected() {
        let mut s = Session::new();
        s.register_bytes_as("t", b"id\n1\n".to_vec(), crate::format::FormatKind::Csv).unwrap();
        let r = s.prepare("COPY (SELECT id FROM t) TO 'out.csv' (FORMAT orc)", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::UnsupportedFeature));
    }

    #[test]
    #[cfg(feature = "csv")]
    fn copy_need_io_fails_clearly_for_unresolved_remote_table() {
        let mut s = Session::new();
        s.register_remote("t", 100).unwrap();
        let r = s.prepare("COPY (SELECT * FROM t) TO 'out.csv'", &[]);
        assert_eq!(crate::error::code_of(r), Some(crate::error::Code::IoFailed));
    }
}
