//! セッション: カタログを持ち、SQL を受け取ってバッチを返す。
//!
//! 非同期 I/O は「実行を止めて必要なバイト範囲を返す」形で表現する。
//! Asyncify を使わずに済むので wasm のコードサイズが増えない（DESIGN.md §6）。

use crate::catalog::{Catalog, Source, TablePart};
use crate::exec::{build, CodecRequest, ExecContext, IoRequest, Operator, Step, Values};
use crate::expr::vm::Vm;
use crate::format::{partitioned::PartitionedFormat, FormatKind, TableFormat};
use crate::plan::bind::{
    bind_query, desugar_pivot, desugar_unpivot, referenced_in_query, resolve_from,
};
use crate::prelude::*;
use crate::sql::ast::{FromItem, Stmt};
use crate::sql::parse;
use crate::vector::{Batch, Field, Ty, Value, Vector};

/// `COPY` が書き出したバイト列と、書き込み先として指定されたパス。
///
/// `ahiru-core` は `no_std` でファイルシステムに触れられないので、実際に
/// `path` へ `data` を書き込むのは呼び出し側（ネイティブなら `ahiru-cli`）の
/// 役目（`write` モジュール doc、DESIGN.md §15 参照）。
#[cfg(feature = "export")]
pub struct CopyResult {
    pub path: String,
    pub data: Vec<u8>,
}

/// 準備済みクエリ。
pub struct Query {
    root: Box<dyn Operator>,
    pub schema: Vec<Field>,
    /// `COPY` の実行結果（`export` フィーチャ）。`Some` のときは `root`/
    /// `schema` は意味を持たない空プレースホルダで、`step` を呼ぶ必要は
    /// ない。実データはここに入っている。
    #[cfg(feature = "export")]
    pub copy: Option<CopyResult>,
}

impl Query {
    /// あらかじめ確定している 1 バッチだけの結果を組み立てる。
    /// `ddl`/`dml` が完了通知（影響行数など）を返すのに使う
    /// （`SHOW TABLES`/`DESCRIBE` の `one_column`/`describe_result` と同じ
    /// 手口）。
    #[cfg(feature = "ddl")]
    pub(crate) fn single_batch(schema: Vec<Field>, batch: Batch) -> Self {
        Query {
            root: Box::new(Values::new(batch)),
            schema,
            #[cfg(feature = "export")]
            copy: None,
        }
    }

    /// `COPY` の結果を保持するだけの `Query` を組み立てる。
    /// `write::copy` から呼ばれる。
    #[cfg(feature = "export")]
    pub(crate) fn copy_result(path: String, data: Vec<u8>) -> Self {
        Query {
            root: Box::new(Values::new(Batch::new(Vec::new()))),
            schema: Vec::new(),
            copy: Some(CopyResult { path, data }),
        }
    }
}

/// `Session::prepare` の結果。
pub enum Prepared {
    Ready(Query),
    /// フッタを読むためにバイトが足りない。
    NeedIo(Vec<IoRequest>),
}

/// `Session::step` の結果。
pub enum QueryStep {
    Batch(Batch),
    NeedIo(Vec<IoRequest>),
    /// 内蔵していないコーデックの展開をホストに依頼する（DESIGN.md §6）。
    NeedCodec(Vec<CodecRequest>),
    Done,
}

pub struct Session {
    pub catalog: Catalog,
    /// `pub(crate)`: `ddl`/`dml` モジュールが行単位の式評価（VALUES/SET/WHERE）
    /// に使う。クレート外には出さない（既存の ABI/JS 面には影響しない）。
    pub(crate) vm: Vm,
    /// `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()` 用のクエリ開始時刻
    /// （エポックからのマイクロ秒、UTC）。wasm コアは時計を持たないので
    /// ホストが `set_now` で明示的に渡す。未設定ならエポック（1970-01-01）
    /// になる — 「時刻を知らないなら黙って嘘をつかず、分かりやすく壊れた
    /// 値を返す」という他の防御的パース方針と同じ考え方。
    now_micros: i64,
}

impl Session {
    pub fn new() -> Self {
        Session { catalog: Catalog::new(), vm: Vm::new(), now_micros: 0 }
    }

    /// クエリ開始時刻を設定する。次回以降の `prepare` で
    /// `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`CURRENT_TIME`/`now()`/`today()`
    /// の値として使われる。ホスト（JS/CLI）がクエリのたびに現在時刻で
    /// 呼ぶ想定（DESIGN.md §2「ホストでできることはホストでやる」）。
    pub fn set_now(&mut self, now_micros: i64) {
        self.now_micros = now_micros;
    }

    /// ファイル全体をメモリに持つテーブルを登録する。
    /// フォーマットは名前（拡張子）から推定する。
    pub fn register_bytes(&mut self, name: &str, bytes: Vec<u8>) -> Result<usize> {
        self.register_bytes_as(name, bytes, FormatKind::Auto)
    }

    pub fn register_bytes_as(
        &mut self,
        name: &str,
        bytes: Vec<u8>,
        kind: FormatKind,
    ) -> Result<usize> {
        self.catalog.register(name, Source::from_bytes(bytes), kind)
    }

    /// ホストがレンジ取得で供給するテーブルを登録する。I/O は発生しない。
    pub fn register_remote(&mut self, name: &str, total_len: u64) -> Result<usize> {
        self.register_remote_as(name, total_len, FormatKind::Auto)
    }

    pub fn register_remote_as(
        &mut self,
        name: &str,
        total_len: u64,
        kind: FormatKind,
    ) -> Result<usize> {
        self.catalog.register(name, Source::remote(total_len), kind)
    }

    /// 複数ファイルを 1 論理テーブルとして、バイト列を渡して登録する。
    ///
    /// `files` は `(パス, バイト列)` の並び。`path` はフォーマット自動判定
    /// （拡張子）と Hive パーティション列の抽出（`key=value` ディレクトリ）
    /// の両方に使う。パスに `key=value` セグメントが無いパートはそのまま、
    /// あるパートだけ `PartitionedFormat` でラップする — 全パートを一律に
    /// ラップすると、パーティションが無い多数派のケースで無駄な間接層が
    /// 常に挟まることになる。
    pub fn register_multi_bytes(
        &mut self,
        name: &str,
        files: Vec<(String, Vec<u8>)>,
        kind: FormatKind,
    ) -> Result<usize> {
        let files: Vec<(String, Source)> =
            files.into_iter().map(|(p, b)| (p, Source::from_bytes(b))).collect();
        self.register_multi(name, files, kind)
    }

    /// 複数ファイルを 1 論理テーブルとして、ホストのレンジ取得で登録する。
    /// `files` は `(パス, 総バイト長)` の並び。I/O は発生しない。
    pub fn register_multi_remote(
        &mut self,
        name: &str,
        files: Vec<(String, u64)>,
        kind: FormatKind,
    ) -> Result<usize> {
        let files: Vec<(String, Source)> =
            files.into_iter().map(|(p, len)| (p, Source::remote(len))).collect();
        self.register_multi(name, files, kind)
    }

    fn register_multi(
        &mut self,
        name: &str,
        files: Vec<(String, Source)>,
        kind: FormatKind,
    ) -> Result<usize> {
        ensure!(!files.is_empty(), Internal);
        let mut parts = Vec::with_capacity(files.len());
        for (path, source) in files {
            let inner = crate::format::make(kind, &path)?;
            let hive_cols = PartitionedFormat::parse_hive_path(&path);
            let format: Box<dyn TableFormat> = if hive_cols.is_empty() {
                inner
            } else {
                Box::new(PartitionedFormat::new(inner, hive_cols))
            };
            parts.push(TablePart { path, source, format });
        }
        self.catalog.register_multi(name, parts)
    }

    /// `NeedIo` で要求したバイト列を渡す。
    pub fn provide(&mut self, table: usize, part: usize, offset: u64, data: Vec<u8>) -> Result<()> {
        let t = match self.catalog.get_mut(table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        match t.parts.get_mut(part) {
            Some(p) => {
                p.source.insert(offset, data);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// SQL をプランに落とす。スキーマ未解決ならバイト範囲を要求して戻る。
    pub fn prepare(&mut self, sql: &str, params: &[Value]) -> Result<Prepared> {
        let mut parsed = parse(sql)?;
        // `CURRENT_DATE`/`CURRENT_TIMESTAMP`/`now()` 等は束縛前に定数へ
        // 置き換える（`sql::now` 参照）。SQL標準の「クエリ内で1回だけ評価
        // する」契約にも自然に一致する。
        crate::sql::substitute_now(&mut parsed.arena, self.now_micros);

        // `PIVOT`/`UNPIVOT` は構文糖衣なので、通常の `SELECT` へ展開してから
        // 下の分岐に合流させる（`plan::bind::desugar_pivot`/`desugar_unpivot`
        // 参照）。展開には（`GROUP BY` 省略時などに）対象表のスキーマが要る
        // ことがあるので、ここでスキーマ解決 → 展開まで済ませてしまい、
        // 下の大きな `match &parsed.stmt` には触れずに済ませる
        // （`Stmt::Pivot`/`Stmt::Unpivot` は `PivotStmt`/`UnpivotStmt` を
        // 所有権ごと消費するので、`&parsed.stmt` からの借用では作れない —
        // `mem::replace` で `parsed.stmt` だけを取り出す）。
        if matches!(parsed.stmt, Stmt::Pivot(_) | Stmt::Unpivot(_)) {
            let stmt = core::mem::replace(&mut parsed.stmt, Stmt::ShowTables);
            let q = match stmt {
                Stmt::Pivot(p) => {
                    let from_schema = if p.group_by.is_empty() {
                        match self.describe(&p.from)? {
                            Ok(f) => f,
                            Err(io) => return Ok(Prepared::NeedIo(io)),
                        }
                    } else {
                        Vec::new()
                    };
                    desugar_pivot(&mut parsed.arena, *p, &from_schema)?
                }
                Stmt::Unpivot(u) => {
                    let from_schema = match self.describe(&u.from)? {
                        Ok(f) => f,
                        Err(io) => return Ok(Prepared::NeedIo(io)),
                    };
                    desugar_unpivot(&mut parsed.arena, *u, &from_schema)?
                }
                _ => unreachable!(),
            };
            return self.prepare_query(&parsed.arena, &q, params);
        }

        match &parsed.stmt {
            Stmt::Select(q) => self.prepare_query(&parsed.arena, q, params),
            // EXPLAIN はプランを組んでからテキストに落とす。実行はしない。
            Stmt::Explain(q) => {
                if let Some(io) = self.resolve_query(&parsed.arena, q)? {
                    return Ok(Prepared::NeedIo(io));
                }
                let plan = bind_query(&self.catalog, &parsed.arena, q, params)?;
                let lines = crate::plan::explain::explain(&plan.root);
                Ok(Prepared::Ready(one_column("plan", lines)))
            }
            Stmt::Describe(from) => {
                let fields = match self.describe(from)? {
                    Ok(f) => f,
                    Err(io) => return Ok(Prepared::NeedIo(io)),
                };
                Ok(Prepared::Ready(describe_result(&fields)))
            }
            Stmt::ShowTables => Ok(Prepared::Ready(one_column("name", self.table_names()))),
            // 上の早期リターンで必ず消費済み（`Stmt::Pivot`/`Stmt::Unpivot` は
            // 展開してから `prepare_query` に合流するので、ここには来ない）。
            Stmt::Pivot(_) | Stmt::Unpivot(_) => unreachable!(),
            // DDL/DML は副作用（カタログの変更）を伴う一発実行の文で、
            // Volcano のストリーミング実行には乗らない。`ddl`/`dml` モジュール
            // がここで完結させ、結果は 1 行だけの `Query`（影響行数など）で
            // 返す（`export::export_all` と同じ「既存の公開経路を外側から叩く」
            // 発想だが、ここは Session の中なので直接呼ぶ）。
            #[cfg(feature = "ddl")]
            Stmt::CreateTable { name, or_replace, if_not_exists, columns, as_select } => {
                crate::ddl::create_table(
                    self,
                    &parsed.arena,
                    name,
                    *or_replace,
                    *if_not_exists,
                    columns,
                    as_select.as_deref(),
                    params,
                )
            }
            #[cfg(feature = "ddl")]
            Stmt::DropTable { name, if_exists } => crate::ddl::drop_table(self, name, *if_exists),
            #[cfg(feature = "ddl")]
            Stmt::AlterTable { name, action } => {
                crate::ddl::alter_table(self, &parsed.arena, name, action, params)
            }
            #[cfg(feature = "ddl")]
            Stmt::CreateView { name, or_replace, query_sql } => {
                crate::ddl::create_view(self, name, query_sql.clone(), *or_replace)
            }
            #[cfg(feature = "ddl")]
            Stmt::DropView { name, if_exists } => crate::ddl::drop_view(self, name, *if_exists),
            #[cfg(feature = "dml")]
            Stmt::Insert { table, columns, source } => {
                crate::dml::insert(self, &parsed.arena, table, columns, source, params)
            }
            #[cfg(feature = "dml")]
            Stmt::Update { table, assignments, filter } => {
                crate::dml::update(self, &parsed.arena, table, assignments, *filter, params)
            }
            #[cfg(feature = "dml")]
            Stmt::Delete { table, filter } => {
                crate::dml::delete(self, &parsed.arena, table, *filter, params)
            }
            // `COPY` も DDL/DML と同様に一発実行の文だが、副作用はカタログ
            // ではなくバイト列の組み立て。実際にファイルへ書くのは
            // `ahiru-core` の外（`write` モジュール doc、DESIGN.md §15）。
            #[cfg(feature = "export")]
            Stmt::Copy { query, path, format } => {
                crate::write::copy(self, &parsed.arena, query, path, format.as_deref(), params)
            }
        }
    }

    /// `SELECT` を束縛してプランに落とす。`Stmt::Select` と、AST を直接持つ
    /// 呼び出し元（`COPY` の内側クエリなど、SQL 文字列への往復を避けたい
    /// 場合。`write::export_query` 参照）の両方から使う共通経路。
    pub(crate) fn prepare_query(
        &mut self,
        arena: &crate::sql::ast::ExprArena,
        q: &crate::sql::ast::QueryStmt,
        params: &[Value],
    ) -> Result<Prepared> {
        if let Some(io) = self.resolve_query(arena, q)? {
            return Ok(Prepared::NeedIo(io));
        }
        let plan = bind_query(&self.catalog, arena, q, params)?;
        let schema = plan.root.schema().to_vec();
        Ok(Prepared::Ready(Query {
            root: build(plan.root)?,
            schema,
            #[cfg(feature = "export")]
            copy: None,
        }))
    }

    /// クエリが参照するテーブルのスキーマをすべて解決する。
    /// 足りない範囲があればまとめて返す（結合や集合演算、複数ファイルテーブル
    /// があると複数になる）。1 テーブルにつき 1 往復で済むよう、そのテーブルの
    /// 全パートが必要とする範囲を `Table::resolve` の時点で束ねてから積む
    /// （`catalog::Table::resolve` のドキュメント参照）。
    fn resolve_query(
        &mut self,
        arena: &crate::sql::ast::ExprArena,
        q: &crate::sql::ast::QueryStmt,
    ) -> Result<Option<Vec<IoRequest>>> {
        let mut tables = Vec::new();
        referenced_in_query(&self.catalog, arena, q, &mut tables, 0)?;
        let mut io = Vec::new();
        for &table in &tables {
            let t = match self.catalog.get_mut(table) {
                Some(t) => t,
                None => err!(TableNotFound),
            };
            if let Err(need) = t.resolve()? {
                for (part, offset, len) in need {
                    io.push(IoRequest { table, part, offset, len });
                }
            }
        }
        Ok(if io.is_empty() { None } else { Some(io) })
    }

    /// ホストが展開した圧縮ブロックを渡す。
    pub fn provide_decoded(
        &mut self,
        table: usize,
        part: usize,
        offset: u64,
        len: u32,
        data: Vec<u8>,
    ) -> Result<()> {
        let t = match self.catalog.get_mut(table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        match t.parts.get_mut(part) {
            Some(p) => {
                p.source.insert_decoded(offset, len, data);
                Ok(())
            }
            None => err!(TableNotFound),
        }
    }

    /// 次のバッチを取り出す。
    pub fn step(&mut self, q: &mut Query) -> Result<QueryStep> {
        let mut ctx = ExecContext {
            catalog: &mut self.catalog,
            vm: &mut self.vm,
            io: Vec::new(),
            codec: Vec::new(),
        };
        match q.root.next(&mut ctx)? {
            Step::Ready(b) => Ok(QueryStep::Batch(b)),
            Step::NeedIo => Ok(QueryStep::NeedIo(ctx.io)),
            Step::NeedCodec => Ok(QueryStep::NeedCodec(ctx.codec)),
            Step::Done => Ok(QueryStep::Done),
        }
    }

    /// テーブル名を列挙する（`SHOW TABLES`）。インメモリ表・ビューも含める
    /// （`ddl`）。
    pub fn table_names(&self) -> Vec<String> {
        #[allow(unused_mut)]
        let mut names: Vec<String> = self.catalog.names().map(String::from).collect();
        #[cfg(feature = "ddl")]
        {
            names.extend(self.catalog.mem_names().map(String::from));
            names.extend(self.catalog.view_names().map(String::from));
        }
        names
    }

    /// `DESCRIBE` 用。フッタが未解決なら要求を返す
    /// （複数ファイルテーブルなら複数パート分まとめて）。
    pub fn describe(
        &mut self,
        from: &FromItem,
    ) -> Result<core::result::Result<Vec<Field>, Vec<IoRequest>>> {
        let table = resolve_from(&self.catalog, from)?;
        let t = match self.catalog.get_mut(table) {
            Some(t) => t,
            None => err!(TableNotFound),
        };
        if let Err(need) = t.resolve()? {
            let io = need
                .into_iter()
                .map(|(part, offset, len)| IoRequest { table, part, offset, len })
                .collect();
            return Ok(Err(io));
        }
        Ok(Ok(t.schema().to_vec()))
    }
}

/// 文字列 1 列だけの結果を作る。`SHOW TABLES` と `EXPLAIN` 用。
fn one_column(name: &str, rows: Vec<String>) -> Query {
    let mut v = Vector::with_capacity(Ty::Varchar, rows.len());
    for r in &rows {
        v.push_value(&Value::Bytes(r.as_bytes().to_vec()));
    }
    let schema = vec![Field::new(name, Ty::Varchar, false)];
    Query {
        root: Box::new(Values::new(Batch::new(vec![v]))),
        schema,
        #[cfg(feature = "export")]
        copy: None,
    }
}

/// `DESCRIBE` の結果。列名・型・NULL 可否の 3 列。
fn describe_result(fields: &[Field]) -> Query {
    let mut names = Vector::with_capacity(Ty::Varchar, fields.len());
    let mut types = Vector::with_capacity(Ty::Varchar, fields.len());
    let mut nulls = Vector::with_capacity(Ty::Varchar, fields.len());
    for f in fields {
        names.push_value(&Value::Bytes(f.name.as_bytes().to_vec()));
        types.push_value(&Value::Bytes(f.ty.name().as_bytes().to_vec()));
        nulls.push_value(&Value::Bytes(if f.nullable { b"YES".to_vec() } else { b"NO".to_vec() }));
    }
    let schema = vec![
        Field::new("column_name", Ty::Varchar, false),
        Field::new("column_type", Ty::Varchar, false),
        Field::new("null", Ty::Varchar, false),
    ];
    Query {
        root: Box::new(Values::new(Batch::new(vec![names, types, nulls]))),
        schema,
        #[cfg(feature = "export")]
        copy: None,
    }
}

impl Default for Session {
    fn default() -> Self {
        Session::new()
    }
}
