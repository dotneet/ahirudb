//! テーブルフォーマットの抽象化。
//!
//! 実行エンジンはこのトレイト越しにしかデータ源を見ない。Parquet 固有の概念
//! （RowGroup、列チャンク、Thrift 統計）は `format::parquet` の内側に閉じる。
//!
//! ## 分割 (split) という単位
//!
//! フォーマットの違いを吸収する鍵は「分割」という 1 つの概念に集約すること。
//!
//! | フォーマット | 分割の実体 | 統計 | 射影で減るバイト |
//! |---|---|---|---|
//! | Parquet | RowGroup | あり | 減る（列チャンク単位で取る） |
//! | CSV / JSONL | 固定長バイトチャンク | なし | 減らない（行指向なので全部読む） |
//!
//! DESIGN.md §6 の「RowGroup 境界 I/O バリア」は、正確には**分割境界**バリア
//! である。分割の開始時点で必要なバイト範囲が確定することだけが要件で、
//! Parquet であることは要件ではない。だから CSV でも同じ実行ループが使える。
//!
//! ## 射影の扱いが 2 段階に分かれる理由
//!
//! `split_ranges` に射影を渡すのは、列指向フォーマットが**取得するバイト自体を
//! 減らせる**ため。行指向フォーマットは射影を渡されても全バイトを読むしかない
//! が、`read_split` 側で不要な列の変換を省ける。この 2 段階を 1 つの引数で
//! 表現しておくと、呼び出し側はフォーマットの性質を知らずに済む。

pub mod parquet;

#[cfg(feature = "csv")]
pub mod csv;

#[cfg(feature = "jsonl")]
pub mod jsonl;

use crate::catalog::Source;
use crate::prelude::*;
use crate::rt::hash::eq_ascii_ci;
use crate::vector::{Field, Value, Vector};

/// 統計による枝刈りに使える単純な範囲述語。
///
/// `WHERE` から抽出できた `列 <op> 定数` の形だけを持つ。
/// `column` は**射影後の列番号**（= スキャン出力での位置）を指す。
pub struct Pruner {
    pub column: usize,
    pub op: PruneOp,
    pub value: Value,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum PruneOp {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

/// 統計の `[min, max]` がこの述語を満たしうるか。偽なら分割を丸ごと飛ばせる。
///
/// 判断できないときは必ず `true` を返す（読む側に倒す）。枝刈りの誤りは
/// 行が消えるという最悪の壊れ方をするので、安全側の定義を 1 か所に固定する。
pub fn range_may_match(p: &Pruner, min: &Value, max: &Value) -> bool {
    use core::cmp::Ordering::*;
    let (cmp_min, cmp_max) = match (min.partial_cmp_same(&p.value), max.partial_cmp_same(&p.value))
    {
        (Some(a), Some(b)) => (a, b),
        // 比較できない（型が違う・NULL）なら枝刈りしない。
        _ => return true,
    };
    match p.op {
        PruneOp::Eq => cmp_min != Greater && cmp_max != Less,
        PruneOp::Lt => cmp_min == Less,
        PruneOp::Le => cmp_min != Greater,
        PruneOp::Gt => cmp_max == Greater,
        PruneOp::Ge => cmp_max != Less,
    }
}

/// スキーマ解決の結果。バイトが足りなければ必要な範囲を返す。
pub type ResolveStep = core::result::Result<(), (u64, u64)>;

/// ホストに展開を委譲する圧縮ブロック。
///
/// wasm コアが内蔵しないコーデック（GZIP / ZSTD）はホスト側で展開する
/// （DESIGN.md §6）。GZIP はブラウザの `DecompressionStream`、ZSTD は別の
/// wasm モジュールが処理する。エンジンから見ればどちらも同じ「ホストに
/// 頼む作業」なので、1 つの経路にまとめてある。
#[derive(Clone, Copy)]
pub struct CodecTask {
    pub codec: crate::parquet::Compression,
    /// 圧縮データ本体のファイル上の位置と長さ。キャッシュのキーでもある。
    pub offset: u64,
    pub len: u32,
    /// 展開後サイズ。ホストはこれを超える出力を返してはならない。
    pub out_len: u32,
}

/// フォーマット非依存のテーブル読み取り。
pub trait TableFormat {
    /// スキーマを解決する。I/O は行わず、足りない範囲を要求して戻る。
    /// 同じ範囲が満たされた状態で呼び直されたら前進しなければならない。
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep>;

    fn is_resolved(&self) -> bool;

    /// 解決済みの列。`resolve` が `Ok(Ok(()))` を返す前は空。
    fn schema(&self) -> &[Field];

    /// スキャン分割の総数。
    fn num_splits(&self) -> usize;

    /// 分割の行数。事前に分からないフォーマットは `None`。
    /// 結合順序の見積りと進捗表示にしか使わないので、精度は問わない。
    fn split_rows(&self, split: usize) -> Option<u64>;

    /// 分割を読むのに必要なバイト範囲を `out` に積む。
    ///
    /// `projection` はスキーマ上の列番号。列指向フォーマットはこれを使って
    /// 取得範囲を絞る。行指向フォーマットは無視してよい。
    fn split_ranges(
        &self,
        split: usize,
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()>;

    /// この分割の復号に必要な、ホスト側での展開作業を列挙する。
    ///
    /// `split_ranges` が示したバイトが揃った**後**に呼ばれる。ページヘッダは
    /// 非圧縮なので、この時点で必要な作業をすべて確定できる。実行の途中で
    /// 止まらずに済むのはこの性質のおかげ（DESIGN.md §6）。
    ///
    /// 内蔵コーデックしか使わないフォーマットは既定実装のままでよい。
    fn codec_tasks(
        &self,
        _src: &Source,
        _split: usize,
        _projection: &[usize],
        _out: &mut Vec<CodecTask>,
    ) -> Result<()> {
        Ok(())
    }

    /// 統計でこの分割を落とせるか。統計を持たないフォーマットは既定実装のまま
    /// `true` を返せばよい。
    ///
    /// `pruners` の `column` は `projection` 上の位置を指す。
    fn may_match(&self, _split: usize, _pruners: &[Pruner], _projection: &[usize]) -> bool {
        true
    }

    /// 分割を復号して列ベクタを返す。
    ///
    /// 返す列は `projection` と同じ順・同じ個数で、すべて同じ長さでなければ
    /// ならない。呼び出し側は `split_ranges` が示した範囲が `src` に揃っている
    /// ことを保証する。
    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>>;
}

/// 対応フォーマット。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
pub enum FormatKind {
    /// 名前（ファイル名・URL）の拡張子から推定する。
    Auto,
    Parquet,
    Csv,
    /// タブ区切り。CSV と同じ実装を区切り文字違いで使う。
    Tsv,
    /// 1 行 1 JSON オブジェクト（NDJSON）。
    Jsonl,
}

impl FormatKind {
    /// 名前の拡張子からフォーマットを推定する。
    ///
    /// URL のクエリ文字列とフラグメントは落としてから見る。判定できない場合は
    /// Parquet とみなす（主対象フォーマットであるため）。誤判定した場合は
    /// マジックバイトの検査ではなくフッタ解決の失敗として現れ、
    /// `BadMagic` が返る。
    pub fn detect(name: &str) -> FormatKind {
        let path = name.split(['?', '#']).next().unwrap_or(name);
        let ext = match path.rfind('.') {
            Some(i) => &path[i + 1..],
            None => return FormatKind::Parquet,
        };
        let e = ext.as_bytes();
        if eq_ascii_ci(e, b"csv") {
            FormatKind::Csv
        } else if eq_ascii_ci(e, b"tsv") || eq_ascii_ci(e, b"tab") {
            FormatKind::Tsv
        } else if eq_ascii_ci(e, b"jsonl") || eq_ascii_ci(e, b"ndjson") {
            FormatKind::Jsonl
        } else {
            FormatKind::Parquet
        }
    }
}

/// フォーマット実装を作る。
///
/// `Auto` は `name` から推定する。未対応（フィーチャ無効）のフォーマットは
/// `UnsupportedFeature` を返す。黙って Parquet として読もうとするより、
/// 対応していないと言う方がよい。
pub fn make(kind: FormatKind, name: &str) -> Result<Box<dyn TableFormat>> {
    let kind = match kind {
        FormatKind::Auto => FormatKind::detect(name),
        k => k,
    };
    match kind {
        FormatKind::Parquet => Ok(Box::new(parquet::ParquetFormat::new())),
        #[cfg(feature = "csv")]
        FormatKind::Csv => Ok(Box::new(csv::CsvFormat::new(b','))),
        #[cfg(feature = "csv")]
        FormatKind::Tsv => Ok(Box::new(csv::CsvFormat::new(b'\t'))),
        #[cfg(feature = "jsonl")]
        FormatKind::Jsonl => Ok(Box::new(jsonl::JsonlFormat::new())),
        #[allow(unreachable_patterns)]
        _ => err!(UnsupportedFeature),
    }
}

/// 行指向フォーマットが分割を切るときのチャンクサイズ。
///
/// 大きすぎると 1 分割のメモリが膨らみ、小さすぎるとレンジ取得の往復が増える。
/// Parquet の典型的な RowGroup（数十 MB）より小さめに寄せてある。
#[cfg(any(feature = "csv", feature = "jsonl"))]
pub const TEXT_SPLIT_BYTES: u64 = 8 * 1024 * 1024;

/// 行指向フォーマットで 1 レコードが分割境界をまたぐときに、次の改行を探して
/// 読み越してよい最大バイト数。
#[cfg(any(feature = "csv", feature = "jsonl"))]
pub const TEXT_MAX_RECORD: u64 = 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_detection() {
        assert_eq!(FormatKind::detect("a.parquet"), FormatKind::Parquet);
        assert_eq!(FormatKind::detect("a.CSV"), FormatKind::Csv);
        assert_eq!(FormatKind::detect("a.tsv"), FormatKind::Tsv);
        assert_eq!(FormatKind::detect("a.jsonl"), FormatKind::Jsonl);
        assert_eq!(FormatKind::detect("a.ndjson"), FormatKind::Jsonl);
        // 拡張子が無い・未知のものは Parquet 扱い。
        assert_eq!(FormatKind::detect("data"), FormatKind::Parquet);
        assert_eq!(FormatKind::detect("a.bin"), FormatKind::Parquet);
    }

    #[test]
    fn url_query_and_fragment_are_ignored() {
        assert_eq!(FormatKind::detect("https://x/y/trips.csv?token=abc"), FormatKind::Csv);
        assert_eq!(FormatKind::detect("https://x/y/trips.jsonl#frag"), FormatKind::Jsonl);
        // クエリ側の拡張子に釣られないこと。
        assert_eq!(FormatKind::detect("https://x/y/data.parquet?name=a.csv"), FormatKind::Parquet);
    }

    #[test]
    fn pruning_is_safe_when_statistics_are_unusable() {
        let p = Pruner { column: 0, op: PruneOp::Eq, value: Value::I64(1) };
        // 型が噛み合わない統計では枝刈りしない。
        assert!(range_may_match(&p, &Value::Bytes(vec![]), &Value::Bytes(vec![])));
        assert!(range_may_match(&p, &Value::Null, &Value::Null));
    }

    #[test]
    fn pruning_boundaries() {
        let gt = Pruner { column: 0, op: PruneOp::Gt, value: Value::I64(100) };
        assert!(!range_may_match(&gt, &Value::I64(0), &Value::I64(100)));
        assert!(range_may_match(&gt, &Value::I64(0), &Value::I64(101)));

        let ge = Pruner { column: 0, op: PruneOp::Ge, value: Value::I64(100) };
        assert!(range_may_match(&ge, &Value::I64(0), &Value::I64(100)));
        assert!(!range_may_match(&ge, &Value::I64(0), &Value::I64(99)));

        let eq = Pruner { column: 0, op: PruneOp::Eq, value: Value::I64(100) };
        assert!(range_may_match(&eq, &Value::I64(100), &Value::I64(100)));
        assert!(!range_may_match(&eq, &Value::I64(101), &Value::I64(200)));
    }
}
