//! Hive スタイルのパーティション列を仮想的に付け足すデコレータ。
//!
//! `s3://bucket/year=2024/month=01/part.parquet` のようなパスの、ファイル名
//! より前のディレクトリ部分から `key=value` セグメントを取り出し、ファイルの
//! 中には存在しない列としてスキーマと読み出し結果に足し込む。
//!
//! `inner: Box<dyn TableFormat>` に丸ごと委譲する純粋なデコレータで、
//! `parquet.rs` / `csv.rs` / `jsonl.rs` は 1 行も変更しなくてよい。
//! これはこのファイルだけで完結させることが設計上の要点。

use alloc::string::ToString;

use crate::catalog::Source;
use crate::format::{range_may_match, CodecTask, Pruner, ResolveStep, TableFormat};
use crate::prelude::*;
use crate::vector::{Field, Ty, Value, Vector};

pub struct PartitionedFormat {
    inner: Box<dyn TableFormat>,
    /// `(列名, 定数値)`。ファイルのパスから 1 度だけ抽出しておく。
    partition_cols: Vec<(String, Value)>,
    /// `inner` のスキーマ + パーティション列。`resolve` が完了するまでは空。
    schema: Vec<Field>,
}

impl PartitionedFormat {
    pub fn new(inner: Box<dyn TableFormat>, partition_cols: Vec<(String, Value)>) -> Self {
        PartitionedFormat { inner, partition_cols, schema: Vec::new() }
    }

    /// パスのディレクトリ部分（ファイル名を除く各セグメント）から
    /// `key=value` を取り出す。見つからなければ空を返す
    /// （= Hive パーティションではない）。
    ///
    /// URL のクエリ文字列・フラグメントは `FormatKind::detect` と同じく
    /// 事前に落とす。`key` か `value` が空のセグメントは無視する
    /// （`=` を含むだけの飾りのディレクトリ名を誤検出しないため）。
    pub fn parse_hive_path(path: &str) -> Vec<(String, Value)> {
        let path = path.split(['?', '#']).next().unwrap_or(path);
        let segs: Vec<&str> = path.split('/').collect();
        if segs.len() < 2 {
            return Vec::new();
        }
        // 最後のセグメントはファイル名なので見ない。
        let mut out = Vec::new();
        for seg in &segs[..segs.len() - 1] {
            let Some(eq) = seg.find('=') else { continue };
            let (k, v) = (&seg[..eq], &seg[eq + 1..]);
            if k.is_empty() || v.is_empty() {
                continue;
            }
            out.push((k.to_string(), infer_value(&percent_decode(v))));
        }
        out
    }

    /// `inner` の列だけを残した射影。パーティション列（`inner` の列数以上の
    /// 添字）は実ファイルには存在しないので、`inner` には見せない。
    ///
    /// 1 本も残らない場合（パーティション列だけを選んだクエリ）でも、行数を
    /// 知るために `inner` の列を最低 1 本は要求する。`plan/bind.rs` が
    /// `COUNT(*)` 用にやっている「射影が空なら列 0 を足す」のと同じ発想を、
    /// ここでは `inner`/パーティション列の境界に対して適用している。
    fn inner_projection(&self, projection: &[usize]) -> Vec<usize> {
        let inner_n = self.inner.schema().len();
        let mut v: Vec<usize> = projection.iter().copied().filter(|&c| c < inner_n).collect();
        if v.is_empty() && inner_n > 0 {
            v.push(0);
        }
        v
    }

    /// `pruners`/`projection` を `inner` の列空間に付け替える。パーティション
    /// 列を触れる pruner は `inner` には見せられない（実ファイルにその列は
    /// 無い）ので、その場で判定してしまう — パーティション列はファイル全体で
    /// 1 定数なので、`min = max = その定数` として `range_may_match` に通せる。
    ///
    /// 戻り値は `(inner 射影, inner 向けに付け替えた pruners, 不一致確定)`。
    /// 3 つ目が `true` なら、パーティション列の pruner だけでこの分割は
    /// 丸ごと読み飛ばせることが分かっている（`inner` に問い合わせるまでもない）。
    ///
    /// `may_match` / `index_ranges` / `refine_with_index` の 3 か所で同じ
    /// 付け替えが要るので、ここに 1 度だけ書く。
    fn remap_pruners(
        &self,
        pruners: &[Pruner],
        projection: &[usize],
    ) -> (Vec<usize>, Vec<Pruner>, bool) {
        let inner_n = self.inner.schema().len();
        let inner_proj = self.inner_projection(projection);
        let mut inner_pruners: Vec<Pruner> = Vec::new();
        for p in pruners {
            let Some(&col) = projection.get(p.column) else { continue };
            if col < inner_n {
                if let Some(pos) = inner_proj.iter().position(|&c| c == col) {
                    inner_pruners.push(Pruner { column: pos, op: p.op, value: p.value.clone() });
                }
            } else {
                let (_, v) = &self.partition_cols[col - inner_n];
                if !range_may_match(p, v, v) {
                    return (inner_proj, inner_pruners, true);
                }
            }
        }
        (inner_proj, inner_pruners, false)
    }
}

/// パーティション値の `%XX` エスケープを元のバイトに戻す。Spark 等が
/// パーティション値にスペースや記号を含めるとき URL エンコードして書き出す
/// ため（`region=us%20east` のように）、DuckDB もここをデコードして返す
/// （`duckdb` CLI で実測して確認済み）。不正な `%` シーケンス（16進数でない、
/// 途中で文字列が終わる）はデコードを諦めてそのまま素通しする。
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    if !b.contains(&b'%') {
        return s.to_string();
    }
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(hi), Some(lo)) = (hex_val(b[i + 1]), hex_val(b[i + 2])) {
                out.push((hi << 4) | lo);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| s.to_string())
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// DuckDB の Hive パーティション型推定に合わせる: 数字だけの値は
/// INTEGER（32bit に収まらなければ BIGINT）、それ以外は VARCHAR。
/// 符号や小数点は「数字だけ」の対象外にしてある（`-1` や `1.5` を含む
/// パーティションは実運用でもまれで、誤って数値扱いにする方が危険）。
fn infer_value(s: &str) -> Value {
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(v) = s.parse::<i64>() {
            return match i32::try_from(v) {
                Ok(v32) => Value::I32(v32),
                Err(_) => Value::I64(v),
            };
        }
    }
    Value::Bytes(s.as_bytes().to_vec())
}

fn value_ty(v: &Value) -> Ty {
    match v {
        Value::I32(_) => Ty::Int,
        Value::I64(_) => Ty::BigInt,
        // `infer_value` はこれ以外を作らない。
        _ => Ty::Varchar,
    }
}

/// 定数値で埋めた `rows` 行のベクタを作る。
fn constant_vector(ty: Ty, v: &Value, rows: usize) -> Vector {
    let mut out = Vector::with_capacity(ty, rows);
    for _ in 0..rows {
        out.push_value(v);
    }
    out
}

impl TableFormat for PartitionedFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        let step = self.inner.resolve(src)?;
        if step.is_ok() && self.schema.is_empty() {
            let mut schema = self.inner.schema().to_vec();
            for (name, v) in &self.partition_cols {
                // パーティション列がファイル自身の列名と衝突すると、同じ名前の
                // 列が2本並ぶスキーマになる。どちらを引くかは列解決の実装
                // 詳細に依存してしまい、意図しない方の値が静かに返りかねない
                // ので、位置ずれ結合を拒否した `catalog::unify_schema` と同じ
                // 方針で明確に拒否する。
                ensure!(
                    !schema
                        .iter()
                        .any(|f| crate::rt::hash::eq_ascii_ci(f.name.as_bytes(), name.as_bytes())),
                    DuplicateColumn
                );
                schema.push(Field::new(name.clone(), value_ty(v), false));
            }
            self.schema = schema;
        }
        Ok(step)
    }

    fn is_resolved(&self) -> bool {
        self.inner.is_resolved()
    }

    fn schema(&self) -> &[Field] {
        &self.schema
    }

    fn num_splits(&self) -> usize {
        self.inner.num_splits()
    }

    fn split_rows(&self, split: usize) -> Option<u64> {
        self.inner.split_rows(split)
    }

    fn split_ranges(
        &self,
        split: usize,
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        self.inner.split_ranges(split, &self.inner_projection(projection), out)
    }

    fn codec_tasks(
        &self,
        src: &Source,
        split: usize,
        projection: &[usize],
        out: &mut Vec<CodecTask>,
    ) -> Result<()> {
        self.inner.codec_tasks(src, split, &self.inner_projection(projection), out)
    }

    fn may_match(&self, split: usize, pruners: &[Pruner], projection: &[usize]) -> bool {
        let (inner_proj, inner_pruners, reject) = self.remap_pruners(pruners, projection);
        if reject {
            return false;
        }
        self.inner.may_match(split, &inner_pruners, &inner_proj)
    }

    fn index_ranges(
        &self,
        split: usize,
        pruners: &[Pruner],
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        let (inner_proj, inner_pruners, reject) = self.remap_pruners(pruners, projection);
        if reject {
            // パーティション列の pruner だけで既に不一致が確定している。
            // ページ選択用のバイトすら要らない。
            return Ok(());
        }
        self.inner.index_ranges(split, &inner_pruners, &inner_proj, out)
    }

    fn refine_with_index(
        &mut self,
        src: &Source,
        split: usize,
        pruners: &[Pruner],
        projection: &[usize],
    ) -> Result<bool> {
        let (inner_proj, inner_pruners, reject) = self.remap_pruners(pruners, projection);
        if reject {
            return Ok(false);
        }
        self.inner.refine_with_index(src, split, &inner_pruners, &inner_proj)
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        let inner_n = self.inner.schema().len();
        let inner_proj = self.inner_projection(projection);
        let inner_cols = self.inner.read_split(src, split, &inner_proj)?;
        ensure!(inner_cols.len() == inner_proj.len(), Internal);
        let rows = inner_cols.first().map_or(0, |c| c.len());

        let mut out = Vec::with_capacity(projection.len());
        for &col in projection {
            if col < inner_n {
                let pos = match inner_proj.iter().position(|&c| c == col) {
                    Some(p) => p,
                    // `inner_projection` は projection の inner 列をすべて含むので
                    // 到達しないはず。
                    None => err!(Internal),
                };
                out.push(inner_cols[pos].clone());
            } else {
                let (_, v) = &self.partition_cols[col - inner_n];
                out.push(constant_vector(value_ty(v), v, rows));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Value;

    #[test]
    fn hive_segments_are_parsed_from_directories_only() {
        let cols = PartitionedFormat::parse_hive_path("data/year=2024/month=01/part.parquet");
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].0, "year");
        assert!(matches!(cols[0].1, Value::I32(2024)));
        assert_eq!(cols[1].0, "month");
        assert!(matches!(cols[1].1, Value::I32(1)));
    }

    #[test]
    fn filename_is_never_treated_as_a_partition_segment() {
        // ファイル名に `=` が含まれていても見ない。
        let cols = PartitionedFormat::parse_hive_path("a=b.parquet");
        assert!(cols.is_empty());
    }

    #[test]
    fn no_key_value_segments_means_no_partitions() {
        assert!(PartitionedFormat::parse_hive_path("data/plain/file.parquet").is_empty());
        assert!(PartitionedFormat::parse_hive_path("file.parquet").is_empty());
    }

    #[test]
    fn non_numeric_values_stay_varchar() {
        let cols = PartitionedFormat::parse_hive_path("data/region=us-east/f.parquet");
        assert!(matches!(&cols[0].1, Value::Bytes(b) if b == b"us-east"));
    }

    #[test]
    fn large_numeric_values_become_bigint() {
        let cols = PartitionedFormat::parse_hive_path("data/ts=99999999999/f.parquet");
        assert!(matches!(cols[0].1, Value::I64(99_999_999_999)));
    }

    #[test]
    fn percent_encoded_values_are_decoded() {
        // duckdb で実測: `region=us%20east` は "us east" になる。
        let cols = PartitionedFormat::parse_hive_path("data/region=us%20east/f.parquet");
        assert!(matches!(&cols[0].1, Value::Bytes(b) if b == b"us east"));
    }

    #[test]
    fn malformed_percent_escape_is_left_as_is() {
        // `%` の後ろが16進数でない、または途中で切れている場合は素通しする。
        let cols = PartitionedFormat::parse_hive_path("data/x=100%off/f.parquet");
        assert!(matches!(&cols[0].1, Value::Bytes(b) if b == b"100%off"));
        let cols = PartitionedFormat::parse_hive_path("data/x=abc%2/f.parquet");
        assert!(matches!(&cols[0].1, Value::Bytes(b) if b == b"abc%2"));
    }

    // --- 分離した TableFormat に対する動作確認用のフェイク --------------------

    struct FakeFormat {
        schema: Vec<Field>,
        rows: usize,
    }

    impl TableFormat for FakeFormat {
        fn resolve(&mut self, _src: &Source) -> Result<ResolveStep> {
            Ok(Ok(()))
        }
        fn is_resolved(&self) -> bool {
            true
        }
        fn schema(&self) -> &[Field] {
            &self.schema
        }
        fn num_splits(&self) -> usize {
            1
        }
        fn split_rows(&self, _split: usize) -> Option<u64> {
            Some(self.rows as u64)
        }
        fn split_ranges(
            &self,
            _split: usize,
            _projection: &[usize],
            _out: &mut Vec<(u64, u64)>,
        ) -> Result<()> {
            Ok(())
        }
        fn read_split(
            &self,
            _src: &Source,
            _split: usize,
            projection: &[usize],
        ) -> Result<Vec<Vector>> {
            Ok(projection
                .iter()
                .map(|&c| {
                    let mut v = Vector::with_capacity(self.schema[c].ty, self.rows);
                    for i in 0..self.rows {
                        v.push_value(&Value::I32(i as i32));
                    }
                    v
                })
                .collect())
        }
    }

    fn fake_source() -> Source {
        Source::from_bytes(Vec::new())
    }

    #[test]
    fn schema_is_extended_with_partition_columns() {
        let inner =
            Box::new(FakeFormat { schema: vec![Field::new("id", Ty::Int, false)], rows: 3 });
        let mut f = PartitionedFormat::new(
            inner,
            vec![
                ("year".to_string(), Value::I32(2024)),
                ("region".to_string(), Value::Bytes(b"us".to_vec())),
            ],
        );
        assert!(f.resolve(&fake_source()).unwrap().is_ok());
        let schema = f.schema();
        assert_eq!(schema.len(), 3);
        assert_eq!(schema[0].name, "id");
        assert_eq!(schema[1].name, "year");
        assert_eq!(schema[1].ty, Ty::Int);
        assert_eq!(schema[2].name, "region");
        assert_eq!(schema[2].ty, Ty::Varchar);
    }

    #[test]
    fn read_split_appends_constant_partition_columns() {
        let inner =
            Box::new(FakeFormat { schema: vec![Field::new("id", Ty::Int, false)], rows: 4 });
        let mut f = PartitionedFormat::new(inner, vec![("year".to_string(), Value::I32(2024))]);
        f.resolve(&fake_source()).unwrap().unwrap();

        let cols = f.read_split(&fake_source(), 0, &[0, 1]).unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].len(), 4);
        assert_eq!(cols[1].len(), 4);
        for i in 0..4 {
            assert!(matches!(cols[1].value_at(i), Value::I32(2024)));
        }
    }

    #[test]
    fn partition_column_colliding_with_a_real_file_column_is_rejected() {
        // ファイル自身に "year" 列があり、かつパスにも `year=...` があると、
        // 同名の列が2本並ぶスキーマになってしまう。どちらを引くか未定義に
        // なるくらいなら、明確なエラーで拒否するべき。
        let inner = Box::new(FakeFormat {
            schema: vec![Field::new("id", Ty::Int, false), Field::new("year", Ty::Int, false)],
            rows: 3,
        });
        let mut f = PartitionedFormat::new(inner, vec![("year".to_string(), Value::I32(2024))]);
        assert!(f.resolve(&fake_source()).is_err());
    }

    #[test]
    fn read_split_works_when_only_partition_columns_are_projected() {
        // inner の列を 1 つも選んでいなくても、行数はちゃんと分かる。
        let inner =
            Box::new(FakeFormat { schema: vec![Field::new("id", Ty::Int, false)], rows: 5 });
        let mut f = PartitionedFormat::new(inner, vec![("year".to_string(), Value::I32(2024))]);
        f.resolve(&fake_source()).unwrap().unwrap();

        let cols = f.read_split(&fake_source(), 0, &[1]).unwrap();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].len(), 5);
    }
}
