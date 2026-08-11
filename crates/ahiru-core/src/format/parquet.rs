//! Parquet を `TableFormat` に適合させるアダプタ。
//!
//! Parquet 固有の概念（RowGroup、列チャンク、Thrift 統計）はこのファイルの
//! 内側で完結する。実行エンジンからは「統計を持つ分割」に見えるだけ。

use crate::catalog::Source;
use crate::format::{range_may_match, CodecTask, Pruner, ResolveStep, TableFormat};
use crate::parquet::file::{footer_probe_range, parse_footer, ParquetFile};
use crate::parquet::reader::{collect_codec_pages, read_column_chunk};
use crate::prelude::*;
use crate::vector::{Field, PhysType, Ty, Value, Vector};

pub struct ParquetFormat {
    file: Option<ParquetFile>,
    /// 実行側に見せるスキーマ。`ParquetFile` から 1 度だけ写しておく。
    schema: Vec<Field>,
}

impl ParquetFormat {
    pub fn new() -> Self {
        ParquetFormat { file: None, schema: Vec::new() }
    }

    fn file(&self) -> Result<&ParquetFile> {
        match &self.file {
            Some(f) => Ok(f),
            None => err!(Internal),
        }
    }

    /// 分割 `split` の列 `col`（Parquet 上の列番号）のメタデータ。
    fn chunk(&self, split: usize, col: usize) -> Result<&crate::parquet::meta::ColumnMetaData> {
        let f = self.file()?;
        let rg = match f.meta.row_groups.get(split) {
            Some(r) => r,
            None => err!(Internal),
        };
        match rg.columns.get(col).and_then(|c| c.meta.as_ref()) {
            Some(m) => Ok(m),
            None => err!(BadThrift),
        }
    }
}

impl Default for ParquetFormat {
    fn default() -> Self {
        ParquetFormat::new()
    }
}

impl TableFormat for ParquetFormat {
    fn resolve(&mut self, src: &Source) -> Result<ResolveStep> {
        if self.file.is_some() {
            return Ok(Ok(()));
        }
        ensure!(src.total_len >= 12, BadMagic);

        // フッタは末尾にあるので、まず末尾 64 KiB を投機的に要求する。
        let (off, len) = footer_probe_range(src.total_len);
        let tail = match src.get(off, len as usize) {
            Some(t) => t,
            None => return Ok(Err((off, len))),
        };
        match parse_footer(tail, off, src.total_len)? {
            Ok(f) => {
                self.schema = f
                    .schema
                    .columns
                    .iter()
                    .map(|c| Field::new(c.name.clone(), c.ty, c.nullable))
                    .collect();
                self.file = Some(f);
                Ok(Ok(()))
            }
            // 投機取得に収まらなかった。正確な範囲を要求し直す。
            Err(range) => Ok(Err(range)),
        }
    }

    fn is_resolved(&self) -> bool {
        self.file.is_some()
    }

    fn schema(&self) -> &[Field] {
        &self.schema
    }

    fn num_splits(&self) -> usize {
        self.file.as_ref().map_or(0, |f| f.meta.row_groups.len())
    }

    fn split_rows(&self, split: usize) -> Option<u64> {
        let f = self.file.as_ref()?;
        f.meta.row_groups.get(split).map(|rg| rg.num_rows.max(0) as u64)
    }

    fn split_ranges(
        &self,
        split: usize,
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        // 列指向なので、射影された列チャンクの範囲だけを要求する。
        // ここが「読むバイトを減らす」最も効く場所。
        for &c in projection {
            out.push(self.chunk(split, c)?.byte_range());
        }
        Ok(())
    }

    fn codec_tasks(
        &self,
        src: &Source,
        split: usize,
        projection: &[usize],
        out: &mut Vec<CodecTask>,
    ) -> Result<()> {
        let f = self.file()?;
        let rg = match f.meta.row_groups.get(split) {
            Some(r) => r,
            None => err!(Internal),
        };
        let nrows = rg.num_rows.max(0) as usize;
        let mut pages = Vec::new();
        for &c in projection {
            let meta = self.chunk(split, c)?;
            if meta.codec.is_builtin() {
                continue;
            }
            let (start, end) = meta.byte_range();
            let buf = match src.get(start, (end - start) as usize) {
                Some(b) => b,
                None => err!(Internal),
            };
            pages.clear();
            collect_codec_pages(meta, buf, start, nrows, &mut pages)?;
            for p in &pages {
                out.push(CodecTask {
                    codec: p.codec,
                    offset: p.offset,
                    len: p.len,
                    out_len: p.out_len,
                });
            }
        }
        Ok(())
    }

    fn may_match(&self, split: usize, pruners: &[Pruner], projection: &[usize]) -> bool {
        for p in pruners {
            let Some(&col) = projection.get(p.column) else { continue };
            let Ok(meta) = self.chunk(split, col) else { continue };
            let Some(stats) = &meta.statistics else { continue };
            // min/max (フィールド 1,2) は符号の扱いが writer 依存で信用できない
            // ので使わない。min_value/max_value (5,6) だけを見る。
            let (Some(min), Some(max)) = (&stats.min_value, &stats.max_value) else {
                continue;
            };
            let Some(ty) = self.schema.get(col).map(|f| f.ty) else { continue };
            let (Some(min), Some(max)) = (stat_value(ty, min), stat_value(ty, max)) else {
                continue;
            };
            if !range_may_match(p, &min, &max) {
                return false;
            }
        }
        true
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        let f = self.file()?;
        let rg = match f.meta.row_groups.get(split) {
            Some(r) => r,
            None => err!(Internal),
        };
        let nrows = rg.num_rows.max(0) as usize;

        let mut cols = Vec::with_capacity(projection.len());
        for &c in projection {
            let meta = self.chunk(split, c)?;
            let (start, end) = meta.byte_range();
            let buf = match src.get(start, (end - start) as usize) {
                Some(b) => b,
                // split_ranges が示した範囲は呼び出し側が揃えている約束。
                None => err!(Internal),
            };
            let desc = match f.schema.columns.get(c) {
                Some(d) => d,
                None => err!(Internal),
            };
            // 展開済みページのキャッシュは `Source` が持つ。内蔵コーデックの
            // ファイルでは一度も引かれない。
            cols.push(read_column_chunk(desc, meta, buf, start, nrows, src)?);
        }
        Ok(cols)
    }
}

/// 統計バイト列を、その列の型に合わせて比較可能な `Value` にする。
///
/// Parquet の統計は物理型のリトルエンディアン表現で書かれる。論理型が
/// INT64 相当でも物理型が INT32 なら 4 バイトしかない（DATE、TIME_MILLIS）。
pub fn stat_value(ty: Ty, bytes: &[u8]) -> Option<Value> {
    match ty.phys() {
        PhysType::I32 => {
            if bytes.len() < 4 {
                return None;
            }
            Some(Value::I32(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])))
        }
        PhysType::I64 => match bytes.len() {
            4 => Some(Value::I64(
                i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
            )),
            n if n >= 8 => Some(Value::I64(i64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]))),
            _ => None,
        },
        PhysType::F64 => match bytes.len() {
            4 => Some(Value::F64(
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
            )),
            n if n >= 8 => Some(Value::F64(f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]))),
            _ => None,
        },
        // 文字列統計は writer が切り詰めることがあるので枝刈りに使わない。
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_value_reads_physical_width() {
        assert_eq!(stat_value(Ty::Int, &7i32.to_le_bytes()), Some(Value::I32(7)));
        assert_eq!(stat_value(Ty::BigInt, &7i64.to_le_bytes()), Some(Value::I64(7)));
        // DATE は論理的に I32 だが、TIME_MILLIS のように物理 INT32 で
        // 論理 I64 の列もある。4 バイトを 64 ビットに広げて読む。
        assert_eq!(stat_value(Ty::Time, &7i32.to_le_bytes()), Some(Value::I64(7)));
        assert_eq!(stat_value(Ty::Double, &1.5f64.to_le_bytes()), Some(Value::F64(1.5)));
        assert_eq!(stat_value(Ty::Float, &1.5f32.to_le_bytes()), Some(Value::F64(1.5)));
        // 文字列は使わない。
        assert_eq!(stat_value(Ty::Varchar, b"abc"), None);
        // 短すぎる統計は読まない。
        assert_eq!(stat_value(Ty::BigInt, &[1, 2]), None);
    }

    #[test]
    fn unresolved_format_reports_no_splits() {
        let f = ParquetFormat::new();
        assert!(!f.is_resolved());
        assert_eq!(f.num_splits(), 0);
        assert!(f.schema().is_empty());
        assert_eq!(f.split_rows(0), None);
    }

    #[test]
    fn tiny_source_is_rejected_before_any_io() {
        let mut f = ParquetFormat::new();
        let src = Source::remote(4);
        assert_eq!(crate::error::code_of(f.resolve(&src)), Some(crate::error::Code::BadMagic));
    }

    #[test]
    fn resolve_requests_the_footer_probe_range() {
        let mut f = ParquetFormat::new();
        // バイトが 1 つも無いので、末尾 64 KiB を要求して戻るはず。
        let src = Source::remote(1_000_000);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off + len, 1_000_000);
                assert_eq!(len, 64 * 1024);
            }
            Ok(()) => panic!("バイトが無いのに解決できるはずがない"),
        }
    }
}
