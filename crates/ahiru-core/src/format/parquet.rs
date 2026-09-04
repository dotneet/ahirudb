//! The adapter fitting Parquet to `TableFormat`.
//!
//! Parquet-specific concepts (RowGroups, column chunks, Thrift statistics) are contained within
//! this file. To the execution engine it merely looks like "a split carrying statistics".

use crate::catalog::Source;
use crate::format::{
    get_or_internal, range_may_match, CodecTask, PruneOp, Pruner, ResolveStep, TableFormat,
};
use crate::parquet::bloom::BloomFilter;
use crate::parquet::file::{footer_probe_range, parse_footer, ParquetFile};
use crate::parquet::meta::{
    decode_bloom_filter_header, decode_column_index, decode_offset_index, ColumnIndex,
    ColumnMetaData, OffsetIndex, MAX_PAGES_PER_COLUMN,
};
use crate::parquet::nested::read_nested_column;
use crate::parquet::reader::{
    collect_codec_pages, collect_codec_pages_all, collect_codec_pages_selected, is_unsigned,
    read_column_chunk, read_selected_pages, CodecPage,
};
use crate::parquet::schema::ColumnDesc;
use crate::parquet::{PType, TimeUnit};
use crate::prelude::*;
use crate::vector::{Field, PhysType, Ty, Value, Vector};

pub struct ParquetFormat {
    file: Option<ParquetFile>,
    /// The schema shown to the execution side. Copied from `ParquetFile` exactly once.
    schema: Vec<Field>,
    /// The current split's page selection result, as settled by the most recent `refine_with_index`.
    /// Used only for calls whose `split`/`projection` match (otherwise it falls back to the
    /// conventional "no filtering" path).
    page_plan: Option<PagePlan>,
}

/// The page selection result for one split (RowGroup).
struct PagePlan {
    split: usize,
    projection: Vec<usize>,
    /// The "keep" intervals in absolute row numbers within the RowGroup (ascending, merged, no duplicates).
    kept_ranges: Vec<(u64, u64)>,
    /// In the same order as `projection`. `Some` per column means page selection applies (fewer
    /// bytes); `None` means that column alone reads the whole column chunk and gathers into
    /// `kept_ranges` after decoding to line the row counts up.
    columns: Vec<Option<ColumnPagePlan>>,
}

struct ColumnPagePlan {
    /// The byte range of the dictionary page (if any).
    dict_range: Option<(u64, u64)>,
    /// The data pages to read, as `(start, end, the absolute RowGroup row number of the first row)`.
    /// In ascending byte-offset order.
    pages: Vec<(u64, u64, i64)>,
}

impl ParquetFormat {
    pub fn new() -> Self {
        ParquetFormat { file: None, schema: Vec::new(), page_plan: None }
    }

    fn file(&self) -> Result<&ParquetFile> {
        match &self.file {
            Some(f) => Ok(f),
            None => err!(Internal),
        }
    }

    /// The `ColumnDesc` of logical column `col` (an index into `self.schema`/`f.schema.columns`).
    ///
    /// A nested column (LIST/MAP and so on) has one logical column consuming several physical
    /// column chunks, so `col` is not necessarily a physical column chunk number. Physical numbers
    /// come from `ColumnDesc::phys_cols`.
    fn desc(&self, col: usize) -> Result<&ColumnDesc> {
        let f = self.file()?;
        match f.schema.columns.get(col) {
            Some(d) => Ok(d),
            None => err!(Internal),
        }
    }

    /// The metadata of physical column chunk number `phys` (an index into `row_group.columns`
    /// itself) of split `split`.
    fn chunk_phys(&self, split: usize, phys: usize) -> Result<&ColumnMetaData> {
        let f = self.file()?;
        let rg = match f.meta.row_groups.get(split) {
            Some(r) => r,
            None => err!(Internal),
        };
        match rg.columns.get(phys).and_then(|c| c.meta.as_ref()) {
            Some(m) => Ok(m),
            None => err!(BadThrift),
        }
    }

    /// The metadata of the representative physical column chunk of logical column `col` of split `split`.
    /// Statistics and pruning look only at the representative column (the first leaf), a
    /// simplification that always coincides with the sole physical column for flat columns. Nested
    /// columns are outside pruning's scope to begin with (`stat_value` always returns `None` for
    /// `Ty::Json` statistics), so the representative column alone safely errs to "cannot decide, so no filtering".
    fn chunk(&self, split: usize, col: usize) -> Result<&ColumnMetaData> {
        let phys = match self.desc(col)?.phys_cols.first() {
            Some(&p) => p,
            None => err!(Internal),
        };
        self.chunk_phys(split, phys)
    }

    /// The `ColumnChunk` of the representative physical column chunk of logical column `col` of
    /// split `split` (the ColumnIndex/OffsetIndex offsets live here). The same representative-column
    /// simplification as `chunk`.
    fn column_chunk(&self, split: usize, col: usize) -> Result<&crate::parquet::meta::ColumnChunk> {
        let phys = match self.desc(col)?.phys_cols.first() {
            Some(&p) => p,
            None => err!(Internal),
        };
        let f = self.file()?;
        let rg = match f.meta.row_groups.get(split) {
            Some(r) => r,
            None => err!(Internal),
        };
        match rg.columns.get(phys) {
            Some(c) => Ok(c),
            None => err!(Internal),
        }
    }

    fn num_rows(&self, split: usize) -> Result<usize> {
        let f = self.file()?;
        match f.meta.row_groups.get(split) {
            Some(rg) => {
                ensure!(rg.num_rows >= 0, BadThrift);
                match usize::try_from(rg.num_rows) {
                    Ok(n) => Ok(n),
                    Err(_) => err!(LimitExceeded),
                }
            }
            None => err!(Internal),
        }
    }

    /// Returns the cached page selection result only when `split`/`projection` match.
    /// On a mismatch it is safest to use the conventional "no filtering" path.
    fn matching_plan(&self, split: usize, projection: &[usize]) -> Option<&PagePlan> {
        let plan = self.page_plan.as_ref()?;
        if plan.split == split && plan.projection == projection {
            Some(plan)
        } else {
            None
        }
    }

    /// Pushes into `out` the ranges of every physical column chunk of logical column `col`. One for
    /// a flat column, and every leaf of `ColumnDesc::phys_cols` for a nested one (LIST/MAP and so on).
    fn push_full_chunk_ranges(
        &self,
        split: usize,
        col: usize,
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        let phys_cols = self.desc(col)?.phys_cols.clone();
        for p in phys_cols {
            out.push(self.chunk_phys(split, p)?.byte_range());
        }
        Ok(())
    }

    /// Collects, for every physical column chunk of logical column `col`, the pages needing host
    /// delegation and pushes them into `out`. A nested column's physical leaves can contain
    /// REPEATED, so the termination condition is exhausting the buffer rather than reaching
    /// `num_rows` (`collect_codec_pages_all`). Flat columns stay row-count-driven as before.
    fn push_full_chunk_codec_tasks(
        &self,
        src: &Source,
        split: usize,
        col: usize,
        nrows: usize,
        pages: &mut Vec<CodecPage>,
        out: &mut Vec<CodecTask>,
    ) -> Result<()> {
        let desc = self.desc(col)?;
        let nested = desc.nested.is_some();
        let phys_cols = desc.phys_cols.clone();
        for p in phys_cols {
            let meta = self.chunk_phys(split, p)?;
            if meta.codec.is_builtin() {
                continue;
            }
            let (s, e) = meta.byte_range();
            let buf = get_or_internal(src, s, e)?;
            pages.clear();
            if nested {
                collect_codec_pages_all(meta, buf, s, pages)?;
            } else {
                collect_codec_pages(meta, buf, s, nrows, pages)?;
            }
            push_codec_tasks(pages, out)?;
        }
        Ok(())
    }

    /// Fetches every physical leaf of a nested column (LIST/MAP and so on) as a whole column chunk
    /// each and assembles them into a single `Ty::Json` vector via Dremel assembly.
    fn read_full_nested_column(
        &self,
        src: &Source,
        split: usize,
        desc: &ColumnDesc,
        num_rows: usize,
    ) -> Result<Vector> {
        // A 0-row row group (pyarrow writes one for an empty table or an empty batch)
        // has no pages to assemble, and its column chunk metadata need not describe a
        // usable byte range at all. The nested read path walks the chunk buffer to
        // exhaustion rather than to a row count, so it must not be entered here.
        if num_rows == 0 {
            return Ok(Vector::with_capacity(desc.ty, 0));
        }
        let mut metas = Vec::with_capacity(desc.phys_cols.len());
        for &p in &desc.phys_cols {
            metas.push(self.chunk_phys(split, p)?);
        }
        let mut chunks = Vec::with_capacity(metas.len());
        for meta in &metas {
            let (s, e) = meta.byte_range();
            let buf = get_or_internal(src, s, e)?;
            chunks.push((*meta, buf, s));
        }
        read_nested_column(desc, &chunks, num_rows, src)
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

        // The footer is at the end, so the last 64 KiB is requested speculatively first.
        let (off, len) = footer_probe_range(src.total_len);
        let tail = match src.get(off, len as usize) {
            Some(t) => t,
            None => return Ok(Err((off, len))),
        };
        let f = match parse_footer(tail, off, src.total_len)? {
            Ok(f) => f,
            // The footer did not fit the speculative fetch. If the exact range has
            // already been satisfied, parse it; otherwise request it. Re-probing the
            // same 64 KiB tail here would return the same request forever and break
            // the `TableFormat::resolve` contract ("called again with the same ranges
            // satisfied, it must make progress"). Mirrors `parquet::file::open_bytes`.
            Err((start, flen)) => {
                let buf = match src.get(start, flen as usize) {
                    Some(b) => b,
                    None => return Ok(Err((start, flen))),
                };
                match parse_footer(buf, start, src.total_len)? {
                    Ok(f) => f,
                    // The refetched range spans the whole footer, so a second miss
                    // means the declared length is inconsistent.
                    Err(_) => err!(BadMagic),
                }
            }
        };
        self.schema =
            f.schema.columns.iter().map(|c| Field::new(c.name.clone(), c.ty, c.nullable)).collect();
        self.file = Some(f);
        Ok(Ok(()))
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
        f.meta.row_groups.get(split).and_then(|rg| rg.num_rows.try_into().ok())
    }

    fn split_ranges(
        &self,
        split: usize,
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        if let Some(plan) = self.matching_plan(split, projection) {
            for (i, &c) in projection.iter().enumerate() {
                match &plan.columns[i] {
                    Some(cp) => {
                        if let Some(d) = cp.dict_range {
                            out.push(d);
                        }
                        for &(s, e, _) in &cp.pages {
                            out.push((s, e));
                        }
                    }
                    // This column could not be filtered (nested columns always land here).
                    // The whole column chunk as before -- several of them for a nested column.
                    None => self.push_full_chunk_ranges(split, c, out)?,
                }
            }
            return Ok(());
        }
        // Being columnar, only the ranges of the projected column chunks are requested.
        // This is the single most effective place for "reading fewer bytes".
        for &c in projection {
            self.push_full_chunk_ranges(split, c, out)?;
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
        let nrows = self.num_rows(split)?;
        let mut pages = Vec::new();

        if let Some(plan) = self.matching_plan(split, projection) {
            for (i, &c) in projection.iter().enumerate() {
                match &plan.columns[i] {
                    Some(cp) => {
                        let meta = self.chunk(split, c)?;
                        if meta.codec.is_builtin() {
                            continue;
                        }
                        pages.clear();
                        let dict = match cp.dict_range {
                            Some((s, e)) => Some((get_or_internal(src, s, e)?, s)),
                            None => None,
                        };
                        let mut bufs = Vec::with_capacity(cp.pages.len());
                        for &(s, e, _) in &cp.pages {
                            bufs.push((get_or_internal(src, s, e)?, s));
                        }
                        collect_codec_pages_selected(meta, dict, &bufs, &mut pages)?;
                        push_codec_tasks(&pages, out)?;
                    }
                    // This column could not be filtered (nested columns always land here).
                    None => {
                        self.push_full_chunk_codec_tasks(src, split, c, nrows, &mut pages, out)?
                    }
                }
            }
            return Ok(());
        }

        for &c in projection {
            self.push_full_chunk_codec_tasks(src, split, c, nrows, &mut pages, out)?;
        }
        Ok(())
    }

    /// Evaluates `pruners` using RowGroup statistics (min_value/max_value) alone.
    /// When it cannot decide -- no statistics, unknown type, and so on -- that pruner is skipped
    /// (erring safe = treated as "cannot filter").
    fn may_match(&self, split: usize, pruners: &[Pruner], projection: &[usize]) -> bool {
        for p in pruners {
            let Some(&col) = projection.get(p.column) else { continue };
            let Ok(meta) = self.chunk(split, col) else { continue };
            let Some(stats) = &meta.statistics else { continue };
            // min/max (fields 1 and 2) cannot be trusted, as the sign handling is writer-dependent,
            // so they are not used. Only min_value/max_value (5 and 6) are consulted.
            let (Some(min), Some(max)) = (&stats.min_value, &stats.max_value) else {
                continue;
            };
            let Ok(desc) = self.desc(col) else { continue };
            let (Some(min), Some(max)) = (stat_value(desc, min), stat_value(desc, max)) else {
                continue;
            };
            if !range_may_match(p, &min, &max) {
                return false;
            }
        }
        true
    }

    fn index_ranges(
        &self,
        split: usize,
        pruners: &[Pruner],
        projection: &[usize],
        out: &mut Vec<(u64, u64)>,
    ) -> Result<()> {
        if pruners.is_empty() {
            return Ok(());
        }
        // ColumnIndex (min/max) and the Bloom filter are fetched only for columns a pruner touches.
        for p in pruners {
            let Some(&col) = projection.get(p.column) else { continue };
            let Ok(cc) = self.column_chunk(split, col) else { continue };
            if let Some(r) = cc.column_index_range() {
                out.push(r);
            }
            if p.op == PruneOp::Eq || p.op == PruneOp::In {
                if let Ok(meta) = self.chunk(split, col) {
                    if let Some(r) = meta.bloom_filter_probe_range() {
                        out.push(r);
                    }
                }
            }
        }
        // The OffsetIndex of every projected column is fetched, so the "rows to keep" settled by
        // page selection can be applied to columns without a pruner as well, reducing bytes.
        // Nested columns span several physical column chunks and are outside page selection's scope
        // (they always read whole column chunks), so nothing is fetched for them here.
        for &c in projection {
            if self.desc(c).is_ok_and(|d| d.nested.is_some()) {
                continue;
            }
            if let Ok(cc) = self.column_chunk(split, c) {
                if let Some(r) = cc.offset_index_range() {
                    out.push(r);
                }
            }
        }
        Ok(())
    }

    fn refine_with_index(
        &mut self,
        src: &Source,
        split: usize,
        pruners: &[Pruner],
        projection: &[usize],
    ) -> Result<bool> {
        // The previous cache is discarded unconditionally. Only this call's result is trusted.
        self.page_plan = None;
        if pruners.is_empty() {
            return Ok(true);
        }

        // 1) The Bloom filter: if an equality/IN pruner's value is definitely absent, the whole
        //    split is dropped. It additionally catches the "wide range but sparse actual values"
        //    case that RowGroup min/max statistics cannot. IN becomes an OR decision, "keep if any
        //    candidate exists".
        for p in pruners {
            if p.op != PruneOp::Eq && p.op != PruneOp::In {
                continue;
            }
            let Some(&col) = projection.get(p.column) else { continue };
            // Filtering by statistics or a Bloom filter is meaningless for a nested column
            // (`Ty::Json`), since no single comparable physical value exists, so it is explicitly
            // excluded to avoid mistakenly using the representative leaf's Bloom filter.
            if self.desc(col)?.nested.is_some() {
                continue;
            }
            let Ok(meta) = self.chunk(split, col) else { continue };
            let Some((start, end)) = meta.bloom_filter_probe_range() else { continue };
            let Some(buf) = src.get(start, (end - start) as usize) else { continue };
            let Ok((hdr, used)) = decode_bloom_filter_header(buf) else { continue };
            let need = used + hdr.num_bytes as usize;
            if need > buf.len() {
                // The speculative fetch did not reach the actual size. It errs safe and gives up.
                continue;
            }
            let Some(bf) = BloomFilter::new(&buf[used..need]) else { continue };
            let Ok(desc) = self.desc(col) else { continue };

            // The split can be dropped only once every candidate is encodable and every one is
            // known to be absent from the filter. If even one candidate cannot be encoded, it
            // errs safe and is treated as "possible".
            let mut any_maybe_present = false;
            let mut all_encoded = true;
            for v in core::iter::once(&p.value).chain(p.in_values.iter()) {
                let Some(key) = plain_encode_for_bloom(desc, v) else {
                    all_encoded = false;
                    break;
                };
                if bf.contains(&key) {
                    any_maybe_present = true;
                    break;
                }
            }
            if all_encoded && !any_maybe_present {
                return Ok(false);
            }
        }

        // 2) Page-level min/max: the surviving row ranges are narrowed from the columns with pruners.
        let num_rows = self.num_rows(split)? as u64;
        let mut kept: Option<Vec<(u64, u64)>> = None;
        for p in pruners {
            let Some(&col) = projection.get(p.column) else { continue };
            if self.desc(col)?.nested.is_some() {
                continue;
            }
            let Ok(cc) = self.column_chunk(split, col) else { continue };
            let (Some((ci_s, ci_e)), Some((oi_s, oi_e))) =
                (cc.column_index_range(), cc.offset_index_range())
            else {
                continue;
            };
            let Some(ci_buf) = src.get(ci_s, (ci_e - ci_s) as usize) else { continue };
            let Some(oi_buf) = src.get(oi_s, (oi_e - oi_s) as usize) else { continue };
            let Ok(ci) = decode_column_index(ci_buf) else { continue };
            let Ok(oi) = decode_offset_index(oi_buf) else { continue };
            if ci.null_pages.len() != oi.page_locations.len() || oi.page_locations.is_empty() {
                continue;
            }
            let Ok(desc) = self.desc(col) else { continue };
            let ranges = page_ranges_for_pruner(p, desc, &ci, &oi, num_rows);
            kept = Some(match kept {
                Some(prev) => intersect_ranges(&prev, &ranges),
                None => ranges,
            });
        }

        let Some(kept_ranges) = kept else {
            // No pruner could decide at page level. Behavior stays as before.
            return Ok(true);
        };
        if kept_ranges.is_empty() {
            // No page can match. The whole RowGroup can be skipped.
            return Ok(false);
        }

        // 3) For each projected column, the surviving row ranges are mapped onto its own OffsetIndex.
        //    A column without an OffsetIndex reads the whole column chunk and gathers afterwards.
        //    A nested column (LIST/MAP and so on) spans several physical column chunks, and
        //    `ColumnPagePlan` can only express a selection for one column chunk, so it is always
        //    made to read the whole column chunk (`None`).
        let mut columns = Vec::with_capacity(projection.len());
        for &c in projection {
            if self.desc(c)?.nested.is_some() {
                columns.push(None);
                continue;
            }
            let Ok(cc) = self.column_chunk(split, c) else {
                columns.push(None);
                continue;
            };
            let Ok(meta) = self.chunk(split, c) else {
                columns.push(None);
                continue;
            };
            let plan = (|| -> Option<ColumnPagePlan> {
                let (oi_s, oi_e) = cc.offset_index_range()?;
                let oi_buf = src.get(oi_s, (oi_e - oi_s) as usize)?;
                let oi = decode_offset_index(oi_buf).ok()?;
                if oi.page_locations.is_empty() {
                    return None;
                }
                let mut pages = Vec::new();
                for (i, loc) in oi.page_locations.iter().enumerate() {
                    let row_start = loc.first_row_index as u64;
                    let row_end = oi
                        .page_locations
                        .get(i + 1)
                        .map(|n| n.first_row_index as u64)
                        .unwrap_or(num_rows);
                    if !ranges_overlap(&kept_ranges, row_start, row_end) {
                        continue;
                    }
                    let start = loc.offset as u64;
                    let end = start + loc.compressed_page_size as u64;
                    pages.push((start, end, loc.first_row_index));
                }
                if pages.is_empty() {
                    // No page was selected = this column need not be read at all.
                    // It does not fall back, though, so an empty vector can still be built.
                    return Some(ColumnPagePlan { dict_range: None, pages });
                }
                // A dictionary-encoded chunk must never be read page-by-page without its
                // dictionary page. The OffsetIndex's first entry is where the data pages
                // really start, which `data_page_offset` does not give under the
                // parquet-mr layout. When the range cannot be worked out, give up on page
                // selection (`None`) so the whole chunk -- dictionary page included -- is
                // read instead.
                let dict_range = if meta.has_dictionary() {
                    let first = oi.page_locations.first()?.offset;
                    Some(meta.dictionary_page_range(first)?)
                } else {
                    None
                };
                Some(ColumnPagePlan { dict_range, pages })
            })();
            columns.push(plan);
        }

        self.page_plan =
            Some(PagePlan { split, projection: projection.to_vec(), kept_ranges, columns });
        Ok(true)
    }

    fn read_split(&self, src: &Source, split: usize, projection: &[usize]) -> Result<Vec<Vector>> {
        let f = self.file()?;
        let nrows = self.num_rows(split)?;

        if let Some(plan) = self.matching_plan(split, projection) {
            let mut cols = Vec::with_capacity(projection.len());
            for (i, &c) in projection.iter().enumerate() {
                let meta = self.chunk(split, c)?;
                let desc = match f.schema.columns.get(c) {
                    Some(d) => d,
                    None => err!(Internal),
                };
                match &plan.columns[i] {
                    Some(cp) if !cp.pages.is_empty() => {
                        let dict_buf = match cp.dict_range {
                            Some((s, e)) => Some((get_or_internal(src, s, e)?, s)),
                            None => None,
                        };
                        let mut pagebufs = Vec::with_capacity(cp.pages.len());
                        for &(s, e, fr) in &cp.pages {
                            pagebufs.push((get_or_internal(src, s, e)?, s, fr));
                        }
                        let (v, abs_rows) =
                            read_selected_pages(desc, meta, dict_buf, &pagebufs, src)?;
                        let idx = select_indices(&abs_rows, &plan.kept_ranges);
                        cols.push(v.gather(&idx));
                    }
                    Some(_) => {
                        // Zero pages selected: no rows of this column survive.
                        cols.push(Vector::with_capacity(desc.ty, 0));
                    }
                    None if desc.nested.is_some() => {
                        let v = self.read_full_nested_column(src, split, desc, nrows)?;
                        let idx = select_indices_full(nrows, &plan.kept_ranges);
                        cols.push(v.gather(&idx));
                    }
                    None => {
                        let (s, e) = meta.byte_range();
                        let buf = get_or_internal(src, s, e)?;
                        let v = read_column_chunk(desc, meta, buf, s, nrows, src)?;
                        let idx = select_indices_full(nrows, &plan.kept_ranges);
                        cols.push(v.gather(&idx));
                    }
                }
            }
            return Ok(cols);
        }

        let mut cols = Vec::with_capacity(projection.len());
        for &c in projection {
            let desc = match f.schema.columns.get(c) {
                Some(d) => d,
                None => err!(Internal),
            };
            if desc.nested.is_some() {
                cols.push(self.read_full_nested_column(src, split, desc, nrows)?);
                continue;
            }
            let meta = self.chunk(split, c)?;
            let (start, end) = meta.byte_range();
            let buf = get_or_internal(src, start, end)?;
            // The cache of decompressed pages is held by `Source`. For a file using only built-in
            // codecs it is never consulted.
            cols.push(read_column_chunk(desc, meta, buf, start, nrows, src)?);
        }
        Ok(cols)
    }
}

/// Converts a sequence of `CodecPage`s into `CodecTask`s and pushes them into `out`.
fn push_codec_tasks(pages: &[CodecPage], out: &mut Vec<CodecTask>) -> Result<()> {
    ensure!(pages.len() <= MAX_PAGES_PER_COLUMN.saturating_sub(out.len()), LimitExceeded);
    for p in pages {
        out.push(CodecTask { codec: p.codec, offset: p.offset, len: p.len, out_len: p.out_len });
    }
    Ok(())
}

/// Whether `[start, end)` overlaps any of `ranges` (ascending and merged).
fn ranges_overlap(ranges: &[(u64, u64)], start: u64, end: u64) -> bool {
    // ranges is small (about the number of pages in a RowGroup), so a linear scan suffices.
    ranges.iter().any(|&(s, e)| s < end && start < e)
}

/// From one pruner, builds the "keep" row ranges by consulting that column's per-page min/max.
/// A page whose statistics are unusable (`null_pages` or a type mismatch) errs safe and is kept.
fn page_ranges_for_pruner(
    p: &Pruner,
    desc: &ColumnDesc,
    ci: &ColumnIndex,
    oi: &OffsetIndex,
    num_rows: u64,
) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for (i, loc) in oi.page_locations.iter().enumerate() {
        let row_start = loc.first_row_index as u64;
        let row_end =
            oi.page_locations.get(i + 1).map(|n| n.first_row_index as u64).unwrap_or(num_rows);
        let keep = if ci.null_pages.get(i).copied().unwrap_or(true) {
            // A page with no statistics (all NULL, say) cannot be decided and is kept.
            true
        } else {
            let min = ci.min_values.get(i).and_then(|b| stat_value(desc, b));
            let max = ci.max_values.get(i).and_then(|b| stat_value(desc, b));
            match (min, max) {
                (Some(min), Some(max)) => range_may_match(p, &min, &max),
                _ => true,
            }
        };
        if keep {
            push_merged(&mut out, row_start, row_end);
        }
    }
    out
}

/// Pushes intervals arriving in ascending order, merging with the previous one when adjacent.
fn push_merged(out: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    if start >= end {
        return;
    }
    if let Some(last) = out.last_mut() {
        if last.1 == start {
            last.1 = end;
            return;
        }
    }
    out.push((start, end));
}

/// The intersection of two ascending, merged interval sets. Only the rows in both are kept.
fn intersect_ranges(a: &[(u64, u64)], b: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (s, e) = (a[i].0.max(b[j].0), a[i].1.min(b[j].1));
        if s < e {
            push_merged(&mut out, s, e);
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Of the ascending absolute row numbers `abs_rows`, returns only the positions contained in
/// `kept` (ascending, merged) as indices into `abs_rows` (= indices into the decoded vector).
fn select_indices(abs_rows: &[u64], kept: &[(u64, u64)]) -> Vec<u32> {
    let mut out = Vec::with_capacity(abs_rows.len());
    let mut ki = 0usize;
    for (pos, &r) in abs_rows.iter().enumerate() {
        while ki < kept.len() && r >= kept[ki].1 {
            ki += 1;
        }
        if ki < kept.len() && r >= kept[ki].0 {
            out.push(pos as u32);
        }
    }
    out
}

/// The index sequence expanding only the `kept` intervals against the consecutive `0..nrows`.
/// Used for a fallback column whose whole column chunk was decoded (where absolute row numbers are
/// the consecutive 0..nrows).
fn select_indices_full(nrows: usize, kept: &[(u64, u64)]) -> Vec<u32> {
    let mut out = Vec::new();
    for &(s, e) in kept {
        let s = s.min(nrows as u64) as u32;
        let e = e.min(nrows as u64) as u32;
        out.extend(s..e);
    }
    out
}

/// Unwinds the microseconds-normalization `reader::push_i32_values`/`push_i64_values` apply
/// (`TimeUnit::to_micros`) back to the column's on-disk physical unit.
///
/// `None` (no `time_unit`, e.g. plain integers/DECIMAL) and `Micros` need no change. `Millis`
/// divides back only when the microsecond value has no sub-millisecond remainder -- any remainder
/// means the value could not have come from a MILLIS column in the first place under exact
/// normalization, so there is nothing correct to encode. `Nanos` always returns `None`: the
/// reader's `v / 1000` truncation is lossy (many nanosecond values collapse onto the same
/// microsecond value), so a microsecond `Value` cannot be unwound back to *the* nanosecond bytes a
/// writer would have hashed -- guessing `micros * 1000` would probe the wrong key whenever the
/// original had a nonzero sub-microsecond remainder, which could wrongly report "definitely
/// absent" for a value that is actually present (exactly the false-negative pruning bug this file
/// exists to avoid).
fn unwind_time_unit(micros: i64, unit: Option<TimeUnit>) -> Option<i64> {
    match unit {
        None | Some(TimeUnit::Micros) => Some(micros),
        Some(TimeUnit::Millis) => (micros % 1000 == 0).then_some(micros / 1000),
        Some(TimeUnit::Nanos) => None,
    }
}

/// Turns a `Value` into PLAIN bytes matching that column's Parquet physical type.
/// A Bloom filter hashes the physical type's byte sequence, so a `Value` widened for the logical
/// type (time-unit scaling, UINT32 zero-extension -- DESIGN.md §8) has to be unwound first. This
/// is the exact inverse of `stat_value`'s normalization and of
/// `reader::push_i32_values`/`push_i64_values`.
///
/// Combinations whose conversion is not well-defined (BOOLEAN, INT96, a NANOS-scaled time value,
/// a FIXED_LEN_BYTE_ARRAY/BYTE_ARRAY DECIMAL, and so on) return `None` and give up on the Bloom
/// check (erring safe = treated as "probably matches").
fn plain_encode_for_bloom(desc: &ColumnDesc, v: &Value) -> Option<Vec<u8>> {
    // A nested (LIST/MAP/STRUCT-with-repeated) column has no single physical value to hash --
    // callers already exclude these, but this keeps the function safe standalone too.
    if desc.nested.is_some() {
        return None;
    }
    match desc.ptype {
        PType::Int32 => match v {
            Value::I32(x) => Some(x.to_le_bytes().to_vec()),
            Value::I64(x) => {
                if is_unsigned(desc.ty) {
                    // UINT32 is zero-extended (not sign-extended) to I64 by the reader; narrow
                    // it back, rejecting anything the physical INT32 width cannot hold.
                    u32::try_from(*x).ok().map(|u| (u as i32).to_le_bytes().to_vec())
                } else {
                    // TIME_MILLIS is stored as INT32 and scaled to microseconds by the reader.
                    let x = unwind_time_unit(*x, desc.time_unit)?;
                    i32::try_from(x).ok().map(|x| x.to_le_bytes().to_vec())
                }
            }
            _ => None,
        },
        PType::Int64 => match v {
            Value::I64(x) => {
                let x = unwind_time_unit(*x, desc.time_unit)?;
                Some(x.to_le_bytes().to_vec())
            }
            Value::I32(x) => Some((*x as i64).to_le_bytes().to_vec()),
            _ => None,
        },
        PType::Float => match v {
            Value::F64(x) => Some((*x as f32).to_le_bytes().to_vec()),
            _ => None,
        },
        PType::Double => match v {
            Value::F64(x) => Some(x.to_le_bytes().to_vec()),
            _ => None,
        },
        PType::ByteArray => match v {
            Value::Bytes(b) => Some(b.clone()),
            _ => None,
        },
        PType::FixedLenByteArray => match v {
            // A DECIMAL(p<=18) stored as FIXED_LEN_BYTE_ARRAY has physical type Bytes but is
            // widened to `Value::I64` (big-endian two's complement, see `push_byte_values`), so
            // it never matches this `Value::Bytes` arm and safely falls through to `None` --
            // unwinding that big-endian encoding back is not attempted here. Only UUID/BLOB
            // (whose `Value` stays `Bytes` unwidened) actually take this path.
            Value::Bytes(b) if b.len() == desc.type_length => Some(b.clone()),
            _ => None,
        },
        // For BOOLEAN and INT96 the inverse of widening is not obvious
        // (INT96 is already converted to microseconds from the epoch and unwinding is nontrivial),
        // so the Bloom filter is not used, avoiding a misjudgment.
        PType::Boolean | PType::Int96 => None,
    }
}

/// Turns statistics bytes into a comparable `Value`, applying exactly the same normalization the
/// reader applies when decoding actual column values (DESIGN.md §8: TIME/TIMESTAMP are always
/// microseconds internally, UINT32 is zero-extended rather than sign-extended). Without this, a
/// millis/nanos-scaled or unsigned column's statistics would be compared against a normalized
/// query value on the wrong scale, and RowGroup/page pruning could silently drop data that
/// actually matches.
///
/// Parquet statistics are written in the physical type's little-endian representation (with one
/// exception -- see below). Even when the logical type is INT64-equivalent, a physical INT32
/// gives only 4 bytes (DATE, TIME_MILLIS).
///
/// Returns `None` (skip pruning for this value, erring safe) for:
/// - INT96: the 12-byte statistics encoding is Julian day + nanoseconds-since-midnight, not the
///   epoch-microseconds `Value` the reader produces, and correctly decoding that here is out of
///   scope (mirrors `plain_encode_for_bloom`, which already skips INT96 for the same reason).
/// - DECIMAL stored as FIXED_LEN_BYTE_ARRAY/BYTE_ARRAY: unlike everything else here, those
///   statistics are big-endian two's complement (see `push_byte_values`), a different encoding
///   this little-endian path does not attempt to handle.
/// - Anything else not covered by the I32/I64/F64 physical-type cases below (strings, BOOLEAN,
///   nested/JSON columns -- `Ty::Json.phys()` is `Bytes`, which is not one of the three).
pub fn stat_value(desc: &ColumnDesc, bytes: &[u8]) -> Option<Value> {
    if desc.ptype == PType::Int96 {
        return None;
    }
    if matches!(desc.ptype, PType::FixedLenByteArray | PType::ByteArray)
        && matches!(desc.ty, Ty::Decimal { .. })
    {
        return None;
    }
    match desc.ty.phys() {
        PhysType::I32 => {
            if bytes.len() < 4 {
                return None;
            }
            Some(Value::I32(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])))
        }
        PhysType::I64 => {
            let unsigned = is_unsigned(desc.ty);
            match bytes.len() {
                4 => {
                    let raw = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                    // UINT32 is zero-extended (not sign-extended) to 64 bits, matching
                    // `reader::push_i32_values`. TIME_MILLIS is stored as INT32 and normalized
                    // to microseconds, matching the reader's scaling. A column is never both.
                    let x = if unsigned { raw as u32 as i64 } else { raw as i64 };
                    Some(Value::I64(match desc.time_unit {
                        Some(u) => u.to_micros(x),
                        None => x,
                    }))
                }
                n if n >= 8 => {
                    let raw = i64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ]);
                    Some(Value::I64(match desc.time_unit {
                        Some(u) => u.to_micros(raw),
                        None => raw,
                    }))
                }
                _ => None,
            }
        }
        PhysType::F64 => match bytes.len() {
            4 => Some(Value::F64(
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
            )),
            n if n >= 8 => Some(Value::F64(f64::from_le_bytes([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]))),
            _ => None,
        },
        // String statistics can be truncated by the writer, so they are not used for pruning.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a minimal `ColumnDesc` for a flat (non-nested) leaf column, with only the fields
    /// `stat_value`/`plain_encode_for_bloom` actually look at set meaningfully.
    fn desc_for(ty: Ty, ptype: PType, time_unit: Option<TimeUnit>) -> ColumnDesc {
        ColumnDesc {
            name: String::new(),
            ty,
            nullable: true,
            max_def_level: 1,
            ptype,
            type_length: 0,
            time_unit,
            phys_cols: vec![0],
            leaves: Vec::new(),
            nested: None,
        }
    }

    #[test]
    fn stat_value_reads_physical_width() {
        let int_desc = desc_for(Ty::Int, PType::Int32, None);
        assert_eq!(stat_value(&int_desc, &7i32.to_le_bytes()), Some(Value::I32(7)));

        let bigint_desc = desc_for(Ty::BigInt, PType::Int64, None);
        assert_eq!(stat_value(&bigint_desc, &7i64.to_le_bytes()), Some(Value::I64(7)));

        // TIME_MILLIS is physically INT32 (4 bytes) but logically I64, and DESIGN.md §8 says the
        // internal representation is always microseconds -- so the 4 raw bytes must be widened to
        // I64 *and* scaled by 1000, not just widened. This corrects a previous version of this
        // test that pinned the then-current *buggy* behavior (millis value "7" staying "7"
        // instead of becoming 7_000 microseconds): that mismatch between statistics (millis) and
        // the internal `Value` (micros) is exactly what let equality/range pruning silently drop
        // RowGroups/pages that actually matched the predicate.
        let time_millis_desc = desc_for(Ty::Time, PType::Int32, Some(TimeUnit::Millis));
        assert_eq!(stat_value(&time_millis_desc, &7i32.to_le_bytes()), Some(Value::I64(7_000)));

        let double_desc = desc_for(Ty::Double, PType::Double, None);
        assert_eq!(stat_value(&double_desc, &1.5f64.to_le_bytes()), Some(Value::F64(1.5)));

        let float_desc = desc_for(Ty::Float, PType::Float, None);
        assert_eq!(stat_value(&float_desc, &1.5f32.to_le_bytes()), Some(Value::F64(1.5)));

        // Strings are not used.
        let varchar_desc = desc_for(Ty::Varchar, PType::ByteArray, None);
        assert_eq!(stat_value(&varchar_desc, b"abc"), None);

        // Statistics that are too short are not read.
        assert_eq!(stat_value(&bigint_desc, &[1, 2]), None);
    }

    #[test]
    fn stat_value_scales_timestamp_millis_to_micros() {
        // TIMESTAMP_MILLIS is physically INT64 (8 bytes). A stats value of 1_700_000_000_000
        // milliseconds since the epoch must come out as 1_700_000_000_000_000 microseconds to
        // compare correctly against the micros-normalized query literal.
        let desc = desc_for(Ty::Timestamp, PType::Int64, Some(TimeUnit::Millis));
        let millis = 1_700_000_000_000i64;
        assert_eq!(stat_value(&desc, &millis.to_le_bytes()), Some(Value::I64(millis * 1000)),);
    }

    #[test]
    fn stat_value_scales_timestamp_nanos_to_micros() {
        // TIMESTAMP_NANOS is also physically INT64. Nanoseconds truncate down to microseconds
        // (matching `TimeUnit::to_micros`'s `v / 1000`, the same truncation the reader applies).
        let desc = desc_for(Ty::Timestamp, PType::Int64, Some(TimeUnit::Nanos));
        let nanos = 1_700_000_000_123_456_789i64;
        assert_eq!(stat_value(&desc, &nanos.to_le_bytes()), Some(Value::I64(nanos / 1000)),);
    }

    #[test]
    fn stat_value_zero_extends_uint32_at_and_above_2_pow_31() {
        // A UINT32 value >= 2^31 is stored as 4 raw bytes that are negative when read as a
        // signed INT32. The reader zero-extends (as u32) rather than sign-extends when widening
        // to I64 (`reader::push_i32_values`); statistics comparison must match, or a value like
        // 3_000_000_000 would compare as a large negative number and be pruned away incorrectly.
        let desc = desc_for(Ty::UInt, PType::Int32, None);
        let v: u32 = 3_000_000_000;
        assert_eq!(stat_value(&desc, &v.to_le_bytes()), Some(Value::I64(v as i64)));
    }

    #[test]
    fn stat_value_skips_int96() {
        // INT96 statistics are Julian day + nanoseconds-since-midnight, not the epoch-microseconds
        // `Value` the reader produces for INT96 columns (see `reader::decode_plain`'s INT96 arm).
        // Decoding that correctly is out of scope here, so pruning must be skipped rather than
        // comparing against a misinterpreted value.
        let desc = desc_for(Ty::Timestamp, PType::Int96, None);
        // Bytes shaped like a real INT96 stats value (12 bytes): nanos-since-midnight (i64 LE) +
        // Julian day number (i32 LE). Even though this superficially satisfies the ">= 8 bytes"
        // branch that PhysType::I64 would otherwise take, it must still be rejected.
        let mut bytes = 0i64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&2_440_588i32.to_le_bytes());
        assert_eq!(stat_value(&desc, &bytes), None);
    }

    #[test]
    fn stat_value_skips_decimal_fixed_len_byte_array() {
        // DECIMAL(p<=18) stored as FIXED_LEN_BYTE_ARRAY writes big-endian two's complement bytes
        // (see `reader::push_byte_values`/`be_signed`), not the little-endian encoding every other
        // case here uses. Misreading those bytes as little-endian would silently produce a wrong
        // value, so this combination is skipped rather than guessed at.
        let desc =
            desc_for(Ty::Decimal { precision: 10, scale: 2 }, PType::FixedLenByteArray, None);
        assert_eq!(stat_value(&desc, &100i64.to_be_bytes()), None);
    }

    #[test]
    fn bloom_encode_round_trips_timestamp_millis() {
        // The value the reader would produce for a TIMESTAMP_MILLIS column (microseconds) must
        // encode back to exactly the 8 PLAIN bytes (raw milliseconds, little-endian) a writer
        // would have hashed into the Bloom filter.
        let desc = desc_for(Ty::Timestamp, PType::Int64, Some(TimeUnit::Millis));
        let millis = 1_700_000_000_000i64;
        let reader_value = Value::I64(millis * 1000); // what the reader hands to the rest of the engine
        assert_eq!(
            plain_encode_for_bloom(&desc, &reader_value),
            Some(millis.to_le_bytes().to_vec()),
        );
    }

    #[test]
    fn bloom_encode_round_trips_uint32_at_and_above_2_pow_31() {
        let desc = desc_for(Ty::UInt, PType::Int32, None);
        let v: u32 = 3_000_000_000;
        let reader_value = Value::I64(v as i64);
        assert_eq!(plain_encode_for_bloom(&desc, &reader_value), Some(v.to_le_bytes().to_vec()),);
    }

    #[test]
    fn bloom_encode_skips_timestamp_nanos() {
        // Unwinding micros back to nanos is lossy (the reader truncates), so encoding must be
        // skipped rather than guessing `micros * 1000`, which could silently probe the wrong key
        // and report "definitely absent" for a value that is actually present.
        let desc = desc_for(Ty::Timestamp, PType::Int64, Some(TimeUnit::Nanos));
        let reader_value = Value::I64(1_700_000_000_123_456);
        assert_eq!(plain_encode_for_bloom(&desc, &reader_value), None);
    }

    #[test]
    fn bloom_encode_skips_int96() {
        let desc = desc_for(Ty::Timestamp, PType::Int96, None);
        let reader_value = Value::I64(1_700_000_000_000_000);
        assert_eq!(plain_encode_for_bloom(&desc, &reader_value), None);
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
    fn negative_row_group_cardinality_is_rejected() {
        let bytes = pagetest_bytes();
        let mut f = ParquetFormat::new();
        f.file = Some(crate::parquet::file::open_bytes(&bytes).unwrap());
        f.file.as_mut().unwrap().meta.row_groups[0].num_rows = -1;
        assert_eq!(crate::error::code_of(f.num_rows(0)), Some(crate::error::Code::BadThrift));
        assert_eq!(f.split_rows(0), None);
    }

    #[test]
    fn resolve_requests_the_footer_probe_range() {
        let mut f = ParquetFormat::new();
        // With not a single byte present, it should request the last 64 KiB and return.
        let src = Source::remote(1_000_000);
        match f.resolve(&src).unwrap() {
            Err((off, len)) => {
                assert_eq!(off + len, 1_000_000);
                assert_eq!(len, 64 * 1024);
            }
            Ok(()) => panic!("it cannot possibly resolve with no bytes"),
        }
    }

    // --- Page-level filtering (a real file: tests/data/pagetest.parquet) ----
    //
    // A file with id ascending over 0..50000, written by pyarrow (parquet-cpp) with
    // ColumnIndex/OffsetIndex/Bloom filters. `scripts/gen-testdata.sh` records how it is generated.
    // The DuckDB version in this environment cannot write Bloom filters, so this is used for
    // verification against a real writer.

    fn pagetest_bytes() -> Vec<u8> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/pagetest.parquet");
        std::fs::read(p).unwrap_or_else(|e| panic!("tests/data/pagetest.parquet: {e}"))
    }

    fn testdata(name: &str) -> Vec<u8> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
        std::fs::read(format!("{dir}{name}")).unwrap_or_else(|e| panic!("tests/data/{name}: {e}"))
    }

    /// One `resolve` step that is expected to still need bytes.
    fn resolve_step(fmt: &mut ParquetFormat, src: &Source) -> (u64, u64) {
        match fmt.resolve(src).unwrap() {
            Err(range) => range,
            Ok(()) => panic!("expected a byte request, not a completed resolve"),
        }
    }

    /// For a `Source::remote`, registers the not-yet-fetched parts of `ranges` as though they had
    /// been fetched from the host. Returns the total bytes actually transferred
    /// (equivalent to the bytes requested via `NEED_IO`).
    fn fetch_missing(src: &mut Source, bytes: &[u8], ranges: &[(u64, u64)]) -> u64 {
        let mut total = 0u64;
        for &(start, end) in ranges {
            if let Some((o, l)) = src.missing(start, end - start) {
                src.insert(o, bytes[o as usize..(o + l) as usize].to_vec());
                total += l;
            }
        }
        total
    }

    /// Resolves the footer on a `Source::remote` with a "speculative fetch -> refetch if short"
    /// loop (the same shape as the production I/O round trips).
    fn resolve_remote(fmt: &mut ParquetFormat, bytes: &[u8]) -> Source {
        let mut src = Source::remote(bytes.len() as u64);
        loop {
            match fmt.resolve(&src).unwrap() {
                Ok(()) => return src,
                Err((off, len)) => {
                    fetch_missing(&mut src, bytes, &[(off, off + len)]);
                }
            }
        }
    }

    #[test]
    fn a_footer_larger_than_the_probe_resolves_after_refetching_it() {
        // `tests/data/footer_big.parquet` has a 65529-byte footer: one byte more than
        // the 64 KiB probe can hold once the 8-byte trailer is subtracted. `resolve`
        // used to answer the "did not fit" result by asking for the *same* tail again,
        // so no host could ever open such a file -- it violated the
        // `TableFormat::resolve` contract that a re-entry with the requested ranges
        // satisfied must make progress.
        let bytes = testdata("footer_big.parquet");
        let mut fmt = ParquetFormat::new();
        let mut src = Source::remote(bytes.len() as u64);

        let probe = resolve_step(&mut fmt, &src);
        assert_eq!(probe.1, crate::parquet::file::FOOTER_PROBE);
        fetch_missing(&mut src, &bytes, &[(probe.0, probe.0 + probe.1)]);

        // The second request must differ from the first, and must span footer + trailer.
        let exact = resolve_step(&mut fmt, &src);
        assert_ne!(exact, probe, "re-requesting the same range makes no progress");
        assert_eq!(exact.1, 65529 + 8);
        assert_eq!(exact.0 + exact.1, bytes.len() as u64);
        fetch_missing(&mut src, &bytes, &[(exact.0, exact.0 + exact.1)]);

        assert!(fmt.resolve(&src).unwrap().is_ok());
        assert_eq!(fmt.schema().len(), 12);
        assert_eq!(fmt.num_splits(), 6);
    }

    #[test]
    fn a_footer_exactly_filling_the_probe_still_resolves_in_one_round_trip() {
        // The boundary from the other side: 65528 bytes of footer plus the 8-byte
        // trailer is exactly the 64 KiB probe, the largest footer that still fits.
        // Refetching here would be a wasted round trip.
        let bytes = testdata("footer_fit.parquet");
        let mut fmt = ParquetFormat::new();
        let mut src = Source::remote(bytes.len() as u64);

        let probe = resolve_step(&mut fmt, &src);
        fetch_missing(&mut src, &bytes, &[(probe.0, probe.0 + probe.1)]);
        assert!(fmt.resolve(&src).unwrap().is_ok());
        assert_eq!(fmt.num_splits(), 6);
    }

    #[test]
    fn parquet_mr_dictionary_layout_keeps_the_dictionary_page_when_pages_are_selected() {
        // `tests/data/dict_mr.parquet` is a dictionary-encoded, page-indexed file whose
        // ColumnMetaData was rewritten to the parquet-mr (Spark/Hive) layout, where the
        // chunk start is recorded as `data_page_offset` before the dictionary page is
        // written -- so `dictionary_page_offset == data_page_offset`. Deriving the
        // dictionary page's range from `data_page_offset` yields an empty range, and the
        // selected RLE_DICTIONARY pages then fail to decode, even though reading the
        // whole chunk (which starts at the dictionary page) works.
        let bytes = testdata("dict_mr.parquet");
        let mut fmt = ParquetFormat::new();
        let src = Source::from_bytes(bytes);
        fmt.resolve(&src).unwrap().unwrap();

        let meta = fmt.chunk(0, 2).unwrap();
        assert_eq!(meta.dictionary_page_offset, Some(meta.data_page_offset));

        // id, s -- `s` is the dictionary-encoded string column.
        let projection = vec![0usize, 2];
        let pruners = vec![Pruner {
            column: 0,
            op: PruneOp::Gt,
            value: Value::I32(3900),
            in_values: Vec::new(),
        }];
        assert!(fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap());
        let plan = fmt.matching_plan(0, &projection).expect("page plan should be active");
        let s_plan = plan.columns[1].as_ref().expect("`s` must still be page-selected");
        assert!(
            s_plan.dict_range.is_some(),
            "the dictionary page's range must be recovered from the OffsetIndex"
        );

        let cols = fmt.read_split(&src, 0, &projection).unwrap();
        // duckdb: SELECT count(*), min(s) FROM dict_mr.parquet WHERE id > 3900 -> 99, 'v0'
        let mut n = 0usize;
        let mut min: Option<Vec<u8>> = None;
        for i in 0..cols[0].len() {
            let Value::I32(id) = cols[0].value_at(i) else { panic!("id is INTEGER") };
            if id <= 3900 {
                continue;
            }
            n += 1;
            let Value::Bytes(s) = cols[1].value_at(i) else { panic!("s is VARCHAR") };
            // The dictionary is what turns an index back into text; without the
            // dictionary page these cannot be produced at all.
            assert_eq!(s, format!("v{}", id % 50).into_bytes());
            if min.as_ref().is_none_or(|m| s < *m) {
                min = Some(s);
            }
        }
        assert_eq!(n, 99);
        assert_eq!(min.as_deref(), Some(&b"v0"[..]));
    }

    #[test]
    fn selective_equality_predicate_reads_a_small_fraction_of_the_column() {
        let bytes = pagetest_bytes();
        let mut fmt = ParquetFormat::new();
        let mut src = resolve_remote(&mut fmt, &bytes);

        let projection = vec![0usize, 1]; // id, s
        let pruners = vec![Pruner {
            column: 0,
            op: PruneOp::Eq,
            value: Value::I32(12345),
            in_values: Vec::new(),
        }];

        assert!(fmt.may_match(0, &pruners, &projection));

        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
        let index_bytes = fetch_missing(&mut src, &bytes, &idx_ranges);

        let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
        assert!(keep);
        assert!(fmt.matching_plan(0, &projection).is_some(), "page plan should be active");

        let mut data_ranges = Vec::new();
        fmt.split_ranges(0, &projection, &mut data_ranges).unwrap();
        let data_bytes = fetch_missing(&mut src, &bytes, &data_ranges);

        let cols = fmt.read_split(&src, 0, &projection).unwrap();
        assert_eq!(cols[0].len(), cols[1].len());
        // id is unique in this dataset, so exactly one row should remain in the surviving pages
        // (if page selection were not working at all, the later byte comparison would be
        // meaningless -- the result's validity is confirmed first).
        let hit = (0..cols[0].len()).find(|&i| cols[0].value_at(i) == Value::I32(12345));
        let hit = hit.expect("id=12345 must be present in the selected pages");
        assert_eq!(cols[1].value_at(hit), Value::Bytes(b"v12345".to_vec()));

        let full_id = fmt.chunk(0, 0).unwrap().total_compressed_size as u64;
        let full_s = fmt.chunk(0, 1).unwrap().total_compressed_size as u64;
        let full_total = full_id + full_s;

        // The index bytes themselves (ColumnIndex/OffsetIndex/Bloom) should be small.
        assert!(index_bytes < 128 * 1024, "index bytes unexpectedly large: {index_bytes}");
        // This is the main regression: for a selective equality predicate it must request far fewer
        // bytes than the whole column chunk.
        assert!(
            data_bytes * 10 < full_total,
            "expected order-of-magnitude reduction: fetched {data_bytes} of {full_total} bytes total"
        );
    }

    #[test]
    fn bloom_filter_skips_the_whole_split_for_a_provably_absent_value() {
        // 999_999_999 is outside the RowGroup's min/max (0..49999), so in practice `may_match`
        // (RowGroup statistics) alone already rejects it. The "within range but sparse" case only a
        // Bloom filter can catch cannot be reproduced, since this dataset covers 0..50000 with no
        // gaps (that a Bloom filter alone returns no false negatives is verified separately in
        // `parquet::bloom`'s tests). What is confirmed here is that the whole `refine_with_index`
        // path, Bloom filter check included, correctly reports a non-match as `false`.
        let bytes = pagetest_bytes();
        let mut fmt = ParquetFormat::new();
        let mut src = resolve_remote(&mut fmt, &bytes);

        let projection = vec![0usize];
        let pruners = vec![Pruner {
            column: 0,
            op: PruneOp::Eq,
            value: Value::I32(999_999_999),
            in_values: Vec::new(),
        }];

        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
        fetch_missing(&mut src, &bytes, &idx_ranges);

        let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
        assert!(!keep, "an absent value must let the whole split be skipped");
    }

    #[test]
    fn in_predicate_skips_the_split_when_every_candidate_is_absent() {
        // `IN` is an OR over several candidates, so the split can be dropped only once every
        // candidate is known absent. Every candidate is set out of range to confirm that.
        let bytes = pagetest_bytes();
        let mut fmt = ParquetFormat::new();
        let mut src = resolve_remote(&mut fmt, &bytes);

        let projection = vec![0usize];
        let pruners = vec![Pruner {
            column: 0,
            op: PruneOp::In,
            value: Value::I32(999_999_999),
            in_values: vec![Value::I32(999_999_998), Value::I32(-1)],
        }];

        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
        fetch_missing(&mut src, &bytes, &idx_ranges);

        let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
        assert!(!keep, "all candidates absent must let the whole split be skipped");
    }

    #[test]
    fn in_predicate_keeps_the_split_when_one_candidate_is_present() {
        // If just one candidate is a value that exists, the split is kept even when all the others are absent.
        let bytes = pagetest_bytes();
        let mut fmt = ParquetFormat::new();
        let mut src = resolve_remote(&mut fmt, &bytes);

        let projection = vec![0usize, 1];
        let pruners = vec![Pruner {
            column: 0,
            op: PruneOp::In,
            value: Value::I32(999_999_999),
            in_values: vec![Value::I32(12345), Value::I32(-1)],
        }];

        assert!(fmt.may_match(0, &pruners, &projection));

        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
        fetch_missing(&mut src, &bytes, &idx_ranges);

        let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
        assert!(keep, "a present candidate must keep the split");

        let mut data_ranges = Vec::new();
        fmt.split_ranges(0, &projection, &mut data_ranges).unwrap();
        let data_bytes = fetch_missing(&mut src, &bytes, &data_ranges);

        let cols = fmt.read_split(&src, 0, &projection).unwrap();
        let hit = (0..cols[0].len()).find(|&i| cols[0].value_at(i) == Value::I32(12345));
        let hit = hit.expect("id=12345 must be present in the selected pages");
        assert_eq!(cols[1].value_at(hit), Value::Bytes(b"v12345".to_vec()));

        let full_id = fmt.chunk(0, 0).unwrap().total_compressed_size as u64;
        let full_s = fmt.chunk(0, 1).unwrap().total_compressed_size as u64;
        let full_total = full_id + full_s;
        assert!(
            data_bytes * 10 < full_total,
            "IN pruning should still narrow to page granularity: fetched {data_bytes} of {full_total} bytes total"
        );
    }

    #[test]
    fn page_pruned_results_are_a_superset_containing_every_exact_match() {
        // Page selection is the same "err safe" filtering as may_match, and the selected pages may
        // contain rows that do not satisfy the predicate (exact filtering is the Filter operator
        // above's job). What is confirmed here is that "the correct set obtained without filtering
        // is contained, neither more nor less (values included), in the result after page
        // selection".
        let bytes = pagetest_bytes();

        let full: std::collections::BTreeMap<i32, Value> = {
            let mut fmt = ParquetFormat::new();
            let src = Source::from_bytes(bytes.clone());
            fmt.resolve(&src).unwrap().unwrap();
            let cols = fmt.read_split(&src, 0, &[0, 1]).unwrap();
            (0..cols[0].len())
                .filter_map(|i| match cols[0].value_at(i) {
                    Value::I32(x) if (12_000..=12_010).contains(&x) => {
                        Some((x, cols[1].value_at(i)))
                    }
                    _ => None,
                })
                .collect()
        };
        assert_eq!(full.len(), 11);

        let mut fmt = ParquetFormat::new();
        let mut src = resolve_remote(&mut fmt, &bytes);
        let projection = vec![0usize, 1];
        let pruners = vec![
            Pruner { column: 0, op: PruneOp::Ge, value: Value::I32(12_000), in_values: Vec::new() },
            Pruner { column: 0, op: PruneOp::Le, value: Value::I32(12_010), in_values: Vec::new() },
        ];
        assert!(fmt.may_match(0, &pruners, &projection));
        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
        fetch_missing(&mut src, &bytes, &idx_ranges);
        assert!(fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap());
        let mut data_ranges = Vec::new();
        fmt.split_ranges(0, &projection, &mut data_ranges).unwrap();
        fetch_missing(&mut src, &bytes, &data_ranges);
        let cols = fmt.read_split(&src, 0, &projection).unwrap();

        // Some filtering must be in effect (it is not all 50000 rows).
        assert!(cols[0].len() < 50_000);

        let mut got = std::collections::BTreeMap::new();
        for i in 0..cols[0].len() {
            if let Value::I32(x) = cols[0].value_at(i) {
                got.insert(x, cols[1].value_at(i));
            }
        }
        for (id, s) in &full {
            assert_eq!(got.get(id), Some(s), "row id={id} missing or wrong after page pruning");
        }
    }

    #[test]
    fn fallback_matches_when_the_column_lacks_a_page_index() {
        // Emulating an old file or an unsupported writer, the id column is made to have no
        // ColumnIndex/OffsetIndex. This confirms it stays on the fallback of reading whole column
        // chunks as before.
        let bytes = pagetest_bytes();
        let mut fmt = ParquetFormat::new();
        let src = Source::from_bytes(bytes.clone());
        fmt.resolve(&src).unwrap().unwrap();
        {
            let file = fmt.file.as_mut().unwrap();
            file.meta.row_groups[0].columns[0].column_index_offset = None;
            file.meta.row_groups[0].columns[0].column_index_length = None;
            file.meta.row_groups[0].columns[0].offset_index_offset = None;
            file.meta.row_groups[0].columns[0].offset_index_length = None;
        }

        let projection = vec![0usize];
        let pruners = vec![Pruner {
            column: 0,
            op: PruneOp::Eq,
            value: Value::I32(12345),
            in_values: Vec::new(),
        }];
        assert!(fmt.may_match(0, &pruners, &projection));

        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
        // Only the Bloom filter's range remains (the column metadata side is unchanged).
        assert!(!idx_ranges.is_empty());

        assert!(fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap());
        assert!(fmt.matching_plan(0, &projection).is_none(), "no page index -> no plan");

        let mut ranges = Vec::new();
        fmt.split_ranges(0, &projection, &mut ranges).unwrap();
        assert_eq!(ranges, vec![fmt.chunk(0, 0).unwrap().byte_range()]);

        let cols = fmt.read_split(&src, 0, &projection).unwrap();
        assert_eq!(cols[0].len(), 50_000, "fallback must still read the whole column");
        assert_eq!(cols[0].value_at(12345), Value::I32(12345));
    }

    #[test]
    fn corrupted_index_bytes_fall_back_instead_of_panicking() {
        let bytes = pagetest_bytes();
        let mut fmt = ParquetFormat::new();
        // The footer is resolved from all the in-memory bytes, but the `src` under test is given
        // only the index-byte ranges, as garbage. If the speculative footer fetch's range overlaps
        // the index-byte range, `Source::insert` keeps the real bytes on a "first arrival wins"
        // basis (and the deliberate corruption stops working), so a separate `Source` is prepared
        // for resolution.
        let full_src = Source::from_bytes(bytes.clone());
        fmt.resolve(&full_src).unwrap().unwrap();

        let projection = vec![0usize];
        let pruners = vec![Pruner {
            column: 0,
            op: PruneOp::Eq,
            value: Value::I32(12345),
            in_values: Vec::new(),
        }];
        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &pruners, &projection, &mut idx_ranges).unwrap();
        assert!(!idx_ranges.is_empty());

        let mut src = Source::remote(bytes.len() as u64);
        for (start, end) in &idx_ranges {
            src.insert(*start, vec![0xFFu8; (end - start) as usize]);
        }

        // It does not panic and errs safe (no filtering).
        let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
        assert!(keep);
        assert!(fmt.matching_plan(0, &projection).is_none());

        let mut ranges = Vec::new();
        fmt.split_ranges(0, &projection, &mut ranges).unwrap();
        assert_eq!(ranges, vec![fmt.chunk(0, 0).unwrap().byte_range()]);
    }

    #[test]
    fn no_pruners_never_activates_a_page_plan() {
        // A query with no pruners (`SELECT *` and the like) reads whole column chunks as before,
        // without a single extra index access.
        let bytes = pagetest_bytes();
        let mut fmt = ParquetFormat::new();
        let src = Source::from_bytes(bytes);
        fmt.resolve(&src).unwrap().unwrap();

        let projection = vec![0usize, 1];
        let mut idx_ranges = Vec::new();
        fmt.index_ranges(0, &[], &projection, &mut idx_ranges).unwrap();
        assert!(idx_ranges.is_empty());
        assert!(fmt.refine_with_index(&src, 0, &[], &projection).unwrap());
        assert!(fmt.matching_plan(0, &projection).is_none());
    }
}
