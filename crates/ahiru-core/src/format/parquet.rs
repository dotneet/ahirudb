//! Parquet を `TableFormat` に適合させるアダプタ。
//!
//! Parquet 固有の概念（RowGroup、列チャンク、Thrift 統計）はこのファイルの
//! 内側で完結する。実行エンジンからは「統計を持つ分割」に見えるだけ。

use crate::catalog::Source;
use crate::format::{
    get_or_internal, range_may_match, CodecTask, PruneOp, Pruner, ResolveStep, TableFormat,
};
use crate::parquet::bloom::BloomFilter;
use crate::parquet::file::{footer_probe_range, parse_footer, ParquetFile};
use crate::parquet::meta::{
    decode_bloom_filter_header, decode_column_index, decode_offset_index, ColumnIndex,
    ColumnMetaData, OffsetIndex,
};
use crate::parquet::nested::read_nested_column;
use crate::parquet::reader::{
    collect_codec_pages, collect_codec_pages_all, collect_codec_pages_selected, read_column_chunk,
    read_selected_pages, CodecPage,
};
use crate::parquet::schema::ColumnDesc;
use crate::parquet::PType;
use crate::prelude::*;
use crate::vector::{Field, PhysType, Ty, Value, Vector};

pub struct ParquetFormat {
    file: Option<ParquetFile>,
    /// 実行側に見せるスキーマ。`ParquetFile` から 1 度だけ写しておく。
    schema: Vec<Field>,
    /// 直近の `refine_with_index` が確定した、現在の分割のページ選択結果。
    /// `split`/`projection` が一致する呼び出しにだけ使う（一致しなければ
    /// 「絞り込み無し」の従来経路にフォールバックする）。
    page_plan: Option<PagePlan>,
}

/// 1 分割（RowGroup）ぶんのページ選択結果。
struct PagePlan {
    split: usize,
    projection: Vec<usize>,
    /// RowGroup 内の絶対行番号での「残す」区間（昇順・マージ済み・重複無し）。
    kept_ranges: Vec<(u64, u64)>,
    /// `projection` と同じ並び。列ごとに `Some` ならページ選択が効く
    /// （バイト量が減る）、`None` ならその列だけ列チャンク全体を読み、
    /// 復号後に `kept_ranges` へ gather して行数を揃える。
    columns: Vec<Option<ColumnPagePlan>>,
}

struct ColumnPagePlan {
    /// 辞書ページ（あれば）のバイト範囲。
    dict_range: Option<(u64, u64)>,
    /// 読むデータページの `(開始, 終了, 先頭行の RowGroup 内絶対行番号)`。
    /// バイトオフセット昇順。
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

    /// 論理列 `col`（`self.schema`/`f.schema.columns` の添字）の `ColumnDesc`。
    ///
    /// 入れ子列（LIST/MAP 等）は 1 本の論理列が複数の物理列チャンクを
    /// 消費するので、`col` は物理列チャンク番号とは限らない。物理番号は
    /// `ColumnDesc::phys_cols` から引く。
    fn desc(&self, col: usize) -> Result<&ColumnDesc> {
        let f = self.file()?;
        match f.schema.columns.get(col) {
            Some(d) => Ok(d),
            None => err!(Internal),
        }
    }

    /// 分割 `split` の物理列チャンク番号 `phys`（`row_group.columns` の
    /// 添字そのもの）のメタデータ。
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

    /// 分割 `split` の論理列 `col` の代表物理列チャンクのメタデータ。
    /// 統計・pruning は代表列（先頭リーフ）だけを見る簡略化で、フラット列
    /// では常に唯一の物理列と一致する。入れ子列は元々 pruning の対象外
    /// （`Ty::Json` の統計は `stat_value` が常に `None` を返す）なので、
    /// 代表列だけで安全に「判断できないので絞り込まない」に倒れる。
    fn chunk(&self, split: usize, col: usize) -> Result<&ColumnMetaData> {
        let phys = match self.desc(col)?.phys_cols.first() {
            Some(&p) => p,
            None => err!(Internal),
        };
        self.chunk_phys(split, phys)
    }

    /// 分割 `split` の論理列 `col` の代表物理列チャンクの `ColumnChunk`
    /// （ColumnIndex/OffsetIndex のオフセットはここにある）。`chunk` と同じ
    /// 代表列の簡略化。
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
            Some(rg) => Ok(rg.num_rows.max(0) as usize),
            None => err!(Internal),
        }
    }

    /// キャッシュ済みのページ選択結果を、`split`/`projection` が一致する
    /// ときだけ返す。一致しなければ「絞り込み無し」の従来経路を使うのが安全。
    fn matching_plan(&self, split: usize, projection: &[usize]) -> Option<&PagePlan> {
        let plan = self.page_plan.as_ref()?;
        if plan.split == split && plan.projection == projection {
            Some(plan)
        } else {
            None
        }
    }

    /// 論理列 `col` の全物理列チャンクの範囲を `out` に積む。フラット列は
    /// 1 個、入れ子列（LIST/MAP 等）は `ColumnDesc::phys_cols` の全リーフぶん。
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

    /// 論理列 `col` の全物理列チャンクぶん、ホスト委譲が要るページを集めて
    /// `out` に積む。入れ子列の物理リーフは REPEATED を含みうるので、
    /// `num_rows` 到達ではなくバッファを使い切ることを終了条件にする
    /// （`collect_codec_pages_all`）。フラット列は従来どおり行数駆動。
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
            push_codec_tasks(pages, out);
        }
        Ok(())
    }

    /// 入れ子列 (LIST/MAP 等) の全物理リーフをそれぞれ列チャンク全体ぶん
    /// 取得し、Dremel 組み立てで 1 本の `Ty::Json` ベクタにする。
    fn read_full_nested_column(
        &self,
        src: &Source,
        split: usize,
        desc: &ColumnDesc,
        num_rows: usize,
    ) -> Result<Vector> {
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
                    // この列は絞り込めなかった（入れ子列は常にここ）。
                    // 従来どおり列チャンク全体 ―― 入れ子列は複数ぶん。
                    None => self.push_full_chunk_ranges(split, c, out)?,
                }
            }
            return Ok(());
        }
        // 列指向なので、射影された列チャンクの範囲だけを要求する。
        // ここが「読むバイトを減らす」最も効く場所。
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
                        push_codec_tasks(&pages, out);
                    }
                    // この列は絞り込めなかった（入れ子列は常にここ）。
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

    /// RowGroup 統計 (min_value/max_value) だけで `pruners` を評価する。
    /// 統計が無い・型が分からない等、判断できない場合はその pruner を
    /// スキップする（安全側 = 「絞り込めない」扱い）。
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
        // Pruner が刺さる列だけ ColumnIndex（min/max）と Bloom フィルタを取る。
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
        // 射影された全列の OffsetIndex を取る。ページ選択で確定した「残す
        // 行範囲」を、Pruner の無い列にも適用してバイトを減らすため。
        // 入れ子列は複数の物理列チャンクにまたがりページ選択の対象外
        // （常に列チャンク全体を読む）なので、ここでは取りに行かない。
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
        // 前回のキャッシュは無条件で捨てる。この呼び出しの結果だけを信じる。
        self.page_plan = None;
        if pruners.is_empty() {
            return Ok(true);
        }

        // 1) Bloom フィルタ: 等号・IN pruner の値が確実に無いなら分割ごと落とす。
        //    RowGroup 統計の min/max では拾えない「範囲は広いが実際の値は
        //    疎」なケースを追加で拾える。IN は「候補のどれかが存在すれば残す」
        //    という OR 判定になる。
        for p in pruners {
            if p.op != PruneOp::Eq && p.op != PruneOp::In {
                continue;
            }
            let Some(&col) = projection.get(p.column) else { continue };
            // 入れ子列 (Ty::Json) には統計・Bloom フィルタによる絞り込みが
            // 意味を持たない（比較可能な物理値が 1 つに定まらない）ので、
            // 代表リーフの Bloom フィルタを誤って使わないよう明示的に除外する。
            if self.desc(col)?.nested.is_some() {
                continue;
            }
            let Ok(meta) = self.chunk(split, col) else { continue };
            let Some((start, end)) = meta.bloom_filter_probe_range() else { continue };
            let Some(buf) = src.get(start, (end - start) as usize) else { continue };
            let Ok((hdr, used)) = decode_bloom_filter_header(buf) else { continue };
            let need = used + hdr.num_bytes as usize;
            if need > buf.len() {
                // 投機取得が実サイズに届かなかった。安全側に諦める。
                continue;
            }
            let Some(bf) = BloomFilter::new(&buf[used..need]) else { continue };
            let Ok(cc) = self.column_chunk(split, col) else { continue };
            let ptype = cc.meta.as_ref().map(|m| m.ptype);
            let Some(ptype) = ptype else { continue };
            let desc_ty = self.schema.get(col).map(|f| f.ty);
            let Some(desc_ty) = desc_ty else { continue };
            let type_length =
                self.file().ok().and_then(|f| f.schema.columns.get(col)).map(|d| d.type_length);
            let Some(type_length) = type_length else { continue };

            // 候補全部がエンコードできて、かつ全部フィルタに無いと分かって
            // 初めて分割を落とせる。1 つでもエンコードできない候補があれば
            // 「あり得る」扱いにして安全側に倒す。
            let mut any_maybe_present = false;
            let mut all_encoded = true;
            for v in core::iter::once(&p.value).chain(p.in_values.iter()) {
                let Some(key) = plain_encode_for_bloom(ptype, type_length, desc_ty, v) else {
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

        // 2) ページ単位の min/max: 残る行範囲を Pruner のある列から絞り込む。
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
            let Some(ty) = self.schema.get(col).map(|f| f.ty) else { continue };
            let ranges = page_ranges_for_pruner(p, ty, &ci, &oi, num_rows);
            kept = Some(match kept {
                Some(prev) => intersect_ranges(&prev, &ranges),
                None => ranges,
            });
        }

        let Some(kept_ranges) = kept else {
            // どの pruner もページ単位で判定できなかった。以前の挙動のまま。
            return Ok(true);
        };
        if kept_ranges.is_empty() {
            // どのページも一致し得ない。RowGroup 丸ごと読み飛ばせる。
            return Ok(false);
        }

        // 3) 射影された各列について、残る行範囲を自分の OffsetIndex に写す。
        //    OffsetIndex が無い列は列チャンク全体を読み、あとで gather する。
        //    入れ子列 (LIST/MAP 等) は複数の物理列チャンクにまたがり、
        //    `ColumnPagePlan` は 1 列チャンクぶんの選択しか表現できないので、
        //    常に列チャンク全体を読ませる（`None`）。
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
                    // 選択されたページが無い＝この列は読む必要が無い。
                    // ただし空ベクタを作れるよう、フォールバックはしない。
                    return Some(ColumnPagePlan { dict_range: None, pages });
                }
                Some(ColumnPagePlan { dict_range: meta.dictionary_page_range(), pages })
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
                        // 選択されたページが 0 枚: この列に残る行は無い。
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
            // 展開済みページのキャッシュは `Source` が持つ。内蔵コーデックの
            // ファイルでは一度も引かれない。
            cols.push(read_column_chunk(desc, meta, buf, start, nrows, src)?);
        }
        Ok(cols)
    }
}

/// `CodecPage` 列を `CodecTask` に変換して `out` に積む。
fn push_codec_tasks(pages: &[CodecPage], out: &mut Vec<CodecTask>) {
    for p in pages {
        out.push(CodecTask { codec: p.codec, offset: p.offset, len: p.len, out_len: p.out_len });
    }
}

/// `[start, end)` が `ranges`（昇順・マージ済み）のどれかと重なるか。
fn ranges_overlap(ranges: &[(u64, u64)], start: u64, end: u64) -> bool {
    // ranges は小さい（RowGroup 内のページ数程度）ので線形探索で十分。
    ranges.iter().any(|&(s, e)| s < end && start < e)
}

/// 1 本の pruner から、その列のページ min/max を見て「残す」行範囲を作る。
/// 統計が使えないページ（`null_pages` や型不一致）は安全側に残す。
fn page_ranges_for_pruner(
    p: &Pruner,
    ty: Ty,
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
            // 統計が無い（全 NULL 等）ページは判定できないので残す。
            true
        } else {
            let min = ci.min_values.get(i).and_then(|b| stat_value(ty, b));
            let max = ci.max_values.get(i).and_then(|b| stat_value(ty, b));
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

/// 昇順に来る区間を、直前の区間と隣接していれば結合しながら積む。
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

/// 2 つの昇順マージ済み区間集合の積集合。両方に含まれる行だけを残す。
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

/// 昇順の絶対行番号 `abs_rows` のうち `kept`（昇順・マージ済み）に含まれる
/// 位置だけを、`abs_rows` 上の添字（= 復号済みベクタの添字）として返す。
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

/// `0..nrows` の連番に対して `kept` 区間だけを展開した添字列。
/// 列チャンク全体を復号した（絶対行番号 = 0..nrows の連番の）フォールバック
/// 列に使う。
fn select_indices_full(nrows: usize, kept: &[(u64, u64)]) -> Vec<u32> {
    let mut out = Vec::new();
    for &(s, e) in kept {
        let s = s.min(nrows as u64) as u32;
        let e = e.min(nrows as u64) as u32;
        out.extend(s..e);
    }
    out
}

/// `Value` を、その列の Parquet 物理型に合わせた PLAIN バイト列にする。
/// Bloom フィルタは物理型のバイト列に対してハッシュを取るので、
/// 論理型向けに広げた `Value` を巻き戻す必要がある。
///
/// 変換の正しさを確認できていない組み合わせ（BOOLEAN、INT96、
/// FIXED_LEN_BYTE_ARRAY の DECIMAL 等）は `None` を返して Bloom 判定を諦める
/// （安全側 = 「多分一致する」扱い）。widening の対応表は
/// `format::parquet::stat_value` の逆写像で、そこでカバーされている
/// INT32/INT64/FLOAT/DOUBLE/BYTE_ARRAY だけを扱う。
fn plain_encode_for_bloom(ptype: PType, type_length: usize, ty: Ty, v: &Value) -> Option<Vec<u8>> {
    let _ = ty;
    match ptype {
        PType::Int32 => match v {
            Value::I32(x) => Some(x.to_le_bytes().to_vec()),
            Value::I64(x) => i32::try_from(*x).ok().map(|x| x.to_le_bytes().to_vec()),
            _ => None,
        },
        PType::Int64 => match v {
            Value::I64(x) => Some(x.to_le_bytes().to_vec()),
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
            Value::Bytes(b) if b.len() == type_length => Some(b.clone()),
            _ => None,
        },
        // BOOLEAN・INT96 は widening の逆変換が自明でない
        // （INT96 はエポック起点のマイクロ秒へ変換済みで巻き戻しが非自明）ため、
        // 誤判定を避けて Bloom フィルタを使わない。
        PType::Boolean | PType::Int96 => None,
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

    // --- ページ単位の絞り込み（実ファイル: tests/data/pagetest.parquet）------
    //
    // pyarrow (parquet-cpp) が ColumnIndex/OffsetIndex/Bloom フィルタ付きで
    // 書いた、id 0..50000 の昇順ファイル。`scripts/gen-testdata.sh` に生成
    // 方法を記している。DuckDB のこの環境の版は Bloom フィルタを書けない
    // ため、実 writer 検証にはこちらを使う。

    fn pagetest_bytes() -> Vec<u8> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/pagetest.parquet");
        std::fs::read(p).unwrap_or_else(|e| panic!("tests/data/pagetest.parquet: {e}"))
    }

    /// `Source::remote` に対し、`ranges` のうち未取得の部分をホストから
    /// 取ってきたことにして登録する。実際に転送したバイト数の合計を返す
    /// （`NEED_IO` で要求されるバイト数に相当する）。
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

    /// フッタ解決を「投機取得 → 足りなければ再取得」のループで
    /// `Source::remote` 上で行う（本番の I/O 往復と同じ形）。
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
        // このデータセットは id が一意なので、ちょうど 1 行だけ生存ページに
        // 残っているはず（ページ選択がそもそも効いていないと、この後の
        // バイト比較が意味を持たない ―― まず結果の妥当性を先に確認する）。
        let hit = (0..cols[0].len()).find(|&i| cols[0].value_at(i) == Value::I32(12345));
        let hit = hit.expect("id=12345 must be present in the selected pages");
        assert_eq!(cols[1].value_at(hit), Value::Bytes(b"v12345".to_vec()));

        let full_id = fmt.chunk(0, 0).unwrap().total_compressed_size as u64;
        let full_s = fmt.chunk(0, 1).unwrap().total_compressed_size as u64;
        let full_total = full_id + full_s;

        // 索引バイト（ColumnIndex/OffsetIndex/Bloom）自体は小さいはず。
        assert!(index_bytes < 128 * 1024, "index bytes unexpectedly large: {index_bytes}");
        // これが本命の回帰: 選択的な等号述語では、列チャンク全体よりずっと
        // 少ないバイト数しか要求してはならない。
        assert!(
            data_bytes * 10 < full_total,
            "expected order-of-magnitude reduction: fetched {data_bytes} of {full_total} bytes total"
        );
    }

    #[test]
    fn bloom_filter_skips_the_whole_split_for_a_provably_absent_value() {
        // 999_999_999 は RowGroup の min/max (0..49999) の外なので、実際には
        // `may_match`（RowGroup 統計）だけでも落ちる。「範囲内だが疎」という
        // Bloom フィルタでしか拾えないケースは、このデータセットが 0..50000
        // を隙間無く含むため再現できない（単体の Bloom フィルタが偽陰性を
        // 返さないことは `parquet::bloom` のテストで別途検証済み）。ここでは
        // Bloom フィルタ照合を含む `refine_with_index` の経路全体が、
        // 不一致を正しく `false` として伝えることを確認する。
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
        // `IN` は複数候補の OR なので、全候補が「無い」と分かって初めて
        // 分割を落とせる。全候補を範囲外の値にして確認する。
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
        // 候補の 1 つだけが実在する値なら、他が全部無くても分割は残す。
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
        // ページ選択は may_match と同じ「安全側」の絞り込みで、選ばれた
        // ページの中に述語に一致しない行が混ざっていてもよい（正確な絞り込み
        // は上位の Filter オペレータの仕事）。ここでは「絞り込み無しで得た
        // 正解集合が、ページ選択後の結果に過不足なく（値も含めて）
        // 含まれること」を確認する。
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

        // 何かしら絞り込みが効いていること（50000 行丸ごとではない）。
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
        // 古いファイル・非対応 writer を模して、id 列だけ ColumnIndex/
        // OffsetIndex が無いことにする。以前どおり列チャンク全体を読む
        // フォールバックのままであることを確認する。
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
        // Bloom フィルタの範囲だけは残る（列メタデータ側は変えていないため）。
        assert!(!idx_ranges.is_empty());

        assert!(fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap());
        assert!(fmt.matching_plan(0, &projection).is_none(), "no page index → no plan");

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
        // フッタはメモリ上の全バイトから解決するが、テスト対象の `src` には
        // 索引バイトの範囲だけをゴミとして与える。フッタ投機取得の範囲と
        // 索引バイトの範囲が重なっていると `Source::insert` は「先着優先」
        // で本物のバイトを残してしまう（意図的な壊し方が効かなくなる）ため、
        // 解決用の `Source` とは別に用意する。
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

        // パニックせず、安全側（絞り込み無し）に倒れる。
        let keep = fmt.refine_with_index(&src, 0, &pruners, &projection).unwrap();
        assert!(keep);
        assert!(fmt.matching_plan(0, &projection).is_none());

        let mut ranges = Vec::new();
        fmt.split_ranges(0, &projection, &mut ranges).unwrap();
        assert_eq!(ranges, vec![fmt.chunk(0, 0).unwrap().byte_range()]);
    }

    #[test]
    fn no_pruners_never_activates_a_page_plan() {
        // pruners が無いクエリ（`SELECT *` など）は、以前どおり 1 バイトも
        // 余分な索引アクセスをせずに列チャンク全体を読む。
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
