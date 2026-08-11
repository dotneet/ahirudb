//! Parquet メタデータ構造体と、その Thrift デコーダ。
//!
//! 汎用 Thrift ランタイムは持たない。構造体ごとに「必要なフィールド ID だけを
//! 読み、未知フィールドはスキップする」専用デコーダを手書きする。
//! 汎用実装 + IDL 生成コードは 100 KB 級になるが、この方式なら 45 KB に収まる
//! （DESIGN.md §5）。
//!
//! ここに無いフィールドは意図的に読み捨てている。追加するときは
//! `docs/DESIGN.md` のサイズ予算も更新すること。

use crate::parquet::thrift::{ttype, Thrift};
use crate::parquet::*;
use crate::prelude::*;

// --- 上限値 ---------------------------------------------------------------
// 敵対的な Parquet ファイルによるメモリ爆撃を防ぐ (DESIGN.md §5)。

/// スキーマ要素数の上限。
pub const MAX_SCHEMA_ELEMENTS: usize = 16_384;
/// RowGroup 数の上限。
pub const MAX_ROW_GROUPS: usize = 1_000_000;
/// 1 RowGroup あたりの列数の上限。
pub const MAX_COLUMNS: usize = 16_384;
/// list ヘッダが宣言できる要素数の上限（実バイト数と無関係な巨大値を弾く）。
pub const MAX_LIST_LEN: usize = 1 << 24;
/// 1 列チャンクあたりのページ数の上限（ColumnIndex/OffsetIndex 用）。
/// 現実のページは数百バイト以上あるので、これだけあれば十分すぎるほど余裕がある。
pub const MAX_PAGES_PER_COLUMN: usize = 1_000_000;

/// `FileMetaData`。フッタ全体。
pub struct FileMetaData {
    pub version: i32,
    /// 深さ優先順に並んだスキーマ要素。先頭はルート。
    pub schema: Vec<SchemaElement>,
    pub num_rows: i64,
    pub row_groups: Vec<RowGroup>,
    pub created_by: Option<String>,
}

/// `SchemaElement`。
pub struct SchemaElement {
    pub ptype: Option<PType>,
    pub type_length: Option<i32>,
    pub repetition: Option<Repetition>,
    pub name: String,
    pub num_children: Option<i32>,
    pub converted_type: Option<ConvertedType>,
    pub scale: Option<i32>,
    pub precision: Option<i32>,
    pub logical: Option<LogicalType>,
}

/// `RowGroup`。
pub struct RowGroup {
    pub columns: Vec<ColumnChunk>,
    pub total_byte_size: i64,
    pub num_rows: i64,
}

/// `ColumnChunk`。
pub struct ColumnChunk {
    /// 別ファイル参照。ahirudb では未対応なので `Some` ならエラーにする。
    pub file_path: Option<String>,
    pub file_offset: i64,
    pub meta: Option<ColumnMetaData>,
    pub column_index_offset: Option<i64>,
    pub column_index_length: Option<i32>,
    pub offset_index_offset: Option<i64>,
    pub offset_index_length: Option<i32>,
    /// 暗号化列。検出したら `EncryptionUnsupported` を返す。
    pub encrypted: bool,
}

/// `ColumnMetaData`。
pub struct ColumnMetaData {
    pub ptype: PType,
    pub encodings: Vec<Encoding>,
    pub path_in_schema: Vec<String>,
    pub codec: Compression,
    pub num_values: i64,
    pub total_uncompressed_size: i64,
    pub total_compressed_size: i64,
    pub data_page_offset: i64,
    pub index_page_offset: Option<i64>,
    pub dictionary_page_offset: Option<i64>,
    pub statistics: Option<Statistics>,
    pub bloom_filter_offset: Option<i64>,
    pub bloom_filter_length: Option<i32>,
}

impl ColumnMetaData {
    /// この列チャンクが占めるファイル上のバイト範囲 `[start, end)`。
    /// 辞書ページがあればそこから始まる。
    pub fn byte_range(&self) -> (u64, u64) {
        let start = match self.dictionary_page_offset {
            Some(d) if d > 0 && d < self.data_page_offset => d,
            _ => self.data_page_offset,
        };
        (start as u64, (start + self.total_compressed_size) as u64)
    }

    /// 辞書ページ（あれば）のバイト範囲 `[dictionary_page_offset, data_page_offset)`。
    /// ページ選択で辞書エンコード列を読むときに、データページと別に取得する。
    pub fn dictionary_page_range(&self) -> Option<(u64, u64)> {
        match self.dictionary_page_offset {
            Some(d) if d > 0 && d < self.data_page_offset => {
                Some((d as u64, self.data_page_offset as u64))
            }
            _ => None,
        }
    }

    /// Bloom フィルタの投機取得範囲。`bloom_filter_length`（新しめの writer が
    /// 書く、ヘッダ込みの正確な長さ）が分かっていればそれを使い、無ければ
    /// 「よくある小さいフィルタなら 1 往復で収まる」サイズを投機取得する。
    /// 実サイズがそれを超えていた場合は `refine_with_index` 側が黙って
    /// Bloom フィルタを諦める（`may_match` と同じ「判断できないなら通す」
    /// 安全側の設計）。
    pub fn bloom_filter_probe_range(&self) -> Option<(u64, u64)> {
        let off = self.bloom_filter_offset?;
        if off < 0 {
            return None;
        }
        let len = match self.bloom_filter_length {
            Some(l) if l > 0 => l as u64,
            _ => BLOOM_FILTER_PROBE,
        };
        Some((off as u64, off as u64 + len))
    }
}

/// `bloom_filter_length` が無い場合に投機取得するバイト数。ヘッダ（数十バイト）
/// + ビットセット本体を 1 往復で取れるよう、実運用のフィルタサイズより
///   十分大きく取ってある。
pub const BLOOM_FILTER_PROBE: u64 = 128 * 1024;

impl ColumnChunk {
    /// `ColumnIndex`（ページ単位の min/max/null 統計）のバイト範囲。
    /// offset/length のどちらか、または両方が無い（古いファイル・非対応
    /// writer）なら `None`。ページ単位の枝刈りをせず列チャンク全体を読む
    /// フォールバックの合図として使う。
    pub fn column_index_range(&self) -> Option<(u64, u64)> {
        byte_range_from(self.column_index_offset, self.column_index_length)
    }

    /// `OffsetIndex`（ページごとのバイト位置・先頭行番号）のバイト範囲。
    pub fn offset_index_range(&self) -> Option<(u64, u64)> {
        byte_range_from(self.offset_index_offset, self.offset_index_length)
    }
}

fn byte_range_from(offset: Option<i64>, length: Option<i32>) -> Option<(u64, u64)> {
    let off = offset?;
    let len = length?;
    if off < 0 || len < 0 {
        return None;
    }
    Some((off as u64, off as u64 + len as u64))
}

/// `Statistics`。枝刈りに使う。
///
/// `min`/`max` (フィールド 1,2) は符号の扱いが writer 依存で信用できないため
/// 読むだけに留め、枝刈りには `min_value`/`max_value` (フィールド 5,6) のみを
/// 使う。
#[derive(Default)]
pub struct Statistics {
    pub max: Option<Vec<u8>>,
    pub min: Option<Vec<u8>>,
    pub null_count: Option<i64>,
    pub distinct_count: Option<i64>,
    pub max_value: Option<Vec<u8>>,
    pub min_value: Option<Vec<u8>>,
    pub is_max_value_exact: Option<bool>,
    pub is_min_value_exact: Option<bool>,
}

/// `BoundaryOrder`。ColumnIndex のページ min/max がこの順序で並んでいるかの
/// ヒント。ページ単位の枝刈りは線形走査で十分（DESIGN.md の 1MB 予算では
/// 二分探索用の分岐コードを足すだけの価値が薄い）なので、値は保持するだけで
/// 判定ロジックには使わない。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundaryOrder {
    Unordered = 0,
    Ascending = 1,
    Descending = 2,
}

impl BoundaryOrder {
    /// 未知の値は `Unordered`（＝順序を仮定しない、安全側）として扱う。
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => BoundaryOrder::Ascending,
            2 => BoundaryOrder::Descending,
            _ => BoundaryOrder::Unordered,
        }
    }
}

/// `ColumnIndex`。列チャンク内の各ページの min/max/null 統計。
/// `RowGroup` の `Statistics` と同じ「writer が切り詰めることがある」注意点が
/// 適用されるため、枝刈りに使う値は `stat_value` を経由して物理型の幅で
/// 読める数値型に限る（文字列は使わない）。
pub struct ColumnIndex {
    /// `null_pages[i]` が真なら、そのページは統計が書かれていない
    /// （全 NULL、または writer が省略した）。`min_values[i]`/`max_values[i]`
    /// は意味を持たないので枝刈りに使わず、そのページは無条件に残す。
    pub null_pages: Vec<bool>,
    pub min_values: Vec<Vec<u8>>,
    pub max_values: Vec<Vec<u8>>,
    pub boundary_order: BoundaryOrder,
    pub null_counts: Option<Vec<i64>>,
}

/// `PageLocation`。1 ページのファイル上の位置と、そのページの先頭が
/// RowGroup 内の何行目から始まるか。
#[derive(Clone, Copy)]
pub struct PageLocation {
    /// ページヘッダを含む先頭からのファイル上オフセット。
    pub offset: i64,
    /// ページヘッダ込みの圧縮後バイト数。`[offset, offset+compressed_page_size)`
    /// が丸ごとこのページ。
    pub compressed_page_size: i32,
    /// このページの先頭行が RowGroup の先頭から数えて何行目か。
    pub first_row_index: i64,
}

/// `OffsetIndex`。列チャンク内の各ページのバイト位置と先頭行番号。
pub struct OffsetIndex {
    pub page_locations: Vec<PageLocation>,
}

/// `BloomFilterHeader`。Bloom フィルタ本体（ビットセット）の手前に置かれる。
/// アルゴリズム/ハッシュ/圧縮のいずれかが対応外なら、デコード時点で
/// `UnsupportedFeature` にする（誤って別形式のビット列を SBBF として読む
/// 事故を防ぐため。ここは「分からなければ安全側」を decode の外ではなく
/// 内側でやるべき数少ない箇所 — 対応外のアルゴリズムを読み進めても
/// 意味のある値は作れないので、早期に確実なエラーにする）。
pub struct BloomFilterHeader {
    pub num_bytes: i32,
}

/// `PageHeader`。データページの手前に置かれる。
pub struct PageHeader {
    pub ptype: PageType,
    pub uncompressed_page_size: i32,
    pub compressed_page_size: i32,
    pub crc: Option<i32>,
    pub data_page: Option<DataPageHeader>,
    pub dict_page: Option<DictionaryPageHeader>,
    pub data_page_v2: Option<DataPageHeaderV2>,
}

pub struct DataPageHeader {
    pub num_values: i32,
    pub encoding: Encoding,
    pub definition_level_encoding: Encoding,
    pub repetition_level_encoding: Encoding,
}

pub struct DataPageHeaderV2 {
    pub num_values: i32,
    pub num_nulls: i32,
    pub num_rows: i32,
    pub encoding: Encoding,
    pub definition_levels_byte_length: i32,
    pub repetition_levels_byte_length: i32,
    /// 既定は true（Parquet 仕様）。
    pub is_compressed: bool,
}

pub struct DictionaryPageHeader {
    pub num_values: i32,
    pub encoding: Encoding,
}

// --- デコーダ --------------------------------------------------------------

/// フッタ本体（`FileMetaData` の Thrift バイト列）をデコードする。
/// `buf` はマジックと長さフィールドを含まない、メタデータ本体のみ。
pub fn decode_file_metadata(buf: &[u8]) -> Result<FileMetaData> {
    let mut t = Thrift::new(buf);
    let mut md = FileMetaData {
        version: 0,
        schema: Vec::new(),
        num_rows: 0,
        row_groups: Vec::new(),
        created_by: None,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => md.version = t.read_i32(ft)?,
            2 => {
                let n = t.read_list_begin(MAX_SCHEMA_ELEMENTS)?.1;
                md.schema.reserve(n);
                for _ in 0..n {
                    md.schema.push(decode_schema_element(&mut t)?);
                }
            }
            3 => md.num_rows = t.read_i64(ft)?,
            4 => {
                let n = t.read_list_begin(MAX_ROW_GROUPS)?.1;
                md.row_groups.reserve(n.min(4096));
                for _ in 0..n {
                    md.row_groups.push(decode_row_group(&mut t)?);
                }
            }
            6 => md.created_by = Some(t.read_string(ft)?),
            8 | 9 => err!(EncryptionUnsupported, t.pos()),
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    ensure!(!md.schema.is_empty(), BadThrift, t.pos());
    Ok(md)
}

fn decode_schema_element(t: &mut Thrift) -> Result<SchemaElement> {
    let mut e = SchemaElement {
        ptype: None,
        type_length: None,
        repetition: None,
        name: String::new(),
        num_children: None,
        converted_type: None,
        scale: None,
        precision: None,
        logical: None,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => e.ptype = Some(PType::from_i32(t.read_i32(ft)?)?),
            2 => e.type_length = Some(t.read_i32(ft)?),
            3 => e.repetition = Some(Repetition::from_i32(t.read_i32(ft)?)?),
            4 => e.name = t.read_string(ft)?,
            5 => e.num_children = Some(t.read_i32(ft)?),
            6 => e.converted_type = ConvertedType::from_i32(t.read_i32(ft)?),
            7 => e.scale = Some(t.read_i32(ft)?),
            8 => e.precision = Some(t.read_i32(ft)?),
            10 => e.logical = decode_logical_type(t)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(e)
}

/// `LogicalType` union。フィールド ID がそのまま種別を表す。
fn decode_logical_type(t: &mut Thrift) -> Result<Option<LogicalType>> {
    let mut out = None;
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => {
                t.skip(ft)?;
                out = Some(LogicalType::String);
            }
            2 => {
                t.skip(ft)?;
                out = Some(LogicalType::Map);
            }
            3 => {
                t.skip(ft)?;
                out = Some(LogicalType::List);
            }
            4 => {
                t.skip(ft)?;
                out = Some(LogicalType::Enum);
            }
            5 => out = Some(decode_decimal_type(t)?),
            6 => {
                t.skip(ft)?;
                out = Some(LogicalType::Date);
            }
            7 => out = Some(decode_time_type(t, false)?),
            8 => out = Some(decode_time_type(t, true)?),
            10 => out = Some(decode_int_type(t)?),
            11 => {
                t.skip(ft)?;
                out = Some(LogicalType::Unknown);
            }
            12 => {
                t.skip(ft)?;
                out = Some(LogicalType::Json);
            }
            13 => {
                t.skip(ft)?;
                out = Some(LogicalType::Bson);
            }
            14 => {
                t.skip(ft)?;
                out = Some(LogicalType::Uuid);
            }
            15 => {
                t.skip(ft)?;
                out = Some(LogicalType::Float16);
            }
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(out)
}

fn decode_decimal_type(t: &mut Thrift) -> Result<LogicalType> {
    let (mut scale, mut precision) = (0, 0);
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => scale = t.read_i32(ft)?,
            2 => precision = t.read_i32(ft)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(LogicalType::Decimal { scale, precision })
}

fn decode_time_type(t: &mut Thrift, is_timestamp: bool) -> Result<LogicalType> {
    let mut utc = false;
    let mut unit = TimeUnit::Millis;
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => utc = t.read_bool(ft)?,
            2 => unit = decode_time_unit(t)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(if is_timestamp {
        LogicalType::Timestamp { utc, unit }
    } else {
        LogicalType::Time { utc, unit }
    })
}

fn decode_time_unit(t: &mut Thrift) -> Result<TimeUnit> {
    let mut unit = TimeUnit::Millis;
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => {
                t.skip(ft)?;
                unit = TimeUnit::Millis;
            }
            2 => {
                t.skip(ft)?;
                unit = TimeUnit::Micros;
            }
            3 => {
                t.skip(ft)?;
                unit = TimeUnit::Nanos;
            }
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(unit)
}

fn decode_int_type(t: &mut Thrift) -> Result<LogicalType> {
    let mut bit_width = 32u8;
    let mut signed = true;
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => bit_width = t.read_i8(ft)? as u8,
            2 => signed = t.read_bool(ft)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(LogicalType::Integer { bit_width, signed })
}

fn decode_row_group(t: &mut Thrift) -> Result<RowGroup> {
    let mut rg = RowGroup { columns: Vec::new(), total_byte_size: 0, num_rows: 0 };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => {
                let n = t.read_list_begin(MAX_COLUMNS)?.1;
                rg.columns.reserve(n);
                for _ in 0..n {
                    rg.columns.push(decode_column_chunk(t)?);
                }
            }
            2 => rg.total_byte_size = t.read_i64(ft)?,
            3 => rg.num_rows = t.read_i64(ft)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(rg)
}

fn decode_column_chunk(t: &mut Thrift) -> Result<ColumnChunk> {
    let mut c = ColumnChunk {
        file_path: None,
        file_offset: 0,
        meta: None,
        column_index_offset: None,
        column_index_length: None,
        offset_index_offset: None,
        offset_index_length: None,
        encrypted: false,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => c.file_path = Some(t.read_string(ft)?),
            2 => c.file_offset = t.read_i64(ft)?,
            3 => c.meta = Some(decode_column_meta(t)?),
            4 => c.offset_index_offset = Some(t.read_i64(ft)?),
            5 => c.offset_index_length = Some(t.read_i32(ft)?),
            6 => c.column_index_offset = Some(t.read_i64(ft)?),
            7 => c.column_index_length = Some(t.read_i32(ft)?),
            8 | 9 => {
                c.encrypted = true;
                t.skip(ft)?;
            }
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(c)
}

fn decode_column_meta(t: &mut Thrift) -> Result<ColumnMetaData> {
    let mut m = ColumnMetaData {
        ptype: PType::Boolean,
        encodings: Vec::new(),
        path_in_schema: Vec::new(),
        codec: Compression::Uncompressed,
        num_values: 0,
        total_uncompressed_size: 0,
        total_compressed_size: 0,
        data_page_offset: 0,
        index_page_offset: None,
        dictionary_page_offset: None,
        statistics: None,
        bloom_filter_offset: None,
        bloom_filter_length: None,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => m.ptype = PType::from_i32(t.read_i32(ft)?)?,
            2 => {
                let n = t.read_list_begin(64)?.1;
                for _ in 0..n {
                    m.encodings.push(Encoding::from_i32(t.read_i32(ttype::I32)?)?);
                }
            }
            3 => {
                let (et, n) = t.read_list_begin(MAX_LIST_LEN.min(4096))?;
                for _ in 0..n {
                    m.path_in_schema.push(t.read_string(et)?);
                }
            }
            4 => m.codec = Compression::from_i32(t.read_i32(ft)?)?,
            5 => m.num_values = t.read_i64(ft)?,
            6 => m.total_uncompressed_size = t.read_i64(ft)?,
            7 => m.total_compressed_size = t.read_i64(ft)?,
            9 => m.data_page_offset = t.read_i64(ft)?,
            10 => m.index_page_offset = Some(t.read_i64(ft)?),
            11 => m.dictionary_page_offset = Some(t.read_i64(ft)?),
            12 => m.statistics = Some(decode_statistics(t)?),
            14 => m.bloom_filter_offset = Some(t.read_i64(ft)?),
            15 => m.bloom_filter_length = Some(t.read_i32(ft)?),
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(m)
}

fn decode_statistics(t: &mut Thrift) -> Result<Statistics> {
    let mut s = Statistics::default();
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => s.max = Some(t.read_binary(ft)?.to_vec()),
            2 => s.min = Some(t.read_binary(ft)?.to_vec()),
            3 => s.null_count = Some(t.read_i64(ft)?),
            4 => s.distinct_count = Some(t.read_i64(ft)?),
            5 => s.max_value = Some(t.read_binary(ft)?.to_vec()),
            6 => s.min_value = Some(t.read_binary(ft)?.to_vec()),
            7 => s.is_max_value_exact = Some(t.read_bool(ft)?),
            8 => s.is_min_value_exact = Some(t.read_bool(ft)?),
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(s)
}

/// `ColumnIndex` 本体（`ColumnChunk.column_index_offset` が指す先頭バイトから
/// 始まる、それ単体で完結した Thrift 構造体）をデコードする。
pub fn decode_column_index(buf: &[u8]) -> Result<ColumnIndex> {
    let mut t = Thrift::new(buf);
    let mut ci = ColumnIndex {
        null_pages: Vec::new(),
        min_values: Vec::new(),
        max_values: Vec::new(),
        boundary_order: BoundaryOrder::Unordered,
        null_counts: None,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => {
                // list<bool>。要素は等しく BOOL_TRUE 固定の型ヘッダを持つので、
                // 個々の真偽は `read_bool_elem` で 1 バイトずつ読む。
                let n = t.read_list_begin(MAX_PAGES_PER_COLUMN)?.1;
                ci.null_pages.reserve(n);
                for _ in 0..n {
                    ci.null_pages.push(t.read_bool_elem()?);
                }
            }
            2 => {
                let (et, n) = t.read_list_begin(MAX_PAGES_PER_COLUMN)?;
                ci.min_values.reserve(n);
                for _ in 0..n {
                    ci.min_values.push(t.read_binary(et)?.to_vec());
                }
            }
            3 => {
                let (et, n) = t.read_list_begin(MAX_PAGES_PER_COLUMN)?;
                ci.max_values.reserve(n);
                for _ in 0..n {
                    ci.max_values.push(t.read_binary(et)?.to_vec());
                }
            }
            4 => ci.boundary_order = BoundaryOrder::from_i32(t.read_i32(ft)?),
            5 => {
                let (et, n) = t.read_list_begin(MAX_PAGES_PER_COLUMN)?;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(t.read_i64(et)?);
                }
                ci.null_counts = Some(v);
            }
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    // null_pages が基準。min_values/max_values は null なページでは空バイト列
    // のことがあるので、要素数の食い違いを許すと後段が out-of-range で
    // パニックしかねない。ここで揃っていることを確定させておく。
    ensure!(ci.min_values.len() == ci.null_pages.len(), BadThrift, t.pos());
    ensure!(ci.max_values.len() == ci.null_pages.len(), BadThrift, t.pos());
    if let Some(nc) = &ci.null_counts {
        ensure!(nc.len() == ci.null_pages.len(), BadThrift, t.pos());
    }
    Ok(ci)
}

fn decode_page_location(t: &mut Thrift) -> Result<PageLocation> {
    let mut p = PageLocation { offset: 0, compressed_page_size: 0, first_row_index: 0 };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => p.offset = t.read_i64(ft)?,
            2 => p.compressed_page_size = t.read_i32(ft)?,
            3 => p.first_row_index = t.read_i64(ft)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(p)
}

/// `OffsetIndex` 本体（`ColumnChunk.offset_index_offset` が指す先頭バイトから
/// 始まる、それ単体で完結した Thrift 構造体）をデコードする。
pub fn decode_offset_index(buf: &[u8]) -> Result<OffsetIndex> {
    let mut t = Thrift::new(buf);
    let mut oi = OffsetIndex { page_locations: Vec::new() };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => {
                let n = t.read_list_begin(MAX_PAGES_PER_COLUMN)?.1;
                oi.page_locations.reserve(n);
                for _ in 0..n {
                    oi.page_locations.push(decode_page_location(&mut t)?);
                }
            }
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    for p in &oi.page_locations {
        ensure!(p.offset >= 0, BadThrift, t.pos());
        ensure!(p.compressed_page_size >= 0, BadThrift, t.pos());
        ensure!(p.first_row_index >= 0, BadThrift, t.pos());
    }
    Ok(oi)
}

/// A Thrift union that only recognizes its field-1 variant (used by
/// `BloomFilterAlgorithm`/`BLOCK`, `BloomFilterHash`/`XXHASH`, and
/// `BloomFilterCompression`/`UNCOMPRESSED` — every other variant is
/// unsupported). Returns whether field 1 was present.
fn decode_single_variant_union(t: &mut Thrift) -> Result<bool> {
    let mut ok = false;
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => {
                t.skip(ft)?;
                ok = true;
            }
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(ok)
}

/// `BloomFilterHeader` をデコードし、`(ヘッダ, 消費バイト数)` を返す。
/// `buf` はヘッダに続いてビットセット本体も含んでいてよい（余りは無視する）。
/// `PageHeader` と同様、長さは前置されないので消費量を呼び出し側へ返す。
///
/// アルゴリズム/ハッシュ/圧縮のいずれかが `BLOCK`/`XXHASH`/`UNCOMPRESSED` で
/// ないファイルは `UnsupportedFeature` にする。誤って未知の形式をビット列
/// として読み、"false" 判定を返してしまう（誤ってページを丸ごと読み飛ばす
/// = 行が消える）事故を避けるため。
pub fn decode_bloom_filter_header(buf: &[u8]) -> Result<(BloomFilterHeader, usize)> {
    let mut t = Thrift::new(buf);
    let mut num_bytes: Option<i32> = None;
    let mut algo_ok = false;
    let mut hash_ok = false;
    let mut comp_ok = false;
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => num_bytes = Some(t.read_i32(ft)?),
            // Field 2: BloomFilterAlgorithm union, BLOCK (variant 1) only.
            2 => algo_ok = decode_single_variant_union(&mut t)?,
            // Field 3: BloomFilterHash union, XXHASH (variant 1) only.
            3 => hash_ok = decode_single_variant_union(&mut t)?,
            // Field 4: BloomFilterCompression union, UNCOMPRESSED (variant 1) only.
            4 => comp_ok = decode_single_variant_union(&mut t)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    let num_bytes = match num_bytes {
        Some(n) => n,
        None => err!(BadThrift, t.pos()),
    };
    ensure!(algo_ok && hash_ok && comp_ok, UnsupportedFeature, t.pos());
    // SBBF は 32 バイトブロックの並び。0 または 32 の倍数でなければ壊れている。
    ensure!(num_bytes > 0 && num_bytes % 32 == 0, BadThrift, t.pos());
    Ok((BloomFilterHeader { num_bytes }, t.pos()))
}

/// ページヘッダをデコードし、`(ヘッダ, 消費バイト数)` を返す。
/// ページヘッダは長さが前置されないため、消費量を呼び出し側に返す必要がある。
pub fn decode_page_header(buf: &[u8]) -> Result<(PageHeader, usize)> {
    let mut t = Thrift::new(buf);
    let mut h = PageHeader {
        ptype: PageType::DataPage,
        uncompressed_page_size: 0,
        compressed_page_size: 0,
        crc: None,
        data_page: None,
        dict_page: None,
        data_page_v2: None,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => h.ptype = PageType::from_i32(t.read_i32(ft)?)?,
            2 => h.uncompressed_page_size = t.read_i32(ft)?,
            3 => h.compressed_page_size = t.read_i32(ft)?,
            4 => h.crc = Some(t.read_i32(ft)?),
            5 => h.data_page = Some(decode_data_page_header(&mut t)?),
            7 => h.dict_page = Some(decode_dict_page_header(&mut t)?),
            8 => h.data_page_v2 = Some(decode_data_page_header_v2(&mut t)?),
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    ensure!(h.compressed_page_size >= 0, BadPageHeader, t.pos());
    ensure!(h.uncompressed_page_size >= 0, BadPageHeader, t.pos());
    Ok((h, t.pos()))
}

fn decode_data_page_header(t: &mut Thrift) -> Result<DataPageHeader> {
    let mut d = DataPageHeader {
        num_values: 0,
        encoding: Encoding::Plain,
        definition_level_encoding: Encoding::Rle,
        repetition_level_encoding: Encoding::Rle,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => d.num_values = t.read_i32(ft)?,
            2 => d.encoding = Encoding::from_i32(t.read_i32(ft)?)?,
            3 => d.definition_level_encoding = Encoding::from_i32(t.read_i32(ft)?)?,
            4 => d.repetition_level_encoding = Encoding::from_i32(t.read_i32(ft)?)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(d)
}

fn decode_data_page_header_v2(t: &mut Thrift) -> Result<DataPageHeaderV2> {
    let mut d = DataPageHeaderV2 {
        num_values: 0,
        num_nulls: 0,
        num_rows: 0,
        encoding: Encoding::Plain,
        definition_levels_byte_length: 0,
        repetition_levels_byte_length: 0,
        is_compressed: true,
    };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => d.num_values = t.read_i32(ft)?,
            2 => d.num_nulls = t.read_i32(ft)?,
            3 => d.num_rows = t.read_i32(ft)?,
            4 => d.encoding = Encoding::from_i32(t.read_i32(ft)?)?,
            5 => d.definition_levels_byte_length = t.read_i32(ft)?,
            6 => d.repetition_levels_byte_length = t.read_i32(ft)?,
            7 => d.is_compressed = t.read_bool(ft)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(d)
}

fn decode_dict_page_header(t: &mut Thrift) -> Result<DictionaryPageHeader> {
    let mut d = DictionaryPageHeader { num_values: 0, encoding: Encoding::Plain };
    t.enter()?;
    while let Some((ft, id)) = t.read_field_begin()? {
        match id {
            1 => d.num_values = t.read_i32(ft)?,
            2 => d.encoding = Encoding::from_i32(t.read_i32(ft)?)?,
            _ => t.skip(ft)?,
        }
    }
    t.leave();
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::code_of;

    // --- 手組みの Thrift Compact エンコーダ（テスト専用） ----------------------
    // 本番デコーダとは独立にバイト列を組み立てて往復させることで、デコーダの
    // 実装をそのままなぞるだけのテストにならないようにする。

    fn field_hdr(delta: i16, ttype: u8) -> u8 {
        ((delta as u8) << 4) | ttype
    }

    fn push_uvarint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
    }

    fn push_zigzag(out: &mut Vec<u8>, v: i64) {
        let u = ((v << 1) ^ (v >> 63)) as u64;
        push_uvarint(out, u);
    }

    fn push_binary(out: &mut Vec<u8>, b: &[u8]) {
        push_uvarint(out, b.len() as u64);
        out.extend_from_slice(b);
    }

    fn push_list_hdr(out: &mut Vec<u8>, etype: u8, n: usize) {
        if n < 15 {
            out.push(((n as u8) << 4) | etype);
        } else {
            out.push(0xf0 | etype);
            push_uvarint(out, n as u64);
        }
    }

    /// `BLOCK`/`XXHASH`/`UNCOMPRESSED` を選んだ、対応済みの `BloomFilterHeader`
    /// を組み立てる。
    fn encode_bloom_header(num_bytes: i32) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::I32));
        push_zigzag(&mut out, num_bytes as i64);
        // algorithm (id=2, struct) { BLOCK(id=1, 空struct) { STOP } STOP }
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        // hash (id=3, struct) { XXHASH(id=1, 空struct) { STOP } STOP }
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        // compression (id=4, struct) { UNCOMPRESSED(id=1, 空struct) { STOP } STOP }
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        out
    }

    #[test]
    fn bloom_header_round_trips() {
        let buf = encode_bloom_header(64);
        let (h, used) = decode_bloom_filter_header(&buf).unwrap();
        assert_eq!(h.num_bytes, 64);
        assert_eq!(used, buf.len());
    }

    #[test]
    fn bloom_header_extra_bytes_after_header_are_left_unconsumed() {
        let mut buf = encode_bloom_header(32);
        let header_len = buf.len();
        buf.extend_from_slice(&[0xAAu8; 32]); // ビットセット本体のつもり
        let (h, used) = decode_bloom_filter_header(&buf).unwrap();
        assert_eq!(h.num_bytes, 32);
        assert_eq!(used, header_len);
    }

    #[test]
    fn bloom_header_rejects_non_multiple_of_32() {
        let buf = encode_bloom_header(33);
        assert_eq!(code_of(decode_bloom_filter_header(&buf)), Some(Code::BadThrift));
    }

    #[test]
    fn bloom_header_rejects_unsupported_algorithm() {
        // algorithm を id=1 ではなく未知の id=9（空 struct）にする → 非対応。
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::I32));
        push_zigzag(&mut out, 64);
        out.push(field_hdr(1, ttype::STRUCT)); // algorithm
        out.push(field_hdr(9, ttype::STRUCT)); // 未知の代替アルゴリズム
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        out.push(field_hdr(1, ttype::STRUCT)); // hash: XXHASH
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        out.push(field_hdr(1, ttype::STRUCT)); // compression: UNCOMPRESSED
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        assert_eq!(code_of(decode_bloom_filter_header(&out)), Some(Code::UnsupportedFeature));
    }

    #[test]
    fn bloom_header_truncated_input_is_an_error_not_a_panic() {
        let buf = encode_bloom_header(64);
        for cut in 0..buf.len() {
            let r = decode_bloom_filter_header(&buf[..cut]);
            assert!(r.is_err(), "cut at {cut} should fail, not silently succeed");
        }
    }

    fn encode_column_index(
        null_pages: &[bool],
        min_values: &[&[u8]],
        max_values: &[&[u8]],
        boundary_order: i32,
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::LIST));
        push_list_hdr(&mut out, ttype::BOOL_TRUE, null_pages.len());
        for &b in null_pages {
            out.push(if b { ttype::BOOL_TRUE } else { ttype::BOOL_FALSE });
        }
        out.push(field_hdr(1, ttype::LIST));
        push_list_hdr(&mut out, ttype::BINARY, min_values.len());
        for v in min_values {
            push_binary(&mut out, v);
        }
        out.push(field_hdr(1, ttype::LIST));
        push_list_hdr(&mut out, ttype::BINARY, max_values.len());
        for v in max_values {
            push_binary(&mut out, v);
        }
        out.push(field_hdr(1, ttype::I32));
        push_zigzag(&mut out, boundary_order as i64);
        out.push(ttype::STOP);
        out
    }

    #[test]
    fn column_index_round_trips() {
        let min0 = 10i32.to_le_bytes();
        let max0 = 20i32.to_le_bytes();
        let buf = encode_column_index(&[false, true], &[&min0, &[]], &[&max0, &[]], 1);
        let ci = decode_column_index(&buf).unwrap();
        assert_eq!(ci.null_pages, vec![false, true]);
        assert_eq!(ci.min_values[0], min0.to_vec());
        assert_eq!(ci.max_values[0], max0.to_vec());
        assert_eq!(ci.boundary_order, BoundaryOrder::Ascending);
    }

    #[test]
    fn column_index_unknown_boundary_order_falls_back_to_unordered() {
        let buf = encode_column_index(&[false], &[&[]], &[&[]], 99);
        let ci = decode_column_index(&buf).unwrap();
        assert_eq!(ci.boundary_order, BoundaryOrder::Unordered);
    }

    #[test]
    fn column_index_mismatched_list_lengths_are_rejected() {
        // null_pages は 2 要素だが min_values は 1 要素しかない → 壊れている。
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::LIST));
        push_list_hdr(&mut out, ttype::BOOL_TRUE, 2);
        out.push(ttype::BOOL_FALSE);
        out.push(ttype::BOOL_FALSE);
        out.push(field_hdr(1, ttype::LIST));
        push_list_hdr(&mut out, ttype::BINARY, 1);
        push_binary(&mut out, &[]);
        out.push(field_hdr(1, ttype::LIST));
        push_list_hdr(&mut out, ttype::BINARY, 2);
        push_binary(&mut out, &[]);
        push_binary(&mut out, &[]);
        out.push(ttype::STOP);
        assert_eq!(code_of(decode_column_index(&out)), Some(Code::BadThrift));
    }

    #[test]
    fn column_index_truncated_input_is_an_error_not_a_panic() {
        let buf =
            encode_column_index(&[false, true, false], &[&[1], &[], &[3]], &[&[9], &[], &[11]], 1);
        for cut in 0..buf.len() {
            assert!(decode_column_index(&buf[..cut]).is_err());
        }
    }

    fn encode_offset_index(locs: &[(i64, i32, i64)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::LIST));
        push_list_hdr(&mut out, ttype::STRUCT, locs.len());
        for &(offset, size, first_row) in locs {
            out.push(field_hdr(1, ttype::I64));
            push_zigzag(&mut out, offset);
            out.push(field_hdr(1, ttype::I32));
            push_zigzag(&mut out, size as i64);
            out.push(field_hdr(1, ttype::I64));
            push_zigzag(&mut out, first_row);
            out.push(ttype::STOP);
        }
        out.push(ttype::STOP);
        out
    }

    #[test]
    fn offset_index_round_trips() {
        let buf = encode_offset_index(&[(4, 1000, 0), (1004, 900, 200)]);
        let oi = decode_offset_index(&buf).unwrap();
        assert_eq!(oi.page_locations.len(), 2);
        assert_eq!(oi.page_locations[0].offset, 4);
        assert_eq!(oi.page_locations[0].compressed_page_size, 1000);
        assert_eq!(oi.page_locations[0].first_row_index, 0);
        assert_eq!(oi.page_locations[1].offset, 1004);
        assert_eq!(oi.page_locations[1].first_row_index, 200);
    }

    #[test]
    fn offset_index_rejects_negative_fields() {
        let buf = encode_offset_index(&[(-1, 10, 0)]);
        assert_eq!(code_of(decode_offset_index(&buf)), Some(Code::BadThrift));
    }

    #[test]
    fn offset_index_truncated_input_is_an_error_not_a_panic() {
        let buf = encode_offset_index(&[(4, 1000, 0), (1004, 900, 200), (1904, 700, 400)]);
        for cut in 0..buf.len() {
            assert!(decode_offset_index(&buf[..cut]).is_err());
        }
    }

    #[test]
    fn declared_list_length_far_beyond_buffer_is_rejected_not_oom() {
        // list ヘッダで巨大な要素数を宣言しつつ、実バイトは全く無い。
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::LIST));
        out.push(0xf0 | ttype::BOOL_TRUE);
        push_uvarint(&mut out, 10_000_000);
        assert!(decode_column_index(&out).is_err());
    }

    // --- 実ファイル（pyarrow/parquet-cpp 書き出し）を介した end-to-end 検証 ---
    //
    // DuckDB はこの環境で ColumnIndex/OffsetIndex は書くが Bloom フィルタは
    // 書かない（`scripts/gen-testdata.sh` 参照）。`tests/data/pagetest.parquet`
    // は pyarrow (`write_page_index=True`, `bloom_filter_options`) で生成した、
    // 3 つとも揃っているファイル。Bloom フィルタの正しさは Python の
    // `xxhash` パッケージで独立に実装した SBBF 照合と比較済み（該当する
    // true/false の具体例をそのままここに書き写している）。

    fn pagetest_bytes() -> Vec<u8> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/pagetest.parquet");
        std::fs::read(p).unwrap_or_else(|e| panic!("tests/data/pagetest.parquet: {e}"))
    }

    #[test]
    fn column_index_and_offset_index_partition_the_column_chunk_without_gaps() {
        let bytes = pagetest_bytes();
        let f = crate::parquet::file::open_bytes(&bytes).expect("open pagetest.parquet");
        let rg = &f.meta.row_groups[0];
        let id_col = &rg.columns[0]; // id: INT32, 先頭列
        let meta = id_col.meta.as_ref().unwrap();

        let (ci_start, ci_end) = id_col.column_index_range().expect("id has a ColumnIndex");
        let (oi_start, oi_end) = id_col.offset_index_range().expect("id has an OffsetIndex");
        let ci = decode_column_index(&bytes[ci_start as usize..ci_end as usize]).unwrap();
        let oi = decode_offset_index(&bytes[oi_start as usize..oi_end as usize]).unwrap();

        assert_eq!(ci.null_pages.len(), oi.page_locations.len());
        assert!(oi.page_locations.len() > 1, "test file must have multiple pages to be meaningful");

        // ページはファイル上で連続し、列チャンクの範囲をちょうど覆う
        // （すきま・重なりが無い）。先頭行番号も単調増加で、最初のページは
        // 0 行目から始まる。
        let (chunk_start, chunk_end) = meta.byte_range();
        let mut want_offset = chunk_start as i64;
        let mut prev_row = -1i64;
        for loc in &oi.page_locations {
            assert_eq!(loc.offset, want_offset, "pages must be contiguous, no gaps/overlaps");
            assert!(loc.first_row_index > prev_row, "first_row_index must strictly increase");
            want_offset += loc.compressed_page_size as i64;
            prev_row = loc.first_row_index;
        }
        assert_eq!(oi.page_locations[0].first_row_index, 0);
        assert!(prev_row < rg.num_rows, "last page's first row must be within the row group");
        assert_eq!(want_offset, chunk_end as i64, "last page must end exactly at chunk end");
    }

    #[test]
    fn bloom_filter_cross_checked_against_independent_python_xxhash_computation() {
        let bytes = pagetest_bytes();
        let f = crate::parquet::file::open_bytes(&bytes).expect("open pagetest.parquet");
        let rg = &f.meta.row_groups[0];
        let meta = rg.columns[0].meta.as_ref().unwrap(); // id 列
        let (start, end) = meta.bloom_filter_probe_range().expect("id has a bloom filter");
        let probe = &bytes[start as usize..end.min(bytes.len() as u64) as usize];
        let (hdr, used) = decode_bloom_filter_header(probe).unwrap();
        let bitset = &probe[used..used + hdr.num_bytes as usize];
        let bf = crate::parquet::bloom::BloomFilter::new(bitset).unwrap();

        // Python (`xxhash` パッケージ) で同じビット列に対して独立実装した
        // SBBF 照合と突き合わせ済みの具体例。
        for id in [0i32, 1, 12345, 49999] {
            assert!(bf.contains(&id.to_le_bytes()), "id={id} was inserted, must never be absent");
        }
        for id in [-1_000_000i32, -999_999, -999_998, -999_997, -999_996] {
            assert!(
                !bf.contains(&id.to_le_bytes()),
                "id={id} is confirmed absent by the reference impl"
            );
        }
    }
}
