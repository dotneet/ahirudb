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
