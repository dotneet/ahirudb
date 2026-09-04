//! Parquet metadata structs, and their Thrift decoder.
//!
//! There is no general-purpose Thrift runtime here. For each struct, a dedicated
//! decoder is handwritten that "reads only the field IDs it needs and skips
//! unknown fields." A generic implementation plus IDL-generated code would be on
//! the order of 100 KB, but this approach fits in 45 KB (DESIGN.md §5).
//!
//! Fields not present here are intentionally dropped. When adding one, also
//! update the size budget in `docs/DESIGN.md`.

use crate::parquet::thrift::{ttype, Thrift};
use crate::parquet::*;
use crate::prelude::*;

// --- Limits ---------------------------------------------------------------
// Guards against a memory bombing attack via an adversarial Parquet file (DESIGN.md §5).

/// Upper bound on the number of schema elements.
pub const MAX_SCHEMA_ELEMENTS: usize = 16_384;
/// Upper bound on the number of RowGroups.
pub const MAX_ROW_GROUPS: usize = 1_000_000;
/// Upper bound on the number of columns per RowGroup.
pub const MAX_COLUMNS: usize = 16_384;
/// Upper bound on the element count a list header may declare (rejects huge values unrelated to the actual byte count).
pub const MAX_LIST_LEN: usize = 1 << 24;
/// Upper bound on the number of pages per column chunk (for ColumnIndex/OffsetIndex).
/// Real pages are at least a few hundred bytes, so this leaves more than enough headroom.
pub const MAX_PAGES_PER_COLUMN: usize = 1_000_000;

/// `FileMetaData`. The entire footer.
pub struct FileMetaData {
    pub version: i32,
    /// Schema elements in depth-first order. The first element is the root.
    pub schema: Vec<SchemaElement>,
    pub num_rows: i64,
    pub row_groups: Vec<RowGroup>,
    pub created_by: Option<String>,
}

/// `SchemaElement`.
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

/// `RowGroup`.
pub struct RowGroup {
    pub columns: Vec<ColumnChunk>,
    pub total_byte_size: i64,
    pub num_rows: i64,
}

/// `ColumnChunk`.
pub struct ColumnChunk {
    /// A reference to a separate file. Not supported by ahirudb, so `Some` is treated as an error.
    pub file_path: Option<String>,
    pub file_offset: i64,
    pub meta: Option<ColumnMetaData>,
    pub column_index_offset: Option<i64>,
    pub column_index_length: Option<i32>,
    pub offset_index_offset: Option<i64>,
    pub offset_index_length: Option<i32>,
    /// An encrypted column. Returns `EncryptionUnsupported` when detected.
    pub encrypted: bool,
}

/// `ColumnMetaData`.
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
    /// The byte range `[start, end)` this column chunk occupies in the file.
    /// If a dictionary page exists, the range starts there.
    ///
    /// `data_page_offset` cannot be assumed to be greater than
    /// `dictionary_page_offset`: parquet-mr (Spark/Hive) records the chunk start as
    /// `data_page_offset` before writing the dictionary page, so the two are equal
    /// there, and pyarrow writes `data_page_offset = 0` for a 0-row row group. In
    /// both cases the chunk still begins at the dictionary page.
    pub fn byte_range(&self) -> (u64, u64) {
        let start = match self.dictionary_page_offset {
            Some(d) if d > 0 => {
                if self.data_page_offset > 0 {
                    d.min(self.data_page_offset)
                } else {
                    d
                }
            }
            _ => self.data_page_offset.max(0),
        };
        let len = self.total_compressed_size.max(0);
        let end = (start as u64).saturating_add(len as u64);
        (start as u64, end)
    }

    /// Whether this chunk carries a dictionary page. Such a chunk must never have its
    /// data pages decoded without the dictionary page alongside them.
    pub fn has_dictionary(&self) -> bool {
        self.dictionary_page_offset.is_some() || self.encodings.iter().any(|e| e.is_dictionary())
    }

    /// The byte range `[dictionary_page_offset, first_page_offset)` of the dictionary
    /// page, where `first_page_offset` is the file offset of the chunk's first data
    /// page as recorded in the OffsetIndex. Fetched separately from the data pages when
    /// page selection needs to read a dictionary-encoded column.
    ///
    /// `data_page_offset` cannot stand in for the first data page's offset (see
    /// `byte_range`). `None` means the range could not be determined; the caller must
    /// then fall back to reading the whole column chunk.
    pub fn dictionary_page_range(&self, first_page_offset: i64) -> Option<(u64, u64)> {
        let d = self.dictionary_page_offset.filter(|&d| d > 0)?;
        if first_page_offset <= d {
            return None;
        }
        Some((d as u64, first_page_offset as u64))
    }

    /// The speculative fetch range for the Bloom filter. If `bloom_filter_length`
    /// (an exact length including the header, written by newer writers) is known, it
    /// is used; otherwise, a size that "a typical small filter should fit within one
    /// round trip" is speculatively fetched. If the real size exceeds that,
    /// `refine_with_index` silently gives up on the Bloom filter (the same
    /// safe-by-default design as `may_match`: "let it through when it can't be
    /// determined").
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

/// Number of bytes to speculatively fetch when `bloom_filter_length` is absent.
/// Sized generously above typical real-world filter sizes so that the header (a few
/// dozen bytes) plus the bitset body can be fetched in one round trip.
pub const BLOOM_FILTER_PROBE: u64 = 128 * 1024;

impl ColumnChunk {
    /// The byte range of the `ColumnIndex` (per-page min/max/null statistics).
    /// `None` if either or both of offset/length are absent (an older file or an
    /// unsupported writer). Used as a signal to fall back to reading the whole
    /// column chunk without per-page pruning.
    pub fn column_index_range(&self) -> Option<(u64, u64)> {
        byte_range_from(self.column_index_offset, self.column_index_length)
    }

    /// The byte range of the `OffsetIndex` (per-page byte position and first row number).
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

/// `Statistics`. Used for pruning.
///
/// `min`/`max` (fields 1, 2) are read but never used for pruning, since their sign
/// handling is writer-dependent and untrustworthy; pruning uses only
/// `min_value`/`max_value` (fields 5, 6).
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

/// `BoundaryOrder`. A hint as to whether the per-page min/max values in a
/// ColumnIndex are sorted in this order. Since a linear scan is good enough for
/// per-page pruning (in DESIGN.md's 1 MB budget, adding branch code for binary
/// search has little value), the value is only kept, never used in the pruning
/// logic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundaryOrder {
    Unordered = 0,
    Ascending = 1,
    Descending = 2,
}

impl BoundaryOrder {
    /// An unknown value is treated as `Unordered` (i.e. assume no ordering -- the safe default).
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => BoundaryOrder::Ascending,
            2 => BoundaryOrder::Descending,
            _ => BoundaryOrder::Unordered,
        }
    }
}

/// `ColumnIndex`. Per-page min/max/null statistics within a column chunk.
/// The same caveat as `RowGroup`'s `Statistics` applies -- "a writer may truncate
/// it" -- so the values used for pruning are limited, via `stat_value`, to numeric
/// types that can be read at the physical type's width (strings are not used).
pub struct ColumnIndex {
    /// If `null_pages[i]` is true, that page has no statistics written (either
    /// entirely NULL, or omitted by the writer). `min_values[i]`/`max_values[i]` are
    /// meaningless in that case and are not used for pruning; that page is always
    /// kept unconditionally.
    pub null_pages: Vec<bool>,
    pub min_values: Vec<Vec<u8>>,
    pub max_values: Vec<Vec<u8>>,
    pub boundary_order: BoundaryOrder,
    pub null_counts: Option<Vec<i64>>,
}

/// `PageLocation`. A single page's position in the file, and which row within the
/// RowGroup its first row starts at.
#[derive(Clone, Copy)]
pub struct PageLocation {
    /// The file offset from the start, including the page header.
    pub offset: i64,
    /// The compressed byte size including the page header. `[offset,
    /// offset+compressed_page_size)` is this entire page.
    pub compressed_page_size: i32,
    /// Which row, counted from the start of the RowGroup, this page's first row is.
    pub first_row_index: i64,
}

/// `OffsetIndex`. Byte position and first row number for each page in a column chunk.
pub struct OffsetIndex {
    pub page_locations: Vec<PageLocation>,
}

/// `BloomFilterHeader`. Placed just before the Bloom filter body (the bitset).
/// If the algorithm/hash/compression is anything unsupported, this is treated as
/// `UnsupportedFeature` at decode time (to prevent the accident of misreading a
/// bit stream in a different format as an SBBF. This is one of the few places
/// where "when unsure, be safe" should happen inside `decode` rather than outside
/// it -- continuing to read past an unsupported algorithm can never produce a
/// meaningful value, so it's turned into a definite error early).
pub struct BloomFilterHeader {
    pub num_bytes: i32,
}

/// `PageHeader`. Placed just before a data page.
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
    /// Defaults to true (per the Parquet spec).
    pub is_compressed: bool,
}

pub struct DictionaryPageHeader {
    pub num_values: i32,
    pub encoding: Encoding,
}

// --- Decoders ---------------------------------------------------------------

/// Decodes the footer body (the Thrift byte stream for `FileMetaData`).
/// `buf` is just the metadata body, not including the magic bytes or the length field.
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

/// The `LogicalType` union. The field ID directly indicates the variant.
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

/// Decodes the `ColumnIndex` body (a self-contained Thrift struct starting at the
/// byte offset pointed to by `ColumnChunk.column_index_offset`).
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
                // list<bool>. Every element shares the same fixed BOOL_TRUE type header,
                // so each individual boolean is read one byte at a time via `read_bool_elem`.
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
    // null_pages is the reference. min_values/max_values may be an empty byte string
    // for a null page, so allowing a mismatched element count risks a later
    // out-of-range panic. Confirm here that they line up.
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

/// Decodes the `OffsetIndex` body (a self-contained Thrift struct starting at the
/// byte offset pointed to by `ColumnChunk.offset_index_offset`).
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
    // OffsetIndex entries are consumed in this order by page selection. A
    // corrupted or reordered index would make the selected pages carry the
    // wrong absolute row numbers and could silently drop matching rows.
    if let Some(first) = oi.page_locations.first() {
        ensure!(first.first_row_index == 0, BadThrift, t.pos());
    }
    for pair in oi.page_locations.windows(2) {
        ensure!(pair[1].offset > pair[0].offset, BadThrift, t.pos());
        ensure!(pair[1].first_row_index > pair[0].first_row_index, BadThrift, t.pos());
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

/// Decodes a `BloomFilterHeader` and returns `(header, bytes consumed)`.
/// `buf` may also include the bitset body following the header (any excess is
/// ignored). As with `PageHeader`, the length isn't prefixed, so the amount
/// consumed is returned to the caller.
///
/// A file whose algorithm/hash/compression isn't `BLOCK`/`XXHASH`/`UNCOMPRESSED`
/// is treated as `UnsupportedFeature`. This avoids the accident of reading an
/// unknown format as a bit stream and returning a "false" match (which would
/// silently skip an entire page = rows disappearing).
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
    // An SBBF is a sequence of 32-byte blocks. If it isn't 0 or a multiple of 32, it's corrupted.
    ensure!(num_bytes > 0 && num_bytes % 32 == 0, BadThrift, t.pos());
    Ok((BloomFilterHeader { num_bytes }, t.pos()))
}

/// Decodes a page header and returns `(header, bytes consumed)`.
/// A page header's length is not prefixed, so the amount consumed must be returned to the caller.
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

    fn chunk_meta(data_page_offset: i64, dictionary_page_offset: Option<i64>) -> ColumnMetaData {
        ColumnMetaData {
            ptype: PType::Int32,
            encodings: Vec::new(),
            path_in_schema: Vec::new(),
            codec: Compression::Uncompressed,
            num_values: 0,
            total_uncompressed_size: 100,
            total_compressed_size: 100,
            data_page_offset,
            index_page_offset: None,
            dictionary_page_offset,
            statistics: None,
            bloom_filter_offset: None,
            bloom_filter_length: None,
        }
    }

    #[test]
    fn byte_range_starts_at_the_dictionary_page_in_every_writer_layout() {
        // The usual layout: the dictionary page precedes the data pages.
        assert_eq!(chunk_meta(500, Some(400)).byte_range(), (400, 500));
        // parquet-mr (Spark/Hive) records the chunk start -- i.e. the dictionary page --
        // as `data_page_offset`, so the two are equal.
        assert_eq!(chunk_meta(400, Some(400)).byte_range(), (400, 500));
        // pyarrow writes `data_page_offset = 0` for a 0-row RowGroup. Starting there
        // would hand the reader the file's `PAR1` magic instead of the column chunk.
        assert_eq!(chunk_meta(0, Some(400)).byte_range(), (400, 500));
        // No dictionary page at all.
        assert_eq!(chunk_meta(400, None).byte_range(), (400, 500));
        // A negative offset never turns into a huge unsigned range.
        assert_eq!(chunk_meta(-1, None).byte_range(), (0, 100));
    }

    #[test]
    fn dictionary_page_range_comes_from_the_first_data_page() {
        // `first_page_offset` is the OffsetIndex's first entry, which is where the data
        // pages really begin -- `data_page_offset` is not usable for this (see above).
        assert_eq!(chunk_meta(400, Some(400)).dictionary_page_range(500), Some((400, 500)));
        assert_eq!(chunk_meta(500, Some(400)).dictionary_page_range(500), Some((400, 500)));
        // Undeterminable cases must say so, so the caller falls back to the whole chunk
        // rather than decoding dictionary-encoded pages with no dictionary.
        assert_eq!(chunk_meta(400, None).dictionary_page_range(500), None);
        assert_eq!(chunk_meta(400, Some(400)).dictionary_page_range(400), None);
        assert_eq!(chunk_meta(400, Some(400)).dictionary_page_range(0), None);
    }

    #[test]
    fn has_dictionary_also_believes_the_encoding_list() {
        assert!(chunk_meta(400, Some(400)).has_dictionary());
        assert!(!chunk_meta(400, None).has_dictionary());
        let mut m = chunk_meta(400, None);
        m.encodings.push(Encoding::RleDictionary);
        assert!(m.has_dictionary());
    }

    // --- A handwritten Thrift Compact encoder (test-only) ----------------------
    // Assembles byte streams independently of the production decoder and round-trips
    // them, so the tests don't just retrace the decoder's own implementation.

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

    /// Builds a supported `BloomFilterHeader` that selects
    /// `BLOCK`/`XXHASH`/`UNCOMPRESSED`.
    fn encode_bloom_header(num_bytes: i32) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::I32));
        push_zigzag(&mut out, num_bytes as i64);
        // algorithm (id=2, struct) { BLOCK(id=1, empty struct) { STOP } STOP }
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        // hash (id=3, struct) { XXHASH(id=1, empty struct) { STOP } STOP }
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(field_hdr(1, ttype::STRUCT));
        out.push(ttype::STOP);
        out.push(ttype::STOP);
        // compression (id=4, struct) { UNCOMPRESSED(id=1, empty struct) { STOP } STOP }
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
        buf.extend_from_slice(&[0xAAu8; 32]); // stand-in for the bitset body
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
        // Turn algorithm into an unknown id=9 (empty struct) instead of id=1 -> unsupported.
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::I32));
        push_zigzag(&mut out, 64);
        out.push(field_hdr(1, ttype::STRUCT)); // algorithm
        out.push(field_hdr(9, ttype::STRUCT)); // an unknown alternate algorithm
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
        // null_pages has 2 elements but min_values has only 1 -> corrupted.
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
    fn offset_index_rejects_reordered_pages_and_missing_first_row() {
        for locs in [
            // The pages are not in file order.
            vec![(1004, 900, 200), (4, 1000, 0)],
            // The pages are not in row order.
            vec![(4, 1000, 200), (1004, 900, 0)],
            // A page index that starts after row zero cannot safely drive pruning.
            vec![(4, 1000, 1)],
            // Equal first-row positions cannot describe two consecutive pages.
            vec![(4, 1000, 0), (1004, 900, 0)],
        ] {
            assert_eq!(
                code_of(decode_offset_index(&encode_offset_index(&locs))),
                Some(Code::BadThrift)
            );
        }
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
        // Declares a huge element count in the list header, with no real bytes at all.
        let mut out = Vec::new();
        out.push(field_hdr(1, ttype::LIST));
        out.push(0xf0 | ttype::BOOL_TRUE);
        push_uvarint(&mut out, 10_000_000);
        assert!(decode_column_index(&out).is_err());
    }

    // --- End-to-end verification through real files (pyarrow/parquet-cpp output) ---
    //
    // In this environment, DuckDB writes ColumnIndex/OffsetIndex but not Bloom
    // filters (see `scripts/gen-testdata.sh`). `tests/data/pagetest.parquet` is a
    // file generated with pyarrow (`write_page_index=True`, `bloom_filter_options`)
    // that has all three. The Bloom filter's correctness has been cross-checked
    // against an independently implemented SBBF match using Python's `xxhash`
    // package (the relevant true/false examples are transcribed verbatim here).

    fn pagetest_bytes() -> Vec<u8> {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/pagetest.parquet");
        std::fs::read(p).unwrap_or_else(|e| panic!("tests/data/pagetest.parquet: {e}"))
    }

    #[test]
    fn column_index_and_offset_index_partition_the_column_chunk_without_gaps() {
        let bytes = pagetest_bytes();
        let f = crate::parquet::file::open_bytes(&bytes).expect("open pagetest.parquet");
        let rg = &f.meta.row_groups[0];
        let id_col = &rg.columns[0]; // id: INT32, the first column
        let meta = id_col.meta.as_ref().unwrap();

        let (ci_start, ci_end) = id_col.column_index_range().expect("id has a ColumnIndex");
        let (oi_start, oi_end) = id_col.offset_index_range().expect("id has an OffsetIndex");
        let ci = decode_column_index(&bytes[ci_start as usize..ci_end as usize]).unwrap();
        let oi = decode_offset_index(&bytes[oi_start as usize..oi_end as usize]).unwrap();

        assert_eq!(ci.null_pages.len(), oi.page_locations.len());
        assert!(oi.page_locations.len() > 1, "test file must have multiple pages to be meaningful");

        // The pages are contiguous in the file and exactly cover the column chunk's
        // range (no gaps, no overlaps). The first row number is also strictly
        // increasing, and the first page starts at row 0.
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
        let meta = rg.columns[0].meta.as_ref().unwrap(); // the id column
        let (start, end) = meta.bloom_filter_probe_range().expect("id has a bloom filter");
        let probe = &bytes[start as usize..end.min(bytes.len() as u64) as usize];
        let (hdr, used) = decode_bloom_filter_header(probe).unwrap();
        let bitset = &probe[used..used + hdr.num_bytes as usize];
        let bf = crate::parquet::bloom::BloomFilter::new(bitset).unwrap();

        // A concrete example cross-checked against an independently implemented SBBF
        // match, computed against the same bit stream with Python's `xxhash` package.
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
