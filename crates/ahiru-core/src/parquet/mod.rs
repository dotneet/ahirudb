//! Parquet read layer.
//!
//! This file holds the enums shared across the submodules. From here, the
//! flow is `thrift` (bytes -> Thrift) -> `meta` (Thrift -> metadata structs)
//! -> `reader` (metadata + pages -> vectors).

pub mod bloom;
pub mod codec;
pub mod encoding;
pub mod file;
pub mod meta;
pub mod nested;
pub mod reader;
pub mod schema;
pub mod thrift;

use crate::prelude::*;

/// Parquet physical type (`Type` in parquet.thrift).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PType {
    Boolean = 0,
    Int32 = 1,
    Int64 = 2,
    Int96 = 3,
    Float = 4,
    Double = 5,
    ByteArray = 6,
    FixedLenByteArray = 7,
}

impl PType {
    pub fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            0 => PType::Boolean,
            1 => PType::Int32,
            2 => PType::Int64,
            3 => PType::Int96,
            4 => PType::Float,
            5 => PType::Double,
            6 => PType::ByteArray,
            7 => PType::FixedLenByteArray,
            _ => err!(UnsupportedType),
        })
    }
}

/// Repetition kind (`FieldRepetitionType` in parquet.thrift).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Repetition {
    Required = 0,
    Optional = 1,
    Repeated = 2,
}

impl Repetition {
    pub fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            0 => Repetition::Required,
            1 => Repetition::Optional,
            2 => Repetition::Repeated,
            _ => err!(BadThrift),
        })
    }
}

/// Page encoding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Encoding {
    Plain = 0,
    PlainDictionary = 2,
    Rle = 3,
    BitPacked = 4,
    DeltaBinaryPacked = 5,
    DeltaLengthByteArray = 6,
    DeltaByteArray = 7,
    RleDictionary = 8,
    ByteStreamSplit = 9,
}

impl Encoding {
    pub fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            0 => Encoding::Plain,
            2 => Encoding::PlainDictionary,
            3 => Encoding::Rle,
            4 => Encoding::BitPacked,
            5 => Encoding::DeltaBinaryPacked,
            6 => Encoding::DeltaLengthByteArray,
            7 => Encoding::DeltaByteArray,
            8 => Encoding::RleDictionary,
            9 => Encoding::ByteStreamSplit,
            _ => err!(UnsupportedEncoding),
        })
    }

    /// Whether this encoding references a dictionary page.
    pub fn is_dictionary(self) -> bool {
        matches!(self, Encoding::PlainDictionary | Encoding::RleDictionary)
    }
}

/// Compression codec.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compression {
    Uncompressed = 0,
    Snappy = 1,
    Gzip = 2,
    Lzo = 3,
    Brotli = 4,
    Lz4 = 5,
    Zstd = 6,
    Lz4Raw = 7,
}

impl Compression {
    pub fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            0 => Compression::Uncompressed,
            1 => Compression::Snappy,
            2 => Compression::Gzip,
            3 => Compression::Lzo,
            4 => Compression::Brotli,
            5 => Compression::Lz4,
            6 => Compression::Zstd,
            7 => Compression::Lz4Raw,
            _ => err!(UnsupportedCodec),
        })
    }

    /// Whether this codec is built into the wasm core. Codecs that aren't
    /// built in delegate decompression to the host (JS's DecompressionStream
    /// or a separate wasm module). See DESIGN.md §6.
    ///
    /// ZSTD only counts as built in when the `zstd` feature (enabled by
    /// default) is present. Otherwise it falls back to host delegation (`NeedCodec`) as before.
    pub fn is_builtin(self) -> bool {
        matches!(self, Compression::Uncompressed | Compression::Snappy | Compression::Lz4Raw)
            || (self == Compression::Zstd && cfg!(feature = "zstd"))
    }
}

/// Page type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PageType {
    DataPage = 0,
    IndexPage = 1,
    DictionaryPage = 2,
    DataPageV2 = 3,
}

impl PageType {
    pub fn from_i32(v: i32) -> Result<Self> {
        Ok(match v {
            0 => PageType::DataPage,
            1 => PageType::IndexPage,
            2 => PageType::DictionaryPage,
            3 => PageType::DataPageV2,
            _ => err!(BadPageHeader),
        })
    }
}

/// The legacy `ConvertedType`. Backward compatibility for files without `LogicalType`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConvertedType {
    Utf8 = 0,
    Map = 1,
    MapKeyValue = 2,
    List = 3,
    Enum = 4,
    Decimal = 5,
    Date = 6,
    TimeMillis = 7,
    TimeMicros = 8,
    TimestampMillis = 9,
    TimestampMicros = 10,
    Uint8 = 11,
    Uint16 = 12,
    Uint32 = 13,
    Uint64 = 14,
    Int8 = 15,
    Int16 = 16,
    Int32 = 17,
    Int64 = 18,
    Json = 19,
    Bson = 20,
    Interval = 21,
}

impl ConvertedType {
    pub fn from_i32(v: i32) -> Option<Self> {
        use ConvertedType::*;
        Some(match v {
            0 => Utf8,
            1 => Map,
            2 => MapKeyValue,
            3 => List,
            4 => Enum,
            5 => Decimal,
            6 => Date,
            7 => TimeMillis,
            8 => TimeMicros,
            9 => TimestampMillis,
            10 => TimestampMicros,
            11 => Uint8,
            12 => Uint16,
            13 => Uint32,
            14 => Uint64,
            15 => Int8,
            16 => Int16,
            17 => Int32,
            18 => Int64,
            19 => Json,
            20 => Bson,
            21 => Interval,
            _ => return None,
        })
    }
}

/// Time resolution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeUnit {
    Millis,
    Micros,
    Nanos,
}

impl TimeUnit {
    /// Normalize a value in this unit to microseconds.
    /// Nanoseconds are truncated (per DESIGN.md §8, the internal representation is always microseconds).
    #[inline]
    pub fn to_micros(self, v: i64) -> i64 {
        match self {
            TimeUnit::Millis => v.saturating_mul(1000),
            TimeUnit::Micros => v,
            TimeUnit::Nanos => v / 1000,
        }
    }
}

/// `SchemaElement.logicalType` (a Thrift union).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LogicalType {
    String,
    Map,
    List,
    Enum,
    Decimal { scale: i32, precision: i32 },
    Date,
    Time { utc: bool, unit: TimeUnit },
    Timestamp { utc: bool, unit: TimeUnit },
    Integer { bit_width: u8, signed: bool },
    Unknown,
    Json,
    Bson,
    Uuid,
    Float16,
}

/// Magic bytes at the end of a Parquet file.
pub const MAGIC: &[u8; 4] = b"PAR1";
/// Magic for an encrypted file. Detected and explicitly rejected.
pub const MAGIC_ENCRYPTED: &[u8; 4] = b"PARE";
