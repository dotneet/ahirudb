//! Errors are expressed as numeric codes.
//!
//! The goal is to never pull in `core::fmt`. Message strings are produced by the
//! table on the JS host side (`js/errors.js`). A string table for native debugging
//! is linked only when the `std` feature is on.

/// Error codes. These map 1:1 to the table on the JS side, so existing values never change.
// Debug is derived only in std builds (native tests).
// Deriving it in wasm builds links core::fmt and costs tens of KB.
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
#[repr(u16)]
pub enum Code {
    // 1xx: corrupt input bytes
    UnexpectedEof = 100,
    BadMagic = 101,
    BadThrift = 102,
    BadVarint = 103,
    NestingTooDeep = 104,
    BadPageHeader = 105,
    BadCompressedData = 106,
    ChecksumMismatch = 107,

    // 2xx: unsupported Parquet features
    UnsupportedEncoding = 200,
    UnsupportedCodec = 201,
    UnsupportedType = 202,
    UnsupportedNested = 203,
    EncryptionUnsupported = 204,

    // 3xx: SQL syntax
    SyntaxError = 300,
    UnexpectedToken = 301,
    UnterminatedString = 302,
    NumberOverflow = 303,
    ExpressionTooDeep = 304,

    // 4xx: binding and semantic analysis
    TableNotFound = 400,
    ColumnNotFound = 401,
    AmbiguousColumn = 402,
    FunctionNotFound = 403,
    TypeMismatch = 404,
    InvalidCast = 405,
    WrongArgCount = 406,
    NotAggregate = 407,
    NotGrouped = 408,
    UnsupportedFeature = 409,
    DuplicateTable = 410,
    /// The number of values in an INSERT does not match the number of columns (`ddl`/`dml`).
    ColumnCountMismatch = 411,
    /// DDL/DML was attempted against a read-only table backed by Parquet/CSV/JSONL
    /// (`ddl`/`dml`).
    ReadOnlyTable = 412,
    /// `ALTER TABLE ... ADD COLUMN`/`RENAME COLUMN` would leave duplicate column
    /// names within one table (`ddl`).
    DuplicateColumn = 413,

    // 5xx: runtime
    Oom = 500,
    LimitExceeded = 501,
    DivideByZero = 502,
    ValueOutOfRange = 503,
    IoFailed = 504,
    /// The fixed-point iteration of `WITH RECURSIVE` hit the iteration cap
    /// (`MAX_RECURSIVE_ITERATIONS` in `exec::recursive`). A safety valve that
    /// reliably stops a non-terminating recursive CTE in finite time.
    RecursionLimitExceeded = 505,

    // 9xx: internal inconsistency (bug)
    Internal = 900,
}

/// The error itself. Kept within 16 bytes.
#[derive(Clone, Copy)]
pub struct Error {
    pub code: Code,
    /// The meaning depends on the code: a byte position in the input string for SQL
    /// errors, or an offset into the file/buffer for Parquet errors.
    pub pos: u32,
}

impl Error {
    #[inline]
    pub fn new(code: Code) -> Self {
        Error { code, pos: u32::MAX }
    }

    #[inline]
    pub fn at(code: Code, pos: usize) -> Self {
        Error { code, pos: pos as u32 }
    }

    #[inline]
    pub fn code_u16(&self) -> u16 {
        self.code as u16
    }
}

pub type Result<T> = core::result::Result<T, Error>;

/// Shorthand for `Err(Error::new(code))`.
#[macro_export]
macro_rules! err {
    ($code:ident) => {
        return Err($crate::error::Error::new($crate::error::Code::$code))
    };
    ($code:ident, $pos:expr) => {
        return Err($crate::error::Error::at($crate::error::Code::$code, $pos))
    };
}

/// Returns an error unless the condition holds. Unlike `assert!`, this never panics.
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $code:ident) => {
        if !($cond) {
            return Err($crate::error::Error::new($crate::error::Code::$code));
        }
    };
    ($cond:expr, $code:ident, $pos:expr) => {
        if !($cond) {
            return Err($crate::error::Error::at($crate::error::Code::$code, $pos));
        }
    };
}

// --- Native only: debug rendering -------------------------------------------
// This whole block is not linked in wasm builds.

#[cfg(feature = "std")]
impl Error {
    pub fn message(&self) -> &'static str {
        use Code::*;
        match self.code {
            UnexpectedEof => "unexpected end of input",
            BadMagic => "not a parquet file (bad magic)",
            BadThrift => "malformed thrift data",
            BadVarint => "malformed varint",
            NestingTooDeep => "nesting too deep",
            BadPageHeader => "malformed page header",
            BadCompressedData => "malformed compressed data",
            ChecksumMismatch => "page checksum mismatch",
            UnsupportedEncoding => "unsupported parquet encoding",
            UnsupportedCodec => "unsupported compression codec",
            UnsupportedType => "unsupported parquet type",
            // LIST/MAP/STRUCT themselves are supported (Dremel assembly in
            // parquet::nested, or dot-separated flattening for STRUCT). This is only
            // reached when a broken or hostile schema is detected (child count
            // disagreeing with the actual elements, a leaf with no physical type,
            // exceeding the leaf-count cap, and so on).
            UnsupportedNested => "malformed or oversized nested parquet schema",
            EncryptionUnsupported => "encrypted parquet files are not supported",
            SyntaxError => "syntax error",
            UnexpectedToken => "unexpected token",
            UnterminatedString => "unterminated string literal",
            NumberOverflow => "numeric literal out of range",
            ExpressionTooDeep => "expression nesting too deep",
            TableNotFound => "table not found",
            ColumnNotFound => "column not found",
            AmbiguousColumn => "ambiguous column reference",
            FunctionNotFound => "function not found",
            TypeMismatch => "type mismatch",
            InvalidCast => "invalid cast",
            WrongArgCount => "wrong number of arguments",
            NotAggregate => "aggregate function required",
            NotGrouped => "column must appear in GROUP BY",
            UnsupportedFeature => "unsupported SQL feature",
            DuplicateTable => "table already registered",
            ColumnCountMismatch => "number of values does not match number of columns",
            ReadOnlyTable => "table is read-only (not created by CREATE TABLE)",
            DuplicateColumn => "column already exists",
            Oom => "out of memory",
            LimitExceeded => "resource limit exceeded",
            DivideByZero => "division by zero",
            ValueOutOfRange => "value out of range",
            IoFailed => "io failed",
            RecursionLimitExceeded => "recursive CTE exceeded the maximum number of iterations",
            Internal => "internal error",
        }
    }
}

#[cfg(feature = "std")]
impl core::fmt::Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[E{}] {}", self.code_u16(), self.message())?;
        if self.pos != u32::MAX {
            write!(f, " (at {})", self.pos)?;
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// For tests: extracts the error code from a `Result`. `unwrap_err()` requires
/// `T: Debug`, so this works for types that do not derive Debug.
#[cfg(feature = "std")]
pub fn code_of<T>(r: Result<T>) -> Option<Code> {
    r.err().map(|e| e.code)
}
