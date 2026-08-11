//! エラーは数値コードで表現する。
//!
//! `core::fmt` を一切引かないのが目的。メッセージ文字列は JS ホスト側
//! (`js/errors.js`) のテーブルで生成する。`std` フィーチャ有効時のみ、
//! ネイティブデバッグ用の文字列テーブルをリンクする。

/// エラーコード。JS 側のテーブルと 1:1 で対応するので、既存の値は変更しない。
// Debug は std ビルド (ネイティブのテスト) でだけ導出する。
// wasm ビルドで導出すると core::fmt がリンクされ、数十 KB を失う。
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "std", derive(Debug))]
#[repr(u16)]
pub enum Code {
    // 1xx: 入力バイト列の破損
    UnexpectedEof = 100,
    BadMagic = 101,
    BadThrift = 102,
    BadVarint = 103,
    NestingTooDeep = 104,
    BadPageHeader = 105,
    BadCompressedData = 106,
    ChecksumMismatch = 107,

    // 2xx: 未対応の Parquet 機能
    UnsupportedEncoding = 200,
    UnsupportedCodec = 201,
    UnsupportedType = 202,
    UnsupportedNested = 203,
    EncryptionUnsupported = 204,

    // 3xx: SQL 構文
    SyntaxError = 300,
    UnexpectedToken = 301,
    UnterminatedString = 302,
    NumberOverflow = 303,
    ExpressionTooDeep = 304,

    // 4xx: 束縛・意味解析
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
    /// INSERT の値の個数が列の個数と合わない（`ddl`/`dml`）。
    ColumnCountMismatch = 411,
    /// Parquet/CSV/JSONL 由来の読み取り専用テーブルに DDL/DML を試みた
    /// （`ddl`/`dml`）。
    ReadOnlyTable = 412,
    /// `ALTER TABLE ... ADD COLUMN`/`RENAME COLUMN` の結果、列名が同一表内で
    /// 重複する（`ddl`）。
    DuplicateColumn = 413,

    // 5xx: 実行時
    Oom = 500,
    LimitExceeded = 501,
    DivideByZero = 502,
    ValueOutOfRange = 503,
    IoFailed = 504,
    /// `WITH RECURSIVE` の不動点反復が上限回数に達した（`exec::recursive`
    /// の `MAX_RECURSIVE_ITERATIONS`）。終端しない再帰 CTE を有限時間で
    /// 確実に止めるための安全弁。
    RecursionLimitExceeded = 505,

    // 9xx: 内部矛盾（バグ）
    Internal = 900,
}

/// エラー本体。16 バイトに収まるようにしておく。
#[derive(Clone, Copy)]
pub struct Error {
    pub code: Code,
    /// 意味はコードごと。SQL エラーなら入力文字列上のバイト位置、
    /// Parquet エラーならファイル/バッファ上のオフセット。
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

/// `Err(Error::new(code))` の短縮形。
#[macro_export]
macro_rules! err {
    ($code:ident) => {
        return Err($crate::error::Error::new($crate::error::Code::$code))
    };
    ($code:ident, $pos:expr) => {
        return Err($crate::error::Error::at($crate::error::Code::$code, $pos))
    };
}

/// 条件を満たさなければエラーを返す。`assert!` と違いパニックしない。
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

// --- ネイティブ専用: デバッグ表示 -------------------------------------------
// wasm ビルドではこのブロックごとリンクされない。

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
            // LIST/MAP/STRUCT 自体は対応済み（parquet::nested の Dremel 組み立て、
            // または STRUCT のドット区切りフラット化）。ここに来るのは、壊れた/
            // 敵対的なスキーマ（子数と実要素数の不一致、物理型を持たないリーフ、
            // リーフ数の上限超過など）を検出したときだけ。
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

/// テスト用: `Result` からエラーコードを取り出す。`unwrap_err()` は
/// `T: Debug` を要求するため、Debug を導出していない型でも使えるようにする。
#[cfg(feature = "std")]
pub fn code_of<T>(r: Result<T>) -> Option<Code> {
    r.err().map(|e| e.code)
}
