// エラーコード表。
//
// wasm 側は数値コードしか返さない（`core::fmt` を一切リンクしないため）。
// 文字列を持つのはここだけで、その分 wasm の予算をエンジン本体に回せる。
//
// ★ crates/ahiru-core/src/error.rs の `Code` / `message()` と 1:1 で対応する。
//   どちらか片方だけを書き換えないこと。値は既存クエリの互換性のため変更しない。

/** エラーコード。error.rs の `#[repr(u16)] enum Code` と同じ数値。 */
export const Code = Object.freeze({
  // 1xx: 入力バイト列の破損
  UNEXPECTED_EOF: 100,
  BAD_MAGIC: 101,
  BAD_THRIFT: 102,
  BAD_VARINT: 103,
  NESTING_TOO_DEEP: 104,
  BAD_PAGE_HEADER: 105,
  BAD_COMPRESSED_DATA: 106,
  CHECKSUM_MISMATCH: 107,

  // 2xx: 未対応の Parquet 機能
  UNSUPPORTED_ENCODING: 200,
  UNSUPPORTED_CODEC: 201,
  UNSUPPORTED_TYPE: 202,
  UNSUPPORTED_NESTED: 203,
  ENCRYPTION_UNSUPPORTED: 204,

  // 3xx: SQL 構文
  SYNTAX_ERROR: 300,
  UNEXPECTED_TOKEN: 301,
  UNTERMINATED_STRING: 302,
  NUMBER_OVERFLOW: 303,
  EXPRESSION_TOO_DEEP: 304,

  // 4xx: 束縛・意味解析
  TABLE_NOT_FOUND: 400,
  COLUMN_NOT_FOUND: 401,
  AMBIGUOUS_COLUMN: 402,
  FUNCTION_NOT_FOUND: 403,
  TYPE_MISMATCH: 404,
  INVALID_CAST: 405,
  WRONG_ARG_COUNT: 406,
  NOT_AGGREGATE: 407,
  NOT_GROUPED: 408,
  UNSUPPORTED_FEATURE: 409,
  DUPLICATE_TABLE: 410,
  COLUMN_COUNT_MISMATCH: 411,
  READ_ONLY_TABLE: 412,
  DUPLICATE_COLUMN: 413,

  // 5xx: 実行時
  OOM: 500,
  LIMIT_EXCEEDED: 501,
  DIVIDE_BY_ZERO: 502,
  VALUE_OUT_OF_RANGE: 503,
  IO_FAILED: 504,
  RECURSION_LIMIT_EXCEEDED: 505,

  // 9xx: 内部矛盾（バグ）
  INTERNAL: 900,
});

/** コード → メッセージ。error.rs の `Error::message()` と同じ文字列。 */
const MESSAGES = Object.freeze({
  100: 'unexpected end of input',
  101: 'not a parquet file (bad magic)',
  102: 'malformed thrift data',
  103: 'malformed varint',
  104: 'nesting too deep',
  105: 'malformed page header',
  106: 'malformed compressed data',
  107: 'page checksum mismatch',
  200: 'unsupported parquet encoding',
  201: 'unsupported compression codec',
  202: 'unsupported parquet type',
  // LIST/MAP/STRUCT 自体は対応済み（Dremel 組み立て、または STRUCT のドット
  // 区切りフラット化）。壊れた/敵対的なスキーマの検出時だけこのコードになる。
  203: 'malformed or oversized nested parquet schema',
  204: 'encrypted parquet files are not supported',
  300: 'syntax error',
  301: 'unexpected token',
  302: 'unterminated string literal',
  303: 'numeric literal out of range',
  304: 'expression nesting too deep',
  400: 'table not found',
  401: 'column not found',
  402: 'ambiguous column reference',
  403: 'function not found',
  404: 'type mismatch',
  405: 'invalid cast',
  406: 'wrong number of arguments',
  407: 'aggregate function required',
  408: 'column must appear in GROUP BY',
  409: 'unsupported SQL feature',
  410: 'table already registered',
  411: 'number of values does not match number of columns',
  412: 'table is read-only (not created by CREATE TABLE)',
  413: 'column already exists',
  500: 'out of memory',
  501: 'resource limit exceeded',
  502: 'division by zero',
  503: 'value out of range',
  504: 'io failed',
  505: 'recursive CTE exceeded the maximum number of iterations',
  900: 'internal error',
});

/** コードに対応する英文メッセージ。未知のコードでも必ず文字列を返す。 */
export function errorMessage(code) {
  return MESSAGES[code] ?? `unknown error (code ${code})`;
}

/**
 * エンジンから返るエラー。
 *
 * `code` は error.rs の数値、`message` は上の表から組み立てたもの。
 * 追加の文脈（実行中の SQL、ホスト側の補足説明）は任意。
 */
export class AhiruError extends Error {
  constructor(code, { sql, detail, cause } = {}) {
    const base = errorMessage(code);
    super(detail ? `[E${code}] ${base}: ${detail}` : `[E${code}] ${base}`, { cause });
    this.name = 'AhiruError';
    this.code = code;
    /** コード本来のメッセージ（detail を含まない）。 */
    this.reason = base;
    if (sql !== undefined) this.sql = sql;
    if (detail !== undefined) this.detail = detail;
  }
}
