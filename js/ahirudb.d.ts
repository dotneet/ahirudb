// Type definitions for ahirudb's public API.
// The implementation is ahirudb.js (dependency-free ESM).

/** Logical type names. 1:1 with `ty_code` (abi.rs). */
export type AhiruTypeName =
  | 'NULL' | 'BOOLEAN' | 'TINYINT' | 'SMALLINT' | 'INTEGER' | 'BIGINT' | 'HUGEINT'
  | 'UTINYINT' | 'USMALLINT' | 'UINTEGER' | 'UBIGINT' | 'FLOAT' | 'DOUBLE'
  | 'DECIMAL' | 'VARCHAR' | 'BLOB' | 'DATE' | 'TIME' | 'TIMESTAMP' | 'INTERVAL'
  | 'JSON' | 'UUID' | 'TIMESTAMPTZ';

/** Physical types. The six the execution kernels handle (vector/types.rs). 0=Bool 1=I32 2=I64 3=F64 4=I128 5=Bytes */
export type PhysType = 0 | 1 | 2 | 3 | 4 | 5;

/**
 * A row value.
 * - BOOLEAN -> boolean
 * - INTEGER family (physical I32) -> number
 * - BIGINT / TIME / TIMESTAMP / TIMESTAMPTZ (physical I64) -> bigint (the raw value, e.g. microseconds)
 * - HUGEINT (physical I128) -> bigint
 * - FLOAT / DOUBLE -> number
 * - DECIMAL -> string (precision/scale already applied; a number would lose digits)
 * - VARCHAR -> string, BLOB -> Uint8Array
 * - UUID -> string (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` form)
 * - NULL -> null (the validity bitmap is honored)
 */
export type AhiruValue = boolean | number | bigint | string | Uint8Array | null;

export type Row = Record<string, AhiruValue>;

/** A parameter that can be passed to a query. Pass TIMESTAMP as BigInt microseconds. */
export type AhiruParam = null | boolean | number | bigint | string | Uint8Array | ArrayBuffer;

export interface Field {
  name: string;
  type: AhiruTypeName;
}

/** One column as returned by `decodeSchema`. Only DECIMAL has non-zero precision/scale. */
export interface SchemaField extends Field {
  typeCode: number;
  physType: PhysType;
  precision: number;
  scale: number;
}

/** Raw column data. NULL rows hold a dummy value, so read it together with `valid`. */
export type ColumnValues =
  | Uint8Array // BOOLEAN (0/1)
  | Int32Array
  | BigInt64Array
  | Float64Array
  | bigint[] // HUGEINT
  | string[] // DECIMAL
  | (string | Uint8Array)[]; // VARCHAR / BLOB

export interface Column {
  name: string;
  type: AhiruTypeName;
  typeCode: number;
  physType: PhysType;
  values: ColumnValues;
  /** 1 = valid, 0 = NULL. null when there are no NULLs at all. */
  valid: Uint8Array | null;
}

/** One batch as returned by `stream()` (columnar, 2048 rows by default). */
export declare class Batch {
  readonly numRows: number;
  readonly columns: Column[];
  readonly schema: Field[];
  /** Raw column data. TypedArrays are copies and are unaffected by the next batch. */
  column(key: string | number): ColumnValues;
  isNull(key: string | number, row: number): boolean;
  get(key: string | number, row: number): AhiruValue;
  toRows(): Row[];
}

/** The minimal byte-range cache interface. */
export interface ByteRangeCache {
  get(key: string): Uint8Array | undefined;
  set(key: string, bytes: Uint8Array): void;
  clear(): void;
}

/** An LRU cache keyed by `(source, offset, len)`. */
export declare class MemoryCache implements ByteRangeCache {
  constructor(maxBytes?: number);
  maxBytes: number;
  readonly size: number;
  get(key: string): Uint8Array | undefined;
  set(key: string, bytes: Uint8Array): void;
  clear(): void;
}

/** A custom byte supplier (OPFS, a fake server for tests, and so on). */
export interface ByteSource {
  /** Cache key prefix. Numbered automatically when omitted. */
  key?: string;
  size: number | (() => number | Promise<number>);
  read(offset: number, length: number): Uint8Array | Promise<Uint8Array>;
}

export type TableSource = string | URL | Uint8Array | ArrayBuffer | Blob | ByteSource;
/** @deprecated Use `TableSource`. Formats other than Parquet can be registered too. */
export type ParquetSource = TableSource;

/** Supported formats. Determined by the extension of the registered name. */
export type FormatName = 'parquet' | 'csv' | 'tsv' | 'jsonl';

export interface InitOptions {
  /** URL of the wasm. On Node it is read as a file path. */
  wasmUrl?: string | URL;
  /** When you already have the bytes. */
  wasmBinary?: Uint8Array | ArrayBuffer;
  /** When it is already compiled. */
  wasmModule?: WebAssembly.Module;
  /** The ZSTD side module (crates/ahiru-zstd). Not loaded until ZSTD is first required. */
  zstdUrl?: string | URL;
  zstdBinary?: Uint8Array | ArrayBuffer;
  zstdModule?: WebAssembly.Module;
  /** Upper bound on the wasm heap, in bytes. Exceeding it throws E501. 0 means unlimited. */
  memoryLimit?: number;
  /** "memory" (default) | "cache-api" (degrades to memory on Node) | "none" | your own implementation */
  cache?: 'memory' | 'cache-api' | 'none' | ByteRangeCache;
  /** Byte limit for the memory cache. 64 MiB by default. */
  cacheSize?: number;
  /** The fetch used by URL sources. Defaults to globalThis.fetch. */
  fetch?: typeof globalThis.fetch;
}

export declare class AhiruDB {
  static init(options?: InitOptions): Promise<AhiruDB>;

  /**
   * Registers a table. No I/O happens (it is deferred until the first query).
   *
   * If `format` is given it is used (`ahiru_register_as`) and the name needs no extension.
   * Otherwise the engine infers it from the extension of the registered name.
   */
  register(name: string, source: TableSource, options?: { format?: FormatName }): this;

  /** Alias for `register`. */
  registerParquet(name: string, source: TableSource, options?: { format?: FormatName }): this;

  /** Materializes the entire result and returns it. */
  query(sql: string, params?: readonly AhiruParam[]): Promise<Row[]>;

  /** Yields columnar batches in order. For large results. */
  stream(sql: string, params?: readonly AhiruParam[]): AsyncGenerator<Batch, void, void>;

  close(): void;

  /** wasm heap usage, in bytes. */
  readonly heapUsed: number;
}

/** An engine error. `code` is the number from crates/ahiru-core/src/error.rs. */
export declare class AhiruError extends Error {
  readonly code: number;
  /** The code's own message (without the extra detail). */
  readonly reason: string;
  readonly sql?: string;
  readonly detail?: string;
  constructor(code: number, options?: { sql?: string; detail?: string; cause?: unknown });
}

export declare const Code: Readonly<Record<string, number>>;
export declare function errorMessage(code: number): string;

/** Converts a TIMESTAMP (microseconds since the epoch) to a Date. */
export declare function timestampToDate(micros: bigint | number): Date;
/** Converts a DATE (days since the epoch) to a Date. */
export declare function dateToDate(days: number): Date;
/** Converts a TIMESTAMPTZ (UTC microseconds since the epoch) to a Date. Alias of `timestampToDate`. */
export declare const timestamptzToDate: typeof timestampToDate;

/** Coalesces nearby ranges (default threshold is 1 MiB). */
export declare function coalesceRanges(
  ranges: readonly { offset: number | bigint; len: number | bigint }[],
  gap?: number,
  totalLen?: number,
): { offset: number; len: number }[];

/** Infers the format from the extension of a registered name (a mirror of `FormatKind::detect`). */
export declare function detectFormat(name: string): FormatName;

/** Decoder / encoder for the wire format in abi.rs. */
export declare function decodeIoRequests(
  bytes: Uint8Array,
): { table: number; offset: number; len: number }[];
export declare function decodeCodecRequests(
  bytes: Uint8Array,
): { table: number; codec: number; offset: number; len: number; outLen: number }[];
export declare function decodeSchema(bytes: Uint8Array): SchemaField[];
export declare function decodeBatch(
  bytes: Uint8Array,
  schema: readonly SchemaField[],
  copy?: boolean,
): Batch;
/** Serializes a parameter list into `[count][tag+payload...]`. */
export declare function encodeParams(params?: readonly AhiruParam[] | null): Uint8Array;

export default AhiruDB;
