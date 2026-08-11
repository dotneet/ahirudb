// ahirudb の公開 API の型定義。
// 実装は ahirudb.js（依存ゼロの ESM）。

/** 論理型の名前。`ty_code`（abi.rs）と 1:1。 */
export type AhiruTypeName =
  | 'NULL' | 'BOOLEAN' | 'TINYINT' | 'SMALLINT' | 'INTEGER' | 'BIGINT' | 'HUGEINT'
  | 'UTINYINT' | 'USMALLINT' | 'UINTEGER' | 'UBIGINT' | 'FLOAT' | 'DOUBLE'
  | 'DECIMAL' | 'VARCHAR' | 'BLOB' | 'DATE' | 'TIME' | 'TIMESTAMP' | 'INTERVAL'
  | 'JSON' | 'UUID' | 'TIMESTAMPTZ';

/** 物理型。実行カーネルが扱う 6 種（vector/types.rs）。0=Bool 1=I32 2=I64 3=F64 4=I128 5=Bytes */
export type PhysType = 0 | 1 | 2 | 3 | 4 | 5;

/**
 * 行の値。
 * - BOOLEAN → boolean
 * - INTEGER 系（物理 I32）→ number
 * - BIGINT / TIME / TIMESTAMP / TIMESTAMPTZ（物理 I64）→ bigint（マイクロ秒など生の値）
 * - HUGEINT（物理 I128）→ bigint
 * - FLOAT / DOUBLE → number
 * - DECIMAL → string（precision/scale を適用済み。number にすると桁が落ちる）
 * - VARCHAR → string、BLOB → Uint8Array
 * - UUID → string（`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` 形式）
 * - NULL は null（validity ビットマップを尊重する）
 */
export type AhiruValue = boolean | number | bigint | string | Uint8Array | null;

export type Row = Record<string, AhiruValue>;

/** クエリに渡せるパラメータ。TIMESTAMP は BigInt のマイクロ秒で渡す。 */
export type AhiruParam = null | boolean | number | bigint | string | Uint8Array | ArrayBuffer;

export interface Field {
  name: string;
  type: AhiruTypeName;
}

/** `decodeSchema` が返す 1 列。DECIMAL のみ precision/scale が 0 以外。 */
export interface SchemaField extends Field {
  typeCode: number;
  physType: PhysType;
  precision: number;
  scale: number;
}

/** 列の生データ。NULL 行にはダミー値が入っているので `valid` と併せて見る。 */
export type ColumnValues =
  | Uint8Array // BOOLEAN（0/1）
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
  /** 1 = 有効、0 = NULL。NULL が 1 つも無ければ null。 */
  valid: Uint8Array | null;
}

/** `stream()` が返す 1 バッチ（列指向、既定 2048 行）。 */
export declare class Batch {
  readonly numRows: number;
  readonly columns: Column[];
  readonly schema: Field[];
  /** 列の生データ。TypedArray はコピー済みで、次のバッチに影響されない。 */
  column(key: string | number): ColumnValues;
  isNull(key: string | number, row: number): boolean;
  get(key: string | number, row: number): AhiruValue;
  toRows(): Row[];
}

/** バイト範囲キャッシュの最小インタフェース。 */
export interface ByteRangeCache {
  get(key: string): Uint8Array | undefined;
  set(key: string, bytes: Uint8Array): void;
  clear(): void;
}

/** `(source, offset, len)` をキーにした LRU キャッシュ。 */
export declare class MemoryCache implements ByteRangeCache {
  constructor(maxBytes?: number);
  maxBytes: number;
  readonly size: number;
  get(key: string): Uint8Array | undefined;
  set(key: string, bytes: Uint8Array): void;
  clear(): void;
}

/** 独自のバイト供給元（OPFS、テスト用の偽サーバなど）。 */
export interface ByteSource {
  /** キャッシュキーの前置。省略すると自動採番される。 */
  key?: string;
  size: number | (() => number | Promise<number>);
  read(offset: number, length: number): Uint8Array | Promise<Uint8Array>;
}

export type TableSource = string | URL | Uint8Array | ArrayBuffer | Blob | ByteSource;
/** @deprecated `TableSource` を使う。Parquet 以外も登録できる。 */
export type ParquetSource = TableSource;

/** 対応フォーマット。登録名の拡張子で決まる。 */
export type FormatName = 'parquet' | 'csv' | 'tsv' | 'jsonl';

export interface InitOptions {
  /** wasm の URL。Node ではファイルパスとして読む。 */
  wasmUrl?: string | URL;
  /** 既にバイト列を持っている場合。 */
  wasmBinary?: Uint8Array | ArrayBuffer;
  /** 既にコンパイル済みの場合。 */
  wasmModule?: WebAssembly.Module;
  /** ZSTD サイドモジュール（crates/ahiru-zstd）。初回の ZSTD 要求まで読まない。 */
  zstdUrl?: string | URL;
  zstdBinary?: Uint8Array | ArrayBuffer;
  zstdModule?: WebAssembly.Module;
  /** wasm ヒープの上限（バイト）。超えたら E501 を投げる。0 で無制限。 */
  memoryLimit?: number;
  /** "memory"（既定） | "cache-api"（Node ではメモリに縮退） | "none" | 自前の実装 */
  cache?: 'memory' | 'cache-api' | 'none' | ByteRangeCache;
  /** メモリキャッシュの上限バイト数。既定 64 MiB。 */
  cacheSize?: number;
  /** URL 供給元が使う fetch。省略時は globalThis.fetch。 */
  fetch?: typeof globalThis.fetch;
}

export declare class AhiruDB {
  static init(options?: InitOptions): Promise<AhiruDB>;

  /**
   * テーブルを登録する。I/O は発生しない（初回クエリまで遅延）。
   *
   * `format` を渡せばそれが使われ（`ahiru_register_as`）、名前に拡張子は要らない。
   * 省略した場合はエンジンが登録名の拡張子から推定する。
   */
  register(name: string, source: TableSource, options?: { format?: FormatName }): this;

  /** `register` の別名。 */
  registerParquet(name: string, source: TableSource, options?: { format?: FormatName }): this;

  /** 結果を全部materializeして返す。 */
  query(sql: string, params?: readonly AhiruParam[]): Promise<Row[]>;

  /** 列指向のバッチを順に返す。大きな結果向け。 */
  stream(sql: string, params?: readonly AhiruParam[]): AsyncGenerator<Batch, void, void>;

  close(): void;

  /** wasm ヒープの使用量（バイト）。 */
  readonly heapUsed: number;
}

/** エンジンのエラー。`code` は crates/ahiru-core/src/error.rs の数値。 */
export declare class AhiruError extends Error {
  readonly code: number;
  /** コード本来のメッセージ（補足を含まない）。 */
  readonly reason: string;
  readonly sql?: string;
  readonly detail?: string;
  constructor(code: number, options?: { sql?: string; detail?: string; cause?: unknown });
}

export declare const Code: Readonly<Record<string, number>>;
export declare function errorMessage(code: number): string;

/** TIMESTAMP（エポックからのマイクロ秒）を Date にする。 */
export declare function timestampToDate(micros: bigint | number): Date;
/** DATE（エポックからの日数）を Date にする。 */
export declare function dateToDate(days: number): Date;
/** TIMESTAMPTZ（エポックからの UTC マイクロ秒）を Date にする。`timestampToDate` の別名。 */
export declare const timestamptzToDate: typeof timestampToDate;

/** 近接するレンジを結合する（既定のしきい値は 1 MiB）。 */
export declare function coalesceRanges(
  ranges: readonly { offset: number | bigint; len: number | bigint }[],
  gap?: number,
  totalLen?: number,
): { offset: number; len: number }[];

/** 登録名の拡張子からフォーマットを推定する（`FormatKind::detect` の写し）。 */
export declare function detectFormat(name: string): FormatName;

/** abi.rs のワイヤ形式のデコーダ / エンコーダ。 */
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
/** パラメータ列を `[count][tag+payload...]` に直列化する。 */
export declare function encodeParams(params?: readonly AhiruParam[] | null): Uint8Array;

export default AhiruDB;
