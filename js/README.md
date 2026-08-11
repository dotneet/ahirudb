# ahirudb — JS ホスト層

`ahiru-core.wasm` を駆動する ES モジュール。**依存ゼロ・ビルド不要**で、ブラウザと
Node 18+ の両方でそのまま動く。

```
js/
  ahirudb.js     本体（IO ループ / 結果デコード / キャッシュ）
  errors.js      エラーコード表（crates/ahiru-core/src/error.rs と 1:1）
  ahirudb.d.ts   型定義
  test/          node --test 用のテスト
```

## 使い方

```js
import { AhiruDB, timestampToDate } from './js/ahirudb.js';

const db = await AhiruDB.init({
  wasmUrl: '/ahiru-core.wasm',
  memoryLimit: 512 * 1024 * 1024, // wasm ヒープの上限。超えたら E501
  cache: 'memory',                // "memory" | "cache-api" | "none" | 自前の実装
  zstdUrl: '/ahiru-zstd.wasm',    // ZSTD を含むファイルを読むときだけ必要
});

// 登録では I/O しない。総バイト長の取得もフッタ読みも初回クエリまで遅延する。
db.register('trips', 'https://example.com/trips.parquet'); // HTTP Range
db.register('local', bytes);                               // Uint8Array / ArrayBuffer
db.register('picked', fileFromInputElement);               // Blob / File
db.register('logs.csv', csvBytes);                         // CSV / TSV / JSONL

// 1) 全件取得
const rows = await db.query('SELECT id, name FROM trips LIMIT 5');
// -> [{ id: 0, name: 'name_0' }, ...]

// 2) パラメータバインド（SQL に値を埋め込まない）
await db.query('SELECT * FROM trips WHERE vendor = ? AND fare > ?', ['VTS', 100]);

// 3) ストリーミング（列指向のバッチ、既定 2048 行）
for await (const batch of db.stream('SELECT id, score FROM trips')) {
  batch.numRows;            // 行数
  batch.column('score');    // Float64Array
  batch.isNull('score', 3); // validity ビットマップを見る
  batch.toRows();           // プレーンなオブジェクトの配列
}

db.close();
```

`registerParquet` は `register` の別名（Parquet 以外も登録できる）。

`FROM parquet('https://…/a.parquet')` と書いた場合、そのパスは自動的に同名の
テーブルとして登録される（`plan/bind.rs` の `resolve_from` がそういう約束）。
これは Parquet 用の入口で、CSV / JSONL は `register(name, src, { format })` で
登録してから素の識別子で参照する。

### フォーマット

`format` を渡せばそれが使われる（`ahiru_register_as`）。渡さなければ
**登録名の拡張子**からエンジンが推定する（`format::FormatKind::detect`）。

```js
db.register('logs', bytes, { format: 'csv' }); // 名前は素の識別子でよい
await db.query('SELECT * FROM logs');

db.register('logs.csv', bytes);                // 拡張子に任せる
await db.query('SELECT * FROM "logs.csv"');    // SQL 側は引用符付きで参照する
```

`format` は `parquet` / `csv` / `tsv` / `jsonl`。拡張子と食い違っていても
明示指定が優先される（名前と読み方を切り離せることがこのオプションの目的なので、
そこを検査で塞がない）。綴りを間違えた場合だけ E409 で落とす — Auto に
落とすと Parquet として読まれ、`BadMagic` になって原因が分からなくなるため。

拡張子推定は `.csv` / `.tsv` / `.tab` / `.jsonl` / `.ndjson`、それ以外は Parquet。
CSV と JSONL は wasm 側のフィーチャ（`--features csv,jsonl`）で切れるので、
既定の配布ビルドには入っていない。入っていないビルドに登録すると E409 になる。

### パラメータ

`null` / `boolean` / `number` / `bigint` / `string` / `Uint8Array` を渡せる。
安全な整数と `bigint` は I64、それ以外の `number` は F64 として送る。
`Date` は受け取らない（マイクロ秒との換算を暗黙にやると桁の間違いに気づけない）。
TIMESTAMP と比べるなら `BigInt(d.getTime()) * 1000n` を渡す。

## 値のマッピング

| 論理型 | JS |
|---|---|
| BOOLEAN | `boolean` |
| TINYINT / SMALLINT / INTEGER / DATE | `number` |
| BIGINT / TIME / TIMESTAMP | `bigint`（TIMESTAMP はエポックからのマイクロ秒） |
| HUGEINT / UBIGINT | `bigint` |
| FLOAT / DOUBLE | `number` |
| DECIMAL | `string`（precision/scale 適用済み。例 `"1.0050"`） |
| VARCHAR | `string`（UTF-8 デコード済み） |
| BLOB | `Uint8Array` |
| NULL | `null` |

DECIMAL を `number` にすると 18 桁を超えたところで丸まる。桁を落とさないために
文字列で返している。近似でよければ `Number(row.amount)` すればよい。

TIMESTAMP を `Date` にするヘルパを同梱している。ミリ秒精度に丸まる点に注意。

```js
timestampToDate(row.d);  // BigInt(micros) -> Date
dateToDate(row.day);     // DATE(日数)     -> Date
```

## I/O とキャッシュ

エンジンは決してブロックしない。バイトが足りなくなると `NEED_IO` と
`{table, offset, len}` の列を返してくるので、ホストは次をやる。

1. **結合** — 隙間が 1 MiB 未満のレンジは 1 本にまとめる。100 KB の穴を挟んだ
   400 KB × 2 回より、900 KB を 1 回取る方が速い。エンジンが RowGroup 単位で要求を
   まとめて返すのはこのためなので、1 本ずつ投げてその意図を潰さない。
2. **並列取得** — 結合後のレンジは `Promise.all` でまとめて取る。URL なら
   `Range: bytes=start-end`、メモリ / Blob なら slice。
3. **供給** — `ahiru_provide` で渡してループを続ける。同じ要求が繰り返され、かつ
   1 バイトも増えていない場合はライブロックとみなして `E504` を投げる。

### コーデック委譲

コアは GZIP と ZSTD を持たない。持たせないことがコアが小さい理由そのもの
（DESIGN.md §6）。内蔵しないコーデックに当たるとエンジンは `NEED_CODEC` を返し、
`{table, codec, offset, len, out_len}` を並べてくる。ホストは:

- **GZIP** … `DecompressionStream('gzip')`。ブラウザにも Node 18+ にもあるので
  追加バイトはゼロ。
- **ZSTD** … `crates/ahiru-zstd` を別 wasm として**初回要求時に**読み込む
  （`zstdUrl` / `zstdBinary` / `zstdModule`）。指定が無ければ ZSTD と名指しで
  E201 を投げる。
- それ以外（BROTLI など） … E201「unsupported compression codec」。

圧縮ブロックは直前の `NEED_IO` で取得済みなので、**展開のために取り直しはしない**。
そのために取得済みバイトの控えをテーブルごとに持つ（上限は `cacheSize`、超えたら
古い順に捨てて、必要になったらキャッシュから取り直す）。一度も取っていない範囲を
要求されたら、黙って取りに行かずエンジン側の不整合として E900 を投げる。

キャッシュは `(source, offset, len)` の完全一致キー。`"memory"` は容量上限付きの
LRU（既定 64 MiB、`cacheSize` で変更可）。`MemoryCache` のインスタンスを直接渡せば
複数の `AhiruDB` で共有できる（この場合 `close()` では消さない）。`"cache-api"` は
現状メモリ実装に縮退する。

## wasm メモリの扱い（実装時の注意）

`ahiru_alloc` / `ahiru_provide` は wasm のヒープを伸ばすことがあり、伸びた瞬間に
既存の `TypedArray` ビューは detach する。壊れ方が静かなので方針を固定してある。

- `memory.buffer` へのビューは wasm 呼び出しをまたいで保持しない。呼び出し直後に
  `new Uint8Array(memory.buffer)` で取り直す。
- `ahiru_out_ptr()` のバッファは次の `ahiru_query_step` / `ahiru_schema` で作り直される。
  値を後で使うなら、次の呼び出しの前に JS 側へ移し終えていること。
- `query()` は行オブジェクトへその場で詰め替えるのでコピーを省く。`stream()` は
  バッチを呼び出し側に渡すため、列バッファを必ずコピーしてから yield する。
- 結果バッファは 4 バイト境界しか保証されない。8 バイト境界に乗らない
  `Float64Array` / `BigInt64Array` の列は、ビューではなくコピーの上に作る。

## エラー

wasm はコード（数値）しか返さない。メッセージは `errors.js` の表で組み立てる。
これだけで wasm から 20 KB 前後の文字列を追い出せる（DESIGN.md §10）。

```js
try {
  await db.query('SELECT FROM');
} catch (e) {
  e.code;    // 301
  e.message; // "[E301] unexpected token"
  e.sql;     // 実行しようとした SQL
}
```

`errors.js` は `crates/ahiru-core/src/error.rs` の `Code` と `message()` の写しなので、
**必ず両方を同時に直すこと**。ずれたらテスト（`errors.js は error.rs …`）が落ちる。

## テスト

```sh
./scripts/size.sh          # target/ahiru-core.wasm を作る
node --test 'js/test/*.test.mjs'
```

値の正解は `duckdb` CLI から取っている（インストール済みであること）。
レンジ取得のテストは、`duckdb` で生成した 2 MB ほどの Parquet を一時ディレクトリに
置いて使う（64 KiB のフッタ投機取得だけでファイル全体が読めてしまうと、
射影プッシュダウンの検証にならないため）。

CSV / JSONL のテストには `--features csv,jsonl` 入りの wasm が要る。テストが
`target/ahiru-core-full.wasm` を自動でビルドする（`AHIRU_WASM_FULL` で差し替え可）。
ZSTD のテストは `crates/ahiru-zstd` をビルドして使う。どちらも用意できなければ
そのテストだけ skip される。

> Node 24 では `node --test js/test/`（ディレクトリ指定）が動かない。
> 上のように glob で渡すか、引数なしの `node --test` を使う。

## 制限

- BROTLI / LZO / LZ4（フレーム付き）は展開しない（E201 でコーデック名を出す）。
  委譲の口はコアにあるので、対応を足すときは JS だけの変更で済む。
- SQL のテーブル関数は `parquet('...')` だけ。CSV / JSONL をパス指定で直接
  参照する構文は無いので、`register` してから名前で引く。
