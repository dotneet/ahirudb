# ahirudb

WASM 1MB 以内で動く、Parquet を直接クエリできる軽量 SQL エンジン。

DuckDB-WASM は数十 MB（brotli 後でも約 10 MB）ある。ahirudb は「DuckDB を小さく
する」のではなく、**最初から入れる機能を選ぶ**ことで 1 MiB の予算に収める。

設計の全体像は [docs/DESIGN.md](docs/DESIGN.md) を参照。

## 現在のサイズ

| 構成 | raw | gzip -9 | 予算比 |
|---|---:|---:|---:|
| `ahiru-core.wasm`（Parquet のみ） | 224.5 KiB | 93.5 KiB | 21.9% |
| `ahiru-core.wasm`（+ CSV + JSONL） | 250.2 KiB | 104.4 KiB | **24.4%** |
| `ahiru-zstd.wasm`（別モジュール） | 17.8 KiB | 8.3 KiB | 予算外 |

`wasm-opt` 未適用の値。適用すればさらに 2〜3 割縮む見込み。

`./scripts/size.sh` で計測する。構成ごとの内訳と、CSV / JSONL を足したときの
増分も出る。ゲートは**全部入りの構成**で判定する（配布構成を絞れば通る、では
守れていないため）。1 MiB を超えたら CI が落ちる。

## 実装状況

| 領域 | 状態 |
|---|---|
| フォーマット抽象化（`TableFormat`） | 完了 |
| Thrift Compact / Parquet メタデータ | 完了 |
| スキーマ解決（論理型・変換型・INT96・DECIMAL） | 完了（ネスト型は明示的に非対応） |
| ページデコーダ（PLAIN / RLE / 辞書 / DELTA 系） | 完了 |
| 圧縮（UNCOMPRESSED / SNAPPY / LZ4_RAW） | 完了 |
| 圧縮（GZIP はホスト委譲 / ZSTD は別モジュール） | 完了（コーデック委譲プロトコル） |
| SQL トークナイザ / Pratt パーサ | 完了 |
| バインダ・射影プッシュダウン・統計プルーニング | 完了 |
| 式バイトコード VM・カーネル | 完了 |
| Scan / Filter / Project / Limit | 完了 |
| CSV / TSV | 完了（フィーチャ `csv`） |
| JSONL | 完了（フィーチャ `jsonl`） |
| 集約（GROUP BY / HAVING / DISTINCT） | 完了 |
| ソート（ORDER BY / Top-N） | 完了 |
| 結合（INNER / LEFT / RIGHT / FULL / CROSS / 非等値） | 完了 |
| wasm ABI | 完了 |
| JS ホスト層 | 完了（Node/ブラウザ、レンジ取得・キャッシュ・コーデック委譲） |
| ZSTD 別モジュール（`ahiru-zstd`） | 完了（17.8 KiB、遅延ロード） |
| FROM 句の派生表 | 完了 |
| スカラ関数 / ウィンドウ関数 | 未着手 |

## 使い方（ネイティブ CLI）

開発とテストはネイティブで回す。wasm 越しのデバッグは効率が悪いため。

```bash
cargo run -p ahiru-cli -- schema tests/data/basic.parquet
```

```bash
cargo run -p ahiru-cli -- dump tests/data/basic.parquet 10
```

```bash
cargo run -p ahiru-cli -- query tests/data/basic.parquet "SELECT name, count(*) c FROM t GROUP BY name ORDER BY c DESC"
```

複数ファイルを渡すと `t`, `t2`, ... として結合できる。

```bash
cargo run -p ahiru-cli -- query tests/data/small_a.parquet tests/data/small_b.parquet "SELECT a.k, b.w FROM t AS a LEFT JOIN t2 AS b ON a.k = b.k ORDER BY a.k"
```

## テスト

```bash
cargo test
```

テストデータは DuckDB CLI で生成している。

SQL のエンドツーエンドテスト（`crates/ahiru-cli/tests/sql_e2e.rs`）は**期待値を
書かず、同じクエリを DuckDB でも実行して突き合わせる**。手で書いた期待値は
書き間違いがそのまま仕様になってしまうし、クエリを増やすたびに手計算が要る。
DuckDB を参照実装として使えば、クエリを 1 行足すだけで検証が増える。
DuckDB が入っていない環境では該当テストを飛ばす。

## サイズ計測

```bash
./scripts/size.sh
```

`wasm-opt`（binaryen）と `twiggy` が入っていれば、最適化後のサイズと関数ごとの
内訳も出る。

## 制限

意図的に残している制限は [docs/DESIGN.md §15](docs/DESIGN.md) にまとめてある。
要点だけ挙げると、スピルしない（上限超過は明示エラー）、ネスト型は非対応、
スカラ関数とウィンドウ関数は未実装。丸めの取り決め（浮動小数→整数は偶数丸め、
DECIMAL のスケール縮小は 0 から遠ざかる丸め）も同じ節に書いてある。

## ライセンス

MIT OR Apache-2.0
