#!/usr/bin/env bash
# tests/data/ を再生成する。
#
# 生成物はリポジトリに追跡している（.gitignore の末尾に理由あり）。
# ahiru-core の単体テストがこれらを直接読むので、clean clone で `cargo test` が
# 通ることを優先した。このスクリプトは「どう作ったか」を残すためのもので、
# 普段は実行しなくてよい。
#
# 期待値の参照実装も DuckDB なので、DuckDB のバージョンを上げたときは
# 再生成してテストが通ることを確認すること。
set -euo pipefail

cd "$(dirname "$0")/../tests/data"

command -v duckdb >/dev/null || { echo "duckdb が必要です" >&2; exit 1; }
echo "duckdb: $(duckdb -version)"

# --- 基本のケース ---------------------------------------------------------
# 1000 行。NULL（5 行に 1 つ）、辞書エンコードされる文字列、BOOLEAN、
# TIMESTAMP をひと通り含む。RowGroup を小さくして複数ページを作る。
duckdb -c "
COPY (SELECT i::INTEGER                                            AS id,
             ('name_' || (i % 7))::VARCHAR                         AS name,
             (i * 1.5)::DOUBLE                                     AS score,
             (i % 3 = 0)                                           AS flag,
             CASE WHEN i % 5 = 0 THEN NULL ELSE (i * 100)::BIGINT END AS big,
             DATE '2024-01-01' + INTERVAL (i % 365) DAY            AS d
      FROM range(0, 1000) t(i))
TO 'basic.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 400);"

# 同じ内容を CSV と JSONL でも。3 フォーマットが同じ結果を返すことの検証に使う。
duckdb -c "COPY (SELECT * FROM 'basic.parquet') TO 'basic.csv'   (FORMAT CSV, HEADER);"
duckdb -c "COPY (SELECT * FROM 'basic.parquet') TO 'basic.jsonl' (FORMAT JSON);"
# 同じ内容を、改行区切りではなく単一の JSON 配列ファイルとしても
# （`format::json` = `read_json`/`read_json_auto` 相当のテスト用）。
duckdb -c "COPY (SELECT * FROM 'basic.parquet') TO 'basic_array.json' (FORMAT JSON, ARRAY true);"

# --- 複数 RowGroup / 複数分割 ---------------------------------------------
# 50000 行 × 8192 行 RowGroup = 7 RowGroup。分割境界の取りこぼしを検出する。
duckdb -c "
COPY (SELECT i::INTEGER AS id, (i % 97)::BIGINT AS k,
             ('v' || (i % 1000))::VARCHAR AS s, (i * 0.25)::DOUBLE AS f
      FROM range(0, 50000) t(i))
TO 'multi_rg.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 8192);"
duckdb -c "COPY (SELECT * FROM 'multi_rg.parquet') TO 'multi_rg.csv' (FORMAT CSV, HEADER);"

# --- コーデック別 ---------------------------------------------------------
# 非圧縮と、ホストに展開を委譲する 2 種（DESIGN.md §6）。
duckdb -c "COPY (SELECT i::INTEGER AS id FROM range(0,5000) t(i))
           TO 'plain.parquet' (FORMAT PARQUET, COMPRESSION UNCOMPRESSED);"
duckdb -c "COPY (SELECT i::INTEGER AS id FROM range(0,5000) t(i))
           TO 'zstd.parquet'  (FORMAT PARQUET, COMPRESSION ZSTD);"

# 文字列と浮動小数を含む版。委譲経路を集約クエリで確かめるのに使う。
for spec in "gzip:GZIP" "zstd2:ZSTD"; do
  name="${spec%%:*}"; codec="${spec##*:}"
  duckdb -c "
  COPY (SELECT i::INTEGER AS id, ('v' || (i % 13))::VARCHAR AS s, (i * 1.5)::DOUBLE AS f
        FROM range(0, 5000) t(i))
  TO '${name}.parquet' (FORMAT PARQUET, COMPRESSION ${codec});"
done

# --- 結合用 ---------------------------------------------------------------
# basic.name に対応するディメンション表。
duckdb -c "
COPY (SELECT (i % 7)::INTEGER AS nid, ('name_' || (i % 7))::VARCHAR AS label,
             (i * 3)::BIGINT AS w
      FROM range(0, 7) t(i))
TO 'dim.parquet' (FORMAT PARQUET);"

# キーが 2 つだけ重なる小さい表。外部結合の NULL 補完を目視できる大きさにしてある。
duckdb -c "COPY (SELECT i::INTEGER AS k, (i * 2)::INTEGER AS v FROM range(0,5) t(i))
           TO 'small_a.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT (i + 2)::INTEGER AS k, (i * 10)::INTEGER AS w FROM range(0,5) t(i))
           TO 'small_b.parquet' (FORMAT PARQUET);"

# --- CSV の引用符まわり ---------------------------------------------------
# 二重引用符のエスケープ、フィールド内改行、空フィールド。手書きでないと作れない。
printf 'a,b,c\n1,"he said ""hi""",2.5\n2,"multi\nline",3.5\n3,,4.5\n' > quoted.csv

# --- STRUCT（ネストしたスキーマ） ------------------------------------------
# 単一段の STRUCT。address 列の下に city / zip の 2 リーフ。
# ドット区切り名 (address.city, address.zip) に解決されることを確かめる。
duckdb -c "
COPY (SELECT i::INTEGER AS id, {'city': 'Tokyo', 'zip': (10000+i)::INTEGER} AS address
      FROM range(0,100) t(i))
TO 'struct1.parquet' (FORMAT PARQUET);"

# 3 段のネスト (nested.a.b.c)。深さ優先で辿る再帰が正しく動くかを見る。
duckdb -c "
COPY (SELECT i::INTEGER AS id, {'a': {'b': {'c': i::INTEGER}}} AS nested
      FROM range(0,20) t(i))
TO 'struct_deep.parquet' (FORMAT PARQUET);"

# STRUCT グループそのものが NULL になる行を混ぜたもの。definition level の
# 「途中のどのグループが NULL でも、子リーフは同じ 1 ビットの validity に
# 潰れる」という前提が本当に正しいかを実バイト列で確かめるための版。
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             CASE WHEN i % 3 = 0 THEN NULL
                  ELSE {'city': 'Tokyo', 'zip': (10000+i)::INTEGER} END AS address
      FROM range(0,30) t(i))
TO 'struct_null.parquet' (FORMAT PARQUET);"

# LIST は STRUCT ではなく REPEATED グループの実体。最小のケースとして残す
# （1 本の JSON 列 `[1,2,3]` に組み立てられることを確認する）。
duckdb -c "
COPY (SELECT i::INTEGER AS id, [1,2,3] AS xs FROM range(0,10) t(i))
TO 'list1.parquet' (FORMAT PARQUET);"

# --- LIST/MAP（Dremel 組み立て） -------------------------------------------
# NULL 配列・空配列・要素内 NULL・可変長を 1 本に混ぜる。definition level
# だけで「配列自体が NULL」と「配列は存在するが 0 要素」を区別できるかが
# ここでの本題（どちらも JSON にすると別の見た目になる: null vs []）。
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             CASE WHEN i % 5 = 0 THEN NULL
                  WHEN i % 5 = 1 THEN []::INTEGER[]
                  WHEN i % 5 = 2 THEN [i]
                  WHEN i % 5 = 3 THEN [i, NULL, i * 2]
                  ELSE [i, i + 1, i + 2, i + 3] END AS xs
      FROM range(0, 50) t(i))
TO 'list_varied.parquet' (FORMAT PARQUET);"

# LIST<STRUCT<...>>。配列の要素が構造体。既存の STRUCT フラット化とは別に、
# 部分木ごと JSON にする経路が使われることを確認する。
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             [{'a': i, 'b': ('s' || i)::VARCHAR}, {'a': i + 1, 'b': NULL}] AS items
      FROM range(0, 20) t(i))
TO 'list_of_struct.parquet' (FORMAT PARQUET);"

# STRUCT の中に LIST がある場合。STRUCT フラット化（address.city 方式）とは
# 切り分けて、STRUCT ごと 1 本の JSON 列になることを確認する。
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             {'name': ('n' || i)::VARCHAR, 'tags': ['t' || i, 't' || (i + 1)]} AS s
      FROM range(0, 20) t(i))
TO 'struct_with_list.parquet' (FORMAT PARQUET);"

# LIST<LIST<INT>>。3 段エンコーディングが 2 重になったケース（配列の配列）。
duckdb -c "
COPY (SELECT i::INTEGER AS id, [[i, i + 1], [], [i * 10]] AS xss
      FROM range(0, 10) t(i))
TO 'list_of_list.parquet' (FORMAT PARQUET);"

# MAP<VARCHAR, INT>。文字列キー。
duckdb -c "
COPY (SELECT i::INTEGER AS id, map(['a', 'b', 'c'], [i, i * 2, NULL]) AS m
      FROM range(0, 20) t(i))
TO 'map_basic.parquet' (FORMAT PARQUET);"

# MAP<INT, VARCHAR>。文字列以外のキー（内部表現の判断が問われるケース）。
duckdb -c "
COPY (SELECT i::INTEGER AS id, map([i, i + 1], ['v' || i, 'v' || (i + 1)]) AS m
      FROM range(0, 20) t(i))
TO 'map_int_key.parquet' (FORMAT PARQUET);"

# LIST<STRUCT<..., LIST<...>>>。3 段のネスト（配列 → 構造体 → 配列）。
# list_of_struct/struct_with_list はどちらも 2 段までしか組み合わせていない
# ので、Dremel 組み立てが3段以上でも repetition/definition level を
# 正しく積み重ねられるかをこれで確認する。
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             [{'name': ('n' || i)::VARCHAR, 'tags': ['t' || i, 't' || (i + 1)]},
              {'name': 'x', 'tags': []}] AS items
      FROM range(0, 10) t(i))
TO 'list_of_struct_with_list.parquet' (FORMAT PARQUET);"

# --- 複数ファイル 1 テーブル ------------------------------------------------
# 素の複数ファイル UNION（パーティションなし）。行数をわざと不揃いにして、
# 「各パートの行数を単純に足し合わせただけ」の取りこぼしを検出しやすくする。
mkdir -p multi
duckdb -c "COPY (SELECT i::INTEGER AS id, ('n' || i)::VARCHAR AS name FROM range(0, 100) t(i))
           TO 'multi/a.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT i::INTEGER AS id, ('n' || i)::VARCHAR AS name FROM range(100, 250) t(i))
           TO 'multi/b.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT i::INTEGER AS id, ('n' || i)::VARCHAR AS name FROM range(250, 480) t(i))
           TO 'multi/c.parquet' (FORMAT PARQUET);"

# Hive スタイルのパーティションディレクトリ。`year=`/`month=` をディレクトリ名
# から読み取れることと、パーティション列での絞り込みでファイル単位に絞れる
# ことの両方をこの 1 セットで確認する。
mkdir -p hive/year=2024/month=01 hive/year=2024/month=02 hive/year=2025/month=01
duckdb -c "COPY (SELECT * FROM range(0,300) t(id)) TO 'hive/year=2024/month=01/part.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT * FROM range(300,700) t(id)) TO 'hive/year=2024/month=02/part.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT * FROM range(700,1000) t(id)) TO 'hive/year=2025/month=01/part.parquet' (FORMAT PARQUET);"

# --- PIVOT/UNPIVOT ---------------------------------------------------------
# region × category の小さい表。amount を PIVOT で集約し、q1..q4 を UNPIVOT で
# 畳み込む。id/region/q1..q4 は「GROUP BY 省略時に自動で残る列」の確認に使う
# ので、category/amount 以外の列を複数用意してある。
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             (['north','south','east','west'])[1 + i % 4]::VARCHAR AS region,
             (['a','b','c'])[1 + i % 3]::VARCHAR AS category,
             (i * 10)::INTEGER AS amount,
             (i)::INTEGER AS q1,
             (i * 2)::INTEGER AS q2,
             (i * 3)::INTEGER AS q3,
             (i * 4)::INTEGER AS q4
      FROM range(0, 60) t(i))
TO 'pivot.parquet' (FORMAT PARQUET);"

# 上と同じ列構成だが手書きできる大きさの版。GROUP BY 自動検出や IN リストの
# 別名付けなど、出力を目で数えたいテストはこちらを使う。
duckdb -c "
COPY (SELECT * FROM (VALUES
  ('east', 'a', 10),
  ('east', 'b', 20),
  ('west', 'a', 30),
  ('west', 'b', 40),
  ('west', 'c', 5)
) AS t(region, category, amount))
TO 'pivot_small.parquet' (FORMAT PARQUET);"

# --- ページ単位の枝刈り（ColumnIndex/OffsetIndex/Bloom フィルタ）-----------
# `pagetest.parquet` は DuckDB ではなく pyarrow (parquet-cpp) で生成する。
# この環境の DuckDB（v1.4.4）は ColumnIndex/OffsetIndex は書くが Bloom
# フィルタの書き出しオプションを持たない（`COPY ... (FORMAT PARQUET,
# BLOOM_FILTER_COLUMNS [...])` は "Unrecognized option" で拒否される）。
# ColumnIndex/OffsetIndex/Bloom フィルタが揃った実ファイルが要るので、
# 対応している pyarrow を使う。id は 0..50000 の一意な昇順（等号述語で
# ちょうど 1 ページに絞り込めることを確認するため）、data_page_size を
# 小さくしてページ数を稼いでいる。生成後は
# `crates/ahiru-core/src/parquet/meta.rs` のテストで、pyarrow の書いた
# ColumnIndex/OffsetIndex とバイト単位で突き合わせている。
#
# duckdb が要らないので、他のブロックと違って `command -v duckdb` の外に
# 書いてある。再生成するには `pip install pyarrow` が要る。
if command -v python3 >/dev/null && python3 -c "import pyarrow" >/dev/null 2>&1; then
  python3 - <<'PY'
import pyarrow as pa
import pyarrow.parquet as pq

n = 50_000
ids = pa.array(range(n), type=pa.int32())
vals = pa.array([f"v{i}" for i in range(n)], type=pa.string())
table = pa.table({"id": ids, "s": vals})

pq.write_table(
    table, "pagetest.parquet",
    row_group_size=n,           # 1 RowGroup に固定し、ページ選択だけを見る
    data_page_size=4 * 1024,    # 小さいページを大量に作る
    write_page_index=True,      # ColumnIndex/OffsetIndex を書く
    bloom_filter_options={"id": {"ndv": n, "fpp": 0.01}},
    use_dictionary=False,       # 辞書化すると min/max の傾向が変わるので避ける
    compression="SNAPPY",
)
PY

  # LIST 列とページ単位の絞り込みの組み合わせ。id に等号/範囲 pruner が
  # 効いてページ選択が有効化されたとき、xs（複数物理列にまたがらない単純な
  # LIST だが、入れ子列であることには変わりない）が「ページ選択の対象外
  # として常に列チャンク全体を読み、選択された行範囲へ後から gather する」
  # フォールバック分岐 (`format::parquet::read_split` の
  # `None if desc.nested.is_some()`) を通ることを確認するための版。
  # DuckDB の COPY にはページサイズを制御するオプションが無いので、ここも
  # pyarrow を使う。
  python3 - <<'PY'
import pyarrow as pa
import pyarrow.parquet as pq

n = 2000
ids = pa.array(range(n), type=pa.int32())
xs = pa.array([[i, i + 1] if i % 7 != 0 else [] for i in range(n)], type=pa.list_(pa.int32()))
table = pa.table({"id": ids, "xs": xs})

pq.write_table(
    table, "list_pagetest.parquet",
    row_group_size=n,
    data_page_size=2 * 1024,
    write_page_index=True,
    use_dictionary=False,
    compression="SNAPPY",
)
PY
else
  echo "!! pyarrow が無いので pagetest.parquet / list_pagetest.parquet の再生成をスキップします" >&2
fi

# --- ブラウザデモ用（demo/app.js の cross-format JOIN サンプル） -----------
# customers は Parquet、orders.csv/regions.jsonl は手書きのプレーンテキスト
# （duckdb で作る理由が無いのでここには含めない）。
duckdb -c "
COPY (SELECT * FROM (VALUES
    (1, 'Alice', 'east'), (2, 'Bob', 'west'), (3, 'Carol', 'east'),
    (4, 'Dave', 'west'), (5, 'Erin', 'north'), (6, 'Frank', 'south')
  ) AS t(customer_id, name, region))
TO 'customers.parquet' (FORMAT PARQUET);"

echo
ls -la
echo
echo "OK: tests/data を再生成しました"
