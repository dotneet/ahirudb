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

echo
ls -la
echo
echo "OK: tests/data を再生成しました"
