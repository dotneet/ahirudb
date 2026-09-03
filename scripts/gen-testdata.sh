#!/usr/bin/env bash
# Regenerates tests/data/.
#
# The generated files are tracked in the repository (the reason is at the end of
# .gitignore). ahiru-core's unit tests read them directly, and having `cargo test`
# pass on a clean clone won out. This script exists to record how they were made;
# you do not normally need to run it.
#
# DuckDB is also the reference implementation for expected values, so after
# bumping the DuckDB version, regenerate and check that the tests still pass.
set -euo pipefail

cd "$(dirname "$0")/../tests/data"

command -v duckdb >/dev/null || { echo "duckdb is required" >&2; exit 1; }
echo "duckdb: $(duckdb -version)"

# --- Basic cases ----------------------------------------------------------
# 1000 rows. Covers NULLs (one in every five), dictionary-encoded strings,
# BOOLEAN, and TIMESTAMP. RowGroups are kept small so multiple pages are created.
duckdb -c "
COPY (SELECT i::INTEGER                                            AS id,
             ('name_' || (i % 7))::VARCHAR                         AS name,
             (i * 1.5)::DOUBLE                                     AS score,
             (i % 3 = 0)                                           AS flag,
             CASE WHEN i % 5 = 0 THEN NULL ELSE (i * 100)::BIGINT END AS big,
             DATE '2024-01-01' + INTERVAL (i % 365) DAY            AS d
      FROM range(0, 1000) t(i))
TO 'basic.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 400);"

# The same content as CSV and JSONL, used to verify all three formats agree.
duckdb -c "COPY (SELECT * FROM 'basic.parquet') TO 'basic.csv'   (FORMAT CSV, HEADER);"
duckdb -c "COPY (SELECT * FROM 'basic.parquet') TO 'basic.jsonl' (FORMAT JSON);"
# The same content as a single JSON array file rather than newline-delimited
# (to test `format::json`, i.e. the `read_json`/`read_json_auto` equivalent).
duckdb -c "COPY (SELECT * FROM 'basic.parquet') TO 'basic_array.json' (FORMAT JSON, ARRAY true);"

# --- Multiple RowGroups / multiple splits ---------------------------------
# 50000 rows x 8192-row RowGroups = 7 RowGroups. Detects rows dropped at split boundaries.
duckdb -c "
COPY (SELECT i::INTEGER AS id, (i % 97)::BIGINT AS k,
             ('v' || (i % 1000))::VARCHAR AS s, (i * 0.25)::DOUBLE AS f
      FROM range(0, 50000) t(i))
TO 'multi_rg.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 8192);"
duckdb -c "COPY (SELECT * FROM 'multi_rg.parquet') TO 'multi_rg.csv' (FORMAT CSV, HEADER);"

# --- Per codec ------------------------------------------------------------
# Uncompressed, plus the two codecs whose decompression is delegated to the host (DESIGN.md §6).
duckdb -c "COPY (SELECT i::INTEGER AS id FROM range(0,5000) t(i))
           TO 'plain.parquet' (FORMAT PARQUET, COMPRESSION UNCOMPRESSED);"
duckdb -c "COPY (SELECT i::INTEGER AS id FROM range(0,5000) t(i))
           TO 'zstd.parquet'  (FORMAT PARQUET, COMPRESSION ZSTD);"

# A variant with strings and floats. Used to exercise the delegation path with aggregate queries.
for spec in "gzip:GZIP" "zstd2:ZSTD"; do
  name="${spec%%:*}"; codec="${spec##*:}"
  duckdb -c "
  COPY (SELECT i::INTEGER AS id, ('v' || (i % 13))::VARCHAR AS s, (i * 1.5)::DOUBLE AS f
        FROM range(0, 5000) t(i))
  TO '${name}.parquet' (FORMAT PARQUET, COMPRESSION ${codec});"
done

# --- For joins ------------------------------------------------------------
# A dimension table keyed by basic.name.
duckdb -c "
COPY (SELECT (i % 7)::INTEGER AS nid, ('name_' || (i % 7))::VARCHAR AS label,
             (i * 3)::BIGINT AS w
      FROM range(0, 7) t(i))
TO 'dim.parquet' (FORMAT PARQUET);"

# A small table overlapping on only two keys. Sized so outer-join NULL padding is visible by eye.
duckdb -c "COPY (SELECT i::INTEGER AS k, (i * 2)::INTEGER AS v FROM range(0,5) t(i))
           TO 'small_a.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT (i + 2)::INTEGER AS k, (i * 10)::INTEGER AS w FROM range(0,5) t(i))
           TO 'small_b.parquet' (FORMAT PARQUET);"

# --- CSV quoting ----------------------------------------------------------
# Escaped double quotes, newlines inside fields, empty fields. Only writable by hand.
printf 'a,b,c\n1,"he said ""hi""",2.5\n2,"multi\nline",3.5\n3,,4.5\n' > quoted.csv

# --- STRUCT (nested schemas) ----------------------------------------------
# A single level of STRUCT: two leaves, city / zip, under the address column.
# Checks that they resolve to dotted names (address.city, address.zip).
duckdb -c "
COPY (SELECT i::INTEGER AS id, {'city': 'Tokyo', 'zip': (10000+i)::INTEGER} AS address
      FROM range(0,100) t(i))
TO 'struct1.parquet' (FORMAT PARQUET);"

# Three levels of nesting (nested.a.b.c). Exercises the depth-first recursion.
duckdb -c "
COPY (SELECT i::INTEGER AS id, {'a': {'b': {'c': i::INTEGER}}} AS nested
      FROM range(0,20) t(i))
TO 'struct_deep.parquet' (FORMAT PARQUET);"

# A version mixing in rows where the STRUCT group itself is NULL. Confirms against
# real bytes that the definition-level assumption holds: "whichever intermediate
# group is NULL, the child leaves collapse into the same single validity bit".
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             CASE WHEN i % 3 = 0 THEN NULL
                  ELSE {'city': 'Tokyo', 'zip': (10000+i)::INTEGER} END AS address
      FROM range(0,30) t(i))
TO 'struct_null.parquet' (FORMAT PARQUET);"

# A LIST is a REPEATED group rather than a STRUCT. Kept as the minimal case
# (checks that it assembles into a single JSON column `[1,2,3]`).
duckdb -c "
COPY (SELECT i::INTEGER AS id, [1,2,3] AS xs FROM range(0,10) t(i))
TO 'list1.parquet' (FORMAT PARQUET);"

# --- LIST/MAP (Dremel assembly) -------------------------------------------
# Mixes NULL arrays, empty arrays, NULL elements, and varying lengths into one
# column. The point here is whether definition levels alone distinguish "the
# array itself is NULL" from "the array exists but has 0 elements" (they look
# different in JSON: null vs []).
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             CASE WHEN i % 5 = 0 THEN NULL
                  WHEN i % 5 = 1 THEN []::INTEGER[]
                  WHEN i % 5 = 2 THEN [i]
                  WHEN i % 5 = 3 THEN [i, NULL, i * 2]
                  ELSE [i, i + 1, i + 2, i + 3] END AS xs
      FROM range(0, 50) t(i))
TO 'list_varied.parquet' (FORMAT PARQUET);"

# LIST<STRUCT<...>>: array elements are structs. Confirms that the path emitting
# a whole subtree as JSON is used, separately from the existing STRUCT flattening.
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             [{'a': i, 'b': ('s' || i)::VARCHAR}, {'a': i + 1, 'b': NULL}] AS items
      FROM range(0, 20) t(i))
TO 'list_of_struct.parquet' (FORMAT PARQUET);"

# A LIST inside a STRUCT. Kept distinct from STRUCT flattening (the address.city
# scheme) to confirm the whole STRUCT becomes a single JSON column.
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             {'name': ('n' || i)::VARCHAR, 'tags': ['t' || i, 't' || (i + 1)]} AS s
      FROM range(0, 20) t(i))
TO 'struct_with_list.parquet' (FORMAT PARQUET);"

# LIST<LIST<INT>>: the three-level encoding doubled up (an array of arrays).
duckdb -c "
COPY (SELECT i::INTEGER AS id, [[i, i + 1], [], [i * 10]] AS xss
      FROM range(0, 10) t(i))
TO 'list_of_list.parquet' (FORMAT PARQUET);"

# MAP<VARCHAR, INT>: string keys.
duckdb -c "
COPY (SELECT i::INTEGER AS id, map(['a', 'b', 'c'], [i, i * 2, NULL]) AS m
      FROM range(0, 20) t(i))
TO 'map_basic.parquet' (FORMAT PARQUET);"

# MAP<INT, VARCHAR>: non-string keys (a case that probes the internal representation).
duckdb -c "
COPY (SELECT i::INTEGER AS id, map([i, i + 1], ['v' || i, 'v' || (i + 1)]) AS m
      FROM range(0, 20) t(i))
TO 'map_int_key.parquet' (FORMAT PARQUET);"

# LIST<STRUCT<..., LIST<...>>>: three levels of nesting (array -> struct -> array).
# list_of_struct/struct_with_list only combine two levels, so this checks that
# Dremel assembly stacks repetition/definition levels correctly at three levels
# and beyond.
duckdb -c "
COPY (SELECT i::INTEGER AS id,
             [{'name': ('n' || i)::VARCHAR, 'tags': ['t' || i, 't' || (i + 1)]},
              {'name': 'x', 'tags': []}] AS items
      FROM range(0, 10) t(i))
TO 'list_of_struct_with_list.parquet' (FORMAT PARQUET);"

# --- Multiple files, one table ----------------------------------------------
# A plain multi-file UNION (no partitioning). Row counts are deliberately uneven
# to make it easier to catch "just summed each part's row count" mistakes.
mkdir -p multi
duckdb -c "COPY (SELECT i::INTEGER AS id, ('n' || i)::VARCHAR AS name FROM range(0, 100) t(i))
           TO 'multi/a.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT i::INTEGER AS id, ('n' || i)::VARCHAR AS name FROM range(100, 250) t(i))
           TO 'multi/b.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT i::INTEGER AS id, ('n' || i)::VARCHAR AS name FROM range(250, 480) t(i))
           TO 'multi/c.parquet' (FORMAT PARQUET);"

# Hive-style partition directories. This one set confirms both that `year=`/`month=`
# can be read out of directory names, and that filtering on a partition column
# narrows things down to individual files.
mkdir -p hive/year=2024/month=01 hive/year=2024/month=02 hive/year=2025/month=01
duckdb -c "COPY (SELECT * FROM range(0,300) t(id)) TO 'hive/year=2024/month=01/part.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT * FROM range(300,700) t(id)) TO 'hive/year=2024/month=02/part.parquet' (FORMAT PARQUET);"
duckdb -c "COPY (SELECT * FROM range(700,1000) t(id)) TO 'hive/year=2025/month=01/part.parquet' (FORMAT PARQUET);"

# --- PIVOT/UNPIVOT ---------------------------------------------------------
# A small region x category table. amount is aggregated with PIVOT and q1..q4 are
# folded with UNPIVOT. id/region/q1..q4 are used to check "columns retained
# automatically when GROUP BY is omitted", hence several columns beyond category/amount.
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

# The same column layout at a size you can write out by hand. Use this one for
# tests where you want to count the output by eye, such as automatic GROUP BY
# detection or aliasing in IN lists.
duckdb -c "
COPY (SELECT * FROM (VALUES
  ('east', 'a', 10),
  ('east', 'b', 20),
  ('west', 'a', 30),
  ('west', 'b', 40),
  ('west', 'c', 5)
) AS t(region, category, amount))
TO 'pivot_small.parquet' (FORMAT PARQUET);"

# --- Pruning a DECIMAL column ---------------------------------------------
# The literal in `d1 = 150` carries no scale, but the column stores 1500, so the
# pruner has to rescale it before comparing against statistics or hashing it into
# the Bloom filter. d1 is a DECIMAL(5,1) (physically INT32) and d2 a DECIMAL(15,2)
# (INT64), so both integer widths are covered; `d` adds a DATE column for the
# `DATE '...'` typed-literal pruners. Several RowGroups so pruning actually has
# something to drop, and the columns are dictionary-encoded, which is when DuckDB
# writes a Bloom filter.
duckdb -c "
COPY (SELECT i::INTEGER                                AS id,
             (100 + (i % 101))::DECIMAL(5,1)           AS d1,
             (3000000000 + (i % 101))::DECIMAL(15,2)   AS d2,
             DATE '2024-01-01' + ((i % 400)::INTEGER)  AS d
      FROM range(0, 800) t(i))
TO 'decimal_pruning.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 200);"

# The same columns with d1 at a different scale. Unioned with the file above as one
# multi-file table it reads back as DECIMAL(9,3), so a pruner scaled into the table's
# type no longer lines up with the first file's statistics -- `exec::pruners_fit_part`
# has to notice and read that part in full.
duckdb -c "
COPY (SELECT i::INTEGER                                AS id,
             (100 + (i % 101))::DECIMAL(8,3)           AS d1,
             (3000000000 + (i % 101))::DECIMAL(15,2)   AS d2,
             DATE '2024-01-01' + ((i % 400)::INTEGER)  AS d
      FROM range(0, 200) t(i))
TO 'decimal_scale3.parquet' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 100);"

# --- Page-level pruning (ColumnIndex/OffsetIndex/Bloom filter) -------------
# `pagetest.parquet` is generated with pyarrow (parquet-cpp), not DuckDB.
# The DuckDB in this environment (v1.4.4) writes ColumnIndex/OffsetIndex but has
# no option for writing Bloom filters (`COPY ... (FORMAT PARQUET,
# BLOOM_FILTER_COLUMNS [...])` is rejected with "Unrecognized option").
# A real file carrying ColumnIndex, OffsetIndex, and a Bloom filter is needed,
# so pyarrow, which supports it, is used instead. id is unique and ascending over
# 0..50000 (so an equality predicate can be confirmed to narrow to exactly one
# page), and data_page_size is kept small to get a decent page count. Once
# generated, the tests in `crates/ahiru-core/src/parquet/meta.rs` cross-check it
# byte for byte against the ColumnIndex/OffsetIndex pyarrow wrote.
#
# Since duckdb is not needed, unlike the other blocks this sits outside the
# `command -v duckdb` check. Regenerating it requires `pip install pyarrow`.
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
    row_group_size=n,           # pin to 1 RowGroup, so only page selection is exercised
    data_page_size=4 * 1024,    # create lots of small pages
    write_page_index=True,      # write ColumnIndex/OffsetIndex
    bloom_filter_options={"id": {"ndv": n, "fpp": 0.01}},
    use_dictionary=False,       # dictionary encoding changes the min/max distribution, so avoid it
    compression="SNAPPY",
)
PY

  # A LIST column combined with page-level filtering. Confirms that when an
  # equality/range pruner fires on id and page selection kicks in, xs (a simple
  # LIST that does not span multiple physical columns, but a nested column all
  # the same) takes the fallback branch that reads the whole column chunk,
  # exempt from page selection, and gathers the selected row ranges afterwards
  # (`None if desc.nested.is_some()` in `format::parquet::read_split`).
  # DuckDB's COPY has no option to control page size, so pyarrow is used here
  # as well.
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

  # A DOUBLE column with NaN rows. Writers (pyarrow here, and parquet-cpp/parquet-mr
  # generally) leave NaN out of min/max, while this engine orders NaN above every
  # other value -- so `d > 100.0` is true for the NaN rows even though they sit
  # outside `max`, and `format::range_may_match` must not prune on `max` for a
  # floating-point column. DuckDB writes NaN *into* max, which makes the statistics
  # incomparable and hides the bug, so pyarrow is used here too.
  python3 - <<'PY'
import pyarrow as pa
import pyarrow.parquet as pq

n = 800
vals = [float('nan') if i % 200 == 100 else float(i % 10) for i in range(n)]
table = pa.table({
    "id": pa.array(range(n), type=pa.int32()),
    "d": pa.array(vals, type=pa.float64()),
})
pq.write_table(
    table, "nan_stats.parquet",
    row_group_size=200,         # 4 RowGroups, one NaN in each
    write_statistics=True,
    use_dictionary=False,
    compression="SNAPPY",
)
PY

  # A footer that does not fit the 64 KiB speculative tail fetch
  # (`parquet::file::FOOTER_PROBE`). `format::parquet::resolve` has to notice and
  # refetch the exact footer range instead of probing the same tail again.
  #
  # The two files bracket the boundary exactly: the probe covers the last 65536
  # bytes and the trailer is 8 of them, so a footer of 65528 bytes is the largest
  # that still fits, and 65529 is the first that does not. Padding the footer to
  # a precise byte count is done by appending an unknown Thrift field (id 9999,
  # binary) to FileMetaData just before its stop byte -- valid Parquet that every
  # reader skips, and far smaller on disk than the hundreds of columns x row
  # groups it would otherwise take to grow a footer past 64 KiB.
  python3 - <<'PY'
import io, struct
import pyarrow as pa
import pyarrow.parquet as pq

NCOL, ROWS, NRG = 12, 24, 6
cols = {f"c{c}": pa.array([i * (c + 1) for i in range(ROWS)], type=pa.int32())
        for c in range(NCOL)}
buf = io.BytesIO()
pq.write_table(pa.table(cols), buf, row_group_size=ROWS // NRG,
               compression="SNAPPY", use_dictionary=False, store_schema=False)
src = buf.getvalue()


def uvarint(v):
    out = bytearray()
    while True:
        x = v & 0x7F
        v >>= 7
        out.append(x | 0x80 if v else x)
        if not v:
            return bytes(out)


def zigzag(v):
    return uvarint((v << 1) ^ (v >> 63))


def pad_to(src, target):
    n = struct.unpack("<I", src[-8:-4])[0]
    meta = src[-8 - n:-8]
    assert meta[-1] == 0, "FileMetaData must end with the Thrift stop byte"
    # 0x08 = (field-id delta 0, type BINARY) -> a long-form field id follows.
    fixed = len(meta) - 1 + 1 + len(zigzag(9999)) + 1
    for lenwidth in range(1, 6):
        pad = target - fixed - lenwidth
        if pad >= 0 and len(uvarint(pad)) == lenwidth:
            new = meta[:-1] + b"\x08" + zigzag(9999) + uvarint(pad) + b"\0" * pad + b"\0"
            assert len(new) == target
            return src[:-8 - n] + new + struct.pack("<I", target) + b"PAR1"
    raise SystemExit("no padding length reaches the target")


for target, name in ((65528, "footer_fit.parquet"), (65529, "footer_big.parquet")):
    open(name, "wb").write(pad_to(src, target))
PY

  # The parquet-mr (Spark/Hive) column-chunk layout: `data_page_offset` records
  # the chunk start, which is written *before* the dictionary page, so it equals
  # `dictionary_page_offset` instead of pointing past it. Page-selected reads must
  # still find the dictionary page. No writer available here produces that layout,
  # so a normal pyarrow file (dictionary-encoded, page index written) is rewritten
  # afterwards: every ColumnMetaData's `data_page_offset` (field 9) is set to its
  # `dictionary_page_offset` (field 11).
  python3 - <<'PY'
import struct
import pyarrow as pa
import pyarrow.parquet as pq

n = 4000
pq.write_table(
    pa.table({
        "id": pa.array(range(n), type=pa.int32()),
        "k": pa.array([i % 100 for i in range(n)], type=pa.int32()),
        "s": pa.array([f"v{i % 50}" for i in range(n)], type=pa.string()),
    }),
    "dict_mr.parquet",
    row_group_size=n, data_page_size=2 * 1024,
    write_page_index=True, use_dictionary=True, compression="SNAPPY",
)


def uvarint(b, p):
    r = s = 0
    while True:
        x = b[p]; p += 1
        r |= (x & 0x7F) << s
        if not x & 0x80:
            return r, p
        s += 7


def zz(b, p):
    u, p = uvarint(b, p)
    return (u >> 1) ^ -(u & 1), p


def enc_zz(v):
    v = (v << 1) ^ (v >> 63)
    out = bytearray()
    while True:
        x = v & 0x7F; v >>= 7
        out.append(x | 0x80 if v else x)
        if not v:
            return bytes(out)


patches = []


def walk(b, p, path):
    """Minimal Thrift compact-protocol struct walk; records where each
    ColumnMetaData's data_page_offset varint sits."""
    fields, last = {}, 0
    while True:
        h = b[p]; p += 1
        if h == 0:
            return fields, p
        ty, delta = h & 0xF, h >> 4
        fid, p = (last + delta, p) if delta else zz(b, p)
        last = fid
        vpos = p
        if ty in (1, 2):
            v = ty == 1
        elif ty == 3:
            v = b[p]; p += 1
        elif ty in (4, 5, 6):
            v, p = zz(b, p)
        elif ty == 7:
            v = None; p += 8
        elif ty == 8:
            ln, p = uvarint(b, p); v = b[p:p + ln]; p += ln
        elif ty in (9, 10):
            hh = b[p]; p += 1
            et, cnt = hh & 0xF, hh >> 4
            if cnt == 15:
                cnt, p = uvarint(b, p)
            v = []
            for _ in range(cnt):
                if et == 12:
                    fv, p = walk(b, p, path + [fid]); v.append(fv)
                elif et in (1, 2, 3):
                    p += 1
                elif et == 8:
                    ln, p = uvarint(b, p); p += ln
                elif et in (4, 5, 6):
                    _, p = zz(b, p)
                elif et == 7:
                    p += 8
                else:
                    raise SystemExit(f"unhandled list element type {et}")
        elif ty == 12:
            v, p = walk(b, p, path + [fid])
        else:
            raise SystemExit(f"unhandled field type {ty}")
        fields[fid] = (ty, v, vpos)
        # FileMetaData.row_groups(4) -> RowGroup.columns(1) -> ColumnChunk.meta_data(3)
        if path == [4, 1] and fid == 3 and 9 in v and 11 in v:
            patches.append((v[9][2], enc_zz(v[9][1]), enc_zz(v[11][1])))


src = open("dict_mr.parquet", "rb").read()
n = struct.unpack("<I", src[-8:-4])[0]
foot = bytearray(src[-8 - n:-8])
walk(foot, 0, [])
# Splice from the end so earlier positions stay valid when a varint changes width.
for pos, old, new in sorted(patches, key=lambda p: -p[0]):
    assert foot[pos:pos + len(old)] == old
    foot[pos:pos + len(old)] = new
open("dict_mr.parquet", "wb").write(
    src[:-8 - n] + bytes(foot) + struct.pack("<I", len(foot)) + b"PAR1")
print(f"dict_mr.parquet: rewrote {len(patches)} column chunks to the parquet-mr layout")
PY

  # A 0-row RowGroup either side of the data, alongside nested (LIST/STRUCT)
  # columns. pyarrow writes such a RowGroup for an empty table or an empty batch,
  # and its column metadata is `dictionary_page_offset=<real>, data_page_offset=0,
  # num_values=0` -- so the chunk's byte range has to start at the dictionary page,
  # and the nested read path (which walks the buffer to exhaustion rather than to a
  # row count) must not be entered for it at all.
  python3 - <<'PY'
import pyarrow as pa
import pyarrow.parquet as pq

st = pa.struct([("a", pa.int32()), ("t", pa.list_(pa.string()))])
schema = pa.schema([("id", pa.int32()), ("l", pa.list_(pa.int32())), ("s", st)])
empty = pa.table({"id": pa.array([], type=pa.int32()),
                  "l": pa.array([], type=pa.list_(pa.int32())),
                  "s": pa.array([], type=st)}, schema=schema)
rows = 30
data = pa.table({
    "id": pa.array(range(rows), type=pa.int32()),
    "l": pa.array([[i, i + 1] if i % 3 else [] for i in range(rows)],
                  type=pa.list_(pa.int32())),
    "s": pa.array([{"a": i, "t": [f"t{i}"]} for i in range(rows)], type=st),
}, schema=schema)
with pq.ParquetWriter("empty_rg_nested.parquet", schema, compression="SNAPPY") as w:
    w.write_table(empty)
    w.write_table(data)
    w.write_table(empty)
PY

  # FLOAT16: FIXED_LEN_BYTE_ARRAY(2) annotated with the FLOAT16 logical type, which
  # ahirudb widens to FLOAT. DuckDB's COPY has no half-precision type, so pyarrow
  # writes it. Covers zero/-zero, both infinities, NULL, the smallest normal, the
  # smallest subnormal, the largest finite value, and a value that is not exact in
  # binary16.
  python3 - <<'PY'
import pyarrow as pa
import pyarrow.parquet as pq

vals = [1.5, -2.5, 0.0, -0.0, float("inf"), float("-inf"), None,
        6.103515625e-05, 5.960464477539063e-08, 65504.0, 0.0999755859375]
pq.write_table(
    pa.table({
        "id": pa.array(range(len(vals)), type=pa.int32()),
        "h": pa.array(vals, type=pa.float32()).cast(pa.float16()),
    }),
    "float16.parquet", compression="SNAPPY",
)
PY
else
  echo "!! pyarrow not found; skipping regeneration of pagetest.parquet / list_pagetest.parquet / nan_stats.parquet / footer_fit.parquet / footer_big.parquet / dict_mr.parquet / empty_rg_nested.parquet / float16.parquet" >&2
fi

# --- INTERVAL (FIXED_LEN_BYTE_ARRAY(12)) ----------------------------------
# Months/days/milliseconds as three unsigned 32-bit little-endian integers.
# A plain INTEGER column sits next to it so "one unsupported column must not make
# the whole file unreadable" stays testable if the mapping is ever dropped.
duckdb -c "
COPY (SELECT * FROM (VALUES
  (1, INTERVAL '1 day'),
  (2, INTERVAL '13 months 5 days 3 hours 4 minutes 5.5 seconds'),
  (3, INTERVAL '0 days'),
  (4, INTERVAL '90 minutes')
) AS t(id, iv))
TO 'interval.parquet' (FORMAT PARQUET);"

# --- For the browser demo (the cross-format JOIN sample in demo/app.js) ----
# customers is Parquet; orders.csv/regions.jsonl are hand-written plain text
# (no reason to build them with duckdb, so they are not included here).
duckdb -c "
COPY (SELECT * FROM (VALUES
    (1, 'Alice', 'east'), (2, 'Bob', 'west'), (3, 'Carol', 'east'),
    (4, 'Dave', 'west'), (5, 'Erin', 'north'), (6, 'Frank', 'south')
  ) AS t(customer_id, name, region))
TO 'customers.parquet' (FORMAT PARQUET);"

echo
ls -la
echo
echo "OK: regenerated tests/data"
