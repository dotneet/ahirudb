#!/usr/bin/env bash
# Differential probe: run the same SQL through ahiru and duckdb, show both outputs.
# Usage: probe.sh <sql> [file...]
set -e
cd "$(dirname "$0")/.."
SQL="$1"; shift
FILES=("$@")
echo "=== SQL: $SQL"
echo "--- ahiru:"
cargo run -q -p ahiru-cli -- query "${FILES[@]}" "$SQL" 2>&1 || echo "(error)"
echo "--- duckdb:"
if [ ${#FILES[@]} -gt 0 ]; then
  DUCKSQL=$(python3 -c '
import sys, re, os
sql = sys.argv[1]
files = sys.argv[2:]
names = {("t" if i==0 else f"t{i+1}"): f for i,f in enumerate(files)}
def repl(m):
    w = m.group(0)
    if w in names:
        p = os.path.abspath(names[w])
        if p.endswith(".csv"): return "read_csv_auto(%s)" % ("'"'"'"+p+"'"'"'")
        if p.endswith(".jsonl"): return "read_json_auto(%s)" % ("'"'"'"+p+"'"'"'")
        return "'"'"'%s'"'"'" % p
    return w
print(re.sub(r"\bt\d*\b", repl, sql))
' "$SQL" "${FILES[@]}")
else
  DUCKSQL="$SQL"
fi
duckdb -csv -c "$DUCKSQL" 2>&1 || echo "(error)"
