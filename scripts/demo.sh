#!/usr/bin/env bash
# One-command browser demo: builds the wasm binary the demo needs and
# launches a local static server with the demo page open in a browser.
#
# "Full-featured" here means every read-side format (parquet + csv + jsonl)
# plus zstd, matching the FULL_WASM build used by js/test/host.test.mjs.
# Write-path features (ddl/dml/export) are deliberately left out: this is a
# read-only "query files in your browser" showcase, not a database console.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=target/ahiru-core-full.wasm
PORT=${PORT:-8787}

echo "==> Building $OUT (parquet + csv + jsonl + zstd)"
cargo build --profile wasm --target wasm32-unknown-unknown -p ahiru-core \
  --no-default-features --features csv,jsonl,zstd
cp target/wasm32-unknown-unknown/wasm/ahiru_core.wasm "$OUT"

if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Oz --strip-debug --strip-producers --enable-bulk-memory \
    --enable-nontrapping-float-to-int -o "$OUT.opt" "$OUT"
  mv "$OUT.opt" "$OUT"
else
  echo "!! wasm-opt not found, serving an unoptimized (larger) build"
  echo "   (brew install binaryen / apt install binaryen)"
fi

SIZE=$(wc -c < "$OUT" | tr -d ' ')
if command -v bc >/dev/null 2>&1; then
  printf '    %s bytes (%.1f%% of 1 MiB)\n' "$SIZE" "$(echo "100*$SIZE/1048576" | bc -l)"
else
  printf '    %s bytes\n' "$SIZE"
fi

echo "==> Starting demo server on port $PORT"
PORT="$PORT" node scripts/demo-server.mjs &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

# Give the server a moment to bind before opening a browser tab against it.
sleep 0.3

URL="http://localhost:$PORT/demo/"
echo "==> Demo running at $URL (Ctrl+C to stop)"
if command -v open >/dev/null 2>&1; then
  open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL"
fi

wait "$SERVER_PID"
