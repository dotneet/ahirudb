#!/usr/bin/env bash
# Measures the wasm binary size and fails if it exceeds the 1 MiB hard limit.
#
# 1 MB is not a goal to measure at the end but a constraint every PR upholds (DESIGN.md §11).
# CI uses this script directly as the gate.
#
# The extra formats (csv / jsonl) are feature-gated, so measured sizes are
# listed per configuration. Without making "how many KB does adding CSV cost?"
# visible every time, the feature gates become a formality.
set -euo pipefail

cd "$(dirname "$0")/.."

LIMIT=${LIMIT:-1048576}
OUT=target/ahiru-core.wasm

build_one() { # $1=features, $2=output path
  local features="$1" out="$2"
  local args=(--profile wasm --target wasm32-unknown-unknown -p ahiru-core --no-default-features)
  [ -n "$features" ] && args+=(--features "$features")
  cargo build "${args[@]}" >/dev/null 2>&1
  cp target/wasm32-unknown-unknown/wasm/ahiru_core.wasm "$out"
  if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz --strip-debug --strip-producers --enable-bulk-memory \
    --enable-nontrapping-float-to-int -o "$out.opt" "$out"
    mv "$out.opt" "$out"
  fi
}

if ! command -v wasm-opt >/dev/null 2>&1; then
  echo "!! wasm-opt not found; reporting unoptimized sizes"
  echo "   (install it with brew install binaryen / apt install binaryen)"
  echo
fi

printf '%-24s %10s %10s %8s\n' "config" "raw" "gzip -9" "of budget"
printf '%-24s %10s %10s %8s\n' "------------------------" "----------" "----------" "--------"

# Measurements go in a global. Capturing them from a subshell would swallow the table rows.
SIZE=0
measure() { # $1=label, $2=features
  local tmp="target/size-probe.wasm" gz
  build_one "$2" "$tmp"
  SIZE=$(wc -c < "$tmp" | tr -d ' ')
  gz=$(gzip -9 -c "$tmp" | wc -c | tr -d ' ')
  printf '%-24s %10s %10s %7.1f%%\n' "$1" "$SIZE" "$gz" "$(echo "100*$SIZE/$LIMIT" | bc -l)"
}

# ZSTD is built in by default (the `zstd` feature); at roughly 13 KB its impact
# on the budget is small enough that splitting it into a separate module was
# judged unnecessary (DESIGN.md §6). The reference distribution config
# (Parquet + ZSTD, the default) is measured last so it is what remains in OUT.
# `NOZSTD_SIZE` exists only for the delta, so it is measured before "default"
# to avoid overwriting OUT.
measure "parquet only (no zstd)" "";         NOZSTD_SIZE=$SIZE
measure "parquet + csv" "zstd,csv";           CSV_SIZE=$SIZE
measure "parquet + jsonl" "zstd,jsonl";       JSONL_SIZE=$SIZE
measure "parquet + csv + jsonl" "zstd,csv,jsonl"; ALL_SIZE=$SIZE
measure "parquet only (default)" "zstd";     BASE_SIZE=$SIZE
cp target/size-probe.wasm "$OUT"

# ZSTD is built into the core by default (the `zstd` feature, included in the
# measurements above). The legacy delegation path for configurations without
# `zstd` (ahiru-zstd built standalone as a separate wasm module, loaded by the
# host on first demand) is still checked to build, as the opt-out alternative.
# `crate-type` defaults to rlib only (to avoid building the cdylib, and hitting
# link errors, when `ahiru-core` depends on it), so the standalone build
# overrides it explicitly with `cargo rustc --crate-type cdylib`.
#
# Cargo only uplifts a copy to the profile root for the crate-types declared in
# `[lib]`, so a `cdylib` forced on the command line is written under `deps/`
# with a hash suffix and may never appear at the profile root. Look in both
# places, newest first, rather than reporting the module as missing.
# Prints the newest candidate, or nothing. Careful with `set -euo pipefail`:
# a glob that matches nothing must not abort the whole script, which is what
# the previous unconditional `cp` of the profile-root path did -- it took the
# per-feature summary below down with it whenever the module was not uplifted.
find_zstd_wasm() {
  local dir=target/wasm32-unknown-unknown/wasm
  local candidates=() f
  for f in "$dir/ahiru_zstd.wasm" "$dir"/deps/ahiru_zstd-*.wasm; do
    [ -f "$f" ] && candidates+=("$f")
  done
  [ ${#candidates[@]} -eq 0 ] && return 0
  ls -t "${candidates[@]}" 2>/dev/null | head -1 || true
}

ZSTD_OUT=target/ahiru-zstd.wasm
cargo rustc --profile wasm --target wasm32-unknown-unknown -p ahiru-zstd \
  --no-default-features --features standalone -- --crate-type cdylib >/dev/null 2>&1 || true
ZSTD_SRC=$(find_zstd_wasm)
if [ -n "$ZSTD_SRC" ]; then
  cp "$ZSTD_SRC" "$ZSTD_OUT"
  if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz --strip-debug --strip-producers --enable-bulk-memory \
      --enable-nontrapping-float-to-int \
      -o "$ZSTD_OUT.opt" "$ZSTD_OUT" && mv "$ZSTD_OUT.opt" "$ZSTD_OUT"
  fi
  ZSIZE=$(wc -c < "$ZSTD_OUT" | tr -d ' ')
  ZGZ=$(gzip -9 -c "$ZSTD_OUT" | wc -c | tr -d ' ')
  printf '%-24s %10s %10s %8s\n' "ahiru-zstd (opt-out alt)" "$ZSIZE" "$ZGZ" "ref"
fi

echo
printf 'csv adds       : %+d bytes\n' "$((CSV_SIZE - BASE_SIZE))"
printf 'jsonl adds     : %+d bytes\n' "$((JSONL_SIZE - BASE_SIZE))"
printf 'zstd adds      : %+d bytes\n' "$((BASE_SIZE - NOZSTD_SIZE))"
printf 'everything on  : %d bytes (%.1f%% of %d)\n' \
  "$ALL_SIZE" "$(echo "100*$ALL_SIZE/$LIMIT" | bc -l)" "$LIMIT"
echo

if command -v twiggy >/dev/null 2>&1; then
  echo "==> twiggy top 25 (parquet only)"
  twiggy top -n 25 "$OUT" || true
else
  echo "(run cargo install twiggy to see the per-function breakdown)"
fi

# The gate is judged on the everything-on build. "It passes with a narrower feature set" is not upholding it.
if [ "$ALL_SIZE" -gt "$LIMIT" ]; then
  echo "FAIL: $ALL_SIZE > $LIMIT" >&2
  exit 1
fi
echo "OK: within 1 MiB with all formats included"
