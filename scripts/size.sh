#!/usr/bin/env bash
# wasm バイナリのサイズを測り、1 MiB のハードリミットを超えたら落ちる。
#
# 1MB は「最後に測る目標」ではなく「毎 PR で守る制約」として扱う（DESIGN.md §11）。
# CI ではこのスクリプトをそのままゲートに使う。
#
# 追加フォーマット (csv / jsonl) はフィーチャで切れるので、構成ごとの実測値を
# 並べて出す。「CSV を足すと何 KB 増えるのか」を毎回可視化しておかないと、
# フィーチャゲートが形骸化する。
set -euo pipefail

cd "$(dirname "$0")/.."

LIMIT=${LIMIT:-1048576}
OUT=target/ahiru-core.wasm

build_one() { # $1=フィーチャ, $2=出力先
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
  echo "!! wasm-opt が無いので最適化前のサイズを報告します"
  echo "   (brew install binaryen / apt install binaryen で入ります)"
  echo
fi

printf '%-24s %10s %10s %8s\n' "構成" "raw" "gzip -9" "予算比"
printf '%-24s %10s %10s %8s\n' "------------------------" "----------" "----------" "--------"

# 計測結果はグローバルに置く。サブシェルで受けると表の行まで飲み込まれるため。
SIZE=0
measure() { # $1=ラベル, $2=フィーチャ
  local tmp="target/size-probe.wasm" gz
  build_one "$2" "$tmp"
  SIZE=$(wc -c < "$tmp" | tr -d ' ')
  gz=$(gzip -9 -c "$tmp" | wc -c | tr -d ' ')
  printf '%-24s %10s %10s %7.1f%%\n' "$1" "$SIZE" "$gz" "$(echo "100*$SIZE/$LIMIT" | bc -l)"
}

# 配布の基準となる構成（Parquet のみ）を最後に測って OUT に残す。
measure "parquet + csv" "csv";           CSV_SIZE=$SIZE
measure "parquet + jsonl" "jsonl";       JSONL_SIZE=$SIZE
measure "parquet + csv + jsonl" "csv,jsonl"; ALL_SIZE=$SIZE
measure "parquet のみ (既定)" "";        BASE_SIZE=$SIZE
cp target/size-probe.wasm "$OUT"

# ZSTD は別モジュール。コアの予算には数えないが、実際に配る総量は見えるようにする。
ZSTD_OUT=target/ahiru-zstd.wasm
if cargo build --profile wasm --target wasm32-unknown-unknown -p ahiru-zstd \
     --no-default-features >/dev/null 2>&1; then
  cp target/wasm32-unknown-unknown/wasm/ahiru_zstd.wasm "$ZSTD_OUT"
  if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz --strip-debug --strip-producers --enable-bulk-memory \
      --enable-nontrapping-float-to-int \
      -o "$ZSTD_OUT.opt" "$ZSTD_OUT" && mv "$ZSTD_OUT.opt" "$ZSTD_OUT"
  fi
  ZSIZE=$(wc -c < "$ZSTD_OUT" | tr -d ' ')
  ZGZ=$(gzip -9 -c "$ZSTD_OUT" | wc -c | tr -d ' ')
  printf '%-24s %10s %10s %8s\n' "ahiru-zstd (別モジュール)" "$ZSIZE" "$ZGZ" "予算外"
fi

echo
printf 'csv の追加分   : %+d bytes\n' "$((CSV_SIZE - BASE_SIZE))"
printf 'jsonl の追加分 : %+d bytes\n' "$((JSONL_SIZE - BASE_SIZE))"
printf '全部入り       : %d bytes (%.1f%% of %d)\n' \
  "$ALL_SIZE" "$(echo "100*$ALL_SIZE/$LIMIT" | bc -l)" "$LIMIT"
echo

if command -v twiggy >/dev/null 2>&1; then
  echo "==> twiggy top 25 (parquet のみ)"
  twiggy top -n 25 "$OUT" || true
else
  echo "(cargo install twiggy で関数ごとの内訳が見られます)"
fi

# ゲートは全部入りで判定する。配布構成を絞れば通る、では守れていない。
if [ "$ALL_SIZE" -gt "$LIMIT" ]; then
  echo "FAIL: $ALL_SIZE > $LIMIT" >&2
  exit 1
fi
echo "OK: 全フォーマット込みで 1 MiB 以内"
