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

# ZSTD は既定で内蔵する（`zstd` フィーチャ、13 KB 程度で予算への影響が
# 小さいため別モジュールに分ける理由が無いと判断した。DESIGN.md §6）。
# 配布の基準となる構成（Parquet + ZSTD が既定）を最後に測って OUT に残す。
# `NOZSTD_SIZE` は増分を出すためだけの比較用なので、OUT を上書きしないよう
# 「既定」構成より先に測る。
measure "parquet のみ (zstd 無し)" "";       NOZSTD_SIZE=$SIZE
measure "parquet + csv" "zstd,csv";           CSV_SIZE=$SIZE
measure "parquet + jsonl" "zstd,jsonl";       JSONL_SIZE=$SIZE
measure "parquet + csv + jsonl" "zstd,csv,jsonl"; ALL_SIZE=$SIZE
measure "parquet のみ (既定)" "zstd";        BASE_SIZE=$SIZE
cp target/size-probe.wasm "$OUT"

# ZSTD は既定でコアに内蔵する（`zstd` フィーチャ、上の測定に含まれている）。
# `zstd` を外した構成向けの旧来の委譲経路（ahiru-zstd を別 wasm モジュールと
# して単独ビルドし、ホストが初回要求時に読み込む）も、opt-out した場合の
# 代替として引き続きビルドできることだけ確認しておく。
# `crate-type` は既定で rlib のみ（`ahiru-core` への依存時に cdylib まで
# ビルドしてリンクエラーになるのを避けるため）なので、単独ビルドでは
# `cargo rustc --crate-type cdylib` で明示的に上書きする。
ZSTD_OUT=target/ahiru-zstd.wasm
if cargo rustc --profile wasm --target wasm32-unknown-unknown -p ahiru-zstd \
     --no-default-features --features standalone -- --crate-type cdylib >/dev/null 2>&1; then
  cp target/wasm32-unknown-unknown/wasm/ahiru_zstd.wasm "$ZSTD_OUT"
  if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz --strip-debug --strip-producers --enable-bulk-memory \
      --enable-nontrapping-float-to-int \
      -o "$ZSTD_OUT.opt" "$ZSTD_OUT" && mv "$ZSTD_OUT.opt" "$ZSTD_OUT"
  fi
  ZSIZE=$(wc -c < "$ZSTD_OUT" | tr -d ' ')
  ZGZ=$(gzip -9 -c "$ZSTD_OUT" | wc -c | tr -d ' ')
  printf '%-24s %10s %10s %8s\n' "ahiru-zstd (opt-out 時の代替)" "$ZSIZE" "$ZGZ" "参考"
fi

echo
printf 'csv の追加分   : %+d bytes\n' "$((CSV_SIZE - BASE_SIZE))"
printf 'jsonl の追加分 : %+d bytes\n' "$((JSONL_SIZE - BASE_SIZE))"
printf 'zstd の追加分  : %+d bytes\n' "$((BASE_SIZE - NOZSTD_SIZE))"
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
