#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/build/slang_tile}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi

# Legacy per-tile kernels are currently unused by runtime path.
# To avoid regenerating unused bundle JSON by default, only copy when explicitly requested.
BUNDLE_DIR="${2:-}"
if [[ -n "$BUNDLE_DIR" && "$BUNDLE_DIR" != /* ]]; then
  BUNDLE_DIR="$ROOT_DIR/$BUNDLE_DIR"
fi

SLANG_SRC="$ROOT_DIR/slang/gaussian_tile_kernels.slang"
CONVERTER="$ROOT_DIR/scripts/slang/convert_slang_metal_to_mlx.py"
DOCKERFILE_PATH="$ROOT_DIR/scripts/slang/Dockerfile"
if [[ ! -f "$DOCKERFILE_PATH" && -f "$ROOT_DIR/slang/Dockerfile" ]]; then
  DOCKERFILE_PATH="$ROOT_DIR/slang/Dockerfile"
fi

mkdir -p "$OUT_DIR"

if [[ "$OUT_DIR" == "$ROOT_DIR" ]]; then
  OUT_DIR_REL="."
elif [[ "$OUT_DIR" == "$ROOT_DIR/"* ]]; then
  OUT_DIR_REL="${OUT_DIR#"$ROOT_DIR/"}"
else
  echo "OUT_DIR must be inside repository: $OUT_DIR" >&2
  exit 1
fi

if command -v slangc >/dev/null 2>&1; then
  SLANGC_CMD=(slangc)
  SLANG_SRC_ARG="$SLANG_SRC"
  OUT_DIR_ARG="$OUT_DIR"
elif docker image inspect slang-slang >/dev/null 2>&1; then
  SLANGC_CMD=(docker run --rm -v "$ROOT_DIR:/work" -w /work slang-slang slangc)
  SLANG_SRC_ARG="/work/slang/gaussian_tile_kernels.slang"
  OUT_DIR_ARG="/work/$OUT_DIR_REL"
elif [[ -f "$DOCKERFILE_PATH" ]] && command -v docker >/dev/null 2>&1; then
  echo "[slang-tile] building docker image slang-slang from $DOCKERFILE_PATH"
  docker build -t slang-slang -f "$DOCKERFILE_PATH" "$ROOT_DIR" >/dev/null
  SLANGC_CMD=(docker run --rm -v "$ROOT_DIR:/work" -w /work slang-slang slangc)
  SLANG_SRC_ARG="/work/slang/gaussian_tile_kernels.slang"
  OUT_DIR_ARG="/work/$OUT_DIR_REL"
else
  echo "slangc not found and docker image 'slang-slang' is unavailable." >&2
  echo "Install slangc or build the docker image first." >&2
  exit 1
fi

entries=(
  gaussian_tile_forward
  gaussian_tile_backward
)

for entry in "${entries[@]}"; do
  echo "[slang-tile] compiling $entry"
  "${SLANGC_CMD[@]}" "$SLANG_SRC_ARG" -target metal -entry "$entry" -stage compute -o "$OUT_DIR_ARG/$entry.metal"

  echo "[slang-tile] converting $entry -> MLX JSON"
  input_names=""
  output_names=""
  atomic_outputs=()
  case "$entry" in
    gaussian_tile_forward)
      input_names="tileCoord_1,sortedDepths_1,sortedMeans2d_1,sortedConic_1,sortedOpacity_1,sortedColor_1,counts_1"
      output_names="outColor_1,outDepth_1,outAlpha_1"
      ;;
    gaussian_tile_backward)
      input_names="tileCoord_1,sortedDepths_1,sortedMeans2d_1,sortedConic_1,sortedOpacity_1,sortedColor_1,cotColor_1,cotDepth_1,cotAlpha_1,counts_1"
      output_names="gradSortedDepths_1,gradSortedMeans2d_1,gradSortedConic_1,gradSortedOpacity_1,gradSortedColor_1"
      atomic_outputs=(--atomic-outputs)
      ;;
    *)
      echo "Unknown entry: $entry" >&2
      exit 1
      ;;
  esac

  python3 "$CONVERTER" \
    --metal "$OUT_DIR/$entry.metal" \
    --entry "$entry" \
    --kernel-name "${entry}_slang_v2" \
    --input-names "$input_names" \
    --output-names "$output_names" \
    --swift-out "$OUT_DIR/${entry}_mlx.swift" \
    --json-out "$OUT_DIR/${entry}_mlx.json" \
    "${atomic_outputs[@]}"
done

if [[ -n "$BUNDLE_DIR" ]]; then
  mkdir -p "$BUNDLE_DIR"
  for entry in "${entries[@]}"; do
    cp "$OUT_DIR/${entry}_mlx.json" "$BUNDLE_DIR/"
  done
  echo "[slang-tile] done"
  echo "[slang-tile] generated JSON copied to: $BUNDLE_DIR"
else
  echo "[slang-tile] done"
  echo "[slang-tile] bundle copy skipped (pass BUNDLE_DIR as 2nd arg to copy)."
fi
