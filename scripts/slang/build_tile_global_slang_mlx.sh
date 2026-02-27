#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/build/slang_tile_global}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi

BUNDLE_DIR="${2:-$ROOT_DIR/GaussianSplattingMlx/Slang}"
if [[ "$BUNDLE_DIR" != /* ]]; then
  BUNDLE_DIR="$ROOT_DIR/$BUNDLE_DIR"
fi

SLANG_SRC="$ROOT_DIR/slang/gaussian_tile_global_kernels.slang"
CONVERTER="$ROOT_DIR/scripts/slang/convert_slang_metal_to_mlx.py"

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
  SLANG_SRC_ARG="/work/slang/gaussian_tile_global_kernels.slang"
  OUT_DIR_ARG="/work/$OUT_DIR_REL"
else
  echo "slangc not found and docker image 'slang-slang' is unavailable." >&2
  echo "Install slangc or build the docker image first." >&2
  exit 1
fi

entries=(
  count_tiles_per_gaussian
  generate_keys
  compute_tile_ranges
)

for entry in "${entries[@]}"; do
  echo "[slang-tile-global] compiling $entry"
  "${SLANGC_CMD[@]}" "$SLANG_SRC_ARG" -target metal -entry "$entry" -stage compute -o "$OUT_DIR_ARG/$entry.metal"

  echo "[slang-tile-global] converting $entry -> MLX JSON"
  input_names=""
  output_names=""
  case "$entry" in
    count_tiles_per_gaussian)
      input_names="ct_rectMin_1,ct_rectMax_1,ct_radii_1,ct_counts_1"
      output_names="ct_tilesTouched_1"
      ;;
    generate_keys)
      input_names="gk_depths_1,gk_rectMin_1,gk_rectMax_1,gk_radii_1,gk_offsets_1,gk_counts_1"
      output_names="gk_keysHigh_1,gk_keysLow_1,gk_gaussIdx_1"
      ;;
    compute_tile_ranges)
      input_names="tr_sortedKeysHigh_1,tr_counts_1"
      output_names="tr_tileRanges_1"
      ;;
    *)
      echo "Unknown entry: $entry" >&2
      exit 1
      ;;
  esac

  python3 "$CONVERTER" \
    --metal "$OUT_DIR/$entry.metal" \
    --entry "$entry" \
    --kernel-name "${entry}_slang_v1" \
    --input-names "$input_names" \
    --output-names "$output_names" \
    --swift-out "$OUT_DIR/${entry}_mlx.swift" \
    --json-out "$OUT_DIR/${entry}_mlx.json"
done

mkdir -p "$BUNDLE_DIR"
cp "$OUT_DIR"/*_mlx.json "$BUNDLE_DIR/"

echo "[slang-tile-global] done"
echo "[slang-tile-global] generated JSON copied to: $BUNDLE_DIR"
