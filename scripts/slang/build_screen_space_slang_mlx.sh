#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/build/slang_screen_space}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi

BUNDLE_DIR="${2:-$ROOT_DIR/GaussianSplattingMlx/Slang}"
if [[ "$BUNDLE_DIR" != /* ]]; then
  BUNDLE_DIR="$ROOT_DIR/$BUNDLE_DIR"
fi

SLANG_SRC="$ROOT_DIR/slang/gaussian_screen_space_kernels.slang"
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
  SLANG_SRC_ARG="/work/slang/gaussian_screen_space_kernels.slang"
  OUT_DIR_ARG="/work/$OUT_DIR_REL"
else
  echo "slangc not found and docker image 'slang-slang' is unavailable." >&2
  echo "Install slangc or build the docker image first." >&2
  exit 1
fi

entries=(
  gaussian_screen_color_forward
  gaussian_screen_color_backward
  gaussian_screen_cov2d_forward
  gaussian_screen_cov2d_backward
  gaussian_screen_inverse2d_forward
  gaussian_screen_inverse2d_backward
)

for entry in "${entries[@]}"; do
  echo "[slang-screen] compiling $entry"
  "${SLANGC_CMD[@]}" "$SLANG_SRC_ARG" -target metal -entry "$entry" -stage compute -o "$OUT_DIR_ARG/$entry.metal"

  echo "[slang-screen] converting $entry -> MLX JSON"
  input_names=""
  output_names=""
  case "$entry" in
    gaussian_screen_color_forward)
      input_names="color_means3d_1,color_shs_1,color_cameraCenter_1,color_counts_1"
      output_names="color_outColor_1"
      ;;
    gaussian_screen_color_backward)
      input_names="color_means3d_1,color_shs_1,color_cameraCenter_1,color_cotColor_1,color_counts_1"
      output_names="color_gradMeans3d_1,color_gradShs_1"
      ;;
    gaussian_screen_cov2d_forward)
      input_names="cov_means3d_1,cov_cov3d_1,cov_viewMatrix_1,cov_fovX_1,cov_fovY_1,cov_focalX_1,cov_focalY_1,cov_counts_1"
      output_names="cov_outCov2d_1"
      ;;
    gaussian_screen_cov2d_backward)
      input_names="cov_means3d_1,cov_cov3d_1,cov_viewMatrix_1,cov_fovX_1,cov_fovY_1,cov_focalX_1,cov_focalY_1,cov_cotCov2d_1,cov_counts_1"
      output_names="cov_gradMeans3d_1,cov_gradCov3d_1"
      ;;
    gaussian_screen_inverse2d_forward)
      input_names="inverse_cov2d_1,inverse_counts_1"
      output_names="inverse_outConic_1"
      ;;
    gaussian_screen_inverse2d_backward)
      input_names="inverse_cov2d_1,inverse_cotConic_1,inverse_counts_1"
      output_names="inverse_gradCov2d_1"
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

echo "[slang-screen] done"
echo "[slang-screen] generated JSON copied to: $BUNDLE_DIR"
