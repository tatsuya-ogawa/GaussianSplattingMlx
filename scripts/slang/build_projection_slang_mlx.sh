#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/build/slang_projection}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi

BUNDLE_DIR="${2:-$ROOT_DIR/GaussianSplattingMlx/Slang}"
if [[ "$BUNDLE_DIR" != /* ]]; then
  BUNDLE_DIR="$ROOT_DIR/$BUNDLE_DIR"
fi

SLANG_SRC="$ROOT_DIR/slang/gaussian_projection_kernels.slang"
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
  SLANG_INCLUDE_ARG="$ROOT_DIR/slang"
elif docker image inspect slang-slang >/dev/null 2>&1; then
  SLANGC_CMD=(docker run --rm -v "$ROOT_DIR:/work" -w /work slang-slang slangc)
  SLANG_SRC_ARG="/work/slang/gaussian_projection_kernels.slang"
  OUT_DIR_ARG="/work/$OUT_DIR_REL"
  SLANG_INCLUDE_ARG="/work/slang"
elif [[ -f "$DOCKERFILE_PATH" ]] && command -v docker >/dev/null 2>&1; then
  echo "[slang-projection] building docker image slang-slang from $DOCKERFILE_PATH"
  docker build -t slang-slang -f "$DOCKERFILE_PATH" "$ROOT_DIR" >/dev/null
  SLANGC_CMD=(docker run --rm -v "$ROOT_DIR:/work" -w /work slang-slang slangc)
  SLANG_SRC_ARG="/work/slang/gaussian_projection_kernels.slang"
  OUT_DIR_ARG="/work/$OUT_DIR_REL"
  SLANG_INCLUDE_ARG="/work/slang"
else
  echo "slangc not found and docker image 'slang-slang' is unavailable." >&2
  echo "Install slangc or build the docker image first." >&2
  exit 1
fi

entries=(
  gaussian_projection_screen_fused_forward
  gaussian_projection_screen_fused_backward
)

for entry in "${entries[@]}"; do
  echo "[slang-projection] compiling $entry"
  "${SLANGC_CMD[@]}" "$SLANG_SRC_ARG" -I "$SLANG_INCLUDE_ARG" -target metal -entry "$entry" -stage compute -o "$OUT_DIR_ARG/$entry.metal"

  echo "[slang-projection] converting $entry -> MLX JSON"
  input_names=""
  output_names=""
  case "$entry" in
    gaussian_projection_screen_fused_forward)
      input_names="fused_scales_1,fused_rotations_1,fused_means3d_1,fused_shs_1,fused_cameraCenter_1,fused_viewMatrix_1,fused_projMatrix_1,fused_fovX_1,fused_fovY_1,fused_focalX_1,fused_focalY_1,fused_imageWidth_1,fused_imageHeight_1,fused_counts_1"
      output_names="fused_outMeans2d_1,fused_outDepths_1,fused_outColor_1,fused_outCov2d_1,fused_outConic_1,fused_outRadii_1,fused_outRectMin_1,fused_outRectMax_1"
      ;;
    gaussian_projection_screen_fused_backward)
      input_names="fused_bwd_scales_1,fused_bwd_rotations_1,fused_bwd_means3d_1,fused_bwd_shs_1,fused_bwd_cameraCenter_1,fused_bwd_viewMatrix_1,fused_bwd_projMatrix_1,fused_bwd_fovX_1,fused_bwd_fovY_1,fused_bwd_focalX_1,fused_bwd_focalY_1,fused_bwd_imageWidth_1,fused_bwd_imageHeight_1,fused_bwd_cotDepths_1,fused_bwd_cotMeans2d_1,fused_bwd_cotCov2d_1,fused_bwd_cotColor_1,fused_bwd_cotConic_1,fused_bwd_counts_1"
      output_names="fused_bwd_gradScales_1,fused_bwd_gradRotations_1,fused_bwd_gradMeans3d_1,fused_bwd_gradShs_1,fused_bwd_gradCameraCenterPoint_1"
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
for entry in "${entries[@]}"; do
  cp "$OUT_DIR/${entry}_mlx.json" "$BUNDLE_DIR/"
done

echo "[slang-projection] done"
echo "[slang-projection] generated JSON copied to: $BUNDLE_DIR"
