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

mkdir -p "$OUT_DIR"

mkdir -p "$BUNDLE_DIR"

echo "[slang-screen] done"
echo "[slang-screen] no active screen-space entries; nothing generated"
echo "[slang-screen] bundle directory ensured at: $BUNDLE_DIR"
