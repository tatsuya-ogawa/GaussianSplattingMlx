#!/usr/bin/env python3
"""Generate active screen-space JSON from Slang source.

This script intentionally does not embed Metal shader source. It delegates to
`build_screen_space_slang_mlx.sh`, which compiles active entries in
`slang/gaussian_screen_space_kernels.slang` and runs the MLX JSON converter.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import tempfile


def _resolve_repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        required=True,
        help="Directory for generated .metal/.json artifacts",
    )
    parser.add_argument(
        "--bundle-dir",
        type=pathlib.Path,
        default=None,
        help="Directory to copy *_mlx.json into (default: GaussianSplattingMlx/Slang)",
    )
    args = parser.parse_args()

    repo_root = _resolve_repo_root()
    out_dir = args.out_dir.resolve()
    bundle_dir = (
        args.bundle_dir.resolve()
        if args.bundle_dir is not None
        else (repo_root / "GaussianSplattingMlx" / "Slang").resolve()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    build_script = repo_root / "scripts" / "slang" / "build_screen_space_slang_mlx.sh"

    temp_bundle_dir: pathlib.Path | None = None
    effective_bundle_dir = bundle_dir
    if out_dir == bundle_dir:
        temp_bundle_dir = pathlib.Path(tempfile.mkdtemp(prefix="slang_bundle_"))
        effective_bundle_dir = temp_bundle_dir

    try:
        subprocess.run(
            [
                "bash",
                str(build_script),
                str(out_dir),
                str(effective_bundle_dir),
            ],
            check=True,
        )
    finally:
        if temp_bundle_dir is not None:
            shutil.rmtree(temp_bundle_dir, ignore_errors=True)

    active_json = out_dir / "gaussian_screen_cov3d_backward_mlx.json"
    if not active_json.exists():
        raise FileNotFoundError(f"Expected screen-space JSON was not generated: {active_json}")

    print(f"Generated from Slang: {active_json}")
    if bundle_dir != out_dir:
        print(f"Copied *_mlx.json to: {bundle_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
