# Slang Kernel MLX Integration Workflow

This guide is the canonical workflow for adding or modifying Slang kernels and regenerating MLX JSON specs used by runtime.

## Goal

- Edit Slang kernel logic.
- Regenerate MLX kernel JSON specs.
- Keep Swift loader names, I/O signatures, and bundle files aligned.
- Verify build success.

## 0. Preconditions

- Working directory is repo root (`GaussianSplattingMlx`).
- `slangc` is installed, or Docker image `slang-slang` is available.
- Avoid discarding branch changes unless explicitly instructed.

## 1. Active Kernel Boundaries

1. Projection path: `slang/gaussian_projection_kernels.slang`
2. Tile-global path: `slang/gaussian_tile_global_kernels.slang`
3. SSIM path: `slang/ssim_kernels.slang`

Legacy per-tile kernels (`gaussian_tile_forward` / `gaussian_tile_backward`) are retired and not part of active runtime.
Legacy screen-space standalone kernel file (`gaussian_screen_space_kernels.slang`) is also retired.

## 2. Edit Phase

1. Update Slang source entry points.
2. If entry names or arguments change:
   - Update the corresponding build script `entries=(...)`.
   - Update `input_names` and `output_names` mappings in that build script.
   - Keep `scripts/slang/compose.yml` command list in sync.
3. If runtime wiring changes:
   - Update `SlangKernelSpecLoader.loadKernel(named:)` callsites.
   - Update `GaussianRenderer.swift` custom function glue.

## 3. Generate Phase (Slang -> Metal -> MLX JSON)

Preferred:

```bash
bash scripts/slang/build_projection_slang_mlx.sh
bash scripts/slang/build_screen_space_slang_mlx.sh
bash scripts/slang/build_tile_global_slang_mlx.sh
bash scripts/slang/build_ssim_slang_mlx.sh
```

Docker compose:

```bash
docker compose -f scripts/slang/compose.yml build
docker compose -f scripts/slang/compose.yml up
```

## 4. Integration Phase

1. Confirm regenerated JSON files exist in `GaussianSplattingMlx/Slang/`.
2. Confirm JSON kernel IDs match Swift `loadKernel(named:)` names exactly.
3. Confirm input/output tensor shape and dtype conventions still match runtime assumptions.

## 5. Verification Phase

```bash
/bin/zsh -lc "set -o pipefail; xcodebuild -project GaussianSplattingMlx.xcodeproj -scheme GaussianSplattingMlx -configuration Debug -sdk iphonesimulator build 2>&1 | rg 'error:|BUILD SUCCEEDED'"
```

Expected: `** BUILD SUCCEEDED **`

## 6. Critical Pitfalls

### A. Address-space / alias artifacts in generated source

If MLX compile fails around `threadgroup`/`device` pointer mismatches, inspect generated JSON source for pointer-alias artifacts.

```bash
rg -n "= &gs|\\(\\*gs" GaussianSplattingMlx/Slang/*_mlx.json
```

Expected: no matches.

### B. Missing script registration

Adding a Slang entry without adding it to build script mapping causes missing JSON bundles at runtime.

### C. Loader ID mismatch

`loadKernel(named: "..._mlx")` must match JSON filename stem exactly.

## 7. Pre-flight Checklist

- [ ] Slang source edits are complete.
- [ ] Build script mappings are updated.
- [ ] JSON specs regenerated into `GaussianSplattingMlx/Slang/`.
- [ ] Swift runtime integration points updated if needed.
- [ ] `xcodebuild` verified.
- [ ] `git status` reviewed for unintended files.

## 8. Primary Target Files

- `slang/gaussian_projection_kernels.slang`
- `slang/gaussian_tile_global_kernels.slang`
- `slang/ssim_kernels.slang`
- `scripts/slang/build_projection_slang_mlx.sh`
- `scripts/slang/build_screen_space_slang_mlx.sh`
- `scripts/slang/build_tile_global_slang_mlx.sh`
- `scripts/slang/build_ssim_slang_mlx.sh`
- `scripts/slang/convert_slang_metal_to_mlx.py`
- `scripts/slang/compose.yml`
- `GaussianSplattingMlx/Slang/*_mlx.json`
- `GaussianSplattingMlx/Trainer/GaussianRenderer.swift`
