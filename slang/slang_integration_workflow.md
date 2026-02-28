# Slang Kernel MLX Integration Workflow

This guide serves as a workflow for AI agents and developers working in this repository to add or modify custom GPU kernels (written in Slang) and safely integrate them into the MLX runtime.

## Goal

- Edit and add Slang kernel logic.
- Regenerate MLX kernel JSON bundles to reflect these changes.
- Correctly load, wire, and execute the kernels from Swift.
- Verify build success and runtime logic consistency.

## 0. Preconditions

- Working directory: The root of the repository (`GaussianSplattingMlx` project root).
- Ensure the Docker image `slang-slang` is available, **or** a local installation of `slangc` is present.
- Do not discard changes on the current branch unless instructed otherwise.

## 1. Plan Phase (Always perform first)

1. Identify the target system boundaries:
   - Tile processing path: `slang/gaussian_tile_kernels.slang`
   - Screen-space path: `slang/gaussian_screen_space_kernels.slang`
2. Identify loaders and invocation points:
   - Loader mechanism: `GaussianSplattingMlx/Trainer/SlangKernelSpecLoader.swift`
   - Invocation site: `GaussianSplattingMlx/Trainer/GaussianRenderer.swift`
3. Verify generation scripts:
   - For Tile operations: `scripts/slang/build_tile_slang_mlx.sh`
   - For Screen-space: `scripts/slang/build_screen_space_slang_mlx.sh`

## 2. Edit Phase

1. Update Slang Source code:
   - Add or edit Forward entry points.
   - If gradients are required, add or edit corresponding Backward entry points.
2. If new entry names are assigned:
   - Make sure to add the new name to the `entries=(...)` list at the top of the relevant bash build script.
   - Register proper `input_names` and `output_names` matching logic within the `case` blocks.
3. If new plumbing is needed in Swift:
   - Request the new spec utilizing `SlangKernelSpecLoader.loadKernel(named: "..._mlx")`.
   - Manually wrap inputs and outputs through `CustomFunction { Forward ... VJP ... }` inside `GaussianRenderer`.

## 3. Generate Phase (Slang -> Metal -> MLX JSON)

Preferred route via scripting:

```bash
bash scripts/slang/build_tile_slang_mlx.sh
bash scripts/slang/build_screen_space_slang_mlx.sh
```

In scenarios where environment checks block the shell script, apply a manual fallback (example below using Tile Forward / Backward setup):

```bash
mkdir -p build/slang_tile
docker run --rm -v "$PWD:/work" -w /work slang-slang \
  slangc /work/slang/gaussian_tile_kernels.slang -target metal \
  -entry gaussian_tile_forward -stage compute -o /work/build/slang_tile/gaussian_tile_forward.metal
docker run --rm -v "$PWD:/work" -w /work slang-slang \
  slangc /work/slang/gaussian_tile_kernels.slang -target metal \
  -entry gaussian_tile_backward -stage compute -o /work/build/slang_tile/gaussian_tile_backward.metal
python3 scripts/slang/convert_slang_metal_to_mlx.py --metal build/slang_tile/gaussian_tile_forward.metal \
  --entry gaussian_tile_forward --kernel-name gaussian_tile_forward_slang_v2 \
  --input-names "tileCoord_1,sortedDepths_1,sortedMeans2d_1,sortedConic_1,sortedOpacity_1,sortedColor_1,counts_1" \
  --output-names "outColor_1,outDepth_1,outAlpha_1" \
  --swift-out build/slang_tile/gaussian_tile_forward_mlx.swift \
  --json-out build/slang_tile/gaussian_tile_forward_mlx.json
python3 scripts/slang/convert_slang_metal_to_mlx.py --metal build/slang_tile/gaussian_tile_backward.metal \
  --entry gaussian_tile_backward --kernel-name gaussian_tile_backward_slang_v2 \
  --input-names "tileCoord_1,sortedDepths_1,sortedMeans2d_1,sortedConic_1,sortedOpacity_1,sortedColor_1,cotColor_1,cotDepth_1,cotAlpha_1,counts_1" \
  --output-names "gradSortedDepths_1,gradSortedMeans2d_1,gradSortedConic_1,gradSortedOpacity_1,gradSortedColor_1" \
  --swift-out build/slang_tile/gaussian_tile_backward_mlx.swift \
  --json-out build/slang_tile/gaussian_tile_backward_mlx.json
cp build/slang_tile/*_mlx.json GaussianSplattingMlx/Slang/
```

## 4. Integration Phase

1. Check that regenerated JSON files physically exist inside the application bundle target directory (`GaussianSplattingMlx/Slang/*.json`).
2. Verify that Kernel names embedded in the JSON directly match what the Swift loader asks for (`loadKernel(named:)`) down to the word.
3. Guarantee that shapes and DTypes constructed at the execution point in Swift perfectly complement the intended runtime semantic properties written in the Slang arguments.

## 5. Verification Phase

Run the project build and ascertain there are zero compilation errors:

```bash
/bin/zsh -lc "set -o pipefail; xcodebuild -project GaussianSplattingMlx.xcodeproj -scheme GaussianSplattingMlx -configuration Debug -sdk iphonesimulator build 2>&1 | rg 'error:|BUILD SUCCEEDED'"
```

Expected result: The console resolves with `** BUILD SUCCEEDED **`.

## 6. Critical Pitfalls

### A. Compile errors regarding `threadgroup` or `address_space`

Symptoms manifesting in exception logs may look like:
- `metal_array ... this object is in address space 'threadgroup', but method expects ... 'device'`
- Unnamed pointer reference warnings or mismatch faults.

**[Core Root Cause in this Repo]**  
The underlying issue occurs when temporary alias references implicitly introduced by the Slang compiler (e.g., a pointer alias assignment `x_0 = &x_1` invoking indexing access later like `(*x_0)[i]`) inadvertently leak straight into the raw MLX JSON `source` field string.

**[Current Mitigation Measures]**  
The Python generation script `scripts/slang/convert_slang_metal_to_mlx.py` runs an embedded logic `simplify_threadgroup_aliases(...)` to aggressively strip these. Nonetheless, always verify post-generation that there are zero assignments using these aliases:

```bash
rg -n "= &gs|\\(\\*gs" GaussianSplattingMlx/Slang/gaussian_tile_*_mlx.json
```
Expected result: No match found.

### B. Omission of script Entry registration

Neglecting to link `.slang` kernel additions back into the bash script `entries` arrays and `case` properties results in missing JSON bundles, producing fatal execution errors at runtime.

### C. Loader ID mismatch

The identifier called at the Swift side inside `loadKernel(named: "...")` must reflect the exact resulting filename representing `..._mlx.json` absent the extension. 

## 7. Pre-flight Checklist

- [ ] Changed and verified Slang target source variables.
- [ ] Updated bash routing variables mapping correctly (if necessary).
- [ ] Conclusive JSON bundles fully generated and relocated correctly in `GaussianSplattingMlx/Slang/`.
- [ ] Associated Swift caller components fixed/adjusted alongside kernel modifications.
- [ ] Verified `xcodebuild` registers without disruption.
- [ ] Enforced that unintended files do not get checked in per `git status`.

## 8. Primary Target Files

- `slang/gaussian_tile_kernels.slang`
- `slang/gaussian_screen_space_kernels.slang`
- `scripts/slang/build_tile_slang_mlx.sh`
- `scripts/slang/build_screen_space_slang_mlx.sh`
- `scripts/slang/convert_slang_metal_to_mlx.py`
- `GaussianSplattingMlx/Slang/*_mlx.json`
- `GaussianSplattingMlx/Trainer/GaussianRenderer.swift`
