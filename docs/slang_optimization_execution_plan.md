# Slang Optimization Execution Plan

## Goal
Improve end-to-end training/render performance in the Slang path by reducing memory traffic and high-overhead preprocessing while preserving numerical behavior.

## Scope
- Main target: `GaussianSplattingMlx/Trainer/GaussianRenderer.swift`
- Main kernels: `slang/gaussian_tile_global_kernels.slang`, `slang/gaussian_tile_kernels.slang`, `slang/gaussian_projection_kernels.slang`
- Reference: `google/slang-gaussian-rasterization`

## Baseline First
1. Add per-stage timing logs (or profiler scopes) for:
   - tile key generation and sort
   - tile range/count build
   - global forward composite
   - global backward composite
2. Record baseline with a fixed scene and fixed resolution.
3. Keep one comparison artifact (CSV or markdown table) before each phase.

## Phase 1 (Highest Impact): Replace Bitwise Radix Loop
### Problem
Current sort path performs many passes from Swift (`extract bit -> cumsum -> scatter`) and likely dominates preprocessing cost.

### Action
1. Remove iterative bit-loop sort in `radixBitPassForTileKeys`/`radixSortTileKeys`.
2. Replace with one GPU-side key-value sort path.
3. Keep key format equivalent (`tileID` high bits + depth bits low bits).

### Success Criteria
- Fewer kernel launches in sort stage.
- Tile sorting time reduced significantly at medium/high Gaussian counts.

## Phase 2: Remove Dense `packedTileIndices` Expansion
### Problem
Current pipeline builds `numTiles * maxTilePairs` dense matrix, which increases memory footprint and bandwidth.

### Action
1. Keep `sortedGaussIdx` and `tileRanges` as primary representation.
2. Update global forward/backward kernels to consume range slices directly.
3. Delete `build_packed_tile_indices` stage after migration.

### Success Criteria
- Lower temporary memory usage.
- One less preprocessing kernel stage.
- Equal rendering output.

## Phase 3: Tile-Block Forward Kernel (Shared Memory Reuse)
### Problem
`gaussian_tile_global_forward` currently iterates per pixel with repeated global loads.

### Action
1. Move forward composite to tile-block style dispatch (`block ~= tileW x tileH`).
2. Load Gaussian chunks into `groupshared` memory once per chunk.
3. Reuse chunk data across all pixels in the tile.

### Success Criteria
- Reduced global memory transactions.
- Better forward kernel time at larger tile counts.

## Phase 4: Specialize SH Color Evaluation
### Problem
Current fused projection kernel uses dynamic loops and a 25-element basis array, which can increase register pressure.

### Action
1. Add degree-specialized code paths (at least for commonly used active SH degrees).
2. Keep output compatibility with current SH math.

### Success Criteria
- Lower register pressure/occupancy risk.
- Faster projection+screen fused forward at same output quality.

## Phase 5: Fallback Path Cost Control
### Problem
Fallback tile path still does per-tile CPU-side orchestration and repeated slicing/sorting.

### Action
1. Ensure global fused path is selected whenever kernel availability allows.
2. Minimize fallback work for debug-only path.
3. Keep fallback behavior but reduce its accidental runtime usage.

### Success Criteria
- Stable preference for global fused path.
- Reduced host-side overhead in production runs.

## Early-Termination Audit (Explicit)
Track and preserve existing early-out behavior during refactor:
1. Alpha transmittance cutoff (`T < 1e-4`) in forward composite.
2. Contributor count carry-over (`lastContrib`) into backward.
3. Radius-based culling (`radii <= 0`) in tile touch/key generation.
4. Optional tile skip in fallback path when masked count is tiny.

## Validation Checklist
1. Numeric checks:
   - Render color/depth/alpha diffs against baseline.
   - Gradient sanity checks for packed gaussian parameters.
2. Performance checks:
   - Stage-wise latency before/after each phase.
   - Total iteration time (forward + backward).
3. Stability checks:
   - No shader compile regressions.
   - No out-of-bounds or NaN growth.

## Execution Order
1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5

## Risks
- Sort-path replacement may require integration work beyond pure Slang.
- Range-based indexing changes both forward and backward kernels simultaneously.
- Shared-memory rewrite can change numerical order and tiny floating-point differences.

## Deliverable per Phase
- Code changes
- Build confirmation
- Short benchmark table
- One-paragraph regression summary
