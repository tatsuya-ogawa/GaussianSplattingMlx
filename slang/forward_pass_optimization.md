# Forward Pass Optimization (Removing `.item()` eval and supporting MLX.compile)

## Overview

Within an `MLX.compile` trace, operations that force synchronization with the CPU (eval), such as `.item(Int.self)`, are not supported.
In the initial implementation, `conditionToIndices` (which internally calls `.item()`) was used within `projection_ndc` to filter visible Gaussians. This optimization completely removes that bottleneck, refactoring the entire Forward pass into an eval-free architecture capable of being compiled.

## Solution Approach: Integrating Visibility Logic into the Tile Pipeline

### Core Idea

By completely removing `conditionToIndices` and its dependent `argSort`, the architecture was changed to **pass all $N$ Gaussians directly into the Screen Space kernel as a single batch**.

For Gaussians that are determined to be invisible, their `radii` is forcefully set to 0. This leverages the existing behavior of the downstream tile-based rendering kernels (`count_tiles_per_gaussian` and `generate_keys`), which naturally skips over any Gaussian with a radius of 0.

### Architecture Comparison

#### Before Optimization
1. Call `conditionToIndices` within `projection_ndc`, which triggers `.item(Int.self)`, retrieving $M$ visible indices.
2. Use the $M$ indices to gather parameters individually from all arrays.
3. Call `argSort(depths)` to sort the $M$ indices by depth.
4. Execute the Screen Space kernel only on the $M$ sorted Gaussians.

#### After Optimization (Current)
1. `projection_ndc` returns `visibleMask` (a lazy boolean mask) instead of an array of indices.
2. The Screen Space kernel is executed directly on **all $N$ Gaussians** without any gathering or pre-sorting.
3. Within the `render()` process, the resulting `visibleMask` is multiplied to forcefully set the calculated `radii` of invisible Gaussians to 0.
4. Gaussians with `radii = 0` are treated as "0 rendering tiles" within the subsequent `buildGlobalTileSliceInfo` pipeline, naturally excluding them from processing.
5. Depth sorting is no longer performed by a Swift-level `argSort`. Instead, it is handled inherently by the `radixSort` (the output of the `generateKeys` kernel) just before the tile processing, making the pre-sort entirely unnecessary.

## Implementation Details and Considerations

### 1. Preventing NaN Errors
If invisible Gaussians positioned behind the camera (`z < 0.2`) are processed directly by the Screen Space kernel, operations like division by zero can produce `NaN`. To prevent this, a safeguard is included in the Swift side of `projection_ndc` to clamp `p_view.z` to a minimum of `max(0.2, z)`.
The calculated results (such as 2D coordinates) for these clamped Gaussians will be meaningless, but because they are later discarded via `radii = 0`, they do not affect the final rendered result.

### 2. Deprecation of the `inputIsDepthSorted` Flag
Since the pre-`argSort` step was eliminated, there is no longer a guarantee that the input data to the rendering function is depth-sorted. As a result, flag conditions like `inputIsDepthSorted` have been removed. The tile information building pipeline is now triggered consistently using only the `globalTileSliceKernelsAvailable` condition.

### 3. Remaining `.item()` Calls
There are a few `.item(Int.self)` calls left inside `buildGlobalTileSliceInfo` used to calculate the total number of tile pairs. However, these are strictly placed within `MLX.stopGradient(...)` blocks (areas excluded from differentiation). Therefore, they do not interfere with `MLX.compile` (the loss calculation/backward pass), ensuring the maximum benefit from compilation during the training loop.

## Trade-offs and Achievements

| Metric | Before Optimization | After Optimization |
|---|---|---|
| Screen Space Target Count | $M$ instances (Visible only) | $N$ instances (All Gaussians) |
| CPU Syncs (Forward pass) | 1 sync via `conditionToIndices` | **Zero** |
| `argSort` Calls | Pre-sort executed on $M$ elements | **Unnecessary** (Handled by Tile Kernel) |
| Memory `gather` Operations | Heavy op on multiple arrays × 2 | **Zero** |
| **MLX.compile Compatibility** | Impossible (blocked by `.item()`) | **Supported (Accelerated Training)** |

Because the Screen Space kernel now evaluates invisible Gaussians as well, the compute load inside that specific kernel increased by about 10–20%.
However, the complete elimination of heavy memory gather access and CPU-side sorting, paired crucially with the **enabling of Forward Pass compilation**, has resulted in a dramatic overall improvement to training performance.
