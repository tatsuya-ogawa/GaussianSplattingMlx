# Backward Kernel Optimization and Slang Porting

## Overview

A fundamental architectural overhaul was implemented for `bwd.globalTileComposite` (Backward kernel), which was the primary performance bottleneck during Gaussian Splatting training.
In its initial implementation, the Backward pass consumption took ~257ms per iteration, accounting for 80% of total training time. Through a three-stage optimization strategy (Tile dispatch, SIMD gradient aggregation, and 1-pass Reverse backward), the processing time was drastically reduced to **~19ms** (approximately a 13.4x speedup). As a result, the Backward pass is no longer a bottleneck.

## Evolution of Optimization

The optimization was implemented progressively through the following three approaches:

### 1. Tile Dispatch and Shared Memory (Approach A)

In the initial implementation, dispatching was extremely fine-grained, with 1 thread = 1 pixel. Adjacent pixels within the same tile fetched the exact same Gaussian data individually straight from global memory.

**Changes:**
- Migrated to a 2D tile-based dispatch (`grid: (pixelsPerTilePadded, numTiles, 1)`).
- Introduced a mechanism where chunks of up to 256 threads (handling pixels within a tile) cooperate to perform a **collaborative batch load into Shared Memory (Threadgroup Memory)** for all 11 fields (mean, color, opacity, etc.) of a Gaussian.
- Because threads reference data through shared memory, global memory bandwidth usage was significantly reduced.

### 2. SIMD Group Gradient Aggregation (Approach A+)

A structural bottleneck in the Backward kernel was severe condition (contention) on `InterlockedAdd` (Atomic operations) caused by multiple pixels simultaneously writing gradients back to the same Gaussian. A single Gaussian could incur up to 12 atomic accesses per pixel.

**Changes:**
- Implemented the `simd_sum()` (or `WaveActiveSum()` in Slang) function to locally aggregate computed gradients from each thread within a SIMD group (32 threads).
- After aggregation, a single representative thread (`simd_lane_id == 0`) performs the global memory Atomic write.
- **Results**: The theoretical number of Atomic writes dropped by a factor of 32. Execution time improved from ~257ms to **~51ms** (a 5x speedup).

### 3. 1-pass Reverse Backward (Approach C)

While the Forward pass requires a single pass, the previous Backward pass operated on a **2-pass scanning** architecture: Pass 1 to calculate the `weightedSum`, and Pass 2 for computing and writing back the gradients.

**Changes:**
- Similar to the official 3DGS CUDA implementation, the scanning direction was changed to a **Reverse Order** (from back to front relative to the camera within a tile), inherently compressing the work into a 1-pass execution.
- During the Forward pass, the number of Gaussians successfully processed by each pixel is recorded (`outLastContrib`) and fed into the Backward kernel. This makes it possible to simultaneously compute the suffix sum and the gradients by dynamically reverse-restoring the Transmittance via division.
- As for numeric stability concerns, the algorithm maintains a safety margin since the alpha $\alpha$ value is clamped to a maximum of 0.99, mitigating precision-loss risks on division.
- **Results**: Heavy operations such as memory access and `exp()` calculations were effectively halved, leading to an execution time of **~19ms**. This performance actually surpasses the Forward pass execution time (~21ms).

## Re-porting from Hand-written Metal to Slang

Initially, to prioritize rapid benchmark testing, the A+ optimizations were embedded manually into the JSON file (`gaussian_tile_global_backward_mlx.json`) via block inline Custom Metal code. To make the architecture maintainable, these changes were later formally ported directly back into the Slang source code (`slang/gaussian_tile_global_kernels.slang`), unifying compiler outputs.

### Syntax Translation Mapping

Advanced GPU control functions executed via the Slang compiler (such as Shared Memory, Barrier syncing, and SIMD aggregation) successfully transpiled flawlessly into the following Metal syntaxes, rendering identical performance to raw hand-written Metal.

| Feature Purpose | Slang Syntax | Transpiled Metal Output |
|---|---|---|
| Memory Array Allocation (Shared) | `groupshared float arr[256]` | `threadgroup array<float, 256>` |
| Local Thread Synchronization | `AllMemoryBarrierWithGroupSync()` | `threadgroup_barrier(...)` |
| SIMD Group Accumulation | `WaveActiveSum(val)` | `simd_sum(val)` |
| SIMD Group Lane ID | `WaveGetLaneIndex()` | `thread_index_in_simdgroup` |
| Local ID in Thread Group | `SV_GroupThreadID` | `thread_position_in_threadgroup` |

## Addendum: Applying Tiled Shared Memory to the Forward Kernel

The Shared Memory mechanism (Approach A), which yielded immense results in the Backward kernel, was also actively tested on the Forward kernel. In practical tests, the **performance impact was non-existent (sometimes regressing slightly)**.
The reason is that Forward processing efficiency is bound tightly by math operations (primarily `exp()`) rather than memory bandwidth, and the hardware L2 Cache natively within Apple Silicon inherently caches shared Gaussian data among tile threadgroups flawlessly without software-level explicit threadgroup boundaries.
Since the Forward pass also never encounters atomic contention (which the Backward pass had to solve), the Forward kernel remains on a 1D Pixel-dispatch model processing bare global memory accesses. This setup is effectively evaluated as the ceiling speed limit for its architectural design.

## Current Policy: AD-Centric Fused Projection+Screen Backward

For the `projection+screen` fused path, the codebase now follows an **AD-centric policy**:

1. Move mathematical logic into large `[Differentiable]` functions under shared Slang modules.
2. Keep kernel entrypoints thin (buffer I/O + packing/unpacking + a small number of AD calls).
3. Prefer a **single `bwd_diff(...)` call** for the fused math block when practical.

### Why an explicit backward kernel still exists

Even with AD, this repository's toolchain is `slangc -> metal -> MLX JSON`, and runtime dispatch expects named Metal entrypoints.
In this path, `-entry fwd.bwd` is not available as a direct dispatch target, so a dedicated backward entrypoint is still required.

### Concrete implementation in this repo

- Shared fused differentiable function:
  - `evaluateProjectionScreenGradOutputs(...)` in `slang/gaussian_projection_screen_shared.slang`
- Forward kernel:
  - `gaussian_projection_screen_fused_forward` calls the same fused function for `means2d/color/cov2d/conic`.
- Backward kernel:
  - `gaussian_projection_screen_fused_backward` acts as a thin wrapper and calls `bwd_diff(evaluateProjectionScreenGradOutputs)(...)`.

### Guideline for future fused kernels

- Do not hand-write long symbolic gradient algebra inside kernels unless required by control-flow constraints.
- Keep differentiable math in reusable shared functions first, then build minimal forward/backward kernel wrappers around them.
- Treat wrapper code as ABI glue for MLX, not as the place for core math logic.
