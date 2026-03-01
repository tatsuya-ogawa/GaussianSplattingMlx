# Slang Forward/Backward Optimization and Design Principles

## Scope

This document is the single source of truth for forward/backward design in this repository's Slang path.

- Projection + Screen-space fused path
- Tile-global slicing path
- Tile-global compositing forward/backward path
- AD usage policy for maintainability and performance

The runtime pipeline is:

`slangc -> metal -> MLX JSON -> Swift CustomFunction (Forward + VJP)`

## Core Philosophy

The codebase follows a hybrid strategy:

1. Put math in reusable `[Differentiable]` Slang functions.
2. Keep kernel entry points as thin ABI wrappers.
3. Hand-write only GPU control flow that AD cannot replace efficiently (reverse traversal, shared-memory staging, wave reduction, atomic accumulation).

This gives both:

- maintainability (single-source math, less symbolic gradient duplication)
- performance (explicit scheduling/reduction where required)

## Forward Design (Current)

### Eval-free forward for compile compatibility

Forward execution avoids CPU sync points in differentiable paths.

- No visibility index gathering via `.item()` in the differentiable projection/screen route.
- All `N` Gaussians are processed in projection+screen.
- Invisible splats are suppressed by `radii = 0`, then naturally skipped by tile slicing kernels (`count_tiles_per_gaussian`, `generate_keys`).

### Tile slicing and sort strategy

- Tile key generation is done on GPU.
- Depth ordering is handled by fused radix sort kernel (`radix_sort_tile_keys_fused_forward`), not by Swift `argSort`.
- Remaining `.item()` usage (for shape/control values) is isolated under `MLX.stopGradient(...)`.

### Numerical safety

For behind-camera or unstable cases, the projection/screen path applies guards (for example minimum z handling) so invalid intermediates do not propagate into rendered output.

## Backward Design (Current)

### AD-centric where practical

For projection+screen fused backward, the policy is:

1. define fused math in shared `[Differentiable]` functions
2. call those from forward
3. call `bwd_diff(...)` from a thin backward wrapper

This avoids duplicating large symbolic math between forward and backward.

### Why explicit backward kernels still exist

With current toolchain/runtime constraints, dispatch targets are explicit named kernels from generated MLX JSON. So dedicated backward entry points are still required for VJP wiring.

## Template Architecture for New Differentiable Tile Renderers

To support "new forward variants with minimal new backward code," use this template.

### Layer 1: Differentiable sample/state math

Define small reusable AD functions:

- `evaluateSample(...) -> Sample`
- `updateState(prevState, sample) -> nextState`

Both should be `[Differentiable]`.

### Layer 2: Reversible state transition

Define:

- `undoState(currentState, sample) -> prevState`

This is not required to be differentiable. It is used to walk backward through the compositing chain.

### Layer 3: Hand-written parallel control

Keep only these parts manual:

- reverse traversal over tile-local sample list
- `groupshared` cooperative loads
- `WaveActiveSum` reductions
- `InterlockedAdd` writes

Do not keep long symbolic gradient algebra here.

### Canonical backward loop pattern

```slang
sample = evaluateSample(...)
prevState = undoState(currentState, sample)

// AD through state update
bwd_diff(updateState)(diffPair(prevState), diffPair(sample), cotCurrentState)
currentState = prevState
cotCurrentState = d_prevState

// AD through sample evaluation
bwd_diff(evaluateSample)(..., d_sample)

// Manual parallel reduction + atomic write
reduce_and_atomic_add(...)
```

## Practical Policy: What to Hand-write vs What to AD

Prefer AD for:

- per-sample scalar/vector math
- per-step compositing state update math

Hand-write for:

- tile scheduling and reverse iteration order
- shared-memory staging
- SIMD/wave aggregation
- atomic write-back topology

## Authoring Checklist (New Variant)

1. Define `Sample` and `PixelState` as `IDifferentiable` structs.
2. Implement `[Differentiable] evaluateSample(...)`.
3. Implement `[Differentiable] updateState(prev, sample)`.
4. Implement `undoState(current, sample)`.
5. Use `evaluateSample + updateState` in forward loop.
6. In backward, keep control/reduction manual and use:
   - `bwd_diff(updateState)`
   - `bwd_diff(evaluateSample)`
7. Cache only required forward tensors for VJP (for example output color/depth/alpha and contributor count).
8. Regenerate JSON via build scripts and verify `xcodebuild`.

## Performance Notes

Historically, the main bottleneck was tile-global backward atomic contention. Performance improved significantly through:

- tile-based dispatch + shared-memory staging
- wave-level reduction before atomics
- reverse one-pass backward traversal

The current template preserves those gains while making future renderer variants easier to implement.
