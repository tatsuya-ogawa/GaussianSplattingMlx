import MLX
import MLXFast
import simd

private func arange(_ count: Int) -> [Int32] {
    Array(0..<Int32(count))
}
func createMeshGrid(shape: [Int]) -> MLXArray {
    let x = MLXArray(arange(shape[1]))
    let y = MLXArray(arange(shape[0]))
    return MLX.stacked(MLX.meshGrid([x, y], indexing: .xy), axis: -1)
}

struct TILE_SIZE_H_W {
    let w: Int
    let h: Int
}

private struct GlobalTileSliceInfo {
    let packedTileIndices: MLXArray
    let tileCounts: MLXArray
    let maxTilePairs: Int
    let renderCounts: MLXArray
}

class GaussianRenderer {
    private enum ScreenSpaceCustomOutputIndex {
        static let means2d = 0
        static let cov2d = 1
        static let color = 2
        static let conic = 3
    }

    private enum ProjectionScreenFusedOutputIndex {
        static let means2d = 0
        static let depths = 1
        static let color = 2
        static let cov2d = 3
        static let conic = 4
        static let radii = 5
    }
    
    private enum PackedGaussianIndex {
        static let means2d = 0..<2
        static let conic = 2..<6
        static let color = 6..<9
        static let opacity = 9..<10
        static let depth = 10
    }

    private static let screenSpaceCustomMinPointCount = 3

    let debug: Bool
    let active_sh_degree: Int
    let W: Int
    let H: Int
    let pix_coord: MLXArray
    let whiteBackground: Bool
    let TILE_SIZE: TILE_SIZE_H_W
    let useScreenSpaceCustomOp: Bool
    private let gridW: Int
    private let gridH: Int

    /// Mutable profiler set by the trainer each iteration.
    /// VJP backward closures read this at execution time.
    var profiler: IntervalProfiler?

    private static func padFirstDimToAtLeast(_ value: MLXArray, count: Int) -> MLXArray {
        let currentCount = value.shape[0]
        guard currentCount < count else { return value }
        let padCount = count - currentCount
        var padShape = [padCount]
        padShape.append(contentsOf: value.shape.dropFirst())
        let padding = MLXArray.zeros(padShape, dtype: value.dtype)
        return MLX.concatenated([value, padding], axis: 0)
    }

    private static func sliceFirstDim(_ value: MLXArray, count: Int) -> MLXArray {
        guard count < value.shape[0] else { return value }
        return value[0..<count]
    }
    
    private func buildPackedGaussians(
        means2d: MLXArray,
        conic: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray
    ) -> MLXArray {
        let conicFlat = conic.reshaped([-1, 4])
        let opacityColumn = opacity.reshaped([-1, 1])
        let depthColumn = depths.reshaped([-1, 1])
        return MLX.concatenated(
            [means2d, conicFlat, color, opacityColumn, depthColumn],
            axis: 1
        )
    }

    private lazy var globalTileCompositeCustomFunction: (([MLXArray]) -> [MLXArray])? = {
        guard
            let forwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_tile_global_forward_mlx"),
            let backwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_tile_global_backward_mlx")
        else {
            Logger.shared.debug(
                "Slang global tile kernels are unavailable. Per-tile path is used.")
            return nil
        }

        let pixelCount = W * H
        let tileW = TILE_SIZE.w
        let tileH = TILE_SIZE.h
        let numTiles = gridW * gridH

        // Captured forward context for 1-pass reverse backward
        var savedOutAlpha: MLXArray?
        var savedLastContrib: MLXArray?

        let forward: ([MLXArray]) -> [MLXArray] = { inputs in
            let packedGaussians = inputs[0]
            let packedTileIndices = inputs[1]
            let tileCounts = inputs[2]
            let renderCounts = inputs[3]

            let allOutputs = forwardKernel(
                [packedGaussians, packedTileIndices, tileCounts, renderCounts],
                grid: (max(pixelCount, 1), 1, 1),
                threadGroup: (min(256, max(pixelCount, 1)), 1, 1),
                outputShapes: [
                    [pixelCount, 3], [pixelCount, 1], [pixelCount, 1], [pixelCount, 1],
                ],
                outputDTypes: [
                    packedGaussians.dtype, packedGaussians.dtype,
                    packedGaussians.dtype, .uint32,
                ]
            )
            savedOutAlpha = allOutputs[2]
            savedLastContrib = allOutputs[3]
            return [allOutputs[0], allOutputs[1], allOutputs[2]]
        }

        let renderer = self
        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
                let outAlpha = savedOutAlpha!
                let lastContrib = savedLastContrib!
                if let profiler = renderer.profiler {
                    return profiler.measure("bwd.globalTileComposite") {
                        let result = Self._globalTileCompositeVJP(
                            primals: primals, cotangents: cotangents,
                            outAlpha: outAlpha, lastContrib: lastContrib,
                            backwardKernel: backwardKernel,
                            pixelCount: pixelCount,
                            tileW: tileW,
                            tileH: tileH,
                            numTiles: numTiles
                        )
                        eval(result)
                        return result
                    }
                }
                return Self._globalTileCompositeVJP(
                    primals: primals, cotangents: cotangents,
                    outAlpha: outAlpha, lastContrib: lastContrib,
                    backwardKernel: backwardKernel,
                    pixelCount: pixelCount,
                    tileW: tileW,
                    tileH: tileH,
                    numTiles: numTiles
                )
            }
        }
    }()

    private static func _globalTileCompositeVJP(
        primals: [MLXArray], cotangents: [MLXArray],
        outAlpha: MLXArray, lastContrib: MLXArray,
        backwardKernel: MLXFast.MLXFastKernel,
        pixelCount: Int,
        tileW: Int,
        tileH: Int,
        numTiles: Int
    ) -> [MLXArray] {
        let packedGaussians = primals[0]
        let packedTileIndices = primals[1]
        let tileCounts = primals[2]
        let renderCounts = primals[3]
        let cotColor = cotangents[0]
        let cotDepth = cotangents[1]
        let cotAlpha = cotangents[2]

        let pixelsPerTile = tileW * tileH
        let pixelsPerTilePadded = ((pixelsPerTile + 255) / 256) * 256

        let gradPacked = backwardKernel(
            [
                packedGaussians, packedTileIndices, tileCounts, cotColor, cotDepth, cotAlpha,
                renderCounts, outAlpha, lastContrib,
            ],
            grid: (max(pixelsPerTilePadded, 1), max(numTiles, 1), 1),
            threadGroup: (min(256, max(pixelsPerTilePadded, 1)), 1, 1),
            outputShapes: [packedGaussians.shape],
            outputDTypes: [packedGaussians.dtype],
            initValue: 0.0
        )[0]

        return [
            gradPacked,
            MLXArray.zeros(packedTileIndices.shape, dtype: packedTileIndices.dtype),
            MLXArray.zeros(tileCounts.shape, dtype: tileCounts.dtype),
            MLXArray.zeros(renderCounts.shape, dtype: renderCounts.dtype),
        ]
    }

    private func renderGlobalTileCompositeCustomOp(
        packedGaussians: MLXArray,
        globalTileSliceInfo: GlobalTileSliceInfo
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard let globalTileCompositeCustomFunction else {
            return nil
        }
        let outputs = globalTileCompositeCustomFunction(
            [
                packedGaussians,
                globalTileSliceInfo.packedTileIndices,
                globalTileSliceInfo.tileCounts,
                globalTileSliceInfo.renderCounts,
            ]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    private static func bitWidthForExclusiveUpperBound(_ value: Int) -> Int {
        guard value > 1 else { return 1 }
        var v = value - 1
        var bits = 0
        while v > 0 {
            bits += 1
            v >>= 1
        }
        return bits
    }

    private static let fusedRadixSortThreadCount = 128

    private lazy var fusedRadixSortTileKeysKernel: MLXFast.MLXFastKernel = {
        if let slangKernel = try? SlangKernelSpecLoader.loadKernel(
            named: "radix_sort_tile_keys_fused_forward_mlx"
        ) {
            return slangKernel
        }
        Logger.shared.debug(
            "radix_sort_tile_keys_fused_forward_mlx unavailable. Falling back to inline Metal kernel.")
        return MLXFast.metalKernel(
            name: "radix_sort_tile_keys_fused_u32_v1",
            inputNames: ["keysHighIn", "keysLowIn", "valuesIn", "counts"],
            outputNames: [
                "sortedKeysHigh", "sortedKeysLow", "sortedValues",
                "scratchKeysHigh", "scratchKeysLow", "scratchValues",
            ],
            source: """
                const uint THREADS = 128u;
                const uint RADIX_BITS = 4u;
                const uint RADIX = 16u;
                const uint LOW_PASSES = 8u; // 32 bits / 4 bits.

                uint lane = thread_position_in_threadgroup.x;
                if (lane >= THREADS) {
                    return;
                }

                uint n = counts[0];
                if (n == 0) {
                    return;
                }

                uint chunk = (n + THREADS - 1u) / THREADS;
                uint start = lane * chunk;
                uint end = min(start + chunk, n);

                uint highBits = counts[1];
                uint highPasses = (highBits + RADIX_BITS - 1u) / RADIX_BITS;
                if (highPasses < 1u) {
                    highPasses = 1u;
                }
                uint totalPasses = LOW_PASSES + highPasses;

                threadgroup uint localHist[2048];
                threadgroup uint threadBase[2048];

                bool writeToScratch = true;

                for (uint pass = 0u; pass < totalPasses; ++pass) {
                    uint localBase = lane * RADIX;
                    for (uint d = 0u; d < RADIX; ++d) {
                        localHist[localBase + d] = 0u;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint i = start; i < end; ++i) {
                        uint keyHigh;
                        uint keyLow;
                        if (pass == 0u) {
                            keyHigh = keysHighIn[i];
                            keyLow = keysLowIn[i];
                        } else if (writeToScratch) {
                            keyHigh = sortedKeysHigh[i];
                            keyLow = sortedKeysLow[i];
                        } else {
                            keyHigh = scratchKeysHigh[i];
                            keyLow = scratchKeysLow[i];
                        }

                        uint digit;
                        if (pass < LOW_PASSES) {
                            digit = (keyLow >> (pass * RADIX_BITS)) & (RADIX - 1u);
                        } else {
                            uint hiPass = pass - LOW_PASSES;
                            digit = (keyHigh >> (hiPass * RADIX_BITS)) & (RADIX - 1u);
                        }
                        localHist[localBase + digit] += 1u;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (lane == 0u) {
                        uint running = 0u;
                        for (uint d = 0u; d < RADIX; ++d) {
                            uint offset = running;
                            for (uint t = 0u; t < THREADS; ++t) {
                                uint idx = t * RADIX + d;
                                threadBase[idx] = offset;
                                offset += localHist[idx];
                            }
                            running = offset;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    uint localOffsets[16];
                    for (uint d = 0u; d < RADIX; ++d) {
                        localOffsets[d] = 0u;
                    }

                    for (uint i = start; i < end; ++i) {
                        uint keyHigh;
                        uint keyLow;
                        uint value;
                        if (pass == 0u) {
                            keyHigh = keysHighIn[i];
                            keyLow = keysLowIn[i];
                            value = valuesIn[i];
                        } else if (writeToScratch) {
                            keyHigh = sortedKeysHigh[i];
                            keyLow = sortedKeysLow[i];
                            value = sortedValues[i];
                        } else {
                            keyHigh = scratchKeysHigh[i];
                            keyLow = scratchKeysLow[i];
                            value = scratchValues[i];
                        }

                        uint digit;
                        if (pass < LOW_PASSES) {
                            digit = (keyLow >> (pass * RADIX_BITS)) & (RADIX - 1u);
                        } else {
                            uint hiPass = pass - LOW_PASSES;
                            digit = (keyHigh >> (hiPass * RADIX_BITS)) & (RADIX - 1u);
                        }

                        uint dst = threadBase[localBase + digit] + localOffsets[digit];
                        localOffsets[digit] += 1u;

                        if (writeToScratch) {
                            scratchKeysHigh[dst] = keyHigh;
                            scratchKeysLow[dst] = keyLow;
                            scratchValues[dst] = value;
                        } else {
                            sortedKeysHigh[dst] = keyHigh;
                            sortedKeysLow[dst] = keyLow;
                            sortedValues[dst] = value;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    writeToScratch = !writeToScratch;
                }

                // If pass count is odd, final result lives in scratch; copy it out.
                if (!writeToScratch) {
                    for (uint i = start; i < end; ++i) {
                        sortedKeysHigh[i] = scratchKeysHigh[i];
                        sortedKeysLow[i] = scratchKeysLow[i];
                        sortedValues[i] = scratchValues[i];
                    }
                }
                """
        )
    }()

    private func radixSortTileKeys(
        keysHigh: MLXArray,
        keysLow: MLXArray,
        values: MLXArray,
        tileBitCount: Int
    ) -> (sortedKeysHigh: MLXArray, sortedKeysLow: MLXArray, sortedValues: MLXArray) {
        let n = keysHigh.shape[0]
        if n <= 1 {
            return (keysHigh, keysLow, values)
        }

        let outputs = fusedRadixSortTileKeysKernel(
            [
                keysHigh,
                keysLow,
                values,
                MLXArray([
                    UInt32(n),
                    UInt32(Swift.max(tileBitCount, 1)),
                ]),
            ],
            grid: (Self.fusedRadixSortThreadCount, 1, 1),
            threadGroup: (Self.fusedRadixSortThreadCount, 1, 1),
            outputShapes: [[n], [n], [n], [n], [n], [n]],
            outputDTypes: [
                keysHigh.dtype, keysLow.dtype, values.dtype,
                keysHigh.dtype, keysLow.dtype, values.dtype,
            ]
        )
        return (outputs[0], outputs[1], outputs[2])
    }

    // MARK: - Global Tile Slice Kernels

    private lazy var countTilesPerGaussianKernel: MLXFast.MLXFastKernel? = {
        try? SlangKernelSpecLoader.loadKernel(named: "count_tiles_per_gaussian_mlx")
    }()

    private lazy var generateKeysKernel: MLXFast.MLXFastKernel? = {
        try? SlangKernelSpecLoader.loadKernel(named: "generate_keys_mlx")
    }()

    private lazy var computeTileRangesKernel: MLXFast.MLXFastKernel? = {
        try? SlangKernelSpecLoader.loadKernel(named: "compute_tile_ranges_mlx")
    }()

    private lazy var computeTileCountsFromRangesKernel: MLXFast.MLXFastKernel? = {
        try? SlangKernelSpecLoader.loadKernel(named: "compute_tile_counts_from_ranges_mlx")
    }()

    private lazy var buildPackedTileIndicesKernel: MLXFast.MLXFastKernel? = {
        try? SlangKernelSpecLoader.loadKernel(named: "build_packed_tile_indices_mlx")
    }()

    private var globalTileSliceKernelsAvailable: Bool {
        countTilesPerGaussianKernel != nil
            && generateKeysKernel != nil
            && computeTileRangesKernel != nil
            && computeTileCountsFromRangesKernel != nil
            && buildPackedTileIndicesKernel != nil
    }

    private func buildGlobalTileSliceInfo(
        rect: (MLXArray, MLXArray),
        radii: MLXArray,
        depths: MLXArray
    ) -> GlobalTileSliceInfo? {
        guard
            let countTilesPerGaussianKernel,
            let generateKeysKernel,
            let computeTileRangesKernel,
            let computeTileCountsFromRangesKernel,
            let buildPackedTileIndicesKernel
        else {
            return nil
        }

        let gaussianCount = radii.shape[0]
        let numTiles = gridW * gridH
        let pixelCount = W * H
        func makeRenderCounts(maxTilePairs: Int) -> MLXArray {
            MLXArray(
                [
                    UInt32(pixelCount),
                    UInt32(maxTilePairs),
                    UInt32(gridW),
                    UInt32(TILE_SIZE.w),
                    UInt32(TILE_SIZE.h),
                    UInt32(W),
                    UInt32(H),
                    UInt32(whiteBackground ? 1 : 0),
                ]
            )
        }
        if gaussianCount == 0 {
            return GlobalTileSliceInfo(
                packedTileIndices: MLXArray.zeros([0], dtype: .int32),
                tileCounts: MLXArray.zeros([numTiles], dtype: .uint32),
                maxTilePairs: 0,
                renderCounts: makeRenderCounts(maxTilePairs: 0)
            )
        }

        let ctCounts = MLXArray(
            [
                UInt32(gaussianCount),
                UInt32(TILE_SIZE.w),
                UInt32(TILE_SIZE.h),
                UInt32(W),
                UInt32(H),
            ]
        )
        let rectMinFlat = MLX.stopGradient(rect.0).reshaped([-1]).asType(.float32)
        let rectMaxFlat = MLX.stopGradient(rect.1).reshaped([-1]).asType(.float32)
        let radiiFloat = MLX.stopGradient(radii).asType(.float32)
        let depthsFloat = MLX.stopGradient(depths).asType(.float32)

        let tilesTouched = MLX.stopGradient(
            countTilesPerGaussianKernel(
                [rectMinFlat, rectMaxFlat, radiiFloat, ctCounts],
                grid: (max(gaussianCount, 1), 1, 1),
                threadGroup: (min(256, max(gaussianCount, 1)), 1, 1),
                outputShapes: [[gaussianCount]],
                outputDTypes: [.uint32]
            )[0]
        )

        let cumsum = MLX.stopGradient(tilesTouched.cumsum(axis: 0))
        let totalPairs = MLX.stopGradient(cumsum[gaussianCount - 1]).item(Int.self)
        if totalPairs <= 0 {
            return GlobalTileSliceInfo(
                packedTileIndices: MLXArray.zeros([0], dtype: .int32),
                tileCounts: MLXArray.zeros([numTiles], dtype: .uint32),
                maxTilePairs: 0,
                renderCounts: makeRenderCounts(maxTilePairs: 0)
            )
        }

        let offsets = MLX.stopGradient((cumsum - tilesTouched).asType(.uint32))
        let gkCounts = MLXArray(
            [
                UInt32(gaussianCount),
                UInt32(TILE_SIZE.w),
                UInt32(TILE_SIZE.h),
                UInt32(W),
                UInt32(H),
            ]
        )
        let generated = generateKeysKernel(
            [depthsFloat, rectMinFlat, rectMaxFlat, radiiFloat, offsets, gkCounts],
            grid: (max(gaussianCount, 1), 1, 1),
            threadGroup: (min(256, max(gaussianCount, 1)), 1, 1),
            outputShapes: [[totalPairs], [totalPairs], [totalPairs]],
            outputDTypes: [.uint32, .uint32, .uint32]
        )

        let keysHigh = MLX.stopGradient(generated[0])
        let keysLow = MLX.stopGradient(generated[1])
        let gaussIdx = MLX.stopGradient(generated[2])
        let tileBitCount = Self.bitWidthForExclusiveUpperBound(numTiles)
        let radixSorted = radixSortTileKeys(
            keysHigh: keysHigh,
            keysLow: keysLow,
            values: gaussIdx,
            tileBitCount: tileBitCount
        )
        let sortedKeysHigh = MLX.stopGradient(radixSorted.sortedKeysHigh)
        let sortedGaussIdx = MLX.stopGradient(radixSorted.sortedValues)

        let trCounts = MLXArray([UInt32(totalPairs), UInt32(numTiles)])
        let tileRanges = MLX.stopGradient(
            computeTileRangesKernel(
                [sortedKeysHigh, trCounts],
                grid: (max(totalPairs, 1), 1, 1),
                threadGroup: (min(256, max(totalPairs, 1)), 1, 1),
                outputShapes: [[numTiles * 2]],
                outputDTypes: [.uint32],
                initValue: 0
            )[0]
        )

        let tileCountArgs = MLXArray([UInt32(numTiles)])
        let tileCountsArray = MLX.stopGradient(
            computeTileCountsFromRangesKernel(
                [tileRanges, tileCountArgs],
                grid: (max(numTiles, 1), 1, 1),
                threadGroup: (min(256, max(numTiles, 1)), 1, 1),
                outputShapes: [[numTiles]],
                outputDTypes: [.uint32]
            )[0]
        )
        let maxTilePairs = MLX.stopGradient(MLX.max(tileCountsArray)).item(Int.self)
        if maxTilePairs <= 0 {
            return GlobalTileSliceInfo(
                packedTileIndices: MLXArray.zeros([0], dtype: .int32),
                tileCounts: tileCountsArray,
                maxTilePairs: 0,
                renderCounts: makeRenderCounts(maxTilePairs: 0)
            )
        }

        let packArgs = MLXArray([UInt32(numTiles), UInt32(maxTilePairs)])
        let packedTileIndices = MLX.stopGradient(
            buildPackedTileIndicesKernel(
                [sortedGaussIdx, tileRanges, packArgs],
                grid: (max(numTiles * maxTilePairs, 1), 1, 1),
                threadGroup: (min(256, max(numTiles * maxTilePairs, 1)), 1, 1),
                outputShapes: [[numTiles * maxTilePairs]],
                outputDTypes: [.int32],
                initValue: 0
            )[0]
        )

        return GlobalTileSliceInfo(
            packedTileIndices: packedTileIndices,
            tileCounts: tileCountsArray,
            maxTilePairs: maxTilePairs,
            renderCounts: makeRenderCounts(maxTilePairs: maxTilePairs)
        )
    }

    private static func _projectionNdcVJP(
        primals: [MLXArray], cotangents: [MLXArray],
        backwardKernel: MLXFast.MLXFastKernel
    ) -> [MLXArray] {
        let points = primals[0]
        let viewMatrix = primals[1]
        let projMatrix = primals[2]
        let cotMeanNdc = cotangents[0]
        let cotPView = cotangents[1]
        // cotangents[2] = visibleMask cotangent (no gradient, ignored)

        let activeCount = points.shape[0]
        let paddedCount = Swift.max(activeCount, 1)
        let pointsPadded = Self.padFirstDimToAtLeast(points, count: paddedCount)
        let cotMeanNdcPadded = Self.padFirstDimToAtLeast(cotMeanNdc, count: paddedCount)
        let cotPViewPadded = Self.padFirstDimToAtLeast(cotPView, count: paddedCount)
        let counts = MLXArray([UInt32(paddedCount)])

        let grads = backwardKernel(
            [pointsPadded, viewMatrix, projMatrix, counts, cotMeanNdcPadded, cotPViewPadded],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [pointsPadded.shape],
            outputDTypes: [points.dtype]
        )

        return [
            Self.sliceFirstDim(grads[0], count: activeCount),  // grad_points
            MLXArray.zeros(like: viewMatrix),                   // no grad for viewMatrix
            MLXArray.zeros(like: projMatrix),                   // no grad for projMatrix
        ]
    }

    // MARK: - Get radius + mask kernel (forward-only, no backward needed)

    private lazy var getRadiusMaskKernel: MLXFast.MLXFastKernel? = {
        guard useScreenSpaceCustomOp else { return nil }
        guard let kernel = try? SlangKernelSpecLoader.loadKernel(
            named: "get_radius_mask_forward_mlx")
        else {
            Logger.shared.debug("get_radius_mask kernel unavailable. Fallback path is used.")
            return nil
        }
        return kernel
    }()

    private lazy var covariance3DCustomFunction: (([MLXArray]) -> [MLXArray])? = {
        guard useScreenSpaceCustomOp else {
            return nil
        }
        guard
            let forwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov3d_forward_mlx"),
            let backwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov3d_backward_mlx")
        else {
            Logger.shared.debug("Slang cov3d kernels are unavailable. Fallback path is used.")
            return nil
        }

        let forward: ([MLXArray]) -> [MLXArray] = { inputs in
            let scales = inputs[0]
            let rotations = inputs[1]
            let activeCount = scales.shape[0]
            let paddedCount = Swift.max(activeCount, 1)
            let scalesPadded = Self.padFirstDimToAtLeast(scales, count: paddedCount)
            let rotationsPadded = Self.padFirstDimToAtLeast(rotations, count: paddedCount)
            let pointCounts = MLXArray([UInt32(paddedCount)])

            let cov3dPadded = forwardKernel(
                [scalesPadded, rotationsPadded, pointCounts],
                grid: (max(paddedCount, 1), 1, 1),
                threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
                outputShapes: [[paddedCount, 3, 3]],
                outputDTypes: [scales.dtype]
            )[0]
            return [Self.sliceFirstDim(cov3dPadded, count: activeCount)]
        }

        let renderer = self
        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
                if let profiler = renderer.profiler {
                    profiler.measure("bwd.cov3d.inputs") {
                        eval(primals)
                        eval(cotangents)
                    }
                    return profiler.measure("bwd.cov3d") {
                        let result = Self._cov3dVJP(primals: primals, cotangents: cotangents, backwardKernel: backwardKernel)
                        eval(result)
                        return result
                    }
                }
                return Self._cov3dVJP(primals: primals, cotangents: cotangents, backwardKernel: backwardKernel)
            }
        }
    }()

    private static func _cov3dVJP(
        primals: [MLXArray], cotangents: [MLXArray],
        backwardKernel: MLXFast.MLXFastKernel
    ) -> [MLXArray] {
        let scales = primals[0]
        let rotations = primals[1]
        let cotCov3d = cotangents[0]

        let activeCount = scales.shape[0]
        let paddedCount = Swift.max(activeCount, 1)
        let scalesPadded = Self.padFirstDimToAtLeast(scales, count: paddedCount)
        let rotationsPadded = Self.padFirstDimToAtLeast(rotations, count: paddedCount)
        let cotCov3dPadded = Self.padFirstDimToAtLeast(cotCov3d, count: paddedCount)
        let pointCounts = MLXArray([UInt32(paddedCount)])

        let grads = backwardKernel(
            [scalesPadded, rotationsPadded, cotCov3dPadded, pointCounts],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [scalesPadded.shape, rotationsPadded.shape],
            outputDTypes: [scales.dtype, rotations.dtype]
        )

        return [
            Self.sliceFirstDim(grads[0], count: activeCount),
            Self.sliceFirstDim(grads[1], count: activeCount),
        ]
    }

    private func buildCovariance3d(scales: MLXArray, rotations: MLXArray) -> MLXArray {
        if let covariance3DCustomFunction {
            return covariance3DCustomFunction([scales, rotations])[0]
        }
        return build_covariance_3d(s: scales, r: rotations)
    }

    private static func _fusedScreenSpaceVJP(
        primals: [MLXArray], cotangents: [MLXArray],
        cov3dForwardKernel: MLXFast.MLXFastKernel,
        cov3dBackwardKernel: MLXFast.MLXFastKernel,
        colorBackwardKernel: MLXFast.MLXFastKernel,
        covForwardKernel: MLXFast.MLXFastKernel,
        covBackwardKernel: MLXFast.MLXFastKernel,
        inverseBackwardKernel: MLXFast.MLXFastKernel,
        activeShDegree: Int
    ) -> [MLXArray] {
        let meanNdc = primals[0]
        let scales = primals[1]
        let rotations = primals[2]
        let means3d = primals[3]
        let shs = primals[4]
        let cameraCenter = primals[5]
        let viewMatrix = primals[6]
        let fovX = primals[7]
        let fovY = primals[8]
        let focalX = primals[9]
        let focalY = primals[10]
        let imageWidth = primals[11]
        let imageHeight = primals[12]

        let cotMeans2d = cotangents[0]
        let cotCov2d = cotangents[1]
        let cotColor = cotangents[2]
        let cotConic = cotangents[3]

        let activeCount = meanNdc.shape[0]
        let paddedCount = Swift.max(activeCount, screenSpaceCustomMinPointCount)
        let scalesPadded = padFirstDimToAtLeast(scales, count: paddedCount)
        let rotationsPadded = padFirstDimToAtLeast(rotations, count: paddedCount)
        let pointCounts = MLXArray([UInt32(paddedCount)])

        let cov3dPadded = cov3dForwardKernel(
            [scalesPadded, rotationsPadded, pointCounts],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [[paddedCount, 3, 3]],
            outputDTypes: [scales.dtype]
        )[0]
        let cov3d = sliceFirstDim(cov3dPadded, count: activeCount)

        let screenGrads = _screenSpaceVJP(
            primals: [
                meanNdc, means3d, shs, cov3d,
                cameraCenter, viewMatrix, fovX, fovY, focalX, focalY, imageWidth, imageHeight,
            ],
            cotangents: [cotMeans2d, cotCov2d, cotColor, cotConic],
            colorBackwardKernel: colorBackwardKernel,
            covForwardKernel: covForwardKernel,
            covBackwardKernel: covBackwardKernel,
            inverseBackwardKernel: inverseBackwardKernel,
            activeShDegree: activeShDegree
        )

        let cov3dGrads = _cov3dVJP(
            primals: [scales, rotations],
            cotangents: [screenGrads[3]],
            backwardKernel: cov3dBackwardKernel
        )

        return [
            screenGrads[0],   // gradMeanNdc
            cov3dGrads[0],    // gradScales
            cov3dGrads[1],    // gradRotations
            screenGrads[1],   // gradMeans3d
            screenGrads[2],   // gradShs
            screenGrads[4],   // gradCameraCenter
            MLXArray.zeros(viewMatrix.shape, dtype: viewMatrix.dtype),
            MLXArray.zeros(fovX.shape, dtype: fovX.dtype),
            MLXArray.zeros(fovY.shape, dtype: fovY.dtype),
            MLXArray.zeros(focalX.shape, dtype: focalX.dtype),
            MLXArray.zeros(focalY.shape, dtype: focalY.dtype),
            MLXArray.zeros(imageWidth.shape, dtype: imageWidth.dtype),
            MLXArray.zeros(imageHeight.shape, dtype: imageHeight.dtype),
        ]
    }

    // MARK: - Projection + screen-space + radius fused forward

    private lazy var projectionScreenFusedCustomFunction: (([MLXArray]) -> [MLXArray])? = {
        guard useScreenSpaceCustomOp else {
            return nil
        }
        guard
            let fusedForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_projection_screen_fused_forward_mlx"),
            let projectionForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "projection_ndc_forward_mlx"),
            let projectionBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "projection_ndc_backward_mlx"),
            let cov3dForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov3d_forward_mlx"),
            let cov3dBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov3d_backward_mlx"),
            let colorBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_color_backward_mlx"),
            let covForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov2d_forward_mlx"),
            let covBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov2d_backward_mlx"),
            let inverseBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_inverse2d_backward_mlx")
        else {
            Logger.shared.debug(
                "projection+screen fused kernel unavailable. fallback path is used.")
            return nil
        }

        let activeShDegree = self.active_sh_degree
        let forward: ([MLXArray]) -> [MLXArray] = { inputs in
            let scales = inputs[0]
            let rotations = inputs[1]
            let means3d = inputs[2]
            let shs = inputs[3]
            let cameraCenter = inputs[4]
            let viewMatrix = inputs[5]
            let projMatrix = inputs[6]
            let fovX = inputs[7]
            let fovY = inputs[8]
            let focalX = inputs[9]
            let focalY = inputs[10]
            let imageWidth = inputs[11]
            let imageHeight = inputs[12]

            let fovXKernel = fovX.reshaped([1])
            let fovYKernel = fovY.reshaped([1])
            let focalXKernel = focalX.reshaped([1])
            let focalYKernel = focalY.reshaped([1])
            let imageWidthKernel = imageWidth.reshaped([1])
            let imageHeightKernel = imageHeight.reshaped([1])

            let activeCount = means3d.shape[0]
            let paddedCount = Swift.max(activeCount, Self.screenSpaceCustomMinPointCount)
            let scalesPadded = Self.padFirstDimToAtLeast(scales, count: paddedCount)
            let rotationsPadded = Self.padFirstDimToAtLeast(rotations, count: paddedCount)
            let means3dPadded = Self.padFirstDimToAtLeast(means3d, count: paddedCount)
            let shsPadded = Self.padFirstDimToAtLeast(shs, count: paddedCount)
            let counts = MLXArray(
                [UInt32(paddedCount), UInt32(activeShDegree), UInt32(shsPadded.shape[1])]
            )

            let outputs = fusedForwardKernel(
                [
                    scalesPadded, rotationsPadded, means3dPadded, shsPadded,
                    cameraCenter, viewMatrix, projMatrix,
                    fovXKernel, fovYKernel, focalXKernel, focalYKernel,
                    imageWidthKernel, imageHeightKernel, counts,
                ],
                grid: (max(paddedCount, 1), 1, 1),
                threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
                outputShapes: [
                    [paddedCount, 2], [paddedCount], [paddedCount, 3],
                    [paddedCount, 2, 2], [paddedCount, 2, 2], [paddedCount],
                ],
                outputDTypes: [
                    means3d.dtype, means3d.dtype, means3d.dtype,
                    means3d.dtype, means3d.dtype, means3d.dtype,
                ]
            )

            return [
                Self.sliceFirstDim(outputs[0], count: activeCount),  // means2d
                Self.sliceFirstDim(outputs[1], count: activeCount),  // depths
                Self.sliceFirstDim(outputs[2], count: activeCount),  // color
                Self.sliceFirstDim(outputs[3], count: activeCount),  // cov2d
                Self.sliceFirstDim(outputs[4], count: activeCount),  // conic
                Self.sliceFirstDim(outputs[5], count: activeCount),  // radii
            ]
        }

        let renderer = self
        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
                if let profiler = renderer.profiler {
                    profiler.measure("bwd.projectionScreenFused.inputs") {
                        eval(primals)
                        eval(cotangents)
                    }
                    return profiler.measure("bwd.projectionScreenFused") {
                        let result = Self._projectionScreenFusedVJP(
                            primals: primals,
                            cotangents: cotangents,
                            projectionForwardKernel: projectionForwardKernel,
                            projectionBackwardKernel: projectionBackwardKernel,
                            cov3dForwardKernel: cov3dForwardKernel,
                            cov3dBackwardKernel: cov3dBackwardKernel,
                            colorBackwardKernel: colorBackwardKernel,
                            covForwardKernel: covForwardKernel,
                            covBackwardKernel: covBackwardKernel,
                            inverseBackwardKernel: inverseBackwardKernel,
                            activeShDegree: activeShDegree
                        )
                        eval(result)
                        return result
                    }
                }
                return Self._projectionScreenFusedVJP(
                    primals: primals,
                    cotangents: cotangents,
                    projectionForwardKernel: projectionForwardKernel,
                    projectionBackwardKernel: projectionBackwardKernel,
                    cov3dForwardKernel: cov3dForwardKernel,
                    cov3dBackwardKernel: cov3dBackwardKernel,
                    colorBackwardKernel: colorBackwardKernel,
                    covForwardKernel: covForwardKernel,
                    covBackwardKernel: covBackwardKernel,
                    inverseBackwardKernel: inverseBackwardKernel,
                    activeShDegree: activeShDegree
                )
            }
        }
    }()

    private static func _projectionScreenFusedVJP(
        primals: [MLXArray], cotangents: [MLXArray],
        projectionForwardKernel: MLXFast.MLXFastKernel,
        projectionBackwardKernel: MLXFast.MLXFastKernel,
        cov3dForwardKernel: MLXFast.MLXFastKernel,
        cov3dBackwardKernel: MLXFast.MLXFastKernel,
        colorBackwardKernel: MLXFast.MLXFastKernel,
        covForwardKernel: MLXFast.MLXFastKernel,
        covBackwardKernel: MLXFast.MLXFastKernel,
        inverseBackwardKernel: MLXFast.MLXFastKernel,
        activeShDegree: Int
    ) -> [MLXArray] {
        let scales = primals[0]
        let rotations = primals[1]
        let means3d = primals[2]
        let shs = primals[3]
        let cameraCenter = primals[4]
        let viewMatrix = primals[5]
        let projMatrix = primals[6]
        let fovX = primals[7]
        let fovY = primals[8]
        let focalX = primals[9]
        let focalY = primals[10]
        let imageWidth = primals[11]
        let imageHeight = primals[12]

        let cotMeans2d = cotangents[0]
        let cotDepths = cotangents[1]
        let cotColor = cotangents[2]
        let cotCov2d = cotangents[3]
        let cotConic = cotangents[4]
        // cotangents[5] = radii cotangent (ignored; treated as stopGradient)

        let activeCount = means3d.shape[0]
        let paddedCount = Swift.max(activeCount, screenSpaceCustomMinPointCount)

        let means3dPadded = padFirstDimToAtLeast(means3d, count: paddedCount)
        let pointCounts = MLXArray([UInt32(paddedCount)])
        let projectionForward = projectionForwardKernel(
            [means3dPadded, viewMatrix, projMatrix, pointCounts],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [[paddedCount, 4], [paddedCount, 4], [paddedCount]],
            outputDTypes: [means3d.dtype, means3d.dtype, means3d.dtype]
        )
        let meanNdc = sliceFirstDim(projectionForward[0], count: activeCount)

        let scalesPadded = padFirstDimToAtLeast(scales, count: paddedCount)
        let rotationsPadded = padFirstDimToAtLeast(rotations, count: paddedCount)
        let cov3dPadded = cov3dForwardKernel(
            [scalesPadded, rotationsPadded, pointCounts],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [[paddedCount, 3, 3]],
            outputDTypes: [scales.dtype]
        )[0]
        let cov3d = sliceFirstDim(cov3dPadded, count: activeCount)

        let screenGrads = _screenSpaceVJP(
            primals: [
                meanNdc, means3d, shs, cov3d,
                cameraCenter, viewMatrix, fovX, fovY, focalX, focalY, imageWidth, imageHeight,
            ],
            cotangents: [cotMeans2d, cotCov2d, cotColor, cotConic],
            colorBackwardKernel: colorBackwardKernel,
            covForwardKernel: covForwardKernel,
            covBackwardKernel: covBackwardKernel,
            inverseBackwardKernel: inverseBackwardKernel,
            activeShDegree: activeShDegree
        )

        let cotPView = MLXArray.zeros([activeCount, 4], dtype: means3d.dtype)
        cotPView[.ellipsis, 2] = cotDepths
        let projectionGrads = _projectionNdcVJP(
            primals: [means3d, viewMatrix, projMatrix],
            cotangents: [screenGrads[0], cotPView],
            backwardKernel: projectionBackwardKernel
        )

        let cov3dGrads = _cov3dVJP(
            primals: [scales, rotations],
            cotangents: [screenGrads[3]],
            backwardKernel: cov3dBackwardKernel
        )

        return [
            cov3dGrads[0],  // gradScales
            cov3dGrads[1],  // gradRotations
            projectionGrads[0] + screenGrads[1],  // gradMeans3d
            screenGrads[2],  // gradShs
            screenGrads[4],  // gradCameraCenter
            MLXArray.zeros(viewMatrix.shape, dtype: viewMatrix.dtype),
            MLXArray.zeros(projMatrix.shape, dtype: projMatrix.dtype),
            MLXArray.zeros(fovX.shape, dtype: fovX.dtype),
            MLXArray.zeros(fovY.shape, dtype: fovY.dtype),
            MLXArray.zeros(focalX.shape, dtype: focalX.dtype),
            MLXArray.zeros(focalY.shape, dtype: focalY.dtype),
            MLXArray.zeros(imageWidth.shape, dtype: imageWidth.dtype),
            MLXArray.zeros(imageHeight.shape, dtype: imageHeight.dtype),
        ]
    }

    private lazy var screenSpaceCustomFunction: (([MLXArray]) -> [MLXArray])? = {
        guard useScreenSpaceCustomOp else {
            return nil
        }
        guard
            let colorForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_color_forward_mlx"),
            let colorBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_color_backward_mlx"),
            let covForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov2d_forward_mlx"),
            let covBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_cov2d_backward_mlx"),
            let inverseForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_inverse2d_forward_mlx"),
            let inverseBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_inverse2d_backward_mlx")
        else {
            Logger.shared.debug("Slang kernels are unavailable. Fallback path is used.")
            return nil
        }

        let activeShDegree = self.active_sh_degree
        let forward: ([MLXArray]) -> [MLXArray] = { inputs in
            let meanNdc = inputs[0]
            let means3d = inputs[1]
            let shs = inputs[2]
            let cov3d = inputs[3]
            let cameraCenter = inputs[4]
            let viewMatrix = inputs[5]
            let fovX = inputs[6]
            let fovY = inputs[7]
            let focalX = inputs[8]
            let focalY = inputs[9]
            let imageWidth = inputs[10]
            let imageHeight = inputs[11]

            let fovXKernel = fovX.reshaped([1])
            let fovYKernel = fovY.reshaped([1])
            let focalXKernel = focalX.reshaped([1])
            let focalYKernel = focalY.reshaped([1])

            let activeCount = meanNdc.shape[0]
            let paddedCount = Swift.max(activeCount, Self.screenSpaceCustomMinPointCount)
            let meanNdcPadded = Self.padFirstDimToAtLeast(meanNdc, count: paddedCount)
            let means3dPadded = Self.padFirstDimToAtLeast(means3d, count: paddedCount)
            let shsPadded = Self.padFirstDimToAtLeast(shs, count: paddedCount)
            let cov3dPadded = Self.padFirstDimToAtLeast(cov3d, count: paddedCount)
            let colorCounts = MLXArray(
                [UInt32(paddedCount), UInt32(activeShDegree), UInt32(shsPadded.shape[1])]
            )
            let pointCounts = MLXArray([UInt32(paddedCount)])

            let colorPadded = colorForwardKernel(
                [means3dPadded, shsPadded, cameraCenter, colorCounts],
                grid: (max(paddedCount, 1), 1, 1),
                threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
                outputShapes: [[paddedCount, 3]],
                outputDTypes: [means3d.dtype]
            )[0]

            let cov2dPadded = covForwardKernel(
                [
                    means3dPadded, cov3dPadded, viewMatrix, fovXKernel, fovYKernel, focalXKernel,
                    focalYKernel, pointCounts,
                ],
                grid: (max(paddedCount, 1), 1, 1),
                threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
                outputShapes: [[paddedCount, 2, 2]],
                outputDTypes: [cov3d.dtype]
            )[0]

            let conicPadded = inverseForwardKernel(
                [cov2dPadded, pointCounts],
                grid: (max(paddedCount, 1), 1, 1),
                threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
                outputShapes: [[paddedCount, 2, 2]],
                outputDTypes: [cov3d.dtype]
            )[0]

            let meanCoordX = ((meanNdcPadded[.ellipsis, 0] + 1) * imageWidth - 1.0) * 0.5
            let meanCoordY = ((meanNdcPadded[.ellipsis, 1] + 1) * imageHeight - 1.0) * 0.5
            let means2dPadded = MLX.stacked([meanCoordX, meanCoordY], axis: -1)

            return [
                Self.sliceFirstDim(means2dPadded, count: activeCount),
                Self.sliceFirstDim(cov2dPadded, count: activeCount),
                Self.sliceFirstDim(colorPadded, count: activeCount),
                Self.sliceFirstDim(conicPadded, count: activeCount),
            ]
        }

        let renderer = self
        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
                if let profiler = renderer.profiler {
                    profiler.measure("bwd.screenSpace.inputs") {
                        eval(primals)
                        eval(cotangents)
                    }
                    return profiler.measure("bwd.screenSpace") {
                        let result = Self._screenSpaceVJP(
                            primals: primals, cotangents: cotangents,
                            colorBackwardKernel: colorBackwardKernel,
                            covForwardKernel: covForwardKernel,
                            covBackwardKernel: covBackwardKernel,
                            inverseBackwardKernel: inverseBackwardKernel,
                            activeShDegree: activeShDegree
                        )
                        eval(result)
                        return result
                    }
                }
                return Self._screenSpaceVJP(
                    primals: primals, cotangents: cotangents,
                    colorBackwardKernel: colorBackwardKernel,
                    covForwardKernel: covForwardKernel,
                    covBackwardKernel: covBackwardKernel,
                    inverseBackwardKernel: inverseBackwardKernel,
                    activeShDegree: activeShDegree
                )
            }
        }
    }()

    private static func _screenSpaceVJP(
        primals: [MLXArray], cotangents: [MLXArray],
        colorBackwardKernel: MLXFast.MLXFastKernel,
        covForwardKernel: MLXFast.MLXFastKernel,
        covBackwardKernel: MLXFast.MLXFastKernel,
        inverseBackwardKernel: MLXFast.MLXFastKernel,
        activeShDegree: Int
    ) -> [MLXArray] {
        let meanNdc = primals[0]
        let means3d = primals[1]
        let shs = primals[2]
        let cov3d = primals[3]
        let cameraCenter = primals[4]
        let viewMatrix = primals[5]
        let fovX = primals[6]
        let fovY = primals[7]
        let focalX = primals[8]
        let focalY = primals[9]
        let imageWidth = primals[10]
        let imageHeight = primals[11]
        let fovXKernel = fovX.reshaped([1])
        let fovYKernel = fovY.reshaped([1])
        let focalXKernel = focalX.reshaped([1])
        let focalYKernel = focalY.reshaped([1])

        let cotMeans2d = cotangents[0]
        let cotCov2d = cotangents[1]
        let cotColor = cotangents[2]
        let cotConic = cotangents[3]

        let activeCount = meanNdc.shape[0]
        let paddedCount = Swift.max(activeCount, screenSpaceCustomMinPointCount)
        let meanNdcPadded = padFirstDimToAtLeast(meanNdc, count: paddedCount)
        let means3dPadded = padFirstDimToAtLeast(means3d, count: paddedCount)
        let shsPadded = padFirstDimToAtLeast(shs, count: paddedCount)
        let cov3dPadded = padFirstDimToAtLeast(cov3d, count: paddedCount)
        let cotMeans2dPadded = padFirstDimToAtLeast(cotMeans2d, count: paddedCount)
        let cotCov2dPadded = padFirstDimToAtLeast(cotCov2d, count: paddedCount)
        let cotColorPadded = padFirstDimToAtLeast(cotColor, count: paddedCount)
        let cotConicPadded = padFirstDimToAtLeast(cotConic, count: paddedCount)

        let colorCounts = MLXArray(
            [UInt32(paddedCount), UInt32(activeShDegree), UInt32(shsPadded.shape[1])]
        )
        let pointCounts = MLXArray([UInt32(paddedCount)])

        let colorGrads = colorBackwardKernel(
            [means3dPadded, shsPadded, cameraCenter, cotColorPadded, colorCounts],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [means3dPadded.shape, shsPadded.shape],
            outputDTypes: [means3d.dtype, shs.dtype],
            initValue: 0.0
        )

        let cov2dPadded = covForwardKernel(
            [
                means3dPadded, cov3dPadded, viewMatrix, fovXKernel, fovYKernel, focalXKernel,
                focalYKernel, pointCounts,
            ],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [[paddedCount, 2, 2]],
            outputDTypes: [cov3d.dtype]
        )[0]

        let gradCov2dFromConic = inverseBackwardKernel(
            [cov2dPadded, cotConicPadded, pointCounts],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [cov2dPadded.shape],
            outputDTypes: [cov2dPadded.dtype]
        )[0]
        let totalCotCov2d = cotCov2dPadded + gradCov2dFromConic

        let covGrads = covBackwardKernel(
            [
                means3dPadded, cov3dPadded, viewMatrix, fovXKernel, fovYKernel, focalXKernel,
                focalYKernel, totalCotCov2d, pointCounts,
            ],
            grid: (max(paddedCount, 1), 1, 1),
            threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
            outputShapes: [means3dPadded.shape, cov3dPadded.shape],
            outputDTypes: [means3d.dtype, cov3d.dtype]
        )

        let gradMeanNdcPadded = MLXArray.zeros(meanNdcPadded.shape, dtype: meanNdc.dtype)
        gradMeanNdcPadded[.ellipsis, 0] = cotMeans2dPadded[.ellipsis, 0] * imageWidth * 0.5
        gradMeanNdcPadded[.ellipsis, 1] = cotMeans2dPadded[.ellipsis, 1] * imageHeight * 0.5

        let gradMeans3dPadded = colorGrads[0] + covGrads[0]
        let gradShsPadded = colorGrads[1]
        let gradCov3dPadded = covGrads[1]
        let gradCameraCenter = -(colorGrads[0].sum(axis: 0)).expandedDimensions(axes: [0])

        return [
            sliceFirstDim(gradMeanNdcPadded, count: activeCount),
            sliceFirstDim(gradMeans3dPadded, count: activeCount),
            sliceFirstDim(gradShsPadded, count: activeCount),
            sliceFirstDim(gradCov3dPadded, count: activeCount),
            gradCameraCenter,
            MLXArray.zeros(viewMatrix.shape, dtype: viewMatrix.dtype),
            MLXArray.zeros(fovX.shape, dtype: fovX.dtype),
            MLXArray.zeros(fovY.shape, dtype: fovY.dtype),
            MLXArray.zeros(focalX.shape, dtype: focalX.dtype),
            MLXArray.zeros(focalY.shape, dtype: focalY.dtype),
            MLXArray.zeros(imageWidth.shape, dtype: imageWidth.dtype),
            MLXArray.zeros(imageHeight.shape, dtype: imageHeight.dtype),
        ]
    }

    init(
        active_sh_degree: Int,
        W: Int,
        H: Int,
        TILE_SIZE: TILE_SIZE_H_W,
        whiteBackground: Bool,
        useScreenSpaceCustomOp: Bool = true
    ) {
        self.active_sh_degree = active_sh_degree
        self.debug = false
        self.whiteBackground = whiteBackground
        self.useScreenSpaceCustomOp = useScreenSpaceCustomOp
        self.TILE_SIZE = TILE_SIZE
        self.W = W
        self.H = H
        self.gridW = (W + TILE_SIZE.w - 1) / TILE_SIZE.w
        self.gridH = (H + TILE_SIZE.h - 1) / TILE_SIZE.h
        self.pix_coord = createMeshGrid(shape: [H, W])
        precondition(
            projectionScreenFusedCustomFunction != nil,
            "Required Slang kernel `gaussian_projection_screen_fused_forward_mlx` is unavailable."
        )
        precondition(
            globalTileCompositeCustomFunction != nil,
            "Required Slang kernels `gaussian_tile_global_forward_mlx` / `gaussian_tile_global_backward_mlx` are unavailable."
        )
        precondition(
            globalTileSliceKernelsAvailable,
            "Required Slang global tile slice kernels are unavailable."
        )
        _ = fusedRadixSortTileKeysKernel
    }

    func render(
        camera: Camera,
        means2d: MLXArray,
        cov2d: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray,
        radii: MLXArray,
        conic: MLXArray,
        inputIsDepthSorted: Bool = false
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visiility_filter: MLXArray,
        radii: MLXArray
    ) {
        return render(
            imageWidth: camera.imageWidth,
            imageHeight: camera.imageHeight,
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            radii: radii,
            conic: conic,
            inputIsDepthSorted: inputIsDepthSorted
        )
    }

    func render(
        imageWidth: Int,
        imageHeight: Int,
        means2d: MLXArray,
        cov2d: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray,
        radii: MLXArray,
        conic: MLXArray,
        inputIsDepthSorted: Bool = false
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visiility_filter: MLXArray,
        radii: MLXArray
    ) {
        _ = inputIsDepthSorted
        precondition(
            imageWidth == self.W && imageHeight == self.H,
            "Renderer image size mismatch: expected (\(self.W), \(self.H)), got (\(imageWidth), \(imageHeight))"
        )
        Logger.shared.debug("get_rect")
        let rect = get_rect(
            pix_coord: means2d,
            radii: radii,
            width: imageWidth,
            height: imageHeight
        )
        let conicValues = conic
        let packedGaussians = buildPackedGaussians(
            means2d: means2d,
            conic: conicValues,
            color: color,
            opacity: opacity,
            depths: depths
        )
        guard let globalTileSliceInfo = buildGlobalTileSliceInfo(
            rect: rect,
            radii: radii,
            depths: depths
        ) else {
            fatalError("Required global tile slice kernels are unavailable.")
        }
        guard let blended = renderGlobalTileCompositeCustomOp(
            packedGaussians: packedGaussians,
            globalTileSliceInfo: globalTileSliceInfo
        ) else {
            fatalError("Required global tile composite kernels are unavailable.")
        }

        let renderColor = blended.0.reshaped([H, W, 3])
        let renderDepth = blended.1.reshaped([H, W, 1])
        let renderAlpha = blended.2.reshaped([H, W, 1])
        return (renderColor, renderDepth, renderAlpha, radii .> 0, radii)
    }

    func forwardWithCameraParams(
        viewMatrix: MLXArray,
        projMatrix: MLXArray,
        cameraCenter: MLXArray,
        fovX: MLXArray,
        fovY: MLXArray,
        focalX: MLXArray,
        focalY: MLXArray,
        imageWidth: Int,
        imageHeight: Int,
        means3d: MLXArray,
        shs: MLXArray,
        opacity: MLXArray,
        scales: MLXArray,
        rotations: MLXArray
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visiility_filter: MLXArray,
        radii: MLXArray
    ) {
        let imageWidthArray = MLXArray(Float(imageWidth))
        let imageHeightArray = MLXArray(Float(imageHeight))

        guard let projectionScreenFusedCustomFunction else {
            fatalError("Required projection+screen fused kernel is unavailable.")
        }
        Logger.shared.debug("projection_screen_fused")
        let outputs = projectionScreenFusedCustomFunction(
            [
                scales, rotations, means3d, shs, cameraCenter, viewMatrix, projMatrix,
                fovX, fovY, focalX, focalY, imageWidthArray, imageHeightArray,
            ]
        )
        let means2d = outputs[ProjectionScreenFusedOutputIndex.means2d]
        let depths = outputs[ProjectionScreenFusedOutputIndex.depths]
        let color = outputs[ProjectionScreenFusedOutputIndex.color]
        let cov2d = outputs[ProjectionScreenFusedOutputIndex.cov2d]
        let conic = outputs[ProjectionScreenFusedOutputIndex.conic]
        let radii = MLX.stopGradient(outputs[ProjectionScreenFusedOutputIndex.radii])
        Logger.shared.debug("render")
        let rets = render(
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            radii: radii,
            conic: conic
        )
        return rets
    }

    func forward(
        camera: Camera,
        means3d: MLXArray,
        shs: MLXArray,
        opacity: MLXArray,
        scales: MLXArray,
        rotations: MLXArray
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visiility_filter: MLXArray,
        radii: MLXArray
    ) {
        let cameraCenter = MLXArray(
            [Float(camera.cameraCenter.x), Float(camera.cameraCenter.y), Float(camera.cameraCenter.z)]
                as [Float])[.newAxis, .ellipsis]
        let imageWidth = MLXArray(Float(camera.imageWidth))
        let imageHeight = MLXArray(Float(camera.imageHeight))

        guard let projectionScreenFusedCustomFunction else {
            fatalError("Required projection+screen fused kernel is unavailable.")
        }
        Logger.shared.debug("projection_screen_fused")
        let outputs = projectionScreenFusedCustomFunction(
            [
                scales, rotations, means3d, shs, cameraCenter,
                camera.worldViewTransform, camera.projectionMatrix,
                camera.FoVx, camera.FoVy, camera.focalX, camera.focalY, imageWidth, imageHeight,
            ]
        )
        let means2d = outputs[ProjectionScreenFusedOutputIndex.means2d]
        let depths = outputs[ProjectionScreenFusedOutputIndex.depths]
        let color = outputs[ProjectionScreenFusedOutputIndex.color]
        let cov2d = outputs[ProjectionScreenFusedOutputIndex.cov2d]
        let conic = outputs[ProjectionScreenFusedOutputIndex.conic]
        let radii = MLX.stopGradient(outputs[ProjectionScreenFusedOutputIndex.radii])
        Logger.shared.debug("render")
        let rets = render(
            camera: camera,
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            radii: radii,
            conic: conic
        )
        return rets
    }

    func get_scales_from(_ scales: MLXArray) -> MLXArray {
        MLX.exp(scales)
    }
    func normalizeRows(_ x: MLXArray, axis: Int = 1, eps: Float = 1e-8)
        -> MLXArray
    {
        // x: [N, ...]
        let norm = MLX.sqrt(
            MLX.sum(MLX.square(x), axes: [axis], keepDims: true)
        )
        return x / (norm + eps)
    }

    func get_rotation_from(_ rotation: MLXArray) -> MLXArray {
        normalizeRows(rotation)
    }
    func get_xyz_from(_ xyz: MLXArray) -> MLXArray {
        xyz
    }
    func get_features_from(
        _ features_dc: MLXArray,
        _ features_rest: MLXArray
    ) -> MLXArray {
        return MLX.concatenated([features_dc, features_rest], axis: 1)
    }
    func get_opacity_from(_ opacity: MLXArray) -> MLXArray {
        MLX.sigmoid(opacity)
    }
}
