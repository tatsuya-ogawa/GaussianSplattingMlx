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

func computeTileMask(
    h: Int,
    w: Int,
    tileSize: TILE_SIZE_H_W,
    rect: (MLXArray, MLXArray)
) -> MLXArray {
    let over_tl_0 = MLX.clip(rect.0[.ellipsis, 0], min: w)
    let over_tl_1 = MLX.clip(rect.0[.ellipsis, 1], min: h)

    let over_br_0 = MLX.clip(
        rect.1[.ellipsis, 0],
        max: w + tileSize.w - 1
    )
    let over_br_1 = MLX.clip(
        rect.1[.ellipsis, 1],
        max: h + tileSize.h - 1
    )
    return (over_br_0 .> over_tl_0) & (over_br_1 .> over_tl_1)
}

func computeDxDy(
    tile_coord: MLXArray,
    sorted_means2d: MLXArray,
) -> MLXArray {
    return tile_coord.expandedDimensions(axes: [1])
        - sorted_means2d.expandedDimensions(axes: [0])
}

func computeGaussianWeights(
    tile_coord: MLXArray,
    sorted_means2d: MLXArray,
    sorted_conic: MLXArray
) -> MLXArray {
    let dx = computeDxDy(tile_coord: tile_coord, sorted_means2d: sorted_means2d)
    return MLX.exp(
        -0.5
            * (MLX.square(dx[0..., 0..., 0]) * sorted_conic[0..., 0, 0]
                + MLX.square(dx[0..., 0..., 1]) * sorted_conic[0..., 1, 1]
                + dx[0..., 0..., 0] * dx[0..., 0..., 1]
                * sorted_conic[0..., 0, 1]
                + dx[0..., 0..., 0] * dx[0..., 0..., 1]
                * sorted_conic[0..., 1, 0])
    )
}

func getSortedValues(
    means2d: MLXArray,
    cov2d: MLXArray,
    color: MLXArray,
    opacity: MLXArray,
    depths: MLXArray,
    conic: MLXArray? = nil,
    in_mask: MLXArray,
    inputIsDepthSorted: Bool = false
) -> (
    depths: MLXArray,
    means2d: MLXArray,
    cov2d: MLXArray,
    conic: MLXArray,
    opacity: MLXArray,
    color: MLXArray
) {
    let sorted_depth_indices: MLXArray = inputIsDepthSorted
        ? in_mask
        : MLX.stopGradient(MLX.argSort(depths[in_mask]))
    let sorted_depths = inputIsDepthSorted
        ? depths[sorted_depth_indices]
        : depths[in_mask][sorted_depth_indices]
    let sorted_means2d = inputIsDepthSorted
        ? means2d[sorted_depth_indices]
        : means2d[in_mask][sorted_depth_indices]
    let sorted_cov2d = inputIsDepthSorted
        ? cov2d[sorted_depth_indices]
        : cov2d[in_mask][sorted_depth_indices]
    let sorted_conic: MLXArray
    if let conic {
        sorted_conic = inputIsDepthSorted
            ? conic[sorted_depth_indices]
            : conic[in_mask][sorted_depth_indices]
    } else {
        sorted_conic = matrixInverse2d(sorted_cov2d)
    }
    let sorted_opacity = inputIsDepthSorted
        ? opacity[sorted_depth_indices]
        : opacity[in_mask][sorted_depth_indices]
    let sorted_color = inputIsDepthSorted
        ? color[sorted_depth_indices]
        : color[in_mask][sorted_depth_indices]

    return (
        sorted_depths,
        sorted_means2d,
        sorted_cov2d,
        sorted_conic,
        sorted_opacity,
        sorted_color
    )
}
struct TILE_SIZE_H_W {
    let w: Int
    let h: Int
}

private struct TileEntry {
    let h: Int
    let w: Int
    let size: TILE_SIZE_H_W
    let tileCoord: MLXArray
}

private struct GlobalTileSliceInfo {
    let sortedGaussIdx: MLXArray
    let tileRanges: [UInt32]
}

class GaussianRenderer {
    private enum ScreenSpaceCustomOutputIndex {
        static let means2d = 0
        static let cov2d = 1
        static let color = 2
        static let conic = 3
    }
    
    private enum PackedGaussianIndex {
        static let means2d = 0..<2
        static let conic = 2..<6
        static let color = 6..<9
        static let opacity = 9..<10
        static let depth = 10
    }

    private static let fusedTileCustomMinGaussianCount = 9
    private static let screenSpaceCustomMinPointCount = 3
    private static let emptyInt32Indices = MLXArray([] as [Int32])

    let debug: Bool
    let active_sh_degree: Int
    let W: Int
    let H: Int
    let pix_coord: MLXArray
    let whiteBackground: Bool
    let TILE_SIZE: TILE_SIZE_H_W
    let useFusedTileCustomOp: Bool
    let useScreenSpaceCustomOp: Bool
    private let tileEntries: [TileEntry]
    private let gridW: Int
    private let gridH: Int

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

    private lazy var fusedTileCompositeCustomFunction: (([MLXArray]) -> [MLXArray])? = {
        guard useFusedTileCustomOp else {
            return nil
        }
        guard
            let forwardKernel = try? SlangKernelSpecLoader.loadKernel(named: "gaussian_tile_forward_mlx"),
            let backwardKernel = try? SlangKernelSpecLoader.loadKernel(named: "gaussian_tile_backward_mlx")
        else {
            Logger.shared.debug("Slang tile kernels are unavailable. Tile fallback path is used.")
            return nil
        }
        let whiteBackground = self.whiteBackground

        let forward: ([MLXArray]) -> [MLXArray] = { inputs in
            let tileCoord = inputs[0]
            let sortedDepths = inputs[1]
            let sortedMeans2d = inputs[2]
            let sortedConic = inputs[3]
            let sortedOpacity = inputs[4]
            let sortedColor = inputs[5]
            let activeGaussianCount = inputs[6].item(Int.self)

            let pixelCount = tileCoord.shape[0]
            let sortedConicFlat = sortedConic.reshaped([-1, 4])
            let counts = MLXArray(
                [
                    UInt32(pixelCount),
                    UInt32(activeGaussianCount),
                    UInt32(whiteBackground ? 1 : 0),
                ]
            )
            return forwardKernel(
                [
                    tileCoord,
                    sortedDepths,
                    sortedMeans2d,
                    sortedConicFlat,
                    sortedOpacity,
                    sortedColor,
                    counts,
                ],
                grid: (max(pixelCount, 1), 1, 1),
                threadGroup: (min(128, max(pixelCount, 1)), 1, 1),
                outputShapes: [
                    [pixelCount, 3],
                    [pixelCount, 1],
                    [pixelCount, 1],
                ],
                outputDTypes: [
                    sortedColor.dtype,
                    sortedDepths.dtype,
                    sortedOpacity.dtype,
                ]
            )
        }

        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
                let tileCoord = primals[0]
                let sortedDepths = primals[1]
                let sortedMeans2d = primals[2]
                let sortedConic = primals[3]
                let sortedOpacity = primals[4]
                let sortedColor = primals[5]
                let activeGaussianCount = primals[6].item(Int.self)
                let cotColor = cotangents[0]
                let cotDepth = cotangents[1]
                let cotAlpha = cotangents[2]

                let pixelCount = tileCoord.shape[0]
                let sortedConicFlat = sortedConic.reshaped([-1, 4])
                let counts = MLXArray(
                    [
                        UInt32(pixelCount),
                        UInt32(activeGaussianCount),
                        UInt32(whiteBackground ? 1 : 0),
                    ]
                )

                let kernelGrads = backwardKernel(
                    [
                        tileCoord,
                        sortedDepths,
                        sortedMeans2d,
                        sortedConicFlat,
                        sortedOpacity,
                        sortedColor,
                        cotColor,
                        cotDepth,
                        cotAlpha,
                        counts,
                    ],
                    grid: (max(pixelCount, 1), 1, 1),
                    threadGroup: (min(128, max(pixelCount, 1)), 1, 1),
                    outputShapes: [
                        sortedDepths.shape,
                        sortedMeans2d.shape,
                        sortedConicFlat.shape,
                        sortedOpacity.shape,
                        sortedColor.shape,
                    ],
                    outputDTypes: [
                        sortedDepths.dtype,
                        sortedMeans2d.dtype,
                        sortedConicFlat.dtype,
                        sortedOpacity.dtype,
                        sortedColor.dtype,
                    ],
                    initValue: 0.0
                )

                let gradTileCoord = MLXArray.zeros(tileCoord.shape, dtype: tileCoord.dtype)
                let gradSortedConic = kernelGrads[2].reshaped(sortedConic.shape)
                let gradActiveGaussianCount = MLXArray(0.0)
                return [
                    gradTileCoord,
                    kernelGrads[0],
                    kernelGrads[1],
                    gradSortedConic,
                    kernelGrads[3],
                    kernelGrads[4],
                    gradActiveGaussianCount,
                ]
            }
        }
    }()

    private func renderTileCompositeCustomOp(
        tileCoord: MLXArray,
        sortedDepths: MLXArray,
        sortedMeans2d: MLXArray,
        sortedConic: MLXArray,
        sortedOpacity: MLXArray,
        sortedColor: MLXArray,
        activeGaussianCount: Int
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard let fusedTileCompositeCustomFunction else {
            return nil
        }
        let outputs = fusedTileCompositeCustomFunction(
            [
                tileCoord,
                sortedDepths,
                sortedMeans2d,
                sortedConic,
                sortedOpacity,
                sortedColor,
                MLXArray(Float(activeGaussianCount)),
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

    private lazy var radixExtractBitKernel: MLXFast.MLXFastKernel = {
        MLXFast.metalKernel(
            name: "radix_extract_bit_u32_v1",
            inputNames: ["keyPart", "bitValue"],
            outputNames: ["oneMask"],
            source: """
                uint elem = thread_position_in_grid.x;
                uint n = keyPart_shape[0];
                if (elem >= n) {
                    return;
                }

                uint bit = bitValue;
                uint value = keyPart[elem];
                oneMask[elem] = int((value >> bit) & 1u);
                """
        )
    }()

    private lazy var radixScatterTripletKernel: MLXFast.MLXFastKernel = {
        MLXFast.metalKernel(
            name: "radix_scatter_triplet_u32_v1",
            inputNames: ["keysHighIn", "keysLowIn", "valuesIn", "destIndices"],
            outputNames: ["keysHighOut", "keysLowOut", "valuesOut"],
            source: """
                uint elem = thread_position_in_grid.x;
                uint n = keysHighIn_shape[0];
                if (elem >= n) {
                    return;
                }

                uint dst = uint(destIndices[elem]);
                if (dst >= n) {
                    return;
                }
                keysHighOut[dst] = keysHighIn[elem];
                keysLowOut[dst] = keysLowIn[elem];
                valuesOut[dst] = valuesIn[elem];
                """
        )
    }()

    private func radixBitPassForTileKeys(
        keyPart: MLXArray,
        bit: Int,
        keysHigh: MLXArray,
        keysLow: MLXArray,
        values: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let n = keyPart.shape[0]
        let oneMask = radixExtractBitKernel(
            [keyPart, MLXArray(UInt32(bit))],
            grid: (max(n, 1), 1, 1),
            threadGroup: (min(256, max(n, 1)), 1, 1),
            outputShapes: [[n]],
            outputDTypes: [.int32]
        )[0]
        let zeroMask = 1 - oneMask

        let zeroInclusive = zeroMask.cumsum(axis: 0).asType(.int32)
        let oneInclusive = oneMask.cumsum(axis: 0).asType(.int32)
        let zeroExclusive = zeroInclusive - zeroMask
        let oneExclusive = oneInclusive - oneMask
        let zeroCount = zeroInclusive[n - 1]

        let destination =
            zeroExclusive
            + oneMask * (zeroCount + oneExclusive - zeroExclusive)
        let destinationIndices = destination.asType(.int32)

        let scattered = radixScatterTripletKernel(
            [keysHigh, keysLow, values, destinationIndices],
            grid: (max(n, 1), 1, 1),
            threadGroup: (min(256, max(n, 1)), 1, 1),
            outputShapes: [keysHigh.shape, keysLow.shape, values.shape],
            outputDTypes: [keysHigh.dtype, keysLow.dtype, values.dtype]
        )
        return (scattered[0], scattered[1], scattered[2])
    }

    private func radixSortTileKeys(
        keysHigh: MLXArray,
        keysLow: MLXArray,
        values: MLXArray,
        tileBitCount: Int
    ) -> (sortedKeysHigh: MLXArray, sortedKeysLow: MLXArray, sortedValues: MLXArray) {
        var sortedKeysHigh = keysHigh
        var sortedKeysLow = keysLow
        var sortedValues = values

        for bit in 0..<32 {
            (sortedKeysHigh, sortedKeysLow, sortedValues) = radixBitPassForTileKeys(
                keyPart: sortedKeysLow,
                bit: bit,
                keysHigh: sortedKeysHigh,
                keysLow: sortedKeysLow,
                values: sortedValues
            )
        }

        let keyHighBits = Swift.max(tileBitCount, 1)
        for bit in 0..<keyHighBits {
            (sortedKeysHigh, sortedKeysLow, sortedValues) = radixBitPassForTileKeys(
                keyPart: sortedKeysHigh,
                bit: bit,
                keysHigh: sortedKeysHigh,
                keysLow: sortedKeysLow,
                values: sortedValues
            )
        }

        return (sortedKeysHigh, sortedKeysLow, sortedValues)
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

    private var globalTileSliceKernelsAvailable: Bool {
        countTilesPerGaussianKernel != nil
            && generateKeysKernel != nil
            && computeTileRangesKernel != nil
    }

    private func buildGlobalTileSliceInfo(
        rect: (MLXArray, MLXArray),
        radii: MLXArray,
        depths: MLXArray
    ) -> GlobalTileSliceInfo? {
        guard
            let countTilesPerGaussianKernel,
            let generateKeysKernel,
            let computeTileRangesKernel
        else {
            return nil
        }

        let gaussianCount = radii.shape[0]
        let numTiles = gridW * gridH
        if gaussianCount == 0 {
            return GlobalTileSliceInfo(
                sortedGaussIdx: Self.emptyInt32Indices,
                tileRanges: [UInt32](repeating: 0, count: numTiles * 2)
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
                sortedGaussIdx: Self.emptyInt32Indices,
                tileRanges: [UInt32](repeating: 0, count: numTiles * 2)
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

        let tileRangesCpu = tileRanges.asArray(UInt32.self)
        return GlobalTileSliceInfo(sortedGaussIdx: sortedGaussIdx, tileRanges: tileRangesCpu)
    }

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

        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
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
        }
    }()

    private func buildCovariance3d(scales: MLXArray, rotations: MLXArray) -> MLXArray {
        if let covariance3DCustomFunction {
            return covariance3DCustomFunction([scales, rotations])[0]
        }
        return build_covariance_3d(s: scales, r: rotations)
    }

    // MARK: - Fused screen-space kernel (cov3d + color + cov2d + inverse2d in one dispatch)

    private lazy var fusedScreenSpaceCustomFunction: (([MLXArray]) -> [MLXArray])? = {
        guard useScreenSpaceCustomOp else {
            return nil
        }
        guard
            let fusedForwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_fused_forward_mlx"),
            let fusedBackwardKernel = try? SlangKernelSpecLoader.loadKernel(
                named: "gaussian_screen_fused_backward_mlx")
        else {
            Logger.shared.debug("Fused screen-space kernels are unavailable. Falling back to separate kernels.")
            return nil
        }

        let activeShDegree = self.active_sh_degree
        let forward: ([MLXArray]) -> [MLXArray] = { inputs in
            let meanNdc = inputs[0]
            let scales = inputs[1]
            let rotations = inputs[2]
            let means3d = inputs[3]
            let shs = inputs[4]
            let cameraCenter = inputs[5]
            let viewMatrix = inputs[6]
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

            let activeCount = meanNdc.shape[0]
            let paddedCount = Swift.max(activeCount, Self.screenSpaceCustomMinPointCount)
            let meanNdcPadded = Self.padFirstDimToAtLeast(meanNdc, count: paddedCount)
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
                    cameraCenter, viewMatrix, fovXKernel, fovYKernel, focalXKernel, focalYKernel,
                    counts,
                ],
                grid: (max(paddedCount, 1), 1, 1),
                threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
                outputShapes: [[paddedCount, 3], [paddedCount, 2, 2], [paddedCount, 2, 2]],
                outputDTypes: [means3d.dtype, means3d.dtype, means3d.dtype]
            )

            let colorPadded = outputs[0]
            let cov2dPadded = outputs[1]
            let conicPadded = outputs[2]

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

        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
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
                let fovXKernel = fovX.reshaped([1])
                let fovYKernel = fovY.reshaped([1])
                let focalXKernel = focalX.reshaped([1])
                let focalYKernel = focalY.reshaped([1])

                let cotMeans2d = cotangents[0]
                let cotCov2d = cotangents[1]
                let cotColor = cotangents[2]
                let cotConic = cotangents[3]

                let activeCount = meanNdc.shape[0]
                let paddedCount = Swift.max(activeCount, Self.screenSpaceCustomMinPointCount)
                let meanNdcPadded = Self.padFirstDimToAtLeast(meanNdc, count: paddedCount)
                let scalesPadded = Self.padFirstDimToAtLeast(scales, count: paddedCount)
                let rotationsPadded = Self.padFirstDimToAtLeast(rotations, count: paddedCount)
                let means3dPadded = Self.padFirstDimToAtLeast(means3d, count: paddedCount)
                let shsPadded = Self.padFirstDimToAtLeast(shs, count: paddedCount)
                let cotMeans2dPadded = Self.padFirstDimToAtLeast(cotMeans2d, count: paddedCount)
                let cotCov2dPadded = Self.padFirstDimToAtLeast(cotCov2d, count: paddedCount)
                let cotColorPadded = Self.padFirstDimToAtLeast(cotColor, count: paddedCount)
                let cotConicPadded = Self.padFirstDimToAtLeast(cotConic, count: paddedCount)

                let counts = MLXArray(
                    [UInt32(paddedCount), UInt32(activeShDegree), UInt32(shsPadded.shape[1])]
                )

                let grads = fusedBackwardKernel(
                    [
                        scalesPadded, rotationsPadded, means3dPadded, shsPadded,
                        cameraCenter, viewMatrix, fovXKernel, fovYKernel, focalXKernel, focalYKernel,
                        cotColorPadded, cotCov2dPadded, cotConicPadded, counts,
                    ],
                    grid: (max(paddedCount, 1), 1, 1),
                    threadGroup: (min(128, max(paddedCount, 1)), 1, 1),
                    outputShapes: [scalesPadded.shape, rotationsPadded.shape, means3dPadded.shape, shsPadded.shape],
                    outputDTypes: [scales.dtype, rotations.dtype, means3d.dtype, shs.dtype],
                    initValue: 0.0
                )

                let gradScalesPadded = grads[0]
                let gradRotationsPadded = grads[1]
                let gradMeans3dPadded = grads[2]
                let gradShsPadded = grads[3]

                let gradMeanNdcPadded = MLXArray.zeros(meanNdcPadded.shape, dtype: meanNdc.dtype)
                gradMeanNdcPadded[.ellipsis, 0] = cotMeans2dPadded[.ellipsis, 0] * imageWidth * 0.5
                gradMeanNdcPadded[.ellipsis, 1] = cotMeans2dPadded[.ellipsis, 1] * imageHeight * 0.5

                let gradCameraCenter = -(Self.sliceFirstDim(gradMeans3dPadded, count: activeCount).sum(axis: 0)).expandedDimensions(axes: [0])

                return [
                    Self.sliceFirstDim(gradMeanNdcPadded, count: activeCount),
                    Self.sliceFirstDim(gradScalesPadded, count: activeCount),
                    Self.sliceFirstDim(gradRotationsPadded, count: activeCount),
                    Self.sliceFirstDim(gradMeans3dPadded, count: activeCount),
                    Self.sliceFirstDim(gradShsPadded, count: activeCount),
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
        }
    }()

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

        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
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
                let paddedCount = Swift.max(activeCount, Self.screenSpaceCustomMinPointCount)
                let meanNdcPadded = Self.padFirstDimToAtLeast(meanNdc, count: paddedCount)
                let means3dPadded = Self.padFirstDimToAtLeast(means3d, count: paddedCount)
                let shsPadded = Self.padFirstDimToAtLeast(shs, count: paddedCount)
                let cov3dPadded = Self.padFirstDimToAtLeast(cov3d, count: paddedCount)
                let cotMeans2dPadded = Self.padFirstDimToAtLeast(cotMeans2d, count: paddedCount)
                let cotCov2dPadded = Self.padFirstDimToAtLeast(cotCov2d, count: paddedCount)
                let cotColorPadded = Self.padFirstDimToAtLeast(cotColor, count: paddedCount)
                let cotConicPadded = Self.padFirstDimToAtLeast(cotConic, count: paddedCount)

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
                    Self.sliceFirstDim(gradMeanNdcPadded, count: activeCount),
                    Self.sliceFirstDim(gradMeans3dPadded, count: activeCount),
                    Self.sliceFirstDim(gradShsPadded, count: activeCount),
                    Self.sliceFirstDim(gradCov3dPadded, count: activeCount),
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
        }
    }()

    private func buildScreenSpaceFallback(
        meanNdc: MLXArray,
        means3d: MLXArray,
        shs: MLXArray,
        cov3d: MLXArray,
        cameraCenter: MLXArray,
        viewMatrix: MLXArray,
        fovX: MLXArray,
        fovY: MLXArray,
        focalX: MLXArray,
        focalY: MLXArray,
        imageWidth: MLXArray,
        imageHeight: MLXArray
    ) -> [MLXArray] {
        let color = build_color(
            means3d: means3d,
            shs: shs,
            cameraCenter: cameraCenter,
            activeShDegree: self.active_sh_degree
        )
        let cov2d = build_covariance_2d(
            mean3d: means3d,
            cov3d: cov3d,
            viewMatrix: viewMatrix,
            fovX: fovX,
            fovY: fovY,
            focalX: focalX,
            focalY: focalY
        )
        let meanCoordX = ((meanNdc[.ellipsis, 0] + 1) * imageWidth - 1.0) * 0.5
        let meanCoordY = ((meanNdc[.ellipsis, 1] + 1) * imageHeight - 1.0) * 0.5
        let means2d = MLX.stacked([meanCoordX, meanCoordY], axis: -1)
        let conic = matrixInverse2d(cov2d)
        return [means2d, cov2d, color, conic]
    }

    private func buildScreenSpaceFallback(
        camera: Camera,
        meanNdc: MLXArray,
        means3d: MLXArray,
        shs: MLXArray,
        cov3d: MLXArray
    ) -> [MLXArray] {
        let cameraCenter = MLXArray(
            [Float(camera.cameraCenter.x), Float(camera.cameraCenter.y), Float(camera.cameraCenter.z)]
                as [Float])[.newAxis, .ellipsis]
        return buildScreenSpaceFallback(
            meanNdc: meanNdc,
            means3d: means3d,
            shs: shs,
            cov3d: cov3d,
            cameraCenter: cameraCenter,
            viewMatrix: camera.worldViewTransform,
            fovX: camera.FoVx,
            fovY: camera.FoVy,
            focalX: camera.focalX,
            focalY: camera.focalY,
            imageWidth: MLXArray(Float(camera.imageWidth)),
            imageHeight: MLXArray(Float(camera.imageHeight))
        )
    }

    init(
        active_sh_degree: Int,
        W: Int,
        H: Int,
        TILE_SIZE: TILE_SIZE_H_W,
        whiteBackground: Bool,
        useFusedTileCustomOp: Bool = true,
        useScreenSpaceCustomOp: Bool = true
    ) {
        self.active_sh_degree = active_sh_degree
        self.debug = false
        self.whiteBackground = whiteBackground
        self.useFusedTileCustomOp = useFusedTileCustomOp
        self.useScreenSpaceCustomOp = useScreenSpaceCustomOp
        self.TILE_SIZE = TILE_SIZE
        self.W = W
        self.H = H
        self.gridW = (W + TILE_SIZE.w - 1) / TILE_SIZE.w
        self.gridH = (H + TILE_SIZE.h - 1) / TILE_SIZE.h
        self.pix_coord = createMeshGrid(shape: [H, W])
        self.tileEntries = Self.makeTileEntries(
            width: W,
            height: H,
            tileSize: TILE_SIZE,
            pixCoord: self.pix_coord
        )
        if useFusedTileCustomOp {
            _ = fusedTileCompositeCustomFunction
        }
    }

    private static func makeTileEntries(
        width: Int,
        height: Int,
        tileSize: TILE_SIZE_H_W,
        pixCoord: MLXArray
    ) -> [TileEntry] {
        var entries: [TileEntry] = []
        for h in stride(from: 0, to: height, by: tileSize.h) {
            for w in stride(from: 0, to: width, by: tileSize.w) {
                let tileSizeRest = TILE_SIZE_H_W(
                    w: Swift.min(width - w, tileSize.w),
                    h: Swift.min(height - h, tileSize.h)
                )
                let tileCoord = MLX.stopGradient(
                    pixCoord[
                        .stride(from: h, to: h + tileSizeRest.h),
                        .stride(from: w, to: w + tileSizeRest.w)
                    ].flattened(start: 0, end: -2)
                ).asType(.float32)
                entries.append(
                    TileEntry(
                        h: h,
                        w: w,
                        size: tileSizeRest,
                        tileCoord: tileCoord
                    )
                )
            }
        }
        return entries
    }

    private func renderTile(
        tile: TileEntry,
        packedGaussians: MLXArray,
        inputIsDepthSorted: Bool = false,
        rect: (MLXArray, MLXArray),
        precomputedIndices: MLXArray? = nil,
        skipThreshold: Int = 0
    ) -> (MLXArray, MLXArray, MLXArray) {
        let h = tile.h
        let w = tile.w
        let tileSizeRest = tile.size
        let in_mask: MLXArray
        if let precomputedIndices {
            in_mask = precomputedIndices
        } else {
            Logger.shared.debug("computeTileMask")
            let in_mask_condition = computeTileMask(
                h: h,
                w: w,
                tileSize: tileSizeRest,
                rect: rect
            )
            in_mask = conditionToIndices(condition: in_mask_condition)
        }
        if in_mask.shape[0] <= skipThreshold {
            Logger.shared.debug("skip tile")
            return (
                self.whiteBackground
                    ? MLXArray.ones([tileSizeRest.h, tileSizeRest.w, 3])
                    : MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 3]),
                MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 1]),
                MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 1])
            )
        }
        let tile_coord = tile.tileCoord
        let maskedPacked = packedGaussians[in_mask]
        let sortedPacked: MLXArray
        if inputIsDepthSorted || precomputedIndices != nil {
            sortedPacked = maskedPacked
        } else {
            let depthSortIndices = MLX.stopGradient(
                MLX.argSort(
                    maskedPacked[.ellipsis, PackedGaussianIndex.depth]
                )
            )
            sortedPacked = maskedPacked[depthSortIndices]
        }
        
        let sortedMeans2d = sortedPacked[.ellipsis, PackedGaussianIndex.means2d]
        let sortedConic = sortedPacked[.ellipsis, PackedGaussianIndex.conic].reshaped([-1, 2, 2])
        let sortedColor = sortedPacked[.ellipsis, PackedGaussianIndex.color]
        let sortedOpacity = sortedPacked[.ellipsis, PackedGaussianIndex.opacity]
        let sortedDepths = sortedPacked[.ellipsis, PackedGaussianIndex.depth]

        let tile_color: MLXArray
        let tile_depth: MLXArray
        let acc_alpha: MLXArray
        if useFusedTileCustomOp {
            let activeGaussianCount = sortedDepths.shape[0]
            let sortedDepthsPadded = Self.padFirstDimToAtLeast(
                sortedDepths,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedMeans2dPadded = Self.padFirstDimToAtLeast(
                sortedMeans2d,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedConicPadded = Self.padFirstDimToAtLeast(
                sortedConic,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedOpacityPadded = Self.padFirstDimToAtLeast(
                sortedOpacity,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedColorPadded = Self.padFirstDimToAtLeast(
                sortedColor,
                count: Self.fusedTileCustomMinGaussianCount
            )
            if let blended = renderTileCompositeCustomOp(
                tileCoord: tile_coord,
                sortedDepths: sortedDepthsPadded,
                sortedMeans2d: sortedMeans2dPadded,
                sortedConic: sortedConicPadded,
                sortedOpacity: sortedOpacityPadded,
                sortedColor: sortedColorPadded,
                activeGaussianCount: activeGaussianCount
            ) {
                tile_color = blended.0
                tile_depth = blended.1
                acc_alpha = blended.2
            } else {
                Logger.shared.debug("tile custom kernel unavailable. fallback path is used.")
                let gauss_weight = computeGaussianWeights(
                    tile_coord: tile_coord,
                    sorted_means2d: sortedMeans2d,
                    sorted_conic: sortedConic
                )
                let alpha =
                    gauss_weight[.ellipsis, .newAxis]
                    * MLX.clip(
                        sortedOpacity[.newAxis],
                        max: 0.99
                    )
                let beforeT = MLX.concatenated(
                    [
                        MLX.ones(like: alpha[0..., .stride(to: 1)]),
                        1 - alpha[0..., .stride(to: -1)],
                    ],
                    axis: 1
                )
                let T = beforeT.cumprod(axis: 1)
                let acc_alphaFallback = (alpha * T).sum(axis: 1)
                tile_color =
                    (T * alpha * sortedColor[.newAxis]).sum(
                        axis: 1
                    ) + (1 - acc_alphaFallback) * (self.whiteBackground ? 1 : 0)
                tile_depth =
                    ((T * alpha)
                    * sortedDepths.expandedDimensions(axes: [0, -1])).sum(
                        axis: 1
                    )
                acc_alpha = acc_alphaFallback
            }
        } else {
            Logger.shared.debug("computeGaussianWeights")
            let gauss_weight = computeGaussianWeights(
                tile_coord: tile_coord,
                sorted_means2d: sortedMeans2d,
                sorted_conic: sortedConic
            )
            Logger.shared.debug("renderTile alpha")
            let alpha =
                gauss_weight[.ellipsis, .newAxis]
                * MLX.clip(
                    sortedOpacity[.newAxis],
                    max: 0.99
                )
            Logger.shared.debug("renderTile T")
            let beforeT = MLX.concatenated(
                [
                    MLX.ones(like: alpha[0..., .stride(to: 1)]),
                    1 - alpha[0..., .stride(to: -1)],
                ],
                axis: 1
            )
            let T = beforeT.cumprod(axis: 1)

            Logger.shared.debug("renderTile acc_alpha")
            let acc_alphaFallback = (alpha * T).sum(axis: 1)

            Logger.shared.debug("renderTile tile_color")
            tile_color =
                (T * alpha * sortedColor[.newAxis]).sum(
                    axis: 1
                ) + (1 - acc_alphaFallback) * (self.whiteBackground ? 1 : 0)

            Logger.shared.debug("renderTile tile_depth")
            tile_depth =
                ((T * alpha)
                * sortedDepths.expandedDimensions(axes: [0, -1])).sum(
                    axis: 1
                )
            acc_alpha = acc_alphaFallback
        }
        return (
            tile_color.reshaped([tileSizeRest.h, tileSizeRest.w, -1]),
            tile_depth.reshaped([tileSizeRest.h, tileSizeRest.w, -1]),
            acc_alpha.reshaped([tileSizeRest.h, tileSizeRest.w, -1])
        )
    }

    func render(
        camera: Camera,
        means2d: MLXArray,
        cov2d: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray,
        conic: MLXArray? = nil,
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
        conic: MLXArray? = nil,
        inputIsDepthSorted: Bool = false
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visiility_filter: MLXArray,
        radii: MLXArray
    ) {
        precondition(
            imageWidth == self.W && imageHeight == self.H,
            "Renderer image size mismatch: expected (\(self.W), \(self.H)), got (\(imageWidth), \(imageHeight))"
        )
        Logger.shared.debug("get_radius")
        let radii = get_radius(cov2d: cov2d)
        Logger.shared.debug("get_rect")
        let rect = get_rect(
            pix_coord: means2d,
            radii: radii,
            width: imageWidth,
            height: imageHeight
        )
        let conicValues = conic ?? matrixInverse2d(cov2d)
        let packedGaussians = buildPackedGaussians(
            means2d: means2d,
            conic: conicValues,
            color: color,
            opacity: opacity,
            depths: depths
        )

        let globalTileSliceInfo: GlobalTileSliceInfo?
        if inputIsDepthSorted && globalTileSliceKernelsAvailable {
            globalTileSliceInfo = buildGlobalTileSliceInfo(
                rect: rect,
                radii: radii,
                depths: depths
            )
        } else {
            globalTileSliceInfo = nil
        }

        let render_color = MLXArray.ones(self.pix_coord.shape[0..<2] + [3])
        let render_depth = MLXArray.zeros(self.pix_coord.shape[0..<2] + [1])
        let render_alpha = MLXArray.zeros(self.pix_coord.shape[0..<2] + [1])
        for (tileIndex, tile) in tileEntries.enumerated() {
            Logger.shared.debug("before renderTile")
            let precomputedIndices: MLXArray?
            if let globalTileSliceInfo {
                let rangeBase = tileIndex * 2
                if rangeBase + 1 < globalTileSliceInfo.tileRanges.count {
                    let start = Int(globalTileSliceInfo.tileRanges[rangeBase])
                    let end = Int(globalTileSliceInfo.tileRanges[rangeBase + 1])
                    if end > start {
                        precomputedIndices = globalTileSliceInfo.sortedGaussIdx[start..<end].asType(.int32)
                    } else {
                        precomputedIndices = Self.emptyInt32Indices
                    }
                } else {
                    precomputedIndices = nil
                }
            } else {
                precomputedIndices = nil
            }
            let (tile_color, tile_depth, acc_alpha) = renderTile(
                tile: tile,
                packedGaussians: packedGaussians,
                inputIsDepthSorted: inputIsDepthSorted,
                rect: rect,
                precomputedIndices: precomputedIndices
            )
            Logger.shared.debug("after renderTile")
            Logger.shared.debug("before assign")
            render_color[
                .stride(from: tile.h, to: tile.h + tile.size.h),
                .stride(from: tile.w, to: tile.w + tile.size.w)
            ] = tile_color
            render_depth[
                .stride(from: tile.h, to: tile.h + tile.size.h),
                .stride(from: tile.w, to: tile.w + tile.size.w)
            ] = tile_depth
            render_alpha[
                .stride(from: tile.h, to: tile.h + tile.size.h),
                .stride(from: tile.w, to: tile.w + tile.size.w)
            ] = acc_alpha
            Logger.shared.debug("after assign")
        }

        return (render_color, render_depth, render_alpha, radii .> 0, radii)
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
        Logger.shared.debug("projection_ndc")
        var (mean_ndc, mean_view, in_mask) = projection_ndc(
            points: means3d,
            viewMatrix: viewMatrix,
            projMatrix: projMatrix
        )
        mean_ndc = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        let depthsUnsorted = mean_view[0..., 2]
        let depthSortIndices = MLX.stopGradient(MLX.argSort(depthsUnsorted))
        let depths = depthsUnsorted[depthSortIndices]
        mean_ndc = mean_ndc[depthSortIndices]
        let means3d = means3d[in_mask][depthSortIndices]
        let shs = shs[in_mask][depthSortIndices]
        let opacity = opacity[in_mask][depthSortIndices]
        let scales = scales[in_mask][depthSortIndices]
        let rotations = rotations[in_mask][depthSortIndices]
        let imageWidthArray = MLXArray(Float(imageWidth))
        let imageHeightArray = MLXArray(Float(imageHeight))

        let screenSpace: [MLXArray]
        if let fusedScreenSpaceCustomFunction {
            Logger.shared.debug("build_screen_space_fused")
            screenSpace = fusedScreenSpaceCustomFunction(
                [
                    mean_ndc, scales, rotations, means3d, shs, cameraCenter, viewMatrix,
                    fovX, fovY, focalX, focalY, imageWidthArray, imageHeightArray,
                ]
            )
        } else {
            Logger.shared.debug("build_covariance_3d")
            let cov3d = buildCovariance3d(scales: scales, rotations: rotations)
            if let screenSpaceCustomFunction {
                Logger.shared.debug("build_screen_space_custom")
                screenSpace = screenSpaceCustomFunction(
                    [
                        mean_ndc, means3d, shs, cov3d, cameraCenter, viewMatrix,
                        fovX, fovY, focalX, focalY, imageWidthArray, imageHeightArray,
                    ]
                )
            } else {
                Logger.shared.debug("build_screen_space_fallback")
                screenSpace = buildScreenSpaceFallback(
                    meanNdc: mean_ndc,
                    means3d: means3d,
                    shs: shs,
                    cov3d: cov3d,
                    cameraCenter: cameraCenter,
                    viewMatrix: viewMatrix,
                    fovX: fovX,
                    fovY: fovY,
                    focalX: focalX,
                    focalY: focalY,
                    imageWidth: imageWidthArray,
                    imageHeight: imageHeightArray
                )
            }
        }

        let means2d = screenSpace[ScreenSpaceCustomOutputIndex.means2d]
        let cov2d = screenSpace[ScreenSpaceCustomOutputIndex.cov2d]
        let color = screenSpace[ScreenSpaceCustomOutputIndex.color]
        let conic = screenSpace[ScreenSpaceCustomOutputIndex.conic]
        Logger.shared.debug("render")
        let rets = render(
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            conic: conic,
            inputIsDepthSorted: true
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
        Logger.shared.debug("projection_ndc")
        var (mean_ndc, mean_view, in_mask) = projection_ndc(
            points: means3d,
            viewMatrix: camera.worldViewTransform,
            projMatrix: camera.projectionMatrix
        )
        mean_ndc = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        let depthsUnsorted = mean_view[0..., 2]
        let depthSortIndices = MLX.stopGradient(MLX.argSort(depthsUnsorted))
        let depths = depthsUnsorted[depthSortIndices]
        mean_ndc = mean_ndc[depthSortIndices]
        let means3d = means3d[in_mask][depthSortIndices]
        let shs = shs[in_mask][depthSortIndices]
        let opacity = opacity[in_mask][depthSortIndices]
        let scales = scales[in_mask][depthSortIndices]
        let rotations = rotations[in_mask][depthSortIndices]
        let cameraCenter = MLXArray(
            [Float(camera.cameraCenter.x), Float(camera.cameraCenter.y), Float(camera.cameraCenter.z)]
                as [Float])[.newAxis, .ellipsis]
        let imageWidth = MLXArray(Float(camera.imageWidth))
        let imageHeight = MLXArray(Float(camera.imageHeight))

        let screenSpace: [MLXArray]
        if let fusedScreenSpaceCustomFunction {
            Logger.shared.debug("build_screen_space_fused")
            screenSpace = fusedScreenSpaceCustomFunction(
                [
                    mean_ndc, scales, rotations, means3d, shs, cameraCenter,
                    camera.worldViewTransform, camera.FoVx, camera.FoVy,
                    camera.focalX, camera.focalY, imageWidth, imageHeight,
                ]
            )
        } else {
            Logger.shared.debug("build_covariance_3d")
            let cov3d = buildCovariance3d(scales: scales, rotations: rotations)
            if let screenSpaceCustomFunction {
                Logger.shared.debug("build_screen_space_custom")
                screenSpace = screenSpaceCustomFunction(
                    [
                        mean_ndc, means3d, shs, cov3d, cameraCenter, camera.worldViewTransform,
                        camera.FoVx, camera.FoVy, camera.focalX, camera.focalY, imageWidth, imageHeight,
                    ]
                )
            } else {
                Logger.shared.debug("build_screen_space_fallback")
                screenSpace = buildScreenSpaceFallback(
                    camera: camera,
                    meanNdc: mean_ndc,
                    means3d: means3d,
                    shs: shs,
                    cov3d: cov3d
                )
            }
        }

        let means2d = screenSpace[ScreenSpaceCustomOutputIndex.means2d]
        let cov2d = screenSpace[ScreenSpaceCustomOutputIndex.cov2d]
        let color = screenSpace[ScreenSpaceCustomOutputIndex.color]
        let conic = screenSpace[ScreenSpaceCustomOutputIndex.conic]
        Logger.shared.debug("render")
        let rets = render(
            camera: camera,
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            conic: conic,
            inputIsDepthSorted: true
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
