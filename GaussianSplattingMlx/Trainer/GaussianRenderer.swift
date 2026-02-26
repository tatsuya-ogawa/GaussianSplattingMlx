import MLX
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

class GaussianRenderer {
    private enum ScreenSpaceCustomOutputIndex {
        static let means2d = 0
        static let cov2d = 1
        static let color = 2
        static let conic = 3
    }

    private static let fusedTileCustomMinGaussianCount = 9
    private static let screenSpaceCustomMinPointCount = 3

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
        camera: Camera,
        meanNdc: MLXArray,
        means3d: MLXArray,
        shs: MLXArray,
        cov3d: MLXArray
    ) -> [MLXArray] {
        let color = build_color(
            means3d: means3d,
            shs: shs,
            camera: camera,
            activeShDegree: self.active_sh_degree
        )
        let cov2d = build_covariance_2d(
            mean3d: means3d,
            cov3d: cov3d,
            viewMatrix: camera.worldViewTransform,
            fovX: camera.FoVx,
            fovY: camera.FoVy,
            focalX: camera.focalX,
            focalY: camera.focalY
        )
        let meanCoordX = ((meanNdc[.ellipsis, 0] + 1) * camera.imageWidth - 1.0) * 0.5
        let meanCoordY = ((meanNdc[.ellipsis, 1] + 1) * camera.imageHeight - 1.0) * 0.5
        let means2d = MLX.stacked([meanCoordX, meanCoordY], axis: -1)
        let conic = matrixInverse2d(cov2d)
        return [means2d, cov2d, color, conic]
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
        means2d: MLXArray,
        cov2d: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray,
        conic: MLXArray? = nil,
        inputIsDepthSorted: Bool = false,
        rect: (MLXArray, MLXArray),
        skipThreshold: Int = 0
    ) -> (MLXArray, MLXArray, MLXArray) {
        let h = tile.h
        let w = tile.w
        let tileSizeRest = tile.size
        Logger.shared.debug("computeTileMask")
        let in_mask_condition = computeTileMask(
            h: h,
            w: w,
            tileSize: tileSizeRest,
            rect: rect
        )
        let in_mask = conditionToIndices(condition: in_mask_condition)
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
        Logger.shared.debug("getSortedValues")
        let sortedValues = getSortedValues(
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            conic: conic,
            in_mask: in_mask,
            inputIsDepthSorted: inputIsDepthSorted
        )

        let tile_color: MLXArray
        let tile_depth: MLXArray
        let acc_alpha: MLXArray
        if useFusedTileCustomOp {
            let activeGaussianCount = sortedValues.depths.shape[0]
            let sortedDepthsPadded = Self.padFirstDimToAtLeast(
                sortedValues.depths,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedMeans2dPadded = Self.padFirstDimToAtLeast(
                sortedValues.means2d,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedConicPadded = Self.padFirstDimToAtLeast(
                sortedValues.conic,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedOpacityPadded = Self.padFirstDimToAtLeast(
                sortedValues.opacity,
                count: Self.fusedTileCustomMinGaussianCount
            )
            let sortedColorPadded = Self.padFirstDimToAtLeast(
                sortedValues.color,
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
                    sorted_means2d: sortedValues.means2d,
                    sorted_conic: sortedValues.conic
                )
                let alpha =
                    gauss_weight[.ellipsis, .newAxis]
                    * MLX.clip(
                        sortedValues.opacity[.newAxis],
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
                    (T * alpha * sortedValues.color[.newAxis]).sum(
                        axis: 1
                    ) + (1 - acc_alphaFallback) * (self.whiteBackground ? 1 : 0)
                tile_depth =
                    ((T * alpha)
                    * sortedValues.depths.expandedDimensions(axes: [0, -1])).sum(
                        axis: 1
                    )
                acc_alpha = acc_alphaFallback
            }
        } else {
            Logger.shared.debug("computeGaussianWeights")
            let gauss_weight = computeGaussianWeights(
                tile_coord: tile_coord,
                sorted_means2d: sortedValues.means2d,
                sorted_conic: sortedValues.conic
            )
            Logger.shared.debug("renderTile alpha")
            let alpha =
                gauss_weight[.ellipsis, .newAxis]
                * MLX.clip(
                    sortedValues.opacity[.newAxis],
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
                (T * alpha * sortedValues.color[.newAxis]).sum(
                    axis: 1
                ) + (1 - acc_alphaFallback) * (self.whiteBackground ? 1 : 0)

            Logger.shared.debug("renderTile tile_depth")
            tile_depth =
                ((T * alpha)
                * sortedValues.depths.expandedDimensions(axes: [0, -1])).sum(
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
        precondition(
            camera.imageWidth == self.W && camera.imageHeight == self.H,
            "Renderer image size mismatch: expected (\(self.W), \(self.H)), got (\(camera.imageWidth), \(camera.imageHeight))"
        )
        Logger.shared.debug("get_radius")
        let radii = get_radius(cov2d: cov2d)
        Logger.shared.debug("get_rect")
        let rect = get_rect(
            pix_coord: means2d,
            radii: radii,
            width: camera.imageWidth,
            height: camera.imageHeight
        )

        let render_color = MLXArray.ones(self.pix_coord.shape[0..<2] + [3])
        let render_depth = MLXArray.zeros(self.pix_coord.shape[0..<2] + [1])
        let render_alpha = MLXArray.zeros(self.pix_coord.shape[0..<2] + [1])
        for tile in tileEntries {
            Logger.shared.debug("before renderTile")
            let (tile_color, tile_depth, acc_alpha) = renderTile(
                tile: tile,
                means2d: means2d,
                cov2d: cov2d,
                color: color,
                opacity: opacity,
                depths: depths,
                conic: conic,
                inputIsDepthSorted: inputIsDepthSorted,
                rect: rect
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
        Logger.shared.debug("build_covariance_3d")
        let cov3d = build_covariance_3d(s: scales, r: rotations)
        let cameraCenter = MLXArray(
            [Float(camera.cameraCenter.x), Float(camera.cameraCenter.y), Float(camera.cameraCenter.z)]
                as [Float])[.newAxis, .ellipsis]
        let imageWidth = MLXArray(Float(camera.imageWidth))
        let imageHeight = MLXArray(Float(camera.imageHeight))

        let screenSpace: [MLXArray]
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
