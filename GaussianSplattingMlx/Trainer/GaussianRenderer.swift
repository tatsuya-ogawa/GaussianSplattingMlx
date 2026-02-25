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

struct TILE_SIZE_H_W {
    let w: Int
    let h: Int
}
class GaussianRenderer {
    private struct TileEntry {
        let h: Int
        let w: Int
        let size: TILE_SIZE_H_W
        let tileCoord: MLXArray
    }

    private struct TileMaskCacheEntry {
        let inMask: MLXArray
        let pointCount: Int
        let refreshedRenderIndex: Int
    }

    private enum ScreenSpaceInputIndex {
        static let meanNdc = 0
        static let means3d = 1
        static let shs = 2
        static let scales = 3
        static let rotations = 4
        static let cameraCenter = 5
        static let viewMatrix = 6
        static let fovX = 7
        static let fovY = 8
        static let focalX = 9
        static let focalY = 10
        static let imageWidth = 11
        static let imageHeight = 12
    }

    private enum ScreenSpaceOutputIndex {
        static let means2d = 0
        static let cov2d = 1
        static let color = 2
        static let conic = 3
    }

    private enum RenderPrepInputIndex {
        static let means2d = 0
        static let cov2d = 1
        static let imageWidth = 2
        static let imageHeight = 3
    }

    private enum RenderPrepOutputIndex {
        static let radii = 0
        static let rectMin = 1
        static let rectMax = 2
    }

    private enum TileBlendInputIndex {
        static let tileCoord = 0
        static let packedGaussians = 1
        static let inMask = 2
    }

    private enum TileBlendOutputIndex {
        static let tileColor = 0
        static let tileDepth = 1
        static let accAlpha = 2
    }

    private enum PackedGaussianIndex {
        static let means2d = 0..<2
        static let conic = 2..<6
        static let color = 6..<9
        static let opacity = 9
        static let depth = 10
    }

    let debug: Bool
    let active_sh_degree: Int
    let W: Int
    let H: Int
    let pix_coord: MLXArray
    private let tileEntries: [TileEntry]
    private var tileMaskCache: [Int: TileMaskCacheEntry] = [:]
    private var renderPassIndex: Int = 0
    let tileMaskCacheReuseInterval: Int
    let whiteBackground: Bool
    let TILE_SIZE: TILE_SIZE_H_W
    private lazy var buildScreenSpaceCompiled: ([MLXArray]) -> [MLXArray] = {
        MLX.compile(shapeless: true) { [active_sh_degree] inputs in
            let meanNdc = inputs[ScreenSpaceInputIndex.meanNdc]
            let means3d = inputs[ScreenSpaceInputIndex.means3d]
            let shs = inputs[ScreenSpaceInputIndex.shs]
            let scales = inputs[ScreenSpaceInputIndex.scales]
            let rotations = inputs[ScreenSpaceInputIndex.rotations]
            let cameraCenter = inputs[ScreenSpaceInputIndex.cameraCenter]
            let viewMatrix = inputs[ScreenSpaceInputIndex.viewMatrix]
            let fovX = inputs[ScreenSpaceInputIndex.fovX]
            let fovY = inputs[ScreenSpaceInputIndex.fovY]
            let focalX = inputs[ScreenSpaceInputIndex.focalX]
            let focalY = inputs[ScreenSpaceInputIndex.focalY]
            let imageWidth = inputs[ScreenSpaceInputIndex.imageWidth]
            let imageHeight = inputs[ScreenSpaceInputIndex.imageHeight]

            let color = build_color(
                means3d: means3d,
                shs: shs,
                cameraCenter: cameraCenter,
                activeShDegree: active_sh_degree
            )
            let cov3d = build_covariance_3d(s: scales, r: rotations)
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
    }()
    private lazy var buildRenderPrepCompiled: ([MLXArray]) -> [MLXArray] = {
        MLX.compile(shapeless: true) { inputs in
            let means2d = inputs[RenderPrepInputIndex.means2d]
            let cov2d = inputs[RenderPrepInputIndex.cov2d]
            let imageWidth = inputs[RenderPrepInputIndex.imageWidth]
            let imageHeight = inputs[RenderPrepInputIndex.imageHeight]

            let radii = get_radius(cov2d: cov2d)
            let rectMinBase = means2d - radii.expandedDimensions(axes: [1])
            let rectMaxBase = means2d + radii.expandedDimensions(axes: [1])

            let rectMin = MLXArray.zeros(like: rectMinBase)
            rectMin[.ellipsis, 0] = MLX.clip(
                rectMinBase[.ellipsis, 0],
                min: 0.0,
                max: imageWidth - 1.0
            )
            rectMin[.ellipsis, 1] = MLX.clip(
                rectMinBase[.ellipsis, 1],
                min: 0.0,
                max: imageHeight - 1.0
            )

            let rectMax = MLXArray.zeros(like: rectMaxBase)
            rectMax[.ellipsis, 0] = MLX.clip(
                rectMaxBase[.ellipsis, 0],
                min: 0.0,
                max: imageWidth - 1.0
            )
            rectMax[.ellipsis, 1] = MLX.clip(
                rectMaxBase[.ellipsis, 1],
                min: 0.0,
                max: imageHeight - 1.0
            )

            return [radii, rectMin, rectMax]
        }
    }()
    private lazy var buildPackedGaussiansCompiled: ([MLXArray]) -> [MLXArray] = {
        MLX.compile(shapeless: true) { inputs in
            let means2d = inputs[0]
            let conic = inputs[1]
            let color = inputs[2]
            let opacity = inputs[3].reshaped([-1, 1])
            let depths = inputs[4].reshaped([-1, 1])
            let conicFlat = conic.reshaped([-1, 4])
            let packed = MLX.concatenated(
                [means2d, conicFlat, color, opacity, depths],
                axis: -1
            )
            return [packed]
        }
    }()
    private lazy var buildTileBlendCompiled: ([MLXArray]) -> [MLXArray] = {
        let background = MLXArray(whiteBackground ? Float(1.0) : Float(0.0))
        return MLX.compile(shapeless: true) { inputs in
            let tileCoord = inputs[TileBlendInputIndex.tileCoord]
            let packedGaussians = inputs[TileBlendInputIndex.packedGaussians]
            let inMask = inputs[TileBlendInputIndex.inMask]

            let sortedPacked = packedGaussians[inMask]
            let sortedMeans2d = sortedPacked[.ellipsis, PackedGaussianIndex.means2d]
            let sortedConic = sortedPacked[.ellipsis, PackedGaussianIndex.conic]
                .reshaped([-1, 2, 2])
            let sortedColor = sortedPacked[.ellipsis, PackedGaussianIndex.color]
            let sortedOpacity = sortedPacked[.ellipsis, PackedGaussianIndex.opacity]
            let sortedDepths = sortedPacked[.ellipsis, PackedGaussianIndex.depth]

            let gaussWeight = computeGaussianWeights(
                tile_coord: tileCoord,
                sorted_means2d: sortedMeans2d,
                sorted_conic: sortedConic
            )
            let alpha =
                gaussWeight[.ellipsis, .newAxis]
                * MLX.clip(
                    sortedOpacity[.newAxis, .ellipsis, .newAxis],
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
            let accAlpha = (alpha * T).sum(axis: 1)
            let tileColor =
                (T * alpha * sortedColor[.newAxis]).sum(axis: 1)
                + (1 - accAlpha) * background
            let tileDepth =
                ((T * alpha)
                * sortedDepths.expandedDimensions(axes: [0, -1])).sum(axis: 1)
            return [tileColor, tileDepth, accAlpha]
        }
    }()

    init(
        active_sh_degree: Int,
        W: Int,
        H: Int,
        TILE_SIZE: TILE_SIZE_H_W,
        whiteBackground: Bool,
        tileMaskCacheReuseInterval: Int = 2
    ) {
        self.active_sh_degree = active_sh_degree
        self.debug = false
        self.whiteBackground = whiteBackground
        self.TILE_SIZE = TILE_SIZE
        self.W = W
        self.H = H
        self.tileMaskCacheReuseInterval = Swift.max(1, tileMaskCacheReuseInterval)
        self.pix_coord = createMeshGrid(shape: [H, W])
        self.tileEntries = GaussianRenderer.makeTileEntries(
            width: W,
            height: H,
            tileSize: TILE_SIZE,
            pixCoord: self.pix_coord
        )
    }

    private func cachedTileMask(
        tileIndex: Int,
        pointCount: Int,
        renderIndex: Int
    ) -> MLXArray? {
        guard tileMaskCacheReuseInterval > 1 else { return nil }
        guard let entry = tileMaskCache[tileIndex] else { return nil }
        guard entry.pointCount == pointCount else {
            tileMaskCache.removeValue(forKey: tileIndex)
            return nil
        }
        let age = renderIndex - entry.refreshedRenderIndex
        guard age > 0 && age < tileMaskCacheReuseInterval else {
            return nil
        }
        return entry.inMask
    }

    private func putTileMask(
        tileIndex: Int,
        pointCount: Int,
        renderIndex: Int,
        inMask: MLXArray
    ) {
        tileMaskCache[tileIndex] = TileMaskCacheEntry(
            inMask: inMask,
            pointCount: pointCount,
            refreshedRenderIndex: renderIndex
        )
    }

    func clearTileMaskCache() {
        tileMaskCache.removeAll(keepingCapacity: true)
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
                )
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

    func renderTile(
        tileSize: TILE_SIZE_H_W,
        tileCoord: MLXArray,
        packedGaussians: MLXArray,
        inMask: MLXArray,
        skipThreshold: Int = 0
    ) -> (MLXArray, MLXArray, MLXArray) {
        if inMask.shape[0] <= skipThreshold {
            Logger.shared.debug("skip tile")
            return (
                self.whiteBackground
                    ? MLXArray.ones([tileSize.h, tileSize.w, 3])
                    : MLXArray.zeros([tileSize.h, tileSize.w, 3]),
                MLXArray.zeros([tileSize.h, tileSize.w, 1]),
                MLXArray.zeros([tileSize.h, tileSize.w, 1])
            )
        }
        Logger.shared.debug("build_tile_blend_compiled")
        let blended = buildTileBlendCompiled([
            tileCoord,
            packedGaussians,
            inMask,
        ])
        let tile_color = blended[TileBlendOutputIndex.tileColor]
        let tile_depth = blended[TileBlendOutputIndex.tileDepth]
        let acc_alpha = blended[TileBlendOutputIndex.accAlpha]
        return (
            tile_color.reshaped([tileSize.h, tileSize.w, -1]),
            tile_depth.reshaped([tileSize.h, tileSize.w, -1]),
            acc_alpha.reshaped([tileSize.h, tileSize.w, -1])
        )
    }

    func render(
        imageWidth: Int,
        imageHeight: Int,
        means2d: MLXArray,
        cov2d: MLXArray,
        conic: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray
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
        Logger.shared.debug("build_render_prep_compiled")
        let renderPrep = buildRenderPrepCompiled([
            means2d,
            cov2d,
            MLXArray(Float(imageWidth)),
            MLXArray(Float(imageHeight)),
        ])
        let radii = renderPrep[RenderPrepOutputIndex.radii]
        let rect = (
            renderPrep[RenderPrepOutputIndex.rectMin],
            renderPrep[RenderPrepOutputIndex.rectMax]
        )
        let packedGaussians = buildPackedGaussiansCompiled([
            means2d,
            conic,
            color,
            opacity,
            depths,
        ])[0]
        let pointCount = means2d.shape[0]
        let renderIndex = renderPassIndex
        renderPassIndex += 1

        let render_color = MLXArray.ones(self.pix_coord.shape[0..<2] + [3])
        let render_depth = MLXArray.zeros(self.pix_coord.shape[0..<2] + [1])
        let render_alpha = MLXArray.zeros(self.pix_coord.shape[0..<2] + [1])
        for (tileIndex, tile) in tileEntries.enumerated() {
            let inMask: MLXArray
            if let cached = cachedTileMask(
                tileIndex: tileIndex,
                pointCount: pointCount,
                renderIndex: renderIndex
            ) {
                inMask = cached
            } else {
                Logger.shared.debug("computeTileMask")
                let in_mask_condition = computeTileMask(
                    h: tile.h,
                    w: tile.w,
                    tileSize: tile.size,
                    rect: rect
                )
                let computed = conditionToIndices(condition: in_mask_condition)
                putTileMask(
                    tileIndex: tileIndex,
                    pointCount: pointCount,
                    renderIndex: renderIndex,
                    inMask: computed
                )
                inMask = computed
            }
            Logger.shared.debug("before renderTile")
            let (tile_color, tile_depth, acc_alpha) = renderTile(
                tileSize: tile.size,
                tileCoord: tile.tileCoord,
                packedGaussians: packedGaussians,
                inMask: inMask
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

    func render(
        camera: Camera,
        means2d: MLXArray,
        cov2d: MLXArray,
        conic: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visiility_filter: MLXArray,
        radii: MLXArray
    ) {
        render(
            imageWidth: camera.imageWidth,
            imageHeight: camera.imageHeight,
            means2d: means2d,
            cov2d: cov2d,
            conic: conic,
            color: color,
            opacity: opacity,
            depths: depths
        )
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
        mean_ndc = mean_ndc[depthSortIndices]
        let depths = depthsUnsorted[depthSortIndices]
        let means3d = means3d[in_mask][depthSortIndices]
        let shs = shs[in_mask][depthSortIndices]
        let opacity = opacity[in_mask][depthSortIndices]
        let scales = scales[in_mask][depthSortIndices]
        let rotations = rotations[in_mask][depthSortIndices]
        Logger.shared.debug("build_screen_space_compiled")
        let screenSpace = buildScreenSpaceCompiled([
            mean_ndc,
            means3d,
            shs,
            scales,
            rotations,
            cameraCenter,
            viewMatrix,
            fovX,
            fovY,
            focalX,
            focalY,
            MLXArray(Float(imageWidth)),
            MLXArray(Float(imageHeight)),
        ])
        let means2d = screenSpace[ScreenSpaceOutputIndex.means2d]
        let cov2d = screenSpace[ScreenSpaceOutputIndex.cov2d]
        let color = screenSpace[ScreenSpaceOutputIndex.color]
        let conic = screenSpace[ScreenSpaceOutputIndex.conic]
        Logger.shared.debug("render")
        let rets = render(
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            means2d: means2d,
            cov2d: cov2d,
            conic: conic,
            color: color,
            opacity: opacity,
            depths: depths
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
        forwardWithCameraParams(
            viewMatrix: camera.worldViewTransform,
            projMatrix: camera.projectionMatrix,
            cameraCenter: MLXArray(
                [Float(camera.cameraCenter.x), Float(camera.cameraCenter.y), Float(camera.cameraCenter.z)]
                    as [Float]
            )[.newAxis, .ellipsis],
            fovX: camera.FoVx,
            fovY: camera.FoVy,
            focalX: camera.focalX,
            focalY: camera.focalY,
            imageWidth: camera.imageWidth,
            imageHeight: camera.imageHeight,
            means3d: means3d,
            shs: shs,
            opacity: opacity,
            scales: scales,
            rotations: rotations
        )
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
