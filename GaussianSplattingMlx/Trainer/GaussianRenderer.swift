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
    in_mask: MLXArray
) -> (
    depths: MLXArray,
    means2d: MLXArray,
    cov2d: MLXArray,
    conic: MLXArray,
    opacity: MLXArray,
    color: MLXArray
) {
    let sorted_depth_indices = MLX.stopGradient(MLX.argSort(depths[in_mask]))
    let sorted_depths = depths[in_mask][sorted_depth_indices]
    let sorted_means2d = means2d[in_mask][sorted_depth_indices]
    let sorted_cov2d = cov2d[in_mask][sorted_depth_indices]
    let sorted_conic = matrixInverse2d(sorted_cov2d)
    let sorted_opacity = opacity[in_mask][sorted_depth_indices]
    let sorted_color = color[in_mask][sorted_depth_indices]

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
class GaussianRenderer {
    let debug: Bool
    let active_sh_degree: Int
    let W: Int
    let H: Int
    let pix_coord: MLXArray
    let whiteBackground: Bool
    let TILE_SIZE: TILE_SIZE_H_W

    init(
        active_sh_degree: Int,
        W: Int,
        H: Int,
        TILE_SIZE: TILE_SIZE_H_W,
        whiteBackground: Bool
    ) {
        self.active_sh_degree = active_sh_degree
        self.debug = false
        self.whiteBackground = whiteBackground
        self.TILE_SIZE = TILE_SIZE
        self.W = W
        self.H = H
        self.pix_coord = createMeshGrid(shape: [H, W])
    }

    func renderTile(
        h: Int,
        w: Int,
        tileSize: TILE_SIZE_H_W,
        means2d: MLXArray,
        cov2d: MLXArray,
        color: MLXArray,
        opacity: MLXArray,
        depths: MLXArray,
        rect: (MLXArray, MLXArray),
        skipThreshold: Int = 0
    ) -> (MLXArray, MLXArray, MLXArray) {
        let tileSizeRest = TILE_SIZE_H_W(
            w: Swift.min(self.W - w, tileSize.w),
            h: Swift.min(self.H - h, tileSize.h)
        )
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
        let tile_coord = MLX.stopGradient(
            self.pix_coord[
                .stride(from: h, to: h + tileSizeRest.h),
                .stride(from: w, to: w + tileSizeRest.w)
            ].flattened(start: 0, end: -2)
        )
        Logger.shared.debug("getSortedValues")
        let sortedValues = getSortedValues(
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            in_mask: in_mask
        )

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
        let acc_alpha = (alpha * T).sum(axis: 1)

        Logger.shared.debug("renderTile tile_color")
        let tile_color =
            (T * alpha * sortedValues.color[.newAxis]).sum(
                axis: 1
            ) + (1 - acc_alpha) * (self.whiteBackground ? 1 : 0)

        Logger.shared.debug("renderTile tile_depth")
        let tile_depth =
            ((T * alpha)
            * sortedValues.depths.expandedDimensions(axes: [0, -1])).sum(
                axis: 1
            )
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
        depths: MLXArray
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visiility_filter: MLXArray,
        radii: MLXArray
    ) {
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
        for h in stride(from: 0, to: camera.imageHeight, by: TILE_SIZE.h) {
            for w in stride(from: 0, to: camera.imageWidth, by: TILE_SIZE.w) {
                Logger.shared.debug("before renderTile")
                let (tile_color, tile_depth, acc_alpha) = renderTile(
                    h: h,
                    w: w,
                    tileSize: TILE_SIZE,
                    means2d: means2d,
                    cov2d: cov2d,
                    color: color,
                    opacity: opacity,
                    depths: depths,
                    rect: rect
                )
                Logger.shared.debug("after renderTile")
                Logger.shared.debug("before assign")
                render_color[
                    .stride(from: h, to: h + TILE_SIZE.h),
                    .stride(from: w, to: w + TILE_SIZE.w)
                ] = tile_color
                render_depth[
                    .stride(from: h, to: h + TILE_SIZE.h),
                    .stride(from: w, to: w + TILE_SIZE.w)
                ] = tile_depth
                render_alpha[
                    .stride(from: h, to: h + TILE_SIZE.h),
                    .stride(from: w, to: w + TILE_SIZE.w)
                ] = acc_alpha
                Logger.shared.debug("after assign")
            }
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
        let depths = mean_view[0..., 2]
        let means3d = means3d[in_mask]
        let shs = shs[in_mask]
        let opacity = opacity[in_mask]
        let scales = scales[in_mask]
        let rotations = rotations[in_mask]
        Logger.shared.debug("build_color")
        let color = build_color(
            means3d: means3d,
            shs: shs,
            camera: camera,
            activeShDegree: self.active_sh_degree
        )
        Logger.shared.debug("build_covariance_3d")
        let cov3d = build_covariance_3d(s: scales, r: rotations)
        Logger.shared.debug("build_covariance_2d")
        let cov2d = build_covariance_2d(
            mean3d: means3d,
            cov3d: cov3d,
            viewMatrix: camera.worldViewTransform,
            fovX: camera.FoVx,
            fovY: camera.FoVy,
            focalX: camera.focalX,
            focalY: camera.focalY
        )
        let mean_coord_x =
            ((mean_ndc[.ellipsis, 0] + 1) * camera.imageWidth - 1.0) * 0.5
        let mean_coord_y =
            ((mean_ndc[.ellipsis, 1] + 1) * camera.imageHeight - 1.0) * 0.5
        let means2d = MLX.stacked([mean_coord_x, mean_coord_y], axis: -1)
        Logger.shared.debug("render")
        let rets = render(
            camera: camera,
            means2d: means2d,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths
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
