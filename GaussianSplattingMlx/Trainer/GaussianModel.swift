//
//  GaussModel.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import Foundation
import MLX

func distTopK(_ X: MLXArray, k: Int) -> MLXArray {
    // Compute average distance to k nearest neighbors (squared distance root applied later outside caller if needed)
    let N = X.shape[0]
    if N == 0 { return MLXArray.zeros([0]) }
    let chunkSize: Int = 1 << 8
    var averageDist2 = MLXArray.zeros(like: X[.ellipsis, 0]) // shape [N]
    // Pre-expand once to avoid inside-loop broadcasting overhead
    let X2 = X.expandedDimensions(axes: [0])  // [1,N,3]
    for i in stride(from: 0, to: N, by: chunkSize) {
        let end = Swift.min(i + chunkSize, N)
        let batch = X[i..<end].expandedDimensions(axes: [1]) // [B,1,3]
        // diff: [B,N,3]
        let diff = batch - X2
        let dist2 = MLX.sum(MLX.square(diff), axes: [-1]) // [B,N]
        // Use negative to leverage top (largest) -> smallest distances selection
        let negTop = MLX.top(-dist2, k: k, axis: -1) // [B,k] (negative values)
        var mean = -MLX.mean(negTop, axes: [-1]) // back to positive
        mean = MLX.stopGradient(mean)
        eval(mean)
        averageDist2[i..<end] = mean
        MLX.GPU.clearCache()
    }
    return averageDist2
}

class GaussModel {
    var max_sh_degree: Int
    var _xyz: MLXArray
    var _features_dc: MLXArray
    var _features_rest: MLXArray
    var _scales: MLXArray
    var _rotation: MLXArray
    var _opacity: MLXArray

    // Activation functions
    let scaling_inverse_activation = MLX.log
    let covariance_activation: (MLXArray, Double, MLXArray) -> MLXArray

    func getParams() -> [MLXArray] {
        return [
            _xyz,
            _features_dc,
            _features_rest,
            _scales,
            _rotation,
            _opacity,
        ]
    }
    func getLearningRates(current: Int, total: Int) -> [Float] {
        return [
            0.00016 * Swift.max((1.0 - Float(current) / Float(total)), 0.01),  // scale 0.00016 to 0.0000016
            0.0025,
            0.0025 / 20,
            0.005,
            0.001,
            0.025,
        ]
    }

    init(sh_degree: Int = 3, debug: Bool = false) {
        self.max_sh_degree = sh_degree
        self._xyz = MLXArray.zeros([0, 3])
        self._features_dc = MLXArray.zeros([0, 3, 1])
        self._features_rest = MLXArray([
            0, 3, (sh_degree + 1) * (sh_degree + 1) - 1,
        ])
        self._scales = MLXArray.zeros([0, 3])
        self._rotation = MLXArray.zeros([0, 4])
        self._opacity = MLXArray.zeros([0, 1])
        self.covariance_activation = { scale, scaling_modifier, rotation in
            let L = build_scaling_rotation(
                s: scale * scaling_modifier,
                r: rotation
            )
            let cov = L.matmul(L.transposed(0, 2, 1))
            return strip_symmetric(cov)
        }
    }

    static func create_from_pcd(
        pcd: PointCloud,
        sh_degree: Int = 3,
        debug: Bool = false
    ) -> GaussModel {
        let model = GaussModel(sh_degree: sh_degree, debug: debug)
        let points = pcd.coords
        let colors = pcd.select_channels(channel_names: ["R", "G", "B"]) / 255.0

        let fused_point_cloud = points
        let fused_color = RGB2SH(rgb: colors)

        let num_pts = fused_point_cloud.shape[0]
        let num_coeff = (sh_degree + 1) * (sh_degree + 1)
        let features = MLXArray.zeros([num_pts, 3, num_coeff])
        features[.ellipsis, 0..<3, 0] = fused_color

        let dist2 = MLX.maximum(distTopK(fused_point_cloud, k: 3), 1e-7)
        var scales = MLX.log(MLX.sqrt(dist2)).reshaped([num_pts, 1])
        scales = MLX.repeated(scales, count: 3, axis: 1)
        let rots = MLXArray.zeros([num_pts, 4])
        rots[.ellipsis, 0] = MLXArray(1.0)

        let opacities = inverse_sigmoid(
            x: 0.1 * MLXArray.ones([fused_point_cloud.shape[0], 1])
        )
        model._xyz = detachedArray(fused_point_cloud)
        model._features_dc = detachedArray(features[.ellipsis, 0..<1].transposed(0, 2, 1))
        model._features_rest = detachedArray(features[.ellipsis, 1...].transposed(0, 2, 1))
        model._scales = detachedArray(scales)
        model._rotation = detachedArray(rots)
        model._opacity = detachedArray(opacities)

        return model
    }
}
