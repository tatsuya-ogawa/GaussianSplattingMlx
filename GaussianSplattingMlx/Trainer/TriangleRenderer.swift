import Foundation
import MLX

final class TriangleRenderer {
    private let gaussianRenderer: GaussianRenderer

    init(gaussianRenderer: GaussianRenderer) {
        self.gaussianRenderer = gaussianRenderer
    }

    func forward(
        camera: Camera,
        vertices: MLXArray,
        featuresDC: MLXArray,
        featuresRest: MLXArray,
        opacityLogits: MLXArray
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visibility: MLXArray,
        radii: MLXArray
    ) {
        guard vertices.shape.count == 3 else {
            fatalError("Triangle vertices must have shape [N, 3, 3]")
        }
        let centers = MLX.mean(vertices, axes: [1])
        let normals = normalizedNormals(from: vertices)
        let scales = areaScales(from: vertices)
        let quaternions = quaternionAligningZ(to: normals)
        let shs = gaussianRenderer.get_features_from(featuresDC, featuresRest)
        let opacity = gaussianRenderer.get_opacity_from(opacityLogits)
        return gaussianRenderer.forward(
            camera: camera,
            means3d: centers,
            shs: shs,
            opacity: opacity,
            scales: scales,
            rotations: quaternions
        )
    }
}

private func normalizedNormals(from vertices: MLXArray) -> MLXArray {
    let v0 = vertices[.ellipsis, 0, 0...]
    let v1 = vertices[.ellipsis, 1, 0...]
    let v2 = vertices[.ellipsis, 2, 0...]
    let edge01 = v1 - v0
    let edge02 = v2 - v0
    var normals = cross(edge01, edge02)
    let norm = MLX.sqrt(MLX.sum(MLX.square(normals), axes: [1], keepDims: true))
    normals = normals / (norm + 1e-6)
    return normals
}

private func areaScales(from vertices: MLXArray) -> MLXArray {
    let v0 = vertices[.ellipsis, 0, 0...]
    let v1 = vertices[.ellipsis, 1, 0...]
    let v2 = vertices[.ellipsis, 2, 0...]
    let crossVec = cross(v1 - v0, v2 - v0)
    let area = 0.5 * MLX.sqrt(MLX.sum(MLX.square(crossVec), axes: [1], keepDims: true))
    let clamped = MLX.maximum(area, 1e-4)
    return MLX.repeated(clamped, count: 3, axis: 1)
}

private func cross(_ lhs: MLXArray, _ rhs: MLXArray) -> MLXArray {
    let result = MLXArray.zeros(lhs.shape)
    result[.ellipsis, 0] = lhs[.ellipsis, 1] * rhs[.ellipsis, 2] - lhs[.ellipsis, 2] * rhs[.ellipsis, 1]
    result[.ellipsis, 1] = lhs[.ellipsis, 2] * rhs[.ellipsis, 0] - lhs[.ellipsis, 0] * rhs[.ellipsis, 2]
    result[.ellipsis, 2] = lhs[.ellipsis, 0] * rhs[.ellipsis, 1] - lhs[.ellipsis, 1] * rhs[.ellipsis, 0]
    return result
}

private func quaternionAligningZ(to normals: MLXArray) -> MLXArray {
    let count = normals.shape[0]
    let ref = MLXArray([0.0 as Float, 0.0, 1.0])[.newAxis, .ellipsis]
    let refDirs = MLX.repeated(ref, count: count, axis: 0)
    let dot = MLX.clip(MLX.sum(refDirs * normals, axes: [1]), min: -1.0, max: 1.0)
    let halfAngle = 0.5 * MLX.acos(dot)
    var axis = cross(refDirs, normals)
    let axisNorm = MLX.sqrt(MLX.sum(MLX.square(axis), axes: [1], keepDims: true))
    axis = axis / (axisNorm + 1e-6)
    let sinHalf = MLX.sin(halfAngle)
    var quaternion = MLXArray.zeros([count, 4])
    quaternion[.ellipsis, 0] = MLX.cos(halfAngle)
    quaternion[.ellipsis, 1] = axis[.ellipsis, 0] * sinHalf
    quaternion[.ellipsis, 2] = axis[.ellipsis, 1] * sinHalf
    quaternion[.ellipsis, 3] = axis[.ellipsis, 2] * sinHalf

    let smallMask = (axisNorm.reshaped([count]) .< 1e-4).expandedDimensions(axes: [1])
    var identity = MLXArray.zeros([count, 4])
    identity[.ellipsis, 0] = MLXArray(1.0 as Float)
    quaternion = MLX.where(smallMask, identity, quaternion)
    return quaternion
}
