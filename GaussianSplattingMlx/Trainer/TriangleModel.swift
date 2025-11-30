import Foundation
import MLX

struct TrianglePseudoGaussian {
    let means: MLXArray
    let shs: MLXArray
    let scales: MLXArray
    let rotations: MLXArray
    let opacity: MLXArray
}

final class TriangleModel {
    enum ParamIndex: Int {
        case vertices = 0
        case featuresDC = 1
        case featuresRest = 2
        case vertexWeights = 3
        case opacity = 4
    }

    let maxShDegree: Int
    private(set) var activeShDegree: Int
    private(set) var vertices: MLXArray
    private(set) var featuresDC: MLXArray
    private(set) var featuresRest: MLXArray
    private(set) var vertexWeights: MLXArray
    private(set) var opacityLogits: MLXArray

    init(shDegree: Int = 3) {
        self.maxShDegree = shDegree
        self.activeShDegree = min(1, shDegree)
        self.vertices = MLXArray.zeros([0, 3, 3])
        self.featuresDC = MLXArray.zeros([0, 3, 1])
        self.featuresRest = MLXArray.zeros([
            0,
            3,
            max(1, (shDegree + 1) * (shDegree + 1) - 1),
        ])
        self.vertexWeights = MLXArray.zeros([0, 3])
        self.opacityLogits = MLXArray.zeros([0, 1])
    }

    var triangleCount: Int { vertices.shape.isEmpty ? 0 : vertices.shape[0] }

    func getParams() -> [MLXArray] {
        return [vertices, featuresDC, featuresRest, vertexWeights, opacityLogits]
    }

    func syncFromParams(_ params: [MLXArray]) {
        guard params.count == 5 else { return }
        vertices = params[ParamIndex.vertices.rawValue]
        featuresDC = params[ParamIndex.featuresDC.rawValue]
        featuresRest = params[ParamIndex.featuresRest.rawValue]
        vertexWeights = params[ParamIndex.vertexWeights.rawValue]
        opacityLogits = params[ParamIndex.opacity.rawValue]
    }

    func getLearningRates(current: Int, total: Int) -> [Float] {
        let t = Float(current) / Float(max(total, 1))
        let decay = max(0.1, 1.0 - t)
        return [
            5e-4 * decay,    // vertices
            3e-3,            // features dc
            3e-3 / 20.0,     // features rest
            1e-3,            // vertex weights
            5e-3             // opacity logits
        ]
    }

    func promoteShDegree(iteration: Int, step: Int = 1000) {
        guard activeShDegree < maxShDegree else { return }
        if iteration % step == 0 {
            activeShDegree = min(maxShDegree, activeShDegree + 1)
        }
    }

    func pseudoGaussianParams(vertices: MLXArray? = nil,
                               featuresDC: MLXArray? = nil,
                               featuresRest: MLXArray? = nil,
                               opacityLogits: MLXArray? = nil) -> TrianglePseudoGaussian {
        let verts = vertices ?? self.vertices
        let dc = featuresDC ?? self.featuresDC
        let rest = featuresRest ?? self.featuresRest
        let logits = opacityLogits ?? self.opacityLogits

        let centers = MLX.mean(verts, axes: [1])
        let normals = normalizedNormals(from: verts)
        let scales = areaScales(from: verts)
        let quaternions = quaternionAligningZ(to: normals)
        let shs = MLX.concatenated([dc, rest], axis: 1)
        return TrianglePseudoGaussian(
            means: centers,
            shs: shs,
            scales: scales,
            rotations: quaternions,
            opacity: logits
        )
    }

    static func createFrom(pointCloud: PointCloud,
                           triangleBudget: Int,
                           shDegree: Int) -> TriangleModel {
        let model = TriangleModel(shDegree: shDegree)
        let totalPoints = pointCloud.coords.shape[0]
        let clampedBudget = max(1, min(triangleBudget, max(1, totalPoints / 3)))
        let sampleCount = clampedBudget * 3
        let sampledCloud = pointCloud.randomSample(sampleCount)
        var coords = sampledCloud.coords
        if coords.shape[0] > sampleCount {
            coords = coords[0..<sampleCount]
        }
        let reshaped = coords.reshaped([clampedBudget, 3, 3])
        let colors = sampledCloud.select_channels(channel_names: ["R", "G", "B"]).asType(.float32) / 255.0
        var reshapedColor = colors
        if reshapedColor.shape[0] > sampleCount {
            reshapedColor = reshapedColor[0..<sampleCount]
        }
        let triColors = reshapedColor.reshaped([clampedBudget, 3, 3])
        let averageColor = MLX.mean(triColors, axes: [1])
        let shColor = RGB2SH(rgb: averageColor)

        let numCoeff = (shDegree + 1) * (shDegree + 1)
        var features = MLXArray.zeros([clampedBudget, 3, numCoeff])
        features[.ellipsis, 0..<3, 0] = shColor
        let featuresDC = detachedArray(features[.ellipsis, 0..<1].transposed(0, 2, 1))
        let featuresRest = detachedArray(features[.ellipsis, 1...].transposed(0, 2, 1))

        let vertexWeights = detachedArray(MLXArray.ones([clampedBudget, 3]) * 0.5)
        let opacity = inverse_sigmoid(x: 0.1 * MLXArray.ones([clampedBudget, 1]))
        model.vertices = detachedArray(reshaped)
        model.featuresDC = featuresDC
        model.featuresRest = featuresRest
        model.vertexWeights = vertexWeights
        model.opacityLogits = detachedArray(opacity)
        return model
    }
}

private func normalizedNormals(from vertices: MLXArray) -> MLXArray {
    precondition(vertices.shape.count == 3, "vertices must be [N, 3, 3]")
    let v0 = vertices[.ellipsis, 0, 0...]
    let v1 = vertices[.ellipsis, 1, 0...]
    let v2 = vertices[.ellipsis, 2, 0...]
    let edge01 = v1 - v0
    let edge02 = v2 - v0
    var normals = cross(edge01, edge02)
    normals = normalizeVectors(normals)
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

private func normalizeVectors(_ vectors: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let norm = MLX.sqrt(MLX.sum(MLX.square(vectors), axes: [1], keepDims: true))
    return vectors / (norm + eps)
}

private func quaternionAligningZ(to normals: MLXArray) -> MLXArray {
    let count = normals.shape[0]
    let ref = MLXArray([0.0 as Float, 0.0, 1.0])[.newAxis, .ellipsis]
    let refDirs = MLX.repeated(ref, count: count, axis: 0)
    let dot = MLX.clip(MLX.sum(refDirs * normals, axes: [1]), min: -1.0, max: 1.0)
    let halfAngle = 0.5 * MLX.acos(dot)
    var axis = cross(refDirs, normals)
    axis = normalizeVectors(axis)
    let sinHalf = MLX.sin(halfAngle)
    var quaternion = MLXArray.zeros([count, 4])
    quaternion[.ellipsis, 0] = MLX.cos(halfAngle)
    quaternion[.ellipsis, 1] = axis[.ellipsis, 0] * sinHalf
    quaternion[.ellipsis, 2] = axis[.ellipsis, 1] * sinHalf
    quaternion[.ellipsis, 3] = axis[.ellipsis, 2] * sinHalf

    let axisNorm = MLX.sqrt(MLX.sum(MLX.square(axis), axes: [1]))
    let smallMask = (axisNorm .< 1e-4).expandedDimensions(axes: [1])
    var identity = MLXArray.zeros([count, 4])
    identity[.ellipsis, 0] = MLXArray(1.0 as Float)
    quaternion = MLX.where(smallMask, identity, quaternion)
    return quaternion
}
