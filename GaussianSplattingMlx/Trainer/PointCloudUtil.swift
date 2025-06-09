//
//  PointCloudUtil.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/02.
//
import Foundation
import MLX
import MLXLinalg
import simd

/// since MLX.linalg.inv is on gpu is not supported,use custom implementation
func inv3x3(_ m: MLXArray) -> MLXArray {
  let a = m[.ellipsis, 0, 0]
  let b = m[.ellipsis, 0, 1]
  let c = m[.ellipsis, 0, 2]
  let d = m[.ellipsis, 1, 0]
  let e = m[.ellipsis, 1, 1]
  let f = m[.ellipsis, 1, 2]
  let g = m[.ellipsis, 2, 0]
  let h = m[.ellipsis, 2, 1]
  let i = m[.ellipsis, 2, 2]

  let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
  let det_eps = det + 1e-10

  let m00 = e * i - f * h
  let m01 = -(b * i - c * h)
  let m02 = b * f - c * e
  let m10 = -(d * i - f * g)
  let m11 = a * i - c * g
  let m12 = -(a * f - c * d)
  let m20 = d * h - e * g
  let m21 = -(a * h - b * g)
  let m22 = a * e - b * d

  let inv = MLXArray.zeros(m.shape)
  inv[.ellipsis, 0, 0] = m00 / det_eps
  inv[.ellipsis, 0, 1] = m01 / det_eps
  inv[.ellipsis, 0, 2] = m02 / det_eps
  inv[.ellipsis, 1, 0] = m10 / det_eps
  inv[.ellipsis, 1, 1] = m11 / det_eps
  inv[.ellipsis, 1, 2] = m12 / det_eps
  inv[.ellipsis, 2, 0] = m20 / det_eps
  inv[.ellipsis, 2, 1] = m21 / det_eps
  inv[.ellipsis, 2, 2] = m22 / det_eps
  return inv
}
func getRaysFromImages(
  H: Int,
  W: Int,
  intrinsics: MLXArray,  // [B,4,4]
  c2w: MLXArray,  // [B,4,4]
  renderStride: Int = 1
) -> (MLXArray, MLXArray) {
  Logger.shared.debug("Create mesh grid")
  let u = MLXArray(
    (stride(from: 0, to: W, by: renderStride)).map { Float($0) }
  )
  let v = MLXArray(
    (stride(from: 0, to: H, by: renderStride)).map { Float($0) }
  )
  let grids = MLX.meshGrid([u, v], indexing: .xy)  // returns [u_grid, v_grid]
  let uGrid = grids[0].reshaped([-1])
  let vGrid = grids[1].reshaped([-1])
  let ones = MLXArray(Array(repeating: 1.0 as Float, count: uGrid.shape[0]))
  // stack (u, v, 1) shape: [3, HW]
  let pixels = MLX.stacked([uGrid, vGrid, ones], axis: 0)
  // expand to batch: [B, 3, HW]
  let B = intrinsics.shape[0]
  let batchedPixels = MLX.repeated(
    pixels.expandedDimensions(axes: [0]),
    count: B,
    axis: 0
  )

  Logger.shared.debug("Compute rays inv")
  let invIntr = inv3x3(intrinsics[0..., 0..<3, 0..<3])
  let rot = c2w[0..., 0..<3, 0..<3]
  Logger.shared.debug("Compute rays matmul")
  let rays_d = rot.matmul(invIntr).matmul(batchedPixels).transposed(
    0,
    2,
    1
  )

  Logger.shared.debug("Compute rays rest")
  let rays_o = MLX.repeated(
    c2w[0..., 0..<3, 3].expandedDimensions(axes: [1]),
    count: rays_d.shape[1],
    axis: 1
  )
  return (rays_o, rays_d)
}

func getPointCloudsFromTrainData(
  trainData: TrainData
) -> PointCloud {
  let depths = trainData.depthArray
  let alphas = trainData.alphaArray
  let rgbs = trainData.rgbArray

  let (Hs, Ws, intrinsics, c2ws) = trainData.getCameraParams()
  let W = Ws[0].item(Int.self)
  let H = Hs[0].item(Int.self)
  guard let depths = depths else {
    fatalError("unexpected nil depth")
  }
  precondition(depths.shape == alphas.shape)
  let (rays_o, rays_d) = getRaysFromImages(
    H: H,
    W: W,
    intrinsics: intrinsics,
    c2w: c2ws
  )
  Logger.shared.debug("Calculate point cloud")
  let pts = rays_o + rays_d * depths.reshaped([rays_o.shape[0], -1, 1])
  let a = alphas.expandedDimensions(axes: [-1])
  let rgbas = MLX.concatenated([rgbs, a], axis: -1)
  Logger.shared.debug("Get mask")
  let mask = conditionToIndices(condition: alphas.reshaped([-1]) .== 1)
  Logger.shared.debug("Get masked rgba")
  let rgbasArr = rgbas.reshaped([-1, rgbas.shape.last!])[mask]
  Logger.shared.debug("Get masked coords")
  let coords = pts.reshaped([-1, pts.shape.last!])[mask]
  var chDict: [String: MLXArray] = [:]
  chDict["R"] = rgbasArr[.ellipsis, 0]
  chDict["G"] = rgbasArr[.ellipsis, 1]
  chDict["B"] = rgbasArr[.ellipsis, 2]
  chDict["A"] = rgbasArr[.ellipsis, 3]
  return PointCloud(coords: coords, channels: chDict)
}
class PointCloud {
  let coords: MLXArray
  let channels: [String: MLXArray]
  init(coords: MLXArray, channels: [String: MLXArray]) {
    self.coords = coords
    self.channels = channels
  }
  let COLORS = Set(["R", "G", "B", "A"])
  func preprocess(data: MLXArray, channel: String) -> MLXArray {
    if COLORS.contains(channel) {
      return MLX.round(data * 255.0)
    }
    return data
  }
  func select_channels(channel_names: [String]) -> MLXArray {
    return MLX.stacked(
      channel_names.map { name in
        preprocess(data: self.channels[name]!, channel: name)
      },
      axis: -1
    )
  }
  func randomSample(_ numPoints: Int) -> PointCloud {
    let n = coords.shape[0]
    if n <= numPoints {
      return self
    }
    var indices = Array(0..<n)
    indices.shuffle()
    let pick = Array(indices.prefix(numPoints))
    let pickArr = MLXArray(pick.map { Int32($0) })
    let newCoords = coords[pickArr, 0...]  // [numPoints, 3]
    var newChannels: [String: MLXArray] = [:]
    for (k, arr) in channels {
      newChannels[k] = arr[pickArr]
    }
    return PointCloud(coords: newCoords, channels: newChannels)
  }
}
