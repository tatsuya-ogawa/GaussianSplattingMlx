//
//  SsimUtils.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/03.
//
import MLX
import MLXNN

func createWindow(windowSize: Int, channel: Int) -> MLXArray {
  let g1d = gaussian(windowSize: windowSize, sigma: 1.5).reshaped([windowSize, 1])
  let g2d = g1d.matmul(g1d.transposed(1, 0)).reshaped([1, windowSize, windowSize, 1])
  // [ch, 1, W, W]
  let window = MLX.broadcast(g2d, to: [channel, windowSize, windowSize, 1])
  return MLX.stopGradient(window)
}
func ssim(
  img1: MLXArray,
  img2: MLXArray,
  windowSize: Int = 11,
  sizeAverage: Bool = true
) -> MLXArray {
  let channel: Int = img1.shape.last!
  let window = createWindow(windowSize: windowSize, channel: channel)  // [C, 1, K, K]
  let conv2d = { (input: MLXArray) -> MLXArray in
    return MLX.conv2d(input, window, padding: IntOrPair(windowSize / 2), groups: channel)
  }
  let mu1 = conv2d(img1)
  let mu2 = conv2d(img2)
  let mu1_sq = MLX.square(mu1)
  let mu2_sq = MLX.square(mu2)
  let mu1_mu2 = mu1 * mu2

  let sigma1_sq = conv2d(img1 * img1) - mu1_sq
  let sigma2_sq = conv2d(img2 * img2) - mu2_sq
  let sigma12 = conv2d(img1 * img2) - mu1_mu2

  let C1: Float = 0.01 * 0.01
  let C2: Float = 0.03 * 0.03

  let num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
  let denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
  let ssim_map = num / denom

  if sizeAverage {
    return ssim_map.mean()
  } else {
    return ssim_map.mean(axes: [1, 2, 3])
  }
}
