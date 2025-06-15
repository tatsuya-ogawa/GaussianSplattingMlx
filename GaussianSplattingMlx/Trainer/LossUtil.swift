//
//  LossUtil.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import Foundation
import MLX

// MARK: - SmoothL1Loss (SL1Loss)
class SL1Loss {
    let ohem: Bool
    let topk: Double

    init(ohem: Bool = false, topk: Double = 0.6) {
        self.ohem = ohem
        self.topk = topk
    }

    func forward(inputs: MLXArray, targets: MLXArray, mask: MLXArray) -> MLXArray {
        let predMasked = inputs[mask]
        let targetMasked = targets[mask]
        let diff = predMasked - targetMasked
        let absDiff = MLX.abs(diff)
        let loss: MLXArray = MLX.where(absDiff .< 1.0, 0.5 * MLX.square(diff), absDiff - 0.5)

        var flatLoss = loss.reshaped([-1])
        if ohem {
            let numHard = Int(Double(flatLoss.shape[0]) * topk)
            let sorted = MLX.sorted(flatLoss)
            flatLoss = sorted[.stride(to: -numHard)]
        }
        return flatLoss.mean()
    }
}

// MARK: - L1, L2 Loss
func l1Loss(_ prediction: MLXArray, _ gt: MLXArray) -> MLXArray {
    return MLX.abs(prediction - gt).mean()
}
func l2Loss(_ prediction: MLXArray, _ gt: MLXArray) -> MLXArray {
    return MLX.square(prediction - gt).mean()
}

// MARK: - Gaussian window for SSIM
func gaussian(windowSize: Int, sigma: Float) -> MLXArray {
    let center = Float(windowSize) / 2.0
    let vals = (0..<windowSize).map { x in
        exp(-pow(Float(x) - center, 2) / (2 * pow(sigma, 2)))
    }
    let arr = MLXArray(vals)
    return arr / arr.sum()
}

let HUGE_NUMBER: Float = 1e10
let TINY_NUMBER: Float = 1e-6

// MSE
func img2mse(_ x: MLXArray, _ y: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let diff = x - y
    if let mask = mask {
        let expandedMask = mask.expandedDimensions(axes: [mask.shape.count])  // mask shape: [...,1]
        let sq = MLX.square(diff) * expandedMask
        let sumsq = sq.sum()
        let denom = mask.sum() * Float(x.shape.last!) + TINY_NUMBER
        return sumsq / denom
    } else {
        return MLX.square(diff).mean()
    }
}

// MSE→PSNR
func mse2psnr(_ mse: MLXArray) -> MLXArray {
    return -10.0 * log(mse + TINY_NUMBER) / log(10.0)
}

// img2psnr
func img2psnr(_ x: MLXArray, _ y: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    return mse2psnr(img2mse(x, y, mask: mask))
}
