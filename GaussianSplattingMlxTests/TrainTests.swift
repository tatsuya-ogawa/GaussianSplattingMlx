import Foundation
//
//  TrainTests.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/02.
//
import MLX
import MLXOptimizers
import Testing

@testable import GaussianSplattingMlx

struct TrainTests {
    @Test func test_simple_train() throws {
        var param = MLXArray([0, 0, 0, 0] as [Float])
        let matrix = MLXArray(
            [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4] as [Float],
            [4, 4]
        )
        let target = MLXArray([1, 2, 3, 4] as [Float])
        let train: ([MLXArray]) -> [MLXArray] = {
            (param: [MLXArray]) -> [MLXArray] in
            return [(param[0].matmul(matrix) - target).square().mean()]
        }
        let optimizer = Adam(learningRate: 1e-2)
        var state = optimizer.newState(parameter: param)
        for step in 0..<200 {
            let (lossArr, grads) = MLX.valueAndGrad(train)([param])
            let (newParam, newState) = optimizer.applySingle(
                gradient: grads[0],
                parameter: param,
                state: state
            )
            param = newParam
            state = newState
        }
        // Assert convergence
        let finalOut = param.matmul(matrix)
        // Test that the difference with the target is small
        print(param)
        #expect(MLX.allClose(finalOut, target, atol: 1e-2).item())
    }
    func randomNormalArray(count: Int, mean: Float = 0.0, std: Float = 1.0) -> [Float] {
        var arr = [Float](repeating: 0, count: count)
        for i in 0..<count {
            // Generate standard normal random numbers using Box-Muller method
            let u1 = Float.random(in: 0..<1)
            let u2 = Float.random(in: 0..<1)
            let z0 = sqrt(-2.0 * Foundation.log(u1)) * cos(2.0 * .pi * u2)
            arr[i] = z0 * std + mean
        }
        return arr
    }
    @Test func testSSIM() {
        let batch = 1
        let channel = 3
        let h = 32
        let w = 32

        // まったく同じ画像
        let img1 = MLXArray.ones([batch, h, w, channel])
        let img2 = MLXArray.ones([batch, h, w, channel])

        let ssimVal = ssim(img1: img1, img2: img2)
        print("SSIM (identical): \(ssimVal)")  // 1.0 に近いはず

        // ノイズを加えた画像
        let totalCount = batch * channel * h * w
        let noiseData = randomNormalArray(count: totalCount, mean: 0.0, std: 0.1)
        let noise = MLXArray(noiseData, [batch, h, w, channel])
        let img2_noisy = img1 + noise
        let ssimValNoisy = ssim(img1: img1, img2: img2_noisy)
        print("SSIM (noisy): \(ssimValNoisy)")  // 1.0 より小さいはず

        // まったく違う画像
        let img3 = MLXArray.zeros([batch, h, w, channel])
        let ssimValDiff = ssim(img1: img1, img2: img3)
        print("SSIM (all ones vs all zeros): \(ssimValDiff)")  // 0.0に近いはず
    }
}
