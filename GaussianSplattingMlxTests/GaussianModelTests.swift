//
//  GaussianModelTests.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/03.
//

import Foundation
import MLX
import MLXOptimizers
import Testing

@testable import GaussianSplattingMlx

@Suite struct GaussianModelTests {
    @Test func test_distTopK() {
        // Example: 4 points (0,0,0), (1,0,0), (0,1,0), (0,0,1)
        let pts: [Float] = [
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]
        let N = 4
        let D = 3
        let X = MLXArray(pts, [N, D])

        // k=2 nearest neighbors
        let avgDist2 = distTopK(X, k: 2)  // [N]
        let expect = MLXArray([0.5, 0.5, 0.5, 0.5] as [Float])
        print(avgDist2)
        #expect(avgDist2.allClose(expect).item())
    }
}
