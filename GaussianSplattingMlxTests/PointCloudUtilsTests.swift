//
//  PointCloudUtilsTests.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/02.
//

import MLX
import Testing
import simd

@testable import GaussianSplattingMlx

struct PointCloudUtilsTests {
  @Test func test_getRaysFromImages() throws {
    // 1 batch, 2x2 image (4 pixels)
    let B = 1
    let H = 2
    let W = 2

    // Intrinsics and c2w as identity matrices
    let intrinsics = MLX.eye(4).expandedDimensions(axes: [0])  // [1,4,4]
    let c2w = MLX.eye(4).expandedDimensions(axes: [0])  // [1,4,4]

    // Execution
    let (rays_o, rays_d) = getRaysFromImages(
      H: H,
      W: W,
      intrinsics: intrinsics,
      c2w: c2w
    )

    // Verification: shapes
    #expect(rays_o.shape == [1, 4, 3])  // [B, HW, 3]
    #expect(rays_d.shape == [1, 4, 3])

    // Origin check: all [0,0,0] (identity c2w means origin)
    for i in 0..<4 {
      let o = rays_o[0, i]
      #expect(
        MLX.allClose(
          o,
          MLXArray([0.0, 0.0, 0.0] as [Float]),
          atol: 1e-5
        ).item()
      )
    }

    // Direction check: pixels (u,v)=(0,0),(1,0),(0,1),(1,1) -> (x,y,1)
    let directions: [[Float]] = [
      [0.0, 0.0, 1.0],  // (u,v)=(0,0)
      [1.0, 0.0, 1.0],  // (u,v)=(1,0)
      [0.0, 1.0, 1.0],  // (u,v)=(0,1)
      [1.0, 1.0, 1.0],  // (u,v)=(1,1)
    ]
    for i in 0..<4 {
      let d = rays_d[0, i]
      let expected = MLXArray(directions[i])
      #expect(MLX.allClose(d, expected, atol: 1e-5).item())
    }
  }
}
