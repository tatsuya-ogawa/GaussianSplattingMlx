//
//  ShUtilsTests.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import MLX
import Testing
import simd

@testable import GaussianSplattingMlx

struct ShUtilsTests {
    @Test func test_evalSh_deg0() throws {
        // sh: [batch, number of channels=1]
        let shData: [Float] = [1.0, -0.5]
        let sh = MLXArray(shData, [2, 1])
        // dirs can be anything
        let dirsData: [Float] = [0.0, 0.0, 1.0, 0.1, 0.2, 0.3]
        let dirs = MLXArray(dirsData, [2, 3])

        let out = evalSh(deg: 0, sh: sh, dirs: dirs)
        // result = C0 * sh[...,0]
        let expectedData: [Float] = [C0 * 1.0, C0 * -0.5]
        let expected = MLXArray(expectedData, [2])
        print("out:", out)
        print("expected:", expected)
        #expect(MLX.allClose(out, expected, atol: 1e-8).item())
    }
    @Test func test_evalSh_deg1() throws {
        // sh: [1, 4]
        let shData: [Float] = [1.0, 0.2, 0.3, 0.4]
        let sh = MLXArray(shData, [1, 4])
        // x=1, y=2, z=3
        let dirs = MLXArray([1.0, 2.0, 3.0] as [Float], [1, 3])
        let out = evalSh(deg: 1, sh: sh, dirs: dirs)
        // result = C0*1.0 - C1*2.0*0.2 + C1*3.0*0.3 - C1*1.0*0.4
        let expectedValue: Float =
            C0 * 1.0 - C1 * 2.0 * 0.2 + C1 * 3.0 * 0.3 - C1 * 1.0 * 0.4
        let expected = MLXArray([expectedValue], [1])
        print(out, expected)
        #expect(MLX.allClose(out, expected, atol: 1e-8).item())
    }
    @Test func test_evalSh_deg2() throws {
        // sh: [1, 9]
        let shData: [Float] = [0.5, 0.2, -0.1, 0.1, 1.0, -1.0, 2.0, 0.5, -2.0]
        let sh = MLXArray(shData, [1, 9])
        let x: Float = 0.5
        let y: Float = -1.0
        let z: Float = 2.0
        let dirs = MLXArray([x, y, z], [1, 3])
        // degree1まで
        var res = C0 * 0.5 - C1 * y * 0.2 + C1 * z * (-0.1) - C1 * x * 0.1
        // degree2
        let xx = x * x
        let yy = y * y
        let zz = z * z
        let xy = x * y
        let yz = y * z
        let xz = x * z
        res +=
            C2[0] * xy * 1.0 + C2[1] * yz * (-1.0) + C2[2] * (2 * zz - xx - yy)
            * 2.0
            + C2[3] * xz * 0.5 + C2[4] * (xx - yy) * (-2.0)
        let expected = MLXArray([Float(res)], [1])
        let out = evalSh(deg: 2, sh: sh, dirs: dirs)
        print(out, expected)
        #expect(MLX.allClose(out, expected, atol: 1e-8).item())
    }
    @Test func test_evalSh_deg3() throws {
        // sh: [1, 16]
        let shData: [Float] = Array(stride(from: 0.1, to: 1.7, by: 0.1))  // [0.1, 0.2, ..., 1.6]
        let sh = MLXArray(shData, [1, 16])
        let x: Float = 0.1
        let y: Float = -0.2
        let z: Float = 0.3
        let dirs = MLXArray([x, y, z], [1, 3])
        // degree1
        var res = C0 * 0.1 - C1 * y * 0.2 + C1 * z * 0.3 - C1 * x * 0.4
        // degree2
        let xx = x * x
        let yy = y * y
        let zz = z * z
        let xy = x * y
        let yz = y * z
        let xz = x * z
        res +=
            C2[0] * xy * 0.5 + C2[1] * yz * 0.6 + C2[2] * (2 * zz - xx - yy)
            * 0.7
            + C2[3] * xz * 0.8 + C2[4] * (xx - yy) * 0.9
        // degree3
        res +=
            C3[0] * y * (3 * xx - yy) * 1.0
            + C3[1] * xy * z * 1.1
            + C3[2] * y * (4 * zz - xx - yy) * 1.2
            + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * 1.3
            + C3[4] * x * (4 * zz - xx - yy) * 1.4
            + C3[5] * z * (xx - yy) * 1.5
            + C3[6] * x * (xx - 3 * yy) * 1.6
        let expected = MLXArray([Float(res)], [1])
        let out = evalSh(deg: 3, sh: sh, dirs: dirs)
        print(out, expected)
        #expect(MLX.allClose(out, expected, atol: 1e-8).item())
    }
    @Test func test_evalSh_deg4() throws {
        // sh: [1, 25]
        let shData: [Float] = Array(stride(from: 0.1, to: 2.6, by: 0.1))  // [0.1, 0.2, ..., 2.5]
        let sh = MLXArray(shData, [1, 25])
        let x: Float = -0.3
        let y: Float = 0.2
        let z: Float = 0.7
        let dirs = MLXArray([x, y, z], [1, 3])
        // degree1
        var res = C0 * 0.1 - C1 * y * 0.2 + C1 * z * 0.3 - C1 * x * 0.4
        // degree2
        let xx = x * x
        let yy = y * y
        let zz = z * z
        let xy = x * y
        let yz = y * z
        let xz = x * z
        res +=
            C2[0] * xy * 0.5 + C2[1] * yz * 0.6 + C2[2] * (2 * zz - xx - yy)
            * 0.7
            + C2[3] * xz * 0.8 + C2[4] * (xx - yy) * 0.9
        // degree3
        res +=
            C3[0] * y * (3 * xx - yy) * 1.0
            + C3[1] * xy * z * 1.1
            + C3[2] * y * (4 * zz - xx - yy) * 1.2
            + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * 1.3
            + C3[4] * x * (4 * zz - xx - yy) * 1.4
            + C3[5] * z * (xx - yy) * 1.5
            + C3[6] * x * (xx - 3 * yy) * 1.6
        // degree4
        res +=
            C4[0] * xy * (xx - yy) * 1.7
            + C4[1] * yz * (3 * xx - yy) * 1.8
            + C4[2] * xy * (7 * zz - 1) * 1.9
            + C4[3] * yz * (7 * zz - 3) * 2.0
            + C4[4] * (zz * (35 * zz - 30) + 3) * 2.1
            + C4[5] * xz * (7 * zz - 3) * 2.2
            + C4[6] * (xx - yy) * (7 * zz - 1) * 2.3
            + C4[7] * xz * (xx - 3 * yy) * 2.4
            + C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * 2.5
        let expected = MLXArray([Float(res)], [1])
        let out = evalSh(deg: 4, sh: sh, dirs: dirs)
        print(out, expected)
        #expect(MLX.allClose(out, expected, atol: 1e-8).item())
    }
}
