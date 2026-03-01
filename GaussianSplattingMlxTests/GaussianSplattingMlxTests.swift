//
//  GaussianSplattingMlXTests.swift
//  GaussianSplattingMlXTests
//
//  Created by Tatsuya Ogawa on 2025/05/28.
//

import MLX
import Testing
import simd

@testable import GaussianSplattingMlx

struct GaussianSplattingMlxTests {

    @Test func testMatmul() async throws {
        let vector4: [Float] = [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ]
        let matrix4x4: simd_double4x4 = simd_double4x4(1.0)
        let vectorArray = MLXArray(vector4, [2, 4])
        let matrixArray = matrix4x4.toMLXArray()
        let result = vectorArray.matmul(matrixArray)
        #expect(result.shape == [2, 4])
    }

    @Test func testExpanded() async throws {
        let input = MLXArray([1, 2, 3])
        let expandedFirst: MLXArray = input.expandedDimensions(axes: [0])
        #expect(expandedFirst.shape.elementsEqual([1, 3]))
        let expandedLast: MLXArray = input.expandedDimensions(axes: [-1])
        #expect(expandedLast.shape.elementsEqual([3, 1]))
    }

    @Test func test_inverse_sigmoid() throws {
        let values: [Float] = [0.1, 0.5, 0.9]
        let x = MLXArray(values)
        let result = inverse_sigmoid(x: x)

        // Expected values: log(x/(1-x))
        // Explicitly use MLX.log to avoid ambiguity with Foundation.log
        let v1 = MLXArray(0.1) / MLXArray(0.9)
        let v2 = MLXArray(0.5) / MLXArray(0.5)
        let v3 = MLXArray(0.9) / MLXArray(0.1)

        let expected = MLX.stacked([
            MLX.log(v1),
            MLX.log(v2),
            MLX.log(v3),
        ])
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }

    @Test func testHomogeneous() throws {
        let pointsData: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]
        let points = MLXArray(pointsData, [2, 3])
        let result = homogeneous(points: points)

        let expectedData: [Float] = [
            1.0, 2.0, 3.0, 1.0,
            4.0, 5.0, 6.0, 1.0,
        ]
        let expected = MLXArray(expectedData, [2, 4])

        #expect(result.shape == [2, 4])
        #expect(MLX.allClose(result, expected).item())
    }

    @Test func test_build_rotation() throws {
        // Test identity quaternion [0,0,0,1]
        let identityData: [Float] = [1.0, 0.0, 0.0, 0.0]
        let identityQuaternion = MLXArray(identityData, [1, 4])
        let identityResult = build_rotation(quaternion: identityQuaternion)

        // Identity quaternion should produce identity rotation matrix
        let identityExpected = MLXArray.eye(3, m: 3).expandedDimensions(axis: 0)
        #expect(
            MLX.allClose(identityResult, identityExpected, atol: 1e-5).item()
        )

        // Test with multiple quaternions
        let data: [Float] = [
            1.0, 0.0, 0.0, 0.0,  // Identity quaternion
            0.0, 1.0, 0.0, 0.0,  // 180 degree rotation around X-axis
        ]
        let quaternion = MLXArray(data, [2, 4])
        let result = build_rotation(quaternion: quaternion)
        print(result)
        // Expected matrices
        let expectedData: [Float] = [
            // Identity matrix for [0,0,0,1]
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            // Rotation matrix for [1,0,0,0]
            1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, -1.0,
        ]
        let expected = MLXArray(expectedData, [2, 3, 3])
        print(expected)
        #expect(result.shape == [2, 3, 3])
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }

    @Test func test_build_scaling_rotation() throws {
        // Test with identity rotation and uniform scaling
        let scaleData: [Float] = [2.0, 2.0, 2.0]
        let rotData: [Float] = [1.0, 0.0, 0.0, 0.0]  // Identity quaternion

        let s = MLXArray(scaleData, [1, 3])
        let r = MLXArray(rotData, [1, 4])

        let result = build_scaling_rotation(s: s, r: r)

        // Expected: diagonal matrix with scaling factors
        let expectedData: [Float] = [
            2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0,
        ]
        let expected = MLXArray(expectedData, [1, 3, 3])

        #expect(result.shape == [1, 3, 3])
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }

    @Test func test_strip_lowerdiag() throws {
        let matrixData: [Float] = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]
        let matrix = MLXArray(matrixData, [1, 3, 3])
        let result = strip_lowerdiag(L: matrix)

        // Expected: [1.0, 2.0, 3.0, 5.0, 6.0, 9.0]
        let expectedData: [Float] = [1.0, 2.0, 3.0, 5.0, 6.0, 9.0]
        let expected = MLXArray(expectedData, [1, 6])

        #expect(result.shape == [1, 6])
        #expect(MLX.allClose(result, expected).item())
    }
    @Test func test_createMeshGrid_3x3() throws {
        let shape = [3]
        let mesh = createMeshGrid(shape: shape)  // shape: [3,3,2]
        let expectedData: [Int32] = [
            0, 0, 1, 0, 2, 0,
            0, 1, 1, 1, 2, 1,
            0, 2, 1, 2, 2, 2,
        ]
        let expected = MLXArray(expectedData, [3, 3, 2])
        #expect(mesh.shape == [3, 3, 2])
        #expect(MLX.allClose(mesh, expected, atol: 1e-5).item())
    }
}
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map { i in
            Array(self[i..<Swift.min(i + size, count)])
        }
    }
}
