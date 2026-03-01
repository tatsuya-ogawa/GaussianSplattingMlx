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

    @Test func test_build_covariance_3d() throws {
        // Simple scaling and identity rotation
        let scaleData: [Float] = [2.0, 1.0, 0.5]
        let rotData: [Float] = [0.0, 0.0, 0.0, 1.0]  // Identity quaternion

        let s = MLXArray(scaleData, [1, 3])
        let r = MLXArray(rotData, [1, 4])

        let result = build_covariance_3d(s: s, r: r)

        // Expected: diag(s²)
        let expectedData: [Float] = [
            4.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.25,
        ]
        let expected = MLXArray(expectedData, [1, 3, 3])

        #expect(result.shape == [1, 3, 3])
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }

    @Test func test_get_max_covariance() throws {
        // Two covariance matrices (batch of 2)
        let covData: [Float] = [
            2.0, 0.0,  // 1st matrix
            0.0, 1.0,

            1.0, 0.5,  // 2nd matrix
            0.5, 2.0,
        ]
        let cov2d = MLXArray(covData, [2, 2, 2])  // [batch_size, 2, 2] 2, 2]
        let result = get_max_covariance(cov2d: cov2d)

        let expectedCpu: [Float] = [2.0, Float(1.5 + sqrt(0.5))]
        let expected = MLXArray(expectedCpu, [2])

        #expect(result.shape == [2])
        print(result)
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }

    @Test func test_get_radius() throws {
        // Two covariance matrices (batch of 2)
        let covData: [Float] = [
            2.0, 0.0,  // first
            0.0, 1.0,

            1.0, 0.5,  // second
            0.5, 2.0,
        ]
        let cov2d = MLXArray(covData, [2, 2, 2])  // [batch_size, 2, 2]
        let result = get_radius(cov2d: cov2d)

        let expectedCpu: [Float] = [6.0, 6.0]
        let expected = MLXArray(expectedCpu, [2])

        #expect(result.shape == [2])
        print(result)
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }

    @Test func test_projection_ndc() throws {
        // Simple test with identity matrices
        let pointsData: [Float] = [
            0.0, 0.0, 1.0,
            1.0, 1.0, 1.0,
        ]
        let points = MLXArray(pointsData, [2, 3])

        let viewMatrix = simd_double4x4(1.0).toMLXArray()
        let projMatrix = simd_double4x4(1.0).toMLXArray()

        let (p_proj, p_view, in_mask) = projection_ndc(
            points: points,
            viewMatrix: viewMatrix,
            projMatrix: projMatrix
        )

        #expect(p_proj.shape == [2, 4])
        #expect(p_view.shape == [2, 4])
        #expect(in_mask.shape == [2])
        let expectedInMask: [Bool] = [true, true]
        #expect(MLX.allClose(in_mask, expectedInMask).item())
    }

    @Test func test_get_rect() throws {
        let pixCoordData: [Float] = [
            100.0, 100.0,
            200.0, 150.0,
        ]
        let pixCoord = MLXArray(pixCoordData, [2, 2])

        let radii = MLXArray([10.0, 20.0] as [Float], [2])

        let (rect_min, rect_max) = get_rect(
            pix_coord: pixCoord,
            radii: radii,
            width: 400,
            height: 300
        )

        let expectedMinData: [Float] = [
            90.0, 90.0,
            180.0, 130.0,
        ]
        let expectedMin = MLXArray(expectedMinData, [2, 2])

        let expectedMaxData: [Float] = [
            110.0, 110.0,
            220.0, 170.0,
        ]
        let expectedMax = MLXArray(expectedMaxData, [2, 2])

        #expect(rect_min.shape == [2, 2])
        #expect(rect_max.shape == [2, 2])
        #expect(MLX.allClose(rect_min, expectedMin).item())
        #expect(MLX.allClose(rect_max, expectedMax).item())
    }
    @Test func test_build_covariance_2d() throws {
        // Simple case with just one point
        let mean3dData: [Float] = [
            0.0, 0.0, 1.0,  // 1 unit in Z direction from origin
        ]
        let mean3d = MLXArray(mean3dData, [1, 3])

        // 3D covariance: diagonal matrix
        let cov3dData: [Float] = [
            4.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.25,
        ]
        let cov3d = MLXArray(cov3dData, [1, 3, 3])

        // View and projection matrices are identity matrices
        let viewMatrix = simd_double4x4(1.0).toMLXArray()
        // FOV and focal with intuitive values
        let fovX = MLXArray(Float.pi / 2)  // 90°
        let fovY = MLXArray(Float.pi / 2)
        let focalX = MLXArray(1.0 as Float)
        let focalY = MLXArray(1.0 as Float)

        let result = build_covariance_2d(
            mean3d: mean3d,
            cov3d: cov3d,
            viewMatrix: viewMatrix,
            fovX: fovX,
            fovY: fovY,
            focalX: focalX,
            focalY: focalY
        )

        let expectedData: [Float] = [
            4.3, 0.3,  // first point
            0.0, 1.0,

            4.0, 0.0,  // second point
            0.3, 1.3,
        ]
        let expected = MLXArray(expectedData, [2, 2, 2])
        #expect(result.shape == [2, 2, 2])
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
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
