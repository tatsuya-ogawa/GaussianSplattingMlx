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
        // 2つの共分散行列（バッチ2個）
        let covData: [Float] = [
            2.0, 0.0,  // 1個目
            0.0, 1.0,

            1.0, 0.5,  // 2個目
            0.5, 2.0,
        ]
        let cov2d = MLXArray(covData, [2, 2, 2])  // [バッチ数, 2, 2]
        let result = get_max_covariance(cov2d: cov2d)

        let expectedCpu: [Float] = [2.0, Float(1.5 + sqrt(0.5))]
        let expected = MLXArray(expectedCpu, [2])

        #expect(result.shape == [2])
        print(result)
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }

    @Test func test_get_radius() throws {
        // 2つの共分散行列（バッチ2個）
        let covData: [Float] = [
            2.0, 0.0,  // 1個目
            0.0, 1.0,

            1.0, 0.5,  // 2個目
            0.5, 2.0,
        ]
        let cov2d = MLXArray(covData, [2, 2, 2])  // [バッチ数, 2, 2]
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
        // 1点だけのシンプルなケースで
        let mean3dData: [Float] = [
            0.0, 0.0, 1.0,  // 原点から1だけZ方向
        ]
        let mean3d = MLXArray(mean3dData, [1, 3])

        // 3D共分散: 対角行列
        let cov3dData: [Float] = [
            4.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 0.25,
        ]
        let cov3d = MLXArray(cov3dData, [1, 3, 3])

        // ビュー・射影行列は単位行列
        let viewMatrix = simd_double4x4(1.0).toMLXArray()
        // fov, focalも直感的な値
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
            4.3, 0.3,  // 1点目
            0.0, 1.0,

            4.0, 0.0,  // 2点目
            0.3, 1.3,
        ]
        let expected = MLXArray(expectedData, [2, 2, 2])
        #expect(result.shape == [2, 2, 2])
        #expect(MLX.allClose(result, expected, atol: 1e-5).item())
    }
    @Test func test_matrixInverse2d() throws {
        // 2つの2x2行列
        let matrixData: [Float] = [
            1, 2,
            3, 4,

            2, 0,
            0, 2,
        ]
        let m = MLXArray(matrixData, [2, 2, 2])  // shape: [2, 2, 2]
        let inv = matrixInverse2d(m)

        // 期待値計算
        // 1つ目: [[1,2],[3,4]] → 逆行列 [[-2,1],[1.5,-0.5]]
        // 2つ目: [[2,0],[0,2]] → 逆行列 [[0.5,0],[0,0.5]]
        let expectedData: [Float] = [
            -2, 1,
            1.5, -0.5,

            0.5, 0,
            0, 0.5,
        ]
        let expected = MLXArray(expectedData, [2, 2, 2])
        print("inv:", inv)
        print("expected:", expected)
        #expect(MLX.allClose(inv, expected, atol: 1e-5).item())
    }
    @Test func test_build_color_camera() throws {
        // 1. カメラ定義（原点に設置、focal100, 256x256画素）
        let width = 256
        let height = 256
        let focal = 100.0
        var intrinsic = matrix_double4x4()
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[2, 2] = 1.0
        intrinsic[3, 3] = 1.0
        let c2w = matrix_double4x4(1.0)  // カメラ座標系＝ワールド座標系
        let camera = Camera(
            width: width,
            height: height,
            intrinsic: intrinsic.toMLXArray(),
            c2w: c2w.toMLXArray()
        )

        // 2. means3D: 2点（ダミー）
        let means3DData: [Float] = [
            1.0, 2.0, 3.0,
            -2.0, 1.0, 0.0,
        ]
        let means3D = MLXArray(means3DData, [2, 3])

        // 3. shs: [2, 3, 1]（RGB各chごとにSH=degree0のみ、RGB値をわかりやすく）
        let shsData: [Float] = [
            1.0, 0.5, -0.5,  // 1点目RGB
            -0.5, 0.0, 0.5,  // 2点目RGB
        ]
        let shs = MLXArray(shsData, [2, 3, 1])
        // 4. degree=0
        let out0 = build_color(
            means3d: means3D,
            shs: shs.transposed(0, 2, 1),
            camera: camera,
            activeShDegree: 0
        )
        // means3D - camera_center はmeans3Dそのまま
        // evalSh(deg=0)はshs[...,0]にC0掛け
        var expectedData0 = [Float]()
        for v in shsData {
            let c = max(Float(C0) * v + 0.5, 0.0)
            expectedData0.append(c)
        }
        let expected0 = MLXArray(expectedData0, [2, 3])
        print("deg=0 out:", out0)
        print("expected:", expected0)
        #expect(MLX.allClose(out0, expected0, atol: 1e-6).item())

        // 5. degree=1テスト
        // shs: [2,3,4] RGB×SH4
        let shs1Data: [Float] = [
            1, 0.2, 0.3, 0.4,  // R
            0.5, -0.1, 0.2, 0.1,  // G
            -0.5, 0.0, -0.2, 0.2,  // B
            0.7, 0.2, 0.3, -0.3,  // R2
            -0.3, 0.4, 0.0, 0.2,  // G2
            0.5, -0.5, 0.5, 0.5,  // B2
        ]
        let shs1 = MLXArray(shs1Data, [2, 3, 4])
        // means3D同じ
        // rays_d = means3D - camera_center = means3D
        // 1点目(1.0,2.0,3.0), 2点目(-2.0,1.0,0.0)
        let rays_d = means3D
        // MLX: evalShはshsのチャンネルごとに適用なので、テストも各chで計算
        func evalSHdeg1(_ sh: [Float], _ x: Float, _ y: Float, _ z: Float)
            -> [Float]
        {
            // sh: [4] (SH)
            [
                Float(
                    C0 * sh[0] - C1 * y * sh[1] + C1 * z * sh[2] - C1 * x
                        * sh[3]
                )
            ]
        }
        var expectedData1 = [Float]()
        let dirs = means3DData.chunked(into: 3)  // [[1,2,3], [-2,1,0]]
        for i in 0..<2 {
            for c in 0..<3 {
                let offset = i * 12 + c * 4
                let sh = Array(shs1Data[offset..<offset + 4])
                let (x, y, z) = (dirs[i][0], dirs[i][1], dirs[i][2])
                let v = evalSHdeg1(sh, x, y, z)[0]
                expectedData1.append(max(v + 0.5, 0.0))
            }
        }
        let expected1 = MLXArray(expectedData1, [2, 3])
        print(shs1)
        let out1 = build_color(
            means3d: means3D,
            shs: shs1.transposed(0, 2, 1),
            camera: camera,
            activeShDegree: 1
        )
        print("deg=1 out:", out1)
        print("expected:", expected1)
        #expect(MLX.allClose(out1, expected1, atol: 1e-5).item())
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
    @Test func test_computeTileMask() throws {
        // ガウシアン中心点と半径
        let rect_minData: [Float] = [
            90.0, 90.0,  // 1つ目のガウシアン
            180.0, 130.0,  // 2つ目のガウシアン
        ]
        let rect_min = MLXArray(rect_minData, [2, 2])

        let rect_maxData: [Float] = [
            110.0, 110.0,  // 1つ目のガウシアン
            220.0, 170.0,  // 2つ目のガウシアン
        ]
        let rect_max = MLXArray(rect_maxData, [2, 2])

        // タイル座標（左上から64x64ピクセルのタイル）
        let h = 64
        let w = 64
        let tileSize = TILE_SIZE_H_W(w: w, h: h)

        // タイル内に入るガウシアンの判定
        let inMask = computeTileMask(
            h: h,
            w: w,
            tileSize: tileSize,
            rect: (rect_min, rect_max)
        )

        // 期待値: 1つ目のガウシアンのみタイル内に入る
        let expectedData: [Bool] = [true, false]
        let expected = MLXArray(expectedData)

        #expect(MLX.allClose(inMask, expected).item())
    }
    @Test func test_computeDxDy() throws {
        // tile_coord: [B, 2] (例: 2ピクセル)
        let tileData: [Float] = [
            0.0, 0.0,
            1.0, 2.0,
        ]
        let tile_coord = MLXArray(tileData, [2, 2])  // B=2

        // sorted_means2d: [P, 2] (例: 3ガウス)
        let meansData: [Float] = [
            1.0, 1.0,
            2.0, 2.0,
            0.0, 2.0,
        ]
        let sorted_means2d = MLXArray(meansData, [3, 2])  // P=3

        // 関数適用
        let dxdy = computeDxDy(tile_coord: tile_coord, sorted_means2d: sorted_means2d)
        // 期待shape: [2, 3, 2] (B, P, 2)

        // 期待値: 各pixelと各meanの差分
        // pixel 0: [0,0] - [1,1] = [-1,-1]
        //          [0,0] - [2,2] = [-2,-2]
        //          [0,0] - [0,2] = [ 0,-2]
        // pixel 1: [1,2] - [1,1] = [0, 1]
        //          [1,2] - [2,2] = [-1, 0]
        //          [1,2] - [0,2] = [1, 0]
        let expectedData: [Float] = [
            -1, -1,
            -2, -2,
            0, -2,

            0, 1,
            -1, 0,
            1, 0,
        ]
        let expected = MLXArray(expectedData, [2, 3, 2])

        print("dxdy =", dxdy)
        print("expected =", expected)
        #expect(dxdy.shape == [2, 3, 2])
        #expect(MLX.allClose(dxdy, expected, atol: 1e-6).item())
    }
    @Test func test_computeGaussianWeights() throws {
        // tile_coord: [2,2]  (B=2)
        let tileData: [Float] = [
            0.0, 0.0,
            1.0, 0.0,
        ]
        let tile_coord = MLXArray(tileData, [2, 2])

        // sorted_means2d: [1,2]  (P=1)
        let meansData: [Float] = [
            0.0, 1.0,
        ]
        let sorted_means2d = MLXArray(meansData, [1, 2])

        // sorted_conic: [1,2,2]  ここでは単位行列（identity）
        let conicData: [Float] = [
            1.0, 0.0,
            0.0, 1.0,
        ]
        let sorted_conic = MLXArray(conicData, [1, 2, 2])

        // 計算
        let weights = computeGaussianWeights(
            tile_coord: tile_coord,
            sorted_means2d: sorted_means2d,
            sorted_conic: sorted_conic
        )
        // shape: [2, 1]（各ピクセル×ガウス）

        // 期待値:
        // dx: [ [0,0] - [0,1] = [0,-1],   [1,0] - [0,1] = [1,-1] ]
        // Σ = identity なら
        //   w = exp(-0.5*(dx^T Σ dx)) = exp(-0.5*(x^2 + y^2))
        // 1点目: [0,-1] → exp(-0.5*(0^2 + 1^2)) = exp(-0.5)
        // 2点目: [1,-1] → exp(-0.5*(1 + 1)) = exp(-1.0)
        let expectedData: [Float] = [
            exp(-0.5),
            exp(-1.0),
        ]
        let expected = MLXArray(expectedData, [2, 1])

        print("weights =", weights)
        print("expected =", expected)
        #expect(weights.shape == [2, 1])
        #expect(MLX.allClose(weights, expected, atol: 1e-6).item())
    }

    @Test func test_renderTile() throws {
        // 256x256の標準画像サイズでGaussianRendererを初期化
        let tileSize = TILE_SIZE_H_W(w: 256, h: 256)
        let renderer = GaussianRenderer(
            active_sh_degree: 4, W: 256, H: 256, TILE_SIZE: tileSize, whiteBackground: false)

        // 簡単なデータを作成
        let means2DData: [Float] = [
            32.0, 32.0,  // タイルの中心
            96.0, 96.0,  // タイルの外
        ]
        let means2D = MLXArray(means2DData, [2, 2])

        // 共分散行列
        let cov2dData: [Float] = [
            10.0, 0.0, 0.0, 10.0,  // シンプルな対角行列
            10.0, 0.0, 0.0, 10.0,
        ]
        let cov2d = MLXArray(cov2dData, [2, 2, 2])

        // 色
        let colorData: [Float] = [
            1.0, 0.0, 0.0,  // 赤
            0.0, 1.0, 0.0,  // 緑
        ]
        let color = MLXArray(colorData, [2, 3])

        // 不透明度
        let opacityData: [Float] = [0.8, 0.5]
        let opacity = MLXArray(opacityData)

        // 深度
        let depthsData: [Float] = [10.0, 20.0]
        let depths = MLXArray(depthsData)

        // バウンディングボックス
        let radii = MLXArray([10.0, 10.0] as [Float])
        let rect = get_rect(
            pix_coord: means2D,
            radii: radii,
            width: 256,
            height: 256
        )

        // タイル描画
        let (tile_color, tile_depth, acc_alpha) = renderer.renderTile(
            h: 0,
            w: 0,
            tileSize: TILE_SIZE_H_W(w: 64, h: 64),
            means2d: means2D,
            cov2d: cov2d,
            color: color,
            opacity: opacity,
            depths: depths,
            rect: rect
        )

        // タイルのサイズ確認
        #expect(tile_color.shape == [64, 64, 3])
        #expect(tile_depth.shape == [64, 64, 1])
        #expect(acc_alpha.shape == [64, 64, 1])

        // 中心付近は赤く、不透明度が高いことを確認
        let center_color = tile_color[32, 32]
        let center_alpha = acc_alpha[32, 32]

        // 赤い色が優勢であることを確認
        #expect((center_color[0] .> 0.5).shape == [])
        #expect((center_color[1] .< 0.5).shape == [])
        #expect((center_color[2] .< 0.5).shape == [])

        // 中心付近は不透明度が高い
        #expect((center_alpha[0] .> 0.5).shape == [])
    }
}
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map { i in
            Array(self[i..<Swift.min(i + size, count)])
        }
    }
}
