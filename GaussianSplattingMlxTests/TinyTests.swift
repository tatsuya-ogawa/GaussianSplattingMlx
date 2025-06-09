//
//  TinyTests.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/03.
//

import Foundation
import MLX
import Testing
import UIKit
import simd

@testable import GaussianSplattingMlx

struct TinyTests {
  @Test func test_indexing() throws {
    let data = MLXArray([1, 2, 3, 4, 5] as [Float])
    let indices = MLXArray([2, 1] as [Int])
    let ret = data[indices]
    #expect(ret.allClose(MLXArray([3, 2] as [Float])).item())
  }
  @Test func test_indexing_large() throws {
    let dataArray: [Float] = (0..<10_000).map { Float($0) }
    let indexArray: [Int32] = (0..<2000).map { _ in
      Int32(Int.random(in: 0..<10_000))
    }
    let data = MLXArray(dataArray)
    let indices = MLXArray(indexArray)
    let ret = data[indices]
    let expected: [Float] = indexArray.map { i in dataArray[Int(i)] }
    let expectedArray = MLXArray(expected)
    #expect(ret.allClose(expectedArray).item())
  }
  @Test func test_indexing_large_vector3() throws {
    let N = 10_000
    let D = 3

    // 1. Prepare a primitive array of type [N, 3]
    var dataArray: [Float] = []
    for i in 0..<N {
      dataArray.append(Float(i))
      dataArray.append(Float(i) + 0.1)
      dataArray.append(Float(i) + 0.2)
    }
    // 2. Random index array
    let indexArray: [Int32] = (0..<2000).map { _ in
      Int32(Int.random(in: 0..<N))
    }

    // 3. Convert to MLXArray
    let data = MLXArray(dataArray, [N, D])
    let indices = MLXArray(indexArray)

    // 4. MLXでindexing: [2000, 3]になるはず
    let ret = data[indices]

    // 5. 検証用: プリミティブ配列でground truth生成
    var expected: [Float] = []
    for idx in indexArray {
      let base = Int(idx) * D
      expected.append(dataArray[base])
      expected.append(dataArray[base + 1])
      expected.append(dataArray[base + 2])
    }
    let expectedArray = MLXArray(expected, [indexArray.count, D])

    #expect(ret.allClose(expectedArray).item())
  }
  @Test func test_conditional_slice_vector3() throws {
    let N = 10_000
    let D = 3

    // 1. ランダムデータ作成
    let dataArray: [Float] = (0..<(N * D)).map { _ in
      Float.random(in: 0..<1)
    }
    let data = MLXArray(dataArray, [N, D])

    // 2. 条件mask作成: 0番目成分が0.1以下
    let mask = data[.ellipsis, 0] .<= 0.1  // shape: [N]
    let indices: [Int] = mask.enumerated().filter { (i, d) in
      return d.item(Bool.self)
    }.map { (i, d) in
      return i
    }
    // 3. 条件抽出: [M, 3]（Mは該当数）
    //        let selected = data[mask]
    let selected = data[MLXArray(indices)]

    // 4. 検証: プリミティブでground truth
    var expected: [[Float]] = []
    for i in 0..<N {
      let x = dataArray[i * D]
      if x <= 0.1 {
        expected.append([
          dataArray[i * D],
          dataArray[i * D + 1],
          dataArray[i * D + 2],
        ])
      }
    }
    let expectedArray = MLXArray(
      expected.flatMap { $0 },
      [expected.count, D]
    )

    #expect(selected.allClose(expectedArray).item())
  }
  @Test func test_newAxis() {
    let data = MLXArray([1, 2, 3] as [Int], [3])
    #expect(data.shape == [3])
    let newAxisData = data[.newAxis]
    #expect(newAxisData.shape == [1, 3])
    let lastNewAxisData = data[.ellipsis, .newAxis]
    #expect(lastNewAxisData.shape == [3, 1])
  }
  @Test func test_outOfAxis() {
    let data = MLXArray([1, 2, 3, 4, 5, 6] as [Int], [3, 2])
    #expect(data.shape == [3, 2])
    data[3...] = MLXArray(100 as Int)
    print(data)
    #expect(
      data.allClose(MLXArray([1, 2, 3, 4, 5, 6] as [Int], [3, 2])).item()
    )
  }

  @Test func test_where() {
    let data = MLXArray([1, 2, 3, 4, 5, 6] as [Int])
    let filter = data .<= 3
    print(filter.shape)
    let arange = MLX.where(
      filter,
      MLXArray(0..<filter.shape[0]),
      MLXArray(Int32.max)
    )
    print(arange)
    let sorted = MLX.sorted(arange)
    print(sorted)
    let index = MLX.argMax(sorted)
    print(index)
    let result = sorted[0..<index.item(Int.self)]
    print(result)
  }
  @Test func test_simd() {
    var mat = simd_float4x4()
    for i in 0..<4 {
      for j in 0..<4 {
        mat[i, j] = Float(i * 10 + j)
      }
    }
    let v = simd_float4(1, 2, 3, 4)  // [1,2,3,4]（行ベクトル）

    // 2. simdでの演算（行ベクトル × 行列）
    let out_simd = v * mat  // simdは行ベクトル×行列
    #expect(out_simd == simd_float4(20.0, 120.0, 220.0, 320.0))

    let mlxMat = mat.toMLXArray()
    let mlxV = MLXArray([1, 2, 3, 4] as [Float])
    let out_mlx = mlxV.matmul(mlxMat)
    #expect(out_mlx.allClose(MLXArray([20.0, 120.0, 220.0, 320.0] as [Float])).item())
  }
  @Test func test_simd_matmul() {
    // 1. 2つの simd_float4x4 を作る
    var matA = simd_float4x4()
    var matB = simd_float4x4()
    for i in 0..<4 {
      for j in 0..<4 {
        matA[i, j] = Float(i * 10 + j)  // 0,1,2,3... 10,11,12...
        matB[i, j] = Float(100 + i * 10 + j)  // 100,101,102... 110,111...
      }
    }
    // 2. simdでの行列積
    let out_simd = matA * matB  // simdは「右から左」：matA×matB

    // 3. MLXArrayに変換（row-majorでOK）
    let mlxA = matA.toMLXArray()
    let mlxB = matB.toMLXArray()
    let out_mlx = mlxA.matmul(mlxB)

    // 4. 結果比較（要素全一致チェック）
    for i in 0..<4 {
      for j in 0..<4 {
        #expect(out_simd[j, i] == out_mlx[i, j].item())
      }
    }
  }
  func createTestImage(width: Int, height: Int) -> UIImage {
    let format = UIGraphicsImageRendererFormat()
    format.scale = 1
    let renderer = UIGraphicsImageRenderer(
      size: CGSize(width: width, height: height), format: format)
    return renderer.image { ctx in
      UIColor.red.setFill()
      ctx.fill(CGRect(x: 0, y: 0, width: width, height: height / 2))
      UIColor.blue.setFill()
      ctx.fill(CGRect(x: 0, y: height / 2, width: width, height: height / 2))
    }
  }

  // 2. 画像を一時ファイルに保存
  func saveImageToTmp(_ image: UIImage, filename: String = "testimg.png") -> URL {
    print(image.size)
    let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(filename)
    try! image.pngData()!.write(to: url)
    return url
  }

  // 3. 画像をMLXArrayに変換
  func loadRGB(path: URL, scale: Double = 1.0) -> MLXArray {
    guard let image = UIImage(contentsOfFile: path.path) else {
      fatalError("画像が読み込めません: \(path)")
    }
    let img = image  // scaleは省略
    guard let cgImage = img.cgImage else { fatalError("CGImage変換失敗") }
    let width = cgImage.width
    let height = cgImage.height
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    let totalBytes = height * bytesPerRow
    var rawData = [UInt8](repeating: 0, count: totalBytes)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
      data: &rawData,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )
    context?.draw(
      cgImage,
      in: CGRect(x: 0, y: 0, width: width, height: height)
    )

    var floatData = [Float]()
    for i in 0..<(width * height) {
      let offset = i * 4
      floatData.append(Float(rawData[offset + 0]) / 255.0)  // R
      floatData.append(Float(rawData[offset + 1]) / 255.0)  // G
      floatData.append(Float(rawData[offset + 2]) / 255.0)  // B
    }
    return MLXArray(floatData, [height, width, 3])
  }
  @Test func test_loadImage() {
    let width = 4
    let height = 4
    let img = createTestImage(width: width, height: height)
    let tmpURL = saveImageToTmp(img)
    let arr = loadRGB(path: tmpURL)
    #expect(arr.shape == [height, width, 3])
    print(arr)
  }
}
