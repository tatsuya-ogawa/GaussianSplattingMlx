//
//  DataLoaderTests.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/02.
//

import Foundation
import MLX
import MLXOptimizers
import Testing

@testable import GaussianSplattingMlx

@Suite struct DataLoaderTests {
  @Test
  func test_readCamera_TestData() throws {
    let testBundle = Bundle.main
    guard
      let url = testBundle.url(
        forResource: "info",
        withExtension: "json",
        //                subdirectory: "TestData"
      )?.deletingLastPathComponent()
    else {
      fatalError("TestData/info.json not found")
    }
    let loader = BlenderDemoDataLoader()
    let (rgbFiles, poses, intrinsics, maxDepth) = try loader.readCamera(
      folder: url
    )
    print(rgbFiles)
    #expect(!rgbFiles.isEmpty)
    #expect(poses.count == rgbFiles.count)
    #expect(intrinsics.count == rgbFiles.count)
    #expect(maxDepth > 0)
  }
}
