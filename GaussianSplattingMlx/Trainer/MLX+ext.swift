//
//  MLX+ext.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/05.
//

import MLX
import UIKit

extension MLXArray {
  func toRGBToUIImage() -> UIImage? {
    var rgb = self
    let height = Int(rgb.shape[rgb.shape.count - 3])
    let width = Int(rgb.shape[rgb.shape.count - 2])
    if rgb.shape[rgb.shape.count - 1] == 3 {
      let alpha = MLXArray.ones(like: rgb[.ellipsis, 0..<1])
      rgb = MLX.concatenated([rgb, alpha], axis: -1)
    }
    var uint8Pixels = (rgb * 256).asMLXArray(dtype: .uint8).asArray(UInt8.self)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
      data: &uint8Pixels,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )
    guard let cgImage = context?.makeImage() else { return nil }
    return UIImage(cgImage: cgImage)
  }
}
