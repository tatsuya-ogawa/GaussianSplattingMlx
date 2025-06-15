//
//  simd+ext.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/02.
//

import MLX
import simd

extension matrix_float4x4 {
    func toMLXArray() -> MLXArray {
        var arr = [Float](repeating: 0, count: 16)
        // simd stores matrices column-major, so self[c][r] is element (r,c)
        for r in 0..<4 {
            for c in 0..<4 {
                arr[r * 4 + c] = self[c][r]
            }
        }
        return MLXArray(arr, [4, 4])
    }
    static func fromMlxArray(_ arr: MLXArray) -> simd_float4x4 {
        var m: simd_float4x4 = simd_float4x4()
        // simd stores matrices column-major, so self[c][r] is element (r,c)
        for r in 0..<4 {
            for c in 0..<4 {
                m[c][r] = arr[r][c].item(Float.self)
            }
        }
        return m
    }
}
extension simd_double3x3 {
    func toMLXArray() -> MLXArray {
        var arr = [Float](repeating: 0, count: 9)
        // simd stores matrices column-major, so self[c][r] is element (r,c)
        for r in 0..<3 {
            for c in 0..<3 {
                arr[r * 3 + c] = Float(self[c][r])
            }
        }
        return MLXArray(arr, [3, 3])
    }
}
extension simd_double4x4 {
    func toMLXArray() -> MLXArray {
        var arr = [Float](repeating: 0, count: 16)
        // simd stores matrices column-major, so self[c][r] is element (r,c)
        for r in 0..<4 {
            for c in 0..<4 {
                arr[r * 4 + c] = Float(self[c][r])
            }
        }
        return MLXArray(arr, [4, 4])
    }
    static func fromMlxArray(_ arr: MLXArray) -> simd_double4x4 {
        var m: simd_double4x4 = simd_double4x4()
        // simd stores matrices column-major, so self[c][r] is element (r,c)
        for r in 0..<4 {
            for c in 0..<4 {
                m[c][r] = Double(arr[r][c].item(Float.self))
            }
        }
        return m
    }
}
