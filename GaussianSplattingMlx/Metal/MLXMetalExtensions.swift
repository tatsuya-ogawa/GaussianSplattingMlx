import MLX
import simd

extension MLXArray {
    func toFloat4x4() -> simd_float4x4 {
        let array = self.asArray(Float.self)
        return simd_float4x4(
            SIMD4<Float>(array[0], array[1], array[2], array[3]),
            SIMD4<Float>(array[4], array[5], array[6], array[7]),
            SIMD4<Float>(array[8], array[9], array[10], array[11]),
            SIMD4<Float>(array[12], array[13], array[14], array[15])
        )
    }
    
    func toFloat3Array() -> [SIMD3<Float>] {
        let array = self.asArray(Float.self)
        let count = self.shape[0]
        var result: [SIMD3<Float>] = []
        
        for i in 0..<count {
            let baseIdx = i * 3
            result.append(SIMD3<Float>(
                array[baseIdx],
                array[baseIdx + 1],
                array[baseIdx + 2]
            ))
        }
        
        return result
    }
    
    func toFloat4Array() -> [SIMD4<Float>] {
        let array = self.asArray(Float.self)
        let count = self.shape[0]
        var result: [SIMD4<Float>] = []
        
        for i in 0..<count {
            let baseIdx = i * 4
            result.append(SIMD4<Float>(
                array[baseIdx],
                array[baseIdx + 1],
                array[baseIdx + 2],
                array[baseIdx + 3]
            ))
        }
        
        return result
    }
    
    func toFloatArray() -> [Float] {
        return self.asArray(Float.self)
    }
}

extension simd_double4x4 {
    func toFloat4x4() -> simd_float4x4 {
        return simd_float4x4(
            SIMD4<Float>(Float(self[0][0]), Float(self[0][1]), Float(self[0][2]), Float(self[0][3])),
            SIMD4<Float>(Float(self[1][0]), Float(self[1][1]), Float(self[1][2]), Float(self[1][3])),
            SIMD4<Float>(Float(self[2][0]), Float(self[2][1]), Float(self[2][2]), Float(self[2][3])),
            SIMD4<Float>(Float(self[3][0]), Float(self[3][1]), Float(self[3][2]), Float(self[3][3]))
        )
    }
}