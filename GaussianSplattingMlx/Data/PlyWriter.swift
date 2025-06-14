import Foundation
import MLX

extension Array {
    func chunked(into chunkSize: Int) -> [[Element]] {
        stride(from: 0, to: count, by: chunkSize).map {
            Array(self[$0..<Swift.min($0 + chunkSize, count)])
        }
    }
}
struct GaussianPlyData {
    let positions: [[Float]]  // N×3
    let features_dc: [[Float]]  // N×3
    let features_rest: [[[Float]]]  // N×M*3
    let opacities: [Float]  // N
    let scales: [[Float]]  // N×3
    let rotations: [[Float]]  // N×4
}

class PlyWriter {
    /// Gaussian Splatting: binary_little_endian PLY
    static func writeGaussianBinary(
        positions: [[Float]],  // N×3
        features_dc: [[Float]],  // N×3
        features_rest: [[[Float]]],  // N×M×3
        opacities: [Float],  // N
        scales: [[Float]],  // N×3
        rotations: [[Float]],  // N×4
        to url: URL
    ) throws {
        let numPoints = positions.count
        let M = features_rest.first?.count ?? 0

        guard features_dc.count == numPoints,
              features_rest.count == numPoints,
              opacities.count == numPoints,
              scales.count == numPoints,
              rotations.count == numPoints else {
            throw NSError(domain: "writeGaussianBinary", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "Attribute array size mismatch"])
        }

        var header = ""
        header += "ply\n"
        header += "format binary_little_endian 1.0\n"
        header += "comment features_rest_shape \(M) 3\n"
        header += "element vertex \(numPoints)\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        header += "property float f_dc_0\n"
        header += "property float f_dc_1\n"
        header += "property float f_dc_2\n"
        for i in 0..<(M * 3) {
            header += "property float f_rest_\(i)\n"
        }
        header += "property float opacity\n"
        header += "property float scale_0\n"
        header += "property float scale_1\n"
        header += "property float scale_2\n"
        header += "property float rot_0\n"
        header += "property float rot_1\n"
        header += "property float rot_2\n"
        header += "property float rot_3\n"
        header += "end_header\n"

        // Data: little-endian float32
        var data = Data()
        for i in 0..<numPoints {
            func appendFloat(_ v: Float) {
                var val = v.bitPattern.littleEndian
                withUnsafeBytes(of: &val) { data.append(contentsOf: $0) }
            }
            let p = positions[i]
            let dc = features_dc[i]
            let rest = features_rest[i].flatMap { $0 } // [M*3]
            let o = opacities[i]
            let s = scales[i]
            let r = rotations[i]
            appendFloat(p[0]); appendFloat(p[1]); appendFloat(p[2])
            appendFloat(dc[0]); appendFloat(dc[1]); appendFloat(dc[2])
            for val in rest { appendFloat(val) }
            appendFloat(o)
            appendFloat(s[0]); appendFloat(s[1]); appendFloat(s[2])
            appendFloat(r[0]); appendFloat(r[1]); appendFloat(r[2]); appendFloat(r[3])
        }
        guard let headerData = header.data(using: .ascii) else {
            throw NSError(domain: "writeGaussianBinary", code: -2,
                          userInfo: [NSLocalizedDescriptionKey: "Header encoding failed"])
        }
        var output = Data()
        output.append(headerData)
        output.append(data)
        let parentDir = url.deletingLastPathComponent()
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: parentDir.path) {
            try fileManager.createDirectory(at: parentDir, withIntermediateDirectories: true, attributes: nil)
        }
        try output.write(to: url)
    }

    /// MLXArray
    static func writeGaussianBinary(
        positions: MLXArray,       // N×3
        features_dc: MLXArray,     // N×3
        features_rest: MLXArray,   // N×M×3
        opacities: MLXArray,       // N
        scales: MLXArray,          // N×3
        rotations: MLXArray,       // N×4
        to url: URL
    ) throws {
        let pos = positions.reshaped([-1, 3]).asArray(Float.self)
        let dc = features_dc.reshaped([-1, 3]).asArray(Float.self)
        let shape = features_rest.shape
        let n = shape[0]
        let M = shape[1]
        let D = shape[2]
        precondition(D == 3, "features_rest must be [N, M, 3]")
        let rest = features_rest.reshaped([n, M * D]).asArray(Float.self)
        let opa = opacities.reshaped([-1]).asArray(Float.self)
        let scl = scales.reshaped([-1, 3]).asArray(Float.self)
        let rot = rotations.reshaped([-1, 4]).asArray(Float.self)
        // rest: [[Float]] (N, M*3)
        try writeGaussianBinary(
            positions: pos.chunked(into: 3),
            features_dc: dc.chunked(into: 3),
            features_rest: rest.chunked(into: M * D).map { $0.chunked(into: 3) },
            opacities: opa,
            scales: scl.chunked(into: 3),
            rotations: rot.chunked(into: 4),
            to: url
        )
    }

    /// Loader
    static func loadGaussianBinaryPLY(from url: URL) throws -> GaussianPlyData {
        let fileData = try Data(contentsOf: url)
        guard let headerEnd = fileData.range(of: Data([UInt8]("end_header\n".utf8)))?.upperBound else {
            throw NSError(domain: "PLYLoader", code: 1, userInfo: [NSLocalizedDescriptionKey: "No end_header"])
        }
        guard let headerStr = String(data: fileData[..<headerEnd], encoding: .ascii) else {
            throw NSError(domain: "PLYLoader", code: 2, userInfo: [NSLocalizedDescriptionKey: "Header parse error"])
        }
        var numPoints = 0
        var restShape: (M: Int, D: Int)? = nil
        var restCount = 0
        var fieldNames: [String] = []
        for line in headerStr.split(separator: "\n") {
            if line.hasPrefix("comment features_rest_shape") {
                let comps = line.split(separator: " ")
                if comps.count >= 4 {
                    restShape = (Int(comps[2]) ?? 0, Int(comps[3]) ?? 0)
                }
            }
            let parts = line.split(separator: " ")
            if parts.count >= 3 && parts[0] == "element" && parts[1] == "vertex" {
                numPoints = Int(parts[2]) ?? 0
            } else if parts.count == 3 && parts[0] == "property" && parts[1] == "float" {
                fieldNames.append(String(parts[2]))
                if parts[2].hasPrefix("f_rest_") {
                    restCount += 1
                }
            }
        }
        guard let (M, D) = restShape else {
            throw NSError(domain: "PLYLoader", code: 3, userInfo: [NSLocalizedDescriptionKey: "No features_rest_shape comment"])
        }
        func idx(_ name: String) -> Int { fieldNames.firstIndex(of: name)! }
        let floatsPerVertex = fieldNames.count
        let binData = fileData[headerEnd...]

        var positions = [[Float]]()
        var features_dc = [[Float]]()
        var features_rest = [[[Float]]]()
        var opacities = [Float]()
        var scales = [[Float]]()
        var rotations = [[Float]]()
        let stride = floatsPerVertex * MemoryLayout<Float>.size
        for i in 0..<numPoints {
            let offset = i * stride
            let base = binData.startIndex + offset
            var floats: [Float] = []
            for j in 0..<floatsPerVertex {
                let pos = base + j * 4
                let f = binData[pos..<(pos + 4)].withUnsafeBytes {
                    $0.load(as: Float.self)
                }
                floats.append(f)
            }
            positions.append([floats[idx("x")], floats[idx("y")], floats[idx("z")]])
            features_dc.append([floats[idx("f_dc_0")], floats[idx("f_dc_1")], floats[idx("f_dc_2")]])
            // restを [M,3]へ復元
            let restFlat = (0..<(M*D)).map { floats[idx("f_rest_\($0)")] }
            let restChunked = restFlat.chunked(into: D)
            features_rest.append(restChunked)
            opacities.append(floats[idx("opacity")])
            scales.append([floats[idx("scale_0")], floats[idx("scale_1")], floats[idx("scale_2")]])
            rotations.append([floats[idx("rot_0")], floats[idx("rot_1")], floats[idx("rot_2")], floats[idx("rot_3")]])
        }

        return GaussianPlyData(
            positions: positions,
            features_dc: features_dc,
            features_rest: features_rest,
            opacities: opacities,
            scales: scales,
            rotations: rotations
        )
    }

    // MLXArray版は、[N, M, 3]にreshapeするだけでOK
    static func loadGaussianBinaryPLYAsMLX(from url: URL) throws -> (
        positions: MLXArray,
        features_dc: MLXArray,
        features_rest: MLXArray,
        opacities: MLXArray,
        scales: MLXArray,
        rotations: MLXArray
    ) {
        let ply = try loadGaussianBinaryPLY(from: url)
        func toMLX2D(_ arr: [[Float]]) -> MLXArray {
            MLXArray(arr.flatMap { $0 }, [arr.count, arr.first?.count ?? 0])
        }
        func toMLX3D(_ arr: [[[Float]]]) -> MLXArray {
            let N = arr.count
            let M = arr.first?.count ?? 0
            let D = arr.first?.first?.count ?? 0
            return MLXArray(arr.flatMap { $0.flatMap { $0 } }, [N, M, D])
        }
        func toMLX1D(_ arr: [Float]) -> MLXArray {
            MLXArray(arr, [arr.count])
        }
        return (
            positions: toMLX2D(ply.positions),
            features_dc: toMLX2D(ply.features_dc),
            features_rest: toMLX3D(ply.features_rest),
            opacities: toMLX1D(ply.opacities),
            scales: toMLX2D(ply.scales),
            rotations: toMLX2D(ply.rotations)
        )
    }
}
