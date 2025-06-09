import CoreGraphics
//
//  NerfStudioDataLoader.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/07.
//
import Foundation
import MLX
import UIKit
import ZIPFoundation
import simd

private struct TransformsData: Codable {

    let camera_model: String?
    let frames: [Frame]
    let ply_file_path: String

    let w: Int?
    let h: Int?
    let fl_x: Double?
    let fl_y: Double?
    let cx: Double?
    let cy: Double?
    let k1: Double?
    let k2: Double?
    let p1: Double?
    let p2: Double?
    func getIntrinsicMatrix() -> simd_double3x3? {
        if let fl_x, let fl_y, let cx, let cy {
            var P = matrix_double3x3()
            P.columns = (
                SIMD3<Double>(fl_x, 0, 0),
                SIMD3<Double>(0, fl_y, 0),
                SIMD3<Double>(cx, cy, 1),
            )
            return P
        } else {
            return nil
        }
    }
}

private struct Frame: Codable {
    let file_path: String
    let transform_matrix: [[Float]]
    let colmap_im_id: Int?
    let w: Int?
    let h: Int?
    let fl_x: Double?
    let fl_y: Double?
    let cx: Double?
    let cy: Double?
    let k1: Double?
    let k2: Double?
    let k3: Double?
    let p1: Double?
    let p2: Double?
    func getIntrinsicMatrix() -> simd_double3x3? {
        if let fl_x, let fl_y, let cx, let cy {
            var P = matrix_double3x3()
            P.columns = (
                SIMD3<Double>(fl_x, 0, 0),
                SIMD3<Double>(0, fl_y, 0),
                SIMD3<Double>(cx, cy, 1),
            )
            return P
        } else {
            return nil
        }
    }
}

class NerfStudioDataLoader {
    func unzipDataTo(zipUrl: URL, targetDirUrl: URL) throws {
        if FileManager.default.fileExists(atPath: targetDirUrl.path) {
            try FileManager.default.removeItem(at: targetDirUrl)
        }
        try FileManager.default.createDirectory(
            at: targetDirUrl,
            withIntermediateDirectories: true
        )
        try FileManager.default.unzipItem(at: zipUrl, to: targetDirUrl)
    }
    func parsePLY(url: URL) throws -> (
        [Float], [Float], [Float], [UInt8], [UInt8], [UInt8]
    ) {
        let fileData = try Data(contentsOf: url)
        guard
            let headerEnd = fileData.range(
                of: Data([UInt8]("end_header\n".utf8))
            )?.upperBound
        else {
            throw NSError(
                domain: "PLYParse",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No end_header"]
            )
        }

        let headerData = fileData[..<headerEnd]
        guard let headerStr = String(data: headerData, encoding: .ascii) else {
            throw NSError(
                domain: "PLYParse",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Header decode error"]
            )
        }
        let isAscii = headerStr.contains("format ascii")
        guard
            let vertexLine = headerStr.components(separatedBy: "\n").first(
                where: { $0.hasPrefix("element vertex") }),
            let vertexCount = Int(
                vertexLine.components(separatedBy: " ").last ?? ""
            )
        else {
            throw NSError(
                domain: "PLYParse",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "No vertex count"]
            )
        }

        var xs = [Float]()
        var ys = [Float]()
        var zs = [Float]()
        var reds = [UInt8]()
        var greens = [UInt8]()
        var blues = [UInt8]()

        if isAscii {
            // --- ASCII mode ---
            let asciiBody = fileData[headerEnd...]
            guard let bodyStr = String(data: asciiBody, encoding: .ascii) else {
                throw NSError(
                    domain: "PLYParse",
                    code: 5,
                    userInfo: [NSLocalizedDescriptionKey: "ASCII decode error"]
                )
            }
            let lines = bodyStr.split(whereSeparator: \.isNewline)
            for line in lines.prefix(vertexCount) {
                let vals = line.split(separator: " ")
                guard vals.count >= 6 else { continue }
                xs.append(Float(vals[0]) ?? 0)
                ys.append(Float(vals[1]) ?? 0)
                zs.append(Float(vals[2]) ?? 0)
                reds.append(UInt8(vals[3]) ?? 0)
                greens.append(UInt8(vals[4]) ?? 0)
                blues.append(UInt8(vals[5]) ?? 0)
            }
        } else {
            // --- BINARY mode ---
            let binaryStart = headerEnd
            let vertexStride = 4 + 4 + 4 + 1 + 1 + 1  // float*3 + uchar*3 = 15 bytes
            let binaryEnd = binaryStart + vertexCount * vertexStride
            guard binaryEnd <= fileData.count else {
                throw NSError(
                    domain: "PLYParse",
                    code: 4,
                    userInfo: [NSLocalizedDescriptionKey: "File too small"]
                )
            }
            for i in 0..<vertexCount {
                let offset = binaryStart + i * vertexStride
                let xData = fileData.subdata(in: (offset + 0)..<(offset + 4))
                let yData = fileData.subdata(in: (offset + 4)..<(offset + 8))
                let zData = fileData.subdata(in: (offset + 8)..<(offset + 12))
                let x = xData.withUnsafeBytes { $0.load(as: Float.self) }
                let y = yData.withUnsafeBytes { $0.load(as: Float.self) }
                let z = zData.withUnsafeBytes { $0.load(as: Float.self) }
                let r = fileData[offset + 12]
                let g = fileData[offset + 13]
                let b = fileData[offset + 14]
                xs.append(x)
                ys.append(y)
                zs.append(z)
                reds.append(r)
                greens.append(g)
                blues.append(b)
            }
        }

        return (xs, ys, zs, reds, greens, blues)
    }

    func loadPlyPointCloud(plyURL: URL) throws -> PointCloud {
        let (xs, ys, zs, reds, greens, blues) = try parsePLY(
            url: plyURL
        )
        let channels = [
            "R": MLXArray(reds).asType(.float32) / 255.0,
            "G": MLXArray(greens).asType(.float32) / 255.0,
            "B": MLXArray(blues).asType(.float32) / 255.0,
        ]
        let xArray = MLXArray(xs)[.ellipsis, .newAxis]
        let yArray = MLXArray(ys)[.ellipsis, .newAxis]
        let zArray = MLXArray(zs)[.ellipsis, .newAxis]
        let coords = MLX.concatenated([xArray, yArray, zArray], axis: 1)
        return PointCloud(coords: coords, channels: channels)
    }

    fileprivate func readFrameFile(jsonURL: URL) throws -> TransformsData {
        let data = try Data(contentsOf: jsonURL)
        let decoder = JSONDecoder()
        let transformData = try decoder.decode(TransformsData.self, from: data)
        return transformData
    }

    func unzipDataToTmp(zipUrl: URL, targetDirUrl: URL) throws {
        try FileManager.default.removeItem(at: targetDirUrl)
        try FileManager.default.createDirectory(
            at: targetDirUrl,
            withIntermediateDirectories: true
        )
        try FileManager.default.unzipItem(at: zipUrl, to: targetDirUrl)
    }

    func readImage(
        path: URL,
        resizeFactor: Double
    )
        -> (
            rgb: MLXArray, alpha: MLXArray, H: MLXArray, W: MLXArray
        )
    {
        func resizeImage(_ image: UIImage, scale: Double) -> UIImage {
            if scale == 1.0 { return image }
            let newSize = CGSize(
                width: image.size.width * scale,
                height: image.size.height * scale
            )
            UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
            image.draw(in: CGRect(origin: .zero, size: newSize))
            let resized = UIGraphicsGetImageFromCurrentImageContext() ?? image
            UIGraphicsEndImageContext()
            return resized
        }
        func loadRGBA(path: URL, scale: Double) -> MLXArray {
            guard let image = UIImage(contentsOfFile: path.path) else {
                fatalError("Can not load image: \(path)")
            }
            let img = resizeImage(image, scale: scale)
            guard let cgImage = img.cgImage else {
                fatalError("CGImage transform failed")
            }

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
                floatData.append(Float(rawData[offset + 3]) / 255.0)  // A
            }
            return MLXArray(floatData, [height, width, 4])
        }

        let rgba = loadRGBA(path: path, scale: resizeFactor)
        let rgb = rgba[.ellipsis, 0..<3]
        let alpha = rgba[.ellipsis, 3]
        let imageSize = rgb.shape

        return (
            rgb, alpha,
            MLXArray(Float(imageSize[0])),
            MLXArray(Float(imageSize[1]))
        )
    }

    fileprivate func readFrames(
        directoryURL: URL,
        transformsData: TransformsData,
        resizeFactor: Double,
        whiteBackground: Bool
    ) -> TrainData {
        var intrinsics: [MLXArray] = []
        var Hs: [MLXArray] = []
        var Ws: [MLXArray] = []
        var rgbArray: [MLXArray] = []
        var alphaArray: [MLXArray] = []
        var c2wArray: [MLXArray] = []
        for frame in transformsData.frames {
            guard
                let intrinsic =
                    (frame.getIntrinsicMatrix()
                    ?? transformsData.getIntrinsicMatrix())?
                    .toMLXArray()
            else {
                fatalError("Failed to load intrinsic matrix")
            }
            if resizeFactor != 1.0 {
                intrinsic[..<2, ..<3] *= resizeFactor
            }
            intrinsics.append(intrinsic)
            let (rgb, alpha, h, w) = readImage(
                path: directoryURL.appendingPathComponent(frame.file_path),
                resizeFactor: resizeFactor
            )
            rgbArray.append(rgb)
            alphaArray.append(alpha)
            Hs.append(h)
            Ws.append(w)
            var c2w = simd_double4x4()
            // Convert NerfStudio's camera(ie:OpenGL) to OpenCV
            for r in 0..<4 {
                for c in 0..<4 {
                    c2w[c][r] = Double(frame.transform_matrix[r][c])
                }
            }
            var w2c = c2w.inverse
            for r in 1...2 { for c in 0..<4 { w2c[c][r] *= -1 } }
            let c2w_opencv = w2c.inverse

            c2wArray.append(c2w_opencv.toMLXArray())
        }
        let ml_rgbs = MLX.stacked(rgbArray, axis: 0)
        let ml_alphas = MLX.stacked(alphaArray, axis: 0)
        let ml_final_rgbs =
            whiteBackground
            ? ml_alphas.expandedDimensions(axes: [-1]) * ml_rgbs
                + (1 - ml_alphas).expandedDimensions(axes: [-1]) : ml_rgbs
        let ml_Hs = MLX.stacked(Hs, axis: 0)
        let ml_Ws = MLX.stacked(Ws, axis: 0)
        let ml_c2ws = MLX.stacked(c2wArray, axis: 0)
        let ml_intrinsics: MLXArray = MLX.stacked(
            intrinsics
        )
        return TrainData(
            Hs: ml_Hs,
            Ws: ml_Ws,
            intrinsicArray: ml_intrinsics,
            c2wArray: ml_c2ws,
            rgbArray: MLX.stopGradient(ml_final_rgbs[.ellipsis, 0..<3]),
            alphaArray: MLX.stopGradient(ml_alphas),
            depthArray: nil
        )
    }

    func loadTrainDataAndPointCloud(
        directoryURL: URL,
        resizeFactor: Double,
        whiteBackground: Bool
    )
        throws -> (
            TrainData, PointCloud, TILE_SIZE_H_W
        )
    {
        let jsonURL =
            directoryURL
            .appendingPathComponent("transforms.json")
        let frameData = try readFrameFile(jsonURL: jsonURL)
        let pointCloud = try loadPlyPointCloud(
            plyURL: directoryURL.appendingPathComponent(frameData.ply_file_path)
        )
        let trainData = readFrames(
            directoryURL: directoryURL,
            transformsData: frameData,
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )
        return (
            trainData, pointCloud,
            TILE_SIZE_H_W(
                w: trainData.Ws[0].item(Int.self) / 4,
                h: trainData.Hs[0].item(Int.self) / 4
            )
        )
    }
}

class UserNerfStudioDataLoader: NerfStudioDataLoader, DataLoaderProtocol {
    let target_dir_name = "nerf_studio_user"
    func getDataDir() -> URL {
        let tmpPath = NSTemporaryDirectory()
        return URL(fileURLWithPath: tmpPath)
            .appendingPathComponent(target_dir_name)
    }

    func getTargetDir() -> URL {
        let tmpPath = NSTemporaryDirectory()
        return URL(fileURLWithPath: tmpPath)
            .appendingPathComponent(target_dir_name)
    }
    init(zipUrl: URL) {
        self.zipUrl = zipUrl
    }
    let zipUrl: URL
    func load(resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    ) {
        let tempDir = getDataDir()
        try unzipDataTo(zipUrl: self.zipUrl, targetDirUrl: tempDir)
        return try loadTrainDataAndPointCloud(
            directoryURL: getDataDir(),
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )
    }
}
