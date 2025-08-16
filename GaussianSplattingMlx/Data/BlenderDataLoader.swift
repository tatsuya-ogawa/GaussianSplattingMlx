//
//  DataLoader.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/02.
//

import Foundation
import MLX
import UIKit
import simd

public struct SceneData: Codable {
    let backend: String
    let lightMode: String
    let fastMode: Bool
    let formatVersion: Int
    let channels: [String]
    let scale: Double
    let images: [ImageData]
    let bbox: [[Double]]

    enum CodingKeys: String, CodingKey {
        case backend
        case lightMode = "light_mode"
        case fastMode = "fast_mode"
        case formatVersion = "format_version"
        case channels
        case scale
        case images
        case bbox
    }
}

public struct ImageData: Codable {
    let intrinsic: [[Double]]
    let pose: [[Double]]
    let rgb: String
    let depth: String
    let alpha: String
    let maxDepth: Double
    let hw: [Int]

    enum CodingKeys: String, CodingKey {
        case intrinsic, pose, rgb, depth, alpha
        case maxDepth = "max_depth"
        case hw = "HW"
    }
}
class BlenderDemoDataLoader: DataLoaderProtocol {
    func readFileJSON<T: Decodable>(_ url: URL, as type: T.Type) throws -> T {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(T.self, from: data)
    }
    func readCamera(folder: URL) throws -> (
        rgbFiles: [URL], poses: [simd_double4x4], intrinsics: [simd_double4x4],
        maxDepth: Double
    ) {
        let infoURL = folder.appendingPathComponent("info.json")
        let sceneInfo = try readFileJSON(infoURL, as: SceneData.self)
        let maxDepth = sceneInfo.images.first?.maxDepth ?? 1.0

        var rgbFiles: [URL] = []
        var poses: [simd_double4x4] = []
        var intrinsics: [simd_double4x4] = []

        for img in sceneInfo.images {
            rgbFiles.append(folder.appendingPathComponent(img.rgb))
            var c2w = simd_double4x4()
            // Convert Blender's camera to OpenCV
            for r in 0..<4 { for c in 0..<4 { c2w[c][r] = img.pose[r][c] } }
            var w2c = c2w.inverse
            for r in 1...2 { for c in 0..<4 { w2c[c][r] *= -1 } }
            //            let c2w_opencv = w2c.inverse
            let c2w_opencv = w2c.inverse
            poses.append(c2w_opencv)
            // intrinsic
            var I = simd_double4x4()
            for r in 0..<3 { for c in 0..<3 { I[c][r] = img.intrinsic[r][c] } }
            I[3, 3] = 1
            intrinsics.append(I)
        }
        return (rgbFiles, poses, intrinsics, maxDepth)
    }

    func readImage(
        path: URL,
        simd_pose: simd_double4x4,
        simd_intrinsic: simd_double4x4,
        maxDepth: Double,
        resizeFactor: Double,
        whiteBackground: Bool
    )
        -> (MLXArray, MLXArray, MLXArray, MLXArray)
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
        func loadGray(path: URL, scale: Double) -> MLXArray {
            guard let image = UIImage(contentsOfFile: path.path) else {
                fatalError("Can not load image: \(path)")
            }
            let img = resizeImage(image, scale: scale)
            guard let cgImage = img.cgImage else {
                fatalError("CGImage transform failed")
            }

            let width = cgImage.width
            let height = cgImage.height
            let bytesPerPixel = 1
            let bytesPerRow = bytesPerPixel * width
            let totalBytes = height * bytesPerRow
            var rawData = [UInt8](repeating: 0, count: totalBytes)
            let colorSpace = CGColorSpaceCreateDeviceGray()
            let context = CGContext(
                data: &rawData,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            )
            context?.draw(
                cgImage,
                in: CGRect(x: 0, y: 0, width: width, height: height)
            )

            let floatData = rawData.map { Float($0) / 255.0 }
            return MLXArray(floatData, [height, width])
        }
        func loadRGB(path: URL, scale: Double) -> MLXArray {
            guard let image = UIImage(contentsOfFile: path.path) else {
                fatalError("Can not load image: \(path)")
            }
            let img = resizeImage(image, scale: scale)
            guard let cgImage = img.cgImage else {
                fatalError("CGImage transformation failed")
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
            }
            return MLXArray(floatData, [height, width, 3])
        }

        let rgb = loadRGB(path: path, scale: resizeFactor)
        let baseName = path.deletingPathExtension().lastPathComponent.split(
            separator: "_"
        ).first!
        let dir = path.deletingLastPathComponent()
        let depth =
            loadGray(
                path: dir.appendingPathComponent("\(baseName)_depth.png"),
                scale: resizeFactor
            ) * maxDepth
        let alpha = loadGray(
            path: dir.appendingPathComponent("\(baseName)_alpha.png"),
            scale: resizeFactor
        )

        let imageSize = rgb.shape
        let intrinsic = simd_intrinsic.toMLXArray()
        if resizeFactor != 1.0 {
            intrinsic[..<2, ..<3] *= resizeFactor
        }

        let cameraVec = MLX.concatenated(
            [
                MLXArray([
                    Float(imageSize[0]), Float(imageSize[1]),
                ]),
                intrinsic.reshaped([-1]),
                simd_pose.toMLXArray().reshaped([-1]),
            ],
            axis: 0
        )
        return (rgb, depth, alpha, cameraVec)
    }
    func parseCamera(params: MLXArray) -> (
        Hs: MLXArray, Ws: MLXArray, intrinsics: MLXArray, c2w: MLXArray
    ) {
        precondition(
            params.shape.count == 2 && params.shape[1] >= 34,
            "params must be [B,34]"
        )
        let B = params.shape[0]

        // H, Wは[0], [1]
        let Hs = params[.ellipsis, 0]
        let Ws = params[.ellipsis, 1]

        // intrinsics: [B, 16] → [B,4,4]
        let intr = params[.ellipsis, 2..<18].reshaped([B, 4, 4])

        // c2w: [B, 16] → [B,4,4]
        let c2w = params[.ellipsis, 18..<34].reshaped([B, 4, 4])

        return (Hs, Ws, intr, c2w)
    }

    func readAll(folder: URL, resizeFactor: Double, whiteBackground: Bool)
        throws -> TrainData
    {
        let (rgbFiles, poses, intrinsics, maxDepth) = try readCamera(
            folder: folder
        )
        var rgbs: [MLXArray] = []
        var depths: [MLXArray] = []
        var alphas: [MLXArray] = []
        var cameras: [MLXArray] = []
        for i in 0..<rgbFiles.count {
            Logger.shared.debug("read \(i+1)/\(rgbFiles.count) th image")
            let (rgb, depth, alpha, camera) = readImage(
                path: rgbFiles[i],
                simd_pose: poses[i],
                simd_intrinsic: intrinsics[i],
                maxDepth: maxDepth,
                resizeFactor: resizeFactor,
                whiteBackground: whiteBackground
            )
            rgbs.append(rgb)
            depths.append(depth)
            alphas.append(alpha)
            cameras.append(camera)
        }
        let ml_rgbs = MLX.stacked(rgbs, axis: 0)
        let ml_depths = MLX.stacked(depths, axis: 0)
        let ml_alphas = MLX.stacked(alphas, axis: 0)
        let ml_cameras = MLX.stacked(cameras, axis: 0)
        let final_rgbs =
            whiteBackground
            ? ml_alphas.expandedDimensions(axes: [-1]) * ml_rgbs
                + (1 - ml_alphas).expandedDimensions(axes: [-1]) : ml_rgbs
        let (Hs, Ws, intrinsicArray, c2wArray) = parseCamera(params: ml_cameras)
        return TrainData(
            Hs: Hs,
            Ws: Ws,
            intrinsicArray: intrinsicArray,
            c2wArray: c2wArray,
            rgbArray: MLX.stopGradient(final_rgbs[.ellipsis, 0..<3]),
            alphaArray: MLX.stopGradient(ml_alphas),
            depthArray: MLX.stopGradient(ml_depths)
        )
    }
    private func getTargetDir() -> URL {
        let tmpPath = NSTemporaryDirectory()
        return URL(fileURLWithPath: tmpPath)
            .appendingPathComponent(WORKING_DIRECTORY)
            .appendingPathComponent("demo_data")
    }
    private func getDataDir() -> URL {
        return getTargetDir().appendingPathComponent("B075X65R3X")
    }

    func downloadDemoData() throws {
        let tempDir = getTargetDir()
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let infoJsonURL = getDataDir().appendingPathComponent("info.json")
        if FileManager.default.fileExists(atPath: infoJsonURL.path) {
            Logger.shared.debug("already downloaded")
            return
        }
        let zipData = try Data(
            contentsOf: URL(
                string:
                    "https://raw.githubusercontent.com/hbb1/torch-splatting/refs/heads/main/B075X65R3X.zip"
            )!)
        let localZipURL = tempDir.appendingPathComponent("B075X65R3X.zip")
        try zipData.write(to: localZipURL)
        try FileManager.default.unzipItem(at: localZipURL, to: tempDir)
    }
    func load(resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    ) {
        try downloadDemoData()
        let url = getDataDir()
        let dataLoader = BlenderDemoDataLoader()
        let data = try dataLoader.readAll(
            folder: url,
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )
        let pointCloud = getPointCloudsFromTrainData(trainData: data)
        return (
            data, pointCloud,
            TILE_SIZE_H_W(
                w: data.Ws[0].item(Int.self) / 4,
                h: data.Hs[0].item(Int.self) / 4
            )
        )
    }
}
