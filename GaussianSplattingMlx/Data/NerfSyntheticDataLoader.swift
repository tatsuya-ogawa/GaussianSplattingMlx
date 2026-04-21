//
//  NerfSyntheticDataLoader.swift
//  GaussianSplattingMlx
//
//  NeRF Synthetic (Blender) dataset loader.
//  Dataset format: transforms_{train,val,test}.json + RGBA PNG images.
//

import Foundation
import MLX
import MLXRandom
import UIKit
import ZIPFoundation
import simd

// MARK: - JSON Schema

private struct NerfSyntheticTransforms: Codable {
    let camera_angle_x: Double
    let frames: [NerfSyntheticFrame]
}

private struct NerfSyntheticFrame: Codable {
    let file_path: String
    let rotation: Double?
    let transform_matrix: [[Double]]
}

// MARK: - Data Loader

class NerfSyntheticDataLoader {

    func getDirectoryURL() -> URL {
        fatalError("Subclass must implement getDirectoryURL")
    }

    func getOriginalImageSize() throws -> (width: Int, height: Int) {
        let directoryURL = getDirectoryURL()
        let transforms = try readTransforms(
            directoryURL: directoryURL, split: "train")
        guard let firstFrame = transforms.frames.first else {
            throw NSError(
                domain: "NerfSynthetic", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No frames found"])
        }
        let imageURL = resolveImagePath(
            directoryURL: directoryURL, filePath: firstFrame.file_path)
        guard let image = UIImage(contentsOfFile: imageURL.path) else {
            throw NSError(
                domain: "NerfSynthetic", code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey: "Cannot load image: \(imageURL)"
                ])
        }
        return (
            width: Int(image.size.width), height: Int(image.size.height)
        )
    }

    // MARK: - JSON Parsing

    private func readTransforms(directoryURL: URL, split: String) throws
        -> NerfSyntheticTransforms
    {
        let jsonURL = directoryURL.appendingPathComponent(
            "transforms_\(split).json")
        let data = try Data(contentsOf: jsonURL)
        return try JSONDecoder().decode(
            NerfSyntheticTransforms.self, from: data)
    }

    // MARK: - Image Path Resolution

    /// NeRF Synthetic file_path is like "./train/r_0" (no extension).
    private func resolveImagePath(directoryURL: URL, filePath: String)
        -> URL
    {
        var path = filePath
        // Strip leading "./"
        if path.hasPrefix("./") {
            path = String(path.dropFirst(2))
        }
        let url = directoryURL.appendingPathComponent(path)
        // Add .png extension if missing
        if url.pathExtension.isEmpty {
            return url.appendingPathExtension("png")
        }
        return url
    }

    // MARK: - Compute Intrinsics

    private func computeIntrinsics(
        cameraAngleX: Double, width: Int, height: Int, resizeFactor: Double
    ) -> PinholeIntrinsics {
        let fx = Double(width) / (2.0 * tan(cameraAngleX / 2.0))
        let fy = fx  // Square pixels assumed
        let cx = Double(width) / 2.0
        let cy = Double(height) / 2.0
        return PinholeIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy)
            .scaled(by: resizeFactor)
    }

    // MARK: - Load Frames

    private func readFrames(
        directoryURL: URL,
        transforms: NerfSyntheticTransforms,
        resizeFactor: Double,
        whiteBackground: Bool
    ) -> TrainData {
        var intrinsics: [MLXArray] = []
        var Hs: [MLXArray] = []
        var Ws: [MLXArray] = []
        var rgbArray: [MLXArray] = []
        var alphaArray: [MLXArray] = []
        var c2wArray: [MLXArray] = []

        for (i, frame) in transforms.frames.enumerated() {
            Logger.shared.debug(
                "read \(i + 1)/\(transforms.frames.count) th image")

            let imageURL = resolveImagePath(
                directoryURL: directoryURL, filePath: frame.file_path)
            let (rgb, alpha, h, w) = ImageRGBAReader.readImage(
                path: imageURL,
                resizeFactor: resizeFactor
            )

            // Compute intrinsics from camera_angle_x and image size
            let imgW = Int(w.item(Float.self))
            let imgH = Int(h.item(Float.self))
            let pinhole = computeIntrinsics(
                cameraAngleX: transforms.camera_angle_x,
                width: imgW, height: imgH,
                resizeFactor: 1.0  // already applied by readImage
            )
            intrinsics.append(pinhole.toSimdMatrix().toMLXArray())

            rgbArray.append(rgb)
            alphaArray.append(alpha)
            Hs.append(h)
            Ws.append(w)

            // Parse transform_matrix (OpenGL convention) -> OpenCV
            var c2w = simd_double4x4()
            for r in 0..<4 {
                for c in 0..<4 {
                    c2w[c][r] = frame.transform_matrix[r][c]
                }
            }
            var w2c = c2w.inverse
            // Negate rows 1,2 to convert OpenGL -> OpenCV
            for r in 1...2 { for c in 0..<4 { w2c[c][r] *= -1 } }
            let c2w_opencv = w2c.inverse

            c2wArray.append(c2w_opencv.toMLXArray())
        }

        let ml_rgbs = MLX.stacked(rgbArray, axis: 0)
        let ml_alphas = MLX.stacked(alphaArray, axis: 0)
        let ml_final_rgbs =
            whiteBackground
            ? ml_alphas.expandedDimensions(axes: [-1]) * ml_rgbs
                + (1 - ml_alphas).expandedDimensions(axes: [-1])
            : ml_rgbs
        let ml_Hs = MLX.stacked(Hs, axis: 0)
        let ml_Ws = MLX.stacked(Ws, axis: 0)
        let ml_c2ws = MLX.stacked(c2wArray, axis: 0)
        let ml_intrinsics = MLX.stacked(intrinsics)

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

    // MARK: - Initial Point Cloud

    private func extractCameraPositions(from trainData: TrainData) -> [SIMD3<Double>] {
        let numCameras = trainData.getNumCameras()
        var positions: [SIMD3<Double>] = []
        positions.reserveCapacity(numCameras)
        for i in 0..<numCameras {
            let c2w = trainData.c2wArray[i]
            positions.append(
                SIMD3<Double>(
                    Double(c2w[0][3].item(Float.self)),
                    Double(c2w[1][3].item(Float.self)),
                    Double(c2w[2][3].item(Float.self))
                )
            )
        }
        return positions
    }

    private func generateFallbackRandomPointCloud(
        center: SIMD3<Double>,
        radius: Double,
        numPoints: Int
    ) -> PointCloud {
        let sampledRadius = Float(radius * 0.2)
        let centerArray = MLXArray([
            Float(center.x), Float(center.y), Float(center.z),
        ])
        let directions = MLXRandom.normal([numPoints, 3])
        let norms = MLX.sqrt(
            MLX.sum(directions * directions, axis: 1, keepDims: true))
        let normalized = directions / norms
        let radii = MLX.pow(
            MLXRandom.uniform(0.0..<1.0, [numPoints, 1]), Float(1.0 / 3.0)
        ) * sampledRadius
        let coords = normalized * radii + centerArray
        let colors = MLXArray.full([numPoints], values: MLXArray(0.5 as Float))
        let channels: [String: MLXArray] = [
            "R": colors,
            "G": colors,
            "B": colors,
        ]
        return PointCloud(coords: coords, channels: channels)
    }

    /// Build an initial point cloud from foreground silhouettes instead of a
    /// camera-centered random sphere. This keeps the initial Gaussians much
    /// closer to the observed object volume and reduces camera-near floaters.
    private func generateInitialPointCloud(
        trainData: TrainData, numPoints: Int = 16_384
    ) -> PointCloud {
        let cameraPositions = extractCameraPositions(from: trainData)
        guard !cameraPositions.isEmpty else {
            return generateFallbackRandomPointCloud(
                center: SIMD3<Double>(0, 0, 0),
                radius: 1.0,
                numPoints: numPoints
            )
        }

        var sceneCenter = SIMD3<Double>(0, 0, 0)
        for position in cameraPositions {
            sceneCenter += position
        }
        sceneCenter /= Double(cameraPositions.count)

        var cameraRadius = 0.0
        for position in cameraPositions {
            cameraRadius += simd_length(position - sceneCenter)
        }
        cameraRadius /= Double(cameraPositions.count)
        let defaultPointCloud = generateFallbackRandomPointCloud(
            center: sceneCenter,
            radius: max(cameraRadius, 1e-3),
            numPoints: numPoints
        )

        let numCameras = trainData.getNumCameras()
        let perCameraBudget = max(64, numPoints / max(numCameras, 1))
        let stride = max(
            2,
            Int(
                sqrt(
                    Double(trainData.Hs[0].item(Int.self) * trainData.Ws[0].item(Int.self))
                        / Double(max(perCameraBudget, 1))
                )
            )
        )
        let depthJitter = max(cameraRadius * 0.08, 1e-3)
        let tangentJitter = max(cameraRadius * 0.01, 1e-4)

        var coords: [Float] = []
        var reds: [Float] = []
        var greens: [Float] = []
        var blues: [Float] = []
        coords.reserveCapacity(numPoints * 3)
        reds.reserveCapacity(numPoints)
        greens.reserveCapacity(numPoints)
        blues.reserveCapacity(numPoints)

        for cameraIndex in 0..<numCameras {
            let H = trainData.Hs[cameraIndex].item(Int.self)
            let W = trainData.Ws[cameraIndex].item(Int.self)
            let alpha = trainData.alphaArray[cameraIndex].asArray(Float.self)
            let rgb = trainData.rgbArray[cameraIndex].asArray(Float.self)
            let intrinsic = trainData.intrinsicArray[cameraIndex]
            let fx = Double(intrinsic[0][0].item(Float.self))
            let fy = Double(intrinsic[1][1].item(Float.self))
            let cx = Double(intrinsic[0][2].item(Float.self))
            let cy = Double(intrinsic[1][2].item(Float.self))
            let c2w = simd_double4x4.fromMlxArray(trainData.c2wArray[cameraIndex])
            let origin = SIMD3<Double>(c2w[3][0], c2w[3][1], c2w[3][2])
            let rotation = simd_double3x3(
                SIMD3<Double>(c2w[0][0], c2w[0][1], c2w[0][2]),
                SIMD3<Double>(c2w[1][0], c2w[1][1], c2w[1][2]),
                SIMD3<Double>(c2w[2][0], c2w[2][1], c2w[2][2])
            )

            var candidates: [(Int, Int)] = []
            candidates.reserveCapacity(perCameraBudget * 2)
            var y = 0
            while y < H {
                var x = 0
                while x < W {
                    let pixelIndex = y * W + x
                    if alpha[pixelIndex] > 0.6 {
                        candidates.append((x, y))
                    }
                    x += stride
                }
                y += stride
            }
            if candidates.isEmpty { continue }
            if candidates.count > perCameraBudget {
                candidates.shuffle()
                candidates = Array(candidates.prefix(perCameraBudget))
            }

            for (x, y) in candidates {
                let pixelX = Double(x) + 0.5
                let pixelY = Double(y) + 0.5
                let cameraDirection = simd_normalize(
                    SIMD3<Double>(
                        (pixelX - cx) / fx,
                        (pixelY - cy) / fy,
                        1.0
                    )
                )
                let worldDirection = simd_normalize(rotation * cameraDirection)
                let projectedDepth = simd_dot(sceneCenter - origin, worldDirection)
                if projectedDepth <= 0 {
                    continue
                }

                let alongRayOffset = Double.random(in: -depthJitter...depthJitter)
                let basePoint = origin + worldDirection * max(0.05, projectedDepth + alongRayOffset)
                var tangent = simd_cross(worldDirection, SIMD3<Double>(0, 1, 0))
                if simd_length_squared(tangent) < 1e-10 {
                    tangent = simd_cross(worldDirection, SIMD3<Double>(1, 0, 0))
                }
                tangent = simd_normalize(tangent)
                let bitangent = simd_normalize(simd_cross(worldDirection, tangent))
                let offset =
                    tangent * Double.random(in: -tangentJitter...tangentJitter)
                    + bitangent * Double.random(in: -tangentJitter...tangentJitter)
                let point = basePoint + offset

                let rgbIndex = (y * W + x) * 3
                coords.append(Float(point.x))
                coords.append(Float(point.y))
                coords.append(Float(point.z))
                reds.append(rgb[rgbIndex + 0])
                greens.append(rgb[rgbIndex + 1])
                blues.append(rgb[rgbIndex + 2])

                if reds.count >= numPoints {
                    break
                }
            }
            if reds.count >= numPoints {
                break
            }
        }

        guard !reds.isEmpty else {
            return defaultPointCloud
        }

        return PointCloud(
            coords: MLXArray(coords, [reds.count, 3]),
            channels: [
                "R": MLXArray(reds),
                "G": MLXArray(greens),
                "B": MLXArray(blues),
            ]
        )
    }

    // MARK: - Main Load

    func loadTrainDataAndPointCloud(
        directoryURL: URL,
        resizeFactor: Double,
        whiteBackground: Bool
    ) throws -> (TrainData, PointCloud, TILE_SIZE_H_W) {
        let transforms = try readTransforms(
            directoryURL: directoryURL, split: "train")

        Logger.shared.debug(
            "NeRF Synthetic: loading \(transforms.frames.count) training frames"
        )

        let trainData = readFrames(
            directoryURL: directoryURL,
            transforms: transforms,
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )

        let pointCloud = generateInitialPointCloud(trainData: trainData)

        return (
            trainData, pointCloud,
            TILE_SIZE_H_W(
                w: trainData.Ws[0].item(Int.self) / 4,
                h: trainData.Hs[0].item(Int.self) / 4
            )
        )
    }
}

// MARK: - User Data Loader (zip import)

class UserNerfSyntheticDataLoader: NerfSyntheticDataLoader,
    DataLoaderProtocol
{
    private let zipUrl: URL
    private let targetDirName = "nerf_synthetic_user"

    init(zipUrl: URL) {
        self.zipUrl = zipUrl
    }

    private func getDataDir() -> URL {
        let tmpPath = NSTemporaryDirectory()
        return URL(fileURLWithPath: tmpPath)
            .appendingPathComponent(targetDirName)
    }

    override func getDirectoryURL() -> URL {
        return getDataDir()
    }

    func load(resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    ) {
        let tempDir = getDataDir()
        // Clean and unzip
        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }
        try FileManager.default.createDirectory(
            at: tempDir, withIntermediateDirectories: true)
        try FileManager.default.unzipItem(at: zipUrl, to: tempDir)

        // Find the actual data directory (zip may contain a wrapper folder)
        let dataDir = try findDataDirectory(in: tempDir)

        return try loadTrainDataAndPointCloud(
            directoryURL: dataDir,
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )
    }

    /// Find the directory containing transforms_train.json
    /// (handles zips with a single wrapper directory)
    private func findDataDirectory(in dir: URL) throws -> URL {
        let trainJSON = dir.appendingPathComponent("transforms_train.json")
        if FileManager.default.fileExists(atPath: trainJSON.path) {
            return dir
        }
        // Check one level deep for a wrapper directory
        let contents = try FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.isDirectoryKey])
        for item in contents {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(
                atPath: item.path, isDirectory: &isDir)
            if isDir.boolValue {
                let nested = item.appendingPathComponent(
                    "transforms_train.json")
                if FileManager.default.fileExists(atPath: nested.path) {
                    return item
                }
            }
        }
        throw NSError(
            domain: "NerfSynthetic", code: 3,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "transforms_train.json not found in zip"
            ])
    }
}

class DemoNerfSyntheticDataLoader: NerfSyntheticDataLoader, DataLoaderProtocol {
    private static let prepareLock = NSLock()
    private let targetDirName = "nerf_synthetic_demo_hotdog"
    private let downloadURL = URL(
        string:
            "https://huggingface.co/datasets/togawa83/nerf_synthetic_zip/resolve/main/hotdog.zip"
    )!

    private func getDataDir() -> URL {
        let tmpPath = NSTemporaryDirectory()
        return URL(fileURLWithPath: tmpPath)
            .appendingPathComponent(targetDirName)
    }

    override func getDirectoryURL() -> URL {
        return getDataDir()
    }

    override func getOriginalImageSize() throws -> (width: Int, height: Int) {
        _ = try prepareDemoDataIfNeeded()
        return try super.getOriginalImageSize()
    }

    private func prepareDemoDataIfNeeded() throws -> URL {
        Self.prepareLock.lock()
        defer { Self.prepareLock.unlock() }

        let tempDir = getDataDir()
        let existingRoot = try? findDataDirectory(in: tempDir)
        if let existingRoot {
            return existingRoot
        }

        if FileManager.default.fileExists(atPath: tempDir.path) {
            try FileManager.default.removeItem(at: tempDir)
        }
        try FileManager.default.createDirectory(
            at: tempDir,
            withIntermediateDirectories: true
        )

        let zipData = try Data(contentsOf: downloadURL)
        let localZipURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("hotdog_demo_download.zip")
        try zipData.write(to: localZipURL)
        try FileManager.default.unzipItem(at: localZipURL, to: tempDir)
        return try findDataDirectory(in: tempDir)
    }

    private func findDataDirectory(in dir: URL) throws -> URL {
        let trainJSON = dir.appendingPathComponent("transforms_train.json")
        if FileManager.default.fileExists(atPath: trainJSON.path) {
            return dir
        }
        let contents = try FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.isDirectoryKey]
        )
        for item in contents {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: item.path, isDirectory: &isDir)
            if isDir.boolValue {
                let nested = item.appendingPathComponent("transforms_train.json")
                if FileManager.default.fileExists(atPath: nested.path) {
                    return item
                }
            }
        }
        throw NSError(
            domain: "NerfSynthetic",
            code: 4,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "transforms_train.json not found in hotdog demo zip"
            ]
        )
    }

    func load(resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    ) {
        let dataDir = try prepareDemoDataIfNeeded()
        return try loadTrainDataAndPointCloud(
            directoryURL: dataDir,
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )
    }
}
