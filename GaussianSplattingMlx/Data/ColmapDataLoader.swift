import CoreGraphics
//
//  ColmapDataLoader.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/07.
//
import Foundation
import MLX
import UIKit
import ZIPFoundation
import simd

private enum CameraModel: Int {
    case SimplePinhole = 0
    case Pinhole = 1
    case SimpleRadial = 2
    case OpenCV = 3
}
private struct ImageWithPose {
    var fileURL: URL
    var pose: simd_double4x4
    var cameraId: UInt32
}
private struct ColmapCamera {
    var id: UInt32
    var width: UInt64
    var height: UInt64
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
    var k1: Double?
    var k2: Double?
    var p1: Double?
    var p2: Double?
    func getIntrinsicsMatrix() -> simd_double3x3 {
        var P = simd_double3x3()
        P.columns = (
            SIMD3<Double>(fx, 0, 0),
            SIMD3<Double>(0, fy, 0),
            SIMD3<Double>(cx, cy, 1),
        )
        return P
    }
}

private func readBinary<T: FixedWidthInteger>(_ handle: FileHandle) throws -> T
{
    let size = MemoryLayout<T>.size
    let data = try handle.read(upToCount: size) ?? Data()
    return data.withUnsafeBytes { $0.load(as: T.self) }
}
private func readBinaryDouble(_ handle: FileHandle) throws -> Double {
    let size = MemoryLayout<Double>.size
    let data = try handle.read(upToCount: size) ?? Data()
    return data.withUnsafeBytes { $0.load(as: Double.self) }
}

private func quatToRotMat(_ q: SIMD4<Double>) -> simd_double3x3 {
    let q = simd_quatd(ix: q[1], iy: q[2], iz: q[3], r: q[0])
    return simd_double3x3(q)
}

class ColmapDataLoader {
    func unzipDataTo(zipUrl:URL,targetDirUrl:URL) throws {
        if FileManager.default.fileExists(atPath: targetDirUrl.path) {
            try FileManager.default.removeItem(at: targetDirUrl)
        }
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
    private func colmapReadCamerasAndPoses(binRoot:URL,imageRoot:URL) throws -> (
        camMap: [UInt32: ColmapCamera],
        unorientedPoses: [ImageWithPose]
    ) {
        let fm = FileManager.default

        let camerasPath = binRoot.appendingPathComponent("cameras.bin")
        let imagesPath = binRoot.appendingPathComponent("images.bin")

        guard fm.fileExists(atPath: camerasPath.path),
            fm.fileExists(atPath: imagesPath.path)
        else {
            throw NSError(
                domain: "inputDataFromColmap",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Colmap files missing"]
            )
        }

        guard let camHandle = FileHandle(forReadingAtPath: camerasPath.path)
        else { throw NSError(domain: "", code: -1) }
        defer { try? camHandle.close() }

        // read camera
        let numCameras: UInt64 = try readBinary(camHandle)
        var cameras = [ColmapCamera]()
        var camMap: [UInt32: ColmapCamera] = [:]

        for _ in 0..<numCameras {
            let camId: UInt32 = try readBinary(camHandle)
            let modelInt: Int32 = try readBinary(camHandle)
            let model = CameraModel(rawValue: Int(modelInt)) ?? .Pinhole
            let width: UInt64 = try readBinary(camHandle)
            let height: UInt64 = try readBinary(camHandle)

            var fx = 0.0
            var fy = 0.0
            var cx = 0.0
            var cy = 0.0
            var k1: Double?
            var k2: Double?
            var p1: Double?
            var p2: Double?

            switch model {
            case .SimplePinhole:
                fx = try readBinaryDouble(camHandle)
                fy = fx
                cx = try readBinaryDouble(camHandle)
                cy = try readBinaryDouble(camHandle)
            case .Pinhole:
                fx = try readBinaryDouble(camHandle)
                fy = try readBinaryDouble(camHandle)
                cx = try readBinaryDouble(camHandle)
                cy = try readBinaryDouble(camHandle)
            case .SimpleRadial:
                fx = try readBinaryDouble(camHandle)
                fy = fx
                cx = try readBinaryDouble(camHandle)
                cy = try readBinaryDouble(camHandle)
                k1 = try readBinaryDouble(camHandle)
            case .OpenCV:
                fx = try readBinaryDouble(camHandle)
                fy = try readBinaryDouble(camHandle)
                cx = try readBinaryDouble(camHandle)
                cy = try readBinaryDouble(camHandle)
                k1 = try readBinaryDouble(camHandle)
                k2 = try readBinaryDouble(camHandle)
                p1 = try readBinaryDouble(camHandle)
                p2 = try readBinaryDouble(camHandle)
            }

            let cam = ColmapCamera(
                id: camId,
                width: width,
                height: height,
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                k1: k1,
                k2: k2,
                p1: p1,
                p2: p2
            )
            cameras.append(cam)
            camMap[camId] = cam
        }

        // readpose
        guard let imgHandle = FileHandle(forReadingAtPath: imagesPath.path)
        else { throw NSError(domain: "", code: -1) }
        defer { try? imgHandle.close() }
        let numImages: UInt64 = try readBinary(imgHandle)
        var unorientedPoses = [ImageWithPose]()

        for _ in 0..<Int(numImages) {
            _ = try readBinary(imgHandle) as UInt32  // imageId

            let qvec = SIMD4<Double>(
                try readBinaryDouble(imgHandle),
                try readBinaryDouble(imgHandle),
                try readBinaryDouble(imgHandle),
                try readBinaryDouble(imgHandle)
            )
            let R = quatToRotMat(qvec)
            let t = SIMD3<Double>(
                try readBinaryDouble(imgHandle),
                try readBinaryDouble(imgHandle),
                try readBinaryDouble(imgHandle)
            )
            let Rinv = R.transpose
            let Tinv = -(Rinv * t)

            let camId: UInt32 = try readBinary(imgHandle)
            var chars: [UInt8] = []
            while true {
                let ch = try imgHandle.read(upToCount: 1) ?? Data()
                if ch.isEmpty || ch[0] == 0 { break }
                chars.append(ch[0])
            }

            guard let filePath = String(bytes: chars, encoding: .utf8) else {
                fatalError("Invalid file path")
            }
            let fileURL = imageRoot.appendingPathComponent(filePath)

            // 4x4 pose
            var pose = simd_double4x4(1)
            pose.columns.0 = SIMD4<Double>(
                Rinv[0, 0],
                Rinv[0, 1],
                Rinv[0, 2],
                0
            )
            pose.columns.1 = SIMD4<Double>(
                Rinv[1, 0],
                Rinv[1, 1],
                Rinv[1, 2],
                0
            )
            pose.columns.2 = SIMD4<Double>(
                Rinv[2, 0],
                Rinv[2, 1],
                Rinv[2, 2],
                0
            )
            pose.columns.3 = SIMD4<Double>(Tinv[0], Tinv[1], Tinv[2], 1)

            // skip rest of points
            let numPoints2D: UInt64 = try readBinary(imgHandle)
            for _ in 0..<numPoints2D {
                _ = try readBinaryDouble(imgHandle)  // x
                _ = try readBinaryDouble(imgHandle)  // y
                _ = try readBinary(imgHandle) as UInt64  // point3D ID
            }
            unorientedPoses.append(
                ImageWithPose(fileURL: fileURL, pose: pose, cameraId: camId)
            )
        }

        return (camMap: camMap, unorientedPoses: unorientedPoses)
    }
    private func readFrames(
        poses: [ImageWithPose],
        resizeFactor: Double,
        whiteBackground: Bool
    ) -> (
        Hs: MLXArray,
        Ws: MLXArray,
        rgbArray: MLXArray,
        alphaArray: MLXArray
    ) {
        var Hs: [MLXArray] = []
        var Ws: [MLXArray] = []
        var rgbArray: [MLXArray] = []
        var alphaArray: [MLXArray] = []
        for pose in poses {
            let (rgb, alpha, h, w) = readImage(
                path: pose.fileURL,
                resizeFactor: resizeFactor
            )
            rgbArray.append(rgb)
            alphaArray.append(alpha)
            Hs.append(h)
            Ws.append(w)
        }
        let ml_rgbs = MLX.stacked(rgbArray, axis: 0)
        let ml_alphas = MLX.stacked(alphaArray, axis: 0)
        let ml_final_rgbs =
            whiteBackground
            ? ml_alphas.expandedDimensions(axes: [-1]) * ml_rgbs
                + (1 - ml_alphas).expandedDimensions(axes: [-1]) : ml_rgbs
        let ml_Hs = MLX.stacked(Hs, axis: 0)
        let ml_Ws = MLX.stacked(Ws, axis: 0)
        return (ml_Hs, ml_Ws, ml_final_rgbs, ml_alphas)
    }
    func readBinary<T: FixedWidthInteger>(_ handle: FileHandle) throws -> T {
        let size = MemoryLayout<T>.size
        let data = try handle.read(upToCount: size) ?? Data()
        if data.count != size {
            throw NSError(
                domain: "readBinary",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Not enough data"]
            )
        }
        return data.withUnsafeBytes { $0.load(as: T.self) }
    }

    func readBinaryDouble(_ handle: FileHandle) throws -> Double {
        let size = MemoryLayout<Double>.size
        let data = try handle.read(upToCount: size) ?? Data()
        if data.count != size {
            throw NSError(
                domain: "readBinaryDouble",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Not enough data"]
            )
        }
        return data.withUnsafeBytes { $0.load(as: Double.self) }
    }

    private func colmapReadPointSet(points3DUrl: URL) throws -> (
        points: [[Double]], colors: [[UInt8]]
    ) {
        guard let handle = FileHandle(forReadingAtPath: points3DUrl.path) else {
            throw NSError(
                domain: "colmapReadPointSet",
                code: -1,
                userInfo: [
                    NSLocalizedDescriptionKey: "Can not open \(points3DUrl)"
                ]
            )
        }
        defer { try? handle.close() }

        // 1. Read number of points
        let numPoints: UInt64 = try readBinary(handle)
        print("Reading \(numPoints) points")

        var points = Array(repeating: [0.0, 0.0, 0.0], count: Int(numPoints))
        var colors = Array(
            repeating: [UInt8](repeating: 0, count: 3),
            count: Int(numPoints)
        )

        for i in 0..<Int(numPoints) {
            // point id
            _ = try readBinary(handle) as UInt64
            // 3D point
            points[i][0] = try readBinaryDouble(handle)
            points[i][1] = try readBinaryDouble(handle)
            points[i][2] = try readBinaryDouble(handle)
            // Color
            colors[i][0] = try readBinary(handle) as UInt8
            colors[i][1] = try readBinary(handle) as UInt8
            colors[i][2] = try readBinary(handle) as UInt8
            // error
            _ = try readBinaryDouble(handle)
            // trackLen
            let trackLen: UInt64 = try readBinary(handle)
            for _ in 0..<trackLen {
                _ = try readBinary(handle) as UInt32  // imageId
                _ = try readBinary(handle) as UInt32  // point2D Idx
            }
        }

        return (points: points, colors: colors)
    }
    func loadTrainDataAndPointCloud(binRoot:URL,imageRoot:URL,resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    ) {
        let (camMap, poses) = try colmapReadCamerasAndPoses(binRoot: binRoot,imageRoot: imageRoot)
        let intrinsics = MLX.stacked(
            poses.map { p in
                guard let camera = camMap[p.cameraId] else {
                    fatalError("No matching camera")
                }
                return camera.getIntrinsicsMatrix().toMLXArray()
            }
        )
        if resizeFactor != 1.0 {
            intrinsics[.ellipsis, ..<2, ..<3] *= resizeFactor
        }
        let c2ws = MLX.stacked(
            poses.map { p in
                return p.pose.toMLXArray()
            }
        )
        let (Hs, Ws, rgbArray, alphaArray) = readFrames(
            poses: poses,
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )
        let (points, colors) = try colmapReadPointSet(
            points3DUrl: binRoot.appendingPathComponent("points3D.bin")
        )
        let coords = MLX.stacked(
            points.map { p in
                return MLXArray(p.map { Float($0) })
            }
        )
        let channels: MLXArray = MLX.stacked(
            colors.map { c in
                return MLXArray(c).asType(.float32) / 255.0
            }
        )
        let pointCloud = PointCloud(
            coords: coords,
            channels: [
                "R": channels[.ellipsis, 0],
                "G": channels[.ellipsis, 1],
                "B": channels[.ellipsis, 2],
            ]
        )
        let trainData: TrainData = TrainData(
            Hs: Hs,
            Ws: Ws,
            intrinsicArray: intrinsics,
            c2wArray: c2ws,
            rgbArray: rgbArray,
            alphaArray: alphaArray,
            depthArray: nil
        )

        return (
            trainData, pointCloud,
            TILE_SIZE_H_W(
                w: Ws[0].item(Int.self) / 4,
                h: Hs[0].item(Int.self) / 4
            )
        )
    }
}
class UserColmapDataLoader: ColmapDataLoader,DataLoaderProtocol {
    let target_dir_name = "colmap_user"
    func getDataDir() -> URL {
        let tmpPath = NSTemporaryDirectory()
        return URL(fileURLWithPath: tmpPath)
            .appendingPathComponent(target_dir_name)
    }
    func getBinRoot() -> URL {
        return getDataDir().appendingPathComponent("colmap/sparse/0")
    }
    func getImageRoot() -> URL {
        return getDataDir().appendingPathComponent("images")
    }
    init(zipUrl: URL) {
        self.zipUrl = zipUrl
    }
    let zipUrl: URL
    func load(resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    ){
        let tempDir = getDataDir()
        try unzipDataTo(zipUrl: self.zipUrl, targetDirUrl: tempDir)
        return try loadTrainDataAndPointCloud(binRoot: getBinRoot(), imageRoot: getImageRoot(), resizeFactor: resizeFactor, whiteBackground: whiteBackground)
    }
}
class DemoColmapDataLoader: ColmapDataLoader,DataLoaderProtocol {
    let target_dir_name = "colmap_demo"
    func getDataDir() -> URL {
        let tmpPath = NSTemporaryDirectory()
        return URL(fileURLWithPath: tmpPath)
            .appendingPathComponent(target_dir_name)
    }
    func getBinRoot() -> URL {
        return getDataDir().appendingPathComponent("colmap/sparse/0")
    }
    func getImageRoot() -> URL {
        return getDataDir().appendingPathComponent("images")
    }
    func getDownloadRoot()->URL{
        return URL(fileURLWithPath: NSTemporaryDirectory())
    }
    func downloadDemoData() throws->URL?{
        let infoJsonURL = getDataDir().appendingPathComponent("sparse/0/cameras.bin")
        if FileManager.default.fileExists(atPath: infoJsonURL.path) {
            Logger.shared.debug("already downloaded")
            return nil
        }
        let tempDir = getDownloadRoot()
        try FileManager.default.createDirectory(
            at: tempDir,
            withIntermediateDirectories: true
        )
        let zipData = try Data(
            contentsOf: URL(
                string:
                    "https://raw.githubusercontent.com/tatsuya-ogawa/TinyGaussianSplattingDataset/refs/heads/main/colmap/lego.zip"
            )!
        )
        let localZipURL = tempDir.appendingPathComponent("lego.zip")
        try zipData.write(to: localZipURL)
        return localZipURL
    }
    func load(resizeFactor: Double, whiteBackground: Bool) throws -> (
        TrainData, PointCloud, TILE_SIZE_H_W
    ){
        let tempDir = getDataDir()
        if let localZipURL = try downloadDemoData(){
            try unzipDataTo(zipUrl: localZipURL, targetDirUrl: tempDir)
        }
     
        return try loadTrainDataAndPointCloud(binRoot: getBinRoot(), imageRoot: getImageRoot(), resizeFactor: resizeFactor, whiteBackground: whiteBackground)
    }
}
