import CoreGraphics
import Foundation
import MLX
import UIKit
import simd

struct PinholeIntrinsics {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double

    func scaled(by scale: Double) -> PinholeIntrinsics {
        PinholeIntrinsics(
            fx: fx * scale,
            fy: fy * scale,
            cx: cx * scale,
            cy: cy * scale
        )
    }

    func toSimdMatrix() -> simd_double3x3 {
        var matrix = simd_double3x3()
        matrix.columns = (
            SIMD3<Double>(fx, 0, 0),
            SIMD3<Double>(0, fy, 0),
            SIMD3<Double>(cx, cy, 1)
        )
        return matrix
    }
}

struct CameraDistortion {
    let k1: Double
    let k2: Double
    let k3: Double
    let p1: Double
    let p2: Double

    var isIdentity: Bool {
        max(abs(k1), abs(k2), abs(k3), abs(p1), abs(p2)) < 1e-12
    }

    var nonIdentity: CameraDistortion? {
        isIdentity ? nil : self
    }
}

enum ImageRGBAReader {
    static func readImage(
        path: URL,
        resizeFactor: Double,
        intrinsics: PinholeIntrinsics? = nil,
        distortion: CameraDistortion? = nil
    )
        -> (
            rgb: MLXArray, alpha: MLXArray, H: MLXArray, W: MLXArray
        )
    {
        var (rawData, width, height) = loadRawRGBA(path: path, scale: resizeFactor)
        if let intrinsics, let distortion = distortion?.nonIdentity {
            rawData = undistortRGBA(
                rawData,
                width: width,
                height: height,
                intrinsics: intrinsics.scaled(by: resizeFactor),
                distortion: distortion
            )
        }

        var floatData = [Float]()
        floatData.reserveCapacity(width * height * 4)
        for i in 0..<(width * height) {
            let offset = i * 4
            floatData.append(Float(rawData[offset + 0]) / 255.0)
            floatData.append(Float(rawData[offset + 1]) / 255.0)
            floatData.append(Float(rawData[offset + 2]) / 255.0)
            floatData.append(Float(rawData[offset + 3]) / 255.0)
        }

        let rgba = MLXArray(floatData, [height, width, 4])
        let rgb = rgba[.ellipsis, 0..<3]
        let alpha = rgba[.ellipsis, 3]
        return (
            rgb, alpha,
            MLXArray(Float(height)),
            MLXArray(Float(width))
        )
    }

    static func readGray(path: URL, resizeFactor: Double) -> MLXArray {
        guard let image = UIImage(contentsOfFile: path.path) else {
            fatalError("Can not load image: \(path)")
        }
        let img = resizeImage(image, scale: resizeFactor)
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
        guard
            let context = CGContext(
                data: &rawData,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            )
        else {
            fatalError("Failed to create image context")
        }
        context.draw(
            cgImage,
            in: CGRect(x: 0, y: 0, width: width, height: height)
        )

        let floatData = rawData.map { Float($0) / 255.0 }
        return MLXArray(floatData, [height, width])
    }

    private static func resizeImage(_ image: UIImage, scale: Double) -> UIImage {
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

    private static func loadRawRGBA(path: URL, scale: Double) -> (
        data: [UInt8], width: Int, height: Int
    ) {
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
        guard
            let context = CGContext(
                data: &rawData,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else {
            fatalError("Failed to create image context")
        }
        context.draw(
            cgImage,
            in: CGRect(x: 0, y: 0, width: width, height: height)
        )
        return (rawData, width, height)
    }

    private static func undistortRGBA(
        _ source: [UInt8],
        width: Int,
        height: Int,
        intrinsics: PinholeIntrinsics,
        distortion: CameraDistortion
    ) -> [UInt8] {
        guard width > 0, height > 0 else { return source }

        var output = [UInt8](repeating: 0, count: source.count)
        let fx = intrinsics.fx
        let fy = intrinsics.fy
        let cx = intrinsics.cx
        let cy = intrinsics.cy

        for y in 0..<height {
            for x in 0..<width {
                let undistortedX = (Double(x) - cx) / fx
                let undistortedY = (Double(y) - cy) / fy
                let r2 = undistortedX * undistortedX + undistortedY * undistortedY
                let r4 = r2 * r2
                let r6 = r4 * r2
                let radial = 1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6
                let tangentialX =
                    2.0 * distortion.p1 * undistortedX * undistortedY
                    + distortion.p2 * (r2 + 2.0 * undistortedX * undistortedX)
                let tangentialY =
                    distortion.p1 * (r2 + 2.0 * undistortedY * undistortedY)
                    + 2.0 * distortion.p2 * undistortedX * undistortedY
                let distortedX = undistortedX * radial + tangentialX
                let distortedY = undistortedY * radial + tangentialY
                let sourceX = fx * distortedX + cx
                let sourceY = fy * distortedY + cy

                let destinationOffset = (y * width + x) * 4
                sampleBilinearRGBA(
                    source,
                    width: width,
                    height: height,
                    sourceX: sourceX,
                    sourceY: sourceY,
                    into: &output,
                    offset: destinationOffset
                )
            }
        }

        return output
    }

    private static func sampleBilinearRGBA(
        _ source: [UInt8],
        width: Int,
        height: Int,
        sourceX: Double,
        sourceY: Double,
        into output: inout [UInt8],
        offset: Int
    ) {
        guard sourceX >= 0,
            sourceY >= 0,
            sourceX <= Double(width - 1),
            sourceY <= Double(height - 1)
        else {
            output[offset + 0] = 0
            output[offset + 1] = 0
            output[offset + 2] = 0
            output[offset + 3] = 0
            return
        }

        let x0 = Int(floor(sourceX))
        let y0 = Int(floor(sourceY))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)
        let dx = sourceX - Double(x0)
        let dy = sourceY - Double(y0)
        let w00 = (1.0 - dx) * (1.0 - dy)
        let w10 = dx * (1.0 - dy)
        let w01 = (1.0 - dx) * dy
        let w11 = dx * dy

        let offset00 = (y0 * width + x0) * 4
        let offset10 = (y0 * width + x1) * 4
        let offset01 = (y1 * width + x0) * 4
        let offset11 = (y1 * width + x1) * 4

        for channel in 0..<4 {
            let value =
                Double(source[offset00 + channel]) * w00
                + Double(source[offset10 + channel]) * w10
                + Double(source[offset01 + channel]) * w01
                + Double(source[offset11 + channel]) * w11
            output[offset + channel] = UInt8(max(0, min(255, Int(round(value)))))
        }
    }
}
