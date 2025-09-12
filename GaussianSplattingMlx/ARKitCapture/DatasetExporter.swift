import Foundation
import CoreImage
import AVFoundation
import simd
import UIKit

// MARK: - Dataset Exporter for GaussianSplatting Training Data

class DatasetExporter {
    
    private let config: ARKitCaptureManager.DatasetConfig
    private let fileManager = FileManager.default
    
    init(config: ARKitCaptureManager.DatasetConfig) {
        self.config = config
    }
    
    // MARK: - Main Export Function
    
    func exportDataset(
        frames: [CapturedFrame],
        cameraTransforms: [simd_float4x4],
        cameraIntrinsics: [simd_float3x3]
    ) async throws {
        
        // Create dataset directory structure
        let datasetURL = try createDatasetDirectory()
        
        Logger.shared.info("Exporting \(frames.count) frames to: \(datasetURL.path)")
        
        // Export in parallel for better performance
        await withTaskGroup(of: Void.self) { group in
            
            // Export images and depth maps
            group.addTask {
                await self.exportFrames(frames, to: datasetURL)
            }
            
            // Export camera data
            group.addTask {
                try? self.exportCameraData(
                    transforms: cameraTransforms,
                    intrinsics: cameraIntrinsics,
                    to: datasetURL
                )
            }
            
            // Export dataset metadata
            group.addTask {
                try? self.exportMetadata(frames: frames, to: datasetURL)
            }
        }
        
        Logger.shared.info("Dataset export completed successfully")
    }
    
    // MARK: - Directory Management
    
    private func createDatasetDirectory() throws -> URL {
        let datasetURL = config.outputDirectory
            .appendingPathComponent(config.datasetName)
            .appendingPathComponent(ISO8601DateFormatter().string(from: Date()))
        
        // Create directory structure
        let subdirectories = ["images", "depth", "masks", "sparse"]
        
        try fileManager.createDirectory(at: datasetURL, withIntermediateDirectories: true)
        
        for subdir in subdirectories {
            try fileManager.createDirectory(
                at: datasetURL.appendingPathComponent(subdir),
                withIntermediateDirectories: true
            )
        }
        
        return datasetURL
    }
    
    // MARK: - Frame Export
    
    private func exportFrames(_ frames: [CapturedFrame], to datasetURL: URL) async {
        let imagesURL = datasetURL.appendingPathComponent("images")
        let depthURL = datasetURL.appendingPathComponent("depth")
        let masksURL = datasetURL.appendingPathComponent("masks")
        
        for (index, frame) in frames.enumerated() {
            do {
                // Export RGB image
                try await exportRGBImage(
                    frame.rgbImage,
                    to: imagesURL.appendingPathComponent(String(format: "%06d.jpg", index))
                )
                
                // Export depth map
                try await exportDepthMap(
                    frame.depthData.depthMap,
                    to: depthURL.appendingPathComponent(String(format: "%06d.tiff", index))
                )
                
                // Export confidence mask if available
                if let confidenceMap = frame.depthData.confidenceMap {
                    try await exportConfidenceMask(
                        confidenceMap,
                        to: masksURL.appendingPathComponent(String(format: "%06d.png", index))
                    )
                }
                
            } catch {
                Logger.shared.error("Failed to export frame \(index)", error: error)
            }
        }
    }
    
    private func exportRGBImage(_ ciImage: CIImage, to url: URL) async throws {
        let context = CIContext()
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        
        // Convert to JPEG with high quality
        guard let jpegData = context.jpegRepresentation(
            of: ciImage,
            colorSpace: colorSpace,
            options: [:]
        ) else {
            throw ARKitCaptureError.imageProcessingFailed
        }
        
        try jpegData.write(to: url)
    }
    
    private func exportDepthMap(_ depthBuffer: CVPixelBuffer, to url: URL) async throws {
        // Convert depth buffer to EXR format for high precision
        let ciImage = CIImage(cvPixelBuffer: depthBuffer)
        let context = CIContext()
        
        // Convert to TIFF format for high precision (EXR not directly available)
        guard let tiffData = context.tiffRepresentation(
            of: ciImage,
            format: .Rf,  // Single channel float
            colorSpace: CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)!
        ) else {
            throw ARKitCaptureError.imageProcessingFailed
        }
        
        try tiffData.write(to: url)
    }
    
    private func exportConfidenceMask(_ confidenceBuffer: CVPixelBuffer, to url: URL) async throws {
        let ciImage = CIImage(cvPixelBuffer: confidenceBuffer)
        let context = CIContext()
        let colorSpace = CGColorSpace(name: CGColorSpace.genericGrayGamma2_2)!
        
        guard let pngData = context.pngRepresentation(
            of: ciImage,
            format: .L8,
            colorSpace: colorSpace
        ) else {
            throw ARKitCaptureError.imageProcessingFailed
        }
        
        try pngData.write(to: url)
    }
    
    // MARK: - Camera Data Export
    
    private func exportCameraData(
        transforms: [simd_float4x4],
        intrinsics: [simd_float3x3],
        to datasetURL: URL
    ) throws {
        
        // Export camera poses in COLMAP format
        try exportCOLMAPCameras(intrinsics: intrinsics, to: datasetURL)
        try exportCOLMAPImages(transforms: transforms, to: datasetURL)
        
        // Export transforms.json for NeRF/GaussianSplatting
        try exportTransformsJSON(
            transforms: transforms,
            intrinsics: intrinsics,
            to: datasetURL
        )
    }
    
    private func exportCOLMAPCameras(intrinsics: [simd_float3x3], to datasetURL: URL) throws {
        let sparseURL = datasetURL.appendingPathComponent("sparse")
        let camerasURL = sparseURL.appendingPathComponent("cameras.txt")
        
        var cameraContent = "# Camera list with one line of data per camera:\n"
        cameraContent += "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        
        if let firstIntrinsics = intrinsics.first {
            let fx = firstIntrinsics[0][0]
            let fy = firstIntrinsics[1][1]
            let cx = firstIntrinsics[0][2]
            let cy = firstIntrinsics[1][2]
            
            cameraContent += "1 PINHOLE \(Int(config.captureResolution.width)) \(Int(config.captureResolution.height)) \(fx) \(fy) \(cx) \(cy)\n"
        }
        
        try cameraContent.write(to: camerasURL, atomically: true, encoding: .utf8)
    }
    
    private func exportCOLMAPImages(transforms: [simd_float4x4], to datasetURL: URL) throws {
        let sparseURL = datasetURL.appendingPathComponent("sparse")
        let imagesURL = sparseURL.appendingPathComponent("images.txt")
        
        var imageContent = "# Image list with two lines of data per image:\n"
        imageContent += "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        imageContent += "# POINTS2D[] as (X, Y, POINT3D_ID)\n"
        
        for (index, transform) in transforms.enumerated() {
            // Convert transform matrix to quaternion and translation
            let (quaternion, translation) = matrixToQuaternionAndTranslation(transform)
            
            let imageID = index + 1
            let cameraID = 1
            let imageName = String(format: "%06d.jpg", index)
            
            imageContent += "\(imageID) \(quaternion.vector.w) \(quaternion.vector.x) \(quaternion.vector.y) \(quaternion.vector.z) "
            imageContent += "\(translation.x) \(translation.y) \(translation.z) \(cameraID) \(imageName)\n\n"
        }
        
        try imageContent.write(to: imagesURL, atomically: true, encoding: .utf8)
    }
    
    private func exportTransformsJSON(
        transforms: [simd_float4x4],
        intrinsics: [simd_float3x3],
        to datasetURL: URL
    ) throws {
        
        let transformsURL = datasetURL.appendingPathComponent("transforms.json")
        
        var transformsData: [String: Any] = [:]
        
        // Camera parameters
        if let firstIntrinsics = intrinsics.first {
            let fx = firstIntrinsics[0][0]
            let fy = firstIntrinsics[1][1]
            let cx = firstIntrinsics[0][2]
            let cy = firstIntrinsics[1][2]
            
            transformsData["camera_angle_x"] = 2 * atan(config.captureResolution.width / (2 * Double(fx)))
            transformsData["camera_angle_y"] = 2 * atan(config.captureResolution.height / (2 * Double(fy)))
            transformsData["fl_x"] = Double(fx)
            transformsData["fl_y"] = Double(fy)
            transformsData["cx"] = Double(cx)
            transformsData["cy"] = Double(cy)
            transformsData["w"] = Int(config.captureResolution.width)
            transformsData["h"] = Int(config.captureResolution.height)
        }
        
        // Frame data
        var frames: [[String: Any]] = []
        
        for (index, transform) in transforms.enumerated() {
            let frameData: [String: Any] = [
                "file_path": "./images/\(String(format: "%06d.jpg", index))",
                "depth_path": "./depth/\(String(format: "%06d.tiff", index))",
                "transform_matrix": matrixToArray(transform)
            ]
            frames.append(frameData)
        }
        
        transformsData["frames"] = frames
        
        let jsonData = try JSONSerialization.data(withJSONObject: transformsData, options: .prettyPrinted)
        try jsonData.write(to: transformsURL)
    }
    
    // MARK: - Metadata Export
    
    private func exportMetadata(frames: [CapturedFrame], to datasetURL: URL) throws {
        let metadataURL = datasetURL.appendingPathComponent("metadata.json")
        
        let metadata: [String: Any] = [
            "dataset_name": config.datasetName,
            "creation_date": ISO8601DateFormatter().string(from: Date()),
            "frame_count": frames.count,
            "capture_resolution": [
                "width": Int(config.captureResolution.width),
                "height": Int(config.captureResolution.height)
            ],
            "capture_device": UIDevice.current.name,
            "device_model": UIDevice.current.model,
            "ios_version": UIDevice.current.systemVersion,
            "format_version": "1.0"
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted)
        try jsonData.write(to: metadataURL)
    }
    
    // MARK: - Utility Functions
    
    private func matrixToQuaternionAndTranslation(_ matrix: simd_float4x4) -> (simd_quatf, simd_float3) {
        let rotation = simd_float3x3(
            simd_float3(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z),
            simd_float3(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z),
            simd_float3(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z)
        )
        
        let quaternion = simd_quatf(rotation)
        let translation = simd_float3(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z)
        
        return (quaternion, translation)
    }
    
    private func matrixToArray(_ matrix: simd_float4x4) -> [[Double]] {
        return [
            [Double(matrix.columns.0.x), Double(matrix.columns.0.y), Double(matrix.columns.0.z), Double(matrix.columns.0.w)],
            [Double(matrix.columns.1.x), Double(matrix.columns.1.y), Double(matrix.columns.1.z), Double(matrix.columns.1.w)],
            [Double(matrix.columns.2.x), Double(matrix.columns.2.y), Double(matrix.columns.2.z), Double(matrix.columns.2.w)],
            [Double(matrix.columns.3.x), Double(matrix.columns.3.y), Double(matrix.columns.3.z), Double(matrix.columns.3.w)]
        ]
    }
}