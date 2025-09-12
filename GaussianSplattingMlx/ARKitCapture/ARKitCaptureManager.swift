import SwiftUI
import CoreImage
import Vision
#if canImport(ARKit)
import ARKit
import RealityKit
#endif

// MARK: - ARKit Capture Manager for GaussianSplatting Data Collection

#if canImport(ARKit)
@MainActor
class ARKitCaptureManager: NSObject, ObservableObject {
    
    // MARK: - Properties
    @Published var isRunning = false
    @Published var frameCount = 0
    @Published var capturedFrames: [CapturedFrame] = []
    @Published var isCapturing = false
    @Published var errorMessage: String?
    
    private var arView: ARView?
    let session = ARSession()
    private var captureTimer: Timer?
    private let captureInterval: TimeInterval = 0.5 // Capture every 0.5 seconds
    
    // Camera tracking data
    @Published var cameraTransforms: [simd_float4x4] = []
    @Published var cameraIntrinsics: [simd_float3x3] = []
    
    // Dataset configuration
    struct DatasetConfig {
        let outputDirectory: URL
        let datasetName: String
        let targetFrameCount: Int
        let captureResolution: CGSize
        
        static let `default` = DatasetConfig(
            outputDirectory: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!,
            datasetName: "gaussian_splatting_dataset",
            targetFrameCount: 100,
            captureResolution: CGSize(width: 1920, height: 1440)
        )
    }
    
    var config = DatasetConfig.default
    
    // MARK: - Initialization
    override init() {
        super.init()
        session.delegate = self
    }
    
    // MARK: - ARKit Session Management
    
    func startSession() {
        guard ARWorldTrackingConfiguration.isSupported else {
            errorMessage = "ARWorldTracking is not supported on this device"
            return
        }
        
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        configuration.environmentTexturing = .automatic
        configuration.wantsHDREnvironmentTextures = true
        
        // Enable high-resolution capture if available
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
        errorMessage = nil
        
        Logger.shared.info("ARKit session started for data collection")
    }
    
    func stopSession() {
        session.pause()
        isRunning = false
        stopCapture()
        
        Logger.shared.info("ARKit session stopped")
    }
    
    // MARK: - Data Capture Control
    
    func startCapture() {
        guard isRunning else {
            errorMessage = "ARKit session is not running"
            return
        }
        
        isCapturing = true
        frameCount = 0
        capturedFrames.removeAll()
        cameraTransforms.removeAll()
        cameraIntrinsics.removeAll()
        
        captureTimer = Timer.scheduledTimer(withTimeInterval: captureInterval, repeats: true) { _ in
            Task { @MainActor in
                await self.captureCurrentFrame()
            }
        }
        
        Logger.shared.info("Started capturing ARKit frames")
    }
    
    func stopCapture() {
        captureTimer?.invalidate()
        captureTimer = nil
        isCapturing = false
        
        Logger.shared.info("Stopped capturing ARKit frames, total: \(frameCount)")
    }
    
    // MARK: - Frame Capture
    
    private func captureCurrentFrame() async {
        guard isCapturing,
              let currentFrame = session.currentFrame else { return }
        
        do {
            let capturedFrame = try await processCameraFrame(currentFrame)
            capturedFrames.append(capturedFrame)
            cameraTransforms.append(currentFrame.camera.transform)
            cameraIntrinsics.append(currentFrame.camera.intrinsics)
            
            frameCount += 1
            
            // Auto-stop when reaching target frame count
            if frameCount >= config.targetFrameCount {
                stopCapture()
                Logger.shared.info("Reached target frame count: \(frameCount)")
            }
            
        } catch {
            errorMessage = "Failed to capture frame: \(error.localizedDescription)"
            Logger.shared.error("Frame capture error", error: error)
        }
    }
    
    // MARK: - Frame Processing
    
    private func processCameraFrame(_ frame: ARFrame) async throws -> CapturedFrame {
        
        // Extract RGB image
        let rgbImage = try await extractRGBImage(from: frame)
        
        // Extract depth data
        let depthData = try extractDepthData(from: frame)
        
        // Extract camera pose and intrinsics
        let cameraPose = CameraPose(
            transform: frame.camera.transform,
            intrinsics: frame.camera.intrinsics,
            imageResolution: simd_float2(Float(frame.camera.imageResolution.width), 
                                       Float(frame.camera.imageResolution.height)),
            timestamp: frame.timestamp
        )
        
        return CapturedFrame(
            index: frameCount,
            rgbImage: rgbImage,
            depthData: depthData,
            cameraPose: cameraPose,
            timestamp: frame.timestamp
        )
    }
    
    private func extractRGBImage(from frame: ARFrame) async throws -> CIImage {
        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        
        // Apply rotation correction for device orientation
        let rotatedImage = ciImage.oriented(.right) // Adjust based on device orientation
        
        return rotatedImage
    }
    
    private func extractDepthData(from frame: ARFrame) throws -> DepthData {
        guard let sceneDepth = frame.sceneDepth else {
            throw ARKitCaptureError.depthDataNotAvailable
        }
        
        let depthMap = sceneDepth.depthMap
        let confidenceMap = sceneDepth.confidenceMap
        
        return DepthData(
            depthMap: depthMap,
            confidenceMap: confidenceMap,
            depthResolution: simd_float2(Float(CVPixelBufferGetWidth(depthMap)), 
                                       Float(CVPixelBufferGetHeight(depthMap)))
        )
    }
    
    // MARK: - Dataset Export
    
    func exportDataset() async throws {
        guard !capturedFrames.isEmpty else {
            throw ARKitCaptureError.noDataToExport
        }
        
        let exporter = DatasetExporter(config: config)
        try await exporter.exportDataset(
            frames: capturedFrames,
            cameraTransforms: cameraTransforms,
            cameraIntrinsics: cameraIntrinsics
        )
        
        Logger.shared.info("Dataset exported successfully to: \(config.outputDirectory)")
    }
    
    // MARK: - Configuration
    
    func updateConfig(_ newConfig: DatasetConfig) {
        config = newConfig
        Logger.shared.debug("Updated dataset config: \(newConfig.datasetName)")
    }
}

// MARK: - ARSessionDelegate

extension ARKitCaptureManager: ARSessionDelegate {
    
    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor in
            errorMessage = "ARKit session failed: \(error.localizedDescription)"
        }
        Logger.shared.error("ARKit session error", error: error)
    }
    
    nonisolated func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        Task { @MainActor in
            switch camera.trackingState {
            case .normal:
                errorMessage = nil
            case .notAvailable:
                errorMessage = "Camera tracking not available"
            case .limited(let reason):
                switch reason {
                case .excessiveMotion:
                    errorMessage = "Too much motion - move slower"
                case .insufficientFeatures:
                    errorMessage = "Not enough visual features - point at textured surfaces"
                case .initializing:
                    errorMessage = "Initializing camera tracking..."
                case .relocalizing:
                    errorMessage = "Relocalizing camera position..."
                @unknown default:
                    errorMessage = "Limited tracking"
                }
            }
        }
        
        Logger.shared.debug("Camera tracking state: \(camera.trackingState)")
    }
    
    nonisolated func sessionWasInterrupted(_ session: ARSession) {
        Task { @MainActor in
            isRunning = false
            stopCapture()
            errorMessage = "AR session was interrupted"
        }
    }
    
    nonisolated func sessionInterruptionEnded(_ session: ARSession) {
        Task { @MainActor in
            errorMessage = "AR session interruption ended"
        }
    }
}

// MARK: - Data Models

struct CapturedFrame {
    let index: Int
    let rgbImage: CIImage
    let depthData: DepthData
    let cameraPose: CameraPose
    let timestamp: TimeInterval
}

struct DepthData {
    let depthMap: CVPixelBuffer
    let confidenceMap: CVPixelBuffer?
    let depthResolution: simd_float2
}

struct CameraPose {
    let transform: simd_float4x4
    let intrinsics: simd_float3x3
    let imageResolution: simd_float2
    let timestamp: TimeInterval
}

// MARK: - Error Types

enum ARKitCaptureError: LocalizedError {
    case depthDataNotAvailable
    case imageProcessingFailed
    case noDataToExport
    case exportFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .depthDataNotAvailable:
            return "Depth data is not available on this device"
        case .imageProcessingFailed:
            return "Failed to process camera image"
        case .noDataToExport:
            return "No captured data available for export"
        case .exportFailed(let message):
            return "Export failed: \(message)"
        }
    }
}
#else
@MainActor
class ARKitCaptureManager: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var frameCount = 0
    @Published var capturedFrames: [CapturedFrame] = []
    @Published var isCapturing = false
    @Published var errorMessage: String? = "ARKit not available on this platform"
    
    var session: AnyObject? = nil
    
    struct DatasetConfig {
        let outputDirectory: URL
        let datasetName: String
        let targetFrameCount: Int
        let captureResolution: CGSize
        
        static let `default` = DatasetConfig(
            outputDirectory: FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!,
            datasetName: "gaussian_splatting_dataset",
            targetFrameCount: 100,
            captureResolution: CGSize(width: 1920, height: 1440)
        )
    }
    
    var config = DatasetConfig.default
    
    func startSession() { }
    func stopSession() { }
    func startCapture() { }
    func stopCapture() { }
    func exportDataset() async throws { }
    func updateConfig(_ newConfig: DatasetConfig) { }
}

struct CapturedFrame {
    let index: Int
    let timestamp: TimeInterval
}

struct DepthData {
    let depthResolution: simd_float2
}

struct CameraPose {
    let timestamp: TimeInterval
}

enum ARKitCaptureError: LocalizedError {
    case depthDataNotAvailable
    case imageProcessingFailed
    case noDataToExport
    case exportFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .depthDataNotAvailable:
            return "Depth data is not available on this device"
        case .imageProcessingFailed:
            return "Failed to process camera image"
        case .noDataToExport:
            return "No captured data available for export"
        case .exportFailed(let message):
            return "Export failed: \(message)"
        }
    }
}
#endif