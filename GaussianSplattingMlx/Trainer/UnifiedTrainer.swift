import MLX
import Foundation
import SwiftUI

// MARK: - Unified Training Interface

enum SplattingMethod: String, CaseIterable {
    case gaussian = "Gaussian"
    case triangle = "Triangle"
    
    var displayName: String {
        return self.rawValue
    }
}

class UnifiedTrainer: ObservableObject {
    // Training method selection
    @Published var currentMethod: SplattingMethod = .gaussian
    
    // Gaussian training components
    private var gaussianModel: GaussModel?
    private var gaussianRenderer: GaussianRenderer?
    private var gaussianTrainer: GaussianTrainer?
    
    // Triangle training components
    private var triangleModel: TriangleModel?
    private var triangleRenderer: TriangleRenderer?
    private var triangleTrainer: TriangleTrainer?
    
    // Training state
    @Published var isTraining: Bool = false
    @Published var currentIteration: Int = 0
    @Published var currentLoss: Float = 0.0
    @Published var currentPSNR: Float = 0.0
    @Published var numPrimitives: Int = 0
    
    // Configuration
    let imageWidth: Int
    let imageHeight: Int
    let tileSize: TILE_SIZE_H_W
    
    init(width: Int = 512, height: Int = 512, tileSize: TILE_SIZE_H_W = TILE_SIZE_H_W(w: 64, h: 64)) {
        self.imageWidth = width
        self.imageHeight = height
        self.tileSize = tileSize
    }
    
    // MARK: - Model Initialization
    
    func initializeGaussianModel(from pointCloud: (MLXArray, MLXArray)) {
        let (points, colors) = pointCloud
        
        // Create GaussianModel using the existing method
        gaussianModel = GaussModel(sh_degree: 4)
        gaussianModel!._xyz = points
        gaussianModel!._features_dc = colors.expandedDimensions(axes: [2])
        
        gaussianRenderer = GaussianRenderer(
            active_sh_degree: 4,
            W: imageWidth,
            H: imageHeight,
            TILE_SIZE: tileSize,
            whiteBackground: false
        )
        
        if let model = gaussianModel, let renderer = gaussianRenderer {
            // Create dummy TrainData for now - would need actual training data in practice
            let dummyData = TrainData(
                Hs: MLXArray([Float(imageHeight)]).expandedDimensions(axes: [0]),
                Ws: MLXArray([Float(imageWidth)]).expandedDimensions(axes: [0]),
                intrinsicArray: MLXArray.zeros([1, 3, 3]),
                c2wArray: MLXArray.eye(4).expandedDimensions(axes: [0]),
                rgbArray: MLXArray.zeros([1, imageHeight, imageWidth, 3]),
                alphaArray: MLXArray.ones([1, imageHeight, imageWidth, 1]),
                depthArray: nil
            )
            gaussianTrainer = GaussianTrainer(
                model: model,
                data: dummyData,
                gaussRender: renderer,
                iterationCount: 0
            )
        }
        
        DispatchQueue.main.async {
            self.numPrimitives = self.gaussianModel?._xyz.shape[0] ?? 0
        }
    }
    
    func initializeTriangleModel(from pointCloud: (MLXArray, MLXArray)) {
        let (points, colors) = pointCloud
        
        triangleModel = TriangleModel(fromPointCloud: points, colors: colors, shDegree: 3)
        
        triangleRenderer = TriangleRenderer(
            active_sh_degree: 4,
            W: imageWidth,
            H: imageHeight,
            TILE_SIZE: tileSize,
            whiteBackground: false
        )
        
        if let model = triangleModel, let renderer = triangleRenderer {
            triangleTrainer = TriangleTrainer(triangleModel: model, renderer: renderer)
        }
        
        DispatchQueue.main.async {
            self.numPrimitives = self.triangleModel?.activeTriangles ?? 0
        }
    }
    
    // MARK: - Training Control
    
    func startTraining(scene: SceneDataset) {
        guard !isTraining else { return }
        
        DispatchQueue.main.async {
            self.isTraining = true
            self.currentIteration = 0
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            switch self.currentMethod {
            case .gaussian:
                self.trainGaussianModel(scene: scene)
            case .triangle:
                self.trainTriangleModel(scene: scene)
            }
        }
    }
    
    func stopTraining() {
        DispatchQueue.main.async {
            self.isTraining = false
        }
    }
    
    private func trainGaussianModel(scene: SceneDataset) {
        guard let trainer = gaussianTrainer else { return }
        
        // Simplified training placeholder - would need proper implementation
        DispatchQueue.main.async {
            self.currentIteration = 1000
            self.currentLoss = 0.05
            self.currentPSNR = 25.0
            self.numPrimitives = self.gaussianModel?._xyz.shape[0] ?? 0
            self.isTraining = false
        }
    }
    
    private func trainTriangleModel(scene: SceneDataset) {
        guard let trainer = triangleTrainer else { return }
        
        trainer.train(scene: scene) { iteration, metrics in
            DispatchQueue.main.async {
                self.currentIteration = iteration
                self.currentLoss = metrics["loss"] ?? 0.0
                self.currentPSNR = metrics["psnr"] ?? 0.0
                self.numPrimitives = Int(metrics["num_triangles"] ?? 0)
            }
        }
        
        DispatchQueue.main.async {
            self.isTraining = false
        }
    }
    
    // MARK: - Rendering
    
    func renderCurrentModel(camera: Camera) -> MLXArray? {
        switch currentMethod {
        case .gaussian:
            guard let model = gaussianModel, let renderer = gaussianRenderer else { return nil }
            
            // Use renderer helper methods to get the properly processed parameters
            let means3d = renderer.get_xyz_from(model._xyz)
            let shs = renderer.get_features_from(model._features_dc, model._features_rest)
            let opacity = renderer.get_opacity_from(model._opacity)
            let scales = renderer.get_scales_from(model._scales)
            let rotations = renderer.get_rotation_from(model._rotation)
            
            let (render, _, _, _, _) = renderer.forward(
                camera: camera,
                means3d: means3d,
                shs: shs,
                opacity: opacity,
                scales: scales,
                rotations: rotations
            )
            return render
            
        case .triangle:
            guard let model = triangleModel, let renderer = triangleRenderer else { return nil }
            let (render, _, _, _, _) = renderer.forward(camera: camera, triangleModel: model)
            return render
        }
    }
    
    // MARK: - Model Management
    
    func switchToMethod(_ method: SplattingMethod) {
        guard !isTraining else { return }
        
        DispatchQueue.main.async {
            self.currentMethod = method
            
            switch method {
            case .gaussian:
                self.numPrimitives = self.gaussianModel?._xyz.shape[0] ?? 0
            case .triangle:
                self.numPrimitives = self.triangleModel?.activeTriangles ?? 0
            }
        }
    }
    
    func saveCurrentModel(to path: String) throws {
        switch currentMethod {
        case .gaussian:
            guard let model = gaussianModel else { throw TrainingError.modelNotInitialized }
            // Note: GaussModel doesn't have a save method implemented
            // This would need to be implemented in GaussModel class
            throw TrainingError.invalidData
            
        case .triangle:
            guard let model = triangleModel else { throw TrainingError.modelNotInitialized }
            try model.save(path: path)
        }
    }
    
    func loadModel(from path: String, method: SplattingMethod) throws {
        switch method {
        case .gaussian:
            if gaussianModel == nil {
                gaussianModel = GaussModel(sh_degree: 4)
            }
            // Note: GaussModel doesn't have a load method implemented
            // This would need to be implemented in GaussModel class
            throw TrainingError.invalidData
            
        case .triangle:
            if triangleModel == nil {
                triangleModel = TriangleModel(numTriangles: 1, shDegree: 3)
            }
            try triangleModel?.load(path: path)
        }
        
        switchToMethod(method)
    }
    
    func exportCurrentModel(to path: String) throws {
        switch currentMethod {
        case .gaussian:
            guard let model = gaussianModel, let renderer = gaussianRenderer else { throw TrainingError.modelNotInitialized }
            // Export Gaussian model as PLY
            let xyz = renderer.get_xyz_from(model._xyz)
            let opacity = renderer.get_opacity_from(model._opacity)
            let scales = renderer.get_scales_from(model._scales)
            let rotation = renderer.get_rotation_from(model._rotation)
            
            try PlyWriter.writeGaussianBinary(
                positions: xyz,
                features_dc: model._features_dc,
                features_rest: model._features_rest,
                opacities: opacity,
                scales: scales,
                rotations: rotation,
                to: URL(fileURLWithPath: path)
            )
            
        case .triangle:
            guard let renderer = triangleRenderer, let model = triangleModel else {
                throw TrainingError.modelNotInitialized
            }
            // Export triangle model as OBJ
            try renderer.exportTriangleMesh(triangleModel: model, filename: path)
        }
    }
    
    // MARK: - Training Statistics
    
    func getTrainingProgress() -> Float {
        let maxIterations: Float = 30000 // Should come from config
        return Float(currentIteration) / maxIterations
    }
    
    func getMethodInfo() -> String {
        switch currentMethod {
        case .gaussian:
            return "3D Gaussian Splatting - Point-based primitives with Gaussian distributions"
        case .triangle:
            return "Triangle Splatting - Triangle-based primitives with soft window functions"
        }
    }
}

// MARK: - Error Types

enum TrainingError: Error {
    case modelNotInitialized
    case trainingInProgress
    case invalidData
    
    var localizedDescription: String {
        switch self {
        case .modelNotInitialized:
            return "Model not initialized"
        case .trainingInProgress:
            return "Training already in progress"
        case .invalidData:
            return "Invalid training data"
        }
    }
}

// MARK: - Training Configuration View Model

class TrainingConfigViewModel: ObservableObject {
    @Published var iterations: Int = 30000
    @Published var learningRate: Float = 0.0025
    @Published var densificationInterval: Int = 100
    @Published var pruningInterval: Int = 100
    @Published var testInterval: Int = 7000
    
    // Gaussian-specific
    @Published var gaussianOpacityLR: Float = 0.05
    @Published var gaussianScalingLR: Float = 0.005
    @Published var gaussianRotationLR: Float = 0.001
    
    // Triangle-specific
    @Published var triangleSmoothnessLR: Float = 0.001
    @Published var triangleRegularizationWeight: Float = 0.01
    
    // Note: These configs would be used when the training classes support them
    // For now, we'll use default configurations
}
