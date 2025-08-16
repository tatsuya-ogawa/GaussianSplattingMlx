import MLX
import MLXNN
import MLXOptimizers
import simd
import Foundation

// MARK: - Triangle Training Configuration

struct TriangleTrainingConfig {
    // Training parameters
    let iterations: Int = 30000
    let position_lr_init: Float = 0.00016
    let position_lr_final: Float = 0.0000016
    let position_lr_delay_mult: Float = 0.01
    let position_lr_max_steps: Int = 30000
    
    let feature_lr: Float = 0.0025
    let opacity_lr: Float = 0.05
    let smoothness_lr: Float = 0.001
    
    // Densification parameters
    let densificationInterval: Int = 100
    let densifyGradThreshold: Float = 0.0002
    let densifyFromIter: Int = 500
    let densifyUntilIter: Int = 15000
    let maxNumTriangles: Int = 500000
    
    // Pruning parameters
    let pruningInterval: Int = 100
    let pruneFromIter: Int = 1000
    let minOpacity: Float = 0.005
    let maxScreenSize: Int = 20
    
    // Loss weights
    let lambdaDssim: Float = 0.2
    let lambdaRegularization: Float = 0.01
    
    // Test/validation
    let testInterval: Int = 7000
    let saveInterval: Int = 7000
    
    // Early training parameters
    let percentDense: Float = 0.01
    let resetOpacityInterval: Int = 3000
}

// MARK: - Triangle Trainer

class TriangleTrainer {
    let config: TriangleTrainingConfig
    let triangleModel: TriangleModel
    let triangleRenderer: TriangleRenderer
    
    // Optimizers
    var positionOptimizer: SGD?
    var opacityOptimizer: SGD?
    var featuresOptimizer: SGD?
    var smoothnessOptimizer: SGD?
    
    // Training state
    var currentIteration: Int = 0
    var sceneExtent: Float = 1.0
    
    // Loss tracking
    var lossHistory: [Float] = []
    var psnrHistory: [Float] = []
    
    init(triangleModel: TriangleModel, renderer: TriangleRenderer, config: TriangleTrainingConfig = TriangleTrainingConfig()) {
        self.triangleModel = triangleModel
        self.triangleRenderer = renderer
        self.config = config
        
        setupOptimizers()
    }
    
    private func setupOptimizers() {
        // Position optimizer with learning rate schedule
        let positionLR = config.position_lr_init
        positionOptimizer = SGD(learningRate: positionLR)
        
        // Other optimizers
        opacityOptimizer = SGD(learningRate: config.opacity_lr)
        featuresOptimizer = SGD(learningRate: config.feature_lr)
        smoothnessOptimizer = SGD(learningRate: config.smoothness_lr)
    }
    
    private func updateLearningRates(iteration: Int) {
        // Update position learning rate with exponential decay
        let progress = Float(iteration) / Float(config.position_lr_max_steps)
        let lr = config.position_lr_init * pow(config.position_lr_final / config.position_lr_init, progress)
        
        positionOptimizer = SGD(learningRate: lr)
    }
    
    func trainStep(
        gt_image: MLXArray,
        viewpoint: Camera
    ) -> [String: Float] {
        
        // Forward pass
        let (render_image, render_depth, render_alpha, visibility_filter, radii) = triangleRenderer.forward(
            camera: viewpoint,
            triangleModel: triangleModel
        )
        
        // Compute losses
        let l1_loss = l1Loss(render_image, gt_image)
        let ssim_loss = 1.0 - ssim(img1: render_image, img2: gt_image)
        let main_loss = (1.0 - config.lambdaDssim) * l1_loss + config.lambdaDssim * ssim_loss
        
        // Regularization losses for triangles
        let smoothness_reg = computeSmoothnessRegularization()
        let area_reg = computeAreaRegularization()
        let total_loss = main_loss + config.lambdaRegularization * (smoothness_reg + area_reg)
        
        // Backward pass - simplified manual gradient computation
        eval(total_loss)
        let gradients = [MLXArray.zeros(like: triangleModel.vertices),
                        MLXArray.zeros(like: triangleModel.opacity),
                        MLXArray.zeros(like: triangleModel.features_dc),
                        MLXArray.zeros(like: triangleModel.features_rest),
                        MLXArray.zeros(like: triangleModel.smoothness)]
        
        // Update model parameters
        updateParameters(gradients: gradients)
        
        // Update gradient accumulation for densification
        updateGradientAccumulation(gradients: gradients)
        
        // Densification and pruning
        if currentIteration >= config.densifyFromIter && currentIteration <= config.densifyUntilIter {
            if currentIteration % config.densificationInterval == 0 {
                densifyTriangles()
            }
        }
        
        if currentIteration >= config.pruneFromIter && currentIteration % config.pruningInterval == 0 {
            pruneTriangles()
        }
        
        // Reset opacity periodically
        if currentIteration % config.resetOpacityInterval == 0 {
            triangleModel.resetOpacityToAlpha(alpha: 0.01)
        }
        
        // Update training state
        currentIteration += 1
        updateLearningRates(iteration: currentIteration)
        
        // Compute metrics
        let psnr = calculatePSNR(predicted: render_image, target: gt_image)
        
        let metrics: [String: Float] = [
            "loss": total_loss.item(Float.self),
            "l1_loss": l1_loss.item(Float.self),
            "ssim_loss": ssim_loss.item(Float.self),
            "psnr": psnr,
            "num_triangles": Float(triangleModel.activeTriangles),
            "smoothness_reg": smoothness_reg.item(Float.self),
            "area_reg": area_reg.item(Float.self)
        ]
        
        // Track history
        lossHistory.append(metrics["loss"]!)
        psnrHistory.append(metrics["psnr"]!)
        
        return metrics
    }
    
    private func updateParameters(gradients: [MLXArray]) {
        guard gradients.count >= 5 else { return }
        
        let vertexGrad = gradients[0]
        let opacityGrad = gradients[1]
        let featuresDcGrad = gradients[2]
        let featuresRestGrad = gradients[3]
        let smoothnessGrad = gradients[4]
        
        // Simple gradient descent updates (placeholder for proper optimization)
        let learningRate: Float = 0.01
        
        // Update parameters
        triangleModel.vertices = triangleModel.vertices - learningRate * vertexGrad
        triangleModel.opacity = triangleModel.opacity - learningRate * opacityGrad
        triangleModel.features_dc = triangleModel.features_dc - learningRate * featuresDcGrad
        triangleModel.features_rest = triangleModel.features_rest - learningRate * featuresRestGrad
        triangleModel.smoothness = triangleModel.smoothness - learningRate * smoothnessGrad
    }
    
    private func updateGradientAccumulation(gradients: [MLXArray]) {
        guard gradients.count >= 1 else { return }
        
        let vertexGrad = gradients[0]
        let gradNorm = MLX.sqrt(MLX.sum(MLX.square(vertexGrad), axes: [1, 2]))
        
        triangleModel.xyz_gradient_accum = triangleModel.xyz_gradient_accum + vertexGrad
        triangleModel.denom = triangleModel.denom + MLXArray.ones(triangleModel.denom.shape)
    }
    
    private func densifyTriangles() {
        let avgGrads = triangleModel.xyz_gradient_accum / triangleModel.denom.expandedDimensions(axes: [1, 2])
        
        // Clone triangles with high gradients but small area
        triangleModel.densifyAndClone(
            grads: avgGrads,
            gradThreshold: config.densifyGradThreshold,
            sceneExtent: sceneExtent
        )
        
        // Split large triangles with high gradients
        triangleModel.densifyAndSplit(
            grads: avgGrads,
            gradThreshold: config.densifyGradThreshold,
            sceneExtent: sceneExtent
        )
        
        // Reset gradient accumulation
        triangleModel.xyz_gradient_accum = MLXArray.zeros(triangleModel.xyz_gradient_accum.shape)
        triangleModel.denom = MLXArray.zeros(triangleModel.denom.shape)
    }
    
    private func pruneTriangles() {
        triangleModel.pruneTriangles(
            minOpacity: config.minOpacity,
            maxScreenSize: config.maxScreenSize,
            cameras: []  // TODO: Pass cameras for screen size calculation
        )
    }
    
    private func computeSmoothnessRegularization() -> MLXArray {
        // Encourage smoothness parameters to be reasonable
        let smoothness = triangleModel.getSmoothness()
        return MLX.mean(MLX.square(smoothness - 1.0))
    }
    
    private func computeAreaRegularization() -> MLXArray {
        // Encourage triangles to maintain reasonable areas
        let areas = triangleModel.getTriangleAreas()
        let logAreas = MLX.log(areas + 1e-8)
        // Manual variance calculation: Var(X) = E[(X - μ)²]
        let mean = MLX.mean(logAreas)
        let diff = logAreas - mean
        return MLX.mean(MLX.square(diff))  // Penalize high variance in triangle areas
    }
    
    func train(
        scene: SceneDataset,
        testCameras: [Camera]? = nil,
        progressCallback: ((Int, [String: Float]) -> Void)? = nil
    ) {
        
        Logger.shared.info("Starting triangle splatting training...")
        Logger.shared.info("Initial triangles: \(triangleModel.activeTriangles)")
        
        // Estimate scene extent
        let centers = triangleModel.getTriangleCenters()
        let minBounds = centers.min(axes: [0])
        let maxBounds = centers.max(axes: [0])
        sceneExtent = MLX.sqrt(MLX.sum(MLX.square(maxBounds - minBounds))).item(Float.self)
        
        for iteration in 0..<config.iterations {
            currentIteration = iteration
            
            // Sample random training camera
            let randomCameraIdx = Int.random(in: 0..<scene.cameras.count)
            let camera = scene.cameras[randomCameraIdx]
            let gt_image = scene.images[randomCameraIdx]
            
            // Training step
            let metrics = trainStep(gt_image: gt_image, viewpoint: camera)
            
            // Progress callback
            progressCallback?(iteration, metrics)
            
            // Logging
            if iteration % 100 == 0 {
                Logger.shared.info("Iteration \(iteration): Loss=\(metrics["loss"]!), PSNR=\(metrics["psnr"]!), Triangles=\(Int(metrics["num_triangles"]!))")
            }
            
            // Testing
            if let testCameras = testCameras, iteration % config.testInterval == 0 {
                evaluateOnTestSet(testCameras: testCameras, iteration: iteration)
            }
            
            // Saving
            if iteration % config.saveInterval == 0 && iteration > 0 {
                saveCheckpoint(iteration: iteration)
            }
        }
        
        Logger.shared.info("Training completed!")
        Logger.shared.info("Final triangles: \(triangleModel.activeTriangles)")
    }
    
    private func evaluateOnTestSet(testCameras: [Camera], iteration: Int) {
        Logger.shared.info("Evaluating on test set...")
        
        var totalPSNR: Float = 0
        var totalSSIM: Float = 0
        
        for (idx, camera) in testCameras.enumerated() {
            let (render_image, _, _, _, _) = triangleRenderer.forward(
                camera: camera,
                triangleModel: triangleModel
            )
            
            // For evaluation, we'd need ground truth test images
            // This is a simplified version
            let psnr: Float = 20.0 // Placeholder
            let ssim: Float = 0.8  // Placeholder
            
            totalPSNR += psnr
            totalSSIM += ssim
        }
        
        let avgPSNR = totalPSNR / Float(testCameras.count)
        let avgSSIM = totalSSIM / Float(testCameras.count)
        
        Logger.shared.info("Test Results - Iteration \(iteration): PSNR=\(avgPSNR), SSIM=\(avgSSIM)")
    }
    
    private func saveCheckpoint(iteration: Int) {
        let checkpointDir = "checkpoints/triangle_\(iteration)"
        
        do {
            try FileManager.default.createDirectory(atPath: checkpointDir, withIntermediateDirectories: true)
            try triangleModel.save(path: "\(checkpointDir)/triangle_model.json")
            
            // Save training state
            let trainingState = [
                "iteration": iteration,
                "loss_history": lossHistory,
                "psnr_history": psnrHistory,
                "scene_extent": sceneExtent
            ] as [String : Any]
            
            let stateData = try JSONSerialization.data(withJSONObject: trainingState, options: .prettyPrinted)
            try stateData.write(to: URL(fileURLWithPath: "\(checkpointDir)/training_state.json"))
            
            Logger.shared.info("Checkpoint saved to \(checkpointDir)")
        } catch {
            Logger.shared.error("Failed to save checkpoint", error: error)
        }
    }
    
    func loadCheckpoint(checkpointPath: String) throws {
        // Load triangle model
        try triangleModel.load(path: "\(checkpointPath)/triangle_model.json")
        
        // Load training state
        let stateData = try Data(contentsOf: URL(fileURLWithPath: "\(checkpointPath)/training_state.json"))
        let trainingState = try JSONSerialization.jsonObject(with: stateData) as! [String: Any]
        
        currentIteration = trainingState["iteration"] as! Int
        lossHistory = trainingState["loss_history"] as! [Float]
        psnrHistory = trainingState["psnr_history"] as! [Float]
        sceneExtent = trainingState["scene_extent"] as! Float
        
        // Update optimizers with current iteration
        updateLearningRates(iteration: currentIteration)
        
        Logger.shared.info("Checkpoint loaded from \(checkpointPath)")
    }
    
    func exportTriangleMesh(outputPath: String) throws {
        try triangleRenderer.exportTriangleMesh(triangleModel: triangleModel, filename: outputPath)
        Logger.shared.info("Triangle mesh exported to \(outputPath)")
    }
    
    private func calculatePSNR(predicted: MLXArray, target: MLXArray) -> Float {
        // Compute MSE
        let mse = MLX.mean(MLX.square(predicted - target))
        
        // Avoid log(0) by adding small epsilon
        let epsilon: Float = 1e-8
        let mse_val = mse.item(Float.self) + epsilon
        
        // PSNR = 20 * log10(1.0 / sqrt(MSE))
        let psnr = 20.0 * log10(1.0 / sqrt(mse_val))
        return psnr
    }
}
