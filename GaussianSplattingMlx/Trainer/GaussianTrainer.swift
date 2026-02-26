//
//  GaussianTrainer.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import Foundation
import MLX
import MLXOptimizers
import MLXRandom

class TrainData {
    let Hs: MLXArray
    let Ws: MLXArray
    let intrinsicArray: MLXArray
    var c2wArray: MLXArray
    let rgbArray: MLXArray
    let alphaArray: MLXArray
    let depthArray: MLXArray?
    init(
        Hs: MLXArray,
        Ws: MLXArray,
        intrinsicArray: MLXArray,
        c2wArray: MLXArray,
        rgbArray: MLXArray,
        alphaArray: MLXArray,
        depthArray: MLXArray?
    ) {
        self.Hs = Hs
        self.Ws = Ws
        self.intrinsicArray = intrinsicArray
        self.c2wArray = c2wArray
        self.rgbArray = rgbArray
        self.alphaArray = alphaArray
        self.depthArray = depthArray
    }
    func getViewPointCamera(index: Int) -> Camera {
        return Camera(
            width: Ws[index].item(Int.self),
            height: Hs[index].item(Int.self),
            intrinsic: intrinsicArray[index],
            c2w: c2wArray[index]
        )
    }
    func getNumCameras() -> Int {
        return self.Hs.shape[0]
    }

    func getCameraParams() -> (
        Hs: MLXArray, Ws: MLXArray, intrinsics: MLXArray, c2w: MLXArray
    ) {
        return (Hs, Ws, intrinsicArray, c2wArray)
    }
    
    /// Make all MLXArrays contiguous to optimize memory layout
    /// - Returns: A new optimized TrainData instance
    func optimized() -> TrainData {
        return TrainData(
            Hs: MLX.contiguous(Hs),
            Ws: MLX.contiguous(Ws),
            intrinsicArray: MLX.contiguous(intrinsicArray),
            c2wArray: MLX.contiguous(c2wArray),
            rgbArray: MLX.contiguous(rgbArray),
            alphaArray: MLX.contiguous(alphaArray),
            depthArray: depthArray != nil ? MLX.contiguous(depthArray!) : nil
        )
    }
    
    /// Optimize all MLXArrays to be contiguous in-place
    /// - Note: Only c2wArray can be updated directly as it's a var
    func optimizeInPlace() {
        // c2wArray is mutable so it can be updated directly
        c2wArray = MLX.contiguous(c2wArray)
        // Other properties are let, so use optimized() instead
    }
    
    /// Get estimated memory usage in bytes
    /// - Returns: Estimated memory usage
    func estimateMemoryUsage() -> Int {
        var totalSize = 0
        totalSize += Hs.size * 4  // Float32 = 4 bytes
        totalSize += Ws.size * 4
        totalSize += intrinsicArray.size * 4
        totalSize += c2wArray.size * 4
        totalSize += rgbArray.size * 4
        totalSize += alphaArray.size * 4
        if let depthArray = depthArray {
            totalSize += depthArray.size * 4
        }
        return totalSize
    }
    
    /// Get shape information of TrainData
    /// - Returns: Information string containing shapes and memory usage
    func getShapeInfo() -> String {
        var info = "TrainData Information:\n"
        info += "- Batch Size: \(Hs.shape[0])\n"
        info += "- Hs: \(Hs.shape)\n"
        info += "- Ws: \(Ws.shape)\n"
        info += "- IntrinsicArray: \(intrinsicArray.shape)\n"
        info += "- C2wArray: \(c2wArray.shape)\n"
        info += "- RgbArray: \(rgbArray.shape)\n"
        info += "- AlphaArray: \(alphaArray.shape)\n"
        if let depthArray = depthArray {
            info += "- DepthArray: \(depthArray.shape)\n"
        } else {
            info += "- DepthArray: nil\n"
        }
        info += "- Estimated Memory: \(estimateMemoryUsage()) bytes"
        return info
    }
}
protocol GaussianTrainerDelegate: AnyObject {
    func pushLoss(loss: Float, iteration: Int?, fps: Float?, timestamp: Date)
    func pushImageData(
        render: MLXArray, truth: MLXArray, loss: Float, iteration: Int, timestamp: Date)
    func pushSnapshot(url: URL,iteration: Int, timestamp: Date)
}
class GaussianTrainer {
    private struct CameraStateArrays {
        let viewMatrix: MLXArray
        let projectionMatrix: MLXArray
        let fovX: MLXArray
        let fovY: MLXArray
        let focalX: MLXArray
        let focalY: MLXArray
        let cameraCenter: MLXArray
    }

    private enum TrainStepInputIndex {
        static let xyz = 0
        static let featuresDC = 1
        static let featuresRest = 2
        static let scales = 3
        static let rotation = 4
        static let opacity = 5
        static let viewMatrix = 6
        static let projectionMatrix = 7
        static let fovX = 8
        static let fovY = 9
        static let focalX = 10
        static let focalY = 11
        static let cameraCenter = 12
        static let targetRGB = 13
        static let depthMask = 14
        static let targetDepth = 15
        static let parameterCount = 6
    }

    var data: TrainData
    var gaussRender: GaussianRenderer
    var model: GaussModel
    var lambda_dssim: Float = 0.2
    var lambda_depth: Float = 0.0
    var split_and_prune_per_iteration: Int = 100
    var save_snapshot_per_iteration: Int
    var iterationCount: Int
    var outputDirectoryURL: URL?
    var cacheLimit: Int
    var manualClearCache: Bool
    weak var delegate: GaussianTrainerDelegate?
    
    // Split and prune parameters
    var gradientThreshold: Float = 0.0002
    var maxScreenSize: Float = 20.0
    var minOpacity: Float = 0.005
    var maxScale: Float = 0.01
    var pruneInterval: Int = 100
    var densifyFromIter: Int = 500
    var densifyUntilIter: Int = 15000
    var maxGaussians: Int = 1_000_000  // Maximum number of Gaussians to prevent unbounded growth
    
    // Tracking gradients for densification
    var xyzGradAccumulation: MLXArray = MLXArray.zeros([0])  // Scalar gradient magnitudes
    var denomGradAccumulation: MLXArray = MLXArray.zeros([0])
    
    static func defaultCacheLimit() -> Int {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory  // bytes
        let sixteenGB: UInt64 = 16 * 1024 * 1024 * 1024
        let fallback: UInt64 = 2 * 1024 * 1024 * 1024
        let chosen = physicalMemory >= sixteenGB ? physicalMemory / 2 : fallback
        return Int(min(chosen, UInt64(Int.max)))
    }
    
    static func needEagerClearCache() -> Bool {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory  // bytes
        let sixteenGB: UInt64 = 16 * 1024 * 1024 * 1024
        // If the machine is under 16GB, keep clearing to be safe; otherwise skip by default.
        return physicalMemory < sixteenGB
    }
    
    private func clearCacheIfNeeded() {
        guard manualClearCache else { return }
        MLX.GPU.clearCache()
    }
    init(
        model: GaussModel,
        data: TrainData,
        gaussRender: GaussianRenderer,
        iterationCount: Int,
        cacheLimit: Int = GaussianTrainer.defaultCacheLimit(),
        manualClearCache: Bool = GaussianTrainer.needEagerClearCache(),
        outputDirectoryURL: URL? = nil,
        saveSnapshotPerIteration: Int = 100
    ) {
        self.model = model
        self.data = data
        self.gaussRender = gaussRender
        self.iterationCount = iterationCount
        self.cacheLimit = cacheLimit
        self.manualClearCache = manualClearCache
        self.outputDirectoryURL = outputDirectoryURL
        self.save_snapshot_per_iteration = saveSnapshotPerIteration
        
        // Initialize gradient accumulation arrays
        let numPoints = model._xyz.shape[0]
        self.xyzGradAccumulation = MLXArray.zeros([numPoints])
        self.denomGradAccumulation = MLXArray.zeros([numPoints])
    }
    func fetchTrainData() -> (
        camera: Camera, rgb: MLXArray, depthMask: MLXArray, depth: MLXArray
    ) {
        let numCameras = data.getNumCameras()
        let ind = Int.random(in: 0..<numCameras)
        let rgb = data.rgbArray[ind]
        let depthMask = data.alphaArray[ind] .> 0.5
        let depth = data.depthArray?[ind] ?? MLXArray.zeros(depthMask.shape)
        let camera = data.getViewPointCamera(index: ind)
        return (camera, rgb, depthMask, depth)
    }

    private func cameraCenterArray(for camera: Camera) -> MLXArray {
        return MLXArray(
            [Float(camera.cameraCenter.x), Float(camera.cameraCenter.y), Float(camera.cameraCenter.z)]
                as [Float]
        )[.newAxis, .ellipsis]
    }

    private func decomposeCameraState(_ camera: Camera) -> CameraStateArrays {
        CameraStateArrays(
            viewMatrix: camera.worldViewTransform,
            projectionMatrix: camera.projectionMatrix,
            fovX: camera.FoVx,
            fovY: camera.FoVy,
            focalX: camera.focalX,
            focalY: camera.focalY,
            cameraCenter: cameraCenterArray(for: camera)
        )
    }

    private func makeTrainStepInputs(
        params: [MLXArray],
        camera: Camera,
        trainRGB: MLXArray,
        depthMask: MLXArray,
        trainDepth: MLXArray
    ) -> [MLXArray] {
        let cameraState = decomposeCameraState(camera)
        precondition(
            params.count == TrainStepInputIndex.parameterCount,
            "Expected \(TrainStepInputIndex.parameterCount) parameters, got \(params.count)"
        )
        return [
            params[TrainStepInputIndex.xyz],
            params[TrainStepInputIndex.featuresDC],
            params[TrainStepInputIndex.featuresRest],
            params[TrainStepInputIndex.scales],
            params[TrainStepInputIndex.rotation],
            params[TrainStepInputIndex.opacity],
            cameraState.viewMatrix,
            cameraState.projectionMatrix,
            cameraState.fovX,
            cameraState.fovY,
            cameraState.focalX,
            cameraState.focalY,
            cameraState.cameraCenter,
            trainRGB,
            depthMask,
            trainDepth,
        ]
    }

    private func buildCompiledLossAndGrad(
        gaussRender: GaussianRenderer,
        lambdaDepth: Float,
        lambdaDssim: Float
    ) -> ([MLXArray]) -> ([MLXArray], [MLXArray]) {
        // Compile only pure tensor transforms. Keep the dynamic renderer path
        // (which uses conditionToIndices -> item) outside compile.
        let preprocessCompiled: ([MLXArray]) -> [MLXArray] = MLX.compile(shapeless: true) { params in
            let _xyz = params[TrainStepInputIndex.xyz]
            let _features_dc = params[TrainStepInputIndex.featuresDC]
            let _features_rest = params[TrainStepInputIndex.featuresRest]
            let _scales = params[TrainStepInputIndex.scales]
            let _rotation = params[TrainStepInputIndex.rotation]
            let _opacity = params[TrainStepInputIndex.opacity]

            let means3d = gaussRender.get_xyz_from(_xyz)
            let opacity = gaussRender.get_opacity_from(_opacity)
            let scales = gaussRender.get_scales_from(_scales)
            let rotations = gaussRender.get_rotation_from(_rotation)
            let shs = gaussRender.get_features_from(_features_dc, _features_rest)
            return [means3d, shs, opacity, scales, rotations]
        }

        let lossFromRenderCompiled: ([MLXArray]) -> [MLXArray] = MLX.compile(shapeless: true) { inputs in
            let render = inputs[0]
            let depth = inputs[1]
            let trainRGB = inputs[2]
            let depthMask = inputs[3]
            let trainDepth = inputs[4]

            let l1LossValue = l1Loss(render, trainRGB)
            let depthDiff = MLX.abs(depth[.ellipsis, 0] - trainDepth)
            let depthMaskFloat = depthMask.asType(.float32)
            let depthWeight = depthMaskFloat.sum()
            let safeDepthWeight = MLX.maximum(depthWeight, MLXArray(1e-6 as Float))
            let depthLossValue = (depthDiff * depthMaskFloat).sum() / safeDepthWeight
            let ssimLossValue = 1.0 - ssim(img1: render[.newAxis], img2: trainRGB[.newAxis])

            let totalLoss =
                (1.0 - lambdaDssim) * l1LossValue
                + lambdaDssim * ssimLossValue
                + lambdaDepth * depthLossValue
            return [totalLoss, render]
        }

        let lossFn: ([MLXArray]) -> [MLXArray] = { inputs in
            let viewMatrix = inputs[TrainStepInputIndex.viewMatrix]
            let projectionMatrix = inputs[TrainStepInputIndex.projectionMatrix]
            let fovX = inputs[TrainStepInputIndex.fovX]
            let fovY = inputs[TrainStepInputIndex.fovY]
            let focalX = inputs[TrainStepInputIndex.focalX]
            let focalY = inputs[TrainStepInputIndex.focalY]
            let cameraCenter = inputs[TrainStepInputIndex.cameraCenter]
            let trainRGB = inputs[TrainStepInputIndex.targetRGB]
            let depthMask = inputs[TrainStepInputIndex.depthMask]
            let trainDepth = inputs[TrainStepInputIndex.targetDepth]

            let preprocessed = preprocessCompiled(Array(inputs.prefix(TrainStepInputIndex.parameterCount)))
            let means3d = preprocessed[0]
            let shs = preprocessed[1]
            let opacity = preprocessed[2]
            let scales = preprocessed[3]
            let rotations = preprocessed[4]

            let (render, depth, _, _, _) = gaussRender.forwardWithCameraParams(
                viewMatrix: viewMatrix,
                projMatrix: projectionMatrix,
                cameraCenter: cameraCenter,
                fovX: fovX,
                fovY: fovY,
                focalX: focalX,
                focalY: focalY,
                imageWidth: gaussRender.W,
                imageHeight: gaussRender.H,
                means3d: means3d,
                shs: shs,
                opacity: opacity,
                scales: scales,
                rotations: rotations
            )

            return lossFromRenderCompiled([render, depth, trainRGB, depthMask, trainDepth])
        }

        return MLX.valueAndGrad(
            lossFn,
            argumentNumbers: Array(0..<TrainStepInputIndex.parameterCount)
        )
    }
    func addGradientAccumulation(xyzGrad: MLXArray) {
        // Calculate gradient magnitude (L2 norm)
        let gradNorm = MLX.sqrt(MLX.sum(MLX.square(xyzGrad), axes: [1]))
        let numPoints = xyzGrad.shape[0]
        
        if xyzGradAccumulation.shape[0] != numPoints {
            xyzGradAccumulation = MLXArray.zeros([numPoints])
            denomGradAccumulation = MLXArray.zeros([numPoints])
        }
        
        xyzGradAccumulation = xyzGradAccumulation + gradNorm
        denomGradAccumulation = denomGradAccumulation + MLXArray.ones([numPoints])
    }
    
    func resetGradientAccumulation() {
        let numPoints = model._xyz.shape[0]
        xyzGradAccumulation = MLXArray.zeros([numPoints])
        denomGradAccumulation = MLXArray.zeros([numPoints])
    }
    
    func split_and_prune(params: [MLXArray], states: [TupleState], iteration: Int) {
        guard iteration >= densifyFromIter && iteration <= densifyUntilIter else {
            return
        }
        
        var _xyz = params[0]
        var _features_dc = params[1]
        var _features_rest = params[2]
        var _scales = params[3]
        var _rotation = params[4]
        var _opacity = params[5]
        
        let numPoints = _xyz.shape[0]
        
        // Check if we've reached the maximum number of Gaussians
        if numPoints >= maxGaussians {
            Logger.shared.info("Reached maximum Gaussians limit: \(maxGaussians). Skipping densification.")
            // Still perform pruning
            let opacity = MLX.sigmoid(_opacity)
            let pruneMask = (opacity .< minOpacity).reshaped([-1])
            if MLX.sum(pruneMask.asType(.int32)).item(Int.self) > 0 {
                let keepMask: MLXArray = .!pruneMask
                let keepIndices = conditionToIndices(condition: keepMask)
                (_xyz, _features_dc, _features_rest, _scales, _rotation, _opacity) = 
                    pruneGaussians(
                        xyz: _xyz, features_dc: _features_dc, features_rest: _features_rest,
                        scales: _scales, rotation: _rotation, opacity: _opacity,
                        indices: keepIndices
                    )
                model._xyz = _xyz
                model._features_dc = _features_dc
                model._features_rest = _features_rest
                model._scales = _scales
                model._rotation = _rotation
                model._opacity = _opacity
            }
            resetGradientAccumulation()
            return
        }
        
        // Calculate average gradient magnitude (already computed as magnitude in addGradientAccumulation)
        let avgGrads = xyzGradAccumulation / denomGradAccumulation
        let gradMagnitude = avgGrads
        
        // Get current scales and opacity
        let scales = MLX.exp(_scales)
        let opacity = MLX.sigmoid(_opacity)
        
        // Find points to densify (high gradient)
        let gradMask = gradMagnitude .> gradientThreshold
        
        // Find points to split (large scale)
        let maxScalePerGaussian = MLX.max(scales, axes: [1])
        let splitMask = gradMask & (maxScalePerGaussian .> MLXArray(maxScale))
        
        // Find points to clone (small scale) 
        let cloneMask = gradMask & (maxScalePerGaussian .<= MLXArray(maxScale))
        
        // Find points to prune (low opacity)
        let pruneMask = (opacity .< minOpacity).reshaped([-1])
        
        // Perform splitting
        if MLX.sum(splitMask.asType(.int32)).item(Int.self) > 0 {
            let splitIndices = conditionToIndices(condition: splitMask)
            (_xyz, _features_dc, _features_rest, _scales, _rotation, _opacity) = 
                splitGaussians(
                    xyz: _xyz, features_dc: _features_dc, features_rest: _features_rest,
                    scales: _scales, rotation: _rotation, opacity: _opacity,
                    indices: splitIndices
                )
        }
        
        // Perform cloning
        if MLX.sum(cloneMask.asType(.int32)).item(Int.self) > 0 {
            let cloneIndices = conditionToIndices(condition: cloneMask)
            (_xyz, _features_dc, _features_rest, _scales, _rotation, _opacity) = 
                cloneGaussians(
                    xyz: _xyz, features_dc: _features_dc, features_rest: _features_rest,
                    scales: _scales, rotation: _rotation, opacity: _opacity,
                    indices: cloneIndices
                )
        }
        
        // Perform pruning
        if MLX.sum(pruneMask.asType(.int32)).item(Int.self) > 0 {
            let keepMask: MLXArray = .!pruneMask
            let keepIndices = conditionToIndices(condition: keepMask)
            (_xyz, _features_dc, _features_rest, _scales, _rotation, _opacity) = 
                pruneGaussians(
                    xyz: _xyz, features_dc: _features_dc, features_rest: _features_rest,
                    scales: _scales, rotation: _rotation, opacity: _opacity,
                    indices: keepIndices
                )
        }
        
        // Update model parameters
        model._xyz = _xyz
        model._features_dc = _features_dc
        model._features_rest = _features_rest
        model._scales = _scales
        model._rotation = _rotation
        model._opacity = _opacity
        
        // Reset gradient accumulation
        resetGradientAccumulation()
    }
    
    /// Split large Gaussians into two smaller ones
    /// Each Gaussian is replaced by two Gaussians with reduced scale and slightly offset positions
    func splitGaussians(
        xyz: MLXArray, features_dc: MLXArray, features_rest: MLXArray,
        scales: MLXArray, rotation: MLXArray, opacity: MLXArray,
        indices: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        
        let selectedXYZ = xyz[indices]
        let selectedFeaturesDC = features_dc[indices]
        let selectedFeaturesRest = features_rest[indices]
        let selectedScales = scales[indices]
        let selectedRotation = rotation[indices]
        let selectedOpacity = opacity[indices]
        
        // Scale down the selected Gaussians (divide by 1.6, so subtract log(1.6))
        let newScales = selectedScales - MLX.log(MLXArray(1.6))
        
        // Create two new Gaussians for each split
        // Sample positions based on the actual Gaussian scale for geometry-aware splitting
        let numSplit = indices.shape[0]
        let baseNoise = MLXRandom.normal([numSplit, 3])
        let actualScales = MLX.exp(selectedScales)
        let scaledNoise = baseNoise * MLX.mean(actualScales, axes: [1], keepDims: true) * 0.1
        
        let newXYZ1 = selectedXYZ + scaledNoise
        let newXYZ2 = selectedXYZ - scaledNoise
        
        // Create mask to keep Gaussians that are NOT being split
        let totalPoints = xyz.shape[0]
        let indicesRaw = indices.asArray(Int.self)
        var keepMask = Swift.Array(repeating: true, count: totalPoints)
        // Set split indices to false in keep mask
        for i in 0..<indicesRaw.count {
            let idx = indicesRaw[i]
            if idx < totalPoints {
                keepMask[idx] = false
            }
        }
        
        let keepIndices = conditionToIndices(condition: MLXArray(keepMask))
        
        // Keep non-split Gaussians and add the two new ones for each split
        let keptXYZ = xyz[keepIndices]
        let keptFeaturesDC = features_dc[keepIndices]
        let keptFeaturesRest = features_rest[keepIndices]
        let keptScales = scales[keepIndices]
        let keptRotation = rotation[keepIndices]
        let keptOpacity = opacity[keepIndices]
        
        let newXYZAll = MLX.concatenated([keptXYZ, newXYZ1, newXYZ2], axis: 0)
        let newFeaturesDCAll = MLX.concatenated([keptFeaturesDC, selectedFeaturesDC, selectedFeaturesDC], axis: 0)
        let newFeaturesRestAll = MLX.concatenated([keptFeaturesRest, selectedFeaturesRest, selectedFeaturesRest], axis: 0)
        let newScalesAll = MLX.concatenated([keptScales, newScales, newScales], axis: 0)
        let newRotationAll = MLX.concatenated([keptRotation, selectedRotation, selectedRotation], axis: 0)
        let newOpacityAll = MLX.concatenated([keptOpacity, selectedOpacity, selectedOpacity], axis: 0)
        
        return (newXYZAll, newFeaturesDCAll, newFeaturesRestAll, newScalesAll, newRotationAll, newOpacityAll)
    }
    
    /// Clone Gaussians by duplicating them with slight position offset
    /// Used for small Gaussians in high-gradient areas
    func cloneGaussians(
        xyz: MLXArray, features_dc: MLXArray, features_rest: MLXArray,
        scales: MLXArray, rotation: MLXArray, opacity: MLXArray,
        indices: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        
        let selectedXYZ = xyz[indices]
        let selectedFeaturesDC = features_dc[indices]
        let selectedFeaturesRest = features_rest[indices]
        let selectedScales = scales[indices]
        let selectedRotation = rotation[indices]
        let selectedOpacity = opacity[indices]
        
        // Add small noise to position to avoid exact duplicates
        let noise = MLXRandom.normal(selectedXYZ.shape) * 0.01
        let newXYZ = selectedXYZ + noise
        
        // Concatenate cloned Gaussians
        let newXYZAll = MLX.concatenated([xyz, newXYZ], axis: 0)
        let newFeaturesDCAll = MLX.concatenated([features_dc, selectedFeaturesDC], axis: 0)
        let newFeaturesRestAll = MLX.concatenated([features_rest, selectedFeaturesRest], axis: 0)
        let newScalesAll = MLX.concatenated([scales, selectedScales], axis: 0)
        let newRotationAll = MLX.concatenated([rotation, selectedRotation], axis: 0)
        let newOpacityAll = MLX.concatenated([opacity, selectedOpacity], axis: 0)
        
        return (newXYZAll, newFeaturesDCAll, newFeaturesRestAll, newScalesAll, newRotationAll, newOpacityAll)
    }
    
    /// Remove Gaussians by keeping only those at specified indices
    /// Used to prune low-opacity Gaussians
    func pruneGaussians(
        xyz: MLXArray, features_dc: MLXArray, features_rest: MLXArray,
        scales: MLXArray, rotation: MLXArray, opacity: MLXArray,
        indices: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        
        let newXYZ = xyz[indices]
        let newFeaturesDC = features_dc[indices]
        let newFeaturesRest = features_rest[indices]
        let newScales = scales[indices]
        let newRotation = rotation[indices]
        let newOpacity = opacity[indices]
        
        return (newXYZ, newFeaturesDC, newFeaturesRest, newScales, newRotation, newOpacity)
    }
    func save_snapshot(iteration: Int, params: [MLXArray]) {
        //TODO
        if let outputDirectoryURL {
            let xyz = params[0]
            let features_dc = params[1]
            let features_rest = params[2]
            let scales = params[3]
            let rotation = params[4]
            let opacity = params[5]
            do {
                let outputURL = outputDirectoryURL.appendingPathComponent("iteration_\(iteration).ply")
                try PlyWriter.writeGaussianBinary(
                    positions: xyz, features_dc: features_dc, features_rest: features_rest,
                    opacities: opacity, scales: scales, rotations: rotation,
                    to: outputURL)
                delegate?.pushSnapshot(url: outputURL, iteration: iteration, timestamp: Date())
            } catch {
                Logger.shared.error(error: error)
            }
        }
    }
    var forceStop: Bool = false
    func stopTrain() {
        forceStop = true
    }
    func startTrain(earlyStoppingThreshold: Float = 0.0001) {
        let gaussRender = self.gaussRender
        let model = self.model
        var params = model.getParams()

        let optimizer = Adam(
            learningRate: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-15
        )
        var states = params.map {
            optimizer.newState(parameter: $0)
        }
        let effectiveLambdaDepth = data.depthArray != nil ? lambda_depth : 0.0
        let trainWithGrad = buildCompiledLossAndGrad(
            gaussRender: gaussRender,
            lambdaDepth: effectiveLambdaDepth,
            lambdaDssim: lambda_dssim
        )

        var fps: Float? = nil
        for iteration in 0..<iterationCount {
            if forceStop {
                break
            }
            let iterationStartTime = Date()
            Logger.shared.debug("\(iteration)th iteration")
            let (trainCamera, trainRGB, trainDepthMask, trainDepth) =
                fetchTrainData()
            let trainInputs = makeTrainStepInputs(
                params: params,
                camera: trainCamera,
                trainRGB: trainRGB,
                depthMask: trainDepthMask,
                trainDepth: trainDepth
            )
            let (loss, grads) = trainWithGrad(trainInputs)
            eval(loss[0], grads)
            let lossValue = loss[0].item(Float.self)
            
            // Accumulate gradients for densification
            addGradientAccumulation(xyzGrad: grads[0])
            
            // Calculate FPS
            let iterationEndTime = Date()
            let iterationDuration = iterationEndTime.timeIntervalSince(iterationStartTime)
            if iterationDuration > 0 {
                fps = Float(1.0 / iterationDuration)
            }
            
            delegate?.pushLoss(
                loss: lossValue,
                iteration: iteration,
                fps: fps,
                timestamp: Date())
            if iteration % 20 == 0 {
                let image = loss[1]
                eval(image)
                delegate?.pushImageData(
                    render: image,
                    truth: trainRGB,
                    loss: lossValue,
                    iteration: iteration, timestamp: Date()
                )
            }
            if lossValue < earlyStoppingThreshold {
                Logger.shared.info("early stopping")
                return
            }
            let lrs = model.getLearningRates(
                current: iteration,
                total: iterationCount
            )
            for i in 0..<params.count {
                Logger.shared.debug("update \(i)th param start")
                optimizer.learningRate = lrs[i]
                let (newParam, newState) = optimizer.applySingle(
                    gradient: grads[i],
                    parameter: params[i],
                    state: states[i]
                )
                params[i] = newParam
                states[i] = newState
                Logger.shared.debug("update \(i)th param end")
            }
            eval(params)
            eval(optimizer)
            if manualClearCache && MLX.GPU.snapshot().cacheMemory > cacheLimit {
                clearCacheIfNeeded()
            }
            if iteration % self.save_snapshot_per_iteration == 0 {
                self.save_snapshot(iteration: iteration, params: params)
                clearCacheIfNeeded()
            }
            if iteration % self.split_and_prune_per_iteration == 0 {
                self.split_and_prune(params: params, states: states, iteration: iteration)
                // Update params after split_and_prune since number of Gaussians may have changed
                params = model.getParams()
                // Reinitialize optimizer states for new parameters
                // This is necessary because split/clone operations change the parameter count
                states = params.map {
                    optimizer.newState(parameter: $0)
                }
                clearCacheIfNeeded()
            }
        }
        clearCacheIfNeeded()
    }
    static func createModel(
        sh_degree: Int,
        pointCloud: PointCloud,
        sampleCount: Int = 1 << 14,
        manualClearCache: Bool = GaussianTrainer.needEagerClearCache()
    ) -> GaussModel {
        Logger.shared.debug("get point clouds...")
        Logger.shared.debug("Random sample....")
        let raw_points = pointCloud.randomSample(sampleCount)
        let gaussModel = GaussModel.create_from_pcd(
            pcd: raw_points,
            sh_degree: sh_degree,
            debug: false,
            manualClearCache: manualClearCache
        )
        return gaussModel
    }
}
