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
}
protocol GaussianTrainerDelegate: AnyObject {
    func pushLoss(loss: Float, iteration: Int?, timestamp: Date)
    func pushImageData(
        render: MLXArray, truth: MLXArray, loss: Float, iteration: Int, timestamp: Date)
    func pushSnapshot(url: URL,iteration: Int, timestamp: Date)
}
class GaussianTrainer {
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
    weak var delegate: GaussianTrainerDelegate?
    
    // Split and prune parameters
    var gradientThreshold: Float = 0.0002
    var maxScreenSize: Float = 20.0
    var minOpacity: Float = 0.005
    var maxScale: Float = 0.01
    var pruneInterval: Int = 100
    var densifyFromIter: Int = 500
    var densifyUntilIter: Int = 15000
    
    // Tracking gradients for densification
    var xyzGradAccumulation: MLXArray = MLXArray.zeros([0, 3])
    var denomGradAccumulation: MLXArray = MLXArray.zeros([0])
    init(
        model: GaussModel,
        data: TrainData,
        gaussRender: GaussianRenderer,
        iterationCount: Int,
        cacheLimit: Int = 2 * 1024 * 1024 * 1024,
        outputDirectoryURL: URL? = nil,
        saveSnapshotPerIteration: Int = 100
    ) {
        self.model = model
        self.data = data
        self.gaussRender = gaussRender
        self.iterationCount = iterationCount
        self.cacheLimit = cacheLimit
        self.outputDirectoryURL = outputDirectoryURL
        self.save_snapshot_per_iteration = saveSnapshotPerIteration
        
        // Initialize gradient accumulation arrays
        let numPoints = model._xyz.shape[0]
        self.xyzGradAccumulation = MLXArray.zeros([numPoints])
        self.denomGradAccumulation = MLXArray.zeros([numPoints])
    }
    func fetchTrainData() -> (
        camera: Camera, rgb: MLXArray, mask: MLXArray, depth: MLXArray?
    ) {
        let numCameras = data.getNumCameras()
        let ind = Int.random(in: 0..<numCameras)
        let rgb = data.rgbArray[ind]
        let depth = data.depthArray?[ind]
        let mask = conditionToIndices(
            condition: (data.alphaArray[ind] .> 0.5).reshaped([-1])
        )
        let camera = data.getViewPointCamera(index: ind)
        return (camera, rgb, mask, depth)
    }
    func addGradientAccumulation(xyzGrad: MLXArray) {
        let gradNorm = MLX.sum(MLX.square(xyzGrad), axes: [1])
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
        
        // Calculate average gradient magnitude
        let avgGrads = xyzGradAccumulation / denomGradAccumulation.expandedDimensions(axes: [1])
        let gradMagnitude = MLX.sqrt(avgGrads)
        
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
        
        // Scale down the selected Gaussians
        let newScales = selectedScales - MLX.log(MLXArray(1.6))
        
        // Create two new Gaussians for each split
        let numSplit = indices.shape[0]
        let noise = MLXRandom.normal([numSplit, 3]) * 0.1
        
        let newXYZ1 = selectedXYZ + noise
        let newXYZ2 = selectedXYZ - noise
        
        // Create mask to keep Gaussians that are NOT being split
        let totalPoints = xyz.shape[0]
        var keepMask = MLXArray.ones([totalPoints], dtype: .bool)
        
        // Set split indices to false in keep mask
        for i in 0..<indices.shape[0] {
            let idx = indices[i].item(Int.self)
            if idx < totalPoints {
                keepMask[idx] = MLXArray(false)
            }
        }
        
        let keepIndices = conditionToIndices(condition: keepMask)
        
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
        
        // Add small noise to position
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
        let trainer: GaussianTrainer = self
        let gaussRender = trainer.gaussRender
        let model = trainer.model
        let lambda_depth = trainer.lambda_depth
        let lambda_dssim = trainer.lambda_dssim
        var params = model.getParams()

        let optimizer = Adam(
            learningRate: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-15
        )
        var states = params.map {
            optimizer.newState(parameter: $0)
        }
        for iteration in 0..<iterationCount {
            if forceStop {
                break
            }
            Logger.shared.debug("\(iteration)th iteration")
            let (trainCamera, trainRGB, trainMask, trainDepth) =
                fetchTrainData()
            let train: ([MLXArray]) -> [MLXArray] = { params in
                let _xyz = params[0]
                let _features_dc = params[1]
                let _features_rest = params[2]
                let _scales = params[3]
                let _rotation = params[4]
                let _opacity = params[5]
                Logger.shared.debug("Prepare variables")
                let means3d = gaussRender.get_xyz_from(_xyz)
                let opacity = gaussRender.get_opacity_from(_opacity)
                let scales = gaussRender.get_scales_from(_scales)
                let rotations = gaussRender.get_rotation_from(_rotation)
                let shs = gaussRender.get_features_from(
                    _features_dc,
                    _features_rest
                )
                Logger.shared.debug("Prepare forward")
                let (
                    render,
                    depth,
                    _,
                    _,
                    _
                ) = gaussRender.forward(
                    camera: trainCamera,
                    means3d: means3d,
                    shs: shs,
                    opacity: opacity,
                    scales: scales,
                    rotations: rotations
                )
                Logger.shared.debug("Calculate loss")
                let l1_loss = l1Loss(render, trainRGB)
                let depth_loss =
                    trainDepth != nil
                    ? l1Loss(
                        depth[.ellipsis, 0].reshaped([-1])[trainMask],
                        trainDepth!.reshaped([-1])[trainMask]
                    ) : MLXArray(0.0 as Float)
                let ssim_loss =
                    1.0 - ssim(img1: render[.newAxis], img2: trainRGB[.newAxis])

                let total_loss =
                    (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss
                    + lambda_depth * depth_loss
                return [total_loss, render]
            }
            Logger.shared.debug("valueAndGrad")
            let (loss, grads) = MLX.valueAndGrad(
                train,
                argumentNumbers: Array(0..<params.count)
            )(params)
            eval(loss[0], grads)
            let lossValue = loss[0].item(Float.self)
            
            // Accumulate gradients for densification
            addGradientAccumulation(xyzGrad: grads[0])
            
            delegate?.pushLoss(
                loss: lossValue,
                iteration: iteration,
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
                eval(newParam)
                params[i] = newParam
                states[i] = newState
                Logger.shared.debug("update \(i)th param end")
            }
            eval(optimizer)
            if MLX.GPU.snapshot().cacheMemory > cacheLimit {
                MLX.GPU.clearCache()
            }
            if iteration % self.save_snapshot_per_iteration == 0 {
                self.save_snapshot(iteration: iteration, params: params)
                MLX.GPU.clearCache()
            }
            if iteration % self.split_and_prune_per_iteration == 0 {
                self.split_and_prune(params: params, states: states, iteration: iteration)
                // Update params after split_and_prune
                params = model.getParams()
                // Reinitialize optimizer states for new parameters
                states = params.map {
                    optimizer.newState(parameter: $0)
                }
                MLX.GPU.clearCache()
            }
        }
        MLX.GPU.clearCache()
    }
    static func createModel(
        sh_degree: Int,
        pointCloud: PointCloud,
        sampleCount: Int = 1 << 14
    ) -> GaussModel {
        Logger.shared.debug("get point clouds...")
        Logger.shared.debug("Random sample....")
        let raw_points = pointCloud.randomSample(sampleCount)
        let gaussModel = GaussModel.create_from_pcd(
            pcd: raw_points,
            sh_degree: sh_degree,
            debug: false
        )
        return gaussModel
    }
}
