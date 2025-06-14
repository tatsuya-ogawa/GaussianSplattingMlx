//
//  GaussianTrainer.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import Foundation
import MLX
import MLXOptimizers

class TrainData {
    let Hs: MLXArray
    let Ws: MLXArray
    let intrinsicArray: MLXArray
    let c2wArray: MLXArray
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
    init(
        model: GaussModel,
        data: TrainData,
        gaussRender: GaussianRenderer,
        iterationCount: Int,
        cacheLimit: Int =  2 * 1024 * 1024 * 1024,
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
    func split_and_prune(params: [MLXArray], states: [TupleState]) {
        //TODO
    }
    func save_snapshot(iteration: Int, params: [MLXArray]) {
        //TODO
        if let outputDirectoryURL{
            let xyz = params[0]
            let features_dc = params[1]
            let features_rest = params[2]
            let scales = params[3]
            let rotation = params[4]
            let opacity = params[5]
            do{
                try PlyWriter.writeGaussianBinary(positions: xyz, features_dc: features_dc, features_rest: features_rest, opacities: opacity, scales: scales, rotations: rotation, to: outputDirectoryURL.appendingPathComponent("iteration_\(iteration).ply"))
            }catch{
                Logger.shared.error(error: error)
            }
        }
    }
    var forceStop: Bool = false
    func stopTrain(){
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
                let means3d = GaussModel.get_xyz_from(_xyz)
                let opacity = GaussModel.get_opacity_from(_opacity)
                let scales = GaussModel.get_scales_from(_scales)
                let rotations = GaussModel.get_rotation_from(_rotation)
                let shs = GaussModel.get_features_from(
                    _features_dc,
                    _features_rest
                )
                Logger.shared.debug("Prepare forward")
                let (
                    render,
                    depth,
                    alpha,
                    visiility_filter,
                    radii
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

            LossChartData.shared.push(
                loss: lossValue,
                iteration: iteration,
                timestamp: Date()
            )
            if iteration % 20 == 0 {
                let image = loss[1]
                eval(image)
                TrainStatusData.shared.pushImageData(
                    render: image,
                    truth: trainRGB,
                    loss: lossValue,
                    iteration: iteration
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
                self.split_and_prune(params: params, states: states)
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
