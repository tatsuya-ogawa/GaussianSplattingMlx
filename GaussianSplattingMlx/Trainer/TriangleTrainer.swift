import Foundation
import MLX
import MLXOptimizers

final class TriangleTrainer {
    var data: TrainData
    var model: TriangleModel
    private let triangleRenderer: TriangleRenderer
    private let gaussianRenderer: GaussianRenderer
    var iterationCount: Int
    var lambda_dssim: Float = 0.2
    var lambda_depth: Float = 0.0
    var lambda_weight: Float = 0.01
    var pruneOpacity: Float = 0.02
    var pruneInterval: Int = 500
    var saveSnapshotPerIteration: Int
    var cacheLimit: Int
    var manualClearCache: Bool
    var outputDirectoryURL: URL?
    weak var delegate: GaussianTrainerDelegate?

    private var forceStop = false

    init(
        model: TriangleModel,
        data: TrainData,
        tileSize: TILE_SIZE_H_W,
        whiteBackground: Bool,
        iterationCount: Int,
        cacheLimit: Int = GaussianTrainer.defaultCacheLimit(),
        manualClearCache: Bool = GaussianTrainer.needEagerClearCache(),
        outputDirectoryURL: URL? = nil,
        saveSnapshotPerIteration: Int = 250
    ) {
        self.model = model
        self.data = data
        let width = data.Ws[0].item(Int.self)
        let height = data.Hs[0].item(Int.self)
        self.gaussianRenderer = GaussianRenderer(
            active_sh_degree: model.maxShDegree,
            W: width,
            H: height,
            TILE_SIZE: tileSize,
            whiteBackground: whiteBackground
        )
        self.triangleRenderer = TriangleRenderer(gaussianRenderer: gaussianRenderer)
        self.iterationCount = iterationCount
        self.cacheLimit = cacheLimit
        self.manualClearCache = manualClearCache
        self.outputDirectoryURL = outputDirectoryURL
        self.saveSnapshotPerIteration = saveSnapshotPerIteration
    }

    func stopTrain() {
        forceStop = true
    }

    func startTrain(earlyStoppingThreshold: Float = 1e-4) {
        var params = model.getParams()
        let optimizer = Adam(learningRate: 1e-3, betas: (0.9, 0.999), eps: 1e-15)
        var states = params.map { optimizer.newState(parameter: $0) }
        var fps: Float? = nil

        for iteration in 0..<iterationCount {
            if forceStop { break }
            model.promoteShDegree(iteration: iteration)

            let iterationStart = Date()
            let (camera, rgb, mask, depth) = fetchTrainSample()

            let closure: ([MLXArray]) -> [MLXArray] = { tensors in
                let vertices = tensors[TriangleModel.ParamIndex.vertices.rawValue]
                let featuresDC = tensors[TriangleModel.ParamIndex.featuresDC.rawValue]
                let featuresRest = tensors[TriangleModel.ParamIndex.featuresRest.rawValue]
                let opacity = tensors[TriangleModel.ParamIndex.opacity.rawValue]
                let renderPack = self.triangleRenderer.forward(
                    camera: camera,
                    vertices: vertices,
                    featuresDC: featuresDC,
                    featuresRest: featuresRest,
                    opacityLogits: opacity
                )
                let render = renderPack.render
                let l1 = l1Loss(render, rgb)
                let ssimLoss = 1.0 - ssim(img1: render[.newAxis], img2: rgb[.newAxis])
                let depthLoss: MLXArray
                if let gtDepth = depth {
                    depthLoss = l1Loss(
                        renderPack.depth[.ellipsis, 0].reshaped([-1])[mask],
                        gtDepth.reshaped([-1])[mask]
                    )
                } else {
                    depthLoss = MLXArray(0.0 as Float)
                }
                let vertexWeights = tensors[TriangleModel.ParamIndex.vertexWeights.rawValue]
                let weightReg = MLX.mean(MLX.square(vertexWeights))
                let total = (1.0 - self.lambda_dssim) * l1
                    + self.lambda_dssim * ssimLoss
                    + self.lambda_depth * depthLoss
                    + self.lambda_weight * weightReg
                return [total, render]
            }

            let (loss, grads) = MLX.valueAndGrad(
                closure,
                argumentNumbers: Array(0..<params.count)
            )(params)
            eval(loss[0], grads)
            let lossValue = loss[0].item(Float.self)

            let learningRates = model.getLearningRates(current: iteration, total: iterationCount)
            for index in 0..<params.count {
                optimizer.learningRate = learningRates[index]
                let (newTensor, newState) = optimizer.applySingle(
                    gradient: grads[index],
                    parameter: params[index],
                    state: states[index]
                )
                eval(newTensor)
                params[index] = newTensor
                states[index] = newState
            }
            model.syncFromParams(params)

            if iteration % pruneInterval == 0 && iteration > 0 {
                pruneTriangles(
                    optimizer: optimizer,
                    params: &params,
                    states: &states
                )
            }

            if manualClearCache && MLX.GPU.snapshot().cacheMemory > cacheLimit {
                MLX.GPU.clearCache()
            }

            if iteration % saveSnapshotPerIteration == 0 {
                saveSnapshot(iteration: iteration, params: params)
            }

            let iterationDuration = Date().timeIntervalSince(iterationStart)
            if iterationDuration > 1e-6 {
                fps = Float(1.0 / iterationDuration)
            }

            delegate?.pushLoss(
                loss: lossValue,
                iteration: iteration,
                fps: fps,
                timestamp: Date()
            )
            if iteration % 20 == 0 {
                let render = loss[1]
                eval(render)
                delegate?.pushImageData(
                    render: render,
                    truth: rgb,
                    loss: lossValue,
                    iteration: iteration,
                    timestamp: Date()
                )
            }

            if lossValue < earlyStoppingThreshold {
                Logger.shared.info("Triangle trainer early stopping.")
                break
            }
        }
        MLX.GPU.clearCache()
    }

    private func fetchTrainSample() -> (Camera, MLXArray, MLXArray, MLXArray?) {
        let numCameras = data.getNumCameras()
        let index = Int.random(in: 0..<numCameras)
        let rgb = data.rgbArray[index]
        let depth = data.depthArray?[index]
        let mask = conditionToIndices(condition: (data.alphaArray[index] .> 0.5).reshaped([-1]))
        let camera = data.getViewPointCamera(index: index)
        return (camera, rgb, mask, depth)
    }

    private func pruneTriangles(
        optimizer: Adam,
        params: inout [MLXArray],
        states: inout [TupleState]
    ) {
        let opacity = params[TriangleModel.ParamIndex.opacity.rawValue]
        let alpha = MLX.sigmoid(opacity).reshaped([-1])
        let keepMask = alpha .>= MLXArray(pruneOpacity)
        let keepCount = MLX.sum(keepMask.asType(.int32)).item(Int.self)
        guard keepCount > 0 && keepCount < alpha.shape[0] else { return }
        let keepIndices = conditionToIndices(condition: keepMask)
        var updated: [MLXArray] = []
        for tensor in params {
            updated.append(tensor[keepIndices])
        }
        params = updated
        model.syncFromParams(params)
        states = params.map { optimizer.newState(parameter: $0) }
    }

    private func saveSnapshot(iteration: Int, params: [MLXArray]) {
        guard let outputDirectoryURL else { return }
        let pseudo = model.pseudoGaussianParams(
            vertices: params[TriangleModel.ParamIndex.vertices.rawValue],
            featuresDC: params[TriangleModel.ParamIndex.featuresDC.rawValue],
            featuresRest: params[TriangleModel.ParamIndex.featuresRest.rawValue],
            opacityLogits: params[TriangleModel.ParamIndex.opacity.rawValue]
        )
        do {
            let url = outputDirectoryURL.appendingPathComponent("triangles_\(iteration).ply")
            try PlyWriter.writeGaussianBinary(
                positions: pseudo.means,
                features_dc: params[TriangleModel.ParamIndex.featuresDC.rawValue],
                features_rest: params[TriangleModel.ParamIndex.featuresRest.rawValue],
                opacities: params[TriangleModel.ParamIndex.opacity.rawValue],
                scales: pseudo.scales,
                rotations: pseudo.rotations,
                to: url
            )
            delegate?.pushSnapshot(url: url, iteration: iteration, timestamp: Date())
        } catch {
            Logger.shared.error(error: error)
        }
    }
}
