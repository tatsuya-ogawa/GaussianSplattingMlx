//
//  GaussianTrainer.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import Foundation
import MLX
import MLXFast
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

final class IntervalProfiler {
    struct Metric {
        var totalNanoseconds: UInt64 = 0
        var selfNanoseconds: UInt64 = 0
        var count: Int = 0
    }

    private struct Frame {
        let startNanoseconds: UInt64
        var childNanoseconds: UInt64 = 0
    }

    let enabled: Bool
    private var metrics: [String: Metric] = [:]
    private var stack: [Frame] = []

    init(enabled: Bool) {
        self.enabled = enabled
    }

    @inline(__always)
    private func nowNanoseconds() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds
    }

    @discardableResult
    func measure<T>(_ name: String, _ body: () -> T) -> T {
        guard enabled else {
            return body()
        }

        let start = nowNanoseconds()
        stack.append(Frame(startNanoseconds: start))
        let value = body()
        let end = nowNanoseconds()

        guard let frame = stack.popLast() else {
            return value
        }
        let elapsed = end >= frame.startNanoseconds
            ? end - frame.startNanoseconds
            : 0
        let selfElapsed = elapsed >= frame.childNanoseconds
            ? elapsed - frame.childNanoseconds
            : 0

        var metric = metrics[name] ?? Metric()
        metric.totalNanoseconds += elapsed
        metric.selfNanoseconds += selfElapsed
        metric.count += 1
        metrics[name] = metric

        if let parentIndex = stack.indices.last {
            stack[parentIndex].childNanoseconds += elapsed
        }
        return value
    }

    func makeReport(
        iteration: Int,
        iterationNanoseconds: UInt64,
        topK: Int = 12,
        minMilliseconds: Double = 0.0
    ) -> String {
        guard enabled else { return "" }
        let iterationMs = Double(iterationNanoseconds) / 1_000_000.0
        let totalSelfNs = metrics.values.reduce(UInt64(0)) {
            $0 + $1.selfNanoseconds
        }
        if totalSelfNs == 0 {
            return String(
                format: "[Profile] iter=%d wall=%.3f ms (no measured sections)",
                iteration,
                iterationMs
            )
        }

        let sorted = metrics.sorted {
            $0.value.selfNanoseconds > $1.value.selfNanoseconds
        }
        var lines: [String] = [
            String(
                format: "[Profile] iter=%d wall=%.3f ms (top %d)",
                iteration,
                iterationMs,
                topK
            )
        ]
        var shown = 0
        for (name, metric) in sorted {
            if shown >= topK {
                break
            }
            let selfMs = Double(metric.selfNanoseconds) / 1_000_000.0
            if selfMs < minMilliseconds {
                continue
            }
            let totalMs = Double(metric.totalNanoseconds) / 1_000_000.0
            let avgMs = totalMs / Double(max(metric.count, 1))
            let share = Double(metric.selfNanoseconds) / Double(totalSelfNs) * 100.0
            lines.append(
                String(
                    format:
                        "  %@: self %.3f ms, total %.3f ms (%.1f%%, calls=%d, avg=%.4f ms)",
                    name,
                    selfMs,
                    totalMs,
                    share,
                    metric.count,
                    avgMs
                )
            )
            shown += 1
        }
        if shown == 0 {
            lines.append("  (all sections are below threshold)")
        }
        return lines.joined(separator: "\n")
    }
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
    var enableIntervalProfiling: Bool
    var profilingLogInterval: Int
    var profilingTopKSections: Int
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
    var denomGradAccumulation: Int = 0  // Simple counter (same for all points)
    
    // Precomputed cameras to avoid ~20 GPU sync points (.item()) per iteration
    private var cachedCameras: [Camera] = []
    
    // MARK: - Custom Metal Kernels for split_and_prune
    
    /// Fused gradient accumulation kernel: reads xyz_grad [N,3], computes L2 norm per point,
    /// and adds it in-place to the accumulation buffer. Eliminates 4 temporary allocations
    /// (square[N,3] → sum[N] → sqrt[N] → add[N]).
    private lazy var accumGradNormKernel: MLXFast.MLXFastKernel = {
        MLXFast.metalKernel(
            name: "accum_grad_norm",
            inputNames: ["xyz_grad", "accum_in"],
            outputNames: ["accum_out"],
            source: """
                uint tid = thread_position_in_grid.x;
                uint n = accum_in_shape[0];
                if (tid >= n) return;
                
                float gx = xyz_grad[tid * 3 + 0];
                float gy = xyz_grad[tid * 3 + 1];
                float gz = xyz_grad[tid * 3 + 2];
                float norm = sqrt(gx * gx + gy * gy + gz * gz);
                
                accum_out[tid] = accum_in[tid] + norm;
                """
        )
    }()
    
    /// Single-pass classification kernel: determines keep/split/clone/prune per Gaussian.
    /// Actions: 0=keep, 1=split, 2=clone, 3=prune
    /// Output counts: keep=1, split=2, clone=2, prune=0
    private lazy var classifyGaussiansKernel: MLXFast.MLXFastKernel = {
        MLXFast.metalKernel(
            name: "classify_gaussians",
            inputNames: ["grad_accum", "denom_accum", "scales", "opacity",
                          "grad_threshold", "max_scale_thresh", "min_opacity_thresh", "allow_densify"],
            outputNames: ["actions", "output_counts"],
            source: """
                uint tid = thread_position_in_grid.x;
                uint n = grad_accum_shape[0];
                if (tid >= n) return;
                
                float g = grad_accum[tid];
                float d = denom_accum[tid];
                float avg_grad = d > 0.0f ? g / d : 0.0f;
                
                uint scaleStride = uint(scales_shape[1]);
                float s0 = exp(scales[tid * scaleStride + 0]);
                float s1 = exp(scales[tid * scaleStride + 1]);
                float s2 = exp(scales[tid * scaleStride + 2]);
                float max_scale_val = fmax(fmax(s0, s1), s2);
                
                float raw_op = opacity[tid];
                float op_val = 1.0f / (1.0f + exp(-raw_op));
                
                float gthresh = grad_threshold;
                float sthresh = max_scale_thresh;
                float othresh = min_opacity_thresh;
                int densify = allow_densify;
                
                int action;
                int out_count;
                
                if (op_val < othresh) {
                    action = 3; out_count = 0;
                } else if (densify && avg_grad > gthresh) {
                    if (max_scale_val > sthresh) {
                        action = 1; out_count = 2;
                    } else {
                        action = 2; out_count = 2;
                    }
                } else {
                    action = 0; out_count = 1;
                }
                
                actions[tid] = action;
                output_counts[tid] = out_count;
                """
        )
    }()
    
    /// Builds a gather-index map from the per-Gaussian classification.
    /// noise_mode: 0=no modification (keep/clone-original), 1=split-first (+noise, scale-down),
    ///             2=split-second (-noise, scale-down), 3=clone-copy (small noise)
    private lazy var buildDensifyOutputMapKernel: MLXFast.MLXFastKernel = {
        MLXFast.metalKernel(
            name: "build_densify_output_map",
            inputNames: ["actions", "offsets"],
            outputNames: ["gather_indices", "noise_mode"],
            source: """
                uint tid = thread_position_in_grid.x;
                uint n = actions_shape[0];
                if (tid >= n) return;
                
                int action = actions[tid];
                int offset = offsets[tid];
                
                if (action == 0) {
                    gather_indices[offset] = int(tid);
                    noise_mode[offset] = 0;
                } else if (action == 1) {
                    gather_indices[offset]     = int(tid);
                    noise_mode[offset]         = 1;
                    gather_indices[offset + 1] = int(tid);
                    noise_mode[offset + 1]     = 2;
                } else if (action == 2) {
                    gather_indices[offset]     = int(tid);
                    noise_mode[offset]         = 0;
                    gather_indices[offset + 1] = int(tid);
                    noise_mode[offset + 1]     = 3;
                }
                """,
            ensureRowContiguous: false
        )
    }()
    
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
        MLX.Memory.clearCache()
    }
    init(
        model: GaussModel,
        data: TrainData,
        gaussRender: GaussianRenderer,
        iterationCount: Int,
        cacheLimit: Int = GaussianTrainer.defaultCacheLimit(),
        manualClearCache: Bool = GaussianTrainer.needEagerClearCache(),
        enableIntervalProfiling: Bool = false,
        profilingLogInterval: Int = 20,
        profilingTopKSections: Int = 12,
        outputDirectoryURL: URL? = nil,
        saveSnapshotPerIteration: Int = 100
    ) {
        self.model = model
        self.data = data
        self.gaussRender = gaussRender
        self.iterationCount = iterationCount
        self.cacheLimit = cacheLimit
        self.manualClearCache = manualClearCache
        self.enableIntervalProfiling = enableIntervalProfiling
        self.profilingLogInterval = max(1, profilingLogInterval)
        self.profilingTopKSections = max(1, profilingTopKSections)
        self.outputDirectoryURL = outputDirectoryURL
        self.save_snapshot_per_iteration = saveSnapshotPerIteration
        
        // Initialize gradient accumulation arrays
        let numPoints = model._xyz.shape[0]
        self.xyzGradAccumulation = MLXArray.zeros([numPoints])
        self.denomGradAccumulation = 0
    }
    /// Precompute all Camera objects once to avoid ~20 GPU sync points per iteration.
    /// Camera construction calls .item() on H, W, c2w elements (16×), FoV (2×) etc.
    /// These sync points flush the GPU queue, causing stalls after optimizer eval.
    private func precomputeCameras() {
        let numCameras = data.getNumCameras()
        cachedCameras = (0..<numCameras).map { data.getViewPointCamera(index: $0) }
    }

    func fetchTrainData() -> (
        camera: Camera, rgb: MLXArray, depthMask: MLXArray, depth: MLXArray
    ) {
        let numCameras = data.getNumCameras()
        let ind = Int.random(in: 0..<numCameras)
        let rgb = data.rgbArray[ind]
        let depthMask = data.alphaArray[ind] .> 0.5
        let depth = data.depthArray?[ind] ?? MLXArray.zeros(depthMask.shape)
        let camera = cachedCameras.isEmpty
            ? data.getViewPointCamera(index: ind)
            : cachedCameras[ind]
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

    // MARK: - Fused SSIM Metal Kernel (CustomFunction with VJP)

    /// Fused SSIM: 5 grouped convolutions + SSIM formula in a single kernel pass.
    /// Forward outputs ssim_map [H,W,C]; backward kernel computes grad_img1, grad_img2.
    private lazy var ssimCustomFunction: (([MLXArray]) -> [MLXArray])? = {
        guard
            let forwardKernel = try? SlangKernelSpecLoader.loadKernel(named: "ssim_forward_mlx"),
            let backwardKernel = try? SlangKernelSpecLoader.loadKernel(named: "ssim_backward_mlx")
        else {
            Logger.shared.debug("Slang SSIM kernels unavailable. Falling back to MLX SSIM.")
            return nil
        }

        // Gaussian window is generated inside Slang kernels (sigma=1.5).
        let windowSize = 11

        // Saved forward intermediates for backward
        var savedMu1: MLXArray?
        var savedMu2: MLXArray?
        var savedSigma1: MLXArray?
        var savedSigma2: MLXArray?
        var savedSigma12: MLXArray?

        let forward: ([MLXArray]) -> [MLXArray] = { inputs in
            let img1 = inputs[0]  // [H, W, C]
            let img2 = inputs[1]  // [H, W, C]
            let H = img1.shape[0]
            let W = img1.shape[1]
            let C = img1.shape[2]
            let totalPixels = H * W * C
            let params = MLXArray([UInt32(H), UInt32(W), UInt32(C), UInt32(windowSize)])

            let outputs = forwardKernel(
                [img1.reshaped([-1]), img2.reshaped([-1]), params],
                grid: (max(totalPixels, 1), 1, 1),
                threadGroup: (min(256, max(totalPixels, 1)), 1, 1),
                outputShapes: Array(repeating: [totalPixels], count: 6),
                outputDTypes: Array(repeating: img1.dtype, count: 6)
            )
            savedMu1 = outputs[1]
            savedMu2 = outputs[2]
            savedSigma1 = outputs[3]
            savedSigma2 = outputs[4]
            savedSigma12 = outputs[5]
            return [outputs[0].reshaped(img1.shape)]  // ssim_map [H,W,C]
        }

        return CustomFunction {
            Forward(forward)
            VJP { primals, cotangents in
                let img1 = primals[0]  // [H, W, C]
                let img2 = primals[1]  // [H, W, C]
                let H = img1.shape[0]
                let W = img1.shape[1]
                let C = img1.shape[2]
                let totalPixels = H * W * C
                let gradOutput = cotangents[0].reshaped([-1])
                let params = MLXArray([UInt32(H), UInt32(W), UInt32(C), UInt32(windowSize)])

                let grads = backwardKernel(
                    [
                        gradOutput, img1.reshaped([-1]), img2.reshaped([-1]),
                        savedMu1!, savedMu2!, savedSigma1!, savedSigma2!, savedSigma12!,
                        params,
                    ],
                    grid: (max(totalPixels, 1), 1, 1),
                    threadGroup: (min(256, max(totalPixels, 1)), 1, 1),
                    outputShapes: [[totalPixels], [totalPixels]],
                    outputDTypes: [img1.dtype, img2.dtype]
                )
                return [grads[0].reshaped(img1.shape), grads[1].reshaped(img2.shape)]
            }
        }
    }()

    private func buildLossAndGrad(
        gaussRender: GaussianRenderer,
        lambdaDepth: Float,
        lambdaDssim: Float,
        profiler: IntervalProfiler
    ) -> ([MLXArray]) -> ([MLXArray], [MLXArray]) {
        let ssimFn = self.ssimCustomFunction
        let lossFn: ([MLXArray]) -> [MLXArray] = { inputs in
            let _xyz = inputs[TrainStepInputIndex.xyz]
            let _features_dc = inputs[TrainStepInputIndex.featuresDC]
            let _features_rest = inputs[TrainStepInputIndex.featuresRest]
            let _scales = inputs[TrainStepInputIndex.scales]
            let _rotation = inputs[TrainStepInputIndex.rotation]
            let _opacity = inputs[TrainStepInputIndex.opacity]
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

            let means3d = profiler.measure("train.preprocess.xyz") {
                gaussRender.get_xyz_from(_xyz)
            }
            let opacity = profiler.measure("train.preprocess.opacity") {
                gaussRender.get_opacity_from(_opacity)
            }
            let scales = profiler.measure("train.preprocess.scales") {
                gaussRender.get_scales_from(_scales)
            }
            let rotations = profiler.measure("train.preprocess.rotation") {
                gaussRender.get_rotation_from(_rotation)
            }
            let shs = profiler.measure("train.preprocess.features") {
                gaussRender.get_features_from(_features_dc, _features_rest)
            }

            // Forward pass
            let (render, depth, _, _, _) = profiler.measure("train.forward") {
                gaussRender.forwardWithCameraParams(
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
            }

            // Loss computation
            let l1LossValue = profiler.measure("train.loss.l1") {
                l1Loss(render, trainRGB)
            }

            let depthLossValue = profiler.measure("train.loss.depth") {
                let depthDiff = MLX.abs(depth[.ellipsis, 0] - trainDepth)
                let depthMaskFloat = depthMask.asType(.float32)
                let depthWeight = depthMaskFloat.sum()
                let safeDepthWeight = MLX.maximum(depthWeight, MLXArray(1e-6 as Float))
                return (depthDiff * depthMaskFloat).sum() / safeDepthWeight
            }

            let ssimLossValue = profiler.measure("train.loss.ssim") {
                if let fn = ssimFn {
                    let ssimMap = fn([render, trainRGB])[0]
                    return 1.0 - ssimMap.mean()
                } else {
                    return 1.0 - ssim(img1: render[.newAxis], img2: trainRGB[.newAxis])
                }
            }

            let totalLoss = profiler.measure("train.loss.total") {
                (1.0 - lambdaDssim) * l1LossValue
                    + lambdaDssim * ssimLossValue
                    + lambdaDepth * depthLossValue
            }

            return [totalLoss, render]
        }

        return MLX.valueAndGrad(
            lossFn,
            argumentNumbers: Array(0..<TrainStepInputIndex.parameterCount)
        )
    }
    func addGradientAccumulation(xyzGrad: MLXArray) {
        let numPoints = xyzGrad.shape[0]
        
        if xyzGradAccumulation.shape[0] != numPoints {
            xyzGradAccumulation = MLXArray.zeros([numPoints])
            denomGradAccumulation = 0
        }
        
        // Fused kernel: L2 norm + accumulate in a single pass (no temporaries)
        let result = accumGradNormKernel(
            [xyzGrad.reshaped([-1]), xyzGradAccumulation],
            grid: (numPoints, 1, 1),
            threadGroup: (min(256, numPoints), 1, 1),
            outputShapes: [[numPoints]],
            outputDTypes: [xyzGradAccumulation.dtype]
        )
        xyzGradAccumulation = result[0]
        denomGradAccumulation += 1
    }
    
    func resetGradientAccumulation() {
        let numPoints = model._xyz.shape[0]
        xyzGradAccumulation = MLXArray.zeros([numPoints])
        denomGradAccumulation = 0
    }
    
    /// Custom-kernel implementation of Gaussian densification (split/clone) and pruning.
    ///
    /// The work is broken into 3 GPU phases followed by MLX gather + arithmetic:
    ///  1. **classify_gaussians** – a single Metal threadgroup pass classifies every
    ///     Gaussian as keep (0), split (1), clone (2), or prune (3) and emits an
    ///     output-count per element (1, 2, 2, 0 respectively).
    ///  2. **exclusive prefix-sum** (MLX `cumsum`) – turns per-element counts into
    ///     scatter offsets for the output arrays.
    ///  3. **build_densify_output_map** – writes a gather-index + noise-mode per
    ///     output slot so that the final arrays can be assembled with a single
    ///     `MLXArray` gather.
    ///
    /// Compared to the previous implementation this avoids:
    ///  - multiple intermediate mask/index arrays,
    ///  - sequential split → clone → prune passes that invalidate index bookkeeping,
    ///  - CPU round-trips for each `conditionToIndices` / `sum` check.
    func split_and_prune(params: [MLXArray], states: [TupleState], iteration: Int) {
        guard iteration >= densifyFromIter && iteration <= densifyUntilIter else {
            return
        }
        
        let _xyz = params[0]
        let _features_dc = params[1]
        let _features_rest = params[2]
        let _scales = params[3]
        let _rotation = params[4]
        let _opacity = params[5]
        
        let numPoints = _xyz.shape[0]
        guard numPoints > 0 else {
            resetGradientAccumulation()
            return
        }
        
        // When over the Gaussian budget we still prune but disable densification.
        let allowDensify = numPoints < maxGaussians
        if !allowDensify {
            Logger.shared.info("Reached maximum Gaussians limit: \(maxGaussians). Skipping densification.")
        }
        
        // Ensure gradient accumulation arrays match current point count
        if xyzGradAccumulation.shape[0] != numPoints {
            resetGradientAccumulation()
        }
        
        // ---------- Phase 1: classify each Gaussian on GPU ----------
        // denomGradAccumulation is a scalar counter; broadcast to [numPoints] for the kernel.
        let denomArray = MLXArray.full([numPoints], values: MLXArray(Float(denomGradAccumulation)))
        let classifyOutputs = classifyGaussiansKernel(
            [xyzGradAccumulation, denomArray,
             _scales, _opacity.reshaped([-1]),
             MLXArray(gradientThreshold),
             MLXArray(maxScale),
             MLXArray(minOpacity),
             MLXArray(Int32(allowDensify ? 1 : 0))],
            grid: (numPoints, 1, 1),
            threadGroup: (min(256, numPoints), 1, 1),
            outputShapes: [[numPoints], [numPoints]],
            outputDTypes: [.int32, .int32]
        )
        let actions = classifyOutputs[0]
        let outputCounts = classifyOutputs[1]
        
        // ---------- Phase 2: exclusive prefix sum for scatter offsets ----------
        let inclusiveSum = outputCounts.cumsum(axis: 0).asType(.int32)
        let offsets = inclusiveSum - outputCounts
        // .item() triggers evaluation of the full classify + cumsum chain.
        let totalOutput = inclusiveSum[numPoints - 1].item(Int.self)
        
        // Fast path: nothing changed.
        if totalOutput == numPoints {
            // Could still be a no-op (all keep). Check action counts to be sure.
            let numPrune = MLX.sum((actions .== 3).asType(.int32)).item(Int.self)
            if numPrune == 0 {
                Logger.shared.debug("split_and_prune: no changes at iteration \(iteration)")
                resetGradientAccumulation()
                return
            }
        }
        
        guard totalOutput > 0 else {
            Logger.shared.info("split_and_prune: all Gaussians would be pruned – skipping")
            resetGradientAccumulation()
            return
        }
        
        // Action counts (cheap – actions are already evaluated)
        let numSplit = MLX.sum((actions .== 1).asType(.int32)).item(Int.self)
        let numClone = MLX.sum((actions .== 2).asType(.int32)).item(Int.self)
        let numPruneTotal = MLX.sum((actions .== 3).asType(.int32)).item(Int.self)
        let numKeep = numPoints - numSplit - numClone - numPruneTotal
        
        if numSplit == 0 && numClone == 0 && numPruneTotal == 0 {
            Logger.shared.debug("split_and_prune: no changes at iteration \(iteration)")
            resetGradientAccumulation()
            return
        }
        
        Logger.shared.info(
            "split_and_prune iter=\(iteration): keep=\(numKeep) split=\(numSplit) clone=\(numClone) prune=\(numPruneTotal) → total=\(totalOutput)"
        )
        
        // ---------- Phase 3: build gather map ----------
        let mapOutputs = buildDensifyOutputMapKernel(
            [actions, offsets],
            grid: (numPoints, 1, 1),
            threadGroup: (min(256, numPoints), 1, 1),
            outputShapes: [[totalOutput], [totalOutput]],
            outputDTypes: [.int32, .int32],
            initValue: 0
        )
        let gatherIndices = mapOutputs[0]
        let noiseMode = mapOutputs[1]
        eval(gatherIndices, noiseMode)
        
        // ---------- Phase 4: gather all parameter arrays ----------
        var newXYZ = _xyz[gatherIndices]
        let newFeaturesDC = _features_dc[gatherIndices]
        let newFeaturesRest = _features_rest[gatherIndices]
        var newScales = _scales[gatherIndices]
        let newRotation = _rotation[gatherIndices]
        let newOpacity = _opacity[gatherIndices]
        
        // ---------- Phase 5: apply per-slot modifications ----------
        if numSplit > 0 || numClone > 0 {
            // 5a. Scale reduction for split Gaussians (÷ 1.6 ⇔ −log(1.6) in log-space)
            let isSplit = ((noiseMode .== 1) | (noiseMode .== 2)).asType(.float32)
            let scaleReduction = isSplit.expandedDimensions(axis: 1) * Float(-log(1.6))
            newScales = newScales + scaleReduction
            
            // 5b. Position noise
            let baseNoise = MLXRandom.normal([totalOutput, 3])
            
            // Split noise – geometry-aware: ±mean(exp(source_scale)) * 0.1 * N(0,1)
            // Uses the *original* (pre-reduction) source scales for the noise magnitude.
            let sourceScales = MLX.exp(_scales[gatherIndices])  // pre-reduction
            let meanSourceScale = MLX.mean(sourceScales, axes: [1], keepDims: true)
            let isSplitFirst  = (noiseMode .== 1).asType(.float32).expandedDimensions(axis: 1)
            let isSplitSecond = (noiseMode .== 2).asType(.float32).expandedDimensions(axis: 1)
            let splitSign = isSplitFirst - isSplitSecond  // +1 or −1
            let splitNoise = splitSign * meanSourceScale * 0.1 * baseNoise
            
            // Clone noise – small fixed offset (0.01 * N(0,1)) for clone copies only.
            let isCloneCopy = (noiseMode .== 3).asType(.float32).expandedDimensions(axis: 1)
            let cloneNoise = isCloneCopy * 0.01 * baseNoise
            
            newXYZ = newXYZ + splitNoise + cloneNoise
        }
        
        // ---------- Phase 6: commit ----------
        model._xyz = newXYZ
        model._features_dc = newFeaturesDC
        model._features_rest = newFeaturesRest
        model._scales = newScales
        model._rotation = newRotation
        model._opacity = newOpacity
        
        resetGradientAccumulation()
    }    
    func save_snapshot(iteration: Int, params: [MLXArray]) {
        // TODO: Implement snapshot saving logic if needed
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
        let lambda_depth = self.lambda_depth
        let lambda_dssim = self.lambda_dssim
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

        // Precompute all Camera objects once to eliminate GPU sync stalls
        // in fetchTrainData (20+ .item() calls per Camera construction).
        precomputeCameras()

        var fps: Float? = nil
        let fpsInterval = 10
        var fpsWindowStartTime = Date()
        for iteration in 0..<iterationCount {
            if forceStop {
                break
            }
            let shouldProfileIteration =
                enableIntervalProfiling
                && (iteration % profilingLogInterval == 0 || iteration == iterationCount - 1)
            let profiler = IntervalProfiler(enabled: shouldProfileIteration)
            gaussRender.profiler = shouldProfileIteration ? profiler : nil

            let iterationStartNs = DispatchTime.now().uptimeNanoseconds
            Logger.shared.debug("Iteration \(iteration)")
            let (trainCamera, trainRGB, trainDepthMask, trainDepth) =
                profiler.measure("train.fetchTrainData") {
                    fetchTrainData()
                }
            let trainInputs = profiler.measure("train.makeTrainStepInputs") {
                makeTrainStepInputs(
                    params: params,
                    camera: trainCamera,
                    trainRGB: trainRGB,
                    depthMask: trainDepthMask,
                    trainDepth: trainDepth
                )
            }
            let trainWithGrad = profiler.measure("train.valueAndGrad.build") {
                buildLossAndGrad(
                    gaussRender: gaussRender,
                    lambdaDepth: effectiveLambdaDepth,
                    lambdaDssim: lambda_dssim,
                    profiler: profiler
                )
            }
            let (loss, grads) = profiler.measure("train.valueAndGrad.execute") {
                trainWithGrad(trainInputs)
            }

            // Accumulate gradients for densification (lazy — no eval here)
            profiler.measure("train.addGradientAccumulation") {
                addGradientAccumulation(xyzGrad: grads[0])
            }
            
            // --- Periodic CPU sync: FPS + loss + pushLoss + early stopping ---
            // loss[0].item() forces a GPU sync, so batch it with FPS calculation
            // every fpsInterval iterations to avoid per-iteration stalls.
            let isFpsIteration = iteration % fpsInterval == fpsInterval - 1
            let isImageIteration = iteration % 20 == 0
            let needsLossValue = isFpsIteration || isImageIteration

            var lossValue: Float = 0
            if needsLossValue {
                lossValue = profiler.measure("train.loss.toFloat") {
                    loss[0].item(Float.self)
                }
            }

            if isFpsIteration {
                let now = Date()
                let windowDuration = now.timeIntervalSince(fpsWindowStartTime)
                if windowDuration > 0 {
                    fps = Float(Double(fpsInterval) / windowDuration)
                }
                fpsWindowStartTime = now

                profiler.measure("train.delegate.pushLoss") {
                    delegate?.pushLoss(
                        loss: lossValue,
                        iteration: iteration,
                        fps: fps,
                        timestamp: Date())
                }
            }

            if isImageIteration {
                let image = loss[1]
                profiler.measure("train.evalPreviewImage") {
                    eval(image)
                }
                profiler.measure("train.delegate.pushImageData") {
                    delegate?.pushImageData(
                        render: image,
                        truth: trainRGB,
                        loss: lossValue,
                        iteration: iteration, timestamp: Date()
                    )
                }
            }
            if needsLossValue && lossValue < earlyStoppingThreshold {
                if shouldProfileIteration {
                    let iterationNs = DispatchTime.now().uptimeNanoseconds - iterationStartNs
                    Logger.shared.info(
                        profiler.makeReport(
                            iteration: iteration,
                            iterationNanoseconds: iterationNs,
                            topK: profilingTopKSections,
                            minMilliseconds: 0.01
                        )
                    )
                }
                Logger.shared.info("Early stopping")
                return
            }
            let lrs = profiler.measure("train.getLearningRates") {
                model.getLearningRates(
                    current: iteration,
                    total: iterationCount
                )
            }
            for i in 0..<params.count {
                Logger.shared.debug("Updating parameter \(i) start")
                optimizer.learningRate = lrs[i]
                let (newParam, newState) = profiler.measure("train.optimizer.applySingle") {
                    optimizer.applySingle(
                        gradient: grads[i],
                        parameter: params[i],
                        state: states[i]
                    )
                }
                params[i] = newParam
                states[i] = newState
                Logger.shared.debug("Updating parameter \(i) end")
            }
            // Single batched eval for all updated params (was 6× individual evals)
            profiler.measure("train.optimizer.evalAllParams") {
                eval(params)
            }
            profiler.measure("train.optimizer.evalOptimizer") {
                eval(optimizer)
            }
            profiler.measure("train.cacheControl") {
                if manualClearCache && MLX.Memory.snapshot().cacheMemory > cacheLimit {
                    clearCacheIfNeeded()
                }
            }
            if iteration % self.save_snapshot_per_iteration == 0 {
                profiler.measure("train.saveSnapshot") {
                    self.save_snapshot(iteration: iteration, params: params)
                    clearCacheIfNeeded()
                }
            }
            if iteration % self.split_and_prune_per_iteration == 0 {
                profiler.measure("train.splitAndPrune") {
                    self.split_and_prune(params: params, states: states, iteration: iteration)
                }
                profiler.measure("train.rebuildParamsAfterSplit") {
                    // Update params after split_and_prune since number of Gaussians may have changed
                    params = model.getParams()
                    // Reinitialize optimizer states for new parameters
                    // This is necessary because split/clone operations change the parameter count
                    states = params.map {
                        optimizer.newState(parameter: $0)
                    }
                }
                profiler.measure("train.clearCacheAfterSplit") {
                    clearCacheIfNeeded()
                }
            }
            if shouldProfileIteration {
                let iterationNs = DispatchTime.now().uptimeNanoseconds - iterationStartNs
                Logger.shared.info(
                    profiler.makeReport(
                        iteration: iteration,
                        iterationNanoseconds: iterationNs,
                        topK: profilingTopKSections,
                        minMilliseconds: 0.01
                    )
                )
            }
            gaussRender.profiler = nil
        }
        clearCacheIfNeeded()
    }
    static func createModel(
        sh_degree: Int,
        pointCloud: PointCloud,
        sampleCount: Int = 1 << 14,
        manualClearCache: Bool = GaussianTrainer.needEagerClearCache()
    ) -> GaussModel {
        Logger.shared.debug("Getting point clouds...")
        Logger.shared.debug("Random sampling...")
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
