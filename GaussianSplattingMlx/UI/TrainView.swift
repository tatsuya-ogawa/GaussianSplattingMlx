//
//  TrainView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/07.
//

import MLX
import SwiftUI

struct TrainSnapshot: Identifiable ,Hashable{
    let id = UUID()
    let timestamp: Date
    let url: URL
    let iteration: Int
}
class TrainOutputAsset: ObservableObject {
    static let shared = TrainOutputAsset()
    @Published var snapshots: [TrainSnapshot] = []
    func getOutputDirectory() -> URL {
        return FileManager.default.temporaryDirectory.appendingPathComponent("outputs")
    }
    func pushSnapshot(url: URL, iteration: Int, timestamp: Date) {
        DispatchQueue.main.async{
            self.snapshots.append(TrainSnapshot(timestamp: timestamp, url: url, iteration: iteration))
        }
    }
    func loadSnapshotsFromDirectory() {
        let dir = getOutputDirectory()
        let fm = FileManager.default
        guard
            let files = try? fm.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles])
        else {
            return
        }

        let snapshots = files.compactMap { url -> TrainSnapshot? in
            let fname = url.lastPathComponent
            guard let iterStr = fname.split(separator: "_").last?.split(separator: ".").first,
                let iter = Int(iterStr)
            else { return nil }

            let attrs = try? url.resourceValues(forKeys: [.contentModificationDateKey])
            let timestamp = attrs?.contentModificationDate ?? Date()

            return TrainSnapshot(timestamp: timestamp, url: url, iteration: iter)
        }
        .sorted(by: { $0.iteration < $1.iteration })

        DispatchQueue.main.async {
            self.snapshots = snapshots
        }
    }
}

class TrainViewModel: ObservableObject {
    var trainer: GaussianTrainer?
    @Published var unifiedTrainer = UnifiedTrainer()
    @Published var configViewModel = TrainingConfigViewModel()
}
extension TrainViewModel: GaussianTrainerDelegate {
    func pushSnapshot(url: URL, iteration: Int, timestamp: Date) {
        TrainOutputAsset.shared.pushSnapshot(url: url, iteration: iteration, timestamp: timestamp)
    }

    func pushLoss(loss: Float, iteration: Int?, timestamp: Date) {
        LossChartData.shared.push(loss: loss, iteration: iteration, timestamp: timestamp)
    }

    func pushImageData(
        render: MLX.MLXArray, truth: MLX.MLXArray, loss: Float, iteration: Int, timestamp: Date
    ) {
        TrainStatusData.shared.pushImageData(
            render: render, truth: truth, loss: loss, iteration: iteration, timestamp: timestamp)
    }
}
struct TrainView: View {
    @StateObject var viewModel = TrainViewModel()

    func getDataLoader() throws -> DataLoaderProtocol {
        switch selected.format {
        case .colmap:
            guard let url = selected.url else {
                throw NSError(domain: "No zip", code: 1)
            }
            return UserColmapDataLoader(zipUrl: url)
        case .nerfstudio:
            guard let url = selected.url else {
                throw NSError(domain: "No zip", code: 2)
            }
            return UserNerfStudioDataLoader(zipUrl: url)
        case .demo(let kind):
            switch kind {

            case .chair:
                return BlenderDemoDataLoader()
            case .lego:
                return DemoColmapDataLoader()
            }
        }
    }
    func doTrain() throws {
        let dataLoader = try getDataLoader()
        let whiteBackground = false
        let (data, pointCloud, TILE_SIZE) = try dataLoader.load(
            resizeFactor: 0.5,
            whiteBackground: whiteBackground
        )
        pointCloud.centering(data: data)

        let cacheLimit = 2 * 1024 * 1024 * 1024
        MLX.GPU.set(cacheLimit: cacheLimit)
        MLX.GPU.set(memoryLimit: cacheLimit * 2)
        let sh_degree = 4
        let gaussModel = GaussianTrainer.createModel(
            sh_degree: sh_degree,
            pointCloud: pointCloud,
            sampleCount: 1 << 14
        )

        viewModel.trainer = GaussianTrainer(
            model: gaussModel,
            data: data,
            gaussRender: GaussianRenderer(
                active_sh_degree: sh_degree,
                W: data.Ws[0].item(Int.self),
                H: data.Hs[0].item(Int.self),
                TILE_SIZE: TILE_SIZE,
                whiteBackground: whiteBackground
            ),
            iterationCount: 30000,
            cacheLimit: cacheLimit,
            outputDirectoryURL: TrainOutputAsset.shared.getOutputDirectory()
        )
        viewModel.trainer?.delegate = viewModel
        viewModel.trainer?.startTrain()
        MLX.GPU.clearCache()
        isTraining = false
    }
    @State var isTraining: Bool = false
    @State var selected: SelectedDataSet = SelectedDataSet(
        format: .colmap,
        url: nil
    )
    func startTrain() {
        isTraining = true
        DispatchQueue.global().async {
            Logger.shared.debug("Start Train")
            do {
                try doTrain()
            } catch {
                Logger.shared.error(error: error)
            }
        }
    }
    func stopTrain() {
        viewModel.trainer?.stopTrain()
    }
    
    func startUnifiedTrain() {
        isTraining = true
        DispatchQueue.global().async {
            Logger.shared.debug("Start Unified Train")
            do {
                try doUnifiedTrain()
            } catch {
                Logger.shared.error(error: error)
                DispatchQueue.main.async {
                    self.isTraining = false
                }
            }
        }
    }
    
    func doUnifiedTrain() throws {
        let dataLoader = try getDataLoader()
        let whiteBackground = false
        let (data, pointCloud, TILE_SIZE) = try dataLoader.load(
            resizeFactor: 0.5,
            whiteBackground: whiteBackground
        )
        pointCloud.centering(data: data)
        
        let cacheLimit = 2 * 1024 * 1024 * 1024
        MLX.GPU.set(cacheLimit: cacheLimit)
        MLX.GPU.set(memoryLimit: cacheLimit * 2)
        
        // Initialize the unified trainer based on selected method
        switch viewModel.unifiedTrainer.currentMethod {
        case .gaussian:
            viewModel.unifiedTrainer.initializeGaussianModel(from: (pointCloud.coords, pointCloud.select_channels(channel_names:["R","G","B"])))
        case .triangle:
            viewModel.unifiedTrainer.initializeTriangleModel(from: (pointCloud.coords, pointCloud.select_channels(channel_names:["R","G","B"])))
        }
        
        // Create scene dataset
        let scene = SceneDataset(data: data)
        
        // Start training
        viewModel.unifiedTrainer.startTraining(scene: scene)
        
        MLX.GPU.clearCache()
        DispatchQueue.main.async {
            self.isTraining = false
        }
    }
    var body: some View {
        VStack(spacing: 20) {
            if viewModel.unifiedTrainer.isTraining {
                // Training in progress
                VStack(spacing: 15) {
                    Text("Training \(viewModel.unifiedTrainer.currentMethod.displayName) Splatting")
                        .font(.headline)
                    
                    TrainStatusView()
                    
                    // Training metrics
                    HStack {
                        VStack(alignment: .leading) {
                            Text("Iteration: \(viewModel.unifiedTrainer.currentIteration)")
                            Text("Loss: \(String(format: "%.6f", viewModel.unifiedTrainer.currentLoss))")
                            Text("PSNR: \(String(format: "%.2f", viewModel.unifiedTrainer.currentPSNR))")
                        }
                        Spacer()
                        VStack(alignment: .trailing) {
                            Text("Primitives: \(viewModel.unifiedTrainer.numPrimitives)")
                            Text("Progress: \(String(format: "%.1f%%", viewModel.unifiedTrainer.getTrainingProgress() * 100))")
                        }
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    
                    Button(action: {
                        viewModel.unifiedTrainer.stopTraining()
                        stopTrain()
                    }) {
                        Text("Stop Training")
                    }
                    .buttonStyle(.borderedProminent)
                }
            } else {
                // Training setup
                VStack(spacing: 15) {
                    Text("Neural Splatting Training")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    // Method selection
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Splatting Method:")
                            .font(.headline)
                        
                        Picker("Method", selection: $viewModel.unifiedTrainer.currentMethod) {
                            ForEach(SplattingMethod.allCases, id: \.self) { method in
                                Text(method.displayName).tag(method)
                            }
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        
                        Text(viewModel.unifiedTrainer.getMethodInfo())
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(8)
                    
                    // Dataset selection
                    SelectDataSetView(selected: $selected)
                    
                    // Training configuration
                    DisclosureGroup("Training Configuration") {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Iterations:")
                                Spacer()
                                TextField("Iterations", value: $viewModel.configViewModel.iterations, format: .number)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .frame(width: 100)
                            }
                            
                            HStack {
                                Text("Learning Rate:")
                                Spacer()
                                TextField("LR", value: $viewModel.configViewModel.learningRate, format: .number)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .frame(width: 100)
                            }
                            
                            if viewModel.unifiedTrainer.currentMethod == .triangle {
                                HStack {
                                    Text("Smoothness LR:")
                                    Spacer()
                                    TextField("Smoothness LR", value: $viewModel.configViewModel.triangleSmoothnessLR, format: .number)
                                        .textFieldStyle(RoundedBorderTextFieldStyle())
                                        .frame(width: 100)
                                }
                                
                                HStack {
                                    Text("Regularization:")
                                    Spacer()
                                    TextField("Reg", value: $viewModel.configViewModel.triangleRegularizationWeight, format: .number)
                                        .textFieldStyle(RoundedBorderTextFieldStyle())
                                        .frame(width: 100)
                                }
                            }
                        }
                        .padding()
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    
                    // Start training button
                    Button(action: startUnifiedTrain) {
                        Text("Start \(viewModel.unifiedTrainer.currentMethod.displayName) Training")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!selected.isValid)
                }
            }
        }
        .padding()
    }
}
extension SelectedDataSet {
    var isValid: Bool {
        switch format {
        case .colmap, .nerfstudio:
            return url != nil
        case .demo:
            return true
        }
    }
}
