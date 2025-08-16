//
//  TriangleTrainView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/08/16.
//

import MLX
import SwiftUI

class TriangleTrainViewModel: ObservableObject {
    var trainer: TriangleTrainer?
    @Published var isTraining: Bool = false
    @Published var currentIteration: Int = 0
    @Published var currentLoss: Float = 0.0
    @Published var currentPSNR: Float = 0.0
    @Published var numTriangles: Int = 0
    
    private var trainingTimer: Timer?
    
    func startTraining(with dataLoader: DataLoaderProtocol) {
        isTraining = true
        currentIteration = 0
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.doTrain(with: dataLoader)
            } catch {
                Logger.shared.error("Triangle training failed", error: error)
                DispatchQueue.main.async {
                    self.isTraining = false
                }
            }
        }
    }
    
    func stopTraining() {
        trainer?.stopTrain()
        trainingTimer?.invalidate()
        trainingTimer = nil
        DispatchQueue.main.async {
            self.isTraining = false
        }
    }
    
    private func doTrain(with dataLoader: DataLoaderProtocol) throws {
        let whiteBackground = false
        let (data, pointCloud, TILE_SIZE) = try dataLoader.load(
            resizeFactor: 0.5,
            whiteBackground: whiteBackground
        )
        pointCloud.centering(data: data)

        let cacheLimit = 2 * 1024 * 1024 * 1024
        MLX.GPU.set(cacheLimit: cacheLimit)
        MLX.GPU.set(memoryLimit: cacheLimit * 2)

        // Create triangle model from point cloud
        let points = pointCloud.coords
        let colors = pointCloud.select_channels(channel_names: ["R", "G", "B"]) / 255.0
        let triangleModel = TriangleModel(fromPointCloud: points, colors: colors, shDegree: 3)
        
        // Create triangle renderer
        let width = data.Ws[0].item(Int.self)
        let height = data.Hs[0].item(Int.self)
        let triangleRenderer = TriangleRenderer(
            active_sh_degree: 3,
            W: width,
            H: height,
            TILE_SIZE: TILE_SIZE,
            whiteBackground: whiteBackground
        )
        
        // Create triangle trainer
        DispatchQueue.main.async {
            self.trainer = TriangleTrainer(triangleModel: triangleModel, renderer: triangleRenderer)
            self.numTriangles = triangleModel.activeTriangles
        }
        
        guard let trainer = trainer else {
            throw NSError(domain: "Failed to create triangle trainer", code: 1)
        }

        // Convert data to SceneDataset for training
        let sceneDataset = SceneDataset(data: data)
        
        // Start progress monitoring
        startProgressMonitoring()
        
        // Start training
        trainer.train(scene: sceneDataset) { iteration, metrics in
            DispatchQueue.main.async {
                self.currentIteration = iteration
                self.currentLoss = metrics["loss"] ?? 0.0
                self.currentPSNR = metrics["psnr"] ?? 0.0
                self.numTriangles = Int(metrics["num_triangles"] ?? 0)
                
                // Update loss chart
                LossChartData.shared.push(
                    loss: self.currentLoss,
                    iteration: iteration,
                    timestamp: Date()
                )
            }
        }
        
        MLX.GPU.clearCache()
        DispatchQueue.main.async {
            self.isTraining = false
        }
    }
    
    private func startProgressMonitoring() {
        DispatchQueue.main.async {
            self.trainingTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                // Check if training is still active
                if let trainer = self.trainer, !trainer.isTraining {
                    self.stopTraining()
                }
            }
        }
    }
}

// Simple protocol to make trainer compatible
extension TriangleTrainer {
    var isTraining: Bool {
        return currentIteration < config.iterations
    }
    
    func stopTrain() {
        // Triangle trainer doesn't have built-in stop mechanism
        // This would need to be implemented in the trainer
        Logger.shared.info("Triangle training stop requested")
    }
}

struct TriangleTrainView: View {
    @StateObject private var viewModel = TriangleTrainViewModel()
    @State private var selected: SelectedDataSet = SelectedDataSet(format: .colmap, url: nil)
    
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
    
    func startTrain() {
        do {
            let dataLoader = try getDataLoader()
            viewModel.startTraining(with: dataLoader)
        } catch {
            Logger.shared.error("Failed to start triangle training", error: error)
        }
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            Text("Triangle Splatting Training")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Train neural rendering using triangle-based primitives with soft window functions")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            if viewModel.isTraining {
                // Training Progress View
                VStack(spacing: 16) {
                    // Progress metrics
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                        MetricCard(title: "Iteration", value: "\(viewModel.currentIteration)")
                        MetricCard(title: "Loss", value: String(format: "%.4f", viewModel.currentLoss))
                        MetricCard(title: "PSNR", value: String(format: "%.2f dB", viewModel.currentPSNR))
                        MetricCard(title: "Triangles", value: "\(viewModel.numTriangles)")
                    }
                    
                    // Progress bar
                    ProgressView(value: Float(viewModel.currentIteration) / 30000.0)
                        .progressViewStyle(LinearProgressViewStyle())
                    
                    TrainStatusView()
                    
                    Button(action: {
                        viewModel.stopTraining()
                    }) {
                        HStack {
                            Image(systemName: "stop.fill")
                            Text("Stop Training")
                        }
                        .padding()
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                }
            } else {
                // Configuration View
                VStack(spacing: 20) {
                    SelectDataSetView(selected: $selected)
                    
                    Button(action: startTrain) {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Start Triangle Training")
                        }
                        .padding()
                        .background(selected.isValid ? Color.blue : Color.gray)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                    }
                    .disabled(!selected.isValid)
                }
            }
        }
        .padding()
        .navigationTitle("Triangle Splatting")
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.title2)
                .fontWeight(.semibold)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.gray.opacity(0.1))
        )
    }
}

// Extension already defined in TrainView.swift