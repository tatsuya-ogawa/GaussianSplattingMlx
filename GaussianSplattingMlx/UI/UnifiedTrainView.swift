//
//  UnifiedTrainView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/08/16.
//

import MLX
import SwiftUI

// MARK: - Unified Training View Model

class UnifiedTrainViewModel: ObservableObject {
    @Published var trainer: UnifiedTrainer?
    @Published var isTraining: Bool = false
    @Published var selectedMethod: SplattingMethod = .gaussian
    @Published var currentIteration: Int = 0
    @Published var currentLoss: Float = 0.0
    @Published var currentPSNR: Float = 0.0
    @Published var numPrimitives: Int = 0
    
    private var trainingTimer: Timer?
    
    func startTraining(with dataLoader: DataLoaderProtocol) {
        isTraining = true
        currentIteration = 0
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try self.doTrain(with: dataLoader)
            } catch {
                Logger.shared.error("Training failed", error: error)
                DispatchQueue.main.async {
                    self.isTraining = false
                }
            }
        }
    }
    
    func stopTraining() {
        trainer?.stopTraining()
        trainingTimer?.invalidate()
        trainingTimer = nil
        isTraining = false
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

        // Create unified trainer
        let width = data.Ws[0].item(Int.self)
        let height = data.Hs[0].item(Int.self)
        
        DispatchQueue.main.async {
            self.trainer = UnifiedTrainer(width: width, height: height, tileSize: TILE_SIZE)
            self.trainer?.currentMethod = self.selectedMethod
        }
        
        guard let trainer = trainer else {
            throw TrainingError.modelNotInitialized
        }

        // Initialize the selected model
        let points = pointCloud.coords
        let colors = pointCloud.select_channels(channel_names: ["R", "G", "B"]) / 255.0
        
        switch selectedMethod {
        case .gaussian:
            trainer.initializeGaussianModel(from: (points, colors))
        case .triangle:
            trainer.initializeTriangleModel(from: (points, colors))
        }

        // Convert data to SceneDataset format for training
        let sceneDataset = SceneDataset(data: data)
        
        // Start training with progress updates
        trainer.startTraining(scene: sceneDataset)
        
        // Start progress monitoring
        startProgressMonitoring()
        
        MLX.GPU.clearCache()
    }
    
    private func startProgressMonitoring() {
        DispatchQueue.main.async {
            self.trainingTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
                guard let trainer = self.trainer else { return }
                
                // Update training metrics from trainer
                self.currentIteration = trainer.currentIteration
                self.currentLoss = trainer.currentLoss
                self.currentPSNR = trainer.currentPSNR
                self.numPrimitives = trainer.numPrimitives
                
                // Update loss chart
                LossChartData.shared.push(
                    loss: self.currentLoss,
                    iteration: self.currentIteration,
                    timestamp: Date()
                )
                
                // Check if training is complete
                if !trainer.isTraining {
                    self.stopTraining()
                }
            }
        }
    }
    
    func switchMethod(_ method: SplattingMethod) {
        guard !isTraining else { return }
        selectedMethod = method
        trainer?.switchToMethod(method)
    }
}

// MARK: - Method Selection Components

struct MethodSelectionRow: View {
    let method: SplattingMethod
    let selectedMethod: SplattingMethod
    let isTraining: Bool
    let onSelect: () -> Void
    
    var body: some View {
        Button(action: {
            if !isTraining {
                onSelect()
            }
        }) {
            HStack {
                Image(systemName: selectedMethod == method ? "largecircle.fill.circle" : "circle")
                    .foregroundColor(selectedMethod == method ? .blue : .gray)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(method.displayName)
                        .font(.system(size: 14, weight: .medium))
                    Text(getMethodDescription(method))
                        .font(.system(size: 12))
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.leading)
                }
                Spacer()
            }
            .padding(.vertical, 8)
            .padding(.horizontal, 12)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(selectedMethod == method ? Color.blue.opacity(0.1) : Color.clear)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(selectedMethod == method ? Color.blue : Color.gray.opacity(0.3), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
        .disabled(isTraining)
    }
    
    private func getMethodDescription(_ method: SplattingMethod) -> String {
        switch method {
        case .gaussian:
            return "3D Gaussian Splatting with point-based primitives"
        case .triangle:
            return "Triangle Splatting with triangle-based primitives"
        }
    }
}

struct TrainingMethodSelectionView: View {
    @Binding var selectedMethod: SplattingMethod
    let isTraining: Bool
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Training Method")
                .font(.headline)
            
            ForEach(SplattingMethod.allCases, id: \.self) { method in
                MethodSelectionRow(
                    method: method,
                    selectedMethod: selectedMethod,
                    isTraining: isTraining,
                    onSelect: { selectedMethod = method }
                )
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.gray.opacity(0.1))
        )
    }
}

// MARK: - Training Progress Components

struct UnifiedTrainingProgressView: View {
    @ObservedObject var viewModel: UnifiedTrainViewModel
    
    var body: some View {
        VStack(spacing: 16) {
            // Method indicator
            HStack {
                Text("Training Method:")
                    .font(.headline)
                Text(viewModel.selectedMethod.displayName)
                    .font(.headline)
                    .foregroundColor(.blue)
                Spacer()
            }
            
            // Progress metrics
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                UnifiedMetricCard(title: "Iteration", value: "\(viewModel.currentIteration)")
                UnifiedMetricCard(title: "Loss", value: String(format: "%.4f", viewModel.currentLoss))
                UnifiedMetricCard(title: "PSNR", value: String(format: "%.2f dB", viewModel.currentPSNR))
                UnifiedMetricCard(title: "Primitives", value: "\(viewModel.numPrimitives)")
            }
            
            // Progress bar
            ProgressView(value: viewModel.trainer?.getTrainingProgress() ?? 0.0)
                .progressViewStyle(LinearProgressViewStyle())
            
            // Training status view
            TrainStatusView()
            
            // Stop button
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
        .padding()
    }
}

struct UnifiedMetricCard: View {
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

// MARK: - Main Unified Train View

struct UnifiedTrainView: View {
    @StateObject private var viewModel = UnifiedTrainViewModel()
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
            Logger.shared.error("Failed to start training", error: error)
        }
    }
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            Text("Unified Neural Rendering Training")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Train neural rendering using either Gaussian or Triangle splatting methods")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            if viewModel.isTraining {
                UnifiedTrainingProgressView(viewModel: viewModel)
            } else {
                // Configuration View
                VStack(spacing: 20) {
                    // Dataset selection
                    SelectDataSetView(selected: $selected)
                    
                    // Method selection
                    TrainingMethodSelectionView(
                        selectedMethod: $viewModel.selectedMethod,
                        isTraining: viewModel.isTraining
                    )
                    
                    // Start button
                    Button(action: startTrain) {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Start \(viewModel.selectedMethod.displayName) Training")
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
        .navigationTitle("Unified Training")
    }
}

#Preview {
    UnifiedTrainView()
}