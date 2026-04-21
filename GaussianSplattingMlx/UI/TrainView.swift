//
//  TrainView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/07.
//

import MLX
import SwiftUI
import UIKit

struct TrainSnapshot: Identifiable ,Hashable{
    let id = UUID()
    let timestamp: Date
    let url: URL
    let iteration: Int
    let renderCameraIntrinsics: RenderCameraIntrinsics?
}
class TrainOutputAsset: ObservableObject {
    static let shared = TrainOutputAsset()
    @Published var snapshots: [TrainSnapshot] = []
    func getOutputDirectory() -> URL {
        return FileManager.default.temporaryDirectory.appendingPathComponent("outputs")
    }
    func pushSnapshot(
        url: URL, iteration: Int, timestamp: Date,
        renderCameraIntrinsics: RenderCameraIntrinsics? = nil
    ) {
        DispatchQueue.main.async{
            self.snapshots.append(
                TrainSnapshot(
                    timestamp: timestamp, url: url, iteration: iteration,
                    renderCameraIntrinsics: renderCameraIntrinsics))
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

            return TrainSnapshot(
                timestamp: timestamp, url: url, iteration: iter,
                renderCameraIntrinsics: nil)
        }
        .sorted(by: { $0.iteration < $1.iteration })

        DispatchQueue.main.async {
            self.snapshots = snapshots
        }
    }
}

class TrainViewModel: ObservableObject {
    var trainer: GaussianTrainer?
}
extension TrainViewModel: GaussianTrainerDelegate {
    func pushSnapshot(
        url: URL, iteration: Int, timestamp: Date,
        renderCameraIntrinsics: RenderCameraIntrinsics?
    ) {
        TrainOutputAsset.shared.pushSnapshot(
            url: url, iteration: iteration, timestamp: timestamp,
            renderCameraIntrinsics: renderCameraIntrinsics)
    }

    func pushLoss(
        loss: Float,
        iteration: Int?,
        fps: Float?,
        gaussianCount: Int?,
        timestamp: Date
    ) {
        LossChartData.shared.push(
            loss: loss,
            iteration: iteration,
            fps: fps,
            gaussianCount: gaussianCount,
            timestamp: timestamp
        )
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
    @State private var latestCaptureURL: URL?
    @State private var showShareSheet: Bool = false
    @State var captureEnabled: Bool = false
    @State var intervalProfilingEnabled: Bool = false
    @State var profilingLogInterval: Int = 20
    @State var profilingTopKSections: Int = 12
    @State private var maintenanceSettings = GaussianMaintenanceSettings.recommended
    @State private var showMaintenanceCustomization: Bool = false

    func getDataLoader() throws -> DataLoaderProtocol {
        switch selected.format {
        case .colmap:
            guard let url = selected.url else {
                throw NSError(domain: "No zip", code: 1)
            }
            return UserColmapDataLoader(zipUrl: url)
        case .nerfStudio:
            guard let url = selected.url else {
                throw NSError(domain: "No zip", code: 2)
            }
            return UserNerfStudioDataLoader(zipUrl: url)
        case .nerfSynthetic:
            guard let url = selected.url else {
                throw NSError(domain: "No zip", code: 3)
            }
            return UserNerfSyntheticDataLoader(zipUrl: url)
        case .demo(let kind):
            switch kind {

            case .chair:
                return BlenderDemoDataLoader()
            case .hotdog:
                return DemoNerfSyntheticDataLoader()
            }
        }
    }
    private func makeCaptureURL() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "captures", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        let fileName = "mlx_capture_\(formatter.string(from: Date())).gputrace"
        return dir.appendingPathComponent(fileName)
    }
    
    private func loadOriginalResolution() {
        guard selected.isValid else { return }
        
        isLoadingResolution = true
        DispatchQueue.global().async {
            do {
                let dataLoader = try getDataLoader()
                let (width, height) = try dataLoader.getOriginalImageSize()
                
                DispatchQueue.main.async {
                    self.originalWidth = width
                    self.originalHeight = height
                    self.isLoadingResolution = false
                }
            } catch {
                Logger.shared.error(error: error)
                DispatchQueue.main.async {
                    self.isLoadingResolution = false
                }
            }
        }
    }
    func doTrain() throws {
        let captureURL: URL? = captureEnabled ? try makeCaptureURL() : nil
        if let captureURL {
            DispatchQueue.main.async {
                self.latestCaptureURL = captureURL
            }
            MLX.GPU.startCapture(url: captureURL)
        } else {
            DispatchQueue.main.async {
                self.latestCaptureURL = nil
            }
        }
        defer {
            if let captureURL {
                MLX.GPU.stopCapture(url: captureURL)
            }
        }
        let dataLoader = try getDataLoader()
        let whiteBackground = false
        let (data, pointCloud, TILE_SIZE) = try dataLoader.load(
            resizeFactor: resizeFactor,
            whiteBackground: whiteBackground
        )
        pointCloud.centering(data: data)

        let cacheLimit = GaussianTrainer.defaultCacheLimit()
        let manualClearCache = GaussianTrainer.needEagerClearCache()
        let memoryLimit = Int(min(UInt64(cacheLimit) * 2, UInt64(Int.max)))
        MLX.Memory.cacheLimit = cacheLimit
        MLX.Memory.memoryLimit = memoryLimit
        let safeShDegree = min(
            max(shDegree, SHDegreeConfig.supportedRange.lowerBound),
            SHDegreeConfig.supportedRange.upperBound
        )
        let safeSampleCount = max(1, sampleCount)
        let safeIterationCount = max(1, iterationCount)
        let safeMaintenanceSettings = maintenanceSettings.sanitized()
        let gaussModel = GaussianTrainer.createModel(
            sh_degree: safeShDegree,
            pointCloud: pointCloud,
            sampleCount: safeSampleCount,
            manualClearCache: manualClearCache
        )

        viewModel.trainer = GaussianTrainer(
            model: gaussModel,
            data: data.optimized(),
            gaussRender: GaussianRenderer(
                active_sh_degree: safeShDegree,
                W: data.Ws[0].item(Int.self),
                H: data.Hs[0].item(Int.self),
                TILE_SIZE: TILE_SIZE,
                whiteBackground: whiteBackground
            ),
            iterationCount: safeIterationCount,
            cacheLimit: cacheLimit,
            manualClearCache: manualClearCache,
            enableIntervalProfiling: intervalProfilingEnabled,
            profilingLogInterval: max(1, profilingLogInterval),
            profilingTopKSections: max(1, profilingTopKSections),
            maintenanceSettings: safeMaintenanceSettings,
            outputDirectoryURL: TrainOutputAsset.shared.getOutputDirectory()
        )
        viewModel.trainer?.delegate = viewModel
        viewModel.trainer?.startTrain()
        MLX.Memory.clearCache()
        DispatchQueue.main.async {
            self.isTraining = false
        }
    }
    @State var isTraining: Bool = false
    @State var iterationCount: Int = 30000
    @State var sampleCount: Int = 1 << 14
    @State var shDegree: Int = SHDegreeConfig.defaultValue
    @State var resizeFactor: Double = 0.5
    @State var originalWidth: Int = 0
    @State var originalHeight: Int = 0
    @State var isLoadingResolution: Bool = false
    @State var selected: SelectedDataSet = SelectedDataSet(
        format: .colmap,
        url: nil
    )
    func resetMaintenanceSettings() {
        maintenanceSettings = .recommended
        showMaintenanceCustomization = false
    }
    func startTrain() {
        latestCaptureURL = nil
        showShareSheet = false
        isTraining = true
        LossChartData.shared.clear()
        TrainStatusData.shared.clear()
        DispatchQueue.global().async {
            Logger.shared.debug("Start Train")
            do {
                try doTrain()
            } catch {
                Logger.shared.error(error: error)
                DispatchQueue.main.async {
                    self.isTraining = false
                }
            }
        }
    }
    func stopTrain() {
        viewModel.trainer?.stopTrain()
    }

    var maintenanceSummary: String {
        let settings = maintenanceSettings.sanitized()
        return
            "interval \(settings.densifyInterval)/\(settings.pruneInterval), grad \(settings.gradientThreshold), cap \(settings.maxGaussians)"
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if isTraining {
                    TrainStatusView()
                    Button(action: stopTrain) {
                        Text("Stop Train")
                    }.buttonStyle(.borderedProminent)
                } else {
                    SelectDataSetView(selected: $selected)
                        .onChange(of: selected) { oldValue, newValue in
                            loadOriginalResolution()
                        }
                
                VStack(alignment: .leading, spacing: 12) {
                    // Dataset resolution information
                    if selected.isValid {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Dataset Resolution")
                                .font(.headline)
                            if isLoadingResolution {
                                HStack {
                                    ProgressView()
                                        .scaleEffect(0.7)
                                    Text("Loading resolution...")
                                        .foregroundColor(.secondary)
                                }
                            } else if originalWidth > 0 && originalHeight > 0 {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Original: \(originalWidth) × \(originalHeight)")
                                        .font(.subheadline)
                                        .foregroundColor(.primary)
                                    Text("Output: \(Int(Double(originalWidth) * resizeFactor)) × \(Int(Double(originalHeight) * resizeFactor))")
                                        .font(.subheadline)
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    
                    HStack {
                        Text("Iteration Count")
                        TextField(
                            "30000",
                            value: $iterationCount,
                            format: .number
                        )
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.numberPad)
                        .frame(width: 140)
                    }
                    
                    HStack {
                        Text("Resize Factor")
                        TextField(
                            "0.5",
                            value: $resizeFactor,
                            format: .number
                        )
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.decimalPad)
                        .frame(width: 140)
                    }
                    
                    Toggle("Capture GPU trace", isOn: $captureEnabled)
                        .toggleStyle(.switch).frame(width: 240)

                    Toggle("Interval profiling", isOn: $intervalProfilingEnabled)
                        .toggleStyle(.switch)
                        .frame(width: 240)

                    if intervalProfilingEnabled {
                        HStack {
                            Text("Profile Log Interval")
                            TextField(
                                "20",
                                value: $profilingLogInterval,
                                format: .number
                            )
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                            .frame(width: 140)
                        }

                        HStack {
                            Text("Profile Top Sections")
                            TextField(
                                "12",
                                value: $profilingTopKSections,
                                format: .number
                            )
                            .textFieldStyle(.roundedBorder)
                            .keyboardType(.numberPad)
                            .frame(width: 140)
                        }
                    }
                    
                    HStack {
                        Text("Sample Count")
                        TextField(
                            "16384",
                            value: $sampleCount,
                            format: .number
                        )
                        .textFieldStyle(.roundedBorder)
                        .keyboardType(.numberPad)
                        .frame(width: 140)
                    }

                    HStack {
                        Text("SH Degree")
                        Spacer(minLength: 12)
                        Stepper("\(shDegree)", value: $shDegree, in: SHDegreeConfig.supportedRange)
                            .frame(width: 180)
                    }

                    Text("Default is 3. Some scenes render more cleanly at lower SH degrees.")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Gaussian Maintenance")
                                .font(.headline)
                            Spacer()
                            Button("Use Recommended", action: resetMaintenanceSettings)
                                .buttonStyle(.bordered)
                        }

                        Toggle("Enable split / prune", isOn: $maintenanceSettings.isEnabled)
                            .toggleStyle(.switch)

                        HStack(alignment: .firstTextBaseline) {
                            Text("Current")
                                .font(.subheadline.weight(.medium))
                            Text(maintenanceSummary)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Toggle("Customize maintenance settings", isOn: $showMaintenanceCustomization)
                            .toggleStyle(.switch)

                        if showMaintenanceCustomization {
                            Text(
                                "Higher gradient threshold, longer densify interval, and a lower Gaussian cap reduce runaway growth."
                            )
                            .font(.caption)
                            .foregroundColor(.secondary)

                            VStack(alignment: .leading, spacing: 8) {
                                Group {
                                    HStack {
                                        Text("Densify Interval")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "200",
                                            value: $maintenanceSettings.densifyInterval,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.numberPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Prune Interval")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "100",
                                            value: $maintenanceSettings.pruneInterval,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.numberPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Gradient Threshold")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "0.00035",
                                            value: $maintenanceSettings.gradientThreshold,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.decimalPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Min Opacity")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "0.01",
                                            value: $maintenanceSettings.minOpacity,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.decimalPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Max Scale")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "0.01",
                                            value: $maintenanceSettings.maxScale,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.decimalPad)
                                        .frame(width: 110)
                                    }
                                }

                                Group {
                                    HStack {
                                        Text("Densify Start")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "1000",
                                            value: $maintenanceSettings.densifyFromIteration,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.numberPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Densify End")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "12000",
                                            value: $maintenanceSettings.densifyUntilIteration,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.numberPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Max Gaussians")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "300000",
                                            value: $maintenanceSettings.maxGaussians,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.numberPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Split Divisor")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "1.6",
                                            value: $maintenanceSettings.splitScaleDivisor,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.decimalPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Split Noise")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "0.075",
                                            value: $maintenanceSettings.splitPositionNoiseScale,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.decimalPad)
                                        .frame(width: 110)
                                    }

                                    HStack {
                                        Text("Clone Noise")
                                        Spacer(minLength: 12)
                                        TextField(
                                            "0.005",
                                            value: $maintenanceSettings.clonePositionNoiseScale,
                                            format: .number
                                        )
                                        .textFieldStyle(.roundedBorder)
                                        .keyboardType(.decimalPad)
                                        .frame(width: 110)
                                    }
                                }
                            }
                            .font(.subheadline)
                        }
                    }
                    .padding(.top, 8)
                }
                    HStack(spacing: 12) {
                        Button(action: startTrain) {
                            Text("Do Train")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!selected.isValid)

                        if latestCaptureURL != nil {
                            Button(action: { showShareSheet = true }) {
                                Text("Share Capture")
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .sheet(isPresented: $showShareSheet) {
            if let url = latestCaptureURL {
                ShareSheet(items: [url])
            }
        }
        .onAppear {
            if selected.isValid {
                loadOriginalResolution()
            }
        }
    }
}
extension SelectedDataSet {
    var isValid: Bool {
        switch format {
        case .colmap, .nerfStudio, .nerfSynthetic:
            return url != nil
        case .demo:
            return true
        }
    }
}
