//
//  TrainView.swift (now GaussianTrainView)
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

class GaussianTrainViewModel: ObservableObject {
    var trainer: GaussianTrainer?
}
extension GaussianTrainViewModel: GaussianTrainerDelegate {
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
struct GaussianTrainView: View {
    @StateObject var viewModel = GaussianTrainViewModel()

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
    var body: some View {
        VStack(spacing: 20) {
            // Header
            Text("Gaussian Splatting Training")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Text("Train neural rendering using point-based primitives with Gaussian distributions")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            if isTraining {
                TrainStatusView()
                Button(action: stopTrain) {
                    HStack {
                        Image(systemName: "stop.fill")
                        Text("Stop Training")
                    }
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
            } else {
                VStack(spacing: 20) {
                    SelectDataSetView(selected: $selected)
                    
                    Button(action: startTrain) {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Start Gaussian Training")
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
        .navigationTitle("Gaussian Splatting")
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
