//
//  TrainView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/07.
//

import MLX
import SwiftUI

class TrainViewModel: ObservableObject {
    var trainer: GaussianTrainer?
}
struct TrainView: View {
    @StateObject var viewModel = TrainViewModel()
    func getOutputDirectory()->URL{
        return FileManager.default.temporaryDirectory.appendingPathComponent("outputs")
    }
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
            outputDirectoryURL: getOutputDirectory()
        )
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
        VStack {
            if isTraining {
                TrainStatusView()
                Button(action: stopTrain) {
                    Text("Stop Train")
                }.buttonStyle(.borderedProminent)
            } else {
                SelectDataSetView(selected: $selected)
                Button(action: startTrain) {
                    Text("Do Train")
                }.buttonStyle(.borderedProminent).disabled(!selected.isValid)
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
