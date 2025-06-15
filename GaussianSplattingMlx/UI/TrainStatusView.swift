//
//  TrainStatusView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/04.
//

import MLX
import SwiftUI

struct TrainImage: Identifiable {
    let id = UUID()
    let timestamp: Date
    let loss: Float
    let render: UIImage
    let truth: UIImage
    let iteration: Int
}
class TrainStatusData: ObservableObject {
    static let shared = TrainStatusData()
    private init() {}
    @Published private(set) var imageData: [TrainImage] = []
    let maxVisibleCount = 5
    func pushImageData(
        render: MLXArray, truth: MLXArray, loss: Float, iteration: Int, timestamp: Date = Date()
    ) {
        guard let render = render.toRGBToUIImage() else { return }
        guard let truth = truth.toRGBToUIImage() else { return }
        DispatchQueue.main.async { [self] in
            self.imageData.append(
                TrainImage(
                    timestamp: timestamp, loss: loss, render: render, truth: truth,
                    iteration: iteration)
            )
            if self.imageData.count > maxVisibleCount {
                imageData.removeFirst(imageData.count - maxVisibleCount)
            }
        }
    }

    func clear() {
        DispatchQueue.main.async {
            self.imageData.removeAll()
        }
    }
}
struct TrainImageListView: View {
    @ObservedObject var trainStatus = TrainStatusData.shared

    var body: some View {
        List(trainStatus.imageData.reversed()) { image in
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .center) {
                    Text("Render")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Image(uiImage: image.render)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 80, height: 80)
                        .cornerRadius(8)
                }
                VStack(alignment: .center) {
                    Text("GT")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Image(uiImage: image.truth)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 80, height: 80)
                        .cornerRadius(8)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Iteration:\(image.iteration)")
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Text(image.timestamp, style: .time)
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Text("Loss:\(image.loss)")
                        .font(.caption2)
                        .foregroundColor(.gray)
                    Spacer()
                }
            }
            .padding(.vertical, 6)
        }
        .listStyle(.plain)
        .navigationTitle("Train Image List")
    }
}

struct TrainStatusView: View {
    @ObservedObject var chartData = LossChartData.shared
    @State private var selected: LossData? = nil
   
    var body: some View {
        VStack {
            LossChartView()
            HStack {
                TrainImageListView()
                SnapshotRenderView()
            }
        }
    }
}
