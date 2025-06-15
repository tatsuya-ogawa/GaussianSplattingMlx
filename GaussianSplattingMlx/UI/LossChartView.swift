import Charts
//
//  LossChartView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/04.
//
import SwiftUI

struct LossData: Identifiable {
    let id = UUID()
    let timestamp: Date
    let loss: Float
    let iteration: Int?
}

class LossChartData: ObservableObject {
    @Published private(set) var data: [LossData] = []
    @Published private(set) var iteration: Int? = nil
    static let shared = LossChartData()
    private init() {}

    let maxVisibleCount = 100

    func push(loss: Float, iteration: Int? = nil, timestamp: Date = Date()) {
        DispatchQueue.main.async { [self] in
            self.iteration = iteration
            self.data.append(
                LossData(timestamp: timestamp, loss: loss, iteration: iteration)
            )
            if self.data.count > maxVisibleCount {
                data.removeFirst(data.count - maxVisibleCount)
            }
        }
    }

    func clear() {
        DispatchQueue.main.async {
            self.data.removeAll()
        }
    }
}
struct LossChartView: View {
    @ObservedObject var chartData = LossChartData.shared
    @State private var selected: LossData? = nil

    var body: some View {
        VStack {
            Text("Iteration \(chartData.iteration ?? 0)")
            Chart(chartData.data) { item in
                LineMark(
                    x: .value("Time", item.timestamp),
                    y: .value("Loss", item.loss)
                )
                PointMark(
                    x: .value("Time", item.timestamp),
                    y: .value("Loss", item.loss)
                )
                .annotation(position: .top, alignment: .center) {
                    if selected?.id == item.id {
                        VStack(alignment: .leading, spacing: 2) {
                            if let iter = item.iteration {
                                Text("iter: \(iter)").font(.caption2)
                            }
                            Text(String(format: "loss: %.4f", item.loss)).font(.caption2)
                            Text(item.timestamp, style: .time).font(.caption2)
                        }
                        .padding(4)
                        .background(.thinMaterial)
                        .cornerRadius(6)
                        .shadow(radius: 1)
                    }
                }
            }
            .chartXAxis {
                AxisMarks(values: .automatic(desiredCount: 5)) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel(format: .dateTime.hour().minute())
                }
            }
            .chartYScale(domain: 0...0.5)
            .frame(height: 300)
            .padding()
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle().fill(.clear).contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    let location = value.location
                                    if let dt: Date = proxy.value(atX: location.x) {
                                        if let nearest = chartData.data.min(by: {
                                            abs(
                                                $0.timestamp.timeIntervalSince1970
                                                    - dt.timeIntervalSince1970)
                                                < abs(
                                                    $1.timestamp.timeIntervalSince1970
                                                        - dt.timeIntervalSince1970)
                                        }) {
                                            selected = nearest
                                        }
                                    }
                                }
                        )
                }
            }
        }
    }
}
