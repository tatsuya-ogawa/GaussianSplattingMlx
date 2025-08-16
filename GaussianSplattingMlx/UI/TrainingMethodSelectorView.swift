//
//  TrainingMethodSelectorView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/08/16.
//

import SwiftUI

enum TrainingMethod: String, CaseIterable {
    case gaussian = "Gaussian"
    case triangle = "Triangle"
    
    var displayName: String {
        return self.rawValue
    }
    
    var description: String {
        switch self {
        case .gaussian:
            return "3D Gaussian Splatting uses point-based primitives with Gaussian distributions for neural rendering"
        case .triangle:
            return "Triangle Splatting uses triangle-based primitives with soft window functions for neural rendering"
        }
    }
    
    var systemImage: String {
        switch self {
        case .gaussian:
            return "circle.fill"
        case .triangle:
            return "triangle.fill"
        }
    }
}

struct TrainingMethodCard: View {
    let method: TrainingMethod
    let isSelected: Bool
    let onSelect: () -> Void
    
    var body: some View {
        Button(action: onSelect) {
            VStack(spacing: 16) {
                // Icon
                Image(systemName: method.systemImage)
                    .font(.system(size: 40))
                    .foregroundColor(isSelected ? .white : .blue)
                
                // Title
                Text(method.displayName + " Splatting")
                    .font(.title2)
                    .fontWeight(.semibold)
                    .foregroundColor(isSelected ? .white : .primary)
                
                // Description
                Text(method.description)
                    .font(.body)
                    .foregroundColor(isSelected ? .white.opacity(0.9) : .secondary)
                    .multilineTextAlignment(.center)
                    .lineLimit(3)
                
                Spacer()
            }
            .padding(20)
            .frame(maxWidth: .infinity, minHeight: 200)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(isSelected ? Color.blue : Color.gray.opacity(0.1))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(isSelected ? Color.blue : Color.gray.opacity(0.3), lineWidth: 2)
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct TrainingMethodSelectorView: View {
    @State private var selectedMethod: TrainingMethod = .gaussian
    @State private var showTrainingView = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 8) {
                    Text("Neural Rendering Training")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Select a training method to begin")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                // Method Selection Cards
                LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 20), count: 2), spacing: 20) {
                    ForEach(TrainingMethod.allCases, id: \.self) { method in
                        TrainingMethodCard(
                            method: method,
                            isSelected: selectedMethod == method,
                            onSelect: { selectedMethod = method }
                        )
                    }
                }
                
                Spacer()
                
                // Start Training Button
                Button(action: {
                    showTrainingView = true
                }) {
                    HStack {
                        Image(systemName: "play.fill")
                        Text("Start \(selectedMethod.displayName) Training")
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
                    .font(.headline)
                }
                .padding(.horizontal)
            }
            .padding()
            .navigationTitle("Training Methods")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showTrainingView) {
                NavigationView {
                    Group {
                        switch selectedMethod {
                        case .gaussian:
                            GaussianTrainView()
                        case .triangle:
                            TriangleTrainView()
                        }
                    }
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarLeading) {
                            Button("Done") {
                                showTrainingView = false
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Wrapper View (Main TrainView)

struct TrainView: View {
    var body: some View {
        TrainingMethodSelectorView()
    }
}

#Preview {
    TrainView()
}