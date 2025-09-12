import SwiftUI
#if canImport(ARKit)
import ARKit
import RealityKit
#endif

// MARK: - ARKit Data Collection View for GaussianSplatting

#if canImport(ARKit)
struct ARKitCaptureView: View {
    @StateObject private var captureManager = ARKitCaptureManager()
    @State private var showingSettings = false
    @State private var showingExportAlert = false
    @State private var exportProgress: Double = 0
    @State private var isExporting = false
    
    var body: some View {
        NavigationView {
            ZStack {
                // ARKit Camera Feed
                #if canImport(ARKit)
                ARViewContainer(captureManager: captureManager)
                    .ignoresSafeArea(.all)
                #else
                Rectangle()
                    .fill(Color.black)
                    .ignoresSafeArea(.all)
                    .overlay(
                        VStack {
                            Image(systemName: "camera.viewfinder")
                                .font(.system(size: 50))
                                .foregroundColor(.gray)
                            Text("ARKit Preview")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                    )
                #endif
                
                // Overlay UI
                VStack {
                    // Top Status Bar
                    topStatusBar
                    
                    Spacer()
                    
                    // Bottom Control Panel
                    bottomControlPanel
                }
                .padding()
                
                // Export Progress Overlay
                if isExporting {
                    exportProgressOverlay
                }
            }
            .navigationTitle("ARKit Capture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    settingsButton
                }
            }
            .sheet(isPresented: $showingSettings) {
                CaptureSettingsView(captureManager: captureManager)
            }
            .alert("Export Dataset", isPresented: $showingExportAlert) {
                Button("Export") {
                    Task {
                        await exportDataset()
                    }
                }
                Button("Cancel", role: .cancel) { }
            } message: {
                Text("Export \(captureManager.frameCount) captured frames as training dataset?")
            }
        }
    }
    
    // MARK: - UI Components
    
    private var topStatusBar: some View {
        HStack {
            // Session Status
            sessionStatusIndicator
            
            Spacer()
            
            // Frame Counter
            frameCounterView
        }
        .padding()
        .background(Color.black.opacity(0.6))
        .cornerRadius(12)
    }
    
    private var sessionStatusIndicator: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(captureManager.isRunning ? Color.green : Color.red)
                .frame(width: 12, height: 12)
            
            Text(captureManager.isRunning ? "AR Active" : "AR Inactive")
                .foregroundColor(.white)
                .font(.system(size: 14, weight: .medium))
        }
    }
    
    private var frameCounterView: some View {
        VStack(alignment: .trailing, spacing: 4) {
            Text("\(captureManager.frameCount)")
                .font(.system(size: 24, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
            
            Text("frames")
                .font(.system(size: 12, weight: .regular))
                .foregroundColor(.white.opacity(0.8))
        }
    }
    
    private var bottomControlPanel: some View {
        VStack(spacing: 16) {
            // Error Message
            if let errorMessage = captureManager.errorMessage {
                errorMessageView(errorMessage)
            }
            
            // Main Controls
            HStack(spacing: 20) {
                // Start/Stop AR Session
                sessionControlButton
                
                // Capture Controls
                captureControlButton
                
                // Export Button
                exportButton
            }
        }
        .padding()
        .background(Color.black.opacity(0.8))
        .cornerRadius(16)
    }
    
    private func errorMessageView(_ message: String) -> some View {
        Text(message)
            .font(.system(size: 14, weight: .medium))
            .foregroundColor(.white)
            .padding()
            .background(Color.red.opacity(0.8))
            .cornerRadius(8)
    }
    
    private var sessionControlButton: some View {
        Button(action: {
            if captureManager.isRunning {
                captureManager.stopSession()
            } else {
                captureManager.startSession()
            }
        }) {
            VStack(spacing: 4) {
                Image(systemName: captureManager.isRunning ? "stop.circle" : "play.circle")
                    .font(.system(size: 32, weight: .medium))
                
                Text(captureManager.isRunning ? "Stop AR" : "Start AR")
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundColor(captureManager.isRunning ? .red : .green)
        }
        .disabled(captureManager.isCapturing)
    }
    
    private var captureControlButton: some View {
        Button(action: {
            if captureManager.isCapturing {
                captureManager.stopCapture()
            } else {
                captureManager.startCapture()
            }
        }) {
            VStack(spacing: 4) {
                Image(systemName: captureManager.isCapturing ? "record.circle" : "camera.circle")
                    .font(.system(size: 40, weight: .medium))
                    .scaleEffect(captureManager.isCapturing ? 1.2 : 1.0)
                    .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), 
                              value: captureManager.isCapturing)
                
                Text(captureManager.isCapturing ? "Recording" : "Capture")
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundColor(captureManager.isCapturing ? .red : .white)
        }
        .disabled(!captureManager.isRunning)
    }
    
    private var exportButton: some View {
        Button(action: {
            showingExportAlert = true
        }) {
            VStack(spacing: 4) {
                Image(systemName: "square.and.arrow.up.circle")
                    .font(.system(size: 32, weight: .medium))
                
                Text("Export")
                    .font(.system(size: 12, weight: .medium))
            }
            .foregroundColor(captureManager.frameCount > 0 ? .blue : .gray)
        }
        .disabled(captureManager.frameCount == 0 || captureManager.isCapturing || isExporting)
    }
    
    private var settingsButton: some View {
        Button(action: {
            showingSettings = true
        }) {
            Image(systemName: "gear")
                .foregroundColor(.white)
        }
    }
    
    private var exportProgressOverlay: some View {
        ZStack {
            Color.black.opacity(0.7)
                .ignoresSafeArea()
            
            VStack(spacing: 20) {
                ProgressView("Exporting Dataset...", value: exportProgress, total: 1.0)
                    .progressViewStyle(LinearProgressViewStyle())
                    .tint(.blue)
                
                Text("\(Int(exportProgress * 100))%")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundColor(.white)
            }
            .padding(40)
            .background(Color.black.opacity(0.8))
            .cornerRadius(16)
        }
    }
    
    // MARK: - Actions
    
    private func exportDataset() async {
        isExporting = true
        exportProgress = 0
        
        do {
            // Simulate export progress
            for i in 1...10 {
                try await Task.sleep(nanoseconds: 100_000_000) // 0.1 second
                await MainActor.run {
                    exportProgress = Double(i) / 10.0
                }
            }
            
            try await captureManager.exportDataset()
            
            await MainActor.run {
                isExporting = false
                exportProgress = 0
            }
            
        } catch {
            await MainActor.run {
                captureManager.errorMessage = "Export failed: \(error.localizedDescription)"
                isExporting = false
                exportProgress = 0
            }
        }
    }
}

// MARK: - ARView Container

struct ARViewContainer: UIViewRepresentable {
    let captureManager: ARKitCaptureManager
    
    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        
        // Configure AR view for data collection
        arView.session = captureManager.session
        arView.renderOptions = [.disablePersonOcclusion, .disableMotionBlur]
        
        // Enable debug visualization (optional)
        #if DEBUG
        arView.debugOptions = [.showFeaturePoints, .showWorldOrigin]
        #endif
        
        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
        // Updates handled by capture manager
    }
}

// MARK: - Capture Settings View

struct CaptureSettingsView: View {
    @ObservedObject var captureManager: ARKitCaptureManager
    @Environment(\.dismiss) private var dismiss
    
    @State private var datasetName: String = ""
    @State private var targetFrameCount: String = "100"
    @State private var captureInterval: String = "0.5"
    
    var body: some View {
        NavigationView {
            Form {
                Section("Dataset Configuration") {
                    TextField("Dataset Name", text: $datasetName)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    TextField("Target Frame Count", text: $targetFrameCount)
                        .keyboardType(.numberPad)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    TextField("Capture Interval (seconds)", text: $captureInterval)
                        .keyboardType(.decimalPad)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                Section("Output Settings") {
                    HStack {
                        Text("Output Directory:")
                        Spacer()
                        Text(captureManager.config.outputDirectory.lastPathComponent)
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Resolution:")
                        Spacer()
                        Text("\(Int(captureManager.config.captureResolution.width))×\(Int(captureManager.config.captureResolution.height))")
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("Status") {
                    HStack {
                        Text("Current Frames:")
                        Spacer()
                        Text("\(captureManager.frameCount)")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("AR Session:")
                        Spacer()
                        Text(captureManager.isRunning ? "Active" : "Inactive")
                            .foregroundColor(captureManager.isRunning ? .green : .red)
                    }
                }
            }
            .navigationTitle("Capture Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
                
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        saveSettings()
                        dismiss()
                    }
                }
            }
        }
        .onAppear {
            loadCurrentSettings()
        }
    }
    
    private func loadCurrentSettings() {
        datasetName = captureManager.config.datasetName
        targetFrameCount = String(captureManager.config.targetFrameCount)
    }
    
    private func saveSettings() {
        let newConfig = ARKitCaptureManager.DatasetConfig(
            outputDirectory: captureManager.config.outputDirectory,
            datasetName: datasetName.isEmpty ? "gaussian_splatting_dataset" : datasetName,
            targetFrameCount: Int(targetFrameCount) ?? 100,
            captureResolution: captureManager.config.captureResolution
        )
        
        captureManager.updateConfig(newConfig)
    }
}

// MARK: - Preview

#Preview {
    ARKitCaptureView()
}
#else
struct ARKitCaptureView: View {
    var body: some View {
        VStack {
            Image(systemName: "iphone")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            Text("ARKit Capture")
                .font(.largeTitle)
                .fontWeight(.bold)
            Text("Only available on iOS/iPadOS")
                .foregroundColor(.secondary)
        }
        .padding()
    }
}
#endif