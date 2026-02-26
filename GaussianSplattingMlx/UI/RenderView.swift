//
//  RenderView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/15.
//

//
//  TrainView.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/07.
//

import MLX
import MetalKit
import SwiftUI
import simd

private func getInitialCamera(
    boundingBox: ([Float], [Float]),
    width: Int,
    height: Int
) -> (simd_double4x4, Double, Double) {
    let (min, max) = boundingBox
    let center = zip(min, max).map { ($0 + $1) / 2 }
    let size = zip(min, max).map { abs($0 - $1) }
    let diameter = sqrt(size.map { $0 * $0 }.reduce(0, +))

    let camTarget = simd_double3(
        Double(center[0]),
        Double(center[1]),
        Double(center[2])
    )
    let distance = Double(diameter) * 1.5
    let camPos = simd_double3(
        Double(center[0]),
        Double(center[1]),
        Double(center[2]) + distance
    )

    let up = simd_double3(0, 1, 0)
    let viewMatrix = lookAt(eye: camPos, center: camTarget, up: up)

    let focalX = Double(width) / (2.0 * tan(Double.pi / 6.0))
    let focalY = Double(height) / (2.0 * tan(Double.pi / 6.0))
    return (viewMatrix, focalX, focalY)
}

private func lookAt(eye: simd_double3, center: simd_double3, up: simd_double3)
    -> simd_double4x4
{
    let f = simd_normalize(center - eye)
    let s = simd_normalize(simd_cross(f, up))
    let u = simd_cross(s, f)
    var result = matrix_identity_double4x4
    result.columns.0 = simd_double4(s, 0)
    result.columns.1 = simd_double4(u, 0)
    result.columns.2 = simd_double4(f, 0)
    result.columns.3 = simd_double4(eye, 1)
    return result
}
class RenderViewModel: ObservableObject {
    let width: Int
    let height: Int
    
    private let metalRenderer: MetalGaussianRenderer?
    
    private lazy var gaussRender = GaussianRenderer(
        active_sh_degree: 4,
        W: width,
        H: height,
        TILE_SIZE: TILE_SIZE_H_W(w: 64, h: 64),
        whiteBackground: false
    )
    
    var activeRenderer: MetalGaussianRenderer? { metalRenderer }
    
    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.metalRenderer = MetalGaussianRenderer(maxGaussians: 1000000, tileSize: SIMD2<UInt32>(64, 64))
        if self.metalRenderer == nil {
            print("RenderViewModel: Metal renderer initialization failed.")
        }
    }
    
    private var cameraMatrix: simd_double4x4 = simd_double4x4(1)
    private var focalX: Double = 1
    private var focalY: Double = 1

    var orbitYaw: Double = 0
    var orbitPitch: Double = 0
    var orbitDistance: Double = 1.0

    var camTarget: simd_double3 = [0, 0, 0]
    var camBaseDistance: Double = 1.0

    func updateCamera() {
        let x =
            camTarget.x
            + Double(orbitDistance * cos(orbitPitch) * sin(orbitYaw))
        let y = camTarget.y + Double(orbitDistance * sin(orbitPitch))
        let z =
            camTarget.z
            + Double(orbitDistance * cos(orbitPitch) * cos(orbitYaw))
        let eye = simd_double3(x, y, z)
        let up = simd_double3(0, 1, 0)
        self.cameraMatrix = lookAt(eye: eye, center: camTarget, up: up)
    }
    func setInitialCamera(
        boundingBox: ([Float], [Float]),
        width: Int,
        height: Int
    ) {
        let (viewMatrix, focalX, focalY) = getInitialCamera(
            boundingBox: boundingBox,
            width: width,
            height: height
        )
        self.cameraMatrix = viewMatrix
        self.focalX = focalX
        self.focalY = focalY

        let (min, max) = boundingBox
        let center = zip(min, max).map { ($0 + $1) / 2 }
        camTarget = simd_double3(
            Double(center[0]),
            Double(center[1]),
            Double(center[2])
        )
        let size = zip(min, max).map { abs($0 - $1) }
        let diameter = sqrt(size.map { $0 * $0 }.reduce(0, +))
        camBaseDistance = Double(diameter) * 1.5
        orbitDistance = camBaseDistance
    }
    private var cachedMeans3d: MLXArray?
    private var cachedShs: MLXArray?
    private var cachedOpacity: MLXArray?
    private var cachedScales: MLXArray?
    private var cachedRotations: MLXArray?
    func calcBoundingBox(xyzArray: MLXArray) -> ([Float], [Float]) {
        let max = xyzArray.max(axes: [0])
        let min = xyzArray.min(axes: [0])
        return (min.asArray(Float.self), max.asArray(Float.self))
    }
    func load(url: URL) throws {
        let (_xyz, _features_dc, _features_rest, _opacity, _scales, _rotation) =
            try PlyWriter.loadGaussianBinaryPLYAsMLX(from: url)
        
        let means3d = gaussRender.get_xyz_from(_xyz)
        let opacity = gaussRender.get_opacity_from(_opacity)
        let scales = gaussRender.get_scales_from(_scales)
        let rotations = gaussRender.get_rotation_from(_rotation)
        let shs = gaussRender.get_features_from(_features_dc, _features_rest)
        
        eval(means3d)
        eval(opacity)
        eval(scales)
        eval(rotations)
        eval(shs)
        
        self.cachedMeans3d = means3d
        self.cachedOpacity = opacity
        self.cachedScales = scales
        self.cachedRotations = rotations
        self.cachedShs = shs

        setInitialCamera(boundingBox: calcBoundingBox(xyzArray: _xyz), width: width, height: height)
    }
    func render() {
        renderWithMetal()
    }
    
    func clearRenderedFrame() {
        metalRenderer?.clearFrame()
        cachedMeans3d = nil
        cachedShs = nil
        cachedOpacity = nil
        cachedScales = nil
        cachedRotations = nil
    }
    
    private func renderWithMetal() {
        guard let means3d = cachedMeans3d,
              let opacity = cachedOpacity,
              let scales = cachedScales,
              let rotations = cachedRotations,
              let shs = cachedShs,
              let metalRenderer
        else {
            return
        }

        let camera = Camera(
            width: width,
            height: height,
            focalX: Float(focalX),
            focalY: Float(focalY),
            c2w: cameraMatrix
        )
        
        metalRenderer.render(
            camera: camera,
            means3d: means3d,
            shs: shs,
            opacity: opacity,
            scales: scales,
            rotations: rotations,
            width: width,
            height: height
        )
    }
}

private struct MetalTextureView: UIViewRepresentable {
    let renderer: MetalGaussianRenderer
    let drawableSize: CGSize
    
    final class Coordinator {
        let renderer: MetalGaussianRenderer
        
        init(renderer: MetalGaussianRenderer) {
            self.renderer = renderer
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(renderer: renderer)
    }
    
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: renderer.metalDevice)
        context.coordinator.renderer.attach(to: view)
        view.drawableSize = drawableSize
        return view
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        context.coordinator.renderer.attach(to: uiView)
        uiView.drawableSize = drawableSize
    }
    
    static func dismantleUIView(_ uiView: MTKView, coordinator: Coordinator) {
        coordinator.renderer.detach()
    }
}

struct RenderView: View {
    let width: Int
    let height: Int
    private let renderQueue = DispatchQueue(label: "com.GaussianSplattingMlX.renderqueue")
    @Binding var url: URL?
    @State private var lastDrag: CGSize = .zero
    @StateObject var viewModel: RenderViewModel
    init(url: Binding<URL?>, width: Int, height: Int) {
        self.width = width
        self.height = height
        self._url = url
        _viewModel = StateObject(
            wrappedValue: RenderViewModel(width: width, height: height)
        )
    }
    func getOutputDirectory() -> URL {
        return FileManager.default.temporaryDirectory.appendingPathComponent(
            "outputs"
        )
    }
    var body: some View {
        VStack {
            if let renderer = viewModel.activeRenderer {
                ZStack(alignment: .bottomTrailing) {
                    MetalTextureView(
                        renderer: renderer,
                        drawableSize: CGSize(width: self.width, height: self.height)
                    )
                        .frame(
                            width: CGFloat(self.width),
                            height: CGFloat(self.height)
                        )
                        .cornerRadius(8)
                    
                    // Camera orientation gizmo
                    CameraGizmoView(
                        yaw: viewModel.orbitYaw,
                        pitch: viewModel.orbitPitch
                    )
                    .frame(width: 80, height: 80)
                    .padding(16)
                }
            } else {
                ZStack {
                    Rectangle()
                        .fill(Color.gray.opacity(0.2))
                        .frame(
                            width: CGFloat(self.width),
                            height: CGFloat(self.height)
                        )
                        .cornerRadius(8)
                    Text("No Image")
                        .foregroundColor(.gray)
                }
            }
        }
        .gesture(
            DragGesture()
                .onChanged { value in
                    let delta = value.translation
                    let deltaYaw = Double(delta.width - lastDrag.width) * 0.01
                    let deltaPitch =
                        Double(delta.height - lastDrag.height) * 0.01
                    viewModel.orbitYaw += deltaYaw
                    viewModel.orbitPitch += deltaPitch
                    viewModel.orbitPitch = min(
                        .pi / 2 - 0.01,
                        max(-.pi / 2 + 0.01, viewModel.orbitPitch)
                    )
                    lastDrag = value.translation
                    renderQueue.async {
                        viewModel.updateCamera()
                        viewModel.render()
                    }
                }
                .onEnded { _ in lastDrag = .zero }
        )
        .gesture(
            MagnificationGesture()
                .onChanged { scale in
                    viewModel.orbitDistance =
                        viewModel.camBaseDistance / Double(scale)
                    renderQueue.async {
                        viewModel.updateCamera()
                        viewModel.render()
                    }
                }
        )
        .onChange(of: url) { newURL in
            if let url = newURL {
                renderQueue.async {
                    do {
                        try viewModel.load(url: url)
                        viewModel.updateCamera()
                        viewModel.render()
                    } catch {
                        print(error)
                    }
                }
            } else {
                viewModel.clearRenderedFrame()
            }
        }
    }
}
struct SnapshotRenderView: View {
    @ObservedObject var asset = TrainOutputAsset.shared
    @State private var selected: TrainSnapshot?
    @State private var renderURL: URL?
    @State private var showShareSheet = false
    
    var body: some View {
        VStack{
            Text("Snapshot Render")
            HStack {
                RenderView(url: .constant(renderURL), width: 512, height: 512)
                    .frame(width: 512, height: 512)
                Divider()
                VStack {
                    List(selection: $selected) {
                        ForEach(asset.snapshots) { snap in
                            HStack {
                                Text("iter \(snap.iteration)")
                                Spacer()
                                Text(snap.timestamp, style: .time)
                            }
                            .contentShape(Rectangle())
                            .tag(snap)
                        }
                    }
                    .frame(minWidth: 250, idealWidth: 300, maxWidth: 400)
                    
                    if let selected = selected {
                        HStack(spacing: 16) {
                            Button(action: {
                                renderURL = selected.url
                            }) {
                                Label("Render", systemImage: "eye.fill")
                            }
                            .buttonStyle(.borderedProminent)
                            
                            Button(action: {
                                showShareSheet = true
                            }) {
                                Label("Share", systemImage: "square.and.arrow.up")
                            }
                            .buttonStyle(.bordered)
                        }
                        .padding()
                    }
                }
                .onAppear {
                    //                asset.loadSnapshotsFromDirectory()
                }
            }
        }
        .sheet(isPresented: $showShareSheet) {
            if let selected = selected {
                ShareSheet(items: [selected.url])
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        let controller = UIActivityViewController(activityItems: items, applicationActivities: nil)
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// Camera orientation gizmo showing XYZ axes
struct CameraGizmoView: View {
    let yaw: Double
    let pitch: Double
    
    var body: some View {
        GeometryReader { geometry in
            let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height / 2)
            let radius = min(geometry.size.width, geometry.size.height) / 2 - 10
            
            ZStack {
                // Background circle
                Circle()
                    .fill(Color.black.opacity(0.3))
                    .overlay(
                        Circle()
                            .stroke(Color.white.opacity(0.5), lineWidth: 1)
                    )
                
                // X axis (Red)
                AxisArrow(
                    center: center,
                    radius: radius,
                    yaw: yaw,
                    pitch: pitch,
                    axisDirection: SIMD3<Double>(1, 0, 0),
                    color: .red,
                    label: "X"
                )
                
                // Y axis (Green)
                AxisArrow(
                    center: center,
                    radius: radius,
                    yaw: yaw,
                    pitch: pitch,
                    axisDirection: SIMD3<Double>(0, 1, 0),
                    color: .green,
                    label: "Y"
                )
                
                // Z axis (Blue)
                AxisArrow(
                    center: center,
                    radius: radius,
                    yaw: yaw,
                    pitch: pitch,
                    axisDirection: SIMD3<Double>(0, 0, 1),
                    color: .blue,
                    label: "Z"
                )
            }
        }
    }
}

struct AxisArrow: View {
    let center: CGPoint
    let radius: Double
    let yaw: Double
    let pitch: Double
    let axisDirection: SIMD3<Double>
    let color: Color
    let label: String
    
    var body: some View {
        let projected = projectAxis(
            axis: axisDirection,
            yaw: yaw,
            pitch: pitch
        )
        
        let startPoint = center
        let endPoint = CGPoint(
            x: center.x + projected.x * radius,
            y: center.y - projected.y * radius // Flip Y for screen coordinates
        )
        
        let depth = projected.z
        let isInFront = depth > 0
        let opacity = isInFront ? 1.0 : 0.3
        let lineWidth: CGFloat = isInFront ? 2 : 1
        
        ZStack {
            // Arrow line
            Path { path in
                path.move(to: startPoint)
                path.addLine(to: endPoint)
            }
            .stroke(color, lineWidth: lineWidth)
            
            // Arrow head
            Circle()
                .fill(color)
                .frame(width: isInFront ? 8 : 5, height: isInFront ? 8 : 5)
                .position(endPoint)
            
            // Label
            Text(label)
                .font(.system(size: 12, weight: isInFront ? .bold : .regular))
                .foregroundColor(color)
                .position(
                    x: endPoint.x + (projected.x > 0 ? 12 : -12),
                    y: endPoint.y - (projected.y > 0 ? 12 : -12)
                )
        }
        .opacity(opacity)
    }
    
    // Project 3D axis direction to 2D screen space
    private func projectAxis(axis: SIMD3<Double>, yaw: Double, pitch: Double) -> SIMD3<Double> {
        // Create rotation matrices
        let cosYaw = cos(yaw)
        let sinYaw = sin(yaw)
        let cosPitch = cos(pitch)
        let sinPitch = sin(pitch)
        
        // Rotate around Y axis (yaw)
        var rotated = SIMD3<Double>(
            axis.x * cosYaw + axis.z * sinYaw,
            axis.y,
            -axis.x * sinYaw + axis.z * cosYaw
        )
        
        // Rotate around X axis (pitch)
        rotated = SIMD3<Double>(
            rotated.x,
            rotated.y * cosPitch - rotated.z * sinPitch,
            rotated.y * sinPitch + rotated.z * cosPitch
        )
        
        return rotated
    }
}
