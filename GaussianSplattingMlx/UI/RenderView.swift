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
    
    @Published var useMetalRenderer: Bool = false
    @Published var renderingMethod: SplattingMethod = .gaussian
    private let metalRenderer: MetalGaussianRenderer?
    private let metalTriangleRenderer: MetalTriangleRenderer?
    private var triangleRenderer: TriangleRenderer?
    
    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.metalRenderer = MetalGaussianRenderer(maxGaussians: 1000000, tileSize: SIMD2<UInt32>(64, 64))
        self.metalTriangleRenderer = MetalTriangleRenderer(maxTriangles: 500000, tileSize: SIMD2<UInt32>(64, 64))
        self.triangleRenderer = TriangleRenderer(
            active_sh_degree: 4,
            W: width,
            H: height,
            TILE_SIZE: TILE_SIZE_H_W(w: 64, h: 64),
            whiteBackground: false
        )
    }

    @Published var image: UIImage?
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
    var xyzArray: MLXArray?
    var features_dcArray: MLXArray?
    var features_restArray: MLXArray?
    var opacityArray: MLXArray?
    var scalesArray: MLXArray?
    var rotationArray: MLXArray?
    
    // Triangle model data
    var triangleModel: TriangleModel?
    func calcBoundingBox(xyzArray: MLXArray) -> ([Float], [Float]) {
        let max = xyzArray.max(axes: [0])
        let min = xyzArray.min(axes: [0])
        return (min.asArray(Float.self), max.asArray(Float.self))
    }
    func load(url: URL) throws {
        let fileExtension = url.pathExtension.lowercased()
        
        switch renderingMethod {
        case .gaussian:
            let (_xyz, _features_dc, _features_rest, _opacity, _scales, _rotation) =
                try PlyWriter.loadGaussianBinaryPLYAsMLX(from: url)
            self.xyzArray = _xyz
            self.features_dcArray = _features_dc
            self.features_restArray = _features_rest
            self.opacityArray = _opacity
            self.scalesArray = _scales
            self.rotationArray = _rotation
            
            setInitialCamera(boundingBox: calcBoundingBox(xyzArray: _xyz), width: width, height: height)
            
        case .triangle:
            if fileExtension == "json" {
                // Load triangle model from JSON
                triangleModel = TriangleModel(numTriangles: 1000, shDegree: 3)
                try triangleModel!.load(path: url.path)
                
                let centers = triangleModel!.getTriangleCenters()
                setInitialCamera(boundingBox: calcBoundingBox(xyzArray: centers), width: width, height: height)
            } else {
                // Convert Gaussian PLY to triangle model
                let (_xyz, _features_dc, _features_rest, _opacity, _scales, _rotation) =
                    try PlyWriter.loadGaussianBinaryPLYAsMLX(from: url)
                
                triangleModel = TriangleModel(fromPointCloud: _xyz, colors: _features_dc, shDegree: 3)
                setInitialCamera(boundingBox: calcBoundingBox(xyzArray: _xyz), width: width, height: height)
            }
        }
    }
    func render() {
        switch renderingMethod {
        case .gaussian:
            guard let xyzArray, let opacityArray, let scalesArray,
                let rotationArray, let features_dcArray, let features_restArray
            else {
                return
            }
            
            if useMetalRenderer && metalRenderer != nil {
                renderGaussianWithMetal()
            } else {
                renderGaussianWithMLX()
            }
            
        case .triangle:
            guard let triangleModel = triangleModel else { return }
            
            if useMetalRenderer && metalTriangleRenderer != nil {
                renderTriangleWithMetal(triangleModel: triangleModel)
            } else {
                renderTriangleWithMLX(triangleModel: triangleModel)
            }
        }
    }
    
    private func renderGaussianWithMLX() {
        guard let xyzArray, let opacityArray, let scalesArray,
            let rotationArray, let features_dcArray, let features_restArray
        else {
            return
        }
        
        let gaussRender = GaussianRenderer(
            active_sh_degree: 4,
            W: width,
            H: height,
            TILE_SIZE: TILE_SIZE_H_W(w: 64, h: 64),
            whiteBackground: false
        )
        let means3d = gaussRender.get_xyz_from(xyzArray)
        let opacity = gaussRender.get_opacity_from(opacityArray)
        let scales = gaussRender.get_scales_from(scalesArray)
        let rotations = gaussRender.get_rotation_from(rotationArray)
        let shs = gaussRender.get_features_from(
            features_dcArray,
            features_restArray
        )
        let camera = Camera(
            width: width,
            height: height,
            focalX: Float(focalX),
            focalY: Float(focalY),
            c2w: cameraMatrix
        )
        let (
            render,
            _,
            _,
            _,
            _
        ) = gaussRender.forward(
            camera: camera,
            means3d: means3d,
            shs: shs,
            opacity: opacity,
            scales: scales,
            rotations: rotations
        )
        let image = MLX.stopGradient(render)
        eval(image)
        DispatchQueue.main.async {
            self.image = image.toRGBToUIImage()
        }
        MLX.GPU.clearCache()
    }
    
    private func renderGaussianWithMetal() {
        guard let metalRenderer = metalRenderer,
              let xyzArray, let opacityArray, let scalesArray,
              let rotationArray, let features_dcArray, let features_restArray
        else {
            return
        }
        
        let gaussRender = GaussianRenderer(
            active_sh_degree: 4,
            W: width,
            H: height,
            TILE_SIZE: TILE_SIZE_H_W(w: 64, h: 64),
            whiteBackground: false
        )
        
        let means3d = gaussRender.get_xyz_from(xyzArray)
        let opacity = gaussRender.get_opacity_from(opacityArray)
        let scales = gaussRender.get_scales_from(scalesArray)
        let rotations = gaussRender.get_rotation_from(rotationArray)
        let shs = gaussRender.get_features_from(
            features_dcArray,
            features_restArray
        )
        
        let camera = Camera(
            width: width,
            height: height,
            focalX: Float(focalX),
            focalY: Float(focalY),
            c2w: cameraMatrix
        )
        
        if let texture = metalRenderer.render(
            camera: camera,
            means3d: means3d,
            shs: shs,
            opacity: opacity,
            scales: scales,
            rotations: rotations,
            width: width,
            height: height
        ) {
            DispatchQueue.main.async {
                self.image = metalRenderer.textureToUIImage(texture)
            }
        }
    }
    
    private func renderTriangleWithMLX(triangleModel: TriangleModel) {
        guard let triangleRenderer = triangleRenderer else { return }
        
        let camera = Camera(
            width: width,
            height: height,
            focalX: Float(focalX),
            focalY: Float(focalY),
            c2w: cameraMatrix
        )
        
        let (render, _, _, _, _) = triangleRenderer.forward(camera: camera, triangleModel: triangleModel)
        let image = MLX.stopGradient(render)
        eval(image)
        
        DispatchQueue.main.async {
            self.image = image.toRGBToUIImage()
        }
        MLX.GPU.clearCache()
    }
    
    private func renderTriangleWithMetal(triangleModel: TriangleModel) {
        guard let metalTriangleRenderer = metalTriangleRenderer else { return }
        
        let camera = Camera(
            width: width,
            height: height,
            focalX: Float(focalX),
            focalY: Float(focalY),
            c2w: cameraMatrix
        )
        
        if let texture = metalTriangleRenderer.render(
            camera: camera,
            triangleModel: triangleModel,
            width: width,
            height: height
        ) {
            DispatchQueue.main.async {
                self.image = metalTriangleRenderer.textureToUIImage(texture)
            }
        }
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
            VStack(spacing: 10) {
                HStack {
                    Text("Renderer:")
                    Picker("Renderer", selection: $viewModel.useMetalRenderer) {
                        Text("MLX").tag(false)
                        Text("Metal").tag(true)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .frame(width: 150)
                    Spacer()
                }
                
                HStack {
                    Text("Method:")
                    Picker("Method", selection: $viewModel.renderingMethod) {
                        Text("Gaussian").tag(SplattingMethod.gaussian)
                        Text("Triangle").tag(SplattingMethod.triangle)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .frame(width: 150)
                    Spacer()
                }
            }
            .padding(.horizontal)
            
            if let img = viewModel.image {
                Image(uiImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(
                        width: CGFloat(self.width),
                        height: CGFloat(self.height)
                    )
                    .cornerRadius(8)
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
                DispatchQueue.main.async {
                    viewModel.image = nil
                }
            }
        }
        .onChange(of: viewModel.useMetalRenderer) { _ in
            if url != nil && (viewModel.xyzArray != nil || viewModel.triangleModel != nil) {
                renderQueue.async {
                    viewModel.render()
                }
            }
        }
        .onChange(of: viewModel.renderingMethod) { _ in
            if let url = url {
                renderQueue.async {
                    do {
                        try viewModel.load(url: url)
                        viewModel.updateCamera()
                        viewModel.render()
                    } catch {
                        print(error)
                    }
                }
            }
        }
    }
}
struct SnapshotRenderView: View {
    @ObservedObject var asset = TrainOutputAsset.shared
    @State private var selected: TrainSnapshot?
    var body: some View {
        VStack{
            Text("Snapshot Render")
            HStack {
                RenderView(url: .constant(selected?.url), width: 512, height: 512)
                    .frame(width: 512, height: 512)
                Divider()
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
                .frame(minWidth: 200, maxWidth: 300)
                .onAppear {
                    //                asset.loadSnapshotsFromDirectory()
                }
            }
        }
    }
}
