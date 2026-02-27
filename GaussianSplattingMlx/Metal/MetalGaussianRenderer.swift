import Metal
import MetalKit
import simd
import MLX

struct MetalGaussianData {
    var position: SIMD3<Float>
    var rotation: SIMD4<Float>
    var scale: SIMD3<Float>
    var opacity: Float
    var sh_dc: SIMD3<Float>
    var sh_rest: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>,
                  SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>,
                  SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)
}

struct MetalCameraParams {
    var viewMatrix: simd_float4x4
    var projMatrix: simd_float4x4
    var cameraCenter: SIMD3<Float>
    var focalX: Float
    var focalY: Float
    var fovX: Float
    var fovY: Float
    var width: Int32
    var height: Int32
}

struct MetalProjectedGaussian {
    var mean2d: SIMD2<Float>
    var cov2d: SIMD3<Float>
    var color: SIMD3<Float>
    var alpha: Float
    var depth: Float
    var originalIndex: UInt32
}

final class MetalGaussianRenderer: NSObject, MTKViewDelegate {
    private struct FullscreenVertex {
        var position: SIMD2<Float>
        var texCoord: SIMD2<Float>
    }
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    
    // Compute pipelines
    private let projectGaussiansPipeline: MTLComputePipelineState
    private let sortPipeline: MTLComputePipelineState
    private let renderTilesPipeline: MTLComputePipelineState
    private let assignTilesPipeline: MTLComputePipelineState
    private let clearBuffersPipeline: MTLComputePipelineState
    
    // Radix sort pipelines
    private let prepareRadixKeysPipeline: MTLComputePipelineState
    private let radixCountPipeline: MTLComputePipelineState
    private let radixPrefixSumPipeline: MTLComputePipelineState
    private let radixScatterPipeline: MTLComputePipelineState
    
    // Buffers
    private var gaussianBuffer: MTLBuffer?
    private var projectedGaussianBuffer: MTLBuffer?
    private var sortedGaussianBuffer: MTLBuffer?
    private var visibilityMaskBuffer: MTLBuffer?
    private var depthKeysBuffer: MTLBuffer?      // radix keys A (ping)
    private var sortIndicesBuffer: MTLBuffer?     // radix values A (ping)
    private var radixKeysTempBuffer: MTLBuffer?   // radix keys B (pong)
    private var radixValuesTempBuffer: MTLBuffer?  // radix values B (pong)
    private var radixHistogramBuffer: MTLBuffer?   // 256 uints
    private var tileAssignmentsBuffer: MTLBuffer?
    private var tileCountsBuffer: MTLBuffer?
    
    // Textures
    private var outputTexture: MTLTexture?
    
    // Display
    private weak var attachedView: MTKView?
    private let displayPipeline: MTLRenderPipelineState
    private let displayVertexBuffer: MTLBuffer
    
    // Configuration
    private let maxGaussians: Int
    private var tileSize: SIMD2<UInt32>
    private var loadedGaussianCount: Int = 0
    private var preparedSortCount: Int = 0
    
    var metalDevice: MTLDevice { device }
    
    init?(maxGaussians: Int = 1000000, tileSize: SIMD2<UInt32> = SIMD2<UInt32>(64, 64)) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not supported")
            return nil
        }
        
        self.device = device
        self.maxGaussians = maxGaussians
        self.tileSize = tileSize
        
        guard let commandQueue = device.makeCommandQueue() else {
            print("Failed to create command queue")
            return nil
        }
        self.commandQueue = commandQueue
        
        guard let library = device.makeDefaultLibrary() else {
            print("Failed to create Metal library")
            return nil
        }
        self.library = library
        
        guard let displayVertexFunction = library.makeFunction(name: "vertexShader"),
              let displayFragmentFunction = library.makeFunction(name: "fragmentShader")
        else {
            print("Failed to create display Metal functions")
            return nil
        }
        
        // Create compute pipelines
        guard let projectFunction = library.makeFunction(name: "projectGaussians"),
              let sortFunction = library.makeFunction(name: "sortGaussiansByDepth"),
              let renderFunction = library.makeFunction(name: "renderTiles"),
              let assignFunction = library.makeFunction(name: "assignGaussiansToTiles"),
              let clearFunction = library.makeFunction(name: "clearBuffers"),
              let prepareRadixKeysFunction = library.makeFunction(name: "prepareRadixSortKeys"),
              let radixCountFunction = library.makeFunction(name: "radixSortCount"),
              let radixPrefixSumFunction = library.makeFunction(name: "radixSortPrefixSum"),
              let radixScatterFunction = library.makeFunction(name: "radixSortScatter") else {
            print("Failed to create Metal functions")
            return nil
        }
        
        do {
            self.projectGaussiansPipeline = try device.makeComputePipelineState(function: projectFunction)
            self.sortPipeline = try device.makeComputePipelineState(function: sortFunction)
            self.renderTilesPipeline = try device.makeComputePipelineState(function: renderFunction)
            self.assignTilesPipeline = try device.makeComputePipelineState(function: assignFunction)
            self.clearBuffersPipeline = try device.makeComputePipelineState(function: clearFunction)
            self.prepareRadixKeysPipeline = try device.makeComputePipelineState(function: prepareRadixKeysFunction)
            self.radixCountPipeline = try device.makeComputePipelineState(function: radixCountFunction)
            self.radixPrefixSumPipeline = try device.makeComputePipelineState(function: radixPrefixSumFunction)
            self.radixScatterPipeline = try device.makeComputePipelineState(function: radixScatterFunction)
            
            let vertexDescriptor = MTLVertexDescriptor()
            vertexDescriptor.attributes[0].format = .float2
            vertexDescriptor.attributes[0].offset = 0
            vertexDescriptor.attributes[0].bufferIndex = 0
            vertexDescriptor.attributes[1].format = .float2
            vertexDescriptor.attributes[1].offset = MemoryLayout<SIMD2<Float>>.stride
            vertexDescriptor.attributes[1].bufferIndex = 0
            vertexDescriptor.layouts[0].stride = MemoryLayout<FullscreenVertex>.stride
            
            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.vertexFunction = displayVertexFunction
            pipelineDescriptor.fragmentFunction = displayFragmentFunction
            pipelineDescriptor.vertexDescriptor = vertexDescriptor
            pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            self.displayPipeline = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            
            let vertices: [FullscreenVertex] = [
                FullscreenVertex(position: SIMD2<Float>(-1, -1), texCoord: SIMD2<Float>(0, 1)),
                FullscreenVertex(position: SIMD2<Float>(1, -1), texCoord: SIMD2<Float>(1, 1)),
                FullscreenVertex(position: SIMD2<Float>(-1, 1), texCoord: SIMD2<Float>(0, 0)),
                FullscreenVertex(position: SIMD2<Float>(1, 1), texCoord: SIMD2<Float>(1, 0)),
            ]
            let vertexLength = MemoryLayout<FullscreenVertex>.stride * vertices.count
            guard let vertexBuffer = device.makeBuffer(bytes: vertices, length: vertexLength, options: .storageModeShared) else {
                print("Failed to create display vertex buffer")
                return nil
            }
            self.displayVertexBuffer = vertexBuffer
        } catch {
            print("Failed to create compute pipeline states: \(error)")
            return nil
        }
        
        super.init()
        allocateBuffers()
    }
    
    private func allocateBuffers() {
        let gaussianSize = MemoryLayout<MetalGaussianData>.stride
        let projectedSize = MemoryLayout<MetalProjectedGaussian>.stride
        gaussianBuffer = device.makeBuffer(length: gaussianSize * maxGaussians, options: .storageModeShared)
        projectedGaussianBuffer = device.makeBuffer(length: projectedSize * maxGaussians, options: .storageModeShared)
        sortedGaussianBuffer = device.makeBuffer(length: projectedSize * maxGaussians, options: .storageModeShared)
        visibilityMaskBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxGaussians, options: .storageModeShared)
        depthKeysBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxGaussians, options: .storageModeShared)
        sortIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxGaussians, options: .storageModeShared)
        radixKeysTempBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxGaussians, options: .storageModeShared)
        radixValuesTempBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxGaussians, options: .storageModeShared)
        radixHistogramBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * 256, options: .storageModeShared)
        
        // Tile buffers (assuming max 4K resolution with 64x64 tiles = ~4096 tiles)
        let maxTiles = 4096
        let maxGaussiansPerTile = 16384
        tileAssignmentsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTiles * maxGaussiansPerTile, options: .storageModeShared)
        tileCountsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTiles, options: .storageModeShared)
    }
    
    func attach(to view: MTKView) {
        if view.device !== device {
            view.device = device
        }
        view.delegate = self
        view.framebufferOnly = false
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.enableSetNeedsDisplay = true
        view.isPaused = true
        attachedView = view
    }
    
    func detach() {
        if attachedView?.delegate === self {
            attachedView?.delegate = nil
        }
        attachedView = nil
    }
    
    func clearFrame() {
        outputTexture = nil
        DispatchQueue.main.async { [weak self] in
            self?.attachedView?.setNeedsDisplay()
        }
    }
    
    func clearScene() {
        loadedGaussianCount = 0
        preparedSortCount = 0
        clearFrame()
    }
    
    @discardableResult
    func uploadScene(
        means3d: MLXArray,
        shs: MLXArray,
        opacity: MLXArray,
        scales: MLXArray,
        rotations: MLXArray
    ) -> Bool {
        let numGaussians = means3d.shape[0]
        guard numGaussians <= maxGaussians else {
            print("Too many gaussians: \(numGaussians) > \(maxGaussians)")
            return false
        }
        
        prepareGaussianData(
            means3d: means3d,
            shs: shs,
            opacity: opacity,
            scales: scales,
            rotations: rotations,
            numGaussians: numGaussians
        )
        prepareIdentitySortIndices(numGaussians: numGaussians)
        loadedGaussianCount = numGaussians
        return true
    }
    
    func render(
        camera: Camera,
        width: Int,
        height: Int
    ) {
        let numGaussians = loadedGaussianCount
        guard numGaussians > 0 else { return }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }
        
        let cameraParams = prepareCameraParams(camera: camera, width: width, height: height)
        
        // Create output texture
        createOutputTexture(width: width, height: height)
        guard outputTexture != nil else { return }
        
        // 1. Project Gaussians to 2D
        projectGaussians(commandBuffer: commandBuffer, cameraParams: cameraParams, numGaussians: numGaussians)
        
        // 2. GPU Radix Sort by depth (front-to-back)
        gpuRadixSort(commandBuffer: commandBuffer, numGaussians: numGaussians)
        
        // 3. Reorder projected gaussians using sorted indices
        sortGaussiansByDepth(commandBuffer: commandBuffer, numGaussians: numGaussians)
        
        // 4. Assign to tiles
        assignGaussiansToTiles(commandBuffer: commandBuffer, width: width, height: height, numGaussians: numGaussians)
        
        // 5. Render tiles
        renderTiles(commandBuffer: commandBuffer, width: width, height: height)
        
        commandBuffer.addCompletedHandler { [weak self] _ in
            DispatchQueue.main.async {
                self?.attachedView?.setNeedsDisplay()
            }
        }
        commandBuffer.commit()
    }
    
    private func prepareIdentitySortIndices(numGaussians: Int) {
        guard preparedSortCount != numGaussians,
              let sortIndicesBuffer = sortIndicesBuffer else { return }
        
        let indices = sortIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: numGaussians)
        for i in 0..<numGaussians {
            indices[i] = UInt32(i)
        }
        preparedSortCount = numGaussians
    }
    
    /// GPU 8-bit radix sort on projected depth (4 passes, front-to-back).
    /// After completion, `sortIndicesBuffer` contains indices sorted by ascending depth.
    private func gpuRadixSort(commandBuffer: MTLCommandBuffer, numGaussians: Int) {
        guard let projectedGaussianBuffer = projectedGaussianBuffer,
              let depthKeysBuffer = depthKeysBuffer,
              let sortIndicesBuffer = sortIndicesBuffer,
              let radixKeysTempBuffer = radixKeysTempBuffer,
              let radixValuesTempBuffer = radixValuesTempBuffer,
              let radixHistogramBuffer = radixHistogramBuffer else { return }
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numGaussians + 255) / 256, height: 1, depth: 1
        )
        var numElementsVar = UInt32(numGaussians)
        let histogramSize = MemoryLayout<UInt32>.stride * 256
        
        // Step 1: Prepare sortable keys and initial indices from projected depth
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(prepareRadixKeysPipeline)
            encoder.setBuffer(projectedGaussianBuffer, offset: 0, index: 0)
            encoder.setBuffer(depthKeysBuffer, offset: 0, index: 1)
            encoder.setBuffer(sortIndicesBuffer, offset: 0, index: 2)
            encoder.setBytes(&numElementsVar, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        // Ping-pong buffers: pass 0 reads A writes B, pass 1 reads B writes A, ...
        let keysBuffers = [depthKeysBuffer, radixKeysTempBuffer]
        let valuesBuffers = [sortIndicesBuffer, radixValuesTempBuffer]
        
        // Step 2: Four 8-bit radix sort passes
        for pass in 0..<4 {
            let bitOffset = UInt32(pass * 8)
            let srcIdx = pass % 2
            let dstIdx = 1 - srcIdx
            
            // 2a. Clear histogram
            if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
                blitEncoder.fill(buffer: radixHistogramBuffer, range: 0..<histogramSize, value: 0)
                blitEncoder.endEncoding()
            }
            
            // 2b. Count digit occurrences
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                var bitOffsetVar = bitOffset
                encoder.setComputePipelineState(radixCountPipeline)
                encoder.setBuffer(keysBuffers[srcIdx], offset: 0, index: 0)
                encoder.setBuffer(radixHistogramBuffer, offset: 0, index: 1)
                encoder.setBytes(&bitOffsetVar, length: MemoryLayout<UInt32>.stride, index: 2)
                encoder.setBytes(&numElementsVar, length: MemoryLayout<UInt32>.stride, index: 3)
                encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }
            
            // 2c. Exclusive prefix sum
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(radixPrefixSumPipeline)
                encoder.setBuffer(radixHistogramBuffer, offset: 0, index: 0)
                encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
            }
            
            // 2d. Scatter to sorted positions
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                var bitOffsetVar = bitOffset
                encoder.setComputePipelineState(radixScatterPipeline)
                encoder.setBuffer(keysBuffers[srcIdx], offset: 0, index: 0)
                encoder.setBuffer(valuesBuffers[srcIdx], offset: 0, index: 1)
                encoder.setBuffer(keysBuffers[dstIdx], offset: 0, index: 2)
                encoder.setBuffer(valuesBuffers[dstIdx], offset: 0, index: 3)
                encoder.setBuffer(radixHistogramBuffer, offset: 0, index: 4)
                encoder.setBytes(&bitOffsetVar, length: MemoryLayout<UInt32>.stride, index: 5)
                encoder.setBytes(&numElementsVar, length: MemoryLayout<UInt32>.stride, index: 6)
                encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }
        }
        // After 4 passes (even count), results are back in keysBuffers[0]/valuesBuffers[0]
        // = depthKeysBuffer / sortIndicesBuffer. Ready for sortGaussiansByDepth.
    }
    
    private func logFrameStats(numGaussians: Int, width: Int, height: Int) {
        guard let visibilityMaskBuffer = visibilityMaskBuffer,
              let tileCountsBuffer = tileCountsBuffer else { return }
        
        let visibility = visibilityMaskBuffer.contents().bindMemory(to: UInt32.self, capacity: numGaussians)
        var visibleCount = 0
        for i in 0..<numGaussians {
            visibleCount += visibility[i] != 0 ? 1 : 0
        }
        
        let tilesX = (width + Int(tileSize.x) - 1) / Int(tileSize.x)
        let tilesY = (height + Int(tileSize.y) - 1) / Int(tileSize.y)
        let numTiles = min(tilesX * tilesY, tileCountsBuffer.length / MemoryLayout<UInt32>.stride)
        
        let tileCounts = tileCountsBuffer.contents().bindMemory(to: UInt32.self, capacity: numTiles)
        var activeTiles = 0
        var maxTileCount = 0
        for i in 0..<numTiles {
            let c = Int(tileCounts[i])
            if c > 0 {
                activeTiles += 1
                maxTileCount = max(maxTileCount, c)
            }
        }
        
        if visibleCount == 0 || activeTiles == 0 {
            print("MetalGaussianRenderer: visibleGaussians=\(visibleCount), activeTiles=\(activeTiles), maxTileCount=\(maxTileCount)")
        }
    }
    
    private func prepareGaussianData(
        means3d: MLXArray,
        shs: MLXArray,
        opacity: MLXArray,
        scales: MLXArray,
        rotations: MLXArray,
        numGaussians: Int
    ) {
        guard let gaussianBuffer = gaussianBuffer else { return }
        
        let gaussianData = gaussianBuffer.contents().bindMemory(to: MetalGaussianData.self, capacity: numGaussians)
        
        let means3dArray = means3d.asArray(Float.self)
        let shsArray = shs.asArray(Float.self)
        let opacityArray = opacity.asArray(Float.self)
        let scalesArray = scales.asArray(Float.self)
        let rotationsArray = rotations.asArray(Float.self)
        
        for i in 0..<numGaussians {
            let baseIdx = i * 3
            let shBaseIdx = i * shs.shape[1] * 3
            let quatIdx = i * 4
            
            gaussianData[i].position = SIMD3<Float>(
                means3dArray[baseIdx],
                means3dArray[baseIdx + 1],
                means3dArray[baseIdx + 2]
            )
            
            gaussianData[i].scale = SIMD3<Float>(
                scalesArray[baseIdx],
                scalesArray[baseIdx + 1],
                scalesArray[baseIdx + 2]
            )
            
            gaussianData[i].rotation = SIMD4<Float>(
                rotationsArray[quatIdx],
                rotationsArray[quatIdx + 1],
                rotationsArray[quatIdx + 2],
                rotationsArray[quatIdx + 3]
            )
            
            gaussianData[i].opacity = opacityArray[i]
            
            // SH coefficients
            gaussianData[i].sh_dc = SIMD3<Float>(
                shsArray[shBaseIdx],
                shsArray[shBaseIdx + 1],
                shsArray[shBaseIdx + 2]
            )
            
            // Initialize sh_rest with remaining coefficients (simplified)
            gaussianData[i].sh_rest = (
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0),
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0),
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0),
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0),
                SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0)
            )
        }
    }
    
    private func prepareCameraParams(camera: Camera, width: Int, height: Int) -> MetalCameraParams {
        let viewMatrixArray = camera.worldViewTransform.asArray(Float.self)
        let projMatrixArray = camera.projectionMatrix.asArray(Float.self)
        
        let viewMatrix = simd_float4x4(
            SIMD4<Float>(viewMatrixArray[0], viewMatrixArray[1], viewMatrixArray[2], viewMatrixArray[3]),
            SIMD4<Float>(viewMatrixArray[4], viewMatrixArray[5], viewMatrixArray[6], viewMatrixArray[7]),
            SIMD4<Float>(viewMatrixArray[8], viewMatrixArray[9], viewMatrixArray[10], viewMatrixArray[11]),
            SIMD4<Float>(viewMatrixArray[12], viewMatrixArray[13], viewMatrixArray[14], viewMatrixArray[15])
        )
        
        let projMatrix = simd_float4x4(
            SIMD4<Float>(projMatrixArray[0], projMatrixArray[1], projMatrixArray[2], projMatrixArray[3]),
            SIMD4<Float>(projMatrixArray[4], projMatrixArray[5], projMatrixArray[6], projMatrixArray[7]),
            SIMD4<Float>(projMatrixArray[8], projMatrixArray[9], projMatrixArray[10], projMatrixArray[11]),
            SIMD4<Float>(projMatrixArray[12], projMatrixArray[13], projMatrixArray[14], projMatrixArray[15])
        )
        
        return MetalCameraParams(
            viewMatrix: viewMatrix,
            projMatrix: projMatrix,
            cameraCenter: SIMD3<Float>(
                Float(camera.cameraCenter.x),
                Float(camera.cameraCenter.y),
                Float(camera.cameraCenter.z)
            ),
            focalX: camera.focalX.item(Float.self),
            focalY: camera.focalY.item(Float.self),
            fovX: camera.FoVx.item(Float.self),
            fovY: camera.FoVy.item(Float.self),
            width: Int32(width),
            height: Int32(height)
        )
    }
    
    private func createOutputTexture(width: Int, height: Int) {
        if let outputTexture,
           outputTexture.width == width,
           outputTexture.height == height {
            return
        }
        
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .private
        
        outputTexture = device.makeTexture(descriptor: descriptor)
    }
    
    private func projectGaussians(commandBuffer: MTLCommandBuffer, cameraParams: MetalCameraParams, numGaussians: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let gaussianBuffer = gaussianBuffer,
              let projectedGaussianBuffer = projectedGaussianBuffer,
              let visibilityMaskBuffer = visibilityMaskBuffer else { return }
        
        computeEncoder.setComputePipelineState(projectGaussiansPipeline)
        computeEncoder.setBuffer(gaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(projectedGaussianBuffer, offset: 0, index: 1)
        
        var cameraParamsCopy = cameraParams
        computeEncoder.setBytes(&cameraParamsCopy, length: MemoryLayout<MetalCameraParams>.stride, index: 2)
        computeEncoder.setBuffer(visibilityMaskBuffer, offset: 0, index: 3)
        
        var numGaussiansVar = UInt32(numGaussians)
        computeEncoder.setBytes(&numGaussiansVar, length: MemoryLayout<UInt32>.stride, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numGaussians + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func sortGaussiansByDepth(commandBuffer: MTLCommandBuffer, numGaussians: Int) {
        // Simplified sorting - in practice you'd implement radix sort
        // For now, just copy the projected gaussians to sorted buffer
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let projectedGaussianBuffer = projectedGaussianBuffer,
              let sortedGaussianBuffer = sortedGaussianBuffer,
              let sortIndicesBuffer = sortIndicesBuffer else { return }
        
        computeEncoder.setComputePipelineState(sortPipeline)
        computeEncoder.setBuffer(projectedGaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedGaussianBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(sortIndicesBuffer, offset: 0, index: 2)
        
        var numGaussiansVar = UInt32(numGaussians)
        computeEncoder.setBytes(&numGaussiansVar, length: MemoryLayout<UInt32>.stride, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numGaussians + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func assignGaussiansToTiles(commandBuffer: MTLCommandBuffer, width: Int, height: Int, numGaussians: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let sortedGaussianBuffer = sortedGaussianBuffer,
              let tileAssignmentsBuffer = tileAssignmentsBuffer,
              let tileCountsBuffer = tileCountsBuffer,
              let visibilityMaskBuffer = visibilityMaskBuffer else { return }
        
        // Clear tile counts
        memset(tileCountsBuffer.contents(), 0, tileCountsBuffer.length)
        
        computeEncoder.setComputePipelineState(assignTilesPipeline)
        computeEncoder.setBuffer(sortedGaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tileAssignmentsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(tileCountsBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(visibilityMaskBuffer, offset: 0, index: 3)
        
        var imageSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        computeEncoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        computeEncoder.setBytes(&tileSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 5)
        
        var numGaussiansVar = UInt32(numGaussians)
        computeEncoder.setBytes(&numGaussiansVar, length: MemoryLayout<UInt32>.stride, index: 6)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numGaussians + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func renderTiles(commandBuffer: MTLCommandBuffer, width: Int, height: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let sortedGaussianBuffer = sortedGaussianBuffer,
              let tileAssignmentsBuffer = tileAssignmentsBuffer,
              let tileCountsBuffer = tileCountsBuffer,
              let outputTexture = outputTexture else { return }
        
        computeEncoder.setComputePipelineState(renderTilesPipeline)
        computeEncoder.setBuffer(sortedGaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tileAssignmentsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(tileCountsBuffer, offset: 0, index: 2)
        computeEncoder.setTexture(outputTexture, index: 0)
        
        var imageSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        computeEncoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        computeEncoder.setBytes(&tileSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        
        // Use a safe 2D threadgroup size for all Apple GPUs (<= 1024 threads total).
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerGrid = MTLSize(width: width, height: height, depth: 1)
        
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    func draw(in view: MTKView) {
        guard let renderPassDescriptor = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
        else {
            return
        }
        
        if let outputTexture = outputTexture {
            renderEncoder.setRenderPipelineState(displayPipeline)
            renderEncoder.setVertexBuffer(displayVertexBuffer, offset: 0, index: 0)
            renderEncoder.setFragmentTexture(outputTexture, index: 0)
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        renderEncoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}
