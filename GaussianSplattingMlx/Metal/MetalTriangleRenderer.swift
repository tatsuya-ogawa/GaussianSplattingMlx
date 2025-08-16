import Metal
import MetalKit
import simd
import MLX

// MARK: - Metal Triangle Data Structures

struct MetalTriangleData {
    var vertex0: SIMD3<Float>
    var vertex1: SIMD3<Float>
    var vertex2: SIMD3<Float>
    var color: SIMD3<Float>
    var opacity: Float
    var smoothness: Float
    var sh_dc: SIMD3<Float>
    var sh_rest: (SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>,
                  SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>,
                  SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)
}

struct MetalProjectedTriangle {
    var vertex0_2d: SIMD2<Float>
    var vertex1_2d: SIMD2<Float>
    var vertex2_2d: SIMD2<Float>
    var color: SIMD3<Float>
    var alpha: Float
    var depth: Float
    var smoothness: Float
    var originalIndex: UInt32
    var boundingBoxMin: SIMD2<Float>
    var boundingBoxMax: SIMD2<Float>
}

struct MetalTriangleTileInfo {
    var tileCoord: SIMD2<UInt32>
    var triangleCount: UInt32
    var triangleOffset: UInt32
}


// MARK: - Metal Triangle Renderer

class MetalTriangleRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    
    // Compute pipelines
    private let projectTrianglesPipeline: MTLComputePipelineState
    private let sortPipeline: MTLComputePipelineState
    private let assignTilesPipeline: MTLComputePipelineState
    private let renderTilesPipeline: MTLComputePipelineState
    private let clearBuffersPipeline: MTLComputePipelineState
    
    // Render pipelines
    private let renderPipelineState: MTLRenderPipelineState
    
    // Buffers
    private var triangleBuffer: MTLBuffer?
    private var projectedTriangleBuffer: MTLBuffer?
    private var sortedTriangleBuffer: MTLBuffer?
    private var visibilityMaskBuffer: MTLBuffer?
    private var depthKeysBuffer: MTLBuffer?
    private var sortIndicesBuffer: MTLBuffer?
    private var tileInfoBuffer: MTLBuffer?
    private var tileAssignmentsBuffer: MTLBuffer?
    private var tileCountsBuffer: MTLBuffer?
    private var outputImageBuffer: MTLBuffer?
    
    // Textures
    private var outputTexture: MTLTexture?
    
    // Configuration
    private let maxTriangles: Int
    private let tileSize: SIMD2<UInt32>
    
    init?(maxTriangles: Int = 500000, tileSize: SIMD2<UInt32> = SIMD2<UInt32>(64, 64)) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not supported")
            return nil
        }
        
        self.device = device
        self.maxTriangles = maxTriangles
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
        
        // Create compute pipelines
        guard let projectFunction = library.makeFunction(name: "projectTriangles"),
              let sortFunction = library.makeFunction(name: "sortTrianglesByDepth"),
              let assignFunction = library.makeFunction(name: "assignTrianglesToTiles"),
              let renderFunction = library.makeFunction(name: "renderTriangleTiles"),
              let clearFunction = library.makeFunction(name: "clearTriangleBuffers") else {
            print("Failed to create Metal functions")
            return nil
        }
        
        do {
            self.projectTrianglesPipeline = try device.makeComputePipelineState(function: projectFunction)
            self.sortPipeline = try device.makeComputePipelineState(function: sortFunction)
            self.assignTilesPipeline = try device.makeComputePipelineState(function: assignFunction)
            self.renderTilesPipeline = try device.makeComputePipelineState(function: renderFunction)
            self.clearBuffersPipeline = try device.makeComputePipelineState(function: clearFunction)
        } catch {
            print("Failed to create compute pipeline states: \(error)")
            return nil
        }
        
        // Create render pipeline
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexShader")
        renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentShader")
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        do {
            self.renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        } catch {
            print("Failed to create render pipeline state: \(error)")
            return nil
        }
        
        allocateBuffers()
    }
    
    private func allocateBuffers() {
        let triangleSize = MemoryLayout<MetalTriangleData>.stride
        let projectedSize = MemoryLayout<MetalProjectedTriangle>.stride
        let tileInfoSize = MemoryLayout<MetalTriangleTileInfo>.stride
        
        triangleBuffer = device.makeBuffer(length: triangleSize * maxTriangles, options: .storageModeShared)
        projectedTriangleBuffer = device.makeBuffer(length: projectedSize * maxTriangles, options: .storageModeShared)
        sortedTriangleBuffer = device.makeBuffer(length: projectedSize * maxTriangles, options: .storageModeShared)
        visibilityMaskBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTriangles, options: .storageModeShared)
        depthKeysBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * maxTriangles, options: .storageModeShared)
        sortIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTriangles, options: .storageModeShared)
        
        // Tile buffers (assuming max 4K resolution with 64x64 tiles = ~4096 tiles)
        let maxTiles = 4096
        tileInfoBuffer = device.makeBuffer(length: tileInfoSize * maxTiles, options: .storageModeShared)
        tileAssignmentsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTiles * 1024, options: .storageModeShared)
        tileCountsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTiles, options: .storageModeShared)
    }
    
    func render(
        camera: Camera,
        triangleModel: TriangleModel,
        width: Int,
        height: Int
    ) -> MTLTexture? {
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
        
        // Convert triangle model to Metal format
        let numTriangles = triangleModel.activeTriangles
        guard numTriangles <= maxTriangles else {
            print("Too many triangles: \(numTriangles) > \(maxTriangles)")
            return nil
        }
        
        prepareTriangleData(triangleModel: triangleModel, numTriangles: numTriangles)
        let cameraParams = prepareCameraParams(camera: camera, width: width, height: height)
        
        // Create output texture
        createOutputTexture(width: width, height: height)
        guard let outputTexture = outputTexture else { return nil }
        
        // 1. Project triangles to 2D
        projectTriangles(commandBuffer: commandBuffer, cameraParams: cameraParams, numTriangles: numTriangles)
        
        // 2. Sort by depth
        sortTrianglesByDepth(commandBuffer: commandBuffer, numTriangles: numTriangles)
        
        // 3. Assign to tiles
        assignTrianglesToTiles(commandBuffer: commandBuffer, width: width, height: height, numTriangles: numTriangles)
        
        // 4. Render tiles
        renderTiles(commandBuffer: commandBuffer, width: width, height: height)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return outputTexture
    }
    
    private func prepareTriangleData(triangleModel: TriangleModel, numTriangles: Int) {
        guard let triangleBuffer = triangleBuffer else { return }
        
        let triangleData = triangleBuffer.contents().bindMemory(to: MetalTriangleData.self, capacity: numTriangles)
        
        let verticesArray = triangleModel.vertices.asArray(Float.self)
        let shsArray = triangleModel.features_dc.asArray(Float.self)
        let opacityArray = triangleModel.opacity.asArray(Float.self)
        let smoothnessArray = triangleModel.smoothness.asArray(Float.self)
        
        for i in 0..<numTriangles {
            // Extract triangle vertices
            let v0BaseIdx = i * 9     // 3 vertices * 3 coordinates
            let v1BaseIdx = v0BaseIdx + 3
            let v2BaseIdx = v1BaseIdx + 3
            
            triangleData[i].vertex0 = SIMD3<Float>(
                verticesArray[v0BaseIdx],
                verticesArray[v0BaseIdx + 1],
                verticesArray[v0BaseIdx + 2]
            )
            
            triangleData[i].vertex1 = SIMD3<Float>(
                verticesArray[v1BaseIdx],
                verticesArray[v1BaseIdx + 1],
                verticesArray[v1BaseIdx + 2]
            )
            
            triangleData[i].vertex2 = SIMD3<Float>(
                verticesArray[v2BaseIdx],
                verticesArray[v2BaseIdx + 1],
                verticesArray[v2BaseIdx + 2]
            )
            
            let colorBaseIdx = i * 3
            triangleData[i].color = SIMD3<Float>(
                shsArray[colorBaseIdx],
                shsArray[colorBaseIdx + 1],
                shsArray[colorBaseIdx + 2]
            )
            
            triangleData[i].opacity = opacityArray[i]
            triangleData[i].smoothness = smoothnessArray[i]
            
            // SH coefficients
            triangleData[i].sh_dc = SIMD3<Float>(
                shsArray[colorBaseIdx],
                shsArray[colorBaseIdx + 1],
                shsArray[colorBaseIdx + 2]
            )
            
            // Initialize sh_rest with zeros (simplified)
            triangleData[i].sh_rest = (
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
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        descriptor.storageMode = .shared
        
        outputTexture = device.makeTexture(descriptor: descriptor)
        
        // Also create output buffer for compute shader
        outputImageBuffer = device.makeBuffer(
            length: MemoryLayout<SIMD4<Float>>.stride * width * height,
            options: .storageModeShared
        )
    }
    
    private func projectTriangles(commandBuffer: MTLCommandBuffer, cameraParams: MetalCameraParams, numTriangles: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let triangleBuffer = triangleBuffer,
              let projectedTriangleBuffer = projectedTriangleBuffer,
              let visibilityMaskBuffer = visibilityMaskBuffer else { return }
        
        computeEncoder.setComputePipelineState(projectTrianglesPipeline)
        computeEncoder.setBuffer(triangleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(projectedTriangleBuffer, offset: 0, index: 1)
        
        var cameraParamsCopy = cameraParams
        computeEncoder.setBytes(&cameraParamsCopy, length: MemoryLayout<MetalCameraParams>.stride, index: 2)
        computeEncoder.setBuffer(visibilityMaskBuffer, offset: 0, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numTriangles + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func sortTrianglesByDepth(commandBuffer: MTLCommandBuffer, numTriangles: Int) {
        // Simplified sorting - in practice you'd implement radix sort or parallel sorting
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let projectedTriangleBuffer = projectedTriangleBuffer,
              let sortedTriangleBuffer = sortedTriangleBuffer else { return }
        
        computeEncoder.setComputePipelineState(sortPipeline)
        computeEncoder.setBuffer(projectedTriangleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedTriangleBuffer, offset: 0, index: 1)
        
        // Create identity indices for now (should implement proper sorting)
        if let sortIndicesBuffer = sortIndicesBuffer {
            let indices = sortIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: numTriangles)
            for i in 0..<numTriangles {
                indices[i] = UInt32(i)
            }
            computeEncoder.setBuffer(sortIndicesBuffer, offset: 0, index: 2)
        }
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numTriangles + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func assignTrianglesToTiles(commandBuffer: MTLCommandBuffer, width: Int, height: Int, numTriangles: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let sortedTriangleBuffer = sortedTriangleBuffer,
              let tileAssignmentsBuffer = tileAssignmentsBuffer,
              let tileCountsBuffer = tileCountsBuffer else { return }
        
        // Clear tile counts
        memset(tileCountsBuffer.contents(), 0, tileCountsBuffer.length)
        
        computeEncoder.setComputePipelineState(assignTilesPipeline)
        computeEncoder.setBuffer(sortedTriangleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tileAssignmentsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(tileCountsBuffer, offset: 0, index: 2)
        
        var imageSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        var tileSizeCopy = tileSize
        computeEncoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        computeEncoder.setBytes(&tileSizeCopy, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        
        var numTrianglesVar = UInt32(numTriangles)
        computeEncoder.setBytes(&numTrianglesVar, length: MemoryLayout<UInt32>.stride, index: 5)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (numTriangles + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
    
    private func renderTiles(commandBuffer: MTLCommandBuffer, width: Int, height: Int) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let sortedTriangleBuffer = sortedTriangleBuffer,
              let tileInfoBuffer = tileInfoBuffer,
              let outputImageBuffer = outputImageBuffer else { return }
        
        // Clear output buffer
        memset(outputImageBuffer.contents(), 0, outputImageBuffer.length)
        
        // Prepare tile info (simplified - would need proper tile assignment data)
        let tilesX = (width + Int(tileSize.x) - 1) / Int(tileSize.x)
        let tilesY = (height + Int(tileSize.y) - 1) / Int(tileSize.y)
        
        computeEncoder.setComputePipelineState(renderTilesPipeline)
        computeEncoder.setBuffer(sortedTriangleBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tileInfoBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(outputImageBuffer, offset: 0, index: 2)
        
        var imageSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        var tileSizeCopy = tileSize
        computeEncoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        computeEncoder.setBytes(&tileSizeCopy, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: Int(tileSize.x), height: Int(tileSize.y), depth: 1)
        let threadgroupsPerGrid = MTLSize(width: tilesX, height: tilesY, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
}

// MARK: - Extension for texture conversion

extension MetalTriangleRenderer {
    func textureToUIImage(_ texture: MTLTexture) -> UIImage? {
        let width = texture.width
        let height = texture.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let imageByteCount = bytesPerRow * height
        
        let imageBytes = UnsafeMutableRawPointer.allocate(byteCount: imageByteCount, alignment: 1)
        defer { imageBytes.deallocate() }
        
        texture.getBytes(imageBytes, bytesPerRow: bytesPerRow, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: imageBytes, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cgImage = context.makeImage() else {
            return nil
        }
        
        return UIImage(cgImage: cgImage)
    }
}