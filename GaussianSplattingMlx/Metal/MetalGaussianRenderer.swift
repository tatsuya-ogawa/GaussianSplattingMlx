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

struct MetalTileInfo {
    var tileCoord: SIMD2<UInt32>
    var gaussianCount: UInt32
    var gaussianOffset: UInt32
}

class MetalGaussianRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    
    // Compute pipelines
    private let projectGaussiansPipeline: MTLComputePipelineState
    private let sortPipeline: MTLComputePipelineState
    private let renderTilesPipeline: MTLComputePipelineState
    private let assignTilesPipeline: MTLComputePipelineState
    private let clearBuffersPipeline: MTLComputePipelineState
    
    // Note: Using compute-based rendering, no traditional render pipeline needed
    
    // Buffers
    private var gaussianBuffer: MTLBuffer?
    private var projectedGaussianBuffer: MTLBuffer?
    private var sortedGaussianBuffer: MTLBuffer?
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
    private let maxGaussians: Int
    private var tileSize: SIMD2<UInt32>
    
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
        
        // Create compute pipelines
        guard let projectFunction = library.makeFunction(name: "projectGaussians"),
              let sortFunction = library.makeFunction(name: "sortGaussiansByDepth"),
              let renderFunction = library.makeFunction(name: "renderTiles"),
              let assignFunction = library.makeFunction(name: "assignGaussiansToTiles"),
              let clearFunction = library.makeFunction(name: "clearBuffers") else {
            print("Failed to create Metal functions")
            return nil
        }
        
        do {
            self.projectGaussiansPipeline = try device.makeComputePipelineState(function: projectFunction)
            self.sortPipeline = try device.makeComputePipelineState(function: sortFunction)
            self.renderTilesPipeline = try device.makeComputePipelineState(function: renderFunction)
            self.assignTilesPipeline = try device.makeComputePipelineState(function: assignFunction)
            self.clearBuffersPipeline = try device.makeComputePipelineState(function: clearFunction)
        } catch {
            print("Failed to create compute pipeline states: \(error)")
            return nil
        }
        
        // Note: Using compute-based rendering, no traditional render pipeline needed
        
        allocateBuffers()
    }
    
    private func allocateBuffers() {
        let gaussianSize = MemoryLayout<MetalGaussianData>.stride
        let projectedSize = MemoryLayout<MetalProjectedGaussian>.stride
        let tileInfoSize = MemoryLayout<MetalTileInfo>.stride
        
        gaussianBuffer = device.makeBuffer(length: gaussianSize * maxGaussians, options: .storageModeShared)
        projectedGaussianBuffer = device.makeBuffer(length: projectedSize * maxGaussians, options: .storageModeShared)
        sortedGaussianBuffer = device.makeBuffer(length: projectedSize * maxGaussians, options: .storageModeShared)
        visibilityMaskBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxGaussians, options: .storageModeShared)
        depthKeysBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * maxGaussians, options: .storageModeShared)
        sortIndicesBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxGaussians, options: .storageModeShared)
        
        // Tile buffers (assuming max 4K resolution with 64x64 tiles = ~4096 tiles)
        let maxTiles = 4096
        tileInfoBuffer = device.makeBuffer(length: tileInfoSize * maxTiles, options: .storageModeShared)
        tileAssignmentsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTiles * 1024, options: .storageModeShared)
        tileCountsBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride * maxTiles, options: .storageModeShared)
    }
    
    func render(
        camera: Camera,
        means3d: MLXArray,
        shs: MLXArray,
        opacity: MLXArray,
        scales: MLXArray,
        rotations: MLXArray,
        width: Int,
        height: Int
    ) -> MTLTexture? {
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
        
        // Convert MLX data to Metal format
        let numGaussians = means3d.shape[0]
        guard numGaussians <= maxGaussians else {
            print("Too many gaussians: \(numGaussians) > \(maxGaussians)")
            return nil
        }
        
        prepareGaussianData(
            means3d: means3d,
            shs: shs,
            opacity: opacity,
            scales: scales,
            rotations: rotations,
            numGaussians: numGaussians
        )
        
        let cameraParams = prepareCameraParams(camera: camera, width: width, height: height)
        
        // Create output texture
        createOutputTexture(width: width, height: height)
        guard let outputTexture = outputTexture else { return nil }
        
        // 1. Project Gaussians to 2D
        projectGaussians(commandBuffer: commandBuffer, cameraParams: cameraParams, numGaussians: numGaussians)
        
        // 2. Sort by depth
        sortGaussiansByDepth(commandBuffer: commandBuffer, numGaussians: numGaussians)
        
        // 3. Assign to tiles
        assignGaussiansToTiles(commandBuffer: commandBuffer, width: width, height: height, numGaussians: numGaussians)
        
        // 4. Render tiles
        renderTiles(commandBuffer: commandBuffer, width: width, height: height)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return outputTexture
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
              let sortedGaussianBuffer = sortedGaussianBuffer else { return }
        
        computeEncoder.setComputePipelineState(sortPipeline)
        computeEncoder.setBuffer(projectedGaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(sortedGaussianBuffer, offset: 0, index: 1)
        
        // Create identity indices for now
        if let sortIndicesBuffer = sortIndicesBuffer {
            let indices = sortIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: numGaussians)
            for i in 0..<numGaussians {
                indices[i] = UInt32(i)
            }
            computeEncoder.setBuffer(sortIndicesBuffer, offset: 0, index: 2)
        }
        
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
              let tileCountsBuffer = tileCountsBuffer else { return }
        
        // Clear tile counts
        memset(tileCountsBuffer.contents(), 0, tileCountsBuffer.length)
        
        computeEncoder.setComputePipelineState(assignTilesPipeline)
        computeEncoder.setBuffer(sortedGaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tileAssignmentsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(tileCountsBuffer, offset: 0, index: 2)
        
        var imageSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        computeEncoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        computeEncoder.setBytes(&tileSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        
        var numGaussiansVar = UInt32(numGaussians)
        computeEncoder.setBytes(&numGaussiansVar, length: MemoryLayout<UInt32>.stride, index: 5)
        
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
              let tileInfoBuffer = tileInfoBuffer,
              let outputImageBuffer = outputImageBuffer else { return }
        
        // Clear output buffer
        memset(outputImageBuffer.contents(), 0, outputImageBuffer.length)
        
        // Prepare tile info (simplified - would need proper tile assignment data)
        let tilesX = (width + Int(tileSize.x) - 1) / Int(tileSize.x)
        let tilesY = (height + Int(tileSize.y) - 1) / Int(tileSize.y)
        
        computeEncoder.setComputePipelineState(renderTilesPipeline)
        computeEncoder.setBuffer(sortedGaussianBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(tileInfoBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(outputImageBuffer, offset: 0, index: 2)
        
        var imageSize = SIMD2<UInt32>(UInt32(width), UInt32(height))
        computeEncoder.setBytes(&imageSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 3)
        computeEncoder.setBytes(&tileSize, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: Int(tileSize.x), height: Int(tileSize.y), depth: 1)
        let threadgroupsPerGrid = MTLSize(width: tilesX, height: tilesY, depth: 1)
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
}

// Extension to convert Metal texture to UIImage
extension MetalGaussianRenderer {
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
