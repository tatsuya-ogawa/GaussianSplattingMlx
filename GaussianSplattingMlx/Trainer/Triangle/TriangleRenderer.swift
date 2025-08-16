import MLX
import simd

// MARK: - Triangle Splatting Renderer

class TriangleRenderer {
    let debug: Bool
    let active_sh_degree: Int
    let W: Int
    let H: Int
    let pix_coord: MLXArray
    let whiteBackground: Bool
    let TILE_SIZE: TILE_SIZE_H_W
    
    init(
        active_sh_degree: Int,
        W: Int,
        H: Int,
        TILE_SIZE: TILE_SIZE_H_W,
        whiteBackground: Bool
    ) {
        self.active_sh_degree = active_sh_degree
        self.debug = false
        self.whiteBackground = whiteBackground
        self.TILE_SIZE = TILE_SIZE
        self.W = W
        self.H = H
        self.pix_coord = createMeshGrid(shape: [H, W])
    }
    
    /// Render a single triangle tile
    func renderTriangleTile(
        h: Int,
        w: Int,
        tileSize: TILE_SIZE_H_W,
        triangleVertices2D: MLXArray,  // [N, 3, 2] triangles in 2D
        triangleColors: MLXArray,      // [N, 3] triangle colors
        triangleOpacity: MLXArray,     // [N, 1] triangle opacity
        triangleDepths: MLXArray,      // [N] triangle depths
        triangleSmoothness: MLXArray,  // [N, 1] smoothness parameters
        skipThreshold: Int = 0
    ) -> (MLXArray, MLXArray, MLXArray) {
        
        let tileSizeRest = TILE_SIZE_H_W(
            w: Swift.min(self.W - w, tileSize.w),
            h: Swift.min(self.H - h, tileSize.h)
        )
        
        // Get pixel coordinates for this tile
        let tile_coord = MLX.stopGradient(
            self.pix_coord[
                .stride(from: h, to: h + tileSizeRest.h),
                .stride(from: w, to: w + tileSizeRest.w)
            ]
        ) // [tile_h, tile_w, 2]
        
        let numTriangles = triangleVertices2D.shape[0]
        if numTriangles <= skipThreshold {
            Logger.shared.debug("skip tile - no triangles")
            return (
                self.whiteBackground
                    ? MLXArray.ones([tileSizeRest.h, tileSizeRest.w, 3])
                    : MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 3]),
                MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 1]),
                MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 1])
            )
        }
        
        // Sort triangles by depth (back-to-front)
        let sortedIndices = sortTrianglesByDepth(triangleDepths: triangleDepths)
        let sortedVertices2D = triangleVertices2D[sortedIndices]
        let sortedColors = triangleColors[sortedIndices]
        let sortedOpacity = triangleOpacity[sortedIndices]
        let sortedSmoothness = triangleSmoothness[sortedIndices]
        
        // Initialize output buffers
        var tileColor = MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 3])
        var tileDepth = MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 1])
        var tileAlpha = MLXArray.zeros([tileSizeRest.h, tileSizeRest.w, 1])
        
        // Render each triangle
        for triIdx in 0..<sortedVertices2D.shape[0] {
            let triangle2D = sortedVertices2D[triIdx]  // [3, 2]
            let color = sortedColors[triIdx]          // [3]
            let opacity = sortedOpacity[triIdx][0]    // scalar
            let smoothness = sortedSmoothness[triIdx][0]  // scalar
            
            // Check if triangle is front-facing
            if !isTriangleFrontFacing(triangle2D: triangle2D) {
                continue
            }
            
            // Cull degenerate triangles
            if !cullDegenerateTriangles(triangle2D: triangle2D) {
                continue
            }
            
            // Compute triangle bounding box
            let bounds = computeTriangleBounds(triangle2D: triangle2D)
            let minX = max(0, bounds.minX - w)
            let maxX = min(tileSizeRest.w - 1, bounds.maxX - w)
            let minY = max(0, bounds.minY - h)
            let maxY = min(tileSizeRest.h - 1, bounds.maxY - h)
            
            if minX >= maxX || minY >= maxY {
                continue  // Triangle doesn't intersect tile
            }
            
            // Render triangle pixels
            for y in minY...maxY {
                for x in minX...maxX {
                    let pixelCoord = MLX.stacked([Float(x + w), Float(y + h)], axis: -1)
                    
                    // Compute window function weight
                    let weight = computeTriangleWindowFunction(
                        points: pixelCoord.expandedDimensions(axes: [0, 1]),
                        triangleVertices: triangle2D,
                        smoothness: smoothness.item(Float.self)
                    )[0, 0]
                    
                    if weight.item(Float.self) > 1e-6 {
                        let alpha = weight * MLX.clip(opacity, max: 0.99)
                        let currentAlpha = tileAlpha[y, x, 0]
                        let T = 1.0 - currentAlpha
                        
                        // Alpha blending
                        let blendedAlpha = currentAlpha + T * alpha
                        let blendedColor = (tileColor[y, x] * currentAlpha.expandedDimensions(axes: [-1]) + 
                                          color.expandedDimensions(axes: [0, 1]) * T * alpha) / blendedAlpha.expandedDimensions(axes: [-1])
                        
                        tileColor[y, x] = blendedColor[0]
                        tileAlpha[y, x, 0] = blendedAlpha
                    }
                }
            }
        }
        
        // Add background
        if whiteBackground {
            tileColor = tileColor + (1.0 - tileAlpha) * 1.0
        }
        
        return (tileColor, tileDepth, tileAlpha)
    }
    
    /// Main triangle rendering function
    func render(
        camera: Camera,
        triangleModel: TriangleModel
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visibility_filter: MLXArray,
        radii: MLXArray
    ) {
        
        Logger.shared.debug("triangle render start")
        
        // Project triangles to 2D
        var triangles2D: [MLXArray] = []
        var triangleColors: [MLXArray] = []
        var triangleOpacities: [MLXArray] = []
        var triangleDepths: [MLXArray] = []
        var triangleSmoothness: [MLXArray] = []
        var visibilityMask: [Bool] = []
        
        let numTriangles = triangleModel.activeTriangles
        
        for i in 0..<numTriangles {
            let triangle3D = triangleModel.vertices[i]  // [3, 3]
            
            // Project to 2D
            let (triangle2D, depths) = projectTriangleTo2D(
                triangle3D: triangle3D,
                viewMatrix: camera.worldViewTransform,
                projMatrix: camera.projectionMatrix,
                width: camera.imageWidth,
                height: camera.imageHeight
            )
            
            // Check visibility (simple frustum culling)
            let minDepth = depths.min()
            let maxDepth = depths.max()
            
            if minDepth.item(Float.self) < 0.1 || maxDepth.item(Float.self) > 100.0 {
                visibilityMask.append(false)
                continue
            }
            
            // Check if triangle is on screen
            let bounds = computeTriangleBounds(triangle2D: triangle2D)
            if bounds.maxX < 0 || bounds.minX >= camera.imageWidth || 
               bounds.maxY < 0 || bounds.minY >= camera.imageHeight {
                visibilityMask.append(false)
                continue
            }
            
            // Compute triangle color using SH
            let triangleCenter = triangleModel.getTriangleCenters()[i..<(i+1)]
            let triangleFeaturesDc = triangleModel.features_dc[i..<(i+1)]
            let triangleFeaturesRest = triangleModel.features_rest[i..<(i+1)]
            
            let color = build_color(
                means3d: triangleCenter,
                shs: MLX.concatenated([triangleFeaturesDc.expandedDimensions(axes: [1]), triangleFeaturesRest], axis: 1),
                camera: camera,
                activeShDegree: self.active_sh_degree
            )[0]  // Get first (and only) color
            
            triangles2D.append(triangle2D)
            triangleColors.append(color)
            triangleOpacities.append(triangleModel.getOpacity()[i..<(i+1)])
            triangleDepths.append(depths.mean())  // Use average depth
            triangleSmoothness.append(triangleModel.getSmoothness()[i..<(i+1)])
            visibilityMask.append(true)
        }
        
        if triangles2D.isEmpty {
            Logger.shared.debug("no visible triangles")
            return (
                self.whiteBackground
                    ? MLXArray.ones([H, W, 3])
                    : MLXArray.zeros([H, W, 3]),
                MLXArray.zeros([H, W, 1]),
                MLXArray.zeros([H, W, 1]),
                MLXArray.zeros([numTriangles]),
                MLXArray.zeros([numTriangles])
            )
        }
        
        // Stack triangle data
        let allTriangles2D = MLX.stacked(triangles2D, axis: 0)
        let allColors = MLX.stacked(triangleColors, axis: 0)
        let allOpacities = MLX.stacked(triangleOpacities, axis: 0)
        let allDepths = MLX.stacked(triangleDepths, axis: 0)
        let allSmoothness = MLX.stacked(triangleSmoothness, axis: 0)
        
        // Initialize render buffers
        let render_color = MLXArray.zeros([H, W, 3])
        let render_depth = MLXArray.zeros([H, W, 1])
        let render_alpha = MLXArray.zeros([H, W, 1])
        
        // Render tiles
        for h in stride(from: 0, to: camera.imageHeight, by: TILE_SIZE.h) {
            for w in stride(from: 0, to: camera.imageWidth, by: TILE_SIZE.w) {
                Logger.shared.debug("rendering tile (\(h), \(w))")
                
                let (tile_color, tile_depth, tile_alpha) = renderTriangleTile(
                    h: h,
                    w: w,
                    tileSize: TILE_SIZE,
                    triangleVertices2D: allTriangles2D,
                    triangleColors: allColors,
                    triangleOpacity: allOpacities,
                    triangleDepths: allDepths,
                    triangleSmoothness: allSmoothness
                )
                
                let endH = min(h + TILE_SIZE.h, H)
                let endW = min(w + TILE_SIZE.w, W)
                
                render_color[
                    .stride(from: h, to: endH),
                    .stride(from: w, to: endW)
                ] = tile_color[
                    .stride(to: endH - h),
                    .stride(to: endW - w)
                ]
                
                render_depth[
                    .stride(from: h, to: endH),
                    .stride(from: w, to: endW)
                ] = tile_depth[
                    .stride(to: endH - h),
                    .stride(to: endW - w)
                ]
                
                render_alpha[
                    .stride(from: h, to: endH),
                    .stride(from: w, to: endW)
                ] = tile_alpha[
                    .stride(to: endH - h),
                    .stride(to: endW - w)
                ]
            }
        }
        
        // Create visibility filter and radii (simplified for triangles)
        let visibility_filter = MLXArray(visibilityMask.map { $0 ? Float(1) : Float(0) })
        let radii = MLXArray.ones([numTriangles]) * 10.0  // Simplified radii for triangles
        
        Logger.shared.debug("triangle render complete")
        
        return (render_color, render_depth, render_alpha, visibility_filter, radii)
    }
    
    /// Forward pass with triangle model
    func forward(
        camera: Camera,
        triangleModel: TriangleModel
    ) -> (
        render: MLXArray,
        depth: MLXArray,
        alpha: MLXArray,
        visibility_filter: MLXArray,
        radii: MLXArray
    ) {
        return render(camera: camera, triangleModel: triangleModel)
    }
}

// MARK: - Triangle Renderer Extensions

extension TriangleRenderer {
    
    /// Render triangle mesh to image for debugging
    func renderTriangleMesh(
        camera: Camera,
        triangleModel: TriangleModel,
        wireframe: Bool = false
    ) -> MLXArray {
        
        let (render, _, _, _, _) = forward(camera: camera, triangleModel: triangleModel)
        
        if wireframe {
            // TODO: Add wireframe rendering
            return render
        }
        
        return render
    }
    
    /// Export triangle mesh to OBJ format
    func exportTriangleMesh(
        triangleModel: TriangleModel,
        filename: String
    ) throws {
        let vertices = triangleModel.vertices
        let colors = triangleModel.features_dc
        
        var objContent = "# Triangle Splatting Mesh Export\n"
        objContent += "# Vertices: \(triangleModel.activeTriangles * 3)\n"
        objContent += "# Faces: \(triangleModel.activeTriangles)\n\n"
        
        // Write vertices
        for i in 0..<triangleModel.activeTriangles {
            for j in 0..<3 {
                let vertex = vertices[i, j]
                let color = colors[i]
                
                let x = vertex[0].item(Float.self)
                let y = vertex[1].item(Float.self)
                let z = vertex[2].item(Float.self)
                
                let r = color[0].item(Float.self)
                let g = color[1].item(Float.self)
                let b = color[2].item(Float.self)
                
                objContent += "v \(x) \(y) \(z) \(r) \(g) \(b)\n"
            }
        }
        
        objContent += "\n"
        
        // Write faces
        for i in 0..<triangleModel.activeTriangles {
            let v1 = i * 3 + 1
            let v2 = i * 3 + 2
            let v3 = i * 3 + 3
            
            objContent += "f \(v1) \(v2) \(v3)\n"
        }
        
        try objContent.write(toFile: filename, atomically: true, encoding: .utf8)
    }
}