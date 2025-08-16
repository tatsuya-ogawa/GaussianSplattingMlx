import MLX
import simd

// MARK: - Triangle Rasterization Utilities

/// Compute signed distance field for a triangle in 2D
func computeTriangleSignedDistance(
    points: MLXArray,           // [H, W, 2] pixel coordinates
    triangleVertices: MLXArray  // [3, 2] triangle vertices in 2D
) -> MLXArray {
    let v0 = triangleVertices[0]
    let v1 = triangleVertices[1] 
    let v2 = triangleVertices[2]
    
    // Compute edge vectors and normals
    let edge0 = v1 - v0
    let edge1 = v2 - v1
    let edge2 = v0 - v2
    
    // Edge normals (pointing inward)
    let normal0 = MLX.stacked([edge0[1], -edge0[0]], axis: -1)
    let normal1 = MLX.stacked([edge1[1], -edge1[0]], axis: -1)
    let normal2 = MLX.stacked([edge2[1], -edge2[0]], axis: -1)
    
    // Normalize edge normals
    let norm0 = MLX.sqrt(MLX.sum(MLX.square(normal0), axis: -1, keepDims: true))
    let norm1 = MLX.sqrt(MLX.sum(MLX.square(normal1), axis: -1, keepDims: true))
    let norm2 = MLX.sqrt(MLX.sum(MLX.square(normal2), axis: -1, keepDims: true))
    
    let unitNormal0 = normal0 / (norm0 + 1e-8)
    let unitNormal1 = normal1 / (norm1 + 1e-8)
    let unitNormal2 = normal2 / (norm2 + 1e-8)
    
    // Compute signed distances to each edge
    let toV0 = points - v0.expandedDimensions(axes: [0, 1])
    let toV1 = points - v1.expandedDimensions(axes: [0, 1])
    let toV2 = points - v2.expandedDimensions(axes: [0, 1])
    
    let dist0 = MLX.sum(toV0 * unitNormal0.expandedDimensions(axes: [0, 1]), axis: -1)
    let dist1 = MLX.sum(toV1 * unitNormal1.expandedDimensions(axes: [0, 1]), axis: -1)
    let dist2 = MLX.sum(toV2 * unitNormal2.expandedDimensions(axes: [0, 1]), axis: -1)
    
    // For a point to be inside the triangle, all distances should be negative
    // The signed distance field is the maximum of the three edge distances
    return MLX.maximum(MLX.maximum(dist0, dist1), dist2)
}

/// Compute triangle incenter
func computeTriangleIncenter(triangleVertices: MLXArray) -> MLXArray {
    let v0 = triangleVertices[0]
    let v1 = triangleVertices[1]
    let v2 = triangleVertices[2]
    
    // Edge lengths
    let a = MLX.sqrt(MLX.sum(MLX.square(v1 - v2), axis: -1))  // opposite to v0
    let b = MLX.sqrt(MLX.sum(MLX.square(v2 - v0), axis: -1))  // opposite to v1
    let c = MLX.sqrt(MLX.sum(MLX.square(v0 - v1), axis: -1))  // opposite to v2
    
    let perimeter = a + b + c
    
    // Incenter formula: weighted average by opposite edge lengths
    let incenter = (a.expandedDimensions(axes: [-1]) * v0 + 
                   b.expandedDimensions(axes: [-1]) * v1 + 
                   c.expandedDimensions(axes: [-1]) * v2) / perimeter.expandedDimensions(axes: [-1])
    
    return incenter
}

/// Compute window function for triangle splatting
func computeTriangleWindowFunction(
    points: MLXArray,           // [H, W, 2] pixel coordinates
    triangleVertices: MLXArray, // [3, 2] triangle vertices
    smoothness: Float           // σ parameter
) -> MLXArray {
    // Compute signed distance field
    let sdf = computeTriangleSignedDistance(points: points, triangleVertices: triangleVertices)
    
    // Compute incenter and its SDF value
    let incenter = computeTriangleIncenter(triangleVertices: triangleVertices)
    let incenterSdf = computeTriangleSignedDistance(
        points: incenter.expandedDimensions(axes: [0, 1]),
        triangleVertices: triangleVertices
    )[0, 0]
    
    // Window function: ReLU(φ(p)/φ(s))^σ
    let ratio = sdf / (incenterSdf + 1e-8)
    let window = MLX.pow(MLX.maximum(ratio, 0.0), smoothness)
    
    return window
}

/// Project 3D triangle to 2D screen space
func projectTriangleTo2D(
    triangle3D: MLXArray,      // [3, 3] - 3 vertices with 3D coordinates
    viewMatrix: MLXArray,      // [4, 4] view matrix
    projMatrix: MLXArray,      // [4, 4] projection matrix
    width: Int,
    height: Int
) -> (MLXArray, MLXArray) {  // Returns (2D vertices, depths)
    
    // Convert to homogeneous coordinates
    let vertices3D = homogeneous(points: triangle3D)  // [3, 4]
    
    // Transform to view space
    let viewSpace = vertices3D.matmul(viewMatrix)
    let depths = viewSpace[0..., 2]  // Z depths
    
    // Project to clip space
    let clipSpace = viewSpace.matmul(projMatrix)
    
    // Perspective divide
    let ndc = clipSpace[0..., .stride(to: 3)] / clipSpace[0..., .stride(from: -1)].expandedDimensions(axes: [-1])
    
    // Convert to screen coordinates
    let screenX = ((ndc[0..., 0] + 1.0) * Float(width) - 1.0) * 0.5
    let screenY = ((ndc[0..., 1] + 1.0) * Float(height) - 1.0) * 0.5
    
    let screen2D = MLX.stacked([screenX, screenY], axis: -1)
    
    return (screen2D, depths)
}

/// Compute triangle bounding box in screen space
func computeTriangleBounds(triangle2D: MLXArray) -> (minX: Int, maxX: Int, minY: Int, maxY: Int) {
    let minCoords = triangle2D.min(axes: [0])
    let maxCoords = triangle2D.max(axes: [0])
    
    let minX = max(0, Int(floor(minCoords[0].item(Float.self))))
    let maxX = min(Int.max, Int(ceil(maxCoords[0].item(Float.self))))
    let minY = max(0, Int(floor(minCoords[1].item(Float.self))))
    let maxY = min(Int.max, Int(ceil(maxCoords[1].item(Float.self))))
    
    return (minX, maxX, minY, maxY)
}

/// Check if a triangle is front-facing
func isTriangleFrontFacing(triangle2D: MLXArray) -> Bool {
    let v0 = triangle2D[0]
    let v1 = triangle2D[1]
    let v2 = triangle2D[2]
    
    // Compute signed area using cross product
    let edge1 = v1 - v0
    let edge2 = v2 - v0
    
    // 2D cross product (z-component of 3D cross product)
    let signedArea = edge1[0] * edge2[1] - edge1[1] * edge2[0]
    
    return signedArea.item(Float.self) > 0
}

/// Compute barycentric coordinates for a point in a triangle
func computeBarycentricCoordinates(
    point: MLXArray,           // [2] point coordinates
    triangleVertices: MLXArray // [3, 2] triangle vertices
) -> MLXArray {
    let v0 = triangleVertices[0]
    let v1 = triangleVertices[1]
    let v2 = triangleVertices[2]
    
    let v0v1 = v1 - v0
    let v0v2 = v2 - v0
    let v0p = point - v0
    
    let dot00 = MLX.sum(v0v2 * v0v2)
    let dot01 = MLX.sum(v0v2 * v0v1)
    let dot02 = MLX.sum(v0v2 * v0p)
    let dot11 = MLX.sum(v0v1 * v0v1)
    let dot12 = MLX.sum(v0v1 * v0p)
    
    // Compute barycentric coordinates
    let invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    let u = (dot11 * dot02 - dot01 * dot12) * invDenom
    let v = (dot00 * dot12 - dot01 * dot02) * invDenom
    let w = 1.0 - u - v
    
    return MLX.stacked([w, v, u], axis: -1)
}

/// Interpolate triangle vertex attributes using barycentric coordinates
func interpolateTriangleAttributes(
    barycentrics: MLXArray,    // [3] barycentric coordinates
    vertexAttributes: MLXArray // [3, N] attributes at vertices
) -> MLXArray {
    return MLX.sum(barycentrics.expandedDimensions(axes: [-1]) * vertexAttributes, axis: 0)
}

/// Compute triangle area in 2D
func computeTriangleArea2D(triangle2D: MLXArray) -> MLXArray {
    let v0 = triangle2D[0]
    let v1 = triangle2D[1]
    let v2 = triangle2D[2]
    
    let edge1 = v1 - v0
    let edge2 = v2 - v0
    
    // 2D cross product magnitude / 2 = area
    let signedArea = edge1[0] * edge2[1] - edge1[1] * edge2[0]
    return MLX.abs(signedArea) / 2.0
}

/// Cull triangles that are too small or degenerate
func cullDegenerateTriangles(triangle2D: MLXArray, minArea: Float = 0.5) -> Bool {
    let area = computeTriangleArea2D(triangle2D: triangle2D)
    return area.item(Float.self) >= minArea
}

/// Sort triangles by depth for proper alpha blending
func sortTrianglesByDepth(triangleDepths: MLXArray) -> MLXArray {
    return MLX.argSort(triangleDepths)
}