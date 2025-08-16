import MLX
import MLXRandom
import simd
import Foundation

// MARK: - Triangle Splatting Model

class TriangleModel {
    // Core triangle parameters
    var vertices: MLXArray        // [N, 3, 3] - 3 vertices per triangle, each with 3D coordinates
    var opacity: MLXArray         // [N, 1] - opacity per triangle
    var features_dc: MLXArray     // [N, 3] - diffuse color coefficients
    var features_rest: MLXArray   // [N, M, 3] - spherical harmonics coefficients
    var smoothness: MLXArray      // [N, 1] - smoothness parameter σ for window function
    
    // Triangle properties
    let maxTriangles: Int
    let shDegree: Int
    
    // Optimization related
    var xyz_gradient_accum: MLXArray
    var opacity_gradient_accum: MLXArray
    var max_radii2D: MLXArray
    var denom: MLXArray
    
    // Triangle statistics
    var activeTriangles: Int
    
    init(numTriangles: Int, shDegree: Int = 3) {
        self.maxTriangles = numTriangles
        self.shDegree = shDegree
        self.activeTriangles = numTriangles
        
        // Initialize triangle vertices randomly or from point cloud
        self.vertices = MLXArray.zeros([numTriangles, 3, 3])
        self.opacity = inverse_sigmoid(x: MLXArray.ones([numTriangles, 1]) * 0.1)
        self.features_dc = MLXArray.zeros([numTriangles, 3])
        
        let shFeatures = (shDegree + 1) * (shDegree + 1) - 1
        self.features_rest = MLXArray.zeros([numTriangles, shFeatures, 3])
        
        // Initialize smoothness parameter
        self.smoothness = MLXArray.ones([numTriangles, 1]) * 1.0
        
        // Initialize optimization tracking
        self.xyz_gradient_accum = MLXArray.zeros([numTriangles, 3, 3])
        self.opacity_gradient_accum = MLXArray.zeros([numTriangles, 1])
        self.max_radii2D = MLXArray.zeros([numTriangles])
        self.denom = MLXArray.zeros([numTriangles, 1])
    }
    
    convenience init(fromPointCloud points: MLXArray, colors: MLXArray, shDegree: Int = 3) {
        let numPoints = points.shape[0]
        self.init(numTriangles: numPoints / 3, shDegree: shDegree)
        
        // Create triangles from consecutive triplets of points
        for i in 0..<(numPoints / 3) {
            let baseIdx = i * 3
            if baseIdx + 2 < numPoints {
                self.vertices[i, 0] = points[baseIdx]
                self.vertices[i, 1] = points[baseIdx + 1]
                self.vertices[i, 2] = points[baseIdx + 2]
                
                // Set color as average of triangle vertices
                let avgColor = (colors[baseIdx] + colors[baseIdx + 1] + colors[baseIdx + 2]) / 3.0
                self.features_dc[i] = avgColor
            }
        }
    }
    
    // MARK: - Triangle Properties
    
    func getTriangleCenters() -> MLXArray {
        // Compute centroid of each triangle
        return vertices.mean(axis: 1)
    }
    
    func getTriangleAreas() -> MLXArray {
        // Compute area of each triangle using cross product
        let v0 = vertices[0..., 0]
        let v1 = vertices[0..., 1] 
        let v2 = vertices[0..., 2]
        
        let edge1 = v1 - v0
        let edge2 = v2 - v0
        
        // Cross product magnitude / 2 = area
        let cross = MLX.cross(edge1, edge2)
        let areas = MLX.sqrt(MLX.sum(MLX.square(cross), axis: -1)) / 2.0
        
        return areas
    }
    
    func getOpacity() -> MLXArray {
        return MLX.sigmoid(opacity)
    }
    
    func getSmoothness() -> MLXArray {
        return MLX.exp(smoothness)
    }
    
    // MARK: - Training Operations
    
    func resetOpacityToAlpha(alpha: Float) {
        let newOpacity = inverse_sigmoid(x: MLXArray.ones(opacity.shape) * alpha)
        opacity = newOpacity
    }
    
    func pruneTriangles(mask: MLXArray) {
        // Remove triangles based on boolean mask
        vertices = vertices[mask]
        opacity = opacity[mask]
        features_dc = features_dc[mask]
        features_rest = features_rest[mask]
        smoothness = smoothness[mask]
        
        xyz_gradient_accum = xyz_gradient_accum[mask]
        opacity_gradient_accum = opacity_gradient_accum[mask]
        max_radii2D = max_radii2D[mask]
        denom = denom[mask]
        
        activeTriangles = vertices.shape[0]
    }
    
    func catTriangles(newVertices: MLXArray, newOpacity: MLXArray, newFeaturesDc: MLXArray, 
                     newFeaturesRest: MLXArray, newSmoothness: MLXArray) {
        // Add new triangles to the model
        vertices = MLX.concatenated([vertices, newVertices], axis: 0)
        opacity = MLX.concatenated([opacity, newOpacity], axis: 0)
        features_dc = MLX.concatenated([features_dc, newFeaturesDc], axis: 0)
        features_rest = MLX.concatenated([features_rest, newFeaturesRest], axis: 0)
        smoothness = MLX.concatenated([smoothness, newSmoothness], axis: 0)
        
        // Update optimization tracking
        let newGradAccum = MLXArray.zeros([newVertices.shape[0], 3, 3])
        let newOpacityGradAccum = MLXArray.zeros([newOpacity.shape[0], 1])
        let newMaxRadii = MLXArray.zeros([newVertices.shape[0]])
        let newDenom = MLXArray.zeros([newVertices.shape[0], 1])
        
        xyz_gradient_accum = MLX.concatenated([xyz_gradient_accum, newGradAccum], axis: 0)
        opacity_gradient_accum = MLX.concatenated([opacity_gradient_accum, newOpacityGradAccum], axis: 0)
        max_radii2D = MLX.concatenated([max_radii2D, newMaxRadii], axis: 0)
        denom = MLX.concatenated([denom, newDenom], axis: 0)
        
        activeTriangles = vertices.shape[0]
    }
    
    func densifyAndSplit(grads: MLXArray, gradThreshold: Float, sceneExtent: Float, N: Int = 2) {
        // Split triangles with high gradients
        let padGrads = MLX.sum(grads, axes: [1, 2])  // Sum gradients over vertices and coordinates
        let selectedPts = padGrads .>= gradThreshold
        let selectedIndices = conditionToIndices(condition: selectedPts)
        
        if selectedIndices.shape[0] == 0 { return }
        
        // Create new triangles by splitting existing ones
        let selectedVertices = vertices[selectedIndices]
        let selectedOpacity = opacity[selectedIndices]
        let selectedFeaturesDc = features_dc[selectedIndices]
        let selectedFeaturesRest = features_rest[selectedIndices]
        let selectedSmoothness = smoothness[selectedIndices]
        
        // Generate N new triangles per selected triangle by perturbing vertices
        var newVertices: [MLXArray] = []
        var newOpacity: [MLXArray] = []
        var newFeaturesDc: [MLXArray] = []
        var newFeaturesRest: [MLXArray] = []
        var newSmoothness: [MLXArray] = []
        
        let splitScale = sceneExtent / 10.0
        
        for _ in 0..<N {
            let noise = MLXRandom.normal([selectedVertices.shape[0], 3, 3]) * splitScale
            newVertices.append(selectedVertices + noise)
            newOpacity.append(selectedOpacity)
            newFeaturesDc.append(selectedFeaturesDc)
            newFeaturesRest.append(selectedFeaturesRest)
            newSmoothness.append(selectedSmoothness)
        }
        
        if !newVertices.isEmpty {
            catTriangles(
                newVertices: MLX.concatenated(newVertices, axis: 0),
                newOpacity: MLX.concatenated(newOpacity, axis: 0),
                newFeaturesDc: MLX.concatenated(newFeaturesDc, axis: 0),
                newFeaturesRest: MLX.concatenated(newFeaturesRest, axis: 0),
                newSmoothness: MLX.concatenated(newSmoothness, axis: 0)
            )
        }
    }
    
    func densifyAndClone(grads: MLXArray, gradThreshold: Float, sceneExtent: Float) {
        // Clone small triangles with high gradients
        let areas = getTriangleAreas()
        let avgArea = areas.mean()
        
        let padGrads = MLX.sum(grads, axes: [1, 2])
        let selectedPts = (padGrads .>= gradThreshold) & (areas .<= avgArea)
        let selectedIndices = conditionToIndices(condition: selectedPts)
        
        if selectedIndices.shape[0] == 0 { return }
        
        let selectedVertices = vertices[selectedIndices]
        let selectedOpacity = opacity[selectedIndices]
        let selectedFeaturesDc = features_dc[selectedIndices]
        let selectedFeaturesRest = features_rest[selectedIndices]
        let selectedSmoothness = smoothness[selectedIndices]
        
        catTriangles(
            newVertices: selectedVertices,
            newOpacity: selectedOpacity,
            newFeaturesDc: selectedFeaturesDc,
            newFeaturesRest: selectedFeaturesRest,
            newSmoothness: selectedSmoothness
        )
    }
    
    func pruneTriangles(minOpacity: Float, maxScreenSize: Int, cameras: [Camera]) {
        // Remove triangles with low opacity or that are too large
        let opacityMask = getOpacity()[0..., 0] .> minOpacity
        let tooBigMask = max_radii2D .<= Float(maxScreenSize)
        let pruneMask = opacityMask & tooBigMask
        
        pruneTriangles(mask: pruneMask)
    }
    
    // MARK: - Model Management
    
    func oneUpSHDegree() {
        // Increase spherical harmonics degree (not implemented for triangles yet)
        // In practice, triangles use a fixed SH degree
    }
    
    func save(path: String) throws {
        // Save triangle model to file
        let data: [String: Any] = [
            "vertices": vertices.asArray(Float.self),
            "opacity": opacity.asArray(Float.self),
            "features_dc": features_dc.asArray(Float.self),
            "features_rest": features_rest.asArray(Float.self),
            "smoothness": smoothness.asArray(Float.self),
            "active_triangles": activeTriangles,
            "sh_degree": shDegree
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: data, options: .prettyPrinted)
        try jsonData.write(to: URL(fileURLWithPath: path))
    }
    
    func load(path: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        
        let verticesData = json["vertices"] as! [Float]
        let opacityData = json["opacity"] as! [Float]
        let featuresDcData = json["features_dc"] as! [Float]
        let featuresRestData = json["features_rest"] as! [Float]
        let smoothnessData = json["smoothness"] as! [Float]
        
        activeTriangles = json["active_triangles"] as! Int
        
        // Reconstruct MLXArrays
        vertices = MLXArray(verticesData).reshaped([activeTriangles, 3, 3])
        opacity = MLXArray(opacityData).reshaped([activeTriangles, 1])
        features_dc = MLXArray(featuresDcData).reshaped([activeTriangles, 3])
        features_rest = MLXArray(featuresRestData).reshaped([activeTriangles, (shDegree + 1) * (shDegree + 1) - 1, 3])
        smoothness = MLXArray(smoothnessData).reshaped([activeTriangles, 1])
        
        // Reset optimization tracking
        xyz_gradient_accum = MLXArray.zeros([activeTriangles, 3, 3])
        opacity_gradient_accum = MLXArray.zeros([activeTriangles, 1])
        max_radii2D = MLXArray.zeros([activeTriangles])
        denom = MLXArray.zeros([activeTriangles, 1])
    }
}