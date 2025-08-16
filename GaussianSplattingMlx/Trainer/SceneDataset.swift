import MLX
import Foundation

// MARK: - Scene Dataset Wrapper

class SceneDataset {
    let cameras: [Camera]
    let images: [MLXArray]
    let imageNames: [String]
    
    init(data: SceneData) {
        var cameras: [Camera] = []
        var images: [MLXArray] = []
        var imageNames: [String] = []
        
        let numCameras = data.image.shape[0]
        
        for i in 0..<numCameras {
            // Create camera from data
            let camera = Camera(
                width: data.Ws[i].item(Int.self),
                height: data.Hs[i].item(Int.self),
                intrinsic: data.Ks[i],
                c2w: data.c2ws[i],
                znear: 0.1,
                zfar: 100.0
            )
            
            cameras.append(camera)
            images.append(data.image[i])
            imageNames.append("image_\(i)")
        }
        
        self.cameras = cameras
        self.images = images
        self.imageNames = imageNames
    }
    
    convenience init(cameras: [Camera], images: [MLXArray], imageNames: [String]? = nil) {
        // For manual construction
        self.cameras = cameras
        self.images = images
        self.imageNames = imageNames ?? (0..<cameras.count).map { "image_\($0)" }
    }
    
    var count: Int {
        return cameras.count
    }
    
    func randomSample() -> (Camera, MLXArray, String) {
        let index = Int.random(in: 0..<count)
        return (cameras[index], images[index], imageNames[index])
    }
    
    func sample(at index: Int) -> (Camera, MLXArray, String) {
        return (cameras[index], images[index], imageNames[index])
    }
    
    // Split dataset into training and test sets
    func split(testRatio: Float = 0.1) -> (train: SceneDataset, test: SceneDataset) {
        let testCount = max(1, Int(Float(count) * testRatio))
        let trainCount = count - testCount
        
        let indices = Array(0..<count).shuffled()
        let trainIndices = Array(indices[..<trainCount])
        let testIndices = Array(indices[trainCount...])
        
        let trainCameras = trainIndices.map { cameras[$0] }
        let trainImages = trainIndices.map { images[$0] }
        let trainNames = trainIndices.map { imageNames[$0] }
        
        let testCameras = testIndices.map { cameras[$0] }
        let testImages = testIndices.map { images[$0] }
        let testNames = testIndices.map { imageNames[$0] }
        
        return (
            train: SceneDataset(cameras: trainCameras, images: trainImages, imageNames: trainNames),
            test: SceneDataset(cameras: testCameras, images: testImages, imageNames: testNames)
        )
    }
}