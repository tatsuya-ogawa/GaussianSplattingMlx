import MLX
import simd

/// Camera class matching Python implementation
class Camera {
  let imageWidth: Int
  let imageHeight: Int
  let focalX: MLXArray
  let focalY: MLXArray
  let FoVx: MLXArray
  let FoVy: MLXArray
  let worldViewTransform: MLXArray
  let projectionMatrix: MLXArray
  let cameraCenter: SIMD3<Double>

  init(
    width: Int,
    height: Int,
    intrinsic: MLXArray,
    c2w: MLXArray,
    znear: Double = 0.1,
    zfar: Double = 100.0
  ) {
    self.imageWidth = width
    self.imageHeight = height
    self.focalX = intrinsic[0][0]
    self.focalY = intrinsic[1][1]
    self.FoVx = focal2fov(focal: focalX, pixels: Double(width))
    self.FoVy = focal2fov(focal: focalY, pixels: Double(height))
    let worldViewTransform = simd_double4x4.fromMlxArray(c2w).inverse.transpose
    let projectionMatrix = getProjectionMatrix(
      znear: znear, zfar: zfar, fovX: Double(FoVx.item(Float.self)),
      fovY: Double(FoVy.item(Float.self)))
    self.worldViewTransform = worldViewTransform.toMLXArray()
    self.projectionMatrix = projectionMatrix.transpose.toMLXArray()
    let invWV = worldViewTransform.inverse.transpose
    self.cameraCenter = SIMD3<Double>(invWV[3][0], invWV[3][1], invWV[3][2])
  }
}

/// Convert fov to focal length
func fov2focal(fov: Double, pixels: Double) -> Double {
  return pixels / (2.0 * tan(fov / 2.0))
}
func fov2focal(fov: MLXArray, pixels: Double) -> MLXArray {
  return pixels / (2.0 * tan(fov / 2.0))
}

/// Convert focal length to fov
func focal2fov(focal: Double, pixels: Double) -> Double {
  return 2.0 * atan(pixels / (2.0 * focal))
}
func focal2fov(focal: MLXArray, pixels: Double) -> MLXArray {
  return 2.0 * atan(pixels / (2.0 * focal))
}

/// Build projection matrix matching Python getProjectionMatrix
func getProjectionMatrix(znear: Double, zfar: Double, fovX: Double, fovY: Double)
  -> matrix_double4x4
{
  let tanHalfY = tan(fovY / 2.0)
  let tanHalfX = tan(fovX / 2.0)
  let top = tanHalfY * znear
  let bottom = -top
  let right = tanHalfX * znear
  let left = -right

  var P = matrix_double4x4()
  P.columns = (
    SIMD4<Double>(2 * znear / (right - left), 0, 0, 0),
    SIMD4<Double>(0, 2 * znear / (top - bottom), 0, 0),
    SIMD4<Double>(
      (right + left) / (right - left), (top + bottom) / (top - bottom), zfar / (zfar - znear), 1),
    SIMD4<Double>(0, 0, -znear * zfar / (zfar - znear), 0)
  )
  return P
}
