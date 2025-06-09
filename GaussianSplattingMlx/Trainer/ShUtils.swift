import Foundation
import MLX

let C0: Float = 0.28209479177387814
let C1: Float = 0.4886025119029199
let C2: [Float] = [
  1.0925484305920792,
  -1.0925484305920792,
  0.31539156525252005,
  -1.0925484305920792,
  0.5462742152960396,
]
let C3: [Float] = [
  -0.5900435899266435,
  2.890611442640554,
  -0.4570457994644658,
  0.3731763325901154,
  -0.4570457994644658,
  1.445305721320277,
  -0.5900435899266435,
]
let C4: [Float] = [
  2.5033429417967046,
  -1.7701307697799304,
  0.9461746957575601,
  -0.6690465435572892,
  0.10578554691520431,
  -0.6690465435572892,
  0.47308734787878004,
  -1.7701307697799304,
  0.6258357354491761,
]

func RGB2SH(rgb: MLXArray) -> MLXArray {
  return (rgb - 0.5) / C0
}

func SH2RGB(sh: MLXArray) -> MLXArray {
  return sh * C0 + 0.5
}

func evalSh(deg: Int, sh: MLXArray, dirs: MLXArray) -> MLXArray {
  let coeff = pow(Double(deg) + 1, 2)
  if sh.shape[sh.shape.count - 1] < Int(coeff) {
    fatalError("coeff:\(coeff) is too large")
  }
  var result = C0 * sh[.ellipsis, 0]
  if deg > 0 {
    let x = dirs[.ellipsis, .stride(from: 0, to: 1)]
    let y = dirs[.ellipsis, .stride(from: 1, to: 2)]
    let z = dirs[.ellipsis, .stride(from: 2, to: 3)]
    result =
      (result - C1 * y * sh[.ellipsis, 1] + C1 * z * sh[.ellipsis, 2] - C1
        * x * sh[.ellipsis, 3])

    if deg > 1 {
      let xx = x * x
      let yy = y * y
      let zz = z * z
      let xy = x * y
      let yz = y * z
      let xz = x * z
      result =
        (result + C2[0] * xy * sh[.ellipsis, 4] + C2[1] * yz
          * sh[.ellipsis, 5] + C2[2] * (2.0 * zz - xx - yy)
          * sh[.ellipsis, 6] + C2[3] * xz * sh[.ellipsis, 7] + C2[4]
          * (xx - yy) * sh[.ellipsis, 8])
      if deg > 2 {
        result =
          (result + C3[0] * y * (3 * xx - yy) * sh[.ellipsis, 9] + C3[
            1
          ] * xy * z * sh[.ellipsis, 10] + C3[2] * y
            * (4 * zz - xx - yy) * sh[.ellipsis, 11] + C3[3] * z
            * (2 * zz - 3 * xx - 3 * yy) * sh[.ellipsis, 12] + C3[4]
            * x * (4 * zz - xx - yy) * sh[.ellipsis, 13] + C3[5] * z
            * (xx - yy) * sh[.ellipsis, 14] + C3[6] * x
            * (xx - 3 * yy) * sh[.ellipsis, 15])
        if deg > 3 {
          result =
            (result + C4[0] * xy * (xx - yy) * sh[.ellipsis, 16]
              + C4[1] * yz * (3 * xx - yy) * sh[.ellipsis, 17]
              + C4[2] * xy * (7 * zz - 1) * sh[.ellipsis, 18]
              + C4[3] * yz * (7 * zz - 3) * sh[.ellipsis, 19]
              + C4[4] * (zz * (35 * zz - 30) + 3)
              * sh[.ellipsis, 20] + C4[5] * xz * (7 * zz - 3)
              * sh[.ellipsis, 21] + C4[6] * (xx - yy)
              * (7 * zz - 1) * sh[.ellipsis, 22] + C4[7] * xz
              * (xx - 3 * yy) * sh[.ellipsis, 23] + C4[8]
              * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
              * sh[.ellipsis, 24])
        }
      }
    }
  }
  return result
}
