import MLX
import MLXFFT
import MLXFast
import MLXLinalg
import MLXNN
import MLXRandom
import simd

private let conditionToIndicesKernel = MLXFast.metalKernel(
    name: "condition_to_indices",
    inputNames: ["mask"],
    outputNames: ["indices", "count"],
    source: """
        uint elem = thread_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint n = mask_shape[0];

        // Must match conditionToIndices() threadGroup.x = 256.
        threadgroup int localIndices[256];
        threadgroup int localCount;
        threadgroup int blockBase;

        int localValue = -1;
        if ((elem < n) && (mask[elem] != 0)) {
            localValue = (int)elem;
        }
        localIndices[lid] = localValue;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            int compactCount = 0;
            for (uint i = 0; i < threads_per_threadgroup.x; ++i) {
                int v = localIndices[i];
                if (v >= 0) {
                    localIndices[compactCount] = v;
                    compactCount += 1;
                }
            }
            localCount = compactCount;
            blockBase = atomic_fetch_add_explicit(
                &count[0], compactCount, memory_order_relaxed
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < (uint)localCount) {
            int dst = blockBase + (int)lid;
            atomic_store_explicit(&indices[dst], localIndices[lid], memory_order_relaxed);
        }
        """,
    ensureRowContiguous: false,
    atomicOutputs: true
)

func inverse_sigmoid(x: MLXArray) -> MLXArray {
    return MLX.log(x / (1 - x))
}

func homogeneous(points: MLXArray) -> MLXArray {
    return MLX.concatenated(
        [points, MLXArray.ones(like: points[.ellipsis, .stride(to: 1)])],
        axis: -1
    )
}

func build_rotation(quaternion: MLXArray)
    -> MLXArray
{
    let norm =
        (quaternion[.ellipsis, 0] * quaternion[.ellipsis, 0] + quaternion[
            .ellipsis,
            1
        ] * quaternion[.ellipsis, 1] + quaternion[.ellipsis, 2]
        * quaternion[.ellipsis, 2] + quaternion[.ellipsis, 3]
        * quaternion[.ellipsis, 3]).sqrt()
    let safeNorm = MLX.maximum(norm, 1e-8)
    let q = quaternion / safeNorm.expandedDimensions(axes: [-1])
    let R = MLXArray.zeros([q.shape[0], 3, 3], dtype: .float32)

    let r = q[.ellipsis, 0]
    let x = q[.ellipsis, 1]
    let y = q[.ellipsis, 2]
    let z = q[.ellipsis, 3]

    R[.ellipsis, 0, 0] = 1 - 2 * (y * y + z * z)
    R[.ellipsis, 0, 1] = 2 * (x * y - r * z)
    R[.ellipsis, 0, 2] = 2 * (x * z + r * y)
    R[.ellipsis, 1, 0] = 2 * (x * y + r * z)
    R[.ellipsis, 1, 1] = 1 - 2 * (x * x + z * z)
    R[.ellipsis, 1, 2] = 2 * (y * z - r * x)
    R[.ellipsis, 2, 0] = 2 * (x * z - r * y)
    R[.ellipsis, 2, 1] = 2 * (y * z + r * x)
    R[.ellipsis, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
}

func build_scaling_rotation(s: MLXArray, r: MLXArray) -> MLXArray {
    let L = MLXArray.zeros([s.shape[0], 3, 3], dtype: .float32)
    let R = build_rotation(quaternion: r)

    L[.ellipsis, 0, 0] = s[.ellipsis, 0]
    L[.ellipsis, 1, 1] = s[.ellipsis, 1]
    L[.ellipsis, 2, 2] = s[.ellipsis, 2]

    return R.matmul(L)
}

func strip_lowerdiag(L: MLXArray) -> MLXArray {
    let uncertainty = MLXArray.zeros([L.shape[0], 6], dtype: .float32)
    uncertainty[.ellipsis, 0] = L[.ellipsis, 0, 0]
    uncertainty[.ellipsis, 1] = L[.ellipsis, 0, 1]
    uncertainty[.ellipsis, 2] = L[.ellipsis, 0, 2]
    uncertainty[.ellipsis, 3] = L[.ellipsis, 1, 1]
    uncertainty[.ellipsis, 4] = L[.ellipsis, 1, 2]
    uncertainty[.ellipsis, 5] = L[.ellipsis, 2, 2]
    return uncertainty
}
let strip_symmetric = strip_lowerdiag
func conditionToIndices(condition: MLXArray) -> MLXArray {
    let mask = condition.reshaped([-1]).asType(.int32)
    let n = mask.shape[0]
    if n == 0 {
        return MLXArray([] as [Int32])
    }

    let outputs = conditionToIndicesKernel(
        [mask],
        grid: (n, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[n], [1]],
        outputDTypes: [DType.int32, DType.int32],
        initValue: 0
    )
    let indices = outputs[0]
    let count = outputs[1].item(Int.self)
    if count == 0 {
        return MLXArray([] as [Int32])
    }
    return MLX.stopGradient(indices[0..<count])
}
func detachedArray(_ array: MLXArray) -> MLXArray {
    let data = MLX.stopGradient(array).asData(access: .copy)
    return MLXArray(data: data)
}
