#include <metal_stdlib>
using namespace metal;

struct GaussianData {
    float3 position;
    float4 rotation;
    float3 scale;
    float opacity;
    float3 sh_dc;
    float3 sh_rest[15];
};

struct CameraParams {
    float4x4 viewMatrix;
    float4x4 projMatrix;
    float3 cameraCenter;
    float focalX;
    float focalY;
    float fovX;
    float fovY;
    int width;
    int height;
};

struct ProjectedGaussian {
    float2 mean2d;
    float3 cov2d;
    float3 color;
    float alpha;
    float depth;
    uint originalIndex;
};

struct TileInfo {
    uint2 tileCoord;
    uint gaussianCount;
    uint gaussianOffset;
};

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

// Simple vertex shader for full-screen quad
vertex VertexOut vertexShader(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

// Fragment shader for final composition
fragment float4 fragmentShader(VertexOut in [[stage_in]],
                              texture2d<float> renderTexture [[texture(0)]]) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    float4 color = renderTexture.sample(textureSampler, in.texCoord);
    return float4(color.rgb, 1.0);
}

// ---- Radix Sort Kernels (8-bit, 4-pass) ----

// Extract depth from projected gaussians and convert to sortable unsigned integer.
// IEEE 754 float → sortable uint: flip sign bit for positive, flip all bits for negative.
kernel void prepareRadixSortKeys(
    device const ProjectedGaussian* projected [[buffer(0)]],
    device uint* keys [[buffer(1)]],
    device uint* values [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numElements) return;
    float depth = projected[id].depth;
    uint bits = as_type<uint>(depth);
    // Positive floats: flip sign bit so they sort above negative.
    // Negative floats: flip all bits so larger magnitude sorts lower.
    uint mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    keys[id] = bits ^ mask;
    values[id] = id;
}

// Count digit occurrences for one 8-bit radix pass.
kernel void radixSortCount(
    device const uint* keys [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant uint& bitOffset [[buffer(2)]],
    constant uint& numElements [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numElements) return;
    uint digit = (keys[id] >> bitOffset) & 0xFFu;
    atomic_fetch_add_explicit(&histogram[digit], 1, memory_order_relaxed);
}

// Exclusive prefix sum of 256-entry histogram (single-thread, good enough for 256 entries).
kernel void radixSortPrefixSum(
    device uint* histogram [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    if (id != 0) return;
    uint sum = 0;
    for (uint i = 0; i < 256; i++) {
        uint count = histogram[i];
        histogram[i] = sum;
        sum += count;
    }
}

// Scatter keys and values to sorted positions using atomic offset claims.
kernel void radixSortScatter(
    device const uint* inputKeys [[buffer(0)]],
    device const uint* inputValues [[buffer(1)]],
    device uint* outputKeys [[buffer(2)]],
    device uint* outputValues [[buffer(3)]],
    device atomic_uint* histogram [[buffer(4)]],
    constant uint& bitOffset [[buffer(5)]],
    constant uint& numElements [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numElements) return;
    uint key = inputKeys[id];
    uint value = inputValues[id];
    uint digit = (key >> bitOffset) & 0xFFu;
    uint pos = atomic_fetch_add_explicit(&histogram[digit], 1, memory_order_relaxed);
    outputKeys[pos] = key;
    outputValues[pos] = value;
}

// Tile assignment kernel
kernel void assignGaussiansToTiles(
    device const ProjectedGaussian* gaussians [[buffer(0)]],
    device uint* tileAssignments [[buffer(1)]],
    device atomic_uint* tileCounts [[buffer(2)]],
    device const uint* visibilityMask [[buffer(3)]],
    constant uint2& imageSize [[buffer(4)]],
    constant uint2& tileSize [[buffer(5)]],
    constant uint& numGaussians [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numGaussians) return;
    if (visibilityMask[id] == 0) return;
    
    ProjectedGaussian gaussian = gaussians[id];
    if (gaussian.alpha <= 0.0f) return;
    
    // Compute radius
    float det = gaussian.cov2d.x * gaussian.cov2d.z - gaussian.cov2d.y * gaussian.cov2d.y;
    if (det <= 1e-8f) return;
    float mid = 0.5f * (gaussian.cov2d.x + gaussian.cov2d.z);
    float discriminant = max(mid * mid - det, 1e-5f);
    float radius = 3.0f * sqrt(max(mid + sqrt(discriminant), mid - sqrt(discriminant)));
    if (radius <= 0.0f) return;
    
    // Compute tile bounds
    int2 tileMin = int2(floor((gaussian.mean2d - radius) / float2(tileSize)));
    int2 tileMax = int2(floor((gaussian.mean2d + radius) / float2(tileSize)));
    
    uint2 tilesPerAxis = (imageSize + tileSize - 1) / tileSize;
    
    tileMin = max(tileMin, int2(0));
    tileMax = min(tileMax, int2(tilesPerAxis) - 1);
    
    // Assign to tiles
    for (int ty = tileMin.y; ty <= tileMax.y; ty++) {
        for (int tx = tileMin.x; tx <= tileMax.x; tx++) {
            uint tileIndex = ty * tilesPerAxis.x + tx;
            uint assignmentIndex = atomic_fetch_add_explicit(&tileCounts[tileIndex], 1, memory_order_relaxed);
            
            // Store assignment (would need proper indexing in practice)
            uint globalAssignmentIndex = tileIndex * 1024 + assignmentIndex; // Assuming max 1024 gaussians per tile
            if (assignmentIndex < 1024) {
                tileAssignments[globalAssignmentIndex] = id;
            }
        }
    }
}

// Prefix sum for tile offsets
kernel void computeTileOffsets(
    device const uint* tileCounts [[buffer(0)]],
    device uint* tileOffsets [[buffer(1)]],
    constant uint& numTiles [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numTiles) return;
    
    if (id == 0) {
        tileOffsets[0] = 0;
        for (uint i = 1; i < numTiles; i++) {
            tileOffsets[i] = tileOffsets[i-1] + tileCounts[i-1];
        }
    }
}

// Clear buffers kernel
kernel void clearBuffers(
    device float4* image [[buffer(0)]],
    device uint* counts [[buffer(1)]],
    constant uint& imagePixels [[buffer(2)]],
    constant uint& numTiles [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < imagePixels) {
        image[id] = float4(0.0f);
    }
    if (id < numTiles) {
        counts[id] = 0;
    }
}
