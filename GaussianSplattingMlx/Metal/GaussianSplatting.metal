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

float3x3 buildRotationMatrix(float4 quat) {
    float norm = length(quat);
    quat = quat / max(norm, 1e-8f);
    
    float r = quat.x, x = quat.y, y = quat.z, z = quat.w;
    
    float3x3 R;
    R[0][0] = 1 - 2 * (y * y + z * z);
    R[0][1] = 2 * (x * y - r * z);
    R[0][2] = 2 * (x * z + r * y);
    R[1][0] = 2 * (x * y + r * z);
    R[1][1] = 1 - 2 * (x * x + z * z);
    R[1][2] = 2 * (y * z - r * x);
    R[2][0] = 2 * (x * z - r * y);
    R[2][1] = 2 * (y * z + r * x);
    R[2][2] = 1 - 2 * (x * x + y * y);
    
    return R;
}

float3x3 buildCovariance3D(float3 scale, float4 rotation) {
    float3x3 R = buildRotationMatrix(rotation);
    float3x3 S = float3x3(0);
    S[0][0] = exp(scale.x);
    S[1][1] = exp(scale.y);
    S[2][2] = exp(scale.z);
    
    float3x3 L = R * S;
    return L * transpose(L);
}

float3 computeColor(float3 position, float3 sh_dc, device const float3* sh_rest, float3 cameraCenter, int activeShDegree) {
    float3 dir = normalize(position - cameraCenter);
    float3 color = sh_dc + 0.5f;
    
    if (activeShDegree > 0) {
        float x = dir.x, y = dir.y, z = dir.z;
        color += sh_rest[0] * (-y) + sh_rest[1] * z + sh_rest[2] * (-x);
        
        if (activeShDegree > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            color += sh_rest[3] * (xy) + sh_rest[4] * (yz) + sh_rest[5] * (2.0f * zz - xx - yy) + 
                     sh_rest[6] * (xz) + sh_rest[7] * (xx - yy);
        }
    }
    
    return max(color, 0.0f);
}

kernel void projectGaussians(
    device const GaussianData* gaussians [[buffer(0)]],
    device ProjectedGaussian* projectedGaussians [[buffer(1)]],
    device const CameraParams& camera [[buffer(2)]],
    device uint* visibilityMask [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 1000000) return; // Adjust based on max gaussians
    
    GaussianData gaussian = gaussians[id];
    
    // Transform to camera space
    float4 worldPos = float4(gaussian.position, 1.0f);
    float4 viewPos = camera.viewMatrix * worldPos;
    
    // Frustum culling
    if (viewPos.z <= 0.2f) {
        visibilityMask[id] = 0;
        return;
    }
    
    // Project to screen
    float4 clipPos = camera.projMatrix * viewPos;
    float3 ndc = clipPos.xyz / clipPos.w;
    
    // Convert to pixel coordinates
    float2 screenPos;
    screenPos.x = ((ndc.x + 1.0f) * camera.width - 1.0f) * 0.5f;
    screenPos.y = ((ndc.y + 1.0f) * camera.height - 1.0f) * 0.5f;
    
    // Build 3D covariance
    float3x3 cov3d = buildCovariance3D(gaussian.scale, gaussian.rotation);
    
    // Transform covariance to 2D
    float tanFovX = tan(camera.fovX * 0.5f);
    float tanFovY = tan(camera.fovY * 0.5f);
    
    float3 t = viewPos.xyz;
    float tx = t.x / clamp(t.z, -tanFovX * 1.3f, tanFovX * 1.3f) * t.z;
    float ty = t.y / clamp(t.z, -tanFovY * 1.3f, tanFovY * 1.3f) * t.z;
    float tz = t.z;
    
    // Jacobian matrix
    float3x3 J = float3x3(0);
    J[0][0] = camera.focalX / tz;
    J[0][2] = -tx * camera.focalX / (tz * tz);
    J[1][1] = camera.focalY / tz;
    J[1][2] = -ty * camera.focalY / (tz * tz);
    J[2][2] = 1.0f;
    
    float3x3 W = transpose(float3x3(camera.viewMatrix[0].xyz, camera.viewMatrix[1].xyz, camera.viewMatrix[2].xyz));
    float3x3 cov2d_full = J * W * cov3d * transpose(W) * transpose(J);
    
    // Extract 2D part and add filtering
    float3 cov2d;
    cov2d.x = cov2d_full[0][0] + 0.3f;
    cov2d.y = cov2d_full[0][1];
    cov2d.z = cov2d_full[1][1] + 0.3f;
    
    // Compute radius
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    float mid = 0.5f * (cov2d.x + cov2d.z);
    float discriminant = max(mid * mid - det, 1e-5f);
    float sqrtDisc = sqrt(discriminant);
    float lambda1 = mid + sqrtDisc;
    float lambda2 = mid - sqrtDisc;
    float maxEigenvalue = max(lambda1, lambda2);
    float radius = 3.0f * ceil(sqrt(maxEigenvalue));
    
    // Check if visible
    if (radius > 0 && 
        screenPos.x + radius >= 0 && screenPos.x - radius < camera.width &&
        screenPos.y + radius >= 0 && screenPos.y - radius < camera.height) {
        
        visibilityMask[id] = 1;
        
        // Compute color (simplified for now)
        float3 color = gaussian.sh_dc + 0.5f;
        
        // Store projected data
        projectedGaussians[id].mean2d = screenPos;
        projectedGaussians[id].cov2d = cov2d;
        projectedGaussians[id].color = color;
        projectedGaussians[id].alpha = clamp(1.0f / (1.0f + exp(-gaussian.opacity)), 0.0f, 0.99f);
        projectedGaussians[id].depth = viewPos.z;
        projectedGaussians[id].originalIndex = id;
    } else {
        visibilityMask[id] = 0;
    }
}

kernel void sortGaussiansByDepth(
    device const ProjectedGaussian* input [[buffer(0)]],
    device ProjectedGaussian* output [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 1000000) return;
    uint index = indices[id];
    output[id] = input[index];
}

kernel void renderTiles(
    device const ProjectedGaussian* sortedGaussians [[buffer(0)]],
    device const TileInfo* tileInfos [[buffer(1)]],
    device float4* outputImage [[buffer(2)]],
    constant uint2& imageSize [[buffer(3)]],
    constant uint2& tileSize [[buffer(4)]],
    uint2 tileCoord [[threadgroup_position_in_grid]],
    uint2 localCoord [[thread_position_in_threadgroup]]
) {
    uint2 pixelCoord = tileCoord * tileSize + localCoord;
    if (pixelCoord.x >= imageSize.x || pixelCoord.y >= imageSize.y) return;
    
    uint tileIndex = tileCoord.y * ((imageSize.x + tileSize.x - 1) / tileSize.x) + tileCoord.x;
    TileInfo tileInfo = tileInfos[tileIndex];
    
    float3 accumulatedColor = float3(0.0f);
    float accumulatedAlpha = 0.0f;
    float T = 1.0f;
    
    for (uint i = 0; i < tileInfo.gaussianCount && T > 0.001f; i++) {
        ProjectedGaussian gaussian = sortedGaussians[tileInfo.gaussianOffset + i];
        
        float2 delta = float2(pixelCoord) - gaussian.mean2d;
        
        // Compute inverse covariance (2x2 matrix inverse)
        float det = gaussian.cov2d.x * gaussian.cov2d.z - gaussian.cov2d.y * gaussian.cov2d.y;
        if (abs(det) < 1e-6f) continue;
        
        float invDet = 1.0f / det;
        float2x2 invCov;
        invCov[0][0] = gaussian.cov2d.z * invDet;
        invCov[0][1] = -gaussian.cov2d.y * invDet;
        invCov[1][0] = -gaussian.cov2d.y * invDet;
        invCov[1][1] = gaussian.cov2d.x * invDet;
        
        // Compute Gaussian weight
        float power = -0.5f * dot(delta, invCov * delta);
        if (power > -4.0f) { // Cutoff for efficiency
            float weight = exp(power);
            float alpha = min(0.99f, gaussian.alpha * weight);
            
            accumulatedColor += T * alpha * gaussian.color;
            accumulatedAlpha += T * alpha;
            T *= (1.0f - alpha);
        }
    }
    
    // Add background
    float3 backgroundColor = float3(0.0f); // Black background
    float3 finalColor = accumulatedColor + T * backgroundColor;
    
    uint pixelIndex = pixelCoord.y * imageSize.x + pixelCoord.x;
    outputImage[pixelIndex] = float4(finalColor, accumulatedAlpha);
}