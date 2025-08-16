#include <metal_stdlib>
using namespace metal;

// MARK: - Triangle Data Structures

struct TriangleData {
    float3 vertex0;     // First vertex
    float3 vertex1;     // Second vertex  
    float3 vertex2;     // Third vertex
    float3 color;       // RGB color
    float opacity;      // Alpha value
    float smoothness;   // Smoothness parameter σ
    float3 sh_dc;       // Spherical harmonics DC component
    float3 sh_rest[15]; // Spherical harmonics higher order coefficients
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

struct ProjectedTriangle {
    float2 vertex0_2d;
    float2 vertex1_2d;
    float2 vertex2_2d;
    float3 color;
    float alpha;
    float depth;
    float smoothness;
    uint originalIndex;
    float2 boundingBoxMin;
    float2 boundingBoxMax;
};

struct TileInfo {
    uint2 tileCoord;
    uint triangleCount;
    uint triangleOffset;
};

// MARK: - Triangle Utility Functions

float3 computeTriangleNormal(float3 v0, float3 v1, float3 v2) {
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    return normalize(cross(edge1, edge2));
}

float triangleArea(float2 v0, float2 v1, float2 v2) {
    float2 edge1 = v1 - v0;
    float2 edge2 = v2 - v0;
    // 2D cross product magnitude / 2 = area
    return abs(edge1.x * edge2.y - edge1.y * edge2.x) * 0.5f;
}

bool isTriangleFrontFacing(float2 v0, float2 v1, float2 v2) {
    float2 edge1 = v1 - v0;
    float2 edge2 = v2 - v0;
    // 2D cross product (z-component)
    float signedArea = edge1.x * edge2.y - edge1.y * edge2.x;
    return signedArea > 0.0f;
}

float2 computeTriangleIncenter(float2 v0, float2 v1, float2 v2) {
    // Edge lengths
    float a = length(v1 - v2);  // opposite to v0
    float b = length(v2 - v0);  // opposite to v1
    float c = length(v0 - v1);  // opposite to v2
    
    float perimeter = a + b + c;
    
    // Avoid division by zero
    if (perimeter < 1e-6f) {
        return (v0 + v1 + v2) / 3.0f;  // fallback to centroid
    }
    
    // Incenter formula: weighted average by opposite edge lengths
    return (a * v0 + b * v1 + c * v2) / perimeter;
}

float computeSignedDistanceField(float2 point, float2 v0, float2 v1, float2 v2) {
    // Compute edge vectors and normals
    float2 edge0 = v1 - v0;
    float2 edge1 = v2 - v1;
    float2 edge2 = v0 - v2;
    
    // Edge normals (pointing inward)
    float2 normal0 = float2(edge0.y, -edge0.x);
    float2 normal1 = float2(edge1.y, -edge1.x);
    float2 normal2 = float2(edge2.y, -edge2.x);
    
    // Normalize edge normals
    normal0 = normalize(normal0);
    normal1 = normalize(normal1);
    normal2 = normalize(normal2);
    
    // Compute signed distances to each edge
    float dist0 = dot(point - v0, normal0);
    float dist1 = dot(point - v1, normal1);
    float dist2 = dot(point - v2, normal2);
    
    // For a point to be inside the triangle, all distances should be negative
    // The signed distance field is the maximum of the three edge distances
    return max(max(dist0, dist1), dist2);
}

float computeTriangleWindowFunction(float2 point, float2 v0, float2 v1, float2 v2, float smoothness) {
    // Compute signed distance field
    float sdf = computeSignedDistanceField(point, v0, v1, v2);
    
    // Compute incenter and its SDF value
    float2 incenter = computeTriangleIncenter(v0, v1, v2);
    float incenterSdf = computeSignedDistanceField(incenter, v0, v1, v2);
    
    // Avoid division by zero
    if (abs(incenterSdf) < 1e-6f) {
        return 0.0f;
    }
    
    // Window function: ReLU(φ(p)/φ(s))^σ
    float ratio = sdf / incenterSdf;
    float window = pow(max(ratio, 0.0f), smoothness);
    
    return window;
}

float3 evaluateSphericalHarmonics(float3 direction, float3 sh_dc, const float3 sh_rest[15], int sh_degree) {
    float3 color = sh_dc + 0.5f;
    
    if (sh_degree > 0) {
        float x = direction.x;
        float y = direction.y;
        float z = direction.z;
        
        // First order SH
        color += sh_rest[0] * (-y);
        color += sh_rest[1] * z;
        color += sh_rest[2] * (-x);
        
        if (sh_degree > 1) {
            // Second order SH
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;
            
            color += sh_rest[3] * (xy);
            color += sh_rest[4] * (yz);
            color += sh_rest[5] * (2.0f * zz - xx - yy);
            color += sh_rest[6] * (xz);
            color += sh_rest[7] * (xx - yy);
        }
    }
    
    return max(color, 0.0f);
}

// MARK: - Compute Kernels

kernel void projectTriangles(
    device const TriangleData* triangles [[buffer(0)]],
    device ProjectedTriangle* projectedTriangles [[buffer(1)]],
    device const CameraParams& camera [[buffer(2)]],
    device uint* visibilityMask [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 1000000) return; // Adjust based on max triangles
    
    TriangleData triangle = triangles[id];
    
    // Transform vertices to view space
    float4 v0_world = float4(triangle.vertex0, 1.0f);
    float4 v1_world = float4(triangle.vertex1, 1.0f);
    float4 v2_world = float4(triangle.vertex2, 1.0f);
    
    float4 v0_view = camera.viewMatrix * v0_world;
    float4 v1_view = camera.viewMatrix * v1_world;
    float4 v2_view = camera.viewMatrix * v2_world;
    
    // Frustum culling - check if any vertex is behind near plane
    if (v0_view.z >= -0.1f && v1_view.z >= -0.1f && v2_view.z >= -0.1f) {
        visibilityMask[id] = 0;
        return;
    }
    
    // Project to clip space
    float4 v0_clip = camera.projMatrix * v0_view;
    float4 v1_clip = camera.projMatrix * v1_view;
    float4 v2_clip = camera.projMatrix * v2_view;
    
    // Perspective divide
    float3 v0_ndc = v0_clip.xyz / v0_clip.w;
    float3 v1_ndc = v1_clip.xyz / v1_clip.w;
    float3 v2_ndc = v2_clip.xyz / v2_clip.w;
    
    // Convert to screen coordinates
    float2 v0_screen = float2(
        ((v0_ndc.x + 1.0f) * camera.width - 1.0f) * 0.5f,
        ((v0_ndc.y + 1.0f) * camera.height - 1.0f) * 0.5f
    );
    float2 v1_screen = float2(
        ((v1_ndc.x + 1.0f) * camera.width - 1.0f) * 0.5f,
        ((v1_ndc.y + 1.0f) * camera.height - 1.0f) * 0.5f
    );
    float2 v2_screen = float2(
        ((v2_ndc.x + 1.0f) * camera.width - 1.0f) * 0.5f,
        ((v2_ndc.y + 1.0f) * camera.height - 1.0f) * 0.5f
    );
    
    // Check if triangle is front-facing
    if (!isTriangleFrontFacing(v0_screen, v1_screen, v2_screen)) {
        visibilityMask[id] = 0;
        return;
    }
    
    // Compute triangle area and cull degenerate triangles
    float area = triangleArea(v0_screen, v1_screen, v2_screen);
    if (area < 0.5f) {
        visibilityMask[id] = 0;
        return;
    }
    
    // Compute bounding box
    float2 minBounds = min(min(v0_screen, v1_screen), v2_screen);
    float2 maxBounds = max(max(v0_screen, v1_screen), v2_screen);
    
    // Screen bounds check
    if (maxBounds.x < 0 || minBounds.x >= camera.width || 
        maxBounds.y < 0 || minBounds.y >= camera.height) {
        visibilityMask[id] = 0;
        return;
    }
    
    // Compute triangle center for color evaluation
    float3 triangleCenter = (triangle.vertex0 + triangle.vertex1 + triangle.vertex2) / 3.0f;
    float3 viewDirection = normalize(triangleCenter - camera.cameraCenter);
    
    // Evaluate color using spherical harmonics
    float3 color = evaluateSphericalHarmonics(viewDirection, triangle.sh_dc, triangle.sh_rest, 3);
    
    // Compute average depth
    float avgDepth = (v0_view.z + v1_view.z + v2_view.z) / 3.0f;
    
    // Mark as visible and store projected data
    visibilityMask[id] = 1;
    
    projectedTriangles[id].vertex0_2d = v0_screen;
    projectedTriangles[id].vertex1_2d = v1_screen;
    projectedTriangles[id].vertex2_2d = v2_screen;
    projectedTriangles[id].color = color;
    projectedTriangles[id].alpha = clamp(1.0f / (1.0f + exp(-triangle.opacity)), 0.0f, 0.99f);
    projectedTriangles[id].depth = -avgDepth; // Negative for proper sorting
    projectedTriangles[id].smoothness = exp(triangle.smoothness);
    projectedTriangles[id].originalIndex = id;
    projectedTriangles[id].boundingBoxMin = minBounds;
    projectedTriangles[id].boundingBoxMax = maxBounds;
}

kernel void sortTrianglesByDepth(
    device const ProjectedTriangle* input [[buffer(0)]],
    device ProjectedTriangle* output [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= 1000000) return;
    uint index = indices[id];
    output[id] = input[index];
}

kernel void assignTrianglesToTiles(
    device const ProjectedTriangle* triangles [[buffer(0)]],
    device uint* tileAssignments [[buffer(1)]],
    device atomic_uint* tileCounts [[buffer(2)]],
    constant uint2& imageSize [[buffer(3)]],
    constant uint2& tileSize [[buffer(4)]],
    constant uint& numTriangles [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numTriangles) return;
    
    ProjectedTriangle triangle = triangles[id];
    
    // Compute tile bounds for triangle bounding box
    uint2 tileMin = uint2(triangle.boundingBoxMin / float2(tileSize));
    uint2 tileMax = uint2(triangle.boundingBoxMax / float2(tileSize));
    
    uint2 tilesPerAxis = (imageSize + tileSize - 1) / tileSize;
    
    tileMin = min(tileMin, tilesPerAxis - 1);
    tileMax = min(tileMax, tilesPerAxis - 1);
    
    // Assign to tiles
    for (uint ty = tileMin.y; ty <= tileMax.y; ty++) {
        for (uint tx = tileMin.x; tx <= tileMax.x; tx++) {
            uint tileIndex = ty * tilesPerAxis.x + tx;
            uint assignmentIndex = atomic_fetch_add_explicit(&tileCounts[tileIndex], 1, memory_order_relaxed);
            
            // Store assignment (would need proper indexing in practice)
            uint globalAssignmentIndex = tileIndex * 1024 + assignmentIndex; // Max 1024 triangles per tile
            if (assignmentIndex < 1024) {
                tileAssignments[globalAssignmentIndex] = id;
            }
        }
    }
}

kernel void renderTriangleTiles(
    device const ProjectedTriangle* sortedTriangles [[buffer(0)]],
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
    
    float2 pixelPos = float2(pixelCoord);
    
    for (uint i = 0; i < tileInfo.triangleCount && T > 0.001f; i++) {
        ProjectedTriangle triangle = sortedTriangles[tileInfo.triangleOffset + i];
        
        // Check if pixel is within triangle bounding box
        if (pixelPos.x < triangle.boundingBoxMin.x || pixelPos.x > triangle.boundingBoxMax.x ||
            pixelPos.y < triangle.boundingBoxMin.y || pixelPos.y > triangle.boundingBoxMax.y) {
            continue;
        }
        
        // Compute window function weight
        float weight = computeTriangleWindowFunction(
            pixelPos, 
            triangle.vertex0_2d, 
            triangle.vertex1_2d, 
            triangle.vertex2_2d, 
            triangle.smoothness
        );
        
        if (weight > 1e-6f) {
            float alpha = min(0.99f, triangle.alpha * weight);
            
            accumulatedColor += T * alpha * triangle.color;
            accumulatedAlpha += T * alpha;
            T *= (1.0f - alpha);
        }
    }
    
    // Add background (black background)
    float3 backgroundColor = float3(0.0f);
    float3 finalColor = accumulatedColor + T * backgroundColor;
    
    uint pixelIndex = pixelCoord.y * imageSize.x + pixelCoord.x;
    outputImage[pixelIndex] = float4(finalColor, accumulatedAlpha);
}

kernel void clearTriangleBuffers(
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