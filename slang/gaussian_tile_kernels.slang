StructuredBuffer<float> tileCoord;
StructuredBuffer<float> sortedDepths;
StructuredBuffer<float> sortedMeans2d;
StructuredBuffer<float> sortedConic;
StructuredBuffer<float> sortedOpacity;
StructuredBuffer<float> sortedColor;
StructuredBuffer<float> cotColor;
StructuredBuffer<float> cotDepth;
StructuredBuffer<float> cotAlpha;
StructuredBuffer<uint> counts;

RWStructuredBuffer<float> outColor;
RWStructuredBuffer<float> outDepth;
RWStructuredBuffer<float> outAlpha;

RWStructuredBuffer<float> gradTileCoord;
RWStructuredBuffer<float> gradSortedDepths;
RWStructuredBuffer<float> gradSortedMeans2d;
RWStructuredBuffer<float> gradSortedConic;
RWStructuredBuffer<float> gradSortedOpacity;
RWStructuredBuffer<float> gradSortedColor;

static const uint BLOCK_SIZE = 128;

groupshared float2 s_means2d[BLOCK_SIZE];
groupshared float4 s_conic[BLOCK_SIZE];
groupshared float s_opacity[BLOCK_SIZE];
groupshared float3 s_color[BLOCK_SIZE];
groupshared float s_depths[BLOCK_SIZE];

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void gaussian_tile_forward(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint p = tid.x;
    uint pixelCount = counts[0];
    uint gaussianCount = counts[1];
    uint whiteBg = counts[2];

    float px = 0.0;
    float py = 0.0;
    bool isActive = (p < pixelCount);
    if (isActive)
    {
        px = tileCoord[p * 2 + 0];
        py = tileCoord[p * 2 + 1];
    }

    float T = 1.0;
    float accAlpha = 0.0;
    float accumColorX = 0.0;
    float accumColorY = 0.0;
    float accumColorZ = 0.0;
    float accumDepth = 0.0;
    
    uint rounds = (gaussianCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint remaining = gaussianCount;
    
    for (uint i = 0; i < rounds; ++i)
    {
        uint blockCount = min(BLOCK_SIZE, remaining);
        
        // Cooperative Fetch
        GroupMemoryBarrierWithGroupSync();
        uint fetchIdx = i * BLOCK_SIZE + gtid.x;
        if (gtid.x < blockCount)
        {
            s_means2d[gtid.x] = float2(sortedMeans2d[fetchIdx * 2 + 0], sortedMeans2d[fetchIdx * 2 + 1]);
            s_conic[gtid.x] = float4(sortedConic[fetchIdx * 4 + 0], sortedConic[fetchIdx * 4 + 1], sortedConic[fetchIdx * 4 + 2], sortedConic[fetchIdx * 4 + 3]);
            s_opacity[gtid.x] = sortedOpacity[fetchIdx];
            s_color[gtid.x] = float3(sortedColor[fetchIdx * 3 + 0], sortedColor[fetchIdx * 3 + 1], sortedColor[fetchIdx * 3 + 2]);
            s_depths[gtid.x] = sortedDepths[fetchIdx];
        }
        GroupMemoryBarrierWithGroupSync();
        
        if (isActive && T >= 1e-4)
        {
            for (uint j = 0; j < blockCount; ++j)
            {
                float2 mean = s_means2d[j];
                float dx = px - mean.x;
                float dy = py - mean.y;
                float4 conic = s_conic[j];
                float exponent = -0.5
                    * (dx * dx * conic.x
                        + dy * dy * conic.w
                        + dx * dy * conic.y
                        + dx * dy * conic.z);
                float alpha = exp(exponent) * s_opacity[j];
                if (alpha > 0.99)
                {
                    alpha = 0.99;
                }

                float contrib = T * alpha;
                accAlpha += contrib;
                float3 color = s_color[j];
                accumColorX += contrib * color.x;
                accumColorY += contrib * color.y;
                accumColorZ += contrib * color.z;
                accumDepth += contrib * s_depths[j];
                T *= (1.0 - alpha);

                if (T < 1e-4)
                {
                    break;
                }
            }
        }
        remaining -= blockCount;
    }

    if (isActive)
    {
        float bg = (whiteBg != 0) ? (1.0 - accAlpha) : 0.0;
        outColor[p * 3 + 0] = accumColorX + bg;
        outColor[p * 3 + 1] = accumColorY + bg;
        outColor[p * 3 + 2] = accumColorZ + bg;
        outDepth[p] = accumDepth;
        outAlpha[p] = accAlpha;
    }
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void gaussian_tile_backward(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint p = tid.x;
    uint pixelCount = counts[0];
    uint gaussianCount = counts[1];
    uint whiteBg = counts[2];

    float px = 0.0;
    float py = 0.0;
    float gColorX = 0.0;
    float gColorY = 0.0;
    float gColorZ = 0.0;
    float gDepth = 0.0;
    float gAlpha = 0.0;
    float bgCoeff = 0.0;
    bool isActive = (p < pixelCount);

    if (isActive)
    {
        px = tileCoord[p * 2 + 0];
        py = tileCoord[p * 2 + 1];

        gColorX = cotColor[p * 3 + 0];
        gColorY = cotColor[p * 3 + 1];
        gColorZ = cotColor[p * 3 + 2];
        gDepth = cotDepth[p];
        gAlpha = cotAlpha[p];
        bgCoeff = (whiteBg != 0) ? -(gColorX + gColorY + gColorZ) : 0.0;
    }

    // First pass
    float weightedSum = 0.0;
    float trans = 1.0;
    
    uint rounds = (gaussianCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint remaining = gaussianCount;

    for (uint i = 0; i < rounds; ++i)
    {
        uint blockCount = min(BLOCK_SIZE, remaining);
        
        GroupMemoryBarrierWithGroupSync();
        uint fetchIdx = i * BLOCK_SIZE + gtid.x;
        if (gtid.x < blockCount)
        {
            s_means2d[gtid.x] = float2(sortedMeans2d[fetchIdx * 2 + 0], sortedMeans2d[fetchIdx * 2 + 1]);
            s_conic[gtid.x] = float4(sortedConic[fetchIdx * 4 + 0], sortedConic[fetchIdx * 4 + 1], sortedConic[fetchIdx * 4 + 2], sortedConic[fetchIdx * 4 + 3]);
            s_opacity[gtid.x] = sortedOpacity[fetchIdx];
            s_color[gtid.x] = float3(sortedColor[fetchIdx * 3 + 0], sortedColor[fetchIdx * 3 + 1], sortedColor[fetchIdx * 3 + 2]);
            s_depths[gtid.x] = sortedDepths[fetchIdx];
        }
        GroupMemoryBarrierWithGroupSync();

        if (isActive && trans >= 1e-4)
        {
            for (uint j = 0; j < blockCount; ++j)
            {
                float2 mean = s_means2d[j];
                float dx = px - mean.x;
                float dy = py - mean.y;
                float4 conic = s_conic[j];
                float exponent = -0.5
                    * (dx * dx * conic.x
                        + dy * dy * conic.w
                        + dx * dy * conic.y
                        + dx * dy * conic.z);
                float raw = exp(exponent) * s_opacity[j];
                float alpha = raw;
                if (alpha > 0.99)
                {
                    alpha = 0.99;
                }

                float contrib = trans * alpha;
                float3 color = s_color[j];
                float c = (gColorX * color.x
                        + gColorY * color.y
                        + gColorZ * color.z)
                    + gDepth * s_depths[j] + gAlpha + bgCoeff;
                weightedSum += c * contrib;

                trans *= (1.0 - alpha);
                if (trans < 1e-4)
                {
                    break;
                }
            }
        }
        remaining -= blockCount;
    }

    // Since each thread can stop at a different time, we need a common end index for the second pass
    // Or we simply do the second pass exactly same fashion and rely on the same trans check to not add gradients
    trans = 1.0;
    float prefixWeighted = 0.0;
    remaining = gaussianCount;

    for (uint i = 0; i < rounds; ++i)
    {
        uint blockCount = min(BLOCK_SIZE, remaining);
        
        GroupMemoryBarrierWithGroupSync();
        uint fetchIdx = i * BLOCK_SIZE + gtid.x;
        if (gtid.x < blockCount)
        {
            s_means2d[gtid.x] = float2(sortedMeans2d[fetchIdx * 2 + 0], sortedMeans2d[fetchIdx * 2 + 1]);
            s_conic[gtid.x] = float4(sortedConic[fetchIdx * 4 + 0], sortedConic[fetchIdx * 4 + 1], sortedConic[fetchIdx * 4 + 2], sortedConic[fetchIdx * 4 + 3]);
            s_opacity[gtid.x] = sortedOpacity[fetchIdx];
            s_color[gtid.x] = float3(sortedColor[fetchIdx * 3 + 0], sortedColor[fetchIdx * 3 + 1], sortedColor[fetchIdx * 3 + 2]);
            s_depths[gtid.x] = sortedDepths[fetchIdx];
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint j = 0; j < blockCount; ++j)
        {
            bool stopped = (trans < 1e-4);
            // Even if stopped, threads must participate in the loop for InterlockedAdd (actually InterlockedAdd is conditionally executed, which is fine, but they must evaluate conditions)
            
            float2 mean = s_means2d[j];
            float dx = px - mean.x;
            float dy = py - mean.y;
            float4 conic = s_conic[j];
            float c00 = conic.x;
            float c01 = conic.y;
            float c10 = conic.z;
            float c11 = conic.w;
            float exponent = -0.5
                * (dx * dx * c00
                    + dy * dy * c11
                    + dx * dy * c01
                    + dx * dy * c10);
            
            float expVal = exp(exponent);
            float opacityRaw = s_opacity[j];
            float raw = expVal * opacityRaw;
            float alpha = raw;
            if (alpha > 0.99)
            {
                alpha = 0.99;
            }
            float contrib = trans * alpha;

            float3 color = s_color[j];
            float depth = s_depths[j];
            float c = (gColorX * color.x
                    + gColorY * color.y
                    + gColorZ * color.z)
                + gDepth * depth + gAlpha + bgCoeff;

            float suffixWeighted = weightedSum - (prefixWeighted + c * contrib);
            float denom = 1.0 - alpha;
            if (denom < 1e-6)
            {
                denom = 1e-6;
            }
            float gradAlpha = c * trans - suffixWeighted / denom;

            if (isActive && !stopped)
            {
                InterlockedAdd(gradSortedDepths[fetchIdx + j], gDepth * contrib);
                InterlockedAdd(gradSortedColor[(fetchIdx + j) * 3 + 0], gColorX * contrib);
                InterlockedAdd(gradSortedColor[(fetchIdx + j) * 3 + 1], gColorY * contrib);
                InterlockedAdd(gradSortedColor[(fetchIdx + j) * 3 + 2], gColorZ * contrib);

                if (raw <= 0.99)
                {
                    float gradRaw = gradAlpha;
                    InterlockedAdd(gradSortedOpacity[fetchIdx + j], gradRaw * expVal);

                    float gradExponent = gradRaw * raw;
                    float dxdy = dx * dy;
                    InterlockedAdd(gradSortedConic[(fetchIdx + j) * 4 + 0], gradExponent * (-0.5 * dx * dx));
                    InterlockedAdd(gradSortedConic[(fetchIdx + j) * 4 + 1], gradExponent * (-0.5 * dxdy));
                    InterlockedAdd(gradSortedConic[(fetchIdx + j) * 4 + 2], gradExponent * (-0.5 * dxdy));
                    InterlockedAdd(gradSortedConic[(fetchIdx + j) * 4 + 3], gradExponent * (-0.5 * dy * dy));

                    float c01sum = c01 + c10;
                    float gradDx = gradExponent * (-dx * c00 - 0.5 * dy * c01sum);
                    float gradDy = gradExponent * (-dy * c11 - 0.5 * dx * c01sum);
                    InterlockedAdd(gradSortedMeans2d[(fetchIdx + j) * 2 + 0], -gradDx);
                    InterlockedAdd(gradSortedMeans2d[(fetchIdx + j) * 2 + 1], -gradDy);
                }
                
                prefixWeighted += c * contrib;
                trans *= (1.0 - alpha);
            }
        }
        remaining -= blockCount;
    }
}
