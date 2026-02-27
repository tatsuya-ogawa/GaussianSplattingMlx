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

static const uint TILE_THREADS = 128;

groupshared float gsMeansX[TILE_THREADS];
groupshared float gsMeansY[TILE_THREADS];
groupshared float gsConic00[TILE_THREADS];
groupshared float gsConic01[TILE_THREADS];
groupshared float gsConic10[TILE_THREADS];
groupshared float gsConic11[TILE_THREADS];
groupshared float gsOpacity[TILE_THREADS];
groupshared float gsColorX[TILE_THREADS];
groupshared float gsColorY[TILE_THREADS];
groupshared float gsColorZ[TILE_THREADS];
groupshared float gsDepth[TILE_THREADS];

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_tile_forward(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint p = tid.x;
    uint lane = gtid.x;
    uint pixelCount = counts[0];
    uint gaussianCount = counts[1];
    uint whiteBg = counts[2];
    bool activePixel = p < pixelCount;

    float px = 0.0;
    float py = 0.0;
    if (activePixel)
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
    bool done = !activePixel;

    for (uint chunkStart = 0; chunkStart < gaussianCount; chunkStart += TILE_THREADS)
    {
        uint loadIndex = chunkStart + lane;
        if (loadIndex < gaussianCount)
        {
            uint meansBase = loadIndex * 2;
            uint conicBase = loadIndex * 4;
            uint colorBase = loadIndex * 3;
            gsMeansX[lane] = sortedMeans2d[meansBase + 0];
            gsMeansY[lane] = sortedMeans2d[meansBase + 1];
            gsConic00[lane] = sortedConic[conicBase + 0];
            gsConic01[lane] = sortedConic[conicBase + 1];
            gsConic10[lane] = sortedConic[conicBase + 2];
            gsConic11[lane] = sortedConic[conicBase + 3];
            gsOpacity[lane] = sortedOpacity[loadIndex];
            gsColorX[lane] = sortedColor[colorBase + 0];
            gsColorY[lane] = sortedColor[colorBase + 1];
            gsColorZ[lane] = sortedColor[colorBase + 2];
            gsDepth[lane] = sortedDepths[loadIndex];
        }
        AllMemoryBarrierWithGroupSync();

        if (!done)
        {
            uint chunkEnd = min(chunkStart + TILE_THREADS, gaussianCount);
            for (uint i = chunkStart; i < chunkEnd; ++i)
            {
                uint local = i - chunkStart;
                float dx = px - gsMeansX[local];
                float dy = py - gsMeansY[local];
                float exponent = -0.5
                    * (dx * dx * gsConic00[local]
                        + dy * dy * gsConic11[local]
                        + dx * dy * gsConic01[local]
                        + dx * dy * gsConic10[local]);
                float alpha = exp(exponent) * gsOpacity[local];
                if (alpha > 0.99)
                {
                    alpha = 0.99;
                }

                float contrib = T * alpha;
                accAlpha += contrib;
                accumColorX += contrib * gsColorX[local];
                accumColorY += contrib * gsColorY[local];
                accumColorZ += contrib * gsColorZ[local];
                accumDepth += contrib * gsDepth[local];
                T *= (1.0 - alpha);

                if (T < 1e-4)
                {
                    done = true;
                    break;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
    }

    if (activePixel)
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
[numthreads(128, 1, 1)]
void gaussian_tile_backward(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint p = tid.x;
    uint lane = gtid.x;
    uint pixelCount = counts[0];
    uint gaussianCount = counts[1];
    uint whiteBg = counts[2];
    bool activePixel = p < pixelCount;

    float px = 0.0;
    float py = 0.0;
    float gColorX = 0.0;
    float gColorY = 0.0;
    float gColorZ = 0.0;
    float gDepth = 0.0;
    float gAlpha = 0.0;
    if (activePixel)
    {
        px = tileCoord[p * 2 + 0];
        py = tileCoord[p * 2 + 1];
        gColorX = cotColor[p * 3 + 0];
        gColorY = cotColor[p * 3 + 1];
        gColorZ = cotColor[p * 3 + 2];
        gDepth = cotDepth[p];
        gAlpha = cotAlpha[p];
    }
    float bgCoeff = (whiteBg != 0) ? -(gColorX + gColorY + gColorZ) : 0.0;

    float weightedSum = 0.0;
    float trans = 1.0;
    bool pass1Done = !activePixel;

    for (uint chunkStart = 0; chunkStart < gaussianCount; chunkStart += TILE_THREADS)
    {
        uint loadIndex = chunkStart + lane;
        if (loadIndex < gaussianCount)
        {
            uint meansBase = loadIndex * 2;
            uint conicBase = loadIndex * 4;
            uint colorBase = loadIndex * 3;
            gsMeansX[lane] = sortedMeans2d[meansBase + 0];
            gsMeansY[lane] = sortedMeans2d[meansBase + 1];
            gsConic00[lane] = sortedConic[conicBase + 0];
            gsConic01[lane] = sortedConic[conicBase + 1];
            gsConic10[lane] = sortedConic[conicBase + 2];
            gsConic11[lane] = sortedConic[conicBase + 3];
            gsOpacity[lane] = sortedOpacity[loadIndex];
            gsColorX[lane] = sortedColor[colorBase + 0];
            gsColorY[lane] = sortedColor[colorBase + 1];
            gsColorZ[lane] = sortedColor[colorBase + 2];
            gsDepth[lane] = sortedDepths[loadIndex];
        }
        AllMemoryBarrierWithGroupSync();

        if (!pass1Done)
        {
            uint chunkEnd = min(chunkStart + TILE_THREADS, gaussianCount);
            for (uint i = chunkStart; i < chunkEnd; ++i)
            {
                uint local = i - chunkStart;
                float dx = px - gsMeansX[local];
                float dy = py - gsMeansY[local];
                float exponent = -0.5
                    * (dx * dx * gsConic00[local]
                        + dy * dy * gsConic11[local]
                        + dx * dy * gsConic01[local]
                        + dx * dy * gsConic10[local]);
                float raw = exp(exponent) * gsOpacity[local];
                float alpha = raw;
                if (alpha > 0.99)
                {
                    alpha = 0.99;
                }

                float contrib = trans * alpha;
                float c = (gColorX * gsColorX[local]
                        + gColorY * gsColorY[local]
                        + gColorZ * gsColorZ[local])
                    + gDepth * gsDepth[local] + gAlpha + bgCoeff;
                weightedSum += c * contrib;

                trans *= (1.0 - alpha);
                if (trans < 1e-4)
                {
                    pass1Done = true;
                    break;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
    }

    trans = 1.0;
    float prefixWeighted = 0.0;
    bool pass2Done = !activePixel;
    for (uint chunkStart = 0; chunkStart < gaussianCount; chunkStart += TILE_THREADS)
    {
        uint loadIndex = chunkStart + lane;
        if (loadIndex < gaussianCount)
        {
            uint meansBase = loadIndex * 2;
            uint conicBase = loadIndex * 4;
            uint colorBase = loadIndex * 3;
            gsMeansX[lane] = sortedMeans2d[meansBase + 0];
            gsMeansY[lane] = sortedMeans2d[meansBase + 1];
            gsConic00[lane] = sortedConic[conicBase + 0];
            gsConic01[lane] = sortedConic[conicBase + 1];
            gsConic10[lane] = sortedConic[conicBase + 2];
            gsConic11[lane] = sortedConic[conicBase + 3];
            gsOpacity[lane] = sortedOpacity[loadIndex];
            gsColorX[lane] = sortedColor[colorBase + 0];
            gsColorY[lane] = sortedColor[colorBase + 1];
            gsColorZ[lane] = sortedColor[colorBase + 2];
            gsDepth[lane] = sortedDepths[loadIndex];
        }
        AllMemoryBarrierWithGroupSync();

        if (!pass2Done)
        {
            uint chunkEnd = min(chunkStart + TILE_THREADS, gaussianCount);
            for (uint i = chunkStart; i < chunkEnd; ++i)
            {
                uint local = i - chunkStart;
                float dx = px - gsMeansX[local];
                float dy = py - gsMeansY[local];
                float c00 = gsConic00[local];
                float c01 = gsConic01[local];
                float c10 = gsConic10[local];
                float c11 = gsConic11[local];
                float exponent = -0.5
                    * (dx * dx * c00
                        + dy * dy * c11
                        + dx * dy * c01
                        + dx * dy * c10);
                float expVal = exp(exponent);
                float raw = expVal * gsOpacity[local];
                float alpha = raw;
                if (alpha > 0.99)
                {
                    alpha = 0.99;
                }
                float contrib = trans * alpha;

                float c = (gColorX * gsColorX[local]
                        + gColorY * gsColorY[local]
                        + gColorZ * gsColorZ[local])
                    + gDepth * gsDepth[local] + gAlpha + bgCoeff;

                float suffixWeighted = weightedSum - (prefixWeighted + c * contrib);
                float denom = 1.0 - alpha;
                if (denom < 1e-6)
                {
                    denom = 1e-6;
                }
                float gradAlpha = c * trans - suffixWeighted / denom;

                uint colorBase = i * 3;
                uint conicBase = i * 4;

                InterlockedAdd(gradSortedDepths[i], gDepth * contrib);
                InterlockedAdd(gradSortedColor[colorBase + 0], gColorX * contrib);
                InterlockedAdd(gradSortedColor[colorBase + 1], gColorY * contrib);
                InterlockedAdd(gradSortedColor[colorBase + 2], gColorZ * contrib);

                if (raw <= 0.99)
                {
                    float gradRaw = gradAlpha;
                    InterlockedAdd(gradSortedOpacity[i], gradRaw * expVal);

                    float gradExponent = gradRaw * raw;
                    float dxdy = dx * dy;
                    InterlockedAdd(gradSortedConic[conicBase + 0], gradExponent * (-0.5 * dx * dx));
                    InterlockedAdd(gradSortedConic[conicBase + 1], gradExponent * (-0.5 * dxdy));
                    InterlockedAdd(gradSortedConic[conicBase + 2], gradExponent * (-0.5 * dxdy));
                    InterlockedAdd(gradSortedConic[conicBase + 3], gradExponent * (-0.5 * dy * dy));

                    float c01sum = c01 + c10;
                    float gradDx = gradExponent * (-dx * c00 - 0.5 * dy * c01sum);
                    float gradDy = gradExponent * (-dy * c11 - 0.5 * dx * c01sum);
                    InterlockedAdd(gradSortedMeans2d[i * 2 + 0], -gradDx);
                    InterlockedAdd(gradSortedMeans2d[i * 2 + 1], -gradDy);
                }

                prefixWeighted += c * contrib;
                trans *= (1.0 - alpha);
                if (trans < 1e-4)
                {
                    pass2Done = true;
                    break;
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
}
