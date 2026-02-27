// gaussian_tile_global_kernels.slang
//
// Minimal global tile grouping kernels:
//   1. count_tiles_per_gaussian
//   2. generate_keys
//   3. compute_tile_ranges

// ---------------------------------------------------------------------------
// count_tiles_per_gaussian
// ---------------------------------------------------------------------------
StructuredBuffer<float> ct_rectMin;       // [N, 2]
StructuredBuffer<float> ct_rectMax;       // [N, 2]
StructuredBuffer<float> ct_radii;         // [N]
StructuredBuffer<uint> ct_counts;         // [0]=N, [1]=tileW, [2]=tileH, [3]=imageW, [4]=imageH
RWStructuredBuffer<uint> ct_tilesTouched; // [N]

[shader("compute")]
[numthreads(256, 1, 1)]
void count_tiles_per_gaussian(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint N = ct_counts[0];
    uint tileW = ct_counts[1];
    uint tileH = ct_counts[2];
    uint imageW = ct_counts[3];
    uint imageH = ct_counts[4];

    if (idx >= N)
    {
        return;
    }

    if (ct_radii[idx] <= 0.0)
    {
        ct_tilesTouched[idx] = 0;
        return;
    }

    float rMinX = ct_rectMin[idx * 2 + 0];
    float rMinY = ct_rectMin[idx * 2 + 1];
    float rMaxX = ct_rectMax[idx * 2 + 0];
    float rMaxY = ct_rectMax[idx * 2 + 1];

    int tMinX = int(floor(rMinX / float(tileW)));
    int tMinY = int(floor(rMinY / float(tileH)));
    int tMaxX = int(floor(rMaxX / float(tileW))) + 1;
    int tMaxY = int(floor(rMaxY / float(tileH))) + 1;

    int gridW = int((imageW + tileW - 1) / tileW);
    int gridH = int((imageH + tileH - 1) / tileH);

    tMinX = max(0, min(tMinX, gridW));
    tMinY = max(0, min(tMinY, gridH));
    tMaxX = max(0, min(tMaxX, gridW));
    tMaxY = max(0, min(tMaxY, gridH));

    ct_tilesTouched[idx] = uint((tMaxX - tMinX) * (tMaxY - tMinY));
}

// ---------------------------------------------------------------------------
// generate_keys
// ---------------------------------------------------------------------------
StructuredBuffer<float> gk_depths;       // [N]
StructuredBuffer<float> gk_rectMin;      // [N, 2]
StructuredBuffer<float> gk_rectMax;      // [N, 2]
StructuredBuffer<float> gk_radii;        // [N]
StructuredBuffer<uint> gk_offsets;       // [N] exclusive prefix sum
StructuredBuffer<uint> gk_counts;        // [0]=N, [1]=tileW, [2]=tileH, [3]=imageW, [4]=imageH
RWStructuredBuffer<uint> gk_keysHigh;    // [M] tile id
RWStructuredBuffer<uint> gk_keysLow;     // [M] depth bits
RWStructuredBuffer<uint> gk_gaussIdx;    // [M] gaussian index

[shader("compute")]
[numthreads(256, 1, 1)]
void generate_keys(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint N = gk_counts[0];
    uint tileW = gk_counts[1];
    uint tileH = gk_counts[2];
    uint imageW = gk_counts[3];
    uint imageH = gk_counts[4];

    if (idx >= N)
    {
        return;
    }

    if (gk_radii[idx] <= 0.0)
    {
        return;
    }

    uint depthBits = asuint(gk_depths[idx]);

    float rMinX = gk_rectMin[idx * 2 + 0];
    float rMinY = gk_rectMin[idx * 2 + 1];
    float rMaxX = gk_rectMax[idx * 2 + 0];
    float rMaxY = gk_rectMax[idx * 2 + 1];

    int tMinX = int(floor(rMinX / float(tileW)));
    int tMinY = int(floor(rMinY / float(tileH)));
    int tMaxX = int(floor(rMaxX / float(tileW))) + 1;
    int tMaxY = int(floor(rMaxY / float(tileH))) + 1;

    int gridW = int((imageW + tileW - 1) / tileW);
    int gridH = int((imageH + tileH - 1) / tileH);

    tMinX = max(0, min(tMinX, gridW));
    tMinY = max(0, min(tMinY, gridH));
    tMaxX = max(0, min(tMaxX, gridW));
    tMaxY = max(0, min(tMaxY, gridH));

    uint offset = gk_offsets[idx];
    for (int ty = tMinY; ty < tMaxY; ty++)
    {
        for (int tx = tMinX; tx < tMaxX; tx++)
        {
            uint tileID = uint(ty * gridW + tx);
            gk_keysHigh[offset] = tileID;
            gk_keysLow[offset] = depthBits;
            gk_gaussIdx[offset] = idx;
            offset++;
        }
    }
}

// ---------------------------------------------------------------------------
// compute_tile_ranges
// ---------------------------------------------------------------------------
StructuredBuffer<uint> tr_sortedKeysHigh; // [M]
StructuredBuffer<uint> tr_counts;         // [0]=M, [1]=numTiles
RWStructuredBuffer<uint> tr_tileRanges;   // [numTiles,2]

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_tile_ranges(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint M = tr_counts[0];
    if (idx >= M)
    {
        return;
    }

    uint curTile = tr_sortedKeysHigh[idx];
    if (idx == 0)
    {
        tr_tileRanges[curTile * 2 + 0] = 0;
    }
    else
    {
        uint prevTile = tr_sortedKeysHigh[idx - 1];
        if (curTile != prevTile)
        {
            tr_tileRanges[prevTile * 2 + 1] = idx;
            tr_tileRanges[curTile * 2 + 0] = idx;
        }
    }

    if (idx == M - 1)
    {
        tr_tileRanges[curTile * 2 + 1] = M;
    }
}

// ---------------------------------------------------------------------------
// compute_tile_counts_from_ranges
// ---------------------------------------------------------------------------
StructuredBuffer<uint> ctc_tileRanges;    // [numTiles,2]
StructuredBuffer<uint> ctc_tileCountArgs; // [0]=numTiles
RWStructuredBuffer<uint> ctc_tileCounts;  // [numTiles]

[shader("compute")]
[numthreads(256, 1, 1)]
void compute_tile_counts_from_ranges(uint3 tid : SV_DispatchThreadID)
{
    uint tileIndex = tid.x;
    uint numTiles = ctc_tileCountArgs[0];
    if (tileIndex >= numTiles)
    {
        return;
    }

    uint start = ctc_tileRanges[tileIndex * 2 + 0];
    uint end = ctc_tileRanges[tileIndex * 2 + 1];
    ctc_tileCounts[tileIndex] = end > start ? (end - start) : 0;
}

// ---------------------------------------------------------------------------
// build_packed_tile_indices
// ---------------------------------------------------------------------------
StructuredBuffer<uint> bpi_sortedGaussIdx;  // [M]
StructuredBuffer<uint> bpi_tileRanges;      // [numTiles,2]
StructuredBuffer<uint> bpi_packArgs;        // [0]=numTiles, [1]=maxTilePairs
RWStructuredBuffer<int> bpi_packedTileIndices; // [numTiles * maxTilePairs]

[shader("compute")]
[numthreads(256, 1, 1)]
void build_packed_tile_indices(uint3 tid : SV_DispatchThreadID)
{
    uint linear = tid.x;
    uint numTiles = bpi_packArgs[0];
    uint maxTilePairs = bpi_packArgs[1];
    uint total = numTiles * maxTilePairs;
    if (linear >= total)
    {
        return;
    }

    uint tileIndex = linear / maxTilePairs;
    uint slot = linear - tileIndex * maxTilePairs;
    uint start = bpi_tileRanges[tileIndex * 2 + 0];
    uint end = bpi_tileRanges[tileIndex * 2 + 1];
    uint count = end > start ? (end - start) : 0;

    if (slot < count)
    {
        bpi_packedTileIndices[linear] = int(bpi_sortedGaussIdx[start + slot]);
    }
    else
    {
        bpi_packedTileIndices[linear] = 0;
    }
}

// ---------------------------------------------------------------------------
// gaussian_tile_global_forward
// ---------------------------------------------------------------------------
StructuredBuffer<float> gtf_packedGaussians; // [N,11]
StructuredBuffer<int> gtf_packedTileIndices; // [numTiles,maxTilePairs]
StructuredBuffer<uint> gtf_tileCounts;       // [numTiles]
StructuredBuffer<uint> gtf_renderCounts;     // [0]=pixelCount,[1]=maxTilePairs,[2]=gridW,[3]=tileW,[4]=tileH,[5]=imageW,[6]=imageH,[7]=whiteBg

RWStructuredBuffer<float> gtf_outColor; // [pixelCount,3]
RWStructuredBuffer<float> gtf_outDepth; // [pixelCount]
RWStructuredBuffer<float> gtf_outAlpha; // [pixelCount]

[shader("compute")]
[numthreads(256, 1, 1)]
void gaussian_tile_global_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pixelCount = gtf_renderCounts[0];
    if (p >= pixelCount)
    {
        return;
    }

    uint maxTilePairs = gtf_renderCounts[1];
    uint gridW = gtf_renderCounts[2];
    uint tileW = gtf_renderCounts[3];
    uint tileH = gtf_renderCounts[4];
    uint imageW = gtf_renderCounts[5];
    uint imageH = gtf_renderCounts[6];
    uint whiteBg = gtf_renderCounts[7];

    uint y = p / imageW;
    uint x = p - y * imageW;
    if (y >= imageH)
    {
        return;
    }

    uint tileX = x / tileW;
    uint tileY = y / tileH;
    uint tileIndex = tileY * gridW + tileX;
    uint count = gtf_tileCounts[tileIndex];
    uint base = tileIndex * maxTilePairs;

    float px = float(x);
    float py = float(y);
    float T = 1.0;
    float accAlpha = 0.0;
    float accumColorX = 0.0;
    float accumColorY = 0.0;
    float accumColorZ = 0.0;
    float accumDepth = 0.0;

    for (uint i = 0; i < count; ++i)
    {
        uint gi = uint(gtf_packedTileIndices[base + i]);
        uint gBase = gi * 11;

        float meanX = gtf_packedGaussians[gBase + 0];
        float meanY = gtf_packedGaussians[gBase + 1];
        float c00 = gtf_packedGaussians[gBase + 2];
        float c01 = gtf_packedGaussians[gBase + 3];
        float c10 = gtf_packedGaussians[gBase + 4];
        float c11 = gtf_packedGaussians[gBase + 5];
        float colorX = gtf_packedGaussians[gBase + 6];
        float colorY = gtf_packedGaussians[gBase + 7];
        float colorZ = gtf_packedGaussians[gBase + 8];
        float opacity = gtf_packedGaussians[gBase + 9];
        float depth = gtf_packedGaussians[gBase + 10];

        float dx = px - meanX;
        float dy = py - meanY;
        float dxdy = dx * dy;
        float exponent = -0.5 * (dx * dx * c00 + dy * dy * c11 + dxdy * c01 + dxdy * c10);
        float raw = exp(exponent) * opacity;
        float alpha = raw > 0.99 ? 0.99 : raw;

        float contrib = T * alpha;
        accAlpha += contrib;
        accumColorX += contrib * colorX;
        accumColorY += contrib * colorY;
        accumColorZ += contrib * colorZ;
        accumDepth += contrib * depth;
        T *= (1.0 - alpha);
        if (T < 1e-4)
        {
            break;
        }
    }

    float bg = (whiteBg != 0) ? (1.0 - accAlpha) : 0.0;
    uint colorBase = p * 3;
    gtf_outColor[colorBase + 0] = accumColorX + bg;
    gtf_outColor[colorBase + 1] = accumColorY + bg;
    gtf_outColor[colorBase + 2] = accumColorZ + bg;
    gtf_outDepth[p] = accumDepth;
    gtf_outAlpha[p] = accAlpha;
}

// ---------------------------------------------------------------------------
// gaussian_tile_global_backward
// ---------------------------------------------------------------------------
StructuredBuffer<float> gtb_packedGaussians; // [N,11]
StructuredBuffer<int> gtb_packedTileIndices; // [numTiles,maxTilePairs]
StructuredBuffer<uint> gtb_tileCounts;       // [numTiles]
StructuredBuffer<float> gtb_cotColor;        // [pixelCount,3]
StructuredBuffer<float> gtb_cotDepth;        // [pixelCount]
StructuredBuffer<float> gtb_cotAlpha;        // [pixelCount]
StructuredBuffer<uint> gtb_renderCounts;     // [0]=pixelCount,[1]=maxTilePairs,[2]=gridW,[3]=tileW,[4]=tileH,[5]=imageW,[6]=imageH,[7]=whiteBg

RWStructuredBuffer<float> gtb_gradPackedGaussians; // [N,11]

[shader("compute")]
[numthreads(256, 1, 1)]
void gaussian_tile_global_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pixelCount = gtb_renderCounts[0];
    if (p >= pixelCount)
    {
        return;
    }

    uint maxTilePairs = gtb_renderCounts[1];
    uint gridW = gtb_renderCounts[2];
    uint tileW = gtb_renderCounts[3];
    uint tileH = gtb_renderCounts[4];
    uint imageW = gtb_renderCounts[5];
    uint imageH = gtb_renderCounts[6];
    uint whiteBg = gtb_renderCounts[7];

    uint y = p / imageW;
    uint x = p - y * imageW;
    if (y >= imageH)
    {
        return;
    }

    uint tileX = x / tileW;
    uint tileY = y / tileH;
    uint tileIndex = tileY * gridW + tileX;
    uint count = gtb_tileCounts[tileIndex];
    if (count == 0)
    {
        return;
    }
    uint base = tileIndex * maxTilePairs;

    uint colorBaseP = p * 3;
    float gColorX = gtb_cotColor[colorBaseP + 0];
    float gColorY = gtb_cotColor[colorBaseP + 1];
    float gColorZ = gtb_cotColor[colorBaseP + 2];
    float gDepth = gtb_cotDepth[p];
    float gAlpha = gtb_cotAlpha[p];
    float whiteTerm = (whiteBg != 0) ? -(gColorX + gColorY + gColorZ) : 0.0;

    float px = float(x);
    float py = float(y);
    float trans = 1.0;
    float weightedSum = 0.0;
    for (uint i = 0; i < count; ++i)
    {
        uint gi = uint(gtb_packedTileIndices[base + i]);
        uint gBase = gi * 11;

        float meanX = gtb_packedGaussians[gBase + 0];
        float meanY = gtb_packedGaussians[gBase + 1];
        float c00 = gtb_packedGaussians[gBase + 2];
        float c01 = gtb_packedGaussians[gBase + 3];
        float c10 = gtb_packedGaussians[gBase + 4];
        float c11 = gtb_packedGaussians[gBase + 5];
        float colorX = gtb_packedGaussians[gBase + 6];
        float colorY = gtb_packedGaussians[gBase + 7];
        float colorZ = gtb_packedGaussians[gBase + 8];
        float opacity = gtb_packedGaussians[gBase + 9];
        float depth = gtb_packedGaussians[gBase + 10];

        float dx = px - meanX;
        float dy = py - meanY;
        float dxdy = dx * dy;
        float exponent = -0.5 * (dx * dx * c00 + dy * dy * c11 + dxdy * c01 + dxdy * c10);
        float expVal = exp(exponent);
        float raw = expVal * opacity;
        float alpha = raw > 0.99 ? 0.99 : raw;

        float contrib = trans * alpha;
        float c = gColorX * colorX + gColorY * colorY + gColorZ * colorZ + gDepth * depth + gAlpha + whiteTerm;
        weightedSum += c * contrib;

        trans *= (1.0 - alpha);
        if (trans < 1e-4)
        {
            break;
        }
    }

    trans = 1.0;
    float prefixWeighted = 0.0;
    for (uint i = 0; i < count; ++i)
    {
        uint gi = uint(gtb_packedTileIndices[base + i]);
        uint gBase = gi * 11;

        float meanX = gtb_packedGaussians[gBase + 0];
        float meanY = gtb_packedGaussians[gBase + 1];
        float c00 = gtb_packedGaussians[gBase + 2];
        float c01 = gtb_packedGaussians[gBase + 3];
        float c10 = gtb_packedGaussians[gBase + 4];
        float c11 = gtb_packedGaussians[gBase + 5];
        float colorX = gtb_packedGaussians[gBase + 6];
        float colorY = gtb_packedGaussians[gBase + 7];
        float colorZ = gtb_packedGaussians[gBase + 8];
        float opacity = gtb_packedGaussians[gBase + 9];
        float depth = gtb_packedGaussians[gBase + 10];

        float dx = px - meanX;
        float dy = py - meanY;
        float dxdy = dx * dy;
        float exponent = -0.5 * (dx * dx * c00 + dy * dy * c11 + dxdy * c01 + dxdy * c10);
        float expVal = exp(exponent);
        float raw = expVal * opacity;
        float alpha = raw > 0.99 ? 0.99 : raw;
        float contrib = trans * alpha;

        float c = gColorX * colorX + gColorY * colorY + gColorZ * colorZ + gDepth * depth + gAlpha + whiteTerm;
        prefixWeighted += c * contrib;

        float suffixWeighted = weightedSum - prefixWeighted;
        float denom = 1.0 - alpha;
        if (denom < 1e-6)
        {
            denom = 1e-6;
        }
        float gradAlpha = c * trans - suffixWeighted / denom;

        InterlockedAdd(gtb_gradPackedGaussians[gBase + 10], gDepth * contrib);
        InterlockedAdd(gtb_gradPackedGaussians[gBase + 6], gColorX * contrib);
        InterlockedAdd(gtb_gradPackedGaussians[gBase + 7], gColorY * contrib);
        InterlockedAdd(gtb_gradPackedGaussians[gBase + 8], gColorZ * contrib);

        if (raw <= 0.99)
        {
            InterlockedAdd(gtb_gradPackedGaussians[gBase + 9], gradAlpha * expVal);

            float gradExponent = gradAlpha * raw;
            InterlockedAdd(gtb_gradPackedGaussians[gBase + 2], gradExponent * (-0.5 * dx * dx));
            float gradOffDiag = gradExponent * (-0.5 * dxdy);
            InterlockedAdd(gtb_gradPackedGaussians[gBase + 3], gradOffDiag);
            InterlockedAdd(gtb_gradPackedGaussians[gBase + 4], gradOffDiag);
            InterlockedAdd(gtb_gradPackedGaussians[gBase + 5], gradExponent * (-0.5 * dy * dy));

            float c01sum = c01 + c10;
            float gradDy = gradExponent * (-dy * c11 - 0.5 * dx * c01sum);
            InterlockedAdd(
                gtb_gradPackedGaussians[gBase + 0],
                -(gradExponent * (-dx * c00 - 0.5 * dy * c01sum))
            );
            InterlockedAdd(gtb_gradPackedGaussians[gBase + 1], -gradDy);
        }

        trans *= (1.0 - alpha);
        if (trans < 1e-4)
        {
            break;
        }
    }
}
