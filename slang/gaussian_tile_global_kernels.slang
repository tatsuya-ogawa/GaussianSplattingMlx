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
// radix_sort_tile_keys_fused_forward
// ---------------------------------------------------------------------------
StructuredBuffer<uint> rs_keysHighIn;   // [M]
StructuredBuffer<uint> rs_keysLowIn;    // [M]
StructuredBuffer<uint> rs_valuesIn;     // [M]
StructuredBuffer<uint> rs_counts;       // [0]=M, [1]=tileBitCount

RWStructuredBuffer<uint> rs_sortedKeysHigh;   // [M]
RWStructuredBuffer<uint> rs_sortedKeysLow;    // [M]
RWStructuredBuffer<uint> rs_sortedValues;     // [M]
RWStructuredBuffer<uint> rs_scratchKeysHigh;  // [M]
RWStructuredBuffer<uint> rs_scratchKeysLow;   // [M]
RWStructuredBuffer<uint> rs_scratchValues;    // [M]

static const uint RS_THREADS = 128;
static const uint RS_RADIX_BITS = 4;
static const uint RS_RADIX = 16;
static const uint RS_LOW_PASSES = 8; // 32 bits / 4 bits

groupshared uint rs_localHist[RS_THREADS * RS_RADIX];
groupshared uint rs_threadBase[RS_THREADS * RS_RADIX];

[shader("compute")]
[numthreads(128, 1, 1)]
void radix_sort_tile_keys_fused_forward(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint lane = gtid.x;
    uint M = rs_counts[0];
    if (lane >= RS_THREADS || M == 0)
    {
        return;
    }

    uint chunk = (M + RS_THREADS - 1) / RS_THREADS;
    uint start = lane * chunk;
    uint end = min(start + chunk, M);

    uint highBits = rs_counts[1];
    uint highPasses = (highBits + RS_RADIX_BITS - 1) / RS_RADIX_BITS;
    if (highPasses < 1)
    {
        highPasses = 1;
    }
    uint totalPasses = RS_LOW_PASSES + highPasses;

    bool writeToScratch = true;
    for (uint pass = 0; pass < totalPasses; ++pass)
    {
        uint localBase = lane * RS_RADIX;
        for (uint d = 0; d < RS_RADIX; ++d)
        {
            rs_localHist[localBase + d] = 0;
        }
        AllMemoryBarrierWithGroupSync();

        for (uint i = start; i < end; ++i)
        {
            uint keyHigh;
            uint keyLow;
            if (pass == 0)
            {
                keyHigh = rs_keysHighIn[i];
                keyLow = rs_keysLowIn[i];
            }
            else if (writeToScratch)
            {
                keyHigh = rs_sortedKeysHigh[i];
                keyLow = rs_sortedKeysLow[i];
            }
            else
            {
                keyHigh = rs_scratchKeysHigh[i];
                keyLow = rs_scratchKeysLow[i];
            }

            uint digit;
            if (pass < RS_LOW_PASSES)
            {
                digit = (keyLow >> (pass * RS_RADIX_BITS)) & (RS_RADIX - 1);
            }
            else
            {
                uint hiPass = pass - RS_LOW_PASSES;
                digit = (keyHigh >> (hiPass * RS_RADIX_BITS)) & (RS_RADIX - 1);
            }
            rs_localHist[localBase + digit] += 1;
        }
        AllMemoryBarrierWithGroupSync();

        if (lane == 0)
        {
            uint running = 0;
            for (uint d = 0; d < RS_RADIX; ++d)
            {
                uint offset = running;
                for (uint t = 0; t < RS_THREADS; ++t)
                {
                    uint idx = t * RS_RADIX + d;
                    rs_threadBase[idx] = offset;
                    offset += rs_localHist[idx];
                }
                running = offset;
            }
        }
        AllMemoryBarrierWithGroupSync();

        uint localOffsets[RS_RADIX];
        for (uint d = 0; d < RS_RADIX; ++d)
        {
            localOffsets[d] = 0;
        }

        for (uint i = start; i < end; ++i)
        {
            uint keyHigh;
            uint keyLow;
            uint value;
            if (pass == 0)
            {
                keyHigh = rs_keysHighIn[i];
                keyLow = rs_keysLowIn[i];
                value = rs_valuesIn[i];
            }
            else if (writeToScratch)
            {
                keyHigh = rs_sortedKeysHigh[i];
                keyLow = rs_sortedKeysLow[i];
                value = rs_sortedValues[i];
            }
            else
            {
                keyHigh = rs_scratchKeysHigh[i];
                keyLow = rs_scratchKeysLow[i];
                value = rs_scratchValues[i];
            }

            uint digit;
            if (pass < RS_LOW_PASSES)
            {
                digit = (keyLow >> (pass * RS_RADIX_BITS)) & (RS_RADIX - 1);
            }
            else
            {
                uint hiPass = pass - RS_LOW_PASSES;
                digit = (keyHigh >> (hiPass * RS_RADIX_BITS)) & (RS_RADIX - 1);
            }

            uint dst = rs_threadBase[localBase + digit] + localOffsets[digit];
            localOffsets[digit] += 1;
            if (writeToScratch)
            {
                rs_scratchKeysHigh[dst] = keyHigh;
                rs_scratchKeysLow[dst] = keyLow;
                rs_scratchValues[dst] = value;
            }
            else
            {
                rs_sortedKeysHigh[dst] = keyHigh;
                rs_sortedKeysLow[dst] = keyLow;
                rs_sortedValues[dst] = value;
            }
        }
        AllMemoryBarrierWithGroupSync();
        writeToScratch = !writeToScratch;
    }

    // If pass count is odd, final values are in scratch.
    if (!writeToScratch)
    {
        for (uint i = start; i < end; ++i)
        {
            rs_sortedKeysHigh[i] = rs_scratchKeysHigh[i];
            rs_sortedKeysLow[i] = rs_scratchKeysLow[i];
            rs_sortedValues[i] = rs_scratchValues[i];
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
RWStructuredBuffer<uint> gtf_outLastContrib; // [pixelCount]

struct TileGlobalBlendStepOutputs : IDifferentiable
{
    float colorX;
    float colorY;
    float colorZ;
    float depth;
    float alphaAcc;
};

[Differentiable]
float tileGlobalAlphaFromGaussian(
    float meanX,
    float meanY,
    float c00,
    float c01,
    float c10,
    float c11,
    float opacity,
    no_diff float px,
    no_diff float py
)
{
    float dx = px - meanX;
    float dy = py - meanY;
    float dxdy = dx * dy;
    float exponent = -0.5 * (dx * dx * c00 + dy * dy * c11 + dxdy * c01 + dxdy * c10);
    float raw = exp(exponent) * opacity;
    return raw > 0.99 ? 0.99 : raw;
}

[Differentiable]
TileGlobalBlendStepOutputs tileGlobalBlendStep(
    float T_i,
    float alpha,
    float colorX,
    float colorY,
    float colorZ,
    float depth
)
{
    float contrib = T_i * alpha;
    TileGlobalBlendStepOutputs outVals;
    outVals.colorX = contrib * colorX;
    outVals.colorY = contrib * colorY;
    outVals.colorZ = contrib * colorZ;
    outVals.depth = contrib * depth;
    outVals.alphaAcc = contrib;
    return outVals;
}

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
    uint nContrib = count;

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

        float alpha = tileGlobalAlphaFromGaussian(
            meanX,
            meanY,
            c00,
            c01,
            c10,
            c11,
            opacity,
            px,
            py
        );

        TileGlobalBlendStepOutputs stepVals = tileGlobalBlendStep(
            T,
            alpha,
            colorX,
            colorY,
            colorZ,
            depth
        );
        accAlpha += stepVals.alphaAcc;
        accumColorX += stepVals.colorX;
        accumColorY += stepVals.colorY;
        accumColorZ += stepVals.colorZ;
        accumDepth += stepVals.depth;
        T *= (1.0 - alpha);
        if (T < 1e-4)
        {
            nContrib = i + 1;
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
    gtf_outLastContrib[p] = nContrib;
}

// ---------------------------------------------------------------------------
// gaussian_tile_global_backward (1-pass reverse traversal + shared memory + WaveActiveSum)
// ---------------------------------------------------------------------------
StructuredBuffer<float> gtb_packedGaussians; // [N,11]
StructuredBuffer<int> gtb_packedTileIndices; // [numTiles,maxTilePairs]
StructuredBuffer<uint> gtb_tileCounts;       // [numTiles]
StructuredBuffer<float> gtb_cotColor;        // [pixelCount,3]
StructuredBuffer<float> gtb_cotDepth;        // [pixelCount]
StructuredBuffer<float> gtb_cotAlpha;        // [pixelCount]
StructuredBuffer<uint> gtb_renderCounts;     // [0]=pixelCount,[1]=maxTilePairs,[2]=gridW,[3]=tileW,[4]=tileH,[5]=imageW,[6]=imageH,[7]=whiteBg
StructuredBuffer<float> gtb_outAlpha;        // [pixelCount] forward output alpha
StructuredBuffer<uint> gtb_lastContrib;      // [pixelCount] forward nContrib per pixel

RWStructuredBuffer<float> gtb_gradPackedGaussians; // [N,11]

static const uint GTB_THREADS = 256;

groupshared float sgMeanX[GTB_THREADS];
groupshared float sgMeanY[GTB_THREADS];
groupshared float sgC00[GTB_THREADS];
groupshared float sgC01[GTB_THREADS];
groupshared float sgC10[GTB_THREADS];
groupshared float sgC11[GTB_THREADS];
groupshared float sgColorX[GTB_THREADS];
groupshared float sgColorY[GTB_THREADS];
groupshared float sgColorZ[GTB_THREADS];
groupshared float sgOpacity[GTB_THREADS];
groupshared float sgDepth[GTB_THREADS];
groupshared uint sgGaussIdx[GTB_THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void gaussian_tile_global_backward(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    // 2D tile-based dispatch: tid.y = tile index, tid.x = local pixel within tile
    uint tileIndex = tid.y;
    uint localPixel = tid.x;
    uint lane = gtid.x;

    uint maxTilePairs = gtb_renderCounts[1];
    uint gridW = gtb_renderCounts[2];
    uint tileW = gtb_renderCounts[3];
    uint tileH = gtb_renderCounts[4];
    uint imageW = gtb_renderCounts[5];
    uint imageH = gtb_renderCounts[6];
    uint whiteBg = gtb_renderCounts[7];
    uint tilePixels = tileW * tileH;

    // Compute pixel coordinates from tile position
    uint tileX = tileIndex % gridW;
    uint tileY = tileIndex / gridW;
    uint inTileX = localPixel < tilePixels ? (localPixel % tileW) : 0;
    uint inTileY = localPixel < tilePixels ? (localPixel / tileW) : 0;
    uint pixelX = tileX * tileW + inTileX;
    uint pixelY = tileY * tileH + inTileY;
    bool activePixel = (localPixel < tilePixels && pixelX < imageW && pixelY < imageH);

    uint count = gtb_tileCounts[tileIndex];
    uint tileBase = tileIndex * maxTilePairs;

    // Load per-pixel gradient inputs and forward context
    float gColorX = 0.0;
    float gColorY = 0.0;
    float gColorZ = 0.0;
    float gDepth = 0.0;
    float gAlpha = 0.0;
    float whiteTerm = 0.0;
    float px = 0.0;
    float py = 0.0;
    float T_final = 0.0;
    uint n_contrib = 0;
    if (activePixel)
    {
        uint p = pixelY * imageW + pixelX;
        uint colorBase = p * 3;
        gColorX = gtb_cotColor[colorBase];
        gColorY = gtb_cotColor[colorBase + 1];
        gColorZ = gtb_cotColor[colorBase + 2];
        gDepth = gtb_cotDepth[p];
        gAlpha = gtb_cotAlpha[p];
        if (whiteBg != 0)
        {
            whiteTerm = -(gColorX + gColorY + gColorZ);
        }
        px = float(pixelX);
        py = float(pixelY);
        T_final = 1.0 - gtb_outAlpha[p];
        n_contrib = gtb_lastContrib[p];
    }

    // ---- Single reverse pass: gradient computation + WaveActiveSum reduction ----
    uint numChunks = (count + GTB_THREADS - 1) / GTB_THREADS;
    bool passDone = !activePixel;
    float T = T_final;
    float suffix = 0.0;

    for (int c = int(numChunks) - 1; c >= 0; c--)
    {
        uint chunkStart = uint(c) * GTB_THREADS;

        // Cooperative load to shared memory (forward direction is fine)
        uint loadIdx = chunkStart + lane;
        if (loadIdx < count)
        {
            uint gi = uint(gtb_packedTileIndices[tileBase + loadIdx]);
            uint gBase = gi * 11;
            sgGaussIdx[lane] = gi;
            sgMeanX[lane] = gtb_packedGaussians[gBase];
            sgMeanY[lane] = gtb_packedGaussians[gBase + 1];
            sgC00[lane] = gtb_packedGaussians[gBase + 2];
            sgC01[lane] = gtb_packedGaussians[gBase + 3];
            sgC10[lane] = gtb_packedGaussians[gBase + 4];
            sgC11[lane] = gtb_packedGaussians[gBase + 5];
            sgColorX[lane] = gtb_packedGaussians[gBase + 6];
            sgColorY[lane] = gtb_packedGaussians[gBase + 7];
            sgColorZ[lane] = gtb_packedGaussians[gBase + 8];
            sgOpacity[lane] = gtb_packedGaussians[gBase + 9];
            sgDepth[lane] = gtb_packedGaussians[gBase + 10];
        }
        AllMemoryBarrierWithGroupSync();

        uint chunkEnd = min(chunkStart + GTB_THREADS, count);

        // Reverse inner loop within this chunk
        for (int ii = int(chunkEnd) - 1; ii >= int(chunkStart); ii--)
        {
            uint local = uint(ii) - chunkStart;

            // Each thread computes its per-gaussian gradient (inactive threads contribute 0.0)
            float g_meanX = 0.0; float g_meanY = 0.0;
            float g_c00 = 0.0; float g_c01 = 0.0; float g_c10 = 0.0; float g_c11 = 0.0;
            float g_colorX = 0.0; float g_colorY = 0.0; float g_colorZ = 0.0;
            float g_opacity = 0.0; float g_depth = 0.0;

            if (!passDone && uint(ii) < n_contrib)
            {
                float meanX = sgMeanX[local];
                float meanY = sgMeanY[local];
                float c00 = sgC00[local];
                float c01 = sgC01[local];
                float c10 = sgC10[local];
                float c11 = sgC11[local];
                float colorX = sgColorX[local];
                float colorY = sgColorY[local];
                float colorZ = sgColorZ[local];
                float opacity = sgOpacity[local];
                float depth = sgDepth[local];
                float alpha = tileGlobalAlphaFromGaussian(
                    meanX,
                    meanY,
                    c00,
                    c01,
                    c10,
                    c11,
                    opacity,
                    px,
                    py
                );

                float denom = 1.0 - alpha;
                if (denom < 1e-6) { denom = 1e-6; }

                // Recover transmittance before this gaussian
                float T_i = T / denom;
                float contrib = T_i * alpha;
                float cotAlphaAcc = gAlpha + whiteTerm;
                float cVal = gColorX * colorX + gColorY * colorY + gColorZ * colorZ + gDepth * depth + cotAlphaAcc;
                float suffixBefore = suffix;

                // Accumulate suffix for next (earlier) gaussian
                suffix += cVal * contrib;
                T = T_i;

                DifferentialPair<float> dpTi = diffPair(T_i, 0.0);
                DifferentialPair<float> dpAlpha = diffPair(alpha, 0.0);
                DifferentialPair<float> dpColorX = diffPair(colorX, g_colorX);
                DifferentialPair<float> dpColorY = diffPair(colorY, g_colorY);
                DifferentialPair<float> dpColorZ = diffPair(colorZ, g_colorZ);
                DifferentialPair<float> dpDepth = diffPair(depth, g_depth);
                TileGlobalBlendStepOutputs cotBlend;
                cotBlend.colorX = gColorX;
                cotBlend.colorY = gColorY;
                cotBlend.colorZ = gColorZ;
                cotBlend.depth = gDepth;
                cotBlend.alphaAcc = cotAlphaAcc;
                bwd_diff(tileGlobalBlendStep)(
                    dpTi,
                    dpAlpha,
                    dpColorX,
                    dpColorY,
                    dpColorZ,
                    dpDepth,
                    cotBlend
                );
                g_colorX = dpColorX.d;
                g_colorY = dpColorY.d;
                g_colorZ = dpColorZ.d;
                g_depth = dpDepth.d;

                float gradAlpha = dpAlpha.d - suffixBefore / denom;

                // Hybrid AD: keep tile-level reverse traversal/reduction handwritten,
                // and let AD differentiate the per-gaussian alpha math.
                DifferentialPair<float> dpMeanX = diffPair(meanX, g_meanX);
                DifferentialPair<float> dpMeanY = diffPair(meanY, g_meanY);
                DifferentialPair<float> dpC00 = diffPair(c00, g_c00);
                DifferentialPair<float> dpC01 = diffPair(c01, g_c01);
                DifferentialPair<float> dpC10 = diffPair(c10, g_c10);
                DifferentialPair<float> dpC11 = diffPair(c11, g_c11);
                DifferentialPair<float> dpOpacity = diffPair(opacity, g_opacity);
                bwd_diff(tileGlobalAlphaFromGaussian)(
                    dpMeanX,
                    dpMeanY,
                    dpC00,
                    dpC01,
                    dpC10,
                    dpC11,
                    dpOpacity,
                    px,
                    py,
                    gradAlpha
                );

                g_meanX = dpMeanX.d;
                g_meanY = dpMeanY.d;
                g_c00 = dpC00.d;
                g_c01 = dpC01.d;
                g_c10 = dpC10.d;
                g_c11 = dpC11.d;
                g_opacity = dpOpacity.d;
            }

            // SIMD group reduction: sum gradients across 32 threads
            float s_meanX = WaveActiveSum(g_meanX);
            float s_meanY = WaveActiveSum(g_meanY);
            float s_c00 = WaveActiveSum(g_c00);
            float s_c01 = WaveActiveSum(g_c01);
            float s_c10 = WaveActiveSum(g_c10);
            float s_c11 = WaveActiveSum(g_c11);
            float s_colorX = WaveActiveSum(g_colorX);
            float s_colorY = WaveActiveSum(g_colorY);
            float s_colorZ = WaveActiveSum(g_colorZ);
            float s_opacity = WaveActiveSum(g_opacity);
            float s_depth = WaveActiveSum(g_depth);

            // Only lane 0 of each SIMD group writes atomically
            if (WaveGetLaneIndex() == 0)
            {
                uint gBaseOut = sgGaussIdx[local] * 11;
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 0], s_meanX);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 1], s_meanY);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 2], s_c00);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 3], s_c01);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 4], s_c10);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 5], s_c11);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 6], s_colorX);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 7], s_colorY);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 8], s_colorZ);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 9], s_opacity);
                InterlockedAdd(gtb_gradPackedGaussians[gBaseOut + 10], s_depth);
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
}
