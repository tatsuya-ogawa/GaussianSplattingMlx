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
