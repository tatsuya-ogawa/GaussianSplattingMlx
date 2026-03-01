// gaussian_screen_space_kernels.slang
//
// No active kernels are built from this file at the moment.
// `gaussian_screen_cov3d_backward` is kept as a reference implementation.

import gaussian_projection_screen_shared;

StructuredBuffer<float> cov3d_scales;
StructuredBuffer<float> cov3d_rotations;
StructuredBuffer<float> cov3d_cotCov3d;
StructuredBuffer<uint> cov3d_counts;
RWStructuredBuffer<float> cov3d_gradScales;
RWStructuredBuffer<float> cov3d_gradRotations;

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_cov3d_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = cov3d_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    uint scaleBase = p * 3;
    float sx = cov3d_scales[scaleBase + 0];
    float sy = cov3d_scales[scaleBase + 1];
    float sz = cov3d_scales[scaleBase + 2];

    uint rotBase = p * 4;
    float rw = cov3d_rotations[rotBase + 0];
    float rx = cov3d_rotations[rotBase + 1];
    float ry = cov3d_rotations[rotBase + 2];
    float rz = cov3d_rotations[rotBase + 3];

    float sxDiff = 0.0;
    float syDiff = 0.0;
    float szDiff = 0.0;
    float rwDiff = 0.0;
    float rxDiff = 0.0;
    float ryDiff = 0.0;
    float rzDiff = 0.0;
    DifferentialPair<float> dpSx = diffPair(sx, sxDiff);
    DifferentialPair<float> dpSy = diffPair(sy, syDiff);
    DifferentialPair<float> dpSz = diffPair(sz, szDiff);
    DifferentialPair<float> dpRw = diffPair(rw, rwDiff);
    DifferentialPair<float> dpRx = diffPair(rx, rxDiff);
    DifferentialPair<float> dpRy = diffPair(ry, ryDiff);
    DifferentialPair<float> dpRz = diffPair(rz, rzDiff);

    uint cotBase = p * 9;
    Cov3dValues cotCov3d;
    cotCov3d.c00 = cov3d_cotCov3d[cotBase + 0];
    cotCov3d.c01 = cov3d_cotCov3d[cotBase + 1];
    cotCov3d.c02 = cov3d_cotCov3d[cotBase + 2];
    cotCov3d.c10 = cov3d_cotCov3d[cotBase + 3];
    cotCov3d.c11 = cov3d_cotCov3d[cotBase + 4];
    cotCov3d.c12 = cov3d_cotCov3d[cotBase + 5];
    cotCov3d.c20 = cov3d_cotCov3d[cotBase + 6];
    cotCov3d.c21 = cov3d_cotCov3d[cotBase + 7];
    cotCov3d.c22 = cov3d_cotCov3d[cotBase + 8];

    bwd_diff(buildCov3dFromScaleRotation)(
        dpSx,
        dpSy,
        dpSz,
        dpRw,
        dpRx,
        dpRy,
        dpRz,
        cotCov3d
    );

    cov3d_gradScales[scaleBase + 0] = dpSx.d;
    cov3d_gradScales[scaleBase + 1] = dpSy.d;
    cov3d_gradScales[scaleBase + 2] = dpSz.d;

    cov3d_gradRotations[rotBase + 0] = dpRw.d;
    cov3d_gradRotations[rotBase + 1] = dpRx.d;
    cov3d_gradRotations[rotBase + 2] = dpRy.d;
    cov3d_gradRotations[rotBase + 3] = dpRz.d;
}
