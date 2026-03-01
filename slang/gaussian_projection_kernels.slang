// gaussian_projection_kernels.slang
//
// Active kernels:
// - gaussian_projection_screen_fused_forward
// - gaussian_projection_screen_fused_backward

import gaussian_projection_screen_shared;

// ---------------------------------------------------------------------------
// gaussian_projection_screen_fused_forward
// ---------------------------------------------------------------------------
StructuredBuffer<float> fused_scales;        // [N*3]
StructuredBuffer<float> fused_rotations;     // [N*4]
StructuredBuffer<float> fused_means3d;       // [N*3]
StructuredBuffer<float> fused_shs;           // [N*K*3]
StructuredBuffer<float> fused_cameraCenter;  // [1*3]
StructuredBuffer<float> fused_viewMatrix;    // [16]
StructuredBuffer<float> fused_projMatrix;    // [16]
StructuredBuffer<float> fused_fovX;          // [1]
StructuredBuffer<float> fused_fovY;          // [1]
StructuredBuffer<float> fused_focalX;        // [1]
StructuredBuffer<float> fused_focalY;        // [1]
StructuredBuffer<float> fused_imageWidth;    // [1]
StructuredBuffer<float> fused_imageHeight;   // [1]
StructuredBuffer<uint>  fused_counts;        // [pointCount, activeShDegree, shStride]

RWStructuredBuffer<float> fused_outMeans2d;  // [N*2]
RWStructuredBuffer<float> fused_outDepths;   // [N]
RWStructuredBuffer<float> fused_outColor;    // [N*3]
RWStructuredBuffer<float> fused_outCov2d;    // [N*4]
RWStructuredBuffer<float> fused_outConic;    // [N*4]
RWStructuredBuffer<float> fused_outRadii;    // [N]

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_projection_screen_fused_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = fused_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    uint meansBase = p * 3;
    float m0 = fused_means3d[meansBase + 0];
    float m1 = fused_means3d[meansBase + 1];
    float m2 = fused_means3d[meansBase + 2];

    int degree = int(fused_counts[1]);
    int coeffCount = (degree + 1) * (degree + 1);
    if (coeffCount > 25)
    {
        coeffCount = 25;
    }
    int coeffStride = int(fused_counts[2]);
    uint shPointBase = p * uint(coeffStride * 3);
    float3 shCoeffs[25];
    for (int k = 0; k < 25; ++k)
    {
        if (k < coeffCount)
        {
            uint shBase = shPointBase + uint(k * 3);
            shCoeffs[k] = float3(
                fused_shs[shBase + 0],
                fused_shs[shBase + 1],
                fused_shs[shBase + 2]
            );
        }
        else
        {
            shCoeffs[k] = float3(0.0);
        }
    }

    uint scaleBase = p * 3;
    uint rotBase = p * 4;
    ProjectionScreenFusedCoreOutputs fusedOut = evaluateProjectionScreenFusedCoreOutputs(
        float3(
            fused_scales[scaleBase + 0],
            fused_scales[scaleBase + 1],
            fused_scales[scaleBase + 2]
        ),
        float4(
            fused_rotations[rotBase + 0],
            fused_rotations[rotBase + 1],
            fused_rotations[rotBase + 2],
            fused_rotations[rotBase + 3]
        ),
        float3(m0, m1, m2),
        float3(fused_cameraCenter[0], fused_cameraCenter[1], fused_cameraCenter[2]),
        shCoeffs,
        degree,
        fused_viewMatrix[0], fused_viewMatrix[1], fused_viewMatrix[2], fused_viewMatrix[3],
        fused_viewMatrix[4], fused_viewMatrix[5], fused_viewMatrix[6], fused_viewMatrix[7],
        fused_viewMatrix[8], fused_viewMatrix[9], fused_viewMatrix[10], fused_viewMatrix[11],
        fused_viewMatrix[12], fused_viewMatrix[13], fused_viewMatrix[14], fused_viewMatrix[15],
        fused_projMatrix[0], fused_projMatrix[1], fused_projMatrix[2], fused_projMatrix[3],
        fused_projMatrix[4], fused_projMatrix[5], fused_projMatrix[6], fused_projMatrix[7],
        fused_projMatrix[8], fused_projMatrix[9], fused_projMatrix[10], fused_projMatrix[11],
        fused_projMatrix[12], fused_projMatrix[13], fused_projMatrix[14], fused_projMatrix[15],
        fused_fovX[0],
        fused_fovY[0],
        fused_focalX[0],
        fused_focalY[0],
        fused_imageWidth[0],
        fused_imageHeight[0]
    );

    uint means2dBase = p * 2;
    fused_outMeans2d[means2dBase + 0] = fusedOut.means2d.x;
    fused_outMeans2d[means2dBase + 1] = fusedOut.means2d.y;
    fused_outDepths[p] = fusedOut.depth;

    uint colorBase = p * 3;
    fused_outColor[colorBase + 0] = fusedOut.color.x;
    fused_outColor[colorBase + 1] = fusedOut.color.y;
    fused_outColor[colorBase + 2] = fusedOut.color.z;

    uint covBase = p * 4;
    fused_outCov2d[covBase + 0] = fusedOut.cov2d.c00;
    fused_outCov2d[covBase + 1] = fusedOut.cov2d.c01;
    fused_outCov2d[covBase + 2] = fusedOut.cov2d.c10;
    fused_outCov2d[covBase + 3] = fusedOut.cov2d.c11;
    fused_outConic[covBase + 0] = fusedOut.conic.x;
    fused_outConic[covBase + 1] = fusedOut.conic.y;
    fused_outConic[covBase + 2] = fusedOut.conic.z;
    fused_outConic[covBase + 3] = fusedOut.conic.w;

    float radius = computeRadiusFromCov2d(fusedOut.cov2d);
    float visibleMask = (fusedOut.depth >= 0.2f) ? 1.0f : 0.0f;
    fused_outRadii[p] = radius * visibleMask;
}

// ---------------------------------------------------------------------------
// gaussian_projection_screen_fused_backward
//   Thin backward wrapper driven by bwd_diff over shared differentiable funcs.
// ---------------------------------------------------------------------------
StructuredBuffer<float> fused_bwd_scales;               // [N*3]
StructuredBuffer<float> fused_bwd_rotations;            // [N*4]
StructuredBuffer<float> fused_bwd_means3d;              // [N*3]
StructuredBuffer<float> fused_bwd_shs;                  // [N*K*3]
StructuredBuffer<float> fused_bwd_cameraCenter;         // [1*3]
StructuredBuffer<float> fused_bwd_viewMatrix;           // [16]
StructuredBuffer<float> fused_bwd_projMatrix;           // [16]
StructuredBuffer<float> fused_bwd_fovX;                 // [1]
StructuredBuffer<float> fused_bwd_fovY;                 // [1]
StructuredBuffer<float> fused_bwd_focalX;               // [1]
StructuredBuffer<float> fused_bwd_focalY;               // [1]
StructuredBuffer<float> fused_bwd_imageWidth;           // [1]
StructuredBuffer<float> fused_bwd_imageHeight;          // [1]
StructuredBuffer<float> fused_bwd_cotDepths;            // [N]
StructuredBuffer<float> fused_bwd_cotMeans2d;           // [N*2]
StructuredBuffer<float> fused_bwd_cotCov2d;             // [N*4]
StructuredBuffer<float> fused_bwd_cotColor;             // [N*3]
StructuredBuffer<float> fused_bwd_cotConic;             // [N*4]
StructuredBuffer<uint>  fused_bwd_counts;               // [pointCount, activeShDegree, shStride]

RWStructuredBuffer<float> fused_bwd_gradScales;         // [N*3]
RWStructuredBuffer<float> fused_bwd_gradRotations;      // [N*4]
RWStructuredBuffer<float> fused_bwd_gradMeans3d;        // [N*3]
RWStructuredBuffer<float> fused_bwd_gradShs;            // [N*K*3]
RWStructuredBuffer<float> fused_bwd_gradCameraCenterPoint; // [N*3]

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_projection_screen_fused_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = fused_bwd_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    int degree = int(fused_bwd_counts[1]);
    int coeffCount = (degree + 1) * (degree + 1);
    if (coeffCount > 25)
    {
        coeffCount = 25;
    }
    int coeffStride = int(fused_bwd_counts[2]);

    uint meansBase = p * 3;
    uint scaleBase = p * 3;
    uint rotBase = p * 4;
    float3 means3d = float3(
        fused_bwd_means3d[meansBase + 0],
        fused_bwd_means3d[meansBase + 1],
        fused_bwd_means3d[meansBase + 2]
    );
    float3 scales = float3(
        fused_bwd_scales[scaleBase + 0],
        fused_bwd_scales[scaleBase + 1],
        fused_bwd_scales[scaleBase + 2]
    );
    float4 rotations = float4(
        fused_bwd_rotations[rotBase + 0],
        fused_bwd_rotations[rotBase + 1],
        fused_bwd_rotations[rotBase + 2],
        fused_bwd_rotations[rotBase + 3]
    );
    float3 cameraCenter = float3(
        fused_bwd_cameraCenter[0],
        fused_bwd_cameraCenter[1],
        fused_bwd_cameraCenter[2]
    );

    uint shPointBase = p * uint(coeffStride * 3);
    float3 shCoeffs[25];
    float3 shCoeffDiff[25];
    for (int k = 0; k < 25; ++k)
    {
        if (k < coeffCount)
        {
            uint shBase = shPointBase + uint(k * 3);
            shCoeffs[k] = float3(
                fused_bwd_shs[shBase + 0],
                fused_bwd_shs[shBase + 1],
                fused_bwd_shs[shBase + 2]
            );
        }
        else
        {
            shCoeffs[k] = float3(0.0);
        }
        shCoeffDiff[k] = float3(0.0);
    }

    float3 scalesDiff = float3(0.0);
    float4 rotationsDiff = float4(0.0);
    float3 means3dDiff = float3(0.0);
    float3 cameraCenterDiff = float3(0.0);

    DifferentialPair<float3> dpScales = diffPair(scales, scalesDiff);
    DifferentialPair<float4> dpRotations = diffPair(rotations, rotationsDiff);
    DifferentialPair<float3> dpMeans3d = diffPair(means3d, means3dDiff);
    DifferentialPair<float3> dpCameraCenter = diffPair(cameraCenter, cameraCenterDiff);
    DifferentialPair<float3[25]> dpShCoeffs = diffPair(shCoeffs, shCoeffDiff);

    float cotDepth = fused_bwd_cotDepths[p];
    uint cotMeansBase = p * 2;
    uint cotColorBase = p * 3;
    uint cotCovBase = p * 4;
    ProjectionScreenFusedCoreOutputs cotOutputs;
    cotOutputs.depth = cotDepth;
    cotOutputs.means2d = float2(
        fused_bwd_cotMeans2d[cotMeansBase + 0],
        fused_bwd_cotMeans2d[cotMeansBase + 1]
    );
    cotOutputs.color = float3(
        fused_bwd_cotColor[cotColorBase + 0],
        fused_bwd_cotColor[cotColorBase + 1],
        fused_bwd_cotColor[cotColorBase + 2]
    );
    cotOutputs.cov2d = makeCov2d(
        fused_bwd_cotCov2d[cotCovBase + 0],
        fused_bwd_cotCov2d[cotCovBase + 1],
        fused_bwd_cotCov2d[cotCovBase + 2],
        fused_bwd_cotCov2d[cotCovBase + 3]
    );
    cotOutputs.conic = float4(
        fused_bwd_cotConic[cotCovBase + 0],
        fused_bwd_cotConic[cotCovBase + 1],
        fused_bwd_cotConic[cotCovBase + 2],
        fused_bwd_cotConic[cotCovBase + 3]
    );

    bwd_diff(evaluateProjectionScreenFusedCoreOutputs)(
        dpScales,
        dpRotations,
        dpMeans3d,
        dpCameraCenter,
        dpShCoeffs,
        degree,
        fused_bwd_viewMatrix[0], fused_bwd_viewMatrix[1], fused_bwd_viewMatrix[2], fused_bwd_viewMatrix[3],
        fused_bwd_viewMatrix[4], fused_bwd_viewMatrix[5], fused_bwd_viewMatrix[6], fused_bwd_viewMatrix[7],
        fused_bwd_viewMatrix[8], fused_bwd_viewMatrix[9], fused_bwd_viewMatrix[10], fused_bwd_viewMatrix[11],
        fused_bwd_viewMatrix[12], fused_bwd_viewMatrix[13], fused_bwd_viewMatrix[14], fused_bwd_viewMatrix[15],
        fused_bwd_projMatrix[0], fused_bwd_projMatrix[1], fused_bwd_projMatrix[2], fused_bwd_projMatrix[3],
        fused_bwd_projMatrix[4], fused_bwd_projMatrix[5], fused_bwd_projMatrix[6], fused_bwd_projMatrix[7],
        fused_bwd_projMatrix[8], fused_bwd_projMatrix[9], fused_bwd_projMatrix[10], fused_bwd_projMatrix[11],
        fused_bwd_projMatrix[12], fused_bwd_projMatrix[13], fused_bwd_projMatrix[14], fused_bwd_projMatrix[15],
        fused_bwd_fovX[0],
        fused_bwd_fovY[0],
        fused_bwd_focalX[0],
        fused_bwd_focalY[0],
        fused_bwd_imageWidth[0],
        fused_bwd_imageHeight[0],
        cotOutputs
    );

    for (int k = 0; k < coeffCount; ++k)
    {
        uint shBase = shPointBase + uint(k * 3);
        fused_bwd_gradShs[shBase + 0] = dpShCoeffs.d[k].x;
        fused_bwd_gradShs[shBase + 1] = dpShCoeffs.d[k].y;
        fused_bwd_gradShs[shBase + 2] = dpShCoeffs.d[k].z;
    }

    fused_bwd_gradScales[scaleBase + 0] = dpScales.d.x;
    fused_bwd_gradScales[scaleBase + 1] = dpScales.d.y;
    fused_bwd_gradScales[scaleBase + 2] = dpScales.d.z;

    fused_bwd_gradRotations[rotBase + 0] = dpRotations.d.x;
    fused_bwd_gradRotations[rotBase + 1] = dpRotations.d.y;
    fused_bwd_gradRotations[rotBase + 2] = dpRotations.d.z;
    fused_bwd_gradRotations[rotBase + 3] = dpRotations.d.w;

    fused_bwd_gradMeans3d[meansBase + 0] = dpMeans3d.d.x;
    fused_bwd_gradMeans3d[meansBase + 1] = dpMeans3d.d.y;
    fused_bwd_gradMeans3d[meansBase + 2] = dpMeans3d.d.z;

    fused_bwd_gradCameraCenterPoint[meansBase + 0] = dpCameraCenter.d.x;
    fused_bwd_gradCameraCenterPoint[meansBase + 1] = dpCameraCenter.d.y;
    fused_bwd_gradCameraCenterPoint[meansBase + 2] = dpCameraCenter.d.z;
}
