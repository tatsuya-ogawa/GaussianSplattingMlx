// gaussian_projection_kernels.slang
//
// Active kernels:
// - projection_ndc_backward
// - gaussian_projection_screen_fused_forward
// - gaussian_projection_screen_fused_backward

import gaussian_projection_screen_shared;

// ---------------------------------------------------------------------------
// projection_ndc_backward
// ---------------------------------------------------------------------------
StructuredBuffer<float> proj_points;         // [N*3]
StructuredBuffer<float> proj_viewMatrix;     // [16] (4x4 column-major)
StructuredBuffer<float> proj_projMatrix;     // [16] (4x4 column-major)
StructuredBuffer<uint>  proj_counts;         // [1]: pointCount

// Backward inputs (cotangents)
StructuredBuffer<float> proj_cotMeanNdc;     // [N*4]
StructuredBuffer<float> proj_cotPView;       // [N*4]

// Backward outputs (gradients)
RWStructuredBuffer<float> proj_gradPoints;   // [N*3]

// ---------------------------------------------------------------------------
// projection_ndc_backward
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)]
void projection_ndc_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    if (p >= proj_counts[0])
    {
        return;
    }

    uint pBase = p * 3;
    float3 point = float3(
        proj_points[pBase + 0],
        proj_points[pBase + 1],
        proj_points[pBase + 2]
    );
    float3 pointDiff = float3(0.0);
    DifferentialPair<float3> dpPoint = diffPair(point, pointDiff);

    uint gBase = p * 4;
    ProjectionNdcOutputs cotOutputs;
    cotOutputs.meanNdc = float4(
        proj_cotMeanNdc[gBase + 0],
        proj_cotMeanNdc[gBase + 1],
        proj_cotMeanNdc[gBase + 2],
        proj_cotMeanNdc[gBase + 3]
    );
    cotOutputs.pView = float4(
        proj_cotPView[gBase + 0],
        proj_cotPView[gBase + 1],
        proj_cotPView[gBase + 2],
        proj_cotPView[gBase + 3]
    );

    bwd_diff(evaluateProjectionNdcOutputs)(
        dpPoint,
        proj_viewMatrix[0], proj_viewMatrix[1], proj_viewMatrix[2], proj_viewMatrix[3],
        proj_viewMatrix[4], proj_viewMatrix[5], proj_viewMatrix[6], proj_viewMatrix[7],
        proj_viewMatrix[8], proj_viewMatrix[9], proj_viewMatrix[10], proj_viewMatrix[11],
        proj_viewMatrix[12], proj_viewMatrix[13], proj_viewMatrix[14], proj_viewMatrix[15],
        proj_projMatrix[0], proj_projMatrix[1], proj_projMatrix[2], proj_projMatrix[3],
        proj_projMatrix[4], proj_projMatrix[5], proj_projMatrix[6], proj_projMatrix[7],
        proj_projMatrix[8], proj_projMatrix[9], proj_projMatrix[10], proj_projMatrix[11],
        proj_projMatrix[12], proj_projMatrix[13], proj_projMatrix[14], proj_projMatrix[15],
        cotOutputs
    );

    proj_gradPoints[pBase + 0] = dpPoint.d.x;
    proj_gradPoints[pBase + 1] = dpPoint.d.y;
    proj_gradPoints[pBase + 2] = dpPoint.d.z;
}

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

    ProjectionNdcOutputs projectionNdc = evaluateProjectionNdcOutputs(
        float3(m0, m1, m2),
        fused_viewMatrix[0], fused_viewMatrix[1], fused_viewMatrix[2], fused_viewMatrix[3],
        fused_viewMatrix[4], fused_viewMatrix[5], fused_viewMatrix[6], fused_viewMatrix[7],
        fused_viewMatrix[8], fused_viewMatrix[9], fused_viewMatrix[10], fused_viewMatrix[11],
        fused_viewMatrix[12], fused_viewMatrix[13], fused_viewMatrix[14], fused_viewMatrix[15],
        fused_projMatrix[0], fused_projMatrix[1], fused_projMatrix[2], fused_projMatrix[3],
        fused_projMatrix[4], fused_projMatrix[5], fused_projMatrix[6], fused_projMatrix[7],
        fused_projMatrix[8], fused_projMatrix[9], fused_projMatrix[10], fused_projMatrix[11],
        fused_projMatrix[12], fused_projMatrix[13], fused_projMatrix[14], fused_projMatrix[15]
    );
    float visibleMask = (projectionNdc.pView.z >= 0.2f) ? 1.0f : 0.0f;

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
    Cov3dValues cov3d = buildCov3dFromScaleRotation(
        fused_scales[scaleBase + 0],
        fused_scales[scaleBase + 1],
        fused_scales[scaleBase + 2],
        fused_rotations[rotBase + 0],
        fused_rotations[rotBase + 1],
        fused_rotations[rotBase + 2],
        fused_rotations[rotBase + 3]
    );

    float a00 = fused_viewMatrix[0 * 4 + 0];
    float a01 = fused_viewMatrix[0 * 4 + 1];
    float a02 = fused_viewMatrix[0 * 4 + 2];
    float a10 = fused_viewMatrix[1 * 4 + 0];
    float a11 = fused_viewMatrix[1 * 4 + 1];
    float a12 = fused_viewMatrix[1 * 4 + 2];
    float a20 = fused_viewMatrix[2 * 4 + 0];
    float a21 = fused_viewMatrix[2 * 4 + 1];
    float a22 = fused_viewMatrix[2 * 4 + 2];

    ProjectionScreenGradOutputs fusedOut = evaluateProjectionScreenGradOutputs(
        float2(projectionNdc.meanNdc.x, projectionNdc.meanNdc.y),
        float3(m0, m1, m2),
        float3(fused_cameraCenter[0], fused_cameraCenter[1], fused_cameraCenter[2]),
        shCoeffs,
        cov3d,
        degree,
        a00, a01, a02,
        a10, a11, a12,
        a20, a21, a22,
        fused_viewMatrix[3 * 4 + 0],
        fused_viewMatrix[3 * 4 + 1],
        fused_viewMatrix[3 * 4 + 2],
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
    fused_outDepths[p] = projectionNdc.pView.z;

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
StructuredBuffer<float> fused_bwd_fovX;                 // [1]
StructuredBuffer<float> fused_bwd_fovY;                 // [1]
StructuredBuffer<float> fused_bwd_focalX;               // [1]
StructuredBuffer<float> fused_bwd_focalY;               // [1]
StructuredBuffer<float> fused_bwd_imageWidth;           // [1]
StructuredBuffer<float> fused_bwd_imageHeight;          // [1]
StructuredBuffer<float> fused_bwd_cotMeans2d;           // [N*2]
StructuredBuffer<float> fused_bwd_cotCov2d;             // [N*4]
StructuredBuffer<float> fused_bwd_cotColor;             // [N*3]
StructuredBuffer<float> fused_bwd_cotConic;             // [N*4]
StructuredBuffer<uint>  fused_bwd_counts;               // [pointCount, activeShDegree, shStride]

RWStructuredBuffer<float> fused_bwd_gradMeanNdc;        // [N*4]
RWStructuredBuffer<float> fused_bwd_gradMeans3dScreen;  // [N*3]
RWStructuredBuffer<float> fused_bwd_gradShs;            // [N*K*3]
RWStructuredBuffer<float> fused_bwd_gradCov3d;          // [N*9]
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
    float3 means3d = float3(
        fused_bwd_means3d[meansBase + 0],
        fused_bwd_means3d[meansBase + 1],
        fused_bwd_means3d[meansBase + 2]
    );
    float3 cameraCenter = float3(
        fused_bwd_cameraCenter[0],
        fused_bwd_cameraCenter[1],
        fused_bwd_cameraCenter[2]
    );
    uint scaleBase = p * 3;
    uint rotBase = p * 4;
    Cov3dValues cov3d = buildCov3dFromScaleRotation(
        fused_bwd_scales[scaleBase + 0],
        fused_bwd_scales[scaleBase + 1],
        fused_bwd_scales[scaleBase + 2],
        fused_bwd_rotations[rotBase + 0],
        fused_bwd_rotations[rotBase + 1],
        fused_bwd_rotations[rotBase + 2],
        fused_bwd_rotations[rotBase + 3]
    );

    float a00 = fused_bwd_viewMatrix[0 * 4 + 0];
    float a01 = fused_bwd_viewMatrix[0 * 4 + 1];
    float a02 = fused_bwd_viewMatrix[0 * 4 + 2];
    float a10 = fused_bwd_viewMatrix[1 * 4 + 0];
    float a11 = fused_bwd_viewMatrix[1 * 4 + 1];
    float a12 = fused_bwd_viewMatrix[1 * 4 + 2];
    float a20 = fused_bwd_viewMatrix[2 * 4 + 0];
    float a21 = fused_bwd_viewMatrix[2 * 4 + 1];
    float a22 = fused_bwd_viewMatrix[2 * 4 + 2];

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

    float2 meanNdcDiff = float2(0.0);
    float3 means3dDiff = float3(0.0);
    float3 cameraCenterDiff = float3(0.0);
    Cov3dValues cov3dDiff = zeroCov3d();

    DifferentialPair<float2> dpMeanNdc = diffPair(float2(0.0), meanNdcDiff);
    DifferentialPair<float3> dpMeans3d = diffPair(means3d, means3dDiff);
    DifferentialPair<float3> dpCameraCenter = diffPair(cameraCenter, cameraCenterDiff);
    DifferentialPair<float3[25]> dpShCoeffs = diffPair(shCoeffs, shCoeffDiff);
    DifferentialPair<Cov3dValues> dpCov3d = diffPair(cov3d, cov3dDiff);

    uint cotMeansBase = p * 2;
    uint cotColorBase = p * 3;
    uint cotCovBase = p * 4;
    ProjectionScreenGradOutputs cotOutputs;
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

    bwd_diff(evaluateProjectionScreenGradOutputs)(
        dpMeanNdc,
        dpMeans3d,
        dpCameraCenter,
        dpShCoeffs,
        dpCov3d,
        degree,
        a00, a01, a02, a10, a11, a12, a20, a21, a22,
        fused_bwd_viewMatrix[3 * 4 + 0],
        fused_bwd_viewMatrix[3 * 4 + 1],
        fused_bwd_viewMatrix[3 * 4 + 2],
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

    uint gradCov3dBase = p * 9;
    fused_bwd_gradCov3d[gradCov3dBase + 0] = dpCov3d.d.c00;
    fused_bwd_gradCov3d[gradCov3dBase + 1] = dpCov3d.d.c01;
    fused_bwd_gradCov3d[gradCov3dBase + 2] = dpCov3d.d.c02;
    fused_bwd_gradCov3d[gradCov3dBase + 3] = dpCov3d.d.c10;
    fused_bwd_gradCov3d[gradCov3dBase + 4] = dpCov3d.d.c11;
    fused_bwd_gradCov3d[gradCov3dBase + 5] = dpCov3d.d.c12;
    fused_bwd_gradCov3d[gradCov3dBase + 6] = dpCov3d.d.c20;
    fused_bwd_gradCov3d[gradCov3dBase + 7] = dpCov3d.d.c21;
    fused_bwd_gradCov3d[gradCov3dBase + 8] = dpCov3d.d.c22;

    uint gradMeanNdcBase = p * 4;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 0] = dpMeanNdc.d.x;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 1] = dpMeanNdc.d.y;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 2] = 0.0;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 3] = 0.0;

    fused_bwd_gradMeans3dScreen[meansBase + 0] = dpMeans3d.d.x;
    fused_bwd_gradMeans3dScreen[meansBase + 1] = dpMeans3d.d.y;
    fused_bwd_gradMeans3dScreen[meansBase + 2] = dpMeans3d.d.z;

    fused_bwd_gradCameraCenterPoint[meansBase + 0] = dpCameraCenter.d.x;
    fused_bwd_gradCameraCenterPoint[meansBase + 1] = dpCameraCenter.d.y;
    fused_bwd_gradCameraCenterPoint[meansBase + 2] = dpCameraCenter.d.z;
}
