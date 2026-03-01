// gaussian_projection_kernels.slang
//
// Projection NDC (forward/backward) and radius-with-mask (forward-only) kernels.
//
// projection_ndc_forward:
//   points [N,3] x viewMatrix [4,4] x projMatrix [4,4]
//   -> mean_ndc [N,4], p_view [N,4], visibleMask [N]
//
// projection_ndc_backward:
//   VJP: cotangent of mean_ndc + cotangent of p_view -> grad_points [N,3]
//   (viewMatrix/projMatrix have no gradient; visibleMask is stopGradient)
//
// get_radius_mask_forward:
//   cov2d [N,2,2] (flat [N,4]) + visibleMask [N] -> radii [N]
//   Computes max eigenvalue of cov2d, radius = 3*ceil(sqrt(lambda_max)),
//   then masks by visibility. No backward needed (always under stopGradient).

import gaussian_projection_screen_shared;

#define FUSED_SH_COEFF(_shBase, _k) \
    float3( \
        fused_shs[(_shBase) + uint((_k) * 3 + 0)], \
        fused_shs[(_shBase) + uint((_k) * 3 + 1)], \
        fused_shs[(_shBase) + uint((_k) * 3 + 2)] \
    )

#define FUSED_BWD_SH_INDEX(_shBase, _k, _ch) \
    ((_shBase) + uint((_k) * 3 + (_ch)))

// ---------------------------------------------------------------------------
// projection_ndc_forward
// ---------------------------------------------------------------------------
StructuredBuffer<float> proj_points;         // [N*3]
StructuredBuffer<float> proj_viewMatrix;     // [16] (4x4 column-major)
StructuredBuffer<float> proj_projMatrix;     // [16] (4x4 column-major)
StructuredBuffer<uint>  proj_counts;         // [1]: pointCount

// Forward outputs
RWStructuredBuffer<float> proj_outMeanNdc;   // [N*4]
RWStructuredBuffer<float> proj_outPView;     // [N*4]
RWStructuredBuffer<float> proj_outVisibleMask; // [N]

// Backward inputs (cotangents)
StructuredBuffer<float> proj_cotMeanNdc;     // [N*4]
StructuredBuffer<float> proj_cotPView;       // [N*4]

// Backward outputs (gradients)
RWStructuredBuffer<float> proj_gradPoints;   // [N*3]

[shader("compute")]
[numthreads(128, 1, 1)]
void projection_ndc_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    if (p >= proj_counts[0]) return;

    uint pBase = p * 3;
    float px = proj_points[pBase];
    float py = proj_points[pBase + 1];
    float pz = proj_points[pBase + 2];

    // Homogeneous point * viewMatrix  (column-major: row i of viewMatrix = indices [i, i+4, i+8, i+12])
    float pv0 = px * proj_viewMatrix[0] + py * proj_viewMatrix[4] + pz * proj_viewMatrix[8]  + proj_viewMatrix[12];
    float pv1 = px * proj_viewMatrix[1] + py * proj_viewMatrix[5] + pz * proj_viewMatrix[9]  + proj_viewMatrix[13];
    float pv2 = px * proj_viewMatrix[2] + py * proj_viewMatrix[6] + pz * proj_viewMatrix[10] + proj_viewMatrix[14];
    float pv3 = px * proj_viewMatrix[3] + py * proj_viewMatrix[7] + pz * proj_viewMatrix[11] + proj_viewMatrix[15];

    // p_view * projMatrix
    float pc0 = pv0 * proj_projMatrix[0] + pv1 * proj_projMatrix[4] + pv2 * proj_projMatrix[8]  + pv3 * proj_projMatrix[12];
    float pc1 = pv0 * proj_projMatrix[1] + pv1 * proj_projMatrix[5] + pv2 * proj_projMatrix[9]  + pv3 * proj_projMatrix[13];
    float pc2 = pv0 * proj_projMatrix[2] + pv1 * proj_projMatrix[6] + pv2 * proj_projMatrix[10] + pv3 * proj_projMatrix[14];
    float pc3 = pv0 * proj_projMatrix[3] + pv1 * proj_projMatrix[7] + pv2 * proj_projMatrix[11] + pv3 * proj_projMatrix[15];

    float w_inv = 1.0f / (pc3 + 0.000001f);

    uint outBase = p * 4;
    proj_outMeanNdc[outBase]     = pc0 * w_inv;
    proj_outMeanNdc[outBase + 1] = pc1 * w_inv;
    proj_outMeanNdc[outBase + 2] = pc2 * w_inv;
    proj_outMeanNdc[outBase + 3] = pc3 * w_inv;

    proj_outPView[outBase]     = pv0;
    proj_outPView[outBase + 1] = pv1;
    proj_outPView[outBase + 2] = pv2;
    proj_outPView[outBase + 3] = pv3;

    proj_outVisibleMask[p] = (pv2 >= 0.2f) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// projection_ndc_backward
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(128, 1, 1)]
void projection_ndc_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    if (p >= proj_counts[0]) return;

    uint pBase = p * 3;
    float px = proj_points[pBase];
    float py = proj_points[pBase + 1];
    float pz = proj_points[pBase + 2];

    // Recompute forward values
    float pv0 = px * proj_viewMatrix[0] + py * proj_viewMatrix[4] + pz * proj_viewMatrix[8]  + proj_viewMatrix[12];
    float pv1 = px * proj_viewMatrix[1] + py * proj_viewMatrix[5] + pz * proj_viewMatrix[9]  + proj_viewMatrix[13];
    float pv2 = px * proj_viewMatrix[2] + py * proj_viewMatrix[6] + pz * proj_viewMatrix[10] + proj_viewMatrix[14];
    float pv3 = px * proj_viewMatrix[3] + py * proj_viewMatrix[7] + pz * proj_viewMatrix[11] + proj_viewMatrix[15];

    float pc0 = pv0 * proj_projMatrix[0] + pv1 * proj_projMatrix[4] + pv2 * proj_projMatrix[8]  + pv3 * proj_projMatrix[12];
    float pc1 = pv0 * proj_projMatrix[1] + pv1 * proj_projMatrix[5] + pv2 * proj_projMatrix[9]  + pv3 * proj_projMatrix[13];
    float pc2 = pv0 * proj_projMatrix[2] + pv1 * proj_projMatrix[6] + pv2 * proj_projMatrix[10] + pv3 * proj_projMatrix[14];
    float pc3 = pv0 * proj_projMatrix[3] + pv1 * proj_projMatrix[7] + pv2 * proj_projMatrix[11] + pv3 * proj_projMatrix[15];

    float w_inv = 1.0f / (pc3 + 0.000001f);

    // Read cotangents
    uint gBase = p * 4;
    float dm0 = proj_cotMeanNdc[gBase];
    float dm1 = proj_cotMeanNdc[gBase + 1];
    float dm2 = proj_cotMeanNdc[gBase + 2];
    float dm3 = proj_cotMeanNdc[gBase + 3];

    float dpv_ext0 = proj_cotPView[gBase];
    float dpv_ext1 = proj_cotPView[gBase + 1];
    float dpv_ext2 = proj_cotPView[gBase + 2];
    float dpv_ext3 = proj_cotPView[gBase + 3];

    // d(p_proj) / d(p_clip): p_proj = p_clip * w_inv
    // d(p_proj_i)/d(p_clip_i) = w_inv
    // d(p_proj_i)/d(p_clip_3) = -p_clip_i * w_inv^2
    float dpc0 = dm0 * w_inv;
    float dpc1 = dm1 * w_inv;
    float dpc2 = dm2 * w_inv;
    float dpc3 = dm3 * w_inv;

    float dw = dm0 * pc0 + dm1 * pc1 + dm2 * pc2 + dm3 * pc3;
    dpc3 += dw * (-w_inv * w_inv);

    // d(p_clip) / d(p_view): p_clip = p_view * projMatrix
    float dpv0 = dpc0 * proj_projMatrix[0] + dpc1 * proj_projMatrix[1] + dpc2 * proj_projMatrix[2]  + dpc3 * proj_projMatrix[3];
    float dpv1 = dpc0 * proj_projMatrix[4] + dpc1 * proj_projMatrix[5] + dpc2 * proj_projMatrix[6]  + dpc3 * proj_projMatrix[7];
    float dpv2 = dpc0 * proj_projMatrix[8] + dpc1 * proj_projMatrix[9] + dpc2 * proj_projMatrix[10] + dpc3 * proj_projMatrix[11];
    float dpv3 = dpc0 * proj_projMatrix[12] + dpc1 * proj_projMatrix[13] + dpc2 * proj_projMatrix[14] + dpc3 * proj_projMatrix[15];

    // Add external p_view cotangent (from depths = p_view[..., 2])
    dpv0 += dpv_ext0;
    dpv1 += dpv_ext1;
    dpv2 += dpv_ext2;
    dpv3 += dpv_ext3;

    // d(p_view) / d(points): p_view = [px,py,pz,1] * viewMatrix
    float dph0 = dpv0 * proj_viewMatrix[0] + dpv1 * proj_viewMatrix[1] + dpv2 * proj_viewMatrix[2]  + dpv3 * proj_viewMatrix[3];
    float dph1 = dpv0 * proj_viewMatrix[4] + dpv1 * proj_viewMatrix[5] + dpv2 * proj_viewMatrix[6]  + dpv3 * proj_viewMatrix[7];
    float dph2 = dpv0 * proj_viewMatrix[8] + dpv1 * proj_viewMatrix[9] + dpv2 * proj_viewMatrix[10] + dpv3 * proj_viewMatrix[11];

    proj_gradPoints[pBase]     = dph0;
    proj_gradPoints[pBase + 1] = dph1;
    proj_gradPoints[pBase + 2] = dph2;
}

// ---------------------------------------------------------------------------
// get_radius_mask_forward  (no backward — always stopGradient)
// ---------------------------------------------------------------------------
StructuredBuffer<float> rad_cov2d;           // [N*4] (flattened 2x2)
StructuredBuffer<float> rad_visibleMask;     // [N]
StructuredBuffer<uint>  rad_counts;          // [1]: pointCount

RWStructuredBuffer<float> rad_outRadii;      // [N]

[shader("compute")]
[numthreads(128, 1, 1)]
void get_radius_mask_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    if (p >= rad_counts[0]) return;

    uint covBase = p * 4;
    float c00 = rad_cov2d[covBase];
    float c01 = rad_cov2d[covBase + 1];
    float c10 = rad_cov2d[covBase + 2];
    float c11 = rad_cov2d[covBase + 3];

    // Max eigenvalue of 2x2 symmetric matrix
    float det_val = c00 * c11 - c01 * c10;
    float mid = 0.5f * (c00 + c11);
    float delta = max(mid * mid - det_val, 1e-5f);
    float sqrtDelta = sqrt(delta);
    float lambda_max = mid + sqrtDelta;

    float radius = 3.0f * ceil(sqrt(lambda_max));

    float mask = rad_visibleMask[p];
    rad_outRadii[p] = radius * mask;
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

    float pv0 = m0 * fused_viewMatrix[0] + m1 * fused_viewMatrix[4] + m2 * fused_viewMatrix[8]  + fused_viewMatrix[12];
    float pv1 = m0 * fused_viewMatrix[1] + m1 * fused_viewMatrix[5] + m2 * fused_viewMatrix[9]  + fused_viewMatrix[13];
    float pv2 = m0 * fused_viewMatrix[2] + m1 * fused_viewMatrix[6] + m2 * fused_viewMatrix[10] + fused_viewMatrix[14];
    float pv3 = m0 * fused_viewMatrix[3] + m1 * fused_viewMatrix[7] + m2 * fused_viewMatrix[11] + fused_viewMatrix[15];

    float pc0 = pv0 * fused_projMatrix[0] + pv1 * fused_projMatrix[4] + pv2 * fused_projMatrix[8]  + pv3 * fused_projMatrix[12];
    float pc1 = pv0 * fused_projMatrix[1] + pv1 * fused_projMatrix[5] + pv2 * fused_projMatrix[9]  + pv3 * fused_projMatrix[13];
    float pc2 = pv0 * fused_projMatrix[2] + pv1 * fused_projMatrix[6] + pv2 * fused_projMatrix[10] + pv3 * fused_projMatrix[14];
    float pc3 = pv0 * fused_projMatrix[3] + pv1 * fused_projMatrix[7] + pv2 * fused_projMatrix[11] + pv3 * fused_projMatrix[15];

    ProjectionValues projection = projectFromHomogeneous(
        pv0, pv1, pv2, pv3,
        pc0, pc1, pc2, pc3
    );

    float2 means2d = ndcToScreen(
        projection.ndc0,
        projection.ndc1,
        fused_imageWidth[0],
        fused_imageHeight[0]
    );
    uint means2dBase = p * 2;
    fused_outMeans2d[means2dBase + 0] = means2d.x;
    fused_outMeans2d[means2dBase + 1] = means2d.y;
    fused_outDepths[p] = projection.pv2;

    int degree = int(fused_counts[1]);
    int coeffStride = int(fused_counts[2]);

    float x = m0 - fused_cameraCenter[0];
    float y = m1 - fused_cameraCenter[1];
    float z = m2 - fused_cameraCenter[2];
    uint shPointBase = p * uint(coeffStride * 3);

    float3 colorAccum = 0.28209479177387814 * FUSED_SH_COEFF(shPointBase, 0);
    if (degree > 0)
    {
        colorAccum += (-0.4886025119029199 * y) * FUSED_SH_COEFF(shPointBase, 1);
        colorAccum += (0.4886025119029199 * z) * FUSED_SH_COEFF(shPointBase, 2);
        colorAccum += (-0.4886025119029199 * x) * FUSED_SH_COEFF(shPointBase, 3);

        if (degree > 1)
        {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;

            colorAccum += (1.0925484305920792 * xy) * FUSED_SH_COEFF(shPointBase, 4);
            colorAccum += (-1.0925484305920792 * yz) * FUSED_SH_COEFF(shPointBase, 5);
            colorAccum += (0.31539156525252005 * (2.0 * zz - xx - yy)) * FUSED_SH_COEFF(shPointBase, 6);
            colorAccum += (-1.0925484305920792 * xz) * FUSED_SH_COEFF(shPointBase, 7);
            colorAccum += (0.5462742152960396 * (xx - yy)) * FUSED_SH_COEFF(shPointBase, 8);

            if (degree > 2)
            {
                colorAccum += (-0.5900435899266435 * y * (3.0 * xx - yy))
                    * FUSED_SH_COEFF(shPointBase, 9);
                colorAccum += (2.890611442640554 * xy * z)
                    * FUSED_SH_COEFF(shPointBase, 10);
                colorAccum += (-0.4570457994644658 * y * (4.0 * zz - xx - yy))
                    * FUSED_SH_COEFF(shPointBase, 11);
                colorAccum += (0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy))
                    * FUSED_SH_COEFF(shPointBase, 12);
                colorAccum += (-0.4570457994644658 * x * (4.0 * zz - xx - yy))
                    * FUSED_SH_COEFF(shPointBase, 13);
                colorAccum += (1.445305721320277 * z * (xx - yy))
                    * FUSED_SH_COEFF(shPointBase, 14);
                colorAccum += (-0.5900435899266435 * x * (xx - 3.0 * yy))
                    * FUSED_SH_COEFF(shPointBase, 15);

                if (degree > 3)
                {
                    colorAccum += (2.5033429417967046 * xy * (xx - yy))
                        * FUSED_SH_COEFF(shPointBase, 16);
                    colorAccum += (-1.7701307697799304 * yz * (3.0 * xx - yy))
                        * FUSED_SH_COEFF(shPointBase, 17);
                    colorAccum += (0.9461746957575601 * xy * (7.0 * zz - 1.0))
                        * FUSED_SH_COEFF(shPointBase, 18);
                    colorAccum += (-0.6690465435572892 * yz * (7.0 * zz - 3.0))
                        * FUSED_SH_COEFF(shPointBase, 19);
                    colorAccum += (0.10578554691520431 * (zz * (35.0 * zz - 30.0) + 3.0))
                        * FUSED_SH_COEFF(shPointBase, 20);
                    colorAccum += (-0.6690465435572892 * xz * (7.0 * zz - 3.0))
                        * FUSED_SH_COEFF(shPointBase, 21);
                    colorAccum += (0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0))
                        * FUSED_SH_COEFF(shPointBase, 22);
                    colorAccum += (-1.7701307697799304 * xz * (xx - 3.0 * yy))
                        * FUSED_SH_COEFF(shPointBase, 23);
                    colorAccum += (0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)))
                        * FUSED_SH_COEFF(shPointBase, 24);
                }
            }
        }
    }
    colorAccum += 0.5;
    colorAccum = max(colorAccum, 0.0);

    uint colorBase = p * 3;
    fused_outColor[colorBase + 0] = colorAccum.x;
    fused_outColor[colorBase + 1] = colorAccum.y;
    fused_outColor[colorBase + 2] = colorAccum.z;

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

    Cov2dValues cov2d = buildCov2dFromCov3d(
        m0, m1, m2,
        cov3d,
        a00, a01, a02, a10, a11, a12, a20, a21, a22,
        fused_viewMatrix[3 * 4 + 0],
        fused_viewMatrix[3 * 4 + 1],
        fused_viewMatrix[3 * 4 + 2],
        fused_fovX[0],
        fused_fovY[0],
        fused_focalX[0],
        fused_focalY[0]
    );

    uint covBase = p * 4;
    fused_outCov2d[covBase + 0] = cov2d.c00;
    fused_outCov2d[covBase + 1] = cov2d.c01;
    fused_outCov2d[covBase + 2] = cov2d.c10;
    fused_outCov2d[covBase + 3] = cov2d.c11;

    float4 conic = inverseCov2d(cov2d);
    fused_outConic[covBase + 0] = conic.x;
    fused_outConic[covBase + 1] = conic.y;
    fused_outConic[covBase + 2] = conic.z;
    fused_outConic[covBase + 3] = conic.w;

    float radius = computeRadiusFromCov2d(cov2d);
    fused_outRadii[p] = radius * projection.visibleMask;
}

// ---------------------------------------------------------------------------
// gaussian_projection_screen_fused_backward
//   Fused backward for:
//   - gaussian_screen_color_backward
//   - gaussian_screen_cov2d_forward + gaussian_screen_inverse2d_backward
//   - gaussian_screen_cov2d_backward
//   - gaussian_screen_cov3d_forward (recompute only)
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
    int coeffStride = int(fused_bwd_counts[2]);

    uint meansBase = p * 3;
    float m0 = fused_bwd_means3d[meansBase + 0];
    float m1 = fused_bwd_means3d[meansBase + 1];
    float m2 = fused_bwd_means3d[meansBase + 2];

    // ---- color backward ----
    float x = m0 - fused_bwd_cameraCenter[0];
    float y = m1 - fused_bwd_cameraCenter[1];
    float z = m2 - fused_bwd_cameraCenter[2];
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    float basis[25];
    float dbx[25];
    float dby[25];
    float dbz[25];
    for (int i = 0; i < 25; ++i)
    {
        basis[i] = 0.0;
        dbx[i] = 0.0;
        dby[i] = 0.0;
        dbz[i] = 0.0;
    }

    basis[0] = 0.28209479177387814;
    if (degree > 0)
    {
        basis[1] = -0.4886025119029199 * y;
        basis[2] = 0.4886025119029199 * z;
        basis[3] = -0.4886025119029199 * x;
        dby[1] = -0.4886025119029199;
        dbz[2] = 0.4886025119029199;
        dbx[3] = -0.4886025119029199;
    }
    if (degree > 1)
    {
        basis[4] = 1.0925484305920792 * xy;
        basis[5] = -1.0925484305920792 * yz;
        basis[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
        basis[7] = -1.0925484305920792 * xz;
        basis[8] = 0.5462742152960396 * (xx - yy);

        dbx[4] = 1.0925484305920792 * y;
        dby[4] = 1.0925484305920792 * x;
        dby[5] = -1.0925484305920792 * z;
        dbz[5] = -1.0925484305920792 * y;
        dbx[6] = 0.31539156525252005 * (-2.0 * x);
        dby[6] = 0.31539156525252005 * (-2.0 * y);
        dbz[6] = 0.31539156525252005 * (4.0 * z);
        dbx[7] = -1.0925484305920792 * z;
        dbz[7] = -1.0925484305920792 * x;
        dbx[8] = 0.5462742152960396 * (2.0 * x);
        dby[8] = 0.5462742152960396 * (-2.0 * y);
    }
    if (degree > 2)
    {
        basis[9] = -0.5900435899266435 * y * (3.0 * xx - yy);
        basis[10] = 2.890611442640554 * xy * z;
        basis[11] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
        basis[12] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
        basis[13] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
        basis[14] = 1.445305721320277 * z * (xx - yy);
        basis[15] = -0.5900435899266435 * x * (xx - 3.0 * yy);

        dbx[9] = -0.5900435899266435 * (6.0 * x * y);
        dby[9] = -0.5900435899266435 * (3.0 * xx - 3.0 * yy);
        dbx[10] = 2.890611442640554 * yz;
        dby[10] = 2.890611442640554 * xz;
        dbz[10] = 2.890611442640554 * xy;
        dbx[11] = -0.4570457994644658 * (-2.0 * x * y);
        dby[11] = -0.4570457994644658 * (4.0 * zz - xx - 3.0 * yy);
        dbz[11] = -0.4570457994644658 * (8.0 * y * z);
        dbx[12] = 0.3731763325901154 * (-6.0 * x * z);
        dby[12] = 0.3731763325901154 * (-6.0 * y * z);
        dbz[12] = 0.3731763325901154 * (6.0 * zz - 3.0 * xx - 3.0 * yy);
        dbx[13] = -0.4570457994644658 * (4.0 * zz - 3.0 * xx - yy);
        dby[13] = -0.4570457994644658 * (-2.0 * x * y);
        dbz[13] = -0.4570457994644658 * (8.0 * x * z);
        dbx[14] = 1.445305721320277 * (2.0 * x * z);
        dby[14] = 1.445305721320277 * (-2.0 * y * z);
        dbz[14] = 1.445305721320277 * (xx - yy);
        dbx[15] = -0.5900435899266435 * (3.0 * xx - 3.0 * yy);
        dby[15] = -0.5900435899266435 * (-6.0 * x * y);
    }
    if (degree > 3)
    {
        basis[16] = 2.5033429417967046 * xy * (xx - yy);
        basis[17] = -1.7701307697799304 * yz * (3.0 * xx - yy);
        basis[18] = 0.9461746957575601 * xy * (7.0 * zz - 1.0);
        basis[19] = -0.6690465435572892 * yz * (7.0 * zz - 3.0);
        basis[20] = 0.10578554691520431 * (zz * (35.0 * zz - 30.0) + 3.0);
        basis[21] = -0.6690465435572892 * xz * (7.0 * zz - 3.0);
        basis[22] = 0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0);
        basis[23] = -1.7701307697799304 * xz * (xx - 3.0 * yy);
        basis[24] = 0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy));

        dbx[16] = 2.5033429417967046 * y * (3.0 * xx - yy);
        dby[16] = 2.5033429417967046 * x * (xx - 3.0 * yy);
        dbx[17] = -1.7701307697799304 * (6.0 * x * y * z);
        dby[17] = -1.7701307697799304 * (3.0 * z * (xx - yy));
        dbz[17] = -1.7701307697799304 * y * (3.0 * xx - yy);
        dbx[18] = 0.9461746957575601 * y * (7.0 * zz - 1.0);
        dby[18] = 0.9461746957575601 * x * (7.0 * zz - 1.0);
        dbz[18] = 0.9461746957575601 * (14.0 * x * y * z);
        dby[19] = -0.6690465435572892 * z * (7.0 * zz - 3.0);
        dbz[19] = -0.6690465435572892 * y * (21.0 * zz - 3.0);
        dbz[20] = 0.10578554691520431 * (140.0 * z * z * z - 60.0 * z);
        dbx[21] = -0.6690465435572892 * z * (7.0 * zz - 3.0);
        dbz[21] = -0.6690465435572892 * x * (21.0 * zz - 3.0);
        dbx[22] = 0.47308734787878004 * (2.0 * x) * (7.0 * zz - 1.0);
        dby[22] = 0.47308734787878004 * (-2.0 * y) * (7.0 * zz - 1.0);
        dbz[22] = 0.47308734787878004 * (14.0 * z * (xx - yy));
        dbx[23] = -1.7701307697799304 * (3.0 * z * (xx - yy));
        dby[23] = -1.7701307697799304 * (-6.0 * x * y * z);
        dbz[23] = -1.7701307697799304 * x * (xx - 3.0 * yy);
        dbx[24] = 0.6258357354491761 * (4.0 * x * (xx - 3.0 * yy));
        dby[24] = 0.6258357354491761 * (4.0 * y * (yy - 3.0 * xx));
    }

    uint cotColorBase = p * 3;
    uint shPointBase = p * uint(coeffStride * 3);
    float colorGradX = 0.0;
    float colorGradY = 0.0;
    float colorGradZ = 0.0;
    for (int ch = 0; ch < 3; ++ch)
    {
        float value = 0.0;
        for (int k = 0; k < coeffCount; ++k)
        {
            uint shIndex = FUSED_BWD_SH_INDEX(shPointBase, k, ch);
            value += basis[k] * fused_bwd_shs[shIndex];
        }

        float g = fused_bwd_cotColor[cotColorBase + uint(ch)];
        if (value + 0.5 <= 0.0)
        {
            g = 0.0;
        }

        for (int k = 0; k < coeffCount; ++k)
        {
            uint shIndex = FUSED_BWD_SH_INDEX(shPointBase, k, ch);
            float shValue = fused_bwd_shs[shIndex];
            fused_bwd_gradShs[shIndex] = g * basis[k];
            colorGradX += g * shValue * dbx[k];
            colorGradY += g * shValue * dby[k];
            colorGradZ += g * shValue * dbz[k];
        }
    }

    // ---- cov3d recompute ----
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

    Cov2dValues cov2d = buildCov2dFromCov3d(
        m0, m1, m2,
        cov3d,
        a00, a01, a02, a10, a11, a12, a20, a21, a22,
        fused_bwd_viewMatrix[3 * 4 + 0],
        fused_bwd_viewMatrix[3 * 4 + 1],
        fused_bwd_viewMatrix[3 * 4 + 2],
        fused_bwd_fovX[0],
        fused_bwd_fovY[0],
        fused_bwd_focalX[0],
        fused_bwd_focalY[0]
    );

    // ---- inverse2d backward ----
    uint cotCovBase = p * 4;
    float cotCov00 = fused_bwd_cotCov2d[cotCovBase + 0];
    float cotCov01 = fused_bwd_cotCov2d[cotCovBase + 1];
    float cotCov10 = fused_bwd_cotCov2d[cotCovBase + 2];
    float cotCov11 = fused_bwd_cotCov2d[cotCovBase + 3];

    float covA = cov2d.c00;
    float covB = cov2d.c01;
    float covC = cov2d.c10;
    float covD = cov2d.c11;
    float det = covA * covD - covB * covC;
    float inv00 = covD / det;
    float inv01 = -covB / det;
    float inv10 = -covC / det;
    float inv11 = covA / det;

    float cotConic00 = fused_bwd_cotConic[cotCovBase + 0];
    float cotConic01 = fused_bwd_cotConic[cotCovBase + 1];
    float cotConic10 = fused_bwd_cotConic[cotCovBase + 2];
    float cotConic11 = fused_bwd_cotConic[cotCovBase + 3];

    float tt00 = inv00 * cotConic00 + inv10 * cotConic10;
    float tt01 = inv00 * cotConic01 + inv10 * cotConic11;
    float tt10 = inv01 * cotConic00 + inv11 * cotConic10;
    float tt11 = inv01 * cotConic01 + inv11 * cotConic11;

    float gradInvCov00 = -(tt00 * inv00 + tt01 * inv01);
    float gradInvCov01 = -(tt00 * inv10 + tt01 * inv11);
    float gradInvCov10 = -(tt10 * inv00 + tt11 * inv01);
    float gradInvCov11 = -(tt10 * inv10 + tt11 * inv11);

    float totalCotCov00 = cotCov00 + gradInvCov00;
    float totalCotCov01 = cotCov01 + gradInvCov01;
    float totalCotCov10 = cotCov10 + gradInvCov10;
    float totalCotCov11 = cotCov11 + gradInvCov11;

    // ---- cov2d backward ----
    float t0 = m0 * a00 + m1 * a10 + m2 * a20 + fused_bwd_viewMatrix[3 * 4 + 0];
    float t1 = m0 * a01 + m1 * a11 + m2 * a21 + fused_bwd_viewMatrix[3 * 4 + 1];
    float t2 = m0 * a02 + m1 * a12 + m2 * a22 + fused_bwd_viewMatrix[3 * 4 + 2];

    float tanFovX = tan(fused_bwd_fovX[0] * 0.5);
    float tanFovY = tan(fused_bwd_fovY[0] * 0.5);
    float minX = -tanFovX * 1.3;
    float maxX = tanFovX * 1.3;
    float minY = -tanFovY * 1.3;
    float maxY = tanFovY * 1.3;
    float clipX = clamp(t2, minX, maxX);
    float clipY = clamp(t2, minY, maxY);
    float tx = t0 / clipX * t2;
    float ty = t1 / clipY * t2;
    float tz = t2;

    float j00 = fused_bwd_focalX[0] / tz;
    float j02 = -tx * fused_bwd_focalX[0] / (tz * tz);
    float j11 = fused_bwd_focalY[0] / tz;
    float j12 = -ty * fused_bwd_focalY[0] / (tz * tz);

    float w00 = a00;
    float w01 = a10;
    float w02 = a20;
    float w10 = a01;
    float w11 = a11;
    float w12 = a21;
    float w20 = a02;
    float w21 = a12;
    float w22 = a22;

    float b[9];
    b[0] = j00 * w00 + j02 * w20;
    b[1] = j00 * w01 + j02 * w21;
    b[2] = j00 * w02 + j02 * w22;
    b[3] = j11 * w10 + j12 * w20;
    b[4] = j11 * w11 + j12 * w21;
    b[5] = j11 * w12 + j12 * w22;
    b[6] = 0.0;
    b[7] = 0.0;
    b[8] = 0.0;

    float c[9];
    c[0] = cov3d.c00;
    c[1] = cov3d.c01;
    c[2] = cov3d.c02;
    c[3] = cov3d.c10;
    c[4] = cov3d.c11;
    c[5] = cov3d.c12;
    c[6] = cov3d.c20;
    c[7] = cov3d.c21;
    c[8] = cov3d.c22;

    float cotM[9];
    cotM[0] = totalCotCov00;
    cotM[1] = totalCotCov01;
    cotM[2] = 0.0;
    cotM[3] = totalCotCov10;
    cotM[4] = totalCotCov11;
    cotM[5] = 0.0;
    cotM[6] = 0.0;
    cotM[7] = 0.0;
    cotM[8] = 0.0;

    float bt[9];
    bt[0] = b[0];
    bt[1] = b[3];
    bt[2] = b[6];
    bt[3] = b[1];
    bt[4] = b[4];
    bt[5] = b[7];
    bt[6] = b[2];
    bt[7] = b[5];
    bt[8] = b[8];

    float tmpA[9];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float v = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                v += bt[i * 3 + k] * cotM[k * 3 + j];
            }
            tmpA[i * 3 + j] = v;
        }
    }

    float gradC[9];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float v = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                v += tmpA[i * 3 + k] * b[k * 3 + j];
            }
            gradC[i * 3 + j] = v;
        }
    }

    float ct[9];
    ct[0] = c[0];
    ct[1] = c[3];
    ct[2] = c[6];
    ct[3] = c[1];
    ct[4] = c[4];
    ct[5] = c[7];
    ct[6] = c[2];
    ct[7] = c[5];
    ct[8] = c[8];

    float bct[9];
    float bc[9];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float v1 = 0.0;
            float v2 = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                v1 += b[i * 3 + k] * ct[k * 3 + j];
                v2 += b[i * 3 + k] * c[k * 3 + j];
            }
            bct[i * 3 + j] = v1;
            bc[i * 3 + j] = v2;
        }
    }

    float cotMt[9];
    cotMt[0] = cotM[0];
    cotMt[1] = cotM[3];
    cotMt[2] = cotM[6];
    cotMt[3] = cotM[1];
    cotMt[4] = cotM[4];
    cotMt[5] = cotM[7];
    cotMt[6] = cotM[2];
    cotMt[7] = cotM[5];
    cotMt[8] = cotM[8];

    float term1[9];
    float term2[9];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float v1 = 0.0;
            float v2 = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                v1 += cotM[i * 3 + k] * bct[k * 3 + j];
                v2 += cotMt[i * 3 + k] * bc[k * 3 + j];
            }
            term1[i * 3 + j] = v1;
            term2[i * 3 + j] = v2;
        }
    }

    float gradB[9];
    for (int i = 0; i < 9; ++i)
    {
        gradB[i] = term1[i] + term2[i];
    }

    float wt[9];
    wt[0] = w00;
    wt[1] = w10;
    wt[2] = w20;
    wt[3] = w01;
    wt[4] = w11;
    wt[5] = w21;
    wt[6] = w02;
    wt[7] = w12;
    wt[8] = w22;

    float gradJ[9];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float v = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                v += gradB[i * 3 + k] * wt[k * 3 + j];
            }
            gradJ[i * 3 + j] = v;
        }
    }

    float gJ00 = gradJ[0 * 3 + 0];
    float gJ02 = gradJ[0 * 3 + 2];
    float gJ11 = gradJ[1 * 3 + 1];
    float gJ12 = gradJ[1 * 3 + 2];

    float invTz2 = 1.0 / (tz * tz);
    float invTz3 = invTz2 / tz;
    float gTx = gJ02 * (-fused_bwd_focalX[0] * invTz2);
    float gTy = gJ12 * (-fused_bwd_focalY[0] * invTz2);
    float gTz = 0.0;
    gTz += gJ00 * (-fused_bwd_focalX[0] * invTz2);
    gTz += gJ11 * (-fused_bwd_focalY[0] * invTz2);
    gTz += gJ02 * (2.0 * fused_bwd_focalX[0] * tx * invTz3);
    gTz += gJ12 * (2.0 * fused_bwd_focalY[0] * ty * invTz3);

    float ux = t2 / clipX;
    float uy = t2 / clipY;
    float gT0 = gTx * ux;
    float gT1 = gTy * uy;
    float gUx = gTx * t0;
    float gUy = gTy * t1;

    float dClipX = (t2 > minX && t2 < maxX) ? 1.0 : 0.0;
    float dClipY = (t2 > minY && t2 < maxY) ? 1.0 : 0.0;
    float duxDt2 = (clipX - t2 * dClipX) / (clipX * clipX);
    float duyDt2 = (clipY - t2 * dClipY) / (clipY * clipY);
    gTz += gUx * duxDt2 + gUy * duyDt2;

    float covGradX = gT0 * a00 + gT1 * a01 + gTz * a02;
    float covGradY = gT0 * a10 + gT1 * a11 + gTz * a12;
    float covGradZ = gT0 * a20 + gT1 * a21 + gTz * a22;

    fused_bwd_gradMeans3dScreen[meansBase + 0] = colorGradX + covGradX;
    fused_bwd_gradMeans3dScreen[meansBase + 1] = colorGradY + covGradY;
    fused_bwd_gradMeans3dScreen[meansBase + 2] = colorGradZ + covGradZ;

    uint gradCov3dBase = p * 9;
    for (int i = 0; i < 9; ++i)
    {
        fused_bwd_gradCov3d[gradCov3dBase + uint(i)] = gradC[i];
    }

    fused_bwd_gradCameraCenterPoint[meansBase + 0] = -colorGradX;
    fused_bwd_gradCameraCenterPoint[meansBase + 1] = -colorGradY;
    fused_bwd_gradCameraCenterPoint[meansBase + 2] = -colorGradZ;

    uint gradMeanNdcBase = p * 4;
    uint cotMeansBase = p * 2;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 0] =
        fused_bwd_cotMeans2d[cotMeansBase + 0] * fused_bwd_imageWidth[0] * 0.5;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 1] =
        fused_bwd_cotMeans2d[cotMeansBase + 1] * fused_bwd_imageHeight[0] * 0.5;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 2] = 0.0;
    fused_bwd_gradMeanNdc[gradMeanNdcBase + 3] = 0.0;
}
