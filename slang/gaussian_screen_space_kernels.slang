StructuredBuffer<float> color_means3d;
StructuredBuffer<float> color_shs;
StructuredBuffer<float> color_cameraCenter;
StructuredBuffer<float> color_cotColor;
StructuredBuffer<uint> color_counts;
RWStructuredBuffer<float> color_outColor;
RWStructuredBuffer<float> color_gradMeans3d;
RWStructuredBuffer<float> color_gradShs;

StructuredBuffer<float> cov_means3d;
StructuredBuffer<float> cov_cov3d;
StructuredBuffer<float> cov_viewMatrix;
StructuredBuffer<float> cov_fovX;
StructuredBuffer<float> cov_fovY;
StructuredBuffer<float> cov_focalX;
StructuredBuffer<float> cov_focalY;
StructuredBuffer<float> cov_cotCov2d;
StructuredBuffer<uint> cov_counts;
RWStructuredBuffer<float> cov_outCov2d;
RWStructuredBuffer<float> cov_gradMeans3d;
RWStructuredBuffer<float> cov_gradCov3d;

StructuredBuffer<float> cov3d_scales;
StructuredBuffer<float> cov3d_rotations;
StructuredBuffer<float> cov3d_cotCov3d;
StructuredBuffer<uint> cov3d_counts;
RWStructuredBuffer<float> cov3d_outCov3d;
RWStructuredBuffer<float> cov3d_gradScales;
RWStructuredBuffer<float> cov3d_gradRotations;

StructuredBuffer<float> inverse_cov2d;
StructuredBuffer<float> inverse_cotConic;
StructuredBuffer<uint> inverse_counts;
RWStructuredBuffer<float> inverse_outConic;
RWStructuredBuffer<float> inverse_gradCov2d;

StructuredBuffer<float> fused_scales;
StructuredBuffer<float> fused_rotations;
StructuredBuffer<float> fused_means3d;
StructuredBuffer<float> fused_shs;
StructuredBuffer<float> fused_cameraCenter;
StructuredBuffer<float> fused_viewMatrix;
StructuredBuffer<float> fused_fovX;
StructuredBuffer<float> fused_fovY;
StructuredBuffer<float> fused_focalX;
StructuredBuffer<float> fused_focalY;
StructuredBuffer<uint> fused_counts;
RWStructuredBuffer<float> fused_outColor;
RWStructuredBuffer<float> fused_outCov2d;
RWStructuredBuffer<float> fused_outConic;

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_color_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = color_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    int degree = int(color_counts[1]);
    int coeffCount = (degree + 1) * (degree + 1);
    int coeffStride = int(color_counts[2]);
    uint meansBase = p * 3;

    float x = color_means3d[meansBase + 0] - color_cameraCenter[0];
    float y = color_means3d[meansBase + 1] - color_cameraCenter[1];
    float z = color_means3d[meansBase + 2] - color_cameraCenter[2];
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    float basis[25];
    for (int i = 0; i < 25; ++i)
    {
        basis[i] = 0.0;
    }

    basis[0] = 0.28209479177387814;
    if (degree > 0)
    {
        basis[1] = -0.4886025119029199 * y;
        basis[2] = 0.4886025119029199 * z;
        basis[3] = -0.4886025119029199 * x;
    }
    if (degree > 1)
    {
        basis[4] = 1.0925484305920792 * xy;
        basis[5] = -1.0925484305920792 * yz;
        basis[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
        basis[7] = -1.0925484305920792 * xz;
        basis[8] = 0.5462742152960396 * (xx - yy);
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
    }

    uint outBase = p * 3;
    uint shPointBase = p * uint(coeffStride * 3);
    for (int ch = 0; ch < 3; ++ch)
    {
        float value = 0.0;
        for (int k = 0; k < coeffCount; ++k)
        {
            uint shIndex = shPointBase + uint(k * 3 + ch);
            value += basis[k] * color_shs[shIndex];
        }
        float color = value + 0.5;
        color_outColor[outBase + uint(ch)] = color < 0.0 ? 0.0 : color;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_color_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = color_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    int degree = int(color_counts[1]);
    int coeffCount = (degree + 1) * (degree + 1);
    int coeffStride = int(color_counts[2]);
    uint meansBase = p * 3;

    float x = color_means3d[meansBase + 0] - color_cameraCenter[0];
    float y = color_means3d[meansBase + 1] - color_cameraCenter[1];
    float z = color_means3d[meansBase + 2] - color_cameraCenter[2];
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

    uint cotBase = p * 3;
    uint shPointBase = p * uint(coeffStride * 3);
    float gradX = 0.0;
    float gradY = 0.0;
    float gradZ = 0.0;

    for (int ch = 0; ch < 3; ++ch)
    {
        float value = 0.0;
        for (int k = 0; k < coeffCount; ++k)
        {
            uint shIndex = shPointBase + uint(k * 3 + ch);
            value += basis[k] * color_shs[shIndex];
        }

        float g = color_cotColor[cotBase + uint(ch)];
        if (value + 0.5 <= 0.0)
        {
            g = 0.0;
        }

        for (int k = 0; k < coeffCount; ++k)
        {
            uint shIndex = shPointBase + uint(k * 3 + ch);
            color_gradShs[shIndex] = g * basis[k];
            float shValue = color_shs[shIndex];
            gradX += g * shValue * dbx[k];
            gradY += g * shValue * dby[k];
            gradZ += g * shValue * dbz[k];
        }
    }

    color_gradMeans3d[meansBase + 0] = gradX;
    color_gradMeans3d[meansBase + 1] = gradY;
    color_gradMeans3d[meansBase + 2] = gradZ;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_cov2d_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = cov_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    float m0 = cov_means3d[p * 3 + 0];
    float m1 = cov_means3d[p * 3 + 1];
    float m2 = cov_means3d[p * 3 + 2];

    float a00 = cov_viewMatrix[0 * 4 + 0];
    float a01 = cov_viewMatrix[0 * 4 + 1];
    float a02 = cov_viewMatrix[0 * 4 + 2];
    float a10 = cov_viewMatrix[1 * 4 + 0];
    float a11 = cov_viewMatrix[1 * 4 + 1];
    float a12 = cov_viewMatrix[1 * 4 + 2];
    float a20 = cov_viewMatrix[2 * 4 + 0];
    float a21 = cov_viewMatrix[2 * 4 + 1];
    float a22 = cov_viewMatrix[2 * 4 + 2];

    float t0 = m0 * a00 + m1 * a10 + m2 * a20 + cov_viewMatrix[3 * 4 + 0];
    float t1 = m0 * a01 + m1 * a11 + m2 * a21 + cov_viewMatrix[3 * 4 + 1];
    float t2 = m0 * a02 + m1 * a12 + m2 * a22 + cov_viewMatrix[3 * 4 + 2];

    float tanFovX = tan(cov_fovX[0] * 0.5);
    float tanFovY = tan(cov_fovY[0] * 0.5);
    float clipX = clamp(t2, -tanFovX * 1.3, tanFovX * 1.3);
    float clipY = clamp(t2, -tanFovY * 1.3, tanFovY * 1.3);
    float tx = t0 / clipX * t2;
    float ty = t1 / clipY * t2;
    float tz = t2;

    float j00 = cov_focalX[0] / tz;
    float j02 = -tx * cov_focalX[0] / (tz * tz);
    float j11 = cov_focalY[0] / tz;
    float j12 = -ty * cov_focalY[0] / (tz * tz);

    float w00 = a00;
    float w01 = a10;
    float w02 = a20;
    float w10 = a01;
    float w11 = a11;
    float w12 = a21;
    float w20 = a02;
    float w21 = a12;
    float w22 = a22;

    float b00 = j00 * w00 + j02 * w20;
    float b01 = j00 * w01 + j02 * w21;
    float b02 = j00 * w02 + j02 * w22;
    float b10 = j11 * w10 + j12 * w20;
    float b11 = j11 * w11 + j12 * w21;
    float b12 = j11 * w12 + j12 * w22;

    uint covBase = p * 9;
    float c00 = cov_cov3d[covBase + 0];
    float c01 = cov_cov3d[covBase + 1];
    float c02 = cov_cov3d[covBase + 2];
    float c10 = cov_cov3d[covBase + 3];
    float c11 = cov_cov3d[covBase + 4];
    float c12 = cov_cov3d[covBase + 5];
    float c20 = cov_cov3d[covBase + 6];
    float c21 = cov_cov3d[covBase + 7];
    float c22 = cov_cov3d[covBase + 8];

    float t00 = b00 * c00 + b01 * c10 + b02 * c20;
    float t01 = b00 * c01 + b01 * c11 + b02 * c21;
    float t02 = b00 * c02 + b01 * c12 + b02 * c22;
    float t10 = b10 * c00 + b11 * c10 + b12 * c20;
    float t11 = b10 * c01 + b11 * c11 + b12 * c21;
    float t12 = b10 * c02 + b11 * c12 + b12 * c22;

    uint outBase = p * 4;
    cov_outCov2d[outBase + 0] = t00 * b00 + t01 * b01 + t02 * b02 + 0.3;
    cov_outCov2d[outBase + 1] = t00 * b10 + t01 * b11 + t02 * b12;
    cov_outCov2d[outBase + 2] = t10 * b00 + t11 * b01 + t12 * b02;
    cov_outCov2d[outBase + 3] = t10 * b10 + t11 * b11 + t12 * b12 + 0.3;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_cov2d_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = cov_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    float m0 = cov_means3d[p * 3 + 0];
    float m1 = cov_means3d[p * 3 + 1];
    float m2 = cov_means3d[p * 3 + 2];

    float a00 = cov_viewMatrix[0 * 4 + 0];
    float a01 = cov_viewMatrix[0 * 4 + 1];
    float a02 = cov_viewMatrix[0 * 4 + 2];
    float a10 = cov_viewMatrix[1 * 4 + 0];
    float a11 = cov_viewMatrix[1 * 4 + 1];
    float a12 = cov_viewMatrix[1 * 4 + 2];
    float a20 = cov_viewMatrix[2 * 4 + 0];
    float a21 = cov_viewMatrix[2 * 4 + 1];
    float a22 = cov_viewMatrix[2 * 4 + 2];

    float t0 = m0 * a00 + m1 * a10 + m2 * a20 + cov_viewMatrix[3 * 4 + 0];
    float t1 = m0 * a01 + m1 * a11 + m2 * a21 + cov_viewMatrix[3 * 4 + 1];
    float t2 = m0 * a02 + m1 * a12 + m2 * a22 + cov_viewMatrix[3 * 4 + 2];

    float tanFovX = tan(cov_fovX[0] * 0.5);
    float tanFovY = tan(cov_fovY[0] * 0.5);
    float minX = -tanFovX * 1.3;
    float maxX = tanFovX * 1.3;
    float minY = -tanFovY * 1.3;
    float maxY = tanFovY * 1.3;
    float clipX = clamp(t2, minX, maxX);
    float clipY = clamp(t2, minY, maxY);
    float tx = t0 / clipX * t2;
    float ty = t1 / clipY * t2;
    float tz = t2;

    float j00 = cov_focalX[0] / tz;
    float j02 = -tx * cov_focalX[0] / (tz * tz);
    float j11 = cov_focalY[0] / tz;
    float j12 = -ty * cov_focalY[0] / (tz * tz);

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

    uint covBase = p * 9;
    float c[9];
    c[0] = cov_cov3d[covBase + 0];
    c[1] = cov_cov3d[covBase + 1];
    c[2] = cov_cov3d[covBase + 2];
    c[3] = cov_cov3d[covBase + 3];
    c[4] = cov_cov3d[covBase + 4];
    c[5] = cov_cov3d[covBase + 5];
    c[6] = cov_cov3d[covBase + 6];
    c[7] = cov_cov3d[covBase + 7];
    c[8] = cov_cov3d[covBase + 8];

    uint cotBase = p * 4;
    float cotM[9];
    cotM[0] = cov_cotCov2d[cotBase + 0];
    cotM[1] = cov_cotCov2d[cotBase + 1];
    cotM[2] = 0.0;
    cotM[3] = cov_cotCov2d[cotBase + 2];
    cotM[4] = cov_cotCov2d[cotBase + 3];
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
    float gTx = gJ02 * (-cov_focalX[0] * invTz2);
    float gTy = gJ12 * (-cov_focalY[0] * invTz2);
    float gTz = 0.0;
    gTz += gJ00 * (-cov_focalX[0] * invTz2);
    gTz += gJ11 * (-cov_focalY[0] * invTz2);
    gTz += gJ02 * (2.0 * cov_focalX[0] * tx * invTz3);
    gTz += gJ12 * (2.0 * cov_focalY[0] * ty * invTz3);

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

    cov_gradMeans3d[p * 3 + 0] = gT0 * a00 + gT1 * a01 + gTz * a02;
    cov_gradMeans3d[p * 3 + 1] = gT0 * a10 + gT1 * a11 + gTz * a12;
    cov_gradMeans3d[p * 3 + 2] = gT0 * a20 + gT1 * a21 + gTz * a22;

    for (int i = 0; i < 9; ++i)
    {
        cov_gradCov3d[covBase + uint(i)] = gradC[i];
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_cov3d_forward(uint3 tid : SV_DispatchThreadID)
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

    float norm = sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
    float safeNorm = max(norm, 1e-8);
    float w = rw / safeNorm;
    float x = rx / safeNorm;
    float y = ry / safeNorm;
    float z = rz / safeNorm;

    float r00 = 1.0 - 2.0 * (y * y + z * z);
    float r01 = 2.0 * (x * y - w * z);
    float r02 = 2.0 * (x * z + w * y);
    float r10 = 2.0 * (x * y + w * z);
    float r11 = 1.0 - 2.0 * (x * x + z * z);
    float r12 = 2.0 * (y * z - w * x);
    float r20 = 2.0 * (x * z - w * y);
    float r21 = 2.0 * (y * z + w * x);
    float r22 = 1.0 - 2.0 * (x * x + y * y);

    float l00 = r00 * sx;
    float l01 = r01 * sy;
    float l02 = r02 * sz;
    float l10 = r10 * sx;
    float l11 = r11 * sy;
    float l12 = r12 * sz;
    float l20 = r20 * sx;
    float l21 = r21 * sy;
    float l22 = r22 * sz;

    uint outBase = p * 9;
    cov3d_outCov3d[outBase + 0] = l00 * l00 + l01 * l01 + l02 * l02;
    cov3d_outCov3d[outBase + 1] = l00 * l10 + l01 * l11 + l02 * l12;
    cov3d_outCov3d[outBase + 2] = l00 * l20 + l01 * l21 + l02 * l22;
    cov3d_outCov3d[outBase + 3] = l10 * l00 + l11 * l01 + l12 * l02;
    cov3d_outCov3d[outBase + 4] = l10 * l10 + l11 * l11 + l12 * l12;
    cov3d_outCov3d[outBase + 5] = l10 * l20 + l11 * l21 + l12 * l22;
    cov3d_outCov3d[outBase + 6] = l20 * l00 + l21 * l01 + l22 * l02;
    cov3d_outCov3d[outBase + 7] = l20 * l10 + l21 * l11 + l22 * l12;
    cov3d_outCov3d[outBase + 8] = l20 * l20 + l21 * l21 + l22 * l22;
}

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

    float norm = sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
    float safeNorm = max(norm, 1e-8);
    float w = rw / safeNorm;
    float x = rx / safeNorm;
    float y = ry / safeNorm;
    float z = rz / safeNorm;

    float r00 = 1.0 - 2.0 * (y * y + z * z);
    float r01 = 2.0 * (x * y - w * z);
    float r02 = 2.0 * (x * z + w * y);
    float r10 = 2.0 * (x * y + w * z);
    float r11 = 1.0 - 2.0 * (x * x + z * z);
    float r12 = 2.0 * (y * z - w * x);
    float r20 = 2.0 * (x * z - w * y);
    float r21 = 2.0 * (y * z + w * x);
    float r22 = 1.0 - 2.0 * (x * x + y * y);

    float l00 = r00 * sx;
    float l01 = r01 * sy;
    float l02 = r02 * sz;
    float l10 = r10 * sx;
    float l11 = r11 * sy;
    float l12 = r12 * sz;
    float l20 = r20 * sx;
    float l21 = r21 * sy;
    float l22 = r22 * sz;

    uint cotBase = p * 9;
    float g00 = cov3d_cotCov3d[cotBase + 0];
    float g01 = cov3d_cotCov3d[cotBase + 1];
    float g02 = cov3d_cotCov3d[cotBase + 2];
    float g10 = cov3d_cotCov3d[cotBase + 3];
    float g11 = cov3d_cotCov3d[cotBase + 4];
    float g12 = cov3d_cotCov3d[cotBase + 5];
    float g20 = cov3d_cotCov3d[cotBase + 6];
    float g21 = cov3d_cotCov3d[cotBase + 7];
    float g22 = cov3d_cotCov3d[cotBase + 8];

    float h00 = g00 + g00;
    float h01 = g01 + g10;
    float h02 = g02 + g20;
    float h10 = g10 + g01;
    float h11 = g11 + g11;
    float h12 = g12 + g21;
    float h20 = g20 + g02;
    float h21 = g21 + g12;
    float h22 = g22 + g22;

    float gl00 = h00 * l00 + h01 * l10 + h02 * l20;
    float gl01 = h00 * l01 + h01 * l11 + h02 * l21;
    float gl02 = h00 * l02 + h01 * l12 + h02 * l22;
    float gl10 = h10 * l00 + h11 * l10 + h12 * l20;
    float gl11 = h10 * l01 + h11 * l11 + h12 * l21;
    float gl12 = h10 * l02 + h11 * l12 + h12 * l22;
    float gl20 = h20 * l00 + h21 * l10 + h22 * l20;
    float gl21 = h20 * l01 + h21 * l11 + h22 * l21;
    float gl22 = h20 * l02 + h21 * l12 + h22 * l22;

    cov3d_gradScales[scaleBase + 0] = gl00 * r00 + gl10 * r10 + gl20 * r20;
    cov3d_gradScales[scaleBase + 1] = gl01 * r01 + gl11 * r11 + gl21 * r21;
    cov3d_gradScales[scaleBase + 2] = gl02 * r02 + gl12 * r12 + gl22 * r22;

    float gr00 = gl00 * sx;
    float gr01 = gl01 * sy;
    float gr02 = gl02 * sz;
    float gr10 = gl10 * sx;
    float gr11 = gl11 * sy;
    float gr12 = gl12 * sz;
    float gr20 = gl20 * sx;
    float gr21 = gl21 * sy;
    float gr22 = gl22 * sz;

    float gw = 0.0;
    gw += gr01 * (-2.0 * z);
    gw += gr02 * (2.0 * y);
    gw += gr10 * (2.0 * z);
    gw += gr12 * (-2.0 * x);
    gw += gr20 * (-2.0 * y);
    gw += gr21 * (2.0 * x);

    float gx = 0.0;
    gx += gr01 * (2.0 * y);
    gx += gr02 * (2.0 * z);
    gx += gr10 * (2.0 * y);
    gx += gr11 * (-4.0 * x);
    gx += gr12 * (-2.0 * w);
    gx += gr20 * (2.0 * z);
    gx += gr21 * (2.0 * w);
    gx += gr22 * (-4.0 * x);

    float gy = 0.0;
    gy += gr00 * (-4.0 * y);
    gy += gr01 * (2.0 * x);
    gy += gr02 * (2.0 * w);
    gy += gr10 * (2.0 * x);
    gy += gr12 * (2.0 * z);
    gy += gr20 * (-2.0 * w);
    gy += gr21 * (2.0 * z);
    gy += gr22 * (-4.0 * y);

    float gz = 0.0;
    gz += gr00 * (-4.0 * z);
    gz += gr01 * (-2.0 * w);
    gz += gr02 * (2.0 * x);
    gz += gr10 * (2.0 * w);
    gz += gr11 * (-4.0 * z);
    gz += gr12 * (2.0 * y);
    gz += gr20 * (2.0 * x);
    gz += gr21 * (2.0 * y);

    float invNorm = 1.0 / safeNorm;
    float gradRw;
    float gradRx;
    float gradRy;
    float gradRz;
    if (norm > 1e-8)
    {
        float invNorm3 = invNorm * invNorm * invNorm;
        float dot = gw * rw + gx * rx + gy * ry + gz * rz;
        gradRw = gw * invNorm - rw * dot * invNorm3;
        gradRx = gx * invNorm - rx * dot * invNorm3;
        gradRy = gy * invNorm - ry * dot * invNorm3;
        gradRz = gz * invNorm - rz * dot * invNorm3;
    }
    else
    {
        gradRw = gw * invNorm;
        gradRx = gx * invNorm;
        gradRy = gy * invNorm;
        gradRz = gz * invNorm;
    }

    cov3d_gradRotations[rotBase + 0] = gradRw;
    cov3d_gradRotations[rotBase + 1] = gradRx;
    cov3d_gradRotations[rotBase + 2] = gradRy;
    cov3d_gradRotations[rotBase + 3] = gradRz;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_inverse2d_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = inverse_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    uint base = p * 4;
    float a = inverse_cov2d[base + 0];
    float b = inverse_cov2d[base + 1];
    float c = inverse_cov2d[base + 2];
    float d = inverse_cov2d[base + 3];
    float det = a * d - b * c;
    inverse_outConic[base + 0] = d / det;
    inverse_outConic[base + 1] = -b / det;
    inverse_outConic[base + 2] = -c / det;
    inverse_outConic[base + 3] = a / det;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_inverse2d_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = inverse_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    uint base = p * 4;
    float a = inverse_cov2d[base + 0];
    float b = inverse_cov2d[base + 1];
    float c = inverse_cov2d[base + 2];
    float d = inverse_cov2d[base + 3];
    float det = a * d - b * c;
    float inv00 = d / det;
    float inv01 = -b / det;
    float inv10 = -c / det;
    float inv11 = a / det;

    float g00 = inverse_cotConic[base + 0];
    float g01 = inverse_cotConic[base + 1];
    float g10 = inverse_cotConic[base + 2];
    float g11 = inverse_cotConic[base + 3];

    float t00 = inv00 * g00 + inv10 * g10;
    float t01 = inv00 * g01 + inv10 * g11;
    float t10 = inv01 * g00 + inv11 * g10;
    float t11 = inv01 * g01 + inv11 * g11;

    inverse_gradCov2d[base + 0] = -(t00 * inv00 + t01 * inv01);
    inverse_gradCov2d[base + 1] = -(t00 * inv10 + t01 * inv11);
    inverse_gradCov2d[base + 2] = -(t10 * inv00 + t11 * inv01);
    inverse_gradCov2d[base + 3] = -(t10 * inv10 + t11 * inv11);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_screen_fused_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pointCount = fused_counts[0];
    if (p >= pointCount)
    {
        return;
    }

    // ---- cov3d from scales/rotations ----
    uint scaleBase = p * 3;
    float sx = fused_scales[scaleBase + 0];
    float sy = fused_scales[scaleBase + 1];
    float sz = fused_scales[scaleBase + 2];

    uint rotBase = p * 4;
    float rw = fused_rotations[rotBase + 0];
    float rx = fused_rotations[rotBase + 1];
    float ry = fused_rotations[rotBase + 2];
    float rz = fused_rotations[rotBase + 3];

    float norm = sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
    float safeNorm = max(norm, 1e-8);
    float qw = rw / safeNorm;
    float qx = rx / safeNorm;
    float qy = ry / safeNorm;
    float qz = rz / safeNorm;

    float r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
    float r01 = 2.0 * (qx * qy - qw * qz);
    float r02 = 2.0 * (qx * qz + qw * qy);
    float r10 = 2.0 * (qx * qy + qw * qz);
    float r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
    float r12 = 2.0 * (qy * qz - qw * qx);
    float r20 = 2.0 * (qx * qz - qw * qy);
    float r21 = 2.0 * (qy * qz + qw * qx);
    float r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

    float l00 = r00 * sx;
    float l01 = r01 * sy;
    float l02 = r02 * sz;
    float l10 = r10 * sx;
    float l11 = r11 * sy;
    float l12 = r12 * sz;
    float l20 = r20 * sx;
    float l21 = r21 * sy;
    float l22 = r22 * sz;

    float cv3d00 = l00 * l00 + l01 * l01 + l02 * l02;
    float cv3d01 = l00 * l10 + l01 * l11 + l02 * l12;
    float cv3d02 = l00 * l20 + l01 * l21 + l02 * l22;
    float cv3d10 = l10 * l00 + l11 * l01 + l12 * l02;
    float cv3d11 = l10 * l10 + l11 * l11 + l12 * l12;
    float cv3d12 = l10 * l20 + l11 * l21 + l12 * l22;
    float cv3d20 = l20 * l00 + l21 * l01 + l22 * l02;
    float cv3d21 = l20 * l10 + l21 * l11 + l22 * l12;
    float cv3d22 = l20 * l20 + l21 * l21 + l22 * l22;

    // ---- SH basis + color ----
    uint meansBase = p * 3;
    int degree = int(fused_counts[1]);
    int coeffCount = (degree + 1) * (degree + 1);
    int coeffStride = int(fused_counts[2]);

    float x = fused_means3d[meansBase + 0] - fused_cameraCenter[0];
    float y = fused_means3d[meansBase + 1] - fused_cameraCenter[1];
    float z = fused_means3d[meansBase + 2] - fused_cameraCenter[2];
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;

    float basis[25];
    for (int i = 0; i < 25; ++i)
    {
        basis[i] = 0.0;
    }

    basis[0] = 0.28209479177387814;
    if (degree > 0)
    {
        basis[1] = -0.4886025119029199 * y;
        basis[2] = 0.4886025119029199 * z;
        basis[3] = -0.4886025119029199 * x;
    }
    if (degree > 1)
    {
        basis[4] = 1.0925484305920792 * xy;
        basis[5] = -1.0925484305920792 * yz;
        basis[6] = 0.31539156525252005 * (2.0 * zz - xx - yy);
        basis[7] = -1.0925484305920792 * xz;
        basis[8] = 0.5462742152960396 * (xx - yy);
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
    }

    uint colorBase = p * 3;
    uint shPointBase = p * uint(coeffStride * 3);
    for (int ch = 0; ch < 3; ++ch)
    {
        float value = 0.0;
        for (int k = 0; k < coeffCount; ++k)
        {
            uint shIndex = shPointBase + uint(k * 3 + ch);
            value += basis[k] * fused_shs[shIndex];
        }
        float color = value + 0.5;
        fused_outColor[colorBase + uint(ch)] = color < 0.0 ? 0.0 : color;
    }

    // ---- cov2d from cov3d ----
    float m0 = fused_means3d[meansBase + 0];
    float m1 = fused_means3d[meansBase + 1];
    float m2 = fused_means3d[meansBase + 2];

    float a00 = fused_viewMatrix[0 * 4 + 0];
    float a01 = fused_viewMatrix[0 * 4 + 1];
    float a02 = fused_viewMatrix[0 * 4 + 2];
    float a10 = fused_viewMatrix[1 * 4 + 0];
    float a11 = fused_viewMatrix[1 * 4 + 1];
    float a12 = fused_viewMatrix[1 * 4 + 2];
    float a20 = fused_viewMatrix[2 * 4 + 0];
    float a21 = fused_viewMatrix[2 * 4 + 1];
    float a22 = fused_viewMatrix[2 * 4 + 2];

    float t0 = m0 * a00 + m1 * a10 + m2 * a20 + fused_viewMatrix[3 * 4 + 0];
    float t1 = m0 * a01 + m1 * a11 + m2 * a21 + fused_viewMatrix[3 * 4 + 1];
    float t2 = m0 * a02 + m1 * a12 + m2 * a22 + fused_viewMatrix[3 * 4 + 2];

    float tanFovX = tan(fused_fovX[0] * 0.5);
    float tanFovY = tan(fused_fovY[0] * 0.5);
    float clipX = clamp(t2, -tanFovX * 1.3, tanFovX * 1.3);
    float clipY = clamp(t2, -tanFovY * 1.3, tanFovY * 1.3);
    float tx = t0 / clipX * t2;
    float ty = t1 / clipY * t2;
    float tz = t2;

    float j00 = fused_focalX[0] / tz;
    float j02 = -tx * fused_focalX[0] / (tz * tz);
    float j11 = fused_focalY[0] / tz;
    float j12 = -ty * fused_focalY[0] / (tz * tz);

    float w00 = a00;
    float w01 = a10;
    float w02 = a20;
    float w10 = a01;
    float w11 = a11;
    float w12 = a21;
    float w20 = a02;
    float w21 = a12;
    float w22 = a22;

    float b00 = j00 * w00 + j02 * w20;
    float b01 = j00 * w01 + j02 * w21;
    float b02 = j00 * w02 + j02 * w22;
    float b10 = j11 * w10 + j12 * w20;
    float b11 = j11 * w11 + j12 * w21;
    float b12 = j11 * w12 + j12 * w22;

    float tt00 = b00 * cv3d00 + b01 * cv3d10 + b02 * cv3d20;
    float tt01 = b00 * cv3d01 + b01 * cv3d11 + b02 * cv3d21;
    float tt02 = b00 * cv3d02 + b01 * cv3d12 + b02 * cv3d22;
    float tt10 = b10 * cv3d00 + b11 * cv3d10 + b12 * cv3d20;
    float tt11 = b10 * cv3d01 + b11 * cv3d11 + b12 * cv3d21;
    float tt12 = b10 * cv3d02 + b11 * cv3d12 + b12 * cv3d22;

    uint outBase = p * 4;
    float cv2d00 = tt00 * b00 + tt01 * b01 + tt02 * b02 + 0.3;
    float cv2d01 = tt00 * b10 + tt01 * b11 + tt02 * b12;
    float cv2d10 = tt10 * b00 + tt11 * b01 + tt12 * b02;
    float cv2d11 = tt10 * b10 + tt11 * b11 + tt12 * b12 + 0.3;

    fused_outCov2d[outBase + 0] = cv2d00;
    fused_outCov2d[outBase + 1] = cv2d01;
    fused_outCov2d[outBase + 2] = cv2d10;
    fused_outCov2d[outBase + 3] = cv2d11;

    // ---- conic = inv(cov2d) ----
    float det = cv2d00 * cv2d11 - cv2d01 * cv2d10;
    fused_outConic[outBase + 0] = cv2d11 / det;
    fused_outConic[outBase + 1] = -cv2d01 / det;
    fused_outConic[outBase + 2] = -cv2d10 / det;
    fused_outConic[outBase + 3] = cv2d00 / det;
}
