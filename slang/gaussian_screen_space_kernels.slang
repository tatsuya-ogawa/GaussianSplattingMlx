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

StructuredBuffer<float> inverse_cov2d;
StructuredBuffer<float> inverse_cotConic;
StructuredBuffer<uint> inverse_counts;
RWStructuredBuffer<float> inverse_outConic;
RWStructuredBuffer<float> inverse_gradCov2d;

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
