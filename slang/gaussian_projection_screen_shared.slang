struct ProjectionNdcOutputs : IDifferentiable
{
    float4 meanNdc;
    float4 pView;
};

struct Cov3dValues : IDifferentiable
{
    float c00;
    float c01;
    float c02;
    float c10;
    float c11;
    float c12;
    float c20;
    float c21;
    float c22;
};

struct Cov2dValues : IDifferentiable
{
    float c00;
    float c01;
    float c10;
    float c11;
};

Cov2dValues makeCov2d(float c00, float c01, float c10, float c11)
{
    Cov2dValues outVals;
    outVals.c00 = c00;
    outVals.c01 = c01;
    outVals.c10 = c10;
    outVals.c11 = c11;
    return outVals;
}

Cov3dValues zeroCov3d()
{
    Cov3dValues outVals;
    outVals.c00 = 0.0;
    outVals.c01 = 0.0;
    outVals.c02 = 0.0;
    outVals.c10 = 0.0;
    outVals.c11 = 0.0;
    outVals.c12 = 0.0;
    outVals.c20 = 0.0;
    outVals.c21 = 0.0;
    outVals.c22 = 0.0;
    return outVals;
}

[Differentiable]
ProjectionNdcOutputs evaluateProjectionNdcOutputs(
    float3 point,
    no_diff float view00,
    no_diff float view01,
    no_diff float view02,
    no_diff float view03,
    no_diff float view10,
    no_diff float view11,
    no_diff float view12,
    no_diff float view13,
    no_diff float view20,
    no_diff float view21,
    no_diff float view22,
    no_diff float view23,
    no_diff float view30,
    no_diff float view31,
    no_diff float view32,
    no_diff float view33,
    no_diff float proj00,
    no_diff float proj01,
    no_diff float proj02,
    no_diff float proj03,
    no_diff float proj10,
    no_diff float proj11,
    no_diff float proj12,
    no_diff float proj13,
    no_diff float proj20,
    no_diff float proj21,
    no_diff float proj22,
    no_diff float proj23,
    no_diff float proj30,
    no_diff float proj31,
    no_diff float proj32,
    no_diff float proj33
)
{
    ProjectionNdcOutputs outVals;

    float pv0 = point.x * view00 + point.y * view10 + point.z * view20 + view30;
    float pv1 = point.x * view01 + point.y * view11 + point.z * view21 + view31;
    float pv2 = point.x * view02 + point.y * view12 + point.z * view22 + view32;
    float pv3 = point.x * view03 + point.y * view13 + point.z * view23 + view33;

    float pc0 = pv0 * proj00 + pv1 * proj10 + pv2 * proj20 + pv3 * proj30;
    float pc1 = pv0 * proj01 + pv1 * proj11 + pv2 * proj21 + pv3 * proj31;
    float pc2 = pv0 * proj02 + pv1 * proj12 + pv2 * proj22 + pv3 * proj32;
    float pc3 = pv0 * proj03 + pv1 * proj13 + pv2 * proj23 + pv3 * proj33;

    float wInv = 1.0f / (pc3 + 0.000001f);
    outVals.meanNdc = float4(pc0, pc1, pc2, pc3) * wInv;
    outVals.pView = float4(pv0, pv1, pv2, pv3);

    return outVals;
}

[Differentiable]
float2 ndcToScreen(float ndcX, float ndcY, no_diff float imageWidth, no_diff float imageHeight)
{
    float sx = ((ndcX + 1.0f) * imageWidth - 1.0f) * 0.5f;
    float sy = ((ndcY + 1.0f) * imageHeight - 1.0f) * 0.5f;
    return float2(sx, sy);
}

[Differentiable]
Cov3dValues buildCov3dFromScaleRotation(
    float sx,
    float sy,
    float sz,
    float rw,
    float rx,
    float ry,
    float rz
)
{
    Cov3dValues outVals;

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

    outVals.c00 = l00 * l00 + l01 * l01 + l02 * l02;
    outVals.c01 = l00 * l10 + l01 * l11 + l02 * l12;
    outVals.c02 = l00 * l20 + l01 * l21 + l02 * l22;
    outVals.c10 = l10 * l00 + l11 * l01 + l12 * l02;
    outVals.c11 = l10 * l10 + l11 * l11 + l12 * l12;
    outVals.c12 = l10 * l20 + l11 * l21 + l12 * l22;
    outVals.c20 = l20 * l00 + l21 * l01 + l22 * l02;
    outVals.c21 = l20 * l10 + l21 * l11 + l22 * l12;
    outVals.c22 = l20 * l20 + l21 * l21 + l22 * l22;

    return outVals;
}

[Differentiable]
Cov2dValues buildCov2dFromCov3d(
    float m0,
    float m1,
    float m2,
    Cov3dValues cov3d,
    no_diff float a00,
    no_diff float a01,
    no_diff float a02,
    no_diff float a10,
    no_diff float a11,
    no_diff float a12,
    no_diff float a20,
    no_diff float a21,
    no_diff float a22,
    no_diff float t30,
    no_diff float t31,
    no_diff float t32,
    no_diff float fovX,
    no_diff float fovY,
    no_diff float focalX,
    no_diff float focalY
)
{
    Cov2dValues outVals;

    float t0 = m0 * a00 + m1 * a10 + m2 * a20 + t30;
    float t1 = m0 * a01 + m1 * a11 + m2 * a21 + t31;
    float t2 = m0 * a02 + m1 * a12 + m2 * a22 + t32;

    float tanFovX = tan(fovX * 0.5);
    float tanFovY = tan(fovY * 0.5);
    float clipX = clamp(t2, -tanFovX * 1.3, tanFovX * 1.3);
    float clipY = clamp(t2, -tanFovY * 1.3, tanFovY * 1.3);
    float tx = t0 / clipX * t2;
    float ty = t1 / clipY * t2;
    float tz = t2;

    float j00 = focalX / tz;
    float j02 = -tx * focalX / (tz * tz);
    float j11 = focalY / tz;
    float j12 = -ty * focalY / (tz * tz);

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

    float t00 = b00 * cov3d.c00 + b01 * cov3d.c10 + b02 * cov3d.c20;
    float t01 = b00 * cov3d.c01 + b01 * cov3d.c11 + b02 * cov3d.c21;
    float t02 = b00 * cov3d.c02 + b01 * cov3d.c12 + b02 * cov3d.c22;
    float t10 = b10 * cov3d.c00 + b11 * cov3d.c10 + b12 * cov3d.c20;
    float t11 = b10 * cov3d.c01 + b11 * cov3d.c11 + b12 * cov3d.c21;
    float t12 = b10 * cov3d.c02 + b11 * cov3d.c12 + b12 * cov3d.c22;

    outVals.c00 = t00 * b00 + t01 * b01 + t02 * b02 + 0.3;
    outVals.c01 = t00 * b10 + t01 * b11 + t02 * b12;
    outVals.c10 = t10 * b00 + t11 * b01 + t12 * b02;
    outVals.c11 = t10 * b10 + t11 * b11 + t12 * b12 + 0.3;

    return outVals;
}

[Differentiable]
float4 inverseCov2d(Cov2dValues cov2d)
{
    float det = cov2d.c00 * cov2d.c11 - cov2d.c01 * cov2d.c10;
    return float4(
        cov2d.c11 / det,
        -cov2d.c01 / det,
        -cov2d.c10 / det,
        cov2d.c00 / det
    );
}

[Differentiable]
float3 evaluateShColorFromPoint(
    float3 means3d,
    float3 cameraCenter,
    float3 shCoeffs[25],
    no_diff int degree
)
{
    float x = means3d.x - cameraCenter.x;
    float y = means3d.y - cameraCenter.y;
    float z = means3d.z - cameraCenter.z;

    float3 colorAccum = 0.28209479177387814 * shCoeffs[0];
    if (degree > 0)
    {
        colorAccum += (-0.4886025119029199 * y) * shCoeffs[1];
        colorAccum += (0.4886025119029199 * z) * shCoeffs[2];
        colorAccum += (-0.4886025119029199 * x) * shCoeffs[3];

        if (degree > 1)
        {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;

            colorAccum += (1.0925484305920792 * xy) * shCoeffs[4];
            colorAccum += (-1.0925484305920792 * yz) * shCoeffs[5];
            colorAccum += (0.31539156525252005 * (2.0 * zz - xx - yy)) * shCoeffs[6];
            colorAccum += (-1.0925484305920792 * xz) * shCoeffs[7];
            colorAccum += (0.5462742152960396 * (xx - yy)) * shCoeffs[8];

            if (degree > 2)
            {
                colorAccum += (-0.5900435899266435 * y * (3.0 * xx - yy)) * shCoeffs[9];
                colorAccum += (2.890611442640554 * xy * z) * shCoeffs[10];
                colorAccum += (-0.4570457994644658 * y * (4.0 * zz - xx - yy)) * shCoeffs[11];
                colorAccum += (0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * shCoeffs[12];
                colorAccum += (-0.4570457994644658 * x * (4.0 * zz - xx - yy)) * shCoeffs[13];
                colorAccum += (1.445305721320277 * z * (xx - yy)) * shCoeffs[14];
                colorAccum += (-0.5900435899266435 * x * (xx - 3.0 * yy)) * shCoeffs[15];

                if (degree > 3)
                {
                    colorAccum += (2.5033429417967046 * xy * (xx - yy)) * shCoeffs[16];
                    colorAccum += (-1.7701307697799304 * yz * (3.0 * xx - yy)) * shCoeffs[17];
                    colorAccum += (0.9461746957575601 * xy * (7.0 * zz - 1.0)) * shCoeffs[18];
                    colorAccum += (-0.6690465435572892 * yz * (7.0 * zz - 3.0)) * shCoeffs[19];
                    colorAccum += (0.10578554691520431 * (zz * (35.0 * zz - 30.0) + 3.0)) * shCoeffs[20];
                    colorAccum += (-0.6690465435572892 * xz * (7.0 * zz - 3.0)) * shCoeffs[21];
                    colorAccum += (0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0)) * shCoeffs[22];
                    colorAccum += (-1.7701307697799304 * xz * (xx - 3.0 * yy)) * shCoeffs[23];
                    colorAccum += (0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy))) * shCoeffs[24];
                }
            }
        }
    }

    colorAccum += 0.5;
    return max(colorAccum, float3(0.0));
}

struct ProjectionScreenGradOutputs : IDifferentiable
{
    float2 means2d;
    float3 color;
    Cov2dValues cov2d;
    float4 conic;
};

[Differentiable]
ProjectionScreenGradOutputs evaluateProjectionScreenGradOutputs(
    float2 meanNdcXY,
    float3 means3d,
    float3 cameraCenter,
    float3 shCoeffs[25],
    Cov3dValues cov3d,
    no_diff int degree,
    no_diff float a00,
    no_diff float a01,
    no_diff float a02,
    no_diff float a10,
    no_diff float a11,
    no_diff float a12,
    no_diff float a20,
    no_diff float a21,
    no_diff float a22,
    no_diff float t30,
    no_diff float t31,
    no_diff float t32,
    no_diff float fovX,
    no_diff float fovY,
    no_diff float focalX,
    no_diff float focalY,
    no_diff float imageWidth,
    no_diff float imageHeight
)
{
    ProjectionScreenGradOutputs outVals;
    outVals.means2d = ndcToScreen(meanNdcXY.x, meanNdcXY.y, imageWidth, imageHeight);
    outVals.color = evaluateShColorFromPoint(means3d, cameraCenter, shCoeffs, degree);
    outVals.cov2d = buildCov2dFromCov3d(
        means3d.x,
        means3d.y,
        means3d.z,
        cov3d,
        a00, a01, a02,
        a10, a11, a12,
        a20, a21, a22,
        t30, t31, t32,
        fovX, fovY, focalX, focalY
    );
    outVals.conic = inverseCov2d(outVals.cov2d);
    return outVals;
}

float computeRadiusFromCov2d(Cov2dValues cov2d)
{
    float detVal = cov2d.c00 * cov2d.c11 - cov2d.c01 * cov2d.c10;
    float mid = 0.5 * (cov2d.c00 + cov2d.c11);
    float delta = max(mid * mid - detVal, 1e-5);
    float lambdaMax = mid + sqrt(delta);
    return 3.0 * ceil(sqrt(lambdaMax));
}
