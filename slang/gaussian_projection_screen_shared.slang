struct ProjectionValues
{
    float ndc0;
    float ndc1;
    float ndc2;
    float ndc3;
    float pv0;
    float pv1;
    float pv2;
    float pv3;
    float visibleMask;
};

struct Cov3dValues
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

struct Cov2dValues
{
    float c00;
    float c01;
    float c10;
    float c11;
};

ProjectionValues projectFromHomogeneous(
    float pv0,
    float pv1,
    float pv2,
    float pv3,
    float pc0,
    float pc1,
    float pc2,
    float pc3
)
{
    ProjectionValues outVals;

    float wInv = 1.0f / (pc3 + 0.000001f);

    outVals.ndc0 = pc0 * wInv;
    outVals.ndc1 = pc1 * wInv;
    outVals.ndc2 = pc2 * wInv;
    outVals.ndc3 = pc3 * wInv;
    outVals.pv0 = pv0;
    outVals.pv1 = pv1;
    outVals.pv2 = pv2;
    outVals.pv3 = pv3;
    outVals.visibleMask = (pv2 >= 0.2f) ? 1.0f : 0.0f;

    return outVals;
}

float2 ndcToScreen(float ndcX, float ndcY, float imageWidth, float imageHeight)
{
    float sx = ((ndcX + 1.0f) * imageWidth - 1.0f) * 0.5f;
    float sy = ((ndcY + 1.0f) * imageHeight - 1.0f) * 0.5f;
    return float2(sx, sy);
}

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

Cov2dValues buildCov2dFromCov3d(
    float m0,
    float m1,
    float m2,
    Cov3dValues cov3d,
    float a00,
    float a01,
    float a02,
    float a10,
    float a11,
    float a12,
    float a20,
    float a21,
    float a22,
    float t30,
    float t31,
    float t32,
    float fovX,
    float fovY,
    float focalX,
    float focalY
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

float computeRadiusFromCov2d(Cov2dValues cov2d)
{
    float detVal = cov2d.c00 * cov2d.c11 - cov2d.c01 * cov2d.c10;
    float mid = 0.5 * (cov2d.c00 + cov2d.c11);
    float delta = max(mid * mid - detVal, 1e-5);
    float lambdaMax = mid + sqrt(delta);
    return 3.0 * ceil(sqrt(lambdaMax));
}
