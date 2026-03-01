// gaussian_screen_space_kernels.slang
//
// Active kernels:
// - gaussian_screen_cov3d_backward

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
