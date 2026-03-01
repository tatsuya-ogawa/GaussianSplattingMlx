// ssim_kernels.slang
//
// Fused SSIM forward and backward kernels.
// Forward:  5 grouped convolutions + SSIM formula in one pass per pixel.
// Backward: gradient computation via 11×11 neighborhood accumulation.
//
// Image layout: NHWC with N=1, so flat [H*W*C].
// Each channel is processed independently (same as grouped conv with groups=C).
// Gaussian window: precomputed [K*K] flattened (symmetric).

// ---------------------------------------------------------------------------
// ssim_forward
// ---------------------------------------------------------------------------
StructuredBuffer<float> ssf_img1;       // [H*W*C]
StructuredBuffer<float> ssf_img2;       // [H*W*C]
StructuredBuffer<float> ssf_window;     // [K*K]
StructuredBuffer<uint>  ssf_params;     // [0]=H, [1]=W, [2]=C, [3]=K

RWStructuredBuffer<float> ssf_outSsim;    // [H*W*C]
RWStructuredBuffer<float> ssf_outMu1;     // [H*W*C]
RWStructuredBuffer<float> ssf_outMu2;     // [H*W*C]
RWStructuredBuffer<float> ssf_outSigma1;  // [H*W*C]
RWStructuredBuffer<float> ssf_outSigma2;  // [H*W*C]
RWStructuredBuffer<float> ssf_outSigma12; // [H*W*C]

[Differentiable]
float ssimFromCenterAndTargetPixel(
    float v1,
    float v2,
    no_diff float wt,
    no_diff float baseMu1,
    no_diff float baseMu2,
    no_diff float baseSumW1Sq,
    no_diff float baseSumW2Sq,
    no_diff float baseSumW12,
    no_diff float C1,
    no_diff float C2
)
{
    float mu1 = baseMu1 + wt * v1;
    float mu2 = baseMu2 + wt * v2;
    float sumW1Sq = baseSumW1Sq + wt * v1 * v1;
    float sumW2Sq = baseSumW2Sq + wt * v2 * v2;
    float sumW12 = baseSumW12 + wt * v1 * v2;

    float sigma1Sq = sumW1Sq - mu1 * mu1;
    float sigma2Sq = sumW2Sq - mu2 * mu2;
    float sigma12 = sumW12 - mu1 * mu2;

    float a = 2.0 * mu1 * mu2 + C1;
    float b = 2.0 * sigma12 + C2;
    float c_ = mu1 * mu1 + mu2 * mu2 + C1;
    float d = sigma1Sq + sigma2Sq + C2;
    return (a * b) / (c_ * d);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ssim_forward(uint3 tid : SV_DispatchThreadID)
{
    uint H = ssf_params[0];
    uint W = ssf_params[1];
    uint C = ssf_params[2];
    uint K = ssf_params[3];
    uint pad = K / 2;
    uint totalPixels = H * W * C;

    uint idx = tid.x;
    if (idx >= totalPixels) return;

    uint c = idx % C;
    uint tmp = idx / C;
    uint w_pos = tmp % W;
    uint h_pos = tmp / W;

    float sum_w1 = 0.0;
    float sum_w2 = 0.0;
    float sum_w1sq = 0.0;
    float sum_w2sq = 0.0;
    float sum_w12 = 0.0;

    for (uint ki = 0; ki < K; ki++)
    {
        int src_h = int(h_pos) + int(ki) - int(pad);
        if (src_h < 0 || src_h >= int(H)) continue;
        for (uint kj = 0; kj < K; kj++)
        {
            int src_w = int(w_pos) + int(kj) - int(pad);
            if (src_w < 0 || src_w >= int(W)) continue;

            float wt = ssf_window[ki * K + kj];
            uint srcIdx = uint(src_h) * W * C + uint(src_w) * C + c;
            float v1 = ssf_img1[srcIdx];
            float v2 = ssf_img2[srcIdx];

            sum_w1 += wt * v1;
            sum_w2 += wt * v2;
            sum_w1sq += wt * v1 * v1;
            sum_w2sq += wt * v2 * v2;
            sum_w12 += wt * v1 * v2;
        }
    }

    float mu1 = sum_w1;
    float mu2 = sum_w2;
    float sigma1_sq = sum_w1sq - mu1 * mu1;
    float sigma2_sq = sum_w2sq - mu2 * mu2;
    float sigma12 = sum_w12 - mu1 * mu2;

    float C1 = 0.0001;   // (0.01 * 1.0)^2
    float C2 = 0.0009;   // (0.03 * 1.0)^2

    float a = 2.0 * mu1 * mu2 + C1;
    float b = 2.0 * sigma12 + C2;
    float c_ = mu1 * mu1 + mu2 * mu2 + C1;
    float d = sigma1_sq + sigma2_sq + C2;
    float den = c_ * d;

    float ssim_val = (a * b) / den;

    ssf_outSsim[idx] = ssim_val;
    ssf_outMu1[idx] = mu1;
    ssf_outMu2[idx] = mu2;
    ssf_outSigma1[idx] = sigma1_sq;
    ssf_outSigma2[idx] = sigma2_sq;
    ssf_outSigma12[idx] = sigma12;
}

// ---------------------------------------------------------------------------
// ssim_backward
// ---------------------------------------------------------------------------
// For each output pixel (h, w, c), iterate over all 11×11 windows
// that contain this pixel. At each window center, use saved stats and
// recover per-pixel gradient through bwd_diff(ssimFromCenterAndTargetPixel).
//
// (Similarly for grad_img2)

StructuredBuffer<float> ssb_gradOutput; // [H*W*C]
StructuredBuffer<float> ssb_img1;       // [H*W*C]
StructuredBuffer<float> ssb_img2;       // [H*W*C]
StructuredBuffer<float> ssb_mu1;        // [H*W*C]
StructuredBuffer<float> ssb_mu2;        // [H*W*C]
StructuredBuffer<float> ssb_sigma1;     // [H*W*C]
StructuredBuffer<float> ssb_sigma2;     // [H*W*C]
StructuredBuffer<float> ssb_sigma12;    // [H*W*C]
StructuredBuffer<float> ssb_window;     // [K*K]
StructuredBuffer<uint>  ssb_params;     // [0]=H, [1]=W, [2]=C, [3]=K

RWStructuredBuffer<float> ssb_gradImg1; // [H*W*C]
RWStructuredBuffer<float> ssb_gradImg2; // [H*W*C]

[shader("compute")]
[numthreads(256, 1, 1)]
void ssim_backward(uint3 tid : SV_DispatchThreadID)
{
    uint H = ssb_params[0];
    uint W = ssb_params[1];
    uint C = ssb_params[2];
    uint K = ssb_params[3];
    uint pad = K / 2;
    uint totalPixels = H * W * C;

    uint idx = tid.x;
    if (idx >= totalPixels) return;

    uint c = idx % C;
    uint tmp = idx / C;
    uint w_pos = tmp % W;
    uint h_pos = tmp / W;

    float C1 = 0.0001;
    float C2 = 0.0009;

    float v1 = ssb_img1[idx];
    float v2 = ssb_img2[idx];

    float grad1 = 0.0;
    float grad2 = 0.0;

    // Iterate over all window centers (cx, cy) whose window contains (h_pos, w_pos)
    for (uint ki = 0; ki < K; ki++)
    {
        int cx = int(h_pos) - int(ki) + int(pad);
        if (cx < 0 || cx >= int(H)) continue;
        for (uint kj = 0; kj < K; kj++)
        {
            int cy = int(w_pos) - int(kj) + int(pad);
            if (cy < 0 || cy >= int(W)) continue;

            float wt = ssb_window[ki * K + kj];
            uint cIdx = uint(cx) * W * C + uint(cy) * C + c;

            float upstream = ssb_gradOutput[cIdx];
            float m1 = ssb_mu1[cIdx];
            float m2 = ssb_mu2[cIdx];
            float s1 = ssb_sigma1[cIdx];
            float s2 = ssb_sigma2[cIdx];
            float s12 = ssb_sigma12[cIdx];

            // Recover fixed "all other pixels" contributions for this center.
            float sumW1Sq = s1 + m1 * m1;
            float sumW2Sq = s2 + m2 * m2;
            float sumW12 = s12 + m1 * m2;
            float baseMu1 = m1 - wt * v1;
            float baseMu2 = m2 - wt * v2;
            float baseSumW1Sq = sumW1Sq - wt * v1 * v1;
            float baseSumW2Sq = sumW2Sq - wt * v2 * v2;
            float baseSumW12 = sumW12 - wt * v1 * v2;

            float dv1 = 0.0;
            float dv2 = 0.0;
            DifferentialPair<float> dpV1 = diffPair(v1, dv1);
            DifferentialPair<float> dpV2 = diffPair(v2, dv2);
            bwd_diff(ssimFromCenterAndTargetPixel)(
                dpV1,
                dpV2,
                wt,
                baseMu1,
                baseMu2,
                baseSumW1Sq,
                baseSumW2Sq,
                baseSumW12,
                C1,
                C2,
                upstream
            );

            grad1 += dpV1.d;
            grad2 += dpV2.d;
        }
    }

    ssb_gradImg1[idx] = grad1;
    ssb_gradImg2[idx] = grad2;
}
