// ssim_kernels.slang
//
// Fused SSIM forward and backward kernels.
// Forward:  5 grouped convolutions + SSIM formula in one pass per pixel.
// Backward: gradient computation via 11×11 neighborhood accumulation.
//
// Image layout: NHWC with N=1, so flat [H*W*C].
// Each channel is processed independently (same as grouped conv with groups=C).
// Gaussian window: generated in-kernel from K and sigma=1.5.

static const uint SSIM_MAX_K = 32;
static const float SSIM_SIGMA = 1.5;

struct SsimAccumState : IDifferentiable
{
    float mu1;
    float mu2;
    float sumW1Sq;
    float sumW2Sq;
    float sumW12;
};

// ---------------------------------------------------------------------------
// ssim_forward
// ---------------------------------------------------------------------------
StructuredBuffer<float> ssf_img1;       // [H*W*C]
StructuredBuffer<float> ssf_img2;       // [H*W*C]
StructuredBuffer<uint>  ssf_params;     // [0]=H, [1]=W, [2]=C, [3]=K

RWStructuredBuffer<float> ssf_outSsim;    // [H*W*C]
RWStructuredBuffer<float> ssf_outMu1;     // [H*W*C]
RWStructuredBuffer<float> ssf_outMu2;     // [H*W*C]
RWStructuredBuffer<float> ssf_outSigma1;  // [H*W*C]
RWStructuredBuffer<float> ssf_outSigma2;  // [H*W*C]
RWStructuredBuffer<float> ssf_outSigma12; // [H*W*C]

[Differentiable]
SsimAccumState updateSsimAccumState(
    SsimAccumState prevState,
    float v1,
    float v2,
    no_diff float wt
)
{
    SsimAccumState nextState;
    nextState.mu1 = prevState.mu1 + wt * v1;
    nextState.mu2 = prevState.mu2 + wt * v2;
    nextState.sumW1Sq = prevState.sumW1Sq + wt * v1 * v1;
    nextState.sumW2Sq = prevState.sumW2Sq + wt * v2 * v2;
    nextState.sumW12 = prevState.sumW12 + wt * v1 * v2;
    return nextState;
}

SsimAccumState undoSsimAccumState(
    SsimAccumState currentState,
    float v1,
    float v2,
    float wt
)
{
    SsimAccumState prevState;
    prevState.mu1 = currentState.mu1 - wt * v1;
    prevState.mu2 = currentState.mu2 - wt * v2;
    prevState.sumW1Sq = currentState.sumW1Sq - wt * v1 * v1;
    prevState.sumW2Sq = currentState.sumW2Sq - wt * v2 * v2;
    prevState.sumW12 = currentState.sumW12 - wt * v1 * v2;
    return prevState;
}

[Differentiable]
float ssimFromAccumState(
    SsimAccumState state,
    no_diff float C1,
    no_diff float C2
)
{
    float mu1 = state.mu1;
    float mu2 = state.mu2;
    float sumW1Sq = state.sumW1Sq;
    float sumW2Sq = state.sumW2Sq;
    float sumW12 = state.sumW12;

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
    if (K == 0 || K > SSIM_MAX_K) return;
    uint pad = K / 2;
    uint totalPixels = H * W * C;

    uint idx = tid.x;
    if (idx >= totalPixels) return;

    uint c = idx % C;
    uint tmp = idx / C;
    uint w_pos = tmp % W;
    uint h_pos = tmp / W;

    SsimAccumState state;
    state.mu1 = 0.0;
    state.mu2 = 0.0;
    state.sumW1Sq = 0.0;
    state.sumW2Sq = 0.0;
    state.sumW12 = 0.0;
    float w1d[SSIM_MAX_K];
    float center = float(K) * 0.5;
    float invTwoSigmaSq = 1.0 / (2.0 * SSIM_SIGMA * SSIM_SIGMA);
    float sum1d = 0.0;
    for (uint i = 0; i < K; ++i)
    {
        float dx = float(i) - center;
        float w = exp(-(dx * dx) * invTwoSigmaSq);
        w1d[i] = w;
        sum1d += w;
    }
    float invSum1d = 1.0 / max(sum1d, 1e-8);
    for (uint i = 0; i < K; ++i)
    {
        w1d[i] *= invSum1d;
    }

    for (uint ki = 0; ki < K; ki++)
    {
        int src_h = int(h_pos) + int(ki) - int(pad);
        if (src_h < 0 || src_h >= int(H)) continue;
        for (uint kj = 0; kj < K; kj++)
        {
            int src_w = int(w_pos) + int(kj) - int(pad);
            if (src_w < 0 || src_w >= int(W)) continue;

            float wt = w1d[ki] * w1d[kj];
            uint srcIdx = uint(src_h) * W * C + uint(src_w) * C + c;
            float v1 = ssf_img1[srcIdx];
            float v2 = ssf_img2[srcIdx];

            state = updateSsimAccumState(state, v1, v2, wt);
        }
    }

    float mu1 = state.mu1;
    float mu2 = state.mu2;
    float sigma1_sq = state.sumW1Sq - mu1 * mu1;
    float sigma2_sq = state.sumW2Sq - mu2 * mu2;
    float sigma12 = state.sumW12 - mu1 * mu2;

    float C1 = 0.0001;   // (0.01 * 1.0)^2
    float C2 = 0.0009;   // (0.03 * 1.0)^2

    float ssim_val = ssimFromAccumState(state, C1, C2);

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
// recover per-pixel gradient through bwd_diff(ssimFromAccumState) and
// bwd_diff(updateSsimAccumState).
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
    if (K == 0 || K > SSIM_MAX_K) return;
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
    float w1d[SSIM_MAX_K];
    float center = float(K) * 0.5;
    float invTwoSigmaSq = 1.0 / (2.0 * SSIM_SIGMA * SSIM_SIGMA);
    float sum1d = 0.0;
    for (uint i = 0; i < K; ++i)
    {
        float dx = float(i) - center;
        float w = exp(-(dx * dx) * invTwoSigmaSq);
        w1d[i] = w;
        sum1d += w;
    }
    float invSum1d = 1.0 / max(sum1d, 1e-8);
    for (uint i = 0; i < K; ++i)
    {
        w1d[i] *= invSum1d;
    }

    // Iterate over all window centers (cx, cy) whose window contains (h_pos, w_pos)
    for (uint ki = 0; ki < K; ki++)
    {
        int cx = int(h_pos) - int(ki) + int(pad);
        if (cx < 0 || cx >= int(H)) continue;
        for (uint kj = 0; kj < K; kj++)
        {
            int cy = int(w_pos) - int(kj) + int(pad);
            if (cy < 0 || cy >= int(W)) continue;

            float wt = w1d[ki] * w1d[kj];
            uint cIdx = uint(cx) * W * C + uint(cy) * C + c;

            float upstream = ssb_gradOutput[cIdx];
            float m1 = ssb_mu1[cIdx];
            float m2 = ssb_mu2[cIdx];
            float s1 = ssb_sigma1[cIdx];
            float s2 = ssb_sigma2[cIdx];
            float s12 = ssb_sigma12[cIdx];

            SsimAccumState currentState;
            currentState.mu1 = m1;
            currentState.mu2 = m2;
            currentState.sumW1Sq = s1 + m1 * m1;
            currentState.sumW2Sq = s2 + m2 * m2;
            currentState.sumW12 = s12 + m1 * m2;

            DifferentialPair<SsimAccumState> dpCurrentState = diffPair(currentState);
            bwd_diff(ssimFromAccumState)(
                dpCurrentState,
                C1,
                C2,
                upstream
            );

            SsimAccumState prevState = undoSsimAccumState(currentState, v1, v2, wt);
            DifferentialPair<SsimAccumState> dpPrevState = diffPair(prevState);

            float dv1 = 0.0;
            float dv2 = 0.0;
            DifferentialPair<float> dpV1 = diffPair(v1, dv1);
            DifferentialPair<float> dpV2 = diffPair(v2, dv2);
            bwd_diff(updateSsimAccumState)(
                dpPrevState,
                dpV1,
                dpV2,
                wt,
                dpCurrentState.d
            );

            grad1 += dpV1.d;
            grad2 += dpV2.d;
        }
    }

    ssb_gradImg1[idx] = grad1;
    ssb_gradImg2[idx] = grad2;
}
