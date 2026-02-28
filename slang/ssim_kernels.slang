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
// that contain this pixel. At each window center, use the saved stats
// to compute SSIM partial derivatives, then accumulate the gradient
// contribution weighted by the Gaussian kernel.
//
// grad_img1(px,py,c) = sum over (di,dj) in [-pad,pad]:
//   let (cx,cy) = (px+di, py+dj)   // window center
//   w(di+pad, dj+pad) * upstream(cx,cy,c) * [
//     dS/dmu1 * 1
//     + dS/dsigma1_sq * 2*(img1(px,py,c) - mu1(cx,cy,c))
//     + dS/dsigma12   * (img2(px,py,c) - mu2(cx,cy,c))
//   ]
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

            // SSIM formula terms at window center (cx, cy)
            float a = 2.0 * m1 * m2 + C1;
            float b = 2.0 * s12 + C2;
            float c_ = m1 * m1 + m2 * m2 + C1;
            float d = s1 + s2 + C2;
            float den = c_ * d;
            float den2 = den * den;
            float num = a * b;

            // Partial derivatives of ssim w.r.t. statistics
            // d(ssim)/d(mu1) = (d(num)/d(mu1)*den - num*d(den)/d(mu1)) / den^2
            float dS_dmu1 = (2.0 * m2 * b * den - num * 2.0 * m1 * d) / den2;
            float dS_dmu2 = (2.0 * m1 * b * den - num * 2.0 * m2 * d) / den2;
            float dS_ds1 = -num * c_ / den2;  // d(den)/d(sigma1_sq) = c_
            float dS_ds2 = -num * c_ / den2;  // d(den)/d(sigma2_sq) = c_
            float dS_ds12 = 2.0 * a / den;    // d(num)/d(sigma12) = 2*a

            // Derivatives of statistics w.r.t. img1 at pixel (h_pos, w_pos)
            // mu1 = sum w*v1 → d/dv1 = w
            // sigma1_sq = sum w*v1^2 - mu1^2 → d/dv1 = 2*w*(v1 - mu1)
            // sigma12 = sum w*v1*v2 - mu1*mu2 → d/dv1 = w*(v2 - mu2)
            float dmu1_dv1 = wt;
            float ds1_dv1 = 2.0 * wt * (v1 - m1);
            float ds12_dv1 = wt * (v2 - m2);

            grad1 += upstream * (dS_dmu1 * dmu1_dv1 + dS_ds1 * ds1_dv1 + dS_ds12 * ds12_dv1);

            // Derivatives w.r.t. img2
            // mu2 = sum w*v2 → d/dv2 = w
            // sigma2_sq = sum w*v2^2 - mu2^2 → d/dv2 = 2*w*(v2 - mu2)
            // sigma12 = sum w*v1*v2 - mu1*mu2 → d/dv2 = w*(v1 - mu1)
            float dmu2_dv2 = wt;
            float ds2_dv2 = 2.0 * wt * (v2 - m2);
            float ds12_dv2 = wt * (v1 - m1);

            grad2 += upstream * (dS_dmu2 * dmu2_dv2 + dS_ds2 * ds2_dv2 + dS_ds12 * ds12_dv2);
        }
    }

    ssb_gradImg1[idx] = grad1;
    ssb_gradImg2[idx] = grad2;
}
