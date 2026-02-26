StructuredBuffer<float> tileCoord;
StructuredBuffer<float> sortedDepths;
StructuredBuffer<float> sortedMeans2d;
StructuredBuffer<float> sortedConic;
StructuredBuffer<float> sortedOpacity;
StructuredBuffer<float> sortedColor;
StructuredBuffer<float> cotColor;
StructuredBuffer<float> cotDepth;
StructuredBuffer<float> cotAlpha;
StructuredBuffer<uint> counts;

RWStructuredBuffer<float> outColor;
RWStructuredBuffer<float> outDepth;
RWStructuredBuffer<float> outAlpha;

RWStructuredBuffer<float> gradTileCoord;
RWStructuredBuffer<float> gradSortedDepths;
RWStructuredBuffer<float> gradSortedMeans2d;
RWStructuredBuffer<float> gradSortedConic;
RWStructuredBuffer<float> gradSortedOpacity;
RWStructuredBuffer<float> gradSortedColor;

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_tile_forward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pixelCount = counts[0];
    uint gaussianCount = counts[1];
    uint whiteBg = counts[2];

    if (p >= pixelCount)
    {
        return;
    }

    float px = tileCoord[p * 2 + 0];
    float py = tileCoord[p * 2 + 1];

    float T = 1.0;
    float accAlpha = 0.0;
    float accumColorX = 0.0;
    float accumColorY = 0.0;
    float accumColorZ = 0.0;
    float accumDepth = 0.0;

    for (uint i = 0; i < gaussianCount; ++i)
    {
        float dx = px - sortedMeans2d[i * 2 + 0];
        float dy = py - sortedMeans2d[i * 2 + 1];
        uint conicBase = i * 4;
        float exponent = -0.5
            * (dx * dx * sortedConic[conicBase + 0]
                + dy * dy * sortedConic[conicBase + 3]
                + dx * dy * sortedConic[conicBase + 1]
                + dx * dy * sortedConic[conicBase + 2]);
        float alpha = exp(exponent) * sortedOpacity[i];
        if (alpha > 0.99)
        {
            alpha = 0.99;
        }

        float contrib = T * alpha;
        accAlpha += contrib;
        uint colorBase = i * 3;
        accumColorX += contrib * sortedColor[colorBase + 0];
        accumColorY += contrib * sortedColor[colorBase + 1];
        accumColorZ += contrib * sortedColor[colorBase + 2];
        accumDepth += contrib * sortedDepths[i];
        T *= (1.0 - alpha);

        if (T < 1e-4)
        {
            break;
        }
    }

    float bg = (whiteBg != 0) ? (1.0 - accAlpha) : 0.0;
    outColor[p * 3 + 0] = accumColorX + bg;
    outColor[p * 3 + 1] = accumColorY + bg;
    outColor[p * 3 + 2] = accumColorZ + bg;
    outDepth[p] = accumDepth;
    outAlpha[p] = accAlpha;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gaussian_tile_backward(uint3 tid : SV_DispatchThreadID)
{
    uint p = tid.x;
    uint pixelCount = counts[0];
    uint gaussianCount = counts[1];
    uint whiteBg = counts[2];

    if (p >= pixelCount)
    {
        return;
    }

    float px = tileCoord[p * 2 + 0];
    float py = tileCoord[p * 2 + 1];

    float gColorX = cotColor[p * 3 + 0];
    float gColorY = cotColor[p * 3 + 1];
    float gColorZ = cotColor[p * 3 + 2];
    float gDepth = cotDepth[p];
    float gAlpha = cotAlpha[p];
    float bgCoeff = (whiteBg != 0) ? -(gColorX + gColorY + gColorZ) : 0.0;

    float weightedSum = 0.0;
    float trans = 1.0;

    for (uint i = 0; i < gaussianCount; ++i)
    {
        float dx = px - sortedMeans2d[i * 2 + 0];
        float dy = py - sortedMeans2d[i * 2 + 1];
        uint conicBase = i * 4;
        float exponent = -0.5
            * (dx * dx * sortedConic[conicBase + 0]
                + dy * dy * sortedConic[conicBase + 3]
                + dx * dy * sortedConic[conicBase + 1]
                + dx * dy * sortedConic[conicBase + 2]);
        float raw = exp(exponent) * sortedOpacity[i];
        float alpha = raw;
        if (alpha > 0.99)
        {
            alpha = 0.99;
        }

        float contrib = trans * alpha;
        uint colorBase = i * 3;
        float c = (gColorX * sortedColor[colorBase + 0]
                + gColorY * sortedColor[colorBase + 1]
                + gColorZ * sortedColor[colorBase + 2])
            + gDepth * sortedDepths[i] + gAlpha + bgCoeff;
        weightedSum += c * contrib;

        trans *= (1.0 - alpha);
        if (trans < 1e-4)
        {
            break;
        }
    }

    trans = 1.0;
    float prefixWeighted = 0.0;
    for (uint i = 0; i < gaussianCount; ++i)
    {
        float dx = px - sortedMeans2d[i * 2 + 0];
        float dy = py - sortedMeans2d[i * 2 + 1];
        uint conicBase = i * 4;
        float c00 = sortedConic[conicBase + 0];
        float c01 = sortedConic[conicBase + 1];
        float c10 = sortedConic[conicBase + 2];
        float c11 = sortedConic[conicBase + 3];
        float exponent = -0.5
            * (dx * dx * c00
                + dy * dy * c11
                + dx * dy * c01
                + dx * dy * c10);
        float expVal = exp(exponent);
        float raw = expVal * sortedOpacity[i];
        float alpha = raw;
        if (alpha > 0.99)
        {
            alpha = 0.99;
        }
        float contrib = trans * alpha;

        uint colorBase = i * 3;
        float c = (gColorX * sortedColor[colorBase + 0]
                + gColorY * sortedColor[colorBase + 1]
                + gColorZ * sortedColor[colorBase + 2])
            + gDepth * sortedDepths[i] + gAlpha + bgCoeff;

        float suffixWeighted = weightedSum - (prefixWeighted + c * contrib);
        float denom = 1.0 - alpha;
        if (denom < 1e-6)
        {
            denom = 1e-6;
        }
        float gradAlpha = c * trans - suffixWeighted / denom;

        InterlockedAdd(gradSortedDepths[i], gDepth * contrib);
        InterlockedAdd(gradSortedColor[colorBase + 0], gColorX * contrib);
        InterlockedAdd(gradSortedColor[colorBase + 1], gColorY * contrib);
        InterlockedAdd(gradSortedColor[colorBase + 2], gColorZ * contrib);

        if (raw <= 0.99)
        {
            float gradRaw = gradAlpha;
            InterlockedAdd(gradSortedOpacity[i], gradRaw * expVal);

            float gradExponent = gradRaw * raw;
            float dxdy = dx * dy;
            InterlockedAdd(gradSortedConic[conicBase + 0], gradExponent * (-0.5 * dx * dx));
            InterlockedAdd(gradSortedConic[conicBase + 1], gradExponent * (-0.5 * dxdy));
            InterlockedAdd(gradSortedConic[conicBase + 2], gradExponent * (-0.5 * dxdy));
            InterlockedAdd(gradSortedConic[conicBase + 3], gradExponent * (-0.5 * dy * dy));

            float c01sum = c01 + c10;
            float gradDx = gradExponent * (-dx * c00 - 0.5 * dy * c01sum);
            float gradDy = gradExponent * (-dy * c11 - 0.5 * dx * c01sum);
            InterlockedAdd(gradSortedMeans2d[i * 2 + 0], -gradDx);
            InterlockedAdd(gradSortedMeans2d[i * 2 + 1], -gradDy);
        }

        prefixWeighted += c * contrib;
        trans *= (1.0 - alpha);
        if (trans < 1e-4)
        {
            break;
        }
    }
}
