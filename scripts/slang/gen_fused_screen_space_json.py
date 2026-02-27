#!/usr/bin/env python3
"""Generate fused screen-space kernel JSON specs for MLXFast.

Fuses the 4 screen-space kernel pairs (cov3d, color, cov2d, inverse2d) into
a single forward kernel and a single backward kernel.

Performance gains:
  - 8 kernel dispatches -> 2
  - cov3d stays in registers (eliminates N*9*4 bytes intermediate buffer)
  - cov2d recomputed in backward from registers (eliminates recompute dispatch)
"""

import json
import pathlib

HEADER = (
    "#include <metal_stdlib>\n"
    "#include <metal_math>\n"
    "#include <metal_texture>\n"
    "using namespace metal;"
)

# ─────────────────── shared code fragments ───────────────────

COV3D_FROM_SCALES_ROTATIONS = r"""
// ---- cov3d from scales/rotations ----
uint scaleBase_0 = p_0 * 3U;
float sx_0 = fused_scales_1[scaleBase_0];
float sy_0 = fused_scales_1[scaleBase_0 + 1U];
float sz_0 = fused_scales_1[scaleBase_0 + 2U];
uint rotBase_0 = p_0 * 4U;
float rw_0 = fused_rotations_1[rotBase_0];
float rx_0 = fused_rotations_1[rotBase_0 + 1U];
float ry_0 = fused_rotations_1[rotBase_0 + 2U];
float rz_0 = fused_rotations_1[rotBase_0 + 3U];
float norm_0 = sqrt(rw_0*rw_0 + rx_0*rx_0 + ry_0*ry_0 + rz_0*rz_0);
float _sn = max(norm_0, 9.99999993922529029e-09);
float qw = rw_0/_sn; float qx = rx_0/_sn; float qy = ry_0/_sn; float qz = rz_0/_sn;
float qyy=qy*qy, qzz=qz*qz, qxy=qx*qy, qwz=qw*qz;
float qxz=qx*qz, qwy=qw*qy, qxx=qx*qx, qyz=qy*qz, qwx=qw*qx;
float r00=1.0-2.0*(qyy+qzz); float r01=2.0*(qxy-qwz); float r02=2.0*(qxz+qwy);
float r10=2.0*(qxy+qwz); float r11=1.0-2.0*(qxx+qzz); float r12=2.0*(qyz-qwx);
float r20=2.0*(qxz-qwy); float r21=2.0*(qyz+qwx); float r22=1.0-2.0*(qxx+qyy);
float l00=r00*sx_0, l01=r01*sy_0, l02=r02*sz_0;
float l10=r10*sx_0, l11=r11*sy_0, l12=r12*sz_0;
float l20=r20*sx_0, l21=r21*sy_0, l22=r22*sz_0;
float cv3d00=l00*l00+l01*l01+l02*l02;
float cv3d01=l00*l10+l01*l11+l02*l12;
float cv3d02=l00*l20+l01*l21+l02*l22;
float cv3d10=l10*l00+l11*l01+l12*l02;
float cv3d11=l10*l10+l11*l11+l12*l12;
float cv3d12=l10*l20+l11*l21+l12*l22;
float cv3d20=l20*l00+l21*l01+l22*l02;
float cv3d21=l20*l10+l21*l11+l22*l12;
float cv3d22=l20*l20+l21*l21+l22*l22;
"""

SH_BASIS_FORWARD = r"""
// ---- SH basis ----
uint meansBase_0 = p_0 * 3U;
int degree_0 = int(fused_counts_1[int(1)]);
int _S1 = degree_0 + int(1);
int _S2 = _S1 * _S1;
int coeffStride_0 = int(fused_counts_1[int(2)]);
float x_0 = fused_means3d_1[meansBase_0]     - fused_cameraCenter_1[int(0)];
float y_0 = fused_means3d_1[meansBase_0+1U]  - fused_cameraCenter_1[int(1)];
float z_0 = fused_means3d_1[meansBase_0+2U]  - fused_cameraCenter_1[int(2)];
float xx_0=x_0*x_0, yy_0=y_0*y_0, zz_0=z_0*z_0;
float xy_0=x_0*y_0, yz_0=y_0*z_0, xz_0=x_0*z_0;
thread array<float, int(25)> basis_0;
int i_0 = int(0);
for(;;){ if(i_0<int(25)){} else { break; } basis_0[i_0]=0.0; i_0=i_0+int(1); }
basis_0[int(0)] = 0.282094806432724;
if(degree_0 > int(0)){
    basis_0[int(1)] = -0.48860251903533936 * y_0;
    basis_0[int(2)] = 0.48860251903533936 * z_0;
    basis_0[int(3)] = -0.48860251903533936 * x_0;
}
if(degree_0 > int(1)){
    basis_0[int(4)] = 1.09254848957061768 * xy_0;
    basis_0[int(5)] = -1.09254848957061768 * yz_0;
    basis_0[int(6)] = 0.31539157032966614 * (2.0*zz_0-xx_0-yy_0);
    basis_0[int(7)] = -1.09254848957061768 * xz_0;
    basis_0[int(8)] = 0.54627424478530884 * (xx_0-yy_0);
}
if(degree_0 > int(2)){
    float _S3=3.0*xx_0;
    basis_0[int(9)]  = -0.59004360437393188*y_0*(_S3-yy_0);
    basis_0[int(10)] = 2.89061141014099121*xy_0*z_0;
    float _S4=4.0*zz_0-xx_0-yy_0;
    basis_0[int(11)] = -0.4570457935333252*y_0*_S4;
    float _S5=3.0*yy_0;
    basis_0[int(12)] = 0.37317633628845215*z_0*(2.0*zz_0-_S3-_S5);
    basis_0[int(13)] = -0.4570457935333252*x_0*_S4;
    basis_0[int(14)] = 1.44530570507049561*z_0*(xx_0-yy_0);
    basis_0[int(15)] = -0.59004360437393188*x_0*(xx_0-_S5);
}
if(degree_0 > int(3)){
    float _S6=xx_0-yy_0;
    basis_0[int(16)] = 2.50334286689758301*xy_0*_S6;
    float _S7=3.0*xx_0-yy_0;
    basis_0[int(17)] = -1.77013075351715088*yz_0*_S7;
    float _S8=7.0*zz_0; float _S9=_S8-1.0;
    basis_0[int(18)] = 0.94617468118667603*xy_0*_S9;
    float _S10=_S8-3.0;
    basis_0[int(19)] = -0.66904652118682861*yz_0*_S10;
    basis_0[int(20)] = 0.1057855486869812*(zz_0*(35.0*zz_0-30.0)+3.0);
    basis_0[int(21)] = -0.66904652118682861*xz_0*_S10;
    basis_0[int(22)] = 0.47308734059333801*_S6*_S9;
    float _S11=xx_0-3.0*yy_0;
    basis_0[int(23)] = -1.77013075351715088*xz_0*_S11;
    basis_0[int(24)] = 0.62583571672439575*(xx_0*_S11-yy_0*_S7);
}
"""

COLOR_EVAL_FORWARD = r"""
// ---- evaluate color from SH ----
uint _S12 = p_0 * uint(coeffStride_0 * int(3));
int ch_0 = int(0);
for(;;){
    if(ch_0 < int(3)){} else { break; }
    int k_0 = int(0); float value_0 = 0.0;
    for(;;){
        if(k_0 < _S2){} else { break; }
        float value_1 = value_0 + basis_0[k_0] * fused_shs_1[_S12 + uint(k_0*int(3)+ch_0)];
        k_0 = k_0 + int(1); value_0 = value_1;
    }
    float color_0 = value_0 + 0.5;
    float device* _S13 = fused_outColor_1 + (meansBase_0 + uint(ch_0));
    *_S13 = (color_0 < 0.0) ? 0.0 : color_0;
    ch_0 = ch_0 + int(1);
}
"""

COV2D_FROM_COV3D = r"""
// ---- cov2d from cov3d (register) ----
float m0_0 = fused_means3d_1[meansBase_0];
float m1_0 = fused_means3d_1[meansBase_0 + 1U];
float m2_0 = fused_means3d_1[meansBase_0 + 2U];
float a00_0 = fused_viewMatrix_1[int(0)];
float a01_0 = fused_viewMatrix_1[int(1)];
float a02_0 = fused_viewMatrix_1[int(2)];
float a10_0 = fused_viewMatrix_1[int(4)];
float a11_0 = fused_viewMatrix_1[int(5)];
float a12_0 = fused_viewMatrix_1[int(6)];
float a20_0 = fused_viewMatrix_1[int(8)];
float a21_0 = fused_viewMatrix_1[int(9)];
float a22_0 = fused_viewMatrix_1[int(10)];
float t2_0 = m0_0*a02_0 + m1_0*a12_0 + m2_0*a22_0 + fused_viewMatrix_1[int(14)];
float tanFovX_0 = tan(fused_fovX_1[int(0)]*0.5);
float tanFovY_0 = tan(fused_fovY_1[int(0)]*0.5);
float j00_0 = fused_focalX_1[int(0)] / t2_0;
float _tz2 = t2_0 * t2_0;
float j02_0 = -((m0_0*a00_0+m1_0*a10_0+m2_0*a20_0+fused_viewMatrix_1[int(12)]) / clamp(t2_0, -tanFovX_0*1.29999995231628418, tanFovX_0*1.29999995231628418) * t2_0) * fused_focalX_1[int(0)] / _tz2;
float j11_0 = fused_focalY_1[int(0)] / t2_0;
float j12_0 = -((m0_0*a01_0+m1_0*a11_0+m2_0*a21_0+fused_viewMatrix_1[int(13)]) / clamp(t2_0, -tanFovY_0*1.29999995231628418, tanFovY_0*1.29999995231628418) * t2_0) * fused_focalY_1[int(0)] / _tz2;
float b00_0=j00_0*a00_0+j02_0*a02_0;
float b01_0=j00_0*a10_0+j02_0*a12_0;
float b02_0=j00_0*a20_0+j02_0*a22_0;
float b10_0=j11_0*a01_0+j12_0*a02_0;
float b11_0=j11_0*a11_0+j12_0*a12_0;
float b12_0=j11_0*a21_0+j12_0*a22_0;
float tt00=b00_0*cv3d00+b01_0*cv3d10+b02_0*cv3d20;
float tt01=b00_0*cv3d01+b01_0*cv3d11+b02_0*cv3d21;
float tt02=b00_0*cv3d02+b01_0*cv3d12+b02_0*cv3d22;
float tt10=b10_0*cv3d00+b11_0*cv3d10+b12_0*cv3d20;
float tt11=b10_0*cv3d01+b11_0*cv3d11+b12_0*cv3d21;
float tt12=b10_0*cv3d02+b11_0*cv3d12+b12_0*cv3d22;
uint outBase_0 = p_0 * 4U;
float cv2d00=tt00*b00_0+tt01*b01_0+tt02*b02_0+0.30000001192092896;
float cv2d01=tt00*b10_0+tt01*b11_0+tt02*b12_0;
float cv2d10=tt10*b00_0+tt11*b01_0+tt12*b02_0;
float cv2d11=tt10*b10_0+tt11*b11_0+tt12*b12_0+0.30000001192092896;
*(fused_outCov2d_1+outBase_0) = cv2d00;
*(fused_outCov2d_1+(outBase_0+1U)) = cv2d01;
*(fused_outCov2d_1+(outBase_0+2U)) = cv2d10;
*(fused_outCov2d_1+(outBase_0+3U)) = cv2d11;
"""

INVERSE_FORWARD = r"""
// ---- conic = inv(cov2d) ----
float det_0 = cv2d00*cv2d11 - cv2d01*cv2d10;
*(fused_outConic_1+outBase_0)        = cv2d11  / det_0;
*(fused_outConic_1+(outBase_0+1U))   = -cv2d01 / det_0;
*(fused_outConic_1+(outBase_0+2U))   = -cv2d10 / det_0;
*(fused_outConic_1+(outBase_0+3U))   = cv2d00  / det_0;
return;
"""

# ─────────────── backward-only fragments ───────────────

SH_BASIS_BACKWARD = r"""
// ---- SH basis + derivatives for backward ----
uint meansBase_0 = p_0 * 3U;
int degree_0 = int(fused_counts_1[int(1)]);
int _S1 = degree_0 + int(1);
int _S2 = _S1 * _S1;
int coeffStride_0 = int(fused_counts_1[int(2)]);
float x_0 = fused_means3d_1[meansBase_0]     - fused_cameraCenter_1[int(0)];
float y_0 = fused_means3d_1[meansBase_0+1U]  - fused_cameraCenter_1[int(1)];
float z_0 = fused_means3d_1[meansBase_0+2U]  - fused_cameraCenter_1[int(2)];
float xx_0=x_0*x_0, yy_0=y_0*y_0, zz_0=z_0*z_0;
float xy_0=x_0*y_0, yz_0=y_0*z_0, xz_0=x_0*z_0;
thread array<float, int(25)> basis_0;
thread array<float, int(25)> dbx_0;
thread array<float, int(25)> dby_0;
thread array<float, int(25)> dbz_0;
int i_0 = int(0);
for(;;){ if(i_0<int(25)){} else { break; }
  basis_0[i_0]=0.0; dbx_0[i_0]=0.0; dby_0[i_0]=0.0; dbz_0[i_0]=0.0;
  i_0=i_0+int(1); }
basis_0[int(0)] = 0.282094806432724;
if(degree_0 > int(0)){
    basis_0[int(1)] = -0.48860251903533936*y_0;
    basis_0[int(2)] = 0.48860251903533936*z_0;
    basis_0[int(3)] = -0.48860251903533936*x_0;
    dby_0[int(1)] = -0.48860251903533936;
    dbz_0[int(2)] = 0.48860251903533936;
    dbx_0[int(3)] = -0.48860251903533936;
}
if(degree_0 > int(1)){
    basis_0[int(4)] = 1.09254848957061768*xy_0;
    basis_0[int(5)] = -1.09254848957061768*yz_0;
    basis_0[int(6)] = 0.31539157032966614*(2.0*zz_0-xx_0-yy_0);
    basis_0[int(7)] = -1.09254848957061768*xz_0;
    basis_0[int(8)] = 0.54627424478530884*(xx_0-yy_0);
    dbx_0[int(4)] = 1.09254848957061768*y_0;
    dby_0[int(4)] = 1.09254848957061768*x_0;
    float _db5 = -1.09254848957061768*z_0;
    dby_0[int(5)] = _db5;
    dbz_0[int(5)] = -1.09254848957061768*y_0;
    dbx_0[int(6)] = 0.31539157032966614*(-2.0*x_0);
    float _db6 = -2.0*y_0;
    dby_0[int(6)] = 0.31539157032966614*_db6;
    dbz_0[int(6)] = 0.31539157032966614*(4.0*z_0);
    dbx_0[int(7)] = _db5;
    dbz_0[int(7)] = -1.09254848957061768*x_0;
    dbx_0[int(8)] = 0.54627424478530884*(2.0*x_0);
    dby_0[int(8)] = 0.54627424478530884*_db6;
}
if(degree_0 > int(2)){
    float _d7=3.0*xx_0;
    basis_0[int(9)]=-0.59004360437393188*y_0*(_d7-yy_0);
    float _d8=2.89061141014099121*xy_0;
    basis_0[int(10)]=_d8*z_0;
    float _d9=4.0*zz_0; float _d10=_d9-xx_0; float _d11=_d10-yy_0;
    basis_0[int(11)]=-0.4570457935333252*y_0*_d11;
    float _d12=3.0*yy_0;
    basis_0[int(12)]=0.37317633628845215*z_0*(2.0*zz_0-_d7-_d12);
    basis_0[int(13)]=-0.4570457935333252*x_0*_d11;
    float _d13=xx_0-yy_0;
    basis_0[int(14)]=1.44530570507049561*z_0*_d13;
    basis_0[int(15)]=-0.59004360437393188*x_0*(xx_0-_d12);
    dbx_0[int(9)]=-0.59004360437393188*(6.0*x_0*y_0);
    float _d14=-0.59004360437393188*(_d7-_d12);
    dby_0[int(9)]=_d14;
    dbx_0[int(10)]=2.89061141014099121*yz_0;
    dby_0[int(10)]=2.89061141014099121*xz_0;
    dbz_0[int(10)]=_d8;
    float _d15=-0.4570457935333252*(-2.0*x_0*y_0);
    dbx_0[int(11)]=_d15;
    dby_0[int(11)]=-0.4570457935333252*(_d10-_d12);
    dbz_0[int(11)]=-0.4570457935333252*(8.0*y_0*z_0);
    float _d16=-6.0*x_0;
    dbx_0[int(12)]=0.37317633628845215*(_d16*z_0);
    dby_0[int(12)]=0.37317633628845215*(-6.0*y_0*z_0);
    dbz_0[int(12)]=0.37317633628845215*(6.0*zz_0-_d7-_d12);
    dbx_0[int(13)]=-0.4570457935333252*(_d9-_d7-yy_0);
    dby_0[int(13)]=_d15;
    dbz_0[int(13)]=-0.4570457935333252*(8.0*x_0*z_0);
    dbx_0[int(14)]=1.44530570507049561*(2.0*x_0*z_0);
    dby_0[int(14)]=1.44530570507049561*(-2.0*y_0*z_0);
    dbz_0[int(14)]=1.44530570507049561*_d13;
    dbx_0[int(15)]=_d14;
    dby_0[int(15)]=-0.59004360437393188*(_d16*y_0);
}
if(degree_0 > int(3)){
    float _d17=xx_0-yy_0;
    basis_0[int(16)]=2.50334286689758301*xy_0*_d17;
    float _d18=3.0*xx_0; float _d19=_d18-yy_0;
    basis_0[int(17)]=-1.77013075351715088*yz_0*_d19;
    float _d20=7.0*zz_0; float _d21=_d20-1.0;
    basis_0[int(18)]=0.94617468118667603*xy_0*_d21;
    float _d22=_d20-3.0;
    basis_0[int(19)]=-0.66904652118682861*yz_0*_d22;
    basis_0[int(20)]=0.1057855486869812*(zz_0*(35.0*zz_0-30.0)+3.0);
    basis_0[int(21)]=-0.66904652118682861*xz_0*_d22;
    basis_0[int(22)]=0.47308734059333801*_d17*_d21;
    float _d23=xx_0-3.0*yy_0;
    basis_0[int(23)]=-1.77013075351715088*xz_0*_d23;
    basis_0[int(24)]=0.62583571672439575*(xx_0*_d23-yy_0*_d19);
    dbx_0[int(16)]=2.50334286689758301*y_0*_d19;
    dby_0[int(16)]=2.50334286689758301*x_0*_d23;
    dbx_0[int(17)]=-1.77013075351715088*(6.0*x_0*y_0*z_0);
    float _d24=-1.77013075351715088*(3.0*z_0*_d17);
    dby_0[int(17)]=_d24;
    dbz_0[int(17)]=-1.77013075351715088*y_0*_d19;
    dbx_0[int(18)]=0.94617468118667603*y_0*_d21;
    dby_0[int(18)]=0.94617468118667603*x_0*_d21;
    dbz_0[int(18)]=0.94617468118667603*(14.0*x_0*y_0*z_0);
    float _d25=-0.66904652118682861*z_0*_d22;
    dby_0[int(19)]=_d25;
    float _d26=21.0*zz_0-3.0;
    dbz_0[int(19)]=-0.66904652118682861*y_0*_d26;
    dbz_0[int(20)]=0.1057855486869812*(140.0*z_0*z_0*z_0-60.0*z_0);
    dbx_0[int(21)]=_d25;
    dbz_0[int(21)]=-0.66904652118682861*x_0*_d26;
    dbx_0[int(22)]=0.47308734059333801*(2.0*x_0)*_d21;
    dby_0[int(22)]=0.47308734059333801*(-2.0*y_0)*_d21;
    dbz_0[int(22)]=0.47308734059333801*(14.0*z_0*_d17);
    dbx_0[int(23)]=_d24;
    dby_0[int(23)]=-1.77013075351715088*(-6.0*x_0*y_0*z_0);
    dbz_0[int(23)]=-1.77013075351715088*x_0*_d23;
    dbx_0[int(24)]=0.62583571672439575*(4.0*x_0*_d23);
    dby_0[int(24)]=0.62583571672439575*(4.0*y_0*(yy_0-_d18));
}
"""

COLOR_BACKWARD = r"""
// ---- color backward: gradMeans3d_color + gradShs ----
uint _S27 = p_0 * uint(coeffStride_0 * int(3));
int ch_0 = int(0);
float gradX_0 = 0.0; float gradY_0 = 0.0; float gradZ_0 = 0.0;
for(;;){
    if(ch_0 < int(3)){} else { break; }
    int k_0 = int(0); float value_0 = 0.0;
    for(;;){
        if(k_0 < _S2){} else { break; }
        float value_1 = value_0 + basis_0[k_0] * fused_shs_1[_S27 + uint(k_0*int(3)+ch_0)];
        k_0 = k_0 + int(1); value_0 = value_1;
    }
    float _S28 = fused_cotColor_1[meansBase_0 + uint(ch_0)];
    float g_0;
    if((value_0 + 0.5) <= 0.0){ g_0 = 0.0; } else { g_0 = _S28; }
    int k_1 = int(0);
    for(;;){
        if(k_1 < _S2){} else { break; }
        uint shIndex_0 = _S27 + uint(k_1*int(3)+ch_0);
        *(fused_gradShs_1 + shIndex_0) = g_0 * basis_0[k_1];
        float _S29 = g_0 * fused_shs_1[shIndex_0];
        float gradX_1 = gradX_0 + _S29 * dbx_0[k_1];
        float gradY_1 = gradY_0 + _S29 * dby_0[k_1];
        float gradZ_1 = gradZ_0 + _S29 * dbz_0[k_1];
        k_1 = k_1 + int(1);
        gradX_0 = gradX_1; gradY_0 = gradY_1; gradZ_0 = gradZ_1;
    }
    ch_0 = ch_0 + int(1);
}
"""

COV2D_RECOMPUTE_FOR_BACKWARD = r"""
// ---- recompute cov2d (for inverse backward) ----
float m0_0 = fused_means3d_1[meansBase_0];
float m1_0 = fused_means3d_1[meansBase_0 + 1U];
float m2_0 = fused_means3d_1[meansBase_0 + 2U];
float a00_0=fused_viewMatrix_1[int(0)];
float a01_0=fused_viewMatrix_1[int(1)];
float a02_0=fused_viewMatrix_1[int(2)];
float a10_0=fused_viewMatrix_1[int(4)];
float a11_0=fused_viewMatrix_1[int(5)];
float a12_0=fused_viewMatrix_1[int(6)];
float a20_0=fused_viewMatrix_1[int(8)];
float a21_0=fused_viewMatrix_1[int(9)];
float a22_0=fused_viewMatrix_1[int(10)];
float t0_0 = m0_0*a00_0+m1_0*a10_0+m2_0*a20_0+fused_viewMatrix_1[int(12)];
float t1_0 = m0_0*a01_0+m1_0*a11_0+m2_0*a21_0+fused_viewMatrix_1[int(13)];
float t2_0 = m0_0*a02_0+m1_0*a12_0+m2_0*a22_0+fused_viewMatrix_1[int(14)];
float tanFovX_0 = tan(fused_fovX_1[int(0)]*0.5);
float tanFovY_0 = tan(fused_fovY_1[int(0)]*0.5);
float minX_0 = -tanFovX_0*1.29999995231628418;
float maxX_0 =  tanFovX_0*1.29999995231628418;
float minY_0 = -tanFovY_0*1.29999995231628418;
float maxY_0 =  tanFovY_0*1.29999995231628418;
float clipX_0 = clamp(t2_0, minX_0, maxX_0);
float clipY_0 = clamp(t2_0, minY_0, maxY_0);
float tx_0 = t0_0 / clipX_0 * t2_0;
float ty_0 = t1_0 / clipY_0 * t2_0;
float j00_0 = fused_focalX_1[int(0)] / t2_0;
float _tz2 = t2_0*t2_0;
float j02_0 = -tx_0*fused_focalX_1[int(0)]/_tz2;
float j11_0 = fused_focalY_1[int(0)] / t2_0;
float j12_0 = -ty_0*fused_focalY_1[int(0)]/_tz2;
float b00_0=j00_0*a00_0+j02_0*a02_0;
float b01_0=j00_0*a10_0+j02_0*a12_0;
float b02_0=j00_0*a20_0+j02_0*a22_0;
float b10_0=j11_0*a01_0+j12_0*a02_0;
float b11_0=j11_0*a11_0+j12_0*a12_0;
float b12_0=j11_0*a21_0+j12_0*a22_0;
float tt00=b00_0*cv3d00+b01_0*cv3d10+b02_0*cv3d20;
float tt01=b00_0*cv3d01+b01_0*cv3d11+b02_0*cv3d21;
float tt02=b00_0*cv3d02+b01_0*cv3d12+b02_0*cv3d22;
float tt10=b10_0*cv3d00+b11_0*cv3d10+b12_0*cv3d20;
float tt11=b10_0*cv3d01+b11_0*cv3d11+b12_0*cv3d21;
float tt12=b10_0*cv3d02+b11_0*cv3d12+b12_0*cv3d22;
uint cv2dBase = p_0*4U;
float cv2d00=tt00*b00_0+tt01*b01_0+tt02*b02_0+0.30000001192092896;
float cv2d01=tt00*b10_0+tt01*b11_0+tt02*b12_0;
float cv2d10=tt10*b00_0+tt11*b01_0+tt12*b02_0;
float cv2d11=tt10*b10_0+tt11*b11_0+tt12*b12_0+0.30000001192092896;
"""

INVERSE_BACKWARD = r"""
// ---- inverse backward: cotConic -> gradCov2d ----
float det_0 = cv2d00*cv2d11 - cv2d01*cv2d10;
float inv00=cv2d11/det_0, inv01=-cv2d01/det_0;
float inv10=-cv2d10/det_0, inv11=cv2d00/det_0;
float gc00=fused_cotConic_1[cv2dBase];
float gc01=fused_cotConic_1[cv2dBase+1U];
float gc10=fused_cotConic_1[cv2dBase+2U];
float gc11=fused_cotConic_1[cv2dBase+3U];
float ig00=inv00*gc00+inv10*gc10; float ig01=inv00*gc01+inv10*gc11;
float ig10=inv01*gc00+inv11*gc10; float ig11=inv01*gc01+inv11*gc11;
float invGrad00=-(ig00*inv00+ig01*inv01);
float invGrad01=-(ig00*inv10+ig01*inv11);
float invGrad10=-(ig10*inv00+ig11*inv01);
float invGrad11=-(ig10*inv10+ig11*inv11);
"""

COMBINE_COT_COV2D = r"""
// ---- combine cotCov2d ----
float totalCot00=fused_cotCov2d_1[cv2dBase]+invGrad00;
float totalCot01=fused_cotCov2d_1[cv2dBase+1U]+invGrad01;
float totalCot10=fused_cotCov2d_1[cv2dBase+2U]+invGrad10;
float totalCot11=fused_cotCov2d_1[cv2dBase+3U]+invGrad11;
"""

COV2D_BACKWARD = r"""
// ---- cov2d backward: totalCotCov2d -> gradMeans3d_cov, gradCov3d ----
thread array<float, int(9)> b_arr;
b_arr[int(0)]=b00_0; b_arr[int(1)]=b01_0; b_arr[int(2)]=b02_0;
b_arr[int(3)]=b10_0; b_arr[int(4)]=b11_0; b_arr[int(5)]=b12_0;
b_arr[int(6)]=0.0;   b_arr[int(7)]=0.0;   b_arr[int(8)]=0.0;
thread array<float, int(9)> c_arr;
c_arr[int(0)]=cv3d00; c_arr[int(1)]=cv3d01; c_arr[int(2)]=cv3d02;
c_arr[int(3)]=cv3d10; c_arr[int(4)]=cv3d11; c_arr[int(5)]=cv3d12;
c_arr[int(6)]=cv3d20; c_arr[int(7)]=cv3d21; c_arr[int(8)]=cv3d22;
thread array<float, int(9)> cotM_0;
cotM_0[int(0)]=totalCot00; cotM_0[int(1)]=totalCot01; cotM_0[int(2)]=0.0;
cotM_0[int(3)]=totalCot10; cotM_0[int(4)]=totalCot11; cotM_0[int(5)]=0.0;
cotM_0[int(6)]=0.0;        cotM_0[int(7)]=0.0;        cotM_0[int(8)]=0.0;
thread array<float, int(9)> bt_0;
bt_0[int(0)]=b00_0; bt_0[int(1)]=b10_0; bt_0[int(2)]=0.0;
bt_0[int(3)]=b01_0; bt_0[int(4)]=b11_0; bt_0[int(5)]=0.0;
bt_0[int(6)]=b02_0; bt_0[int(7)]=b12_0; bt_0[int(8)]=0.0;
int ii, jj, kk; float vv, vv2;
thread array<float, int(9)> tmpA_0;
ii=int(0);
for(;;){ if(ii<int(3)){} else { break; }
  jj=int(0);
  for(;;){ if(jj<int(3)){} else { break; }
    kk=int(0); vv=0.0;
    for(;;){ if(kk<int(3)){} else { break; }
      vv=vv+bt_0[ii*int(3)+kk]*cotM_0[kk*int(3)+jj]; kk=kk+int(1); }
    tmpA_0[ii*int(3)+jj]=vv; jj=jj+int(1); }
  ii=ii+int(1); }
thread array<float, int(9)> gradC_0;
ii=int(0);
for(;;){ if(ii<int(3)){} else { break; }
  jj=int(0);
  for(;;){ if(jj<int(3)){} else { break; }
    kk=int(0); vv=0.0;
    for(;;){ if(kk<int(3)){} else { break; }
      vv=vv+tmpA_0[ii*int(3)+kk]*b_arr[kk*int(3)+jj]; kk=kk+int(1); }
    gradC_0[ii*int(3)+jj]=vv; jj=jj+int(1); }
  ii=ii+int(1); }
thread array<float, int(9)> ct_0;
ct_0[int(0)]=cv3d00; ct_0[int(1)]=cv3d10; ct_0[int(2)]=cv3d20;
ct_0[int(3)]=cv3d01; ct_0[int(4)]=cv3d11; ct_0[int(5)]=cv3d21;
ct_0[int(6)]=cv3d02; ct_0[int(7)]=cv3d12; ct_0[int(8)]=cv3d22;
thread array<float, int(9)> bct_0;
thread array<float, int(9)> bc_0;
ii=int(0);
for(;;){ if(ii<int(3)){} else { break; }
  jj=int(0);
  for(;;){ if(jj<int(3)){} else { break; }
    kk=int(0); vv=0.0; vv2=0.0;
    for(;;){ if(kk<int(3)){} else { break; }
      int idx=kk*int(3)+jj;
      vv=vv+b_arr[ii*int(3)+kk]*ct_0[idx];
      vv2=vv2+b_arr[ii*int(3)+kk]*c_arr[idx]; kk=kk+int(1); }
    int idx2=ii*int(3)+jj; bct_0[idx2]=vv; bc_0[idx2]=vv2;
    jj=jj+int(1); }
  ii=ii+int(1); }
thread array<float, int(9)> cotMt_0;
cotMt_0[int(0)]=totalCot00; cotMt_0[int(1)]=totalCot10; cotMt_0[int(2)]=0.0;
cotMt_0[int(3)]=totalCot01; cotMt_0[int(4)]=totalCot11; cotMt_0[int(5)]=0.0;
cotMt_0[int(6)]=0.0;        cotMt_0[int(7)]=0.0;        cotMt_0[int(8)]=0.0;
thread array<float, int(9)> term1_0;
thread array<float, int(9)> term2_0;
ii=int(0);
for(;;){ if(ii<int(3)){} else { break; }
  jj=int(0);
  for(;;){ if(jj<int(3)){} else { break; }
    kk=int(0); vv=0.0; vv2=0.0;
    for(;;){ if(kk<int(3)){} else { break; }
      int i3=ii*int(3)+kk; int i4=kk*int(3)+jj;
      vv=vv+cotM_0[i3]*bct_0[i4]; vv2=vv2+cotMt_0[i3]*bc_0[i4]; kk=kk+int(1); }
    int i5=ii*int(3)+jj; term1_0[i5]=vv; term2_0[i5]=vv2;
    jj=jj+int(1); }
  ii=ii+int(1); }
thread array<float, int(9)> gradB_0;
ii=int(0);
for(;;){ if(ii<int(9)){} else { break; }
  gradB_0[ii]=term1_0[ii]+term2_0[ii]; ii=ii+int(1); }
thread array<float, int(9)> wt_0;
wt_0[int(0)]=a00_0; wt_0[int(1)]=a01_0; wt_0[int(2)]=a02_0;
wt_0[int(3)]=a10_0; wt_0[int(4)]=a11_0; wt_0[int(5)]=a12_0;
wt_0[int(6)]=a20_0; wt_0[int(7)]=a21_0; wt_0[int(8)]=a22_0;
thread array<float, int(9)> gradJ_0;
ii=int(0);
for(;;){ if(ii<int(3)){} else { break; }
  jj=int(0);
  for(;;){ if(jj<int(3)){} else { break; }
    kk=int(0); vv=0.0;
    for(;;){ if(kk<int(3)){} else { break; }
      vv=vv+gradB_0[ii*int(3)+kk]*wt_0[kk*int(3)+jj]; kk=kk+int(1); }
    gradJ_0[ii*int(3)+jj]=vv; jj=jj+int(1); }
  ii=ii+int(1); }
float invTz2_0 = 1.0/_tz2;
float invTz3_0 = invTz2_0/t2_0;
float gTx_0 = gradJ_0[int(2)]*(-fused_focalX_1[int(0)]*invTz2_0);
float gTy_0 = gradJ_0[int(5)]*(-fused_focalY_1[int(0)]*invTz2_0);
float gTz_0 = gradJ_0[int(0)]*(-fused_focalX_1[int(0)]*invTz2_0)
            + gradJ_0[int(4)]*(-fused_focalY_1[int(0)]*invTz2_0)
            + gradJ_0[int(2)]*(2.0*fused_focalX_1[int(0)]*tx_0*invTz3_0)
            + gradJ_0[int(5)]*(2.0*fused_focalY_1[int(0)]*ty_0*invTz3_0);
float gT0_0 = gTx_0*(t2_0/clipX_0);
float gT1_0 = gTy_0*(t2_0/clipY_0);
float gUx_0 = gTx_0*t0_0;
float gUy_0 = gTy_0*t1_0;
bool _Sb1; float dClipX_v;
if(t2_0>minX_0){ _Sb1=t2_0<maxX_0; } else { _Sb1=false; }
if(_Sb1){ dClipX_v=1.0; } else { dClipX_v=0.0; }
bool _Sb2; float dClipY_v;
if(t2_0>minY_0){ _Sb2=t2_0<maxY_0; } else { _Sb2=false; }
if(_Sb2){ dClipY_v=1.0; } else { dClipY_v=0.0; }
float gTz_1 = gTz_0 + gUx_0*((clipX_0-t2_0*dClipX_v)/(clipX_0*clipX_0))
                      + gUy_0*((clipY_0-t2_0*dClipY_v)/(clipY_0*clipY_0));
float covGradM0 = gT0_0*a00_0 + gT1_0*a01_0 + gTz_1*a02_0;
float covGradM1 = gT0_0*a10_0 + gT1_0*a11_0 + gTz_1*a12_0;
float covGradM2 = gT0_0*a20_0 + gT1_0*a21_0 + gTz_1*a22_0;
"""

COV3D_BACKWARD = r"""
// ---- cov3d backward: gradCov3d -> gradScales, gradRotations ----
float g3d00=gradC_0[int(0)], g3d01=gradC_0[int(1)], g3d02=gradC_0[int(2)];
float g3d10=gradC_0[int(3)], g3d11=gradC_0[int(4)], g3d12=gradC_0[int(5)];
float g3d20=gradC_0[int(6)], g3d21=gradC_0[int(7)], g3d22=gradC_0[int(8)];
float h3d00=g3d00+g3d00, h3d01=g3d01+g3d10, h3d02=g3d02+g3d20;
float h3d10=g3d10+g3d01, h3d11=g3d11+g3d11, h3d12=g3d12+g3d21;
float h3d20=g3d20+g3d02, h3d21=g3d21+g3d12, h3d22=g3d22+g3d22;
float gl00=h3d00*l00+h3d01*l10+h3d02*l20;
float gl01=h3d00*l01+h3d01*l11+h3d02*l21;
float gl02=h3d00*l02+h3d01*l12+h3d02*l22;
float gl10=h3d10*l00+h3d11*l10+h3d12*l20;
float gl11=h3d10*l01+h3d11*l11+h3d12*l21;
float gl12=h3d10*l02+h3d11*l12+h3d12*l22;
float gl20=h3d20*l00+h3d21*l10+h3d22*l20;
float gl21=h3d20*l01+h3d21*l11+h3d22*l21;
float gl22=h3d20*l02+h3d21*l12+h3d22*l22;
*(fused_gradScales_1+scaleBase_0)     = gl00*r00+gl10*r10+gl20*r20;
*(fused_gradScales_1+(scaleBase_0+1U))= gl01*r01+gl11*r11+gl21*r21;
*(fused_gradScales_1+(scaleBase_0+2U))= gl02*r02+gl12*r12+gl22*r22;
float gr00=gl00*sx_0, gr01=gl01*sy_0, gr02=gl02*sz_0;
float gr10=gl10*sx_0, gr11=gl11*sy_0, gr12=gl12*sz_0;
float gr20=gl20*sx_0, gr21=gl21*sy_0, gr22=gl22*sz_0;
float _tqy2=2.0*qy, _tqz2=2.0*qz, _tqx2=2.0*qx;
float _tqw2=2.0*qw, _nqw2=-2.0*qw;
float gqw = gr01*(-2.0*qz)+gr02*_tqy2+gr10*_tqz2+gr12*(-2.0*qx)+gr20*(-2.0*qy)+gr21*_tqx2;
float gqx = gr01*_tqy2+gr02*_tqz2+gr10*_tqy2+gr11*(-4.0*qx)+gr12*_nqw2+gr20*_tqz2+gr21*_tqw2+gr22*(-4.0*qx);
float gqy = gr00*(-4.0*qy)+gr01*_tqx2+gr02*_tqw2+gr10*_tqx2+gr12*_tqz2+gr20*_nqw2+gr21*_tqz2+gr22*(-4.0*qy);
float gqz = gr00*(-4.0*qz)+gr01*_nqw2+gr02*_tqx2+gr10*_tqw2+gr11*(-4.0*qz)+gr12*_tqy2+gr20*_tqx2+gr21*_tqy2;
float invNorm_0 = 1.0/_sn;
float gradRw_0, gradRx_0, gradRy_0, gradRz_0;
if(norm_0 > 9.99999993922529029e-09){
    float invNorm3_0=invNorm_0*invNorm_0*invNorm_0;
    float dot_0=gqw*rw_0+gqx*rx_0+gqy*ry_0+gqz*rz_0;
    gradRw_0=gqw*invNorm_0-rw_0*dot_0*invNorm3_0;
    gradRx_0=gqx*invNorm_0-rx_0*dot_0*invNorm3_0;
    gradRy_0=gqy*invNorm_0-ry_0*dot_0*invNorm3_0;
    gradRz_0=gqz*invNorm_0-rz_0*dot_0*invNorm3_0;
} else {
    gradRw_0=gqw*invNorm_0;
    gradRx_0=gqx*invNorm_0;
    gradRy_0=gqy*invNorm_0;
    gradRz_0=gqz*invNorm_0;
}
*(fused_gradRotations_1+rotBase_0)     = gradRw_0;
*(fused_gradRotations_1+(rotBase_0+1U))= gradRx_0;
*(fused_gradRotations_1+(rotBase_0+2U))= gradRy_0;
*(fused_gradRotations_1+(rotBase_0+3U))= gradRz_0;
"""

WRITE_GRAD_MEANS3D = r"""
// ---- combined gradMeans3d ----
*(fused_gradMeans3d_1+meansBase_0)     = gradX_0 + covGradM0;
*(fused_gradMeans3d_1+(meansBase_0+1U))= gradY_0 + covGradM1;
*(fused_gradMeans3d_1+(meansBase_0+2U))= gradZ_0 + covGradM2;
return;
"""


def lines_to_source(code: str) -> str:
    """Strip leading/trailing blank lines, keep internal structure."""
    lines = code.strip().split("\n")
    return "\n".join(lines)


def build_forward_source() -> str:
    parts = [
        "uint3 tid_0 = thread_position_in_grid;",
        "uint p_0 = tid_0.x;",
        "if(p_0 >= fused_counts_1[int(0)]){ return; }",
        COV3D_FROM_SCALES_ROTATIONS,
        SH_BASIS_FORWARD,
        COLOR_EVAL_FORWARD,
        COV2D_FROM_COV3D,
        INVERSE_FORWARD,
    ]
    return lines_to_source("\n".join(parts))


def build_backward_source() -> str:
    parts = [
        "uint3 tid_0 = thread_position_in_grid;",
        "uint p_0 = tid_0.x;",
        "if(p_0 >= fused_counts_1[int(0)]){ return; }",
        COV3D_FROM_SCALES_ROTATIONS,
        SH_BASIS_BACKWARD,
        COLOR_BACKWARD,
        COV2D_RECOMPUTE_FOR_BACKWARD,
        INVERSE_BACKWARD,
        COMBINE_COT_COV2D,
        COV2D_BACKWARD,
        COV3D_BACKWARD,
        WRITE_GRAD_MEANS3D,
    ]
    return lines_to_source("\n".join(parts))


def make_forward_json() -> dict:
    return {
        "kernel_name": "gaussian_screen_fused_forward_v1",
        "input_names": [
            "fused_scales_1",
            "fused_rotations_1",
            "fused_means3d_1",
            "fused_shs_1",
            "fused_cameraCenter_1",
            "fused_viewMatrix_1",
            "fused_fovX_1",
            "fused_fovY_1",
            "fused_focalX_1",
            "fused_focalY_1",
            "fused_counts_1",
        ],
        "output_names": [
            "fused_outColor_1",
            "fused_outCov2d_1",
            "fused_outConic_1",
        ],
        "source": build_forward_source(),
        "header": HEADER,
    }


def make_backward_json() -> dict:
    return {
        "kernel_name": "gaussian_screen_fused_backward_v1",
        "input_names": [
            "fused_scales_1",
            "fused_rotations_1",
            "fused_means3d_1",
            "fused_shs_1",
            "fused_cameraCenter_1",
            "fused_viewMatrix_1",
            "fused_fovX_1",
            "fused_fovY_1",
            "fused_focalX_1",
            "fused_focalY_1",
            "fused_cotColor_1",
            "fused_cotCov2d_1",
            "fused_cotConic_1",
            "fused_counts_1",
        ],
        "output_names": [
            "fused_gradScales_1",
            "fused_gradRotations_1",
            "fused_gradMeans3d_1",
            "fused_gradShs_1",
        ],
        "source": build_backward_source(),
        "header": HEADER,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fwd_path = args.out_dir / "gaussian_screen_fused_forward_mlx.json"
    bwd_path = args.out_dir / "gaussian_screen_fused_backward_mlx.json"

    with open(fwd_path, "w") as f:
        json.dump(make_forward_json(), f, indent=2, ensure_ascii=False)
    print(f"wrote {fwd_path}")

    with open(bwd_path, "w") as f:
        json.dump(make_backward_json(), f, indent=2, ensure_ascii=False)
    print(f"wrote {bwd_path}")


if __name__ == "__main__":
    main()
