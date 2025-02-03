#pragma once

#include "precision.cuh"

extern dfloat res;
extern int nx, ny, nz, fpoints, gpoints;
extern dfloat tau, cssq, omega, sharp_c, sigma;
 
extern dfloat *d_f, *d_g, *d_w, *d_w_g, *d_cix, *d_ciy, *d_ciz;
extern dfloat *d_normx, *d_normy, *d_normz, *d_indicator, *d_mod_grad;
extern dfloat *d_curvature, *d_ffx, *d_ffy, *d_ffz;
extern dfloat *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
extern dfloat *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;
//extern dfloat *d_fneq;
//extern dfloat *d_grad_fix, *d_grad_fiy, *d_grad_fiz, *d_uu;
extern dfloat *d_f_coll, *d_g_out ;

void initializeVars();
