#pragma once

extern float res;
extern int nx, ny, nz, fpoints, gpoints;
extern float tau, cssq, omega, sharp_c, sigma;
 
extern float *d_f, *d_g, *d_w, *d_w_g, *d_cix, *d_ciy, *d_ciz;
extern float *d_normx, *d_normy, *d_normz, *d_indicator, *d_mod_grad;
extern float *d_curvature, *d_ffx, *d_ffy, *d_ffz;
extern float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
extern float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;

extern float *d_fneq;

void initializeVars();
