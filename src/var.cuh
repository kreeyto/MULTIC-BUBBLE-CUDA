#pragma once

#include "precision.cuh"

extern int mesh;
extern int nx, ny, nz;

extern __constant__ dfloat TAU, CSSQ, OMEGA, SHARP_C, SIGMA;
extern __constant__ dfloat W[FPOINTS], W_G[GPOINTS];
extern __constant__ int CIX[FPOINTS], CIY[FPOINTS], CIZ[FPOINTS];
 
extern dfloat *d_f, *d_g;
extern dfloat *d_normx, *d_normy, *d_normz, *d_indicator;
extern dfloat *d_curvature, *d_ffx, *d_ffy, *d_ffz;
extern dfloat *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
extern dfloat *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;
extern dfloat *d_g_out; // *d_f_coll

void initializeVars();
