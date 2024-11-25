#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>
#include <cuda_runtime.h>

// Protótipos dos kernels
__global__ void phaseFieldCalc(
    float *phi, const float *g, const float *w, const float *cix, const float *ciy, const float *ciz,
    float *grad_fix, float *grad_fiy, float *grad_fiz, float *mod_grad,
    float *normx, float *normy, float *normz, float *indicator,
    float *curvature, float *ffx, float *ffy, float *ffz,
    int nx, int ny, int nz, int fpoints, float sigma);

__global__ void momentCalc(
    const float *f, const float *rho, const float *ffx, const float *ffy, const float *ffz,
    const float *cix, const float *ciy, const float *ciz,
    float *ux, float *uy, float *uz, float *pxx, float *pyy, float *pzz,
    float *pxy, float *pxz, float *pyz,
    int nx, int ny, int nz, int fpoints, float cssq, const float *w);

__global__ void collisionCalc(
    float *f, float *g, const float *phi, const float *rho, const float *ux, const float *uy, const float *uz,
    const float *ffx, const float *ffy, const float *ffz,
    const float *pxx, const float *pyy, const float *pzz, const float *pxy, const float *pxz, const float *pyz,
    const float *cix, const float *ciy, const float *ciz, const float *w, const float *w_g,
    const float *normx, const float *normy, const float *normz,
    int nx, int ny, int nz, int fpoints, int gpoints, float cssq, float omega, float sharp_c);

__global__ void boundaryAndShiftKernel(
    float *f, float *g, const float *rho, const float *phi,
    const float *w, const float *w_g, const float *cix, const float *ciy, const float *ciz,
    int nx, int ny, int nz, int fpoints, int gpoints);

#endif
