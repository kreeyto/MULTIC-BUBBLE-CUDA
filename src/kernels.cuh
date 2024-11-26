#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void gpuMomCollisionStreamBCS (
    float *f, float *g, float *phi, float *rho, const float *w, const float *w_g,
    const float *cix, const float *ciy, const float *ciz, 
    float *mod_grad, float *normx, float *normy, float *normz, float *indicator,
    float *curvature, float *ffx, float *ffy, float *ffz,
    float *ux, float *uy, float *uz,
    float *pxx, float *pyy, float *pzz, float *pxy, float *pxz, float *pyz,
    int nx, int ny, int nz, int fpoints, int gpoints, 
    float sigma, float cssq, float omega, float sharp_c, int nsteps
);

#endif
