#ifndef NEOKERNELS_CUH
#define NEOKERNELS_CUH

__global__ void phiCalc(
    float *phi, float *g,
    int nx, int ny, int nz
);

__global__ void gradCalc(
    float *w, const float *cix, const float *ciy, const float *ciz, 
    int fpoints, int nx, int ny, int nz, float grad_fix, float grad_fiy, float grad_fiz
);

__global__ void postGrad(
    float *mod_grad, float *normx, float *normy, float *normz, 
    float *indicator, int nx, int ny, int nz, float grad_fix, float grad_fiy, float grad_fiz
);

__global__ void curvatureCalc(
    float *curvature,
    const float *cix, const float *ciy, const float *ciz,
    float *normx, float *normy, float *normz,
    int fpoints, int nx, int ny, int nz
);

__global__ void postCurv(
    float *curvature, float *indicator,
    float *normx, float *normy, float *normz,
    float *ffx, float *ffy, float *ffz, float sigma,
    int nx, int ny, int nz
);

__global__ void momentiCalc(
    float *ux, float *uy, float *uz, float *rho,
    float *ffx, float *ffy, float *ffz, float *f,
    int nx, int ny, int nz
);

__global__ void uuCalc(
    float *ux, float *uy, float *uz,
    int nx, int ny, int nz, float cssq
);

__global__ void rhoCalc(
    float *rho, float *f,
    int nx, int ny, int nz
);

__global__ void fneqCalc(
    float *ux, float *uy, float *uz, float *rho,
    float *ffx, float *ffy, float *ffz, float *w, float *f,
    const float *cix, const float *ciy, const float *ciz,
    float cssq, int nx, int ny, int nz,
    int fpoints, float *fneq
);

__global__ void tensorCalc(
    float *pxx, float *pyy, float *pzz,
    float *pxy, float *pxz, float *pyz,
    int nx, int ny, int nz, float *fneq
);

// call uu again

__global__ void collisionCalc(
    float *ux, float *uy, float *uz, float *w,
    const float *cix, const float *ciy, const float *ciz,
    float *ffx, float *ffy, float *ffz,
    float *rho,float *f,
    float *pxx, float *pyy, float *pzz, float *pxy, float *pxz, float *pyz,
    float cssq, float omega, int fpoints,
    int nx, int ny, int nz
);

__global__ void colStep(
    float *ux, float *uy, float *uz, float *w_g,
    const float *cix, const float *ciy, const float *ciz,
    float *normx, float *normy, float *normz,
    float *phi, float *g,
    float cssq, float sharp_c, int gpoints,
    int nx, int ny, int nz
);

__global__ void streamingCalc(
    float *g, const float *cix, const float *ciy, const float *ciz,
    int nx, int ny, int nz, int gpoints
);

__global__ void fgBoundary(
    float *f, float *g, float *rho, float *phi, float *w, float *w_g,
    const float *cix, const float *ciy, const float *ciz,
    int fpoints, int gpoints, int nx, int ny, int nz
);

__global__ void boundaryConditions(
    float *phi, int nx, int ny, int nz
);

#endif
