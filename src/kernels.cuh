#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "var.cuh"
#include <math.h>

#include "precision.cuh"

__global__ void phiCalc(
    dfloat *phi, dfloat *g, 
    int nx, int ny, int nz
);

__global__ void gradCalc(
    dfloat *phi, dfloat *mod_grad, dfloat *normx, dfloat *normy,
    dfloat *normz, dfloat *indicator, dfloat *w, const dfloat *cix,
    const dfloat *ciy, const dfloat *ciz, int fpoints,
    int nx, int ny, int nz
);

__global__ void curvatureCalc(
    dfloat *curvature, dfloat *indicator, dfloat *w,
    const dfloat *cix, const dfloat *ciy, const dfloat *ciz,
    dfloat *normx, dfloat *normy, dfloat *normz,
    dfloat *ffx, dfloat *ffy, dfloat *ffz, dfloat sigma,
    int fpoints, int nx, int ny, int nz
);

__global__ void momentiCalc(
    dfloat *ux, dfloat *uy, dfloat *uz, dfloat *rho,
    dfloat *ffx, dfloat *ffy, dfloat *ffz, dfloat *w, dfloat *f,
    const dfloat *cix, const dfloat *ciy, const dfloat *ciz,
    dfloat *pxx, dfloat *pyy, dfloat *pzz,
    dfloat *pxy, dfloat *pxz, dfloat *pyz,
    dfloat cssq, int nx, int ny, int nz,
    int fpoints, dfloat *fneq
);

__global__ void collisionCalc(
    dfloat *ux, dfloat *uy, dfloat *uz, dfloat *w, dfloat *w_g,
    const dfloat *cix, const dfloat *ciy, const dfloat *ciz,
    dfloat *normx, dfloat *normy, dfloat *normz,
    dfloat *ffx, dfloat *ffy, dfloat *ffz,
    dfloat *rho, dfloat *phi, dfloat *f, dfloat *g,
    dfloat *pxx, dfloat *pyy, dfloat *pzz, dfloat *pxy, dfloat *pxz, dfloat *pyz,
    dfloat cssq, dfloat omega, dfloat sharp_c, int fpoints, int gpoints,
    int nx, int ny, int nz
);

__global__ void streamingCalc(
    dfloat *g, const dfloat *cix, const dfloat *ciy, const dfloat *ciz,
    int nx, int ny, int nz, int gpoints
);

__global__ void fgBoundary(
    dfloat *f, dfloat *g, dfloat *rho, dfloat *phi, dfloat *w, dfloat *w_g,
    const dfloat *cix, const dfloat *ciy, const dfloat *ciz,
    int fpoints, int gpoints, int nx, int ny, int nz
);

__global__ void boundaryConditions(
    dfloat *phi, int nx, int ny, int nz
);

#endif
