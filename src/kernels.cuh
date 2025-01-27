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
    dfloat *phi, dfloat *w, const dfloat *cix,
    dfloat *grad_fix, dfloat *grad_fiy, dfloat *grad_fiz,
    const dfloat *ciy, const dfloat *ciz, int fpoints,
    int nx, int ny, int nz
);

__global__ void normCalc(
    dfloat *grad_fix, dfloat *grad_fiy, dfloat *grad_fiz,
    dfloat *mod_grad, dfloat *normx, dfloat *normy, dfloat *normz,
    dfloat *indicator, int nx, int ny, int nz
);

__global__ void curvatureCalc(
    dfloat *curvature, dfloat *indicator, dfloat *w,
    const dfloat *cix, const dfloat *ciy, const dfloat *ciz,
    dfloat *normx, dfloat *normy, dfloat *normz,
    dfloat *ffx, dfloat *ffy, dfloat *ffz, dfloat sigma,
    int fpoints, int nx, int ny, int nz

);

__global__ void forceCalc(
    dfloat *ffx, dfloat *ffy, dfloat *ffz,
    dfloat sigma, dfloat *curvature, dfloat *indicator,
    dfloat *normx, dfloat *normy, dfloat *normz,
    int nx, int ny, int nz
);

__global__ void macroCalc(
    dfloat *ux, dfloat *uy, dfloat *uz, dfloat *f,
    dfloat *ffx, dfloat *ffy, dfloat *ffz, dfloat *rho,
    int nx, int ny, int nz
);

__global__ void uuCalc(
    dfloat *ux, dfloat *uy, dfloat *uz, dfloat *uu,
    dfloat cssq, int nx, int ny, int nz
);

__global__ void rhoCalc(
    dfloat *rho, dfloat *f, int nx, int ny, int nz
);

__global__ void momentiCalc(
    dfloat *ux, dfloat *uy, dfloat *uz, dfloat *w,
    dfloat *cix, dfloat *ciy, dfloat *ciz,
    dfloat *ffx, dfloat *ffy, dfloat *ffz,
    dfloat *uu, dfloat *rho, dfloat *fneq,
    dfloat *f, dfloat cssq, int nx, int ny, int nz,
    int fpoints
);

__global__ void tensorCalc(
    dfloat *pxx, dfloat *pyy, dfloat *pzz,
    dfloat *pxy, dfloat *pxz, dfloat *pyz,
    int nx, int ny, int nz, dfloat *fneq
);

__global__ void fCalc(
    dfloat *ux, dfloat *uy, dfloat *uz,
    dfloat *cix, dfloat *ciy, dfloat *ciz,
    dfloat *w, dfloat *rho, dfloat *uu,
    dfloat *ffx, dfloat *ffy, dfloat *ffz,
    dfloat *pxx, dfloat *pyy, dfloat *pzz,
    dfloat *pxy, dfloat *pxz, dfloat *pyz,
    dfloat *f, dfloat omega, dfloat cssq,
    int fpoints, int nx, int ny, int nz
);

__global__ void gCalc(
    dfloat *ux, dfloat *uy, dfloat *uz,
    dfloat *cix, dfloat *ciy, dfloat *ciz,
    dfloat *w_g, dfloat *phi, dfloat *g,
    dfloat *normx, dfloat *normy, dfloat *normz,
    dfloat cssq, int gpoints, int nx, int ny, int nz,
    dfloat sharp_c
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
