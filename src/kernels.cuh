#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "var.cuh"
#include <math.h>

#include "precision.cuh"

__global__ void phiCalc(
    dfloat * __restrict__ phi,
    const dfloat * __restrict__ g,
    int nx, int ny, int nz
);

__global__ void gradCalc(
    const dfloat * __restrict__ phi,
    dfloat * __restrict__ mod_grad,
    dfloat * __restrict__ normx,
    dfloat * __restrict__ normy,
    dfloat * __restrict__ normz,
    dfloat * __restrict__ indicator,
    const dfloat * __restrict__ w,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    int fpoints, int nx, int ny, int nz
);

__global__ void curvatureCalc(
    dfloat * __restrict__ curvature,
    const dfloat * __restrict__ indicator,
    const dfloat * __restrict__ w,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    const dfloat * __restrict__ normx,
    const dfloat * __restrict__ normy,
    const dfloat * __restrict__ normz,
    dfloat * __restrict__ ffx,
    dfloat * __restrict__ ffy,
    dfloat * __restrict__ ffz,
    dfloat sigma,
    int fpoints, int nx, int ny, int nz
);

__global__ void momentiCalc(
    dfloat * __restrict__ ux,
    dfloat * __restrict__ uy,
    dfloat * __restrict__ uz,
    dfloat * __restrict__ rho,
    dfloat * __restrict__ ffx,
    dfloat * __restrict__ ffy,
    dfloat * __restrict__ ffz,
    const dfloat * __restrict__ w,
    const dfloat * __restrict__ f,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    dfloat * __restrict__ pxx,
    dfloat * __restrict__ pyy,
    dfloat * __restrict__ pzz,
    dfloat * __restrict__ pxy,
    dfloat * __restrict__ pxz,
    dfloat * __restrict__ pyz,
    dfloat cssq,
    int nx, int ny, int nz, int fpoints 
);

/*__global__ void collisionCalc(
    const dfloat * __restrict__ ux,
    const dfloat * __restrict__ uy,
    const dfloat * __restrict__ uz,
    const dfloat * __restrict__ w,
    const dfloat * __restrict__ w_g,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    const dfloat * __restrict__ normx,
    const dfloat * __restrict__ normy,
    const dfloat * __restrict__ normz,
    const dfloat * __restrict__ ffx,
    const dfloat * __restrict__ ffy,
    const dfloat * __restrict__ ffz,
    const dfloat * __restrict__ rho,
    const dfloat * __restrict__ phi,
    const dfloat * __restrict__ f,
    dfloat * __restrict__ g,
    const dfloat * __restrict__ pxx,
    const dfloat * __restrict__ pyy,
    const dfloat * __restrict__ pzz,
    const dfloat * __restrict__ pxy,
    const dfloat * __restrict__ pxz,
    const dfloat * __restrict__ pyz,
    dfloat cssq, dfloat omega, dfloat sharp_c,
    int fpoints, int gpoints, int nx, int ny, int nz,
    dfloat * __restrict__ f_coll
);*/
__global__ void collisionCalc(
    dfloat *ux, dfloat *uy, dfloat *uz, dfloat *w, dfloat *w_g,
    const dfloat *cix, const dfloat *ciy, const dfloat *ciz,
    dfloat *normx, dfloat *normy, dfloat *normz,
    dfloat *ffx, dfloat *ffy, dfloat *ffz,
    dfloat *rho, dfloat *phi, dfloat *f, dfloat *g,
    dfloat *pxx, dfloat *pyy, dfloat *pzz, dfloat *pxy, dfloat *pxz, dfloat *pyz,
    dfloat cssq, dfloat omega, dfloat sharp_c, int fpoints, int gpoints,
    int nx, int ny, int nz, dfloat *f_coll // escrita final na memória global
);

__global__ void streamingCalcNew(
    const dfloat * __restrict__ f_coll,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    int nx, int ny, int nz, int fpoints,
    dfloat * __restrict__ f 
);

__global__ void streamingCalc(
    const dfloat * __restrict__ g_in,
    dfloat * __restrict__ g_out,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    int nx, int ny, int nz,
    int gpoints
);

__global__ void fgBoundary(
    dfloat * __restrict__ f,
    dfloat * __restrict__ g,
    const dfloat * __restrict__ rho,
    const dfloat * __restrict__ phi,
    const dfloat * __restrict__ w,
    const dfloat * __restrict__ w_g,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    int fpoints, int gpoints,
    int nx, int ny, int nz
);

__global__ void boundaryConditions(
    dfloat * __restrict__ phi,
    int nx, int ny, int nz
);

#endif
