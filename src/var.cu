#include "var.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "precision.cuh"

int mesh = 128;
int nx = mesh; int ny = mesh; int nz = mesh;  

__constant__ dfloat TAU;
__constant__ dfloat CSSQ;
__constant__ dfloat OMEGA;
__constant__ dfloat SHARP_C;
__constant__ dfloat SIGMA;
__constant__ dfloat W[FPOINTS], W_G[GPOINTS];
__constant__ int CIX[FPOINTS], CIY[FPOINTS], CIZ[FPOINTS];

dfloat *d_f, *d_g;
dfloat *d_normx, *d_normy, *d_normz, *d_indicator;
dfloat *d_curvature, *d_ffx, *d_ffy, *d_ffz;
dfloat *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
dfloat *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;
dfloat *d_g_out; // *d_f_coll

// ========================================================================== parametros ========================================================================== //
dfloat h_tau = 0.505;
dfloat h_cssq = 1.0 / 3.0;
dfloat h_omega = 1.0 / h_tau;
dfloat h_sharp_c = 0.15 * 3.0;
dfloat h_sigma = 0.1;

// fluid velocity set
#ifdef FD3Q19
    int h_cix[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0 };
    int h_ciy[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1 };
    int h_ciz[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1 };
#elif defined(FD3Q27)
    int h_cix[27] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1 };
    int h_ciy[27] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };
    int h_ciz[27] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1 };
#endif

// fluid weights
#ifdef FD3Q19
    dfloat h_w[19] = {
        1.0 / 3.0, 
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
#elif defined(FD3Q27)
    dfloat h_w[27] = {
        8.0 / 27.0,
        2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 
        1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 
        1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0
    };
#endif

// phase field weights
#ifdef PD3Q19
    dfloat h_w_g[19] = {
        1.0 / 3.0, 
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
#endif
// =============================================================================================================================================================== //

void initializeVars() {
    size_t size = nx * ny * nz * sizeof(dfloat);            
    size_t f_size = nx * ny * nz * FPOINTS * sizeof(dfloat); 
    size_t g_size = nx * ny * nz * GPOINTS * sizeof(dfloat); 

    cudaMalloc((void **)&d_rho, size);
    cudaMalloc((void **)&d_phi, size);
    cudaMalloc((void **)&d_ux, size);
    cudaMalloc((void **)&d_uy, size);
    cudaMalloc((void **)&d_uz, size);
    cudaMalloc((void **)&d_normx, size);
    cudaMalloc((void **)&d_normy, size);
    cudaMalloc((void **)&d_normz, size);
    cudaMalloc((void **)&d_curvature, size);
    cudaMalloc((void **)&d_indicator, size);
    cudaMalloc((void **)&d_ffx, size);
    cudaMalloc((void **)&d_ffy, size);
    cudaMalloc((void **)&d_ffz, size);
    cudaMalloc((void **)&d_pxx, size);
    cudaMalloc((void **)&d_pyy, size);
    cudaMalloc((void **)&d_pzz, size);
    cudaMalloc((void **)&d_pxy, size);
    cudaMalloc((void **)&d_pxz, size);
    cudaMalloc((void **)&d_pyz, size);

    cudaMalloc((void **)&d_f, f_size);
    cudaMalloc((void **)&d_g, g_size);

    cudaMalloc((void **)&d_g_out, g_size);

    cudaMemset(d_phi, 0, size);
    cudaMemset(d_ux, 0, size);
    cudaMemset(d_uy, 0, size);
    cudaMemset(d_uz, 0, size);
    
    cudaMemset(d_f, 0, f_size);
    cudaMemset(d_g, 0, g_size);

    cudaMemset(d_normx, 0, size);
    cudaMemset(d_normy, 0, size);
    cudaMemset(d_normz, 0, size);
    cudaMemset(d_curvature, 0, size);
    cudaMemset(d_indicator, 0, size);
    cudaMemset(d_ffx, 0, size);
    cudaMemset(d_ffy, 0, size);
    cudaMemset(d_ffz, 0, size);

    cudaMemcpyToSymbol(TAU, &h_tau, sizeof(dfloat));
    cudaMemcpyToSymbol(CSSQ, &h_cssq, sizeof(dfloat));
    cudaMemcpyToSymbol(OMEGA, &h_omega, sizeof(dfloat));
    cudaMemcpyToSymbol(SHARP_C, &h_sharp_c, sizeof(dfloat));
    cudaMemcpyToSymbol(SIGMA, &h_sigma, sizeof(dfloat));

    cudaMemcpyToSymbol(W, &h_w, FPOINTS * sizeof(dfloat));
    cudaMemcpyToSymbol(W_G, &h_w_g, GPOINTS * sizeof(dfloat));

    cudaMemcpyToSymbol(CIX, &h_cix, FPOINTS * sizeof(int));
    cudaMemcpyToSymbol(CIY, &h_ciy, FPOINTS * sizeof(int));
    cudaMemcpyToSymbol(CIZ, &h_ciz, FPOINTS * sizeof(int));

}

