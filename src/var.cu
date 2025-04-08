#include "var.cuh"

int MESH = 64;
int NX = MESH; int NY = MESH; int NZ = MESH*4;  

__constant__ float CSSQ;
__constant__ float OMEGA;
__constant__ float SHARP_C;
__constant__ float SIGMA;
__constant__ float W[NLINKS];
__constant__ int CIX[NLINKS], CIY[NLINKS], CIZ[NLINKS];

float *d_f, *d_g;
float *d_normx, *d_normy, *d_normz, *d_indicator;
float *d_curvature, *d_ffx, *d_ffy, *d_ffz;
float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;
float *d_g_out;

// ========================================================================== parametros ========================================================================== //
float H_TAU = 0.505f;
float H_CSSQ = 1.0f / 3.0f;
float H_OMEGA = 1.0f / H_TAU;
float H_SHARP_C = 0.15f * 3.0f;
float H_SIGMA = 0.1f;

// velocity set
#ifdef D3Q19
    int H_CIX[19] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0 };
    int H_CIY[19] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1 };
    int H_CIZ[19] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1 };
#elif defined(D3Q27)
    int H_CIX[27] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1 };
    int H_CIY[27] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1 };
    int H_CIZ[27] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1 };
#endif

// vs weights
#ifdef D3Q19
    float H_W[19] = {
        1.0f / 3.0f, 
        1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
    };
#elif defined(D3Q27)
    float H_W[27] = {
        8.0f / 27.0f,
        2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 
        1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 
        1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f
    };
#endif

// =============================================================================================================================================================== //

void initializeVars() {
    size_t SIZE = NX * NY * NZ * sizeof(float);            
    size_t DIST_SIZE = NX * NY * NZ * NLINKS * sizeof(float); 

    cudaMalloc((void **)&d_rho, SIZE);
    cudaMalloc((void **)&d_phi, SIZE);
    cudaMalloc((void **)&d_ux, SIZE);
    cudaMalloc((void **)&d_uy, SIZE);
    cudaMalloc((void **)&d_uz, SIZE);
    cudaMalloc((void **)&d_normx, SIZE);
    cudaMalloc((void **)&d_normy, SIZE);
    cudaMalloc((void **)&d_normz, SIZE);
    cudaMalloc((void **)&d_curvature, SIZE);
    cudaMalloc((void **)&d_indicator, SIZE);
    cudaMalloc((void **)&d_ffx, SIZE);
    cudaMalloc((void **)&d_ffy, SIZE);
    cudaMalloc((void **)&d_ffz, SIZE);
    cudaMalloc((void **)&d_pxx, SIZE);
    cudaMalloc((void **)&d_pyy, SIZE);
    cudaMalloc((void **)&d_pzz, SIZE);
    cudaMalloc((void **)&d_pxy, SIZE);
    cudaMalloc((void **)&d_pxz, SIZE);
    cudaMalloc((void **)&d_pyz, SIZE);

    cudaMalloc((void **)&d_f, DIST_SIZE);
    cudaMalloc((void **)&d_g, DIST_SIZE);

    cudaMalloc((void **)&d_g_out, DIST_SIZE);

    cudaMemset(d_phi, 0, SIZE);
    cudaMemset(d_ux, 0, SIZE);
    cudaMemset(d_uy, 0, SIZE);
    cudaMemset(d_uz, 0, SIZE);
    
    cudaMemset(d_f, 0, DIST_SIZE);
    cudaMemset(d_g, 0, DIST_SIZE);

    cudaMemset(d_normx, 0, SIZE);
    cudaMemset(d_normy, 0, SIZE);
    cudaMemset(d_normz, 0, SIZE);
    cudaMemset(d_curvature, 0, SIZE);
    cudaMemset(d_indicator, 0, SIZE);
    cudaMemset(d_ffx, 0, SIZE);
    cudaMemset(d_ffy, 0, SIZE);
    cudaMemset(d_ffz, 0, SIZE);

    cudaMemcpyToSymbol(CSSQ, &H_CSSQ, sizeof(float));
    cudaMemcpyToSymbol(OMEGA, &H_OMEGA, sizeof(float));
    cudaMemcpyToSymbol(SHARP_C, &H_SHARP_C, sizeof(float));
    cudaMemcpyToSymbol(SIGMA, &H_SIGMA, sizeof(float));

    cudaMemcpyToSymbol(W, &H_W, NLINKS * sizeof(float));

    cudaMemcpyToSymbol(CIX, &H_CIX, NLINKS * sizeof(int));
    cudaMemcpyToSymbol(CIY, &H_CIY, NLINKS * sizeof(int));
    cudaMemcpyToSymbol(CIZ, &H_CIZ, NLINKS * sizeof(int));

}

