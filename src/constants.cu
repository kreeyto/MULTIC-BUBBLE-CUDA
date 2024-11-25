#include "constants.cuh"
#include <cuda_runtime.h>
#include <iostream>

// Constantes globais
int nx = 128, ny = 128, nz = 128, fpoints = 19, gpoints = 15, nsteps = 10000;
float tau = 0.505, cssq = 1.0f / 3.0f, omega = 1.0f / tau, sharp_c = 0.45f, sigma = 0.1f;

// Vetores globais da GPU
float *d_f *d_g, *d_w, *d_w_g, *d_cix, *d_ciy, *d_ciz;
float *d_grad_fix, *d_grad_fiy, *d_grad_fiz, *d_mod_grad;
float *d_normx, *d_normy, *d_normz, *d_indicator;
float *d_curvature, *d_ffx, *d_ffy, *d_ffz;
float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
float *d_pxy, *d_pxz, *d_pyz;

// Pesos e direções no host
const float w[19] = {
    1.0f / 3.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 18.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

const float w_g[15] = {
    2.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
    1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f,
    1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f
};

const float cix[19] = { 0, 1, -1, 0, 0, 0,  0,  1, -1,  1, -1,  0,  0,  0,  1, -1,  1, -1,  0 };
const float ciy[19] = { 0, 0,  0, 1, -1, 0,  0,  1,  1, -1, -1,  1, -1,  0,  0,  0,  0,  0, -1 };
const float ciz[19] = { 0, 0,  0, 0,  0, 1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1 };

// Função para inicializar as constantes na GPU
void initializeConstants() {
    size_t size = nx * ny * nz * sizeof(float);
    size_t f_size = nx * ny * nz * fpoints * sizeof(float);
    size_t g_size = nx * ny * nz * gpoints * sizeof(float);

    cudaMalloc((void **)&d_g, g_size);
    cudaMalloc((void **)&d_w, f_size);
    cudaMalloc((void **)&d_w_g, g_size);
    cudaMalloc((void **)&d_cix, f_size);
    cudaMalloc((void **)&d_ciy, f_size);
    cudaMalloc((void **)&d_ciz, f_size);
    cudaMalloc((void **)&d_grad_fix, size);
    cudaMalloc((void **)&d_grad_fiy, size);
    cudaMalloc((void **)&d_grad_fiz, size);
    cudaMalloc((void **)&d_mod_grad, size);
    cudaMalloc((void **)&d_normx, size);
    cudaMalloc((void **)&d_normy, size);
    cudaMalloc((void **)&d_normz, size);
    cudaMalloc((void **)&d_indicator, size);
    cudaMalloc((void **)&d_curvature, size);
    cudaMalloc((void **)&d_ffx, size);
    cudaMalloc((void **)&d_ffy, size);
    cudaMalloc((void **)&d_ffz, size);
    cudaMalloc((void **)&d_f, f_size);
    cudaMalloc((void **)&d_ux, size);
    cudaMalloc((void **)&d_uy, size);
    cudaMalloc((void **)&d_uz, size);
    cudaMalloc((void **)&d_pxx, size);
    cudaMalloc((void **)&d_pyy, size);
    cudaMalloc((void **)&d_pzz, size);
    cudaMalloc((void **)&d_pxy, size);
    cudaMalloc((void **)&d_pxz, size);
    cudaMalloc((void **)&d_pyz, size);

    // Copiando valores constantes para a GPU
    cudaMemcpy(d_w, w, sizeof(w), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_g, w_g, sizeof(w_g), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cix, cix, sizeof(cix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ciy, ciy, sizeof(ciy), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ciz, ciz, sizeof(ciz), cudaMemcpyHostToDevice);
}
