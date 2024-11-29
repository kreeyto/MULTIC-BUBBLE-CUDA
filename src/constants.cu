#include "constants.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

int nx = 128, ny = 128, nz = 128, fpoints = 19, gpoints = 15, nsteps = 1000;
float tau = 0.505, cssq = 1.0f / 3.0f, omega = 1.0f / tau, sharp_c = 0.45f, sigma = 0.1f;

float *d_f, *d_g, *d_w, *d_w_g, *d_cix, *d_ciy, *d_ciz;
float *d_grad_fix, *d_grad_fiy, *d_grad_fiz, *d_mod_grad;
float *d_normx, *d_normy, *d_normz, *d_indicator;
float *d_curvature, *d_ffx, *d_ffy, *d_ffz;
float *d_ux, *d_uy, *d_uz, *d_pxx, *d_pyy, *d_pzz;
float *d_pxy, *d_pxz, *d_pyz, *d_rho, *d_phi;

float *h_rho = (float *)malloc(nx * ny * nz * sizeof(float));
float *h_pxx = (float *)malloc(nx * ny * nz * sizeof(float));
float *h_pyy = (float *)malloc(nx * ny * nz * sizeof(float));
float *h_pzz = (float *)malloc(nx * ny * nz * sizeof(float));
float *h_pxy = (float *)malloc(nx * ny * nz * sizeof(float));
float *h_pxz = (float *)malloc(nx * ny * nz * sizeof(float));
float *h_pyz = (float *)malloc(nx * ny * nz * sizeof(float));

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

void initializeConstants() {
    size_t size = nx * ny * nz * sizeof(float);            
    size_t f_size = nx * ny * nz * fpoints * sizeof(float); 
    size_t g_size = nx * ny * nz * gpoints * sizeof(float); 

    for (int i = 0; i < nx * ny * nz; i++) {
        h_pxx[i] = 1.0f;
        h_pyy[i] = 1.0f;
        h_pzz[i] = 1.0f;
        h_pxy[i] = 1.0f;
        h_pxz[i] = 1.0f;
        h_pyz[i] = 1.0f;
    }   

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
    cudaMalloc((void **)&d_mod_grad, size);
    cudaMalloc((void **)&d_pxx, size);
    cudaMalloc((void **)&d_pyy, size);
    cudaMalloc((void **)&d_pzz, size);
    cudaMalloc((void **)&d_pxy, size);
    cudaMalloc((void **)&d_pxz, size);
    cudaMalloc((void **)&d_pyz, size);

    cudaMalloc((void **)&d_f, f_size);
    cudaMalloc((void **)&d_g, g_size);
    cudaMalloc((void **)&d_w, fpoints * sizeof(float));
    cudaMalloc((void **)&d_w_g, gpoints * sizeof(float));
    cudaMalloc((void **)&d_cix, fpoints * sizeof(float));
    cudaMalloc((void **)&d_ciy, fpoints * sizeof(float));
    cudaMalloc((void **)&d_ciz, fpoints * sizeof(float));

    cudaMemset(d_ux, 0, size);
    cudaMemset(d_uy, 0, size);
    cudaMemset(d_uz, 0, size);
    cudaMemset(d_normx, 0, size);
    cudaMemset(d_normy, 0, size);
    cudaMemset(d_normz, 0, size);
    cudaMemset(d_curvature, 0, size);
    cudaMemset(d_indicator, 0, size);
    cudaMemset(d_ffx, 0, size);
    cudaMemset(d_ffy, 0, size);
    cudaMemset(d_ffz, 0, size);
    cudaMemset(d_mod_grad, 0, size);
    cudaMemset(d_f, 0, f_size);
    cudaMemset(d_g, 0, g_size);

    cudaMemcpy(d_pxx, h_pxx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pyy, h_pyy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pzz, h_pzz, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxy, h_pxy, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pxz, h_pxz, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pyz, h_pyz, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, sizeof(w), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_g, w_g, sizeof(w_g), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cix, cix, sizeof(cix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ciy, ciy, sizeof(ciy), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ciz, ciz, sizeof(ciz), cudaMemcpyHostToDevice);

    free(h_pxx);
    free(h_pyy);
    free(h_pzz);
    free(h_pxy);
    free(h_pxz);
    free(h_pyz);

}

