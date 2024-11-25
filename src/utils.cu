#include "utils.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>
#include <iostream>

void allocateMemory(float *&h_rho, float *&h_phi, float *&d_rho, float *&d_phi) {
    size_t size = nx * ny * nz * sizeof(float);
    h_rho = (float *)malloc(size);
    h_phi = (float *)malloc(size);
    cudaMalloc((void **)&d_rho, size);
    cudaMalloc((void **)&d_phi, size);
}

void freeMemory(float *h_rho, float *h_phi, float *d_rho, float *d_phi) {
    free(h_rho);
    free(h_phi);
    cudaFree(d_rho);
    cudaFree(d_phi);
}

void copyToDevice(const float *h_rho, const float *h_phi, float *d_rho, float *d_phi) {
    size_t size = nx * ny * nz * sizeof(float);
    cudaMemcpy(d_rho, h_rho, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, h_phi, size, cudaMemcpyHostToDevice);
}

void initializeHostArrays(float *h_rho, float *h_phi) {
    for (int i = 0; i < nx * ny * nz; i++) {
        h_rho[i] = 1.0f;
        h_phi[i] = 0.0f;
    }
}
