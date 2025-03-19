#include "kernels.cuh"

__global__ void initTensor(
    float * __restrict__ pxx,
    float * __restrict__ pyy,
    float * __restrict__ pzz,
    float * __restrict__ pxy,
    float * __restrict__ pxz,
    float * __restrict__ pyz,
    float * __restrict__ rho,
    const int NX, const int NY, const int NZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float val = 1.0f;
    pxx[idx3D] = val; pyy[idx3D] = val; pzz[idx3D] = val;
    pxy[idx3D] = val; pxz[idx3D] = val; pyz[idx3D] = val;
    rho[idx3D] = val;
}

__global__ void initPhase(
    float * __restrict__ phi, 
    const int NX, const int NY, const int NZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ || i == 0 || i == NX-1 || j == 0 || j == NY-1 || k == 0 || k == NZ-1) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float bubble_radius = 20.0f * NX / 150.0f;

    float dx = i - NX * 0.5f;
    float dy = j - NY * 0.5f;
    float dz = k - NZ * 0.5f;
    float Ri = sqrt((dx * dx) / 4.0f + dy * dy + dz * dz);

    float phi_val = 0.5f + 0.5f * tanh(2.0f * (bubble_radius - Ri) / 3.0f);

    phi[idx3D] = phi_val;
}

__global__ void initDist(
    const float * __restrict__ rho, 
    const float * __restrict__ phi, 
    float * __restrict__ f,
    float * __restrict__ g,
    int NX, int NY, int NZ
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= NX || j >= NY || k >= NZ) return;

    int idx3D = inline3D(i,j,k,NX,NY);

    float rho_val = rho[idx3D];
    float phi_val = phi[idx3D];

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int idx4D = inline4D(i,j,k,l,NX,NY,NZ);
        f[idx4D] = W[l] * rho_val;
    }

    #pragma unroll 19
    for (int l = 0; l < NLINKS; ++l) {
        int idx4D = inline4D(i,j,k,l,NX,NY,NZ);
        g[idx4D] = W[l] * phi_val;
    }
}