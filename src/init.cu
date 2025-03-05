#include "kernels.cuh"

__global__ void initTensor(
    float * __restrict__ pxx,
    float * __restrict__ pyy,
    float * __restrict__ pzz,
    float * __restrict__ pxy,
    float * __restrict__ pxz,
    float * __restrict__ pyz,
    float * __restrict__ rho,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx3D = inline3D(i,j,k,nx,ny);

    float val = 1.0f;
    pxx[idx3D] = val; pyy[idx3D] = val; pzz[idx3D] = val;
    pxy[idx3D] = val; pxz[idx3D] = val; pyz[idx3D] = val;
    rho[idx3D] = val;
}

__global__ void initPhase(
    float * __restrict__ phi, 
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz || i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int idx3D = inline3D(i,j,k,nx,ny);

    float bubble_radius = 20.0f * nx / 150.0f;

    float dx = i - nx * 0.5f;
    float dy = j - ny * 0.5f;
    float dz = k - nz * 0.5f;
    float Ri = sqrt((dx * dx) / 4.0f + dy * dy + dz * dz);

    float phi_val = 0.5f + 0.5f * tanh(2.0f * (bubble_radius - Ri) / 3.0f);

    phi[idx3D] = phi_val;
}

__global__ void initDist(
    const float * __restrict__ rho, 
    const float * __restrict__ phi, 
    float * __restrict__ f,
    float * __restrict__ g,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx3D = inline3D(i,j,k,nx,ny);

    float rho_val = rho[idx3D];
    float phi_val = phi[idx3D];

    for (int l = 0; l < FPOINTS; ++l) {
        int idx4D = inline4D(i,j,k,l,nx,ny,nz);
        f[idx4D] = (W[l] * rho_val) - W[l];
    }

    for (int l = 0; l < GPOINTS; ++l) {
        int idx4D = inline4D(i,j,k,l,nx,ny,nz);
        g[idx4D] = W_G[l] * phi_val;
    }
}