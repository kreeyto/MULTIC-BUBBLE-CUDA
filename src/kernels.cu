#include "kernels.cuh"
#include "var.cuh"
#include <math.h>

#include "precision.cuh"

#ifdef FD3Q19
  #define FPOINTS 19
#elif defined(FD3Q27)
  #define FPOINTS 27
#endif

#ifdef PD3Q15
    #define GPOINTS 15
#elif defined(PD3Q19)
    #define GPOINTS 19
#elif defined(PD3Q27)
    #define GPOINTS 27
#endif

__global__ void phiCalc(
    dfloat * __restrict__ phi,
    const dfloat * __restrict__ g,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((k)*(nx*ny) + (j)*(nx) + (i))
    #define IDX4D(i,j,k,l) ((l)*(nx*ny*nz) + (k)*(nx*ny) + (j)*(nx) + (i))

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    dfloat sum = 0.0;
    #pragma unroll
    for (int l = 0; l < GPOINTS; ++l) {
        sum += g[IDX4D(i,j,k,l)];
    }
    phi[IDX3D(i,j,k)] = sum;
}

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
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((k) * (nx * ny) + (j) * nx + (i))

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    dfloat grad_fix = 0.0, grad_fiy = 0.0, grad_fiz = 0.0;
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        int ii = i + __float2int_rn(cix[l]);
        int jj = j + __float2int_rn(ciy[l]);
        int kk = k + __float2int_rn(ciz[l]);
        int offset = kk * (nx * ny) + jj * nx + ii;
        dfloat val = phi[offset];
        dfloat coef = 3.0 * w[l];
        grad_fix += coef * cix[l] * val;
        grad_fiy += coef * ciy[l] * val;
        grad_fiz += coef * ciz[l] * val;
    }
    dfloat gmag = sqrt(grad_fix * grad_fix + grad_fiy * grad_fiy + grad_fiz * grad_fiz);
    mod_grad[IDX3D(i,j,k)] = gmag;
    dfloat inv_gmag = 1.0 / (gmag + 1e-9);
    normx[IDX3D(i,j,k)] = grad_fix * inv_gmag;
    normy[IDX3D(i,j,k)] = grad_fiy * inv_gmag;
    normz[IDX3D(i,j,k)] = grad_fiz * inv_gmag;
    indicator[IDX3D(i,j,k)] = gmag;
}

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
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((k) * (nx * ny) + (j) * nx + (i))

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int offset = IDX3D(i,j,k);

    dfloat curv = 0.0;
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        int ii = i + __float2int_rn(cix[l]);
        int jj = j + __float2int_rn(ciy[l]);
        int kk = k + __float2int_rn(ciz[l]);
        int offsetN = kk * (nx * ny) + jj * nx + ii;
        curv -= 3.0 * w[l] * (
            cix[l] * normx[offsetN] +
            ciy[l] * normy[offsetN] +
            ciz[l] * normz[offsetN]
        );
    }
    curvature[offset] = curv;
    dfloat normx_ = normx[offset];
    dfloat normy_ = normy[offset];
    dfloat normz_ = normz[offset];
    dfloat ind_ = indicator[offset];
    dfloat mult = sigma * curv * ind_;
    ffx[offset] = mult * normx_;
    ffy[offset] = mult * normy_;
    ffz[offset] = mult * normz_;
}

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
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((k)*(nx*ny) + (j)*nx + (i))
    #define IDX4D(i,j,k,l) ((l)*(nx*ny*nz) + (k)*(nx*ny) + (j)*nx + (i))

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int idx = IDX3D(i,j,k);

    dfloat fVal[FPOINTS];
    dfloat fneq[FPOINTS];
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        fVal[l] = f[IDX4D(i,j,k,l)];
    }
    dfloat rhoOld = rho[idx];
    #ifdef FD3Q19
        dfloat sumUx = ( fVal[1] - fVal[2] + fVal[7] - fVal[8] + fVal[9] - fVal[10] + fVal[13] - fVal[14] + fVal[15] - fVal[16] );
        dfloat sumUy = ( fVal[3] - fVal[4] + fVal[7] - fVal[8] + fVal[11] - fVal[12] - fVal[13] + fVal[14] + fVal[17] - fVal[18] );
        dfloat sumUz = ( fVal[5] - fVal[6] + fVal[9] - fVal[10] + fVal[11] - fVal[12] - fVal[15] + fVal[16] - fVal[17] + fVal[18] );
    #endif
    dfloat invRhoOld = 1.0 / rhoOld;
    dfloat halfFx = 0.5 * ffx[idx] * invRhoOld;
    dfloat halfFy = 0.5 * ffy[idx] * invRhoOld;
    dfloat halfFz = 0.5 * ffz[idx] * invRhoOld;
    dfloat uxVal = sumUx * invRhoOld + halfFx;
    dfloat uyVal = sumUy * invRhoOld + halfFy;
    dfloat uzVal = sumUz * invRhoOld + halfFz;
    dfloat rhoNew = 0.0;
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        rhoNew += fVal[l];
    }
    rho[idx] = rhoNew;
    dfloat uu = 0.5 * (uxVal*uxVal + uyVal*uyVal + uzVal*uzVal) / cssq;
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        dfloat udotc = (uxVal * cix[l] + uyVal * ciy[l] + uzVal * ciz[l]) / cssq;
        dfloat eqBase = rhoNew * (udotc + 0.5 * udotc * udotc - uu);
        dfloat feq = w[l] * (rhoNew + eqBase); 
        dfloat HeF = w[l] * (rhoNew + eqBase)
                     * ((cix[l] - uxVal)*ffx[idx] 
                      + (ciy[l] - uyVal)*ffy[idx]
                      + (ciz[l] - uzVal)*ffz[idx])
                     / (rhoNew * cssq);
        feq -= 0.5 * HeF;
        fneq[l] = fVal[l] - feq;
    }
    #ifdef FD3Q19
        pxx[idx] = fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
        pyy[idx] = fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
        pzz[idx] = fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
        pxy[idx] = fneq[7] + fneq[8] - fneq[13] - fneq[14];
        pxz[idx] = fneq[9] + fneq[10] - fneq[15] - fneq[16];
        pyz[idx] = fneq[11] + fneq[12] - fneq[17] - fneq[18];
    #endif
    ux[idx] = uxVal;
    uy[idx] = uyVal;
    uz[idx] = uzVal;
}

__global__ void collisionCalc(
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
    int nx, int ny, int nz,
    dfloat * __restrict__ f_coll
) {     
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    #define IDX3D(i,j,k) ((k) * (nx * ny) + (j) * (nx) + (i))
    #define IDX4D(i,j,k,l) ((l) * (nx * ny * nz) + (k) * (nx * ny) + (j) * (nx) + (i))

    int idx3D = IDX3D(i, j, k);
    dfloat ux_ijk = ux[idx3D];
    dfloat uy_ijk = uy[idx3D];
    dfloat uz_ijk = uz[idx3D];
    dfloat rho_ijk = rho[idx3D];
    dfloat phi_ijk = phi[idx3D];
    dfloat uu = 0.5 * (ux_ijk * ux_ijk + uy_ijk * uy_ijk + uz_ijk * uz_ijk) / cssq;

    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        dfloat cix_l = cix[l];
        dfloat ciy_l = ciy[l];
        dfloat ciz_l = ciz[l];

        dfloat udotc = (ux_ijk * cix_l + uy_ijk * ciy_l + uz_ijk * ciz_l) / cssq;
        dfloat feq = w[l] * (rho_ijk + rho_ijk * (udotc + 0.5 * (udotc * udotc) - uu));
        dfloat HeF = 0.5 * feq *
                ((cix_l - ux_ijk) * ffx[idx3D] +
                 (ciy_l - uy_ijk) * ffy[idx3D] +
                 (ciz_l - uz_ijk) * ffz[idx3D]) / (rho_ijk * cssq);
        dfloat fneq = (cix_l * cix_l - cssq) * pxx[idx3D] +
                      (ciy_l * ciy_l - cssq) * pyy[idx3D] +
                      (ciz_l * ciz_l - cssq) * pzz[idx3D] +
                      2 * cix_l * ciy_l * pxy[idx3D] +
                      2 * cix_l * ciz_l * pxz[idx3D] +
                      2 * ciy_l * ciz_l * pyz[idx3D];
        f_coll[IDX4D(i,j,k,l)] = feq + (1.0 - omega) * (w[l] / (2.0 * cssq * cssq)) * fneq + HeF;
    }
    #pragma unroll
    for (int l = 0; l < GPOINTS; ++l) {
        dfloat cix_l = cix[l];
        dfloat ciy_l = ciy[l];
        dfloat ciz_l = ciz[l];
        dfloat udotc = (ux_ijk * cix_l + uy_ijk * ciy_l + uz_ijk * ciz_l) / cssq;
        dfloat feq = w_g[l] * phi_ijk * (1 + udotc);
        dfloat Hi = sharp_c * phi_ijk * (1 - phi_ijk) *
            (cix_l * normx[idx3D] +
             ciy_l * normy[idx3D] + 
             ciz_l * normz[idx3D]);
        g[IDX4D(i,j,k,l)] = feq + w_g[l] * Hi;
    }
}

__global__ void streamingCalcNew(
    const dfloat * __restrict__ f_coll,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    int nx, int ny, int nz,
    dfloat * __restrict__ f 
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;
    // idk if we check this too

    int NxNy = nx * ny;
    int NxNyNz = NxNy * nz;
    int dstBase = k * NxNy + j * nx + i; 

    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        int src_i = (i - __float2int_rn(cix[l]) + nx) % nx;
        int src_j = (j - __float2int_rn(ciy[l]) + ny) % ny;
        int src_k = (k - __float2int_rn(ciz[l]) + nz) % nz;
        int srcBase = src_k * NxNy + src_j * nx + src_i;
        int dstIdx = l * NxNyNz + dstBase;
        int srcIdx = l * NxNyNz + srcBase;
        f[dstIdx] = f_coll[srcIdx];
    }
}

__global__ void streamingCalc(
    const dfloat * __restrict__ g_in,
    dfloat * __restrict__ g_out,
    const dfloat * __restrict__ cix,
    const dfloat * __restrict__ ciy,
    const dfloat * __restrict__ ciz,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int NxNy   = nx * ny;
    int NxNyNz = NxNy * nz;
    int dstBase = k * NxNy + j * nx + i; 

    #pragma unroll
    for (int l = 0; l < GPOINTS; ++l) {
        int di = __float2int_rn(cix[l]);
        int dj = __float2int_rn(ciy[l]);
        int dk = __float2int_rn(ciz[l]);
        int src_i = (i - di + nx) % nx;
        int src_j = (j - dj + ny) % ny;
        int src_k = (k - dk + nz) % nz;
        int srcBase = src_k * NxNy + src_j * nx + src_i;
        int dstIdx = l * NxNyNz + dstBase;
        int srcIdx = l * NxNyNz + srcBase;
        g_out[dstIdx] = g_in[srcIdx];
    }
}

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
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int idx = IDX3D(i,j,k);

    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) {
        #pragma unroll
        for (int l = 0; l < FPOINTS; ++l) {
            int nb_i = i + __float2int_rn(cix[l]);
            int nb_j = j + __float2int_rn(ciy[l]);
            int nb_k = k + __float2int_rn(ciz[l]);
            if (nb_i >= 0 && nb_i < nx && nb_j >= 0 && nb_j < ny && nb_k >= 0 && nb_k < nz) {
                f[IDX4D(nb_i,nb_j,nb_k,l)] = rho[idx] * w[l];
            }
        }
        #pragma unroll
        for (int l = 0; l < GPOINTS; ++l) {
            int nb_i = i + __float2int_rn(cix[l]);
            int nb_j = j + __float2int_rn(ciy[l]);
            int nb_k = k + __float2int_rn(ciz[l]);

            if (nb_i >= 0 && nb_i < nx && nb_j >= 0 && nb_j < ny && nb_k >= 0 && nb_k < nz) {
                g[IDX4D(nb_i,nb_j,nb_k,l)] = phi[idx] * w_g[l];
            }
        }
    }
}

__global__ void boundaryConditions(
    dfloat * __restrict__ phi, 
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    #define IDX3D(i,j,k) ((k) * (nx * ny) + (j) * nx + (i))

    if (k == 0) phi[IDX3D(i,j,0)] = phi[IDX3D(i,j,1)];
    if (k == nz-1 && nz > 1) phi[IDX3D(i,j,nz-1)] = phi[IDX3D(i,j,nz-2)];

    if (j == 0) phi[IDX3D(i,0,k)] = phi[IDX3D(i,1,k)];
    if (j == ny-1 && ny > 1) phi[IDX3D(i,ny-1,k)] = phi[IDX3D(i,ny-2,k)];

    if (i == 0) phi[IDX3D(0,j,k)] = phi[IDX3D(1,j,k)];
    if (i == nx-1 && nx > 1) phi[IDX3D(nx-1,j,k)] = phi[IDX3D(nx-2,j,k)];
}
