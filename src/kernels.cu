#include "kernels.cuh"
#include "var.cuh"
#include <math.h>

#include "precision.cuh"

#ifdef FD3Q19
  #define FPOINTS 19
#endif

#ifdef PD3Q15
    #define GPOINTS 15
#endif

#define IDX3D(i,j,k)  ((i) + (j) * nx + (k) * nx * ny)
#define IDX4D(i,j,k,l) ((i) + (j) * nx + (k) * nx * ny + (l) * nx * ny * nz)

__global__ void phiCalc(
    dfloat * __restrict__ phi,
    const dfloat * __restrict__ g,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

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
    const int * __restrict__ cix,
    const int * __restrict__ ciy,
    const int * __restrict__ ciz,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    dfloat grad_fix = 0.0, grad_fiy = 0.0, grad_fiz = 0.0;
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        int ii = i + cix[l];
        int jj = j + ciy[l];
        int kk = k + ciz[l];
        int offset = ii + jj * nx + kk * nx * ny;
        dfloat val = phi[offset];
        dfloat coef = 3.0 * w[l];
        grad_fix += coef * cix[l] * val;
        grad_fiy += coef * ciy[l] * val;
        grad_fiz += coef * ciz[l] * val;
    }
    dfloat gmag_sq = grad_fix * grad_fix + grad_fiy * grad_fiy + grad_fiz * grad_fiz;
    dfloat inv_gmag = rsqrtf(gmag_sq + 1e-9);
    dfloat gmag = 1.0 / inv_gmag;  
    mod_grad[IDX3D(i,j,k)] = gmag;
    normx[IDX3D(i,j,k)] = grad_fix * inv_gmag;
    normy[IDX3D(i,j,k)] = grad_fiy * inv_gmag;
    normz[IDX3D(i,j,k)] = grad_fiz * inv_gmag;
    indicator[IDX3D(i,j,k)] = gmag;
}

__global__ void curvatureCalc(
    dfloat * __restrict__ curvature,
    const dfloat * __restrict__ indicator,
    const dfloat * __restrict__ w,
    const int * __restrict__ cix,
    const int * __restrict__ ciy,
    const int * __restrict__ ciz,
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

    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;

    int offset = IDX3D(i,j,k);

    dfloat normx_ = normx[offset];
    dfloat normy_ = normy[offset];
    dfloat normz_ = normz[offset];
    dfloat ind_ = indicator[offset];
    dfloat curv = 0.0;

    int baseIdx = i + j * nx + k * nx * ny;

    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        int offsetN = baseIdx + cix[l] + ciy[l] * nx + ciz[l] * nx * ny;
        dfloat normxN = normx[offsetN];
        dfloat normyN = normy[offsetN];
        dfloat normzN = normz[offsetN];
        dfloat coef = 3.0 * w[l];
        curv -= coef * (cix[l] * normxN + ciy[l] * normyN + ciz[l] * normzN);
    }
    dfloat mult = sigma * curv * ind_;
    curvature[offset] = curv;
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
    const int * __restrict__ cix,
    const int * __restrict__ ciy,
    const int * __restrict__ ciz,
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
    
    if (i >= nx || j >= ny || k >= nz) return;
    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == nz-1) return;
    
    int idx = IDX3D(i,j,k);
    
    dfloat fVal[FPOINTS];
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        fVal[l] = f[IDX4D(i,j,k,l)];
    }
    dfloat rhoOld   = rho[idx];
    dfloat ffx_val  = ffx[idx];
    dfloat ffy_val  = ffy[idx];
    dfloat ffz_val  = ffz[idx];
    #ifdef FD3Q19
        dfloat sumUx = fVal[1] - fVal[2] + fVal[7] - fVal[8] + fVal[9] - fVal[10] + fVal[13] - fVal[14] + fVal[15] - fVal[16];
        dfloat sumUy = fVal[3] - fVal[4] + fVal[7] - fVal[8] + fVal[11] - fVal[12] - fVal[13] + fVal[14] + fVal[17] - fVal[18];
        dfloat sumUz = fVal[5] - fVal[6] + fVal[9] - fVal[10] + fVal[11] - fVal[12] - fVal[15] + fVal[16] - fVal[17] + fVal[18];
    #endif
    dfloat invRhoOld = 1.0 / rhoOld;
    dfloat halfFx = 0.5 * ffx_val * invRhoOld;
    dfloat halfFy = 0.5 * ffy_val * invRhoOld;
    dfloat halfFz = 0.5 * ffz_val * invRhoOld;
    dfloat uxVal = sumUx * invRhoOld + halfFx;
    dfloat uyVal = sumUy * invRhoOld + halfFy;
    dfloat uzVal = sumUz * invRhoOld + halfFz;
    dfloat rhoNew = 0.0;
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        rhoNew += fVal[l];
    }
    rho[idx] = rhoNew;
    dfloat invCssq       = 1.0 / cssq;
    dfloat uu            = 0.5 * (uxVal*uxVal + uyVal*uyVal + uzVal*uzVal) * invCssq;
    dfloat invRhoNewCssq = 1.0 / (rhoNew * cssq);
    dfloat fneq[FPOINTS];
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        int cix_l = cix[l];
        int ciy_l = ciy[l];
        int ciz_l = ciz[l];
        dfloat udotc = (uxVal * cix_l + uyVal * ciy_l + uzVal * ciz_l) * invCssq;
        dfloat eqBase = rhoNew * (udotc + 0.5 * udotc * udotc - uu);
        dfloat common = w[l] * (rhoNew + eqBase);
        dfloat feq = common;
        dfloat HeF = common * ((cix_l - uxVal) * ffx_val +
                               (ciy_l - uyVal) * ffy_val +
                               (ciz_l - uzVal) * ffz_val) * invRhoNewCssq;
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
    const int    * __restrict__ cix,
    const int    * __restrict__ ciy,
    const int    * __restrict__ ciz,
    const dfloat * __restrict__ normx,
    const dfloat * __restrict__ normy,
    const dfloat * __restrict__ normz,
    const dfloat * __restrict__ ffx,
    const dfloat * __restrict__ ffy,
    const dfloat * __restrict__ ffz,
    const dfloat * __restrict__ rho,
    const dfloat * __restrict__ phi,
    const dfloat * __restrict__ f,
    dfloat       * __restrict__ g,
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
    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) return;

    int idx3D = IDX3D(i,j,k);
    int nxyz  = nx * ny * nz;

    dfloat ux_val    = ux[idx3D];
    dfloat uy_val    = uy[idx3D];
    dfloat uz_val    = uz[idx3D];
    dfloat rho_val   = rho[idx3D];
    dfloat phi_val   = phi[idx3D];
    dfloat ffx_val   = ffx[idx3D];
    dfloat ffy_val   = ffy[idx3D];
    dfloat ffz_val   = ffz[idx3D];
    dfloat pxx_val   = pxx[idx3D];
    dfloat pyy_val   = pyy[idx3D];
    dfloat pzz_val   = pzz[idx3D];
    dfloat pxy_val   = pxy[idx3D];
    dfloat pxz_val   = pxz[idx3D];
    dfloat pyz_val   = pyz[idx3D];
    dfloat normx_val = normx[idx3D];
    dfloat normy_val = normy[idx3D];
    dfloat normz_val = normz[idx3D];

    dfloat uu             = 0.5 * (ux_val * ux_val + uy_val * uy_val + uz_val * uz_val) / cssq;
    dfloat one_minus_omega = 1.0 - omega;
    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        dfloat cx = cix[l];
        dfloat cy = ciy[l];
        dfloat cz = ciz[l];
        dfloat udotc = (ux_val * cx + uy_val * cy + uz_val * cz) / cssq;
        dfloat feq    = w[l] * (rho_val + rho_val * (udotc + 0.5 * udotc * udotc - uu));
        dfloat HeF    = 0.5 * feq *
                        ((cx - ux_val) * ffx_val +
                         (cy - uy_val) * ffy_val +
                         (cz - uz_val) * ffz_val) / (rho_val * cssq);
        dfloat fneq   = (cx * cx - cssq) * pxx_val +
                        (cy * cy - cssq) * pyy_val +
                        (cz * cz - cssq) * pzz_val +
                        2 * cx * cy * pxy_val +
                        2 * cx * cz * pxz_val +
                        2 * cy * cz * pyz_val;
        f_coll[idx3D + l * nxyz] = feq + one_minus_omega * (w[l] / (2.0 * cssq * cssq)) * fneq + HeF;
    }
    #pragma unroll
    for (int l = 0; l < GPOINTS; ++l) {
        dfloat cx = cix[l];
        dfloat cy = ciy[l];
        dfloat cz = ciz[l];
        dfloat udotc = (ux_val * cx + uy_val * cy + uz_val * cz) / cssq;
        dfloat feq    = w_g[l] * phi_val * (1 + udotc);
        dfloat Hi     = sharp_c * phi_val * (1 - phi_val) *
                        (cx * normx_val + cy * normy_val + cz * normz_val);
        g[idx3D + l * nxyz] = feq + w_g[l] * Hi;
    }
}

__global__ void streamingCalcNew(
    const dfloat * __restrict__ f_coll,
    const int * __restrict__ cix,
    const int * __restrict__ ciy,
    const int * __restrict__ ciz,
    int nx, int ny, int nz,
    dfloat * __restrict__ f 
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int NxNy = nx * ny;
    int NxNyNz = NxNy * nz;
    int dstBase = i + j * nx + k * NxNy;

    #pragma unroll
    for (int l = 0; l < FPOINTS; ++l) {
        int src_i = (i - cix[l] + nx) % nx;
        int src_j = (j - ciy[l] + ny) % ny;
        int src_k = (k - ciz[l] + nz) % nz;
        int srcBase = src_i + src_j * nx + src_k * NxNy;
        int dstIdx = l * NxNyNz + dstBase;
        int srcIdx = l * NxNyNz + srcBase;
        f[dstIdx] = f_coll[srcIdx];
    }
}

__global__ void streamingCalc(
    const dfloat * __restrict__ g_in,
    dfloat * __restrict__ g_out,
    const int * __restrict__ cix,
    const int * __restrict__ ciy,
    const int * __restrict__ ciz,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int NxNy = nx * ny;
    int NxNyNz = NxNy * nz;
    int dstBase = i + j * nx + k * NxNy;

    #pragma unroll
    for (int l = 0; l < GPOINTS; ++l) {
        int di = cix[l];
        int dj = ciy[l];
        int dk = ciz[l];
        int src_i = (i - di + nx) % nx;
        int src_j = (j - dj + ny) % ny;
        int src_k = (k - dk + nz) % nz;
        int srcBase = src_i + src_j * nx + src_k * NxNy;
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
    const int * __restrict__ cix,
    const int * __restrict__ ciy,
    const int * __restrict__ ciz,
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
            int nb_i = i + cix[l];
            int nb_j = j + ciy[l];
            int nb_k = k + ciz[l];
            if (nb_i >= 0 && nb_i < nx && nb_j >= 0 && nb_j < ny && nb_k >= 0 && nb_k < nz) {
                f[IDX4D(nb_i,nb_j,nb_k,l)] = rho[idx] * w[l];
            }
        }
        #pragma unroll
        for (int l = 0; l < GPOINTS; ++l) {
            int nb_i = i + cix[l];
            int nb_j = j + ciy[l];
            int nb_k = k + ciz[l];

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

    if (k == 0) phi[IDX3D(i,j,0)] = phi[IDX3D(i,j,1)];
    if (k == nz-1 && nz > 1) phi[IDX3D(i,j,nz-1)] = phi[IDX3D(i,j,nz-2)];

    if (j == 0) phi[IDX3D(i,0,k)] = phi[IDX3D(i,1,k)];
    if (j == ny-1 && ny > 1) phi[IDX3D(i,ny-1,k)] = phi[IDX3D(i,ny-2,k)];

    if (i == 0) phi[IDX3D(0,j,k)] = phi[IDX3D(1,j,k)];
    if (i == nx-1 && nx > 1) phi[IDX3D(nx-1,j,k)] = phi[IDX3D(nx-2,j,k)];
}
