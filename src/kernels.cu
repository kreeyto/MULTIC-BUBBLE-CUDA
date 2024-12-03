#include "kernels.cuh"
#include "var.cuh"
#include <math.h>

__global__ void phiCalc(
    float *phi, float *g, int gpoints,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + nx * (j + ny * k);
    #define F_IDX(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        for (int l = 0; l < gpoints; l++) {
            phi[idx] += g[F_IDX(i, j, k, l)];
        }            
    }
}

__global__ void gradCalc(
    float *phi, float *mod_grad, float *normx, float *normy, 
    float *normz, float *indicator, float *w, const float *cix, 
    const float *ciy, const float *ciz, int fpoints,
    int nx, int ny, int nz, float grad_fix, float grad_fiy, float grad_fiz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + nx * (j + ny * k);
    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        grad_fix = 0; 
        grad_fiy = 0;
        grad_fiz = 0;
        for (int l = 0; l < fpoints; l++) {
            grad_fix += 3 * w[l] * cix[l] * phi[IDX3D(i + static_cast<int>(cix[l]),
                                                        j + static_cast<int>(ciy[l]),
                                                        k + static_cast<int>(ciz[l]))];
            grad_fiy += 3 * w[l] * ciy[l] * phi[IDX3D(i + static_cast<int>(cix[l]),
                                                        j + static_cast<int>(ciy[l]),
                                                        k + static_cast<int>(ciz[l]))];
            grad_fiz += 3 * w[l] * ciz[l] * phi[IDX3D(i + static_cast<int>(cix[l]),
                                                        j + static_cast<int>(ciy[l]),
                                                        k + static_cast<int>(ciz[l]))];
        }
        mod_grad[idx] = sqrt(pow(grad_fix,2) + pow(grad_fiy,2) + pow(grad_fiz,2));
        normx[idx] = grad_fix / (mod_grad[idx] + 1e-9);
        normy[idx] = grad_fiy / (mod_grad[idx] + 1e-9);
        normz[idx] = grad_fiz / (mod_grad[idx] + 1e-9);
        indicator[idx] = sqrt(pow(grad_fix,2) + pow(grad_fiy,2) + pow(grad_fiz,2));
    }
}

__global__ void curvatureCalc(
    float *curvature, float *indicator, float *w,
    const float *cix, const float *ciy, const float *ciz,
    float *normx, float *normy, float *normz, 
    float *ffx, float *ffy, float *ffz, float sigma,
    int fpoints, int nx, int ny, int nz
    
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + nx * (j + ny * k);
    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        curvature[idx] = 0;
        for (int l = 0; l < fpoints; l++) {
            curvature[idx] -= 3 * w[l] * 
               (cix[l] * normx[IDX3D(i + static_cast<int>(cix[l]),
                                    j + static_cast<int>(ciy[l]),
                                    k + static_cast<int>(ciz[l]))] + 
                ciy[l] * normy[IDX3D(i + static_cast<int>(cix[l]),
                                    j + static_cast<int>(ciy[l]),
                                    k + static_cast<int>(ciz[l]))] + 
                ciz[l] * normz[IDX3D(i + static_cast<int>(cix[l]),
                                    j + static_cast<int>(ciy[l]),
                                    k + static_cast<int>(ciz[l]))] 
            );
        }
        ffx[idx] = sigma * curvature[idx] * normx[idx] * indicator[idx];
        ffy[idx] = sigma * curvature[idx] * normy[idx] * indicator[idx];
        ffz[idx] = sigma * curvature[idx] * normz[idx] * indicator[idx];
    }
}

__global__ void momentsCalc(
    float *ux, float *uy, float *uz, float *rho,
    float *ffx, float *ffy, float *ffz, float *w, float *f,
    const float *cix, const float *ciy, const float *ciz,
    float *pxx, float *pyy, float *pzz,
    float *pxy, float *pxz, float *pyz,
    float cssq, int nx, int ny, int nz,
    int fpoints, float uu, float udotc, float HeF, float feq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + nx * (j + ny * k);
    #define F_IDX(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    float fneq[19];

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        ux[idx] = (
            (f[F_IDX(i,j,k,1)] + f[F_IDX(i,j,k,15)] + f[F_IDX(i,j,k,9)] + f[F_IDX(i,j,k,7)] + f[F_IDX(i,j,k,13)]) -
            (f[F_IDX(i,j,k,2)] + f[F_IDX(i,j,k,10)] + f[F_IDX(i,j,k,16)] + f[F_IDX(i,j,k,14)] + f[F_IDX(i,j,k,7)])
        ) / rho[idx] +
        ffx[idx] * 0.5 / rho[idx];

        uy[idx] = (
            (f[F_IDX(i,j,k,3)] + f[F_IDX(i,j,k,7)] + f[F_IDX(i,j,k,14)] + f[F_IDX(i,j,k,17)] + f[F_IDX(i,j,k,11)]) -
            (f[F_IDX(i,j,k,4)] + f[F_IDX(i,j,k,13)] + f[F_IDX(i,j,k,8)] + f[F_IDX(i,j,k,12)] + f[F_IDX(i,j,k,18)])
        ) / rho[idx] +
        ffy[idx] * 0.5 / rho[idx];

        uz[idx] = (
            (f[F_IDX(i,j,k,6)] + f[F_IDX(i,j,k,15)] + f[F_IDX(i,j,k,10)] + f[F_IDX(i,j,k,17)] + f[F_IDX(i,j,k,12)]) -
            (f[F_IDX(i,j,k,5)] + f[F_IDX(i,j,k,9)] + f[F_IDX(i,j,k,16)] + f[F_IDX(i,j,k,11)] + f[F_IDX(i,j,k,18)])
        ) / rho[idx] +
        ffz[idx] * 0.5 / rho[idx];

        uu = 0.5 * (pow(ux[idx],2) + pow(uy[idx],2) + pow(uz[idx],2)) / cssq;

        for (int l = 0; l < fpoints; l++) {
            rho[idx] += f[F_IDX(i, j, k, l)];
        }   

        for (int l = 0; l < fpoints; l++) {
            udotc = (ux[idx] * cix[l] + uy[idx] * ciy[l] + uz[idx] * ciz[l]) / cssq;
            HeF = (w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc,2) - uu)))
                    * ((cix[l] - ux[idx]) * ffx[idx] + 
                        (ciy[l] - uy[idx]) * ffy[idx] + 
                        (ciz[l] - uz[idx]) * ffz[idx] 
                    ) / (rho[idx] * cssq);
            feq = w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc,2) - uu)) - 0.5 * HeF;
            fneq[l] = f[F_IDX(i,j,k,l)] - feq;
        }

        pxx[idx] = fneq[2] + fneq[3] + fneq[8] + fneq[9] + fneq[10] + fneq[11] + fneq[14] + fneq[15] + fneq[16] + fneq[17];
        pyy[idx] = fneq[4] + fneq[5] + fneq[8] + fneq[9] + fneq[12] + fneq[13] + fneq[14] + fneq[15] + fneq[18] + fneq[19];
        pzz[idx] = fneq[6] + fneq[7] + fneq[10] + fneq[11] + fneq[12] + fneq[13] + fneq[16] + fneq[17] + fneq[18] + fneq[19];
        pxy[idx] = fneq[8] + fneq[9] - fneq[14] - fneq[15];
        pxz[idx] = fneq[10] + fneq[11] - fneq[16] - fneq[17];
        pyz[idx] = fneq[12] + fneq[13] - fneq[18] - fneq[19];
    }
}

__global__ void collisionCalc(
    float *ux, float *uy, float *uz, float *w, float *w_g,
    const float *cix, const float *ciy, const float *ciz,
    float *normx, float *normy, float *normz,
    float *ffx, float *ffy, float *ffz,
    float *rho, float *phi, float *f, float *g, 
    float *pxx, float *pyy, float *pzz, float *pxy, float *pxz, float *pyz, 
    float cssq, float omega, float sharp_c, int fpoints, int gpoints,
    int nx, int ny, int nz, float uu, float udotc, float HeF, float feq, float Hi
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + nx * (j + ny * k);
    #define F_IDX(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {

        uu = 0.5 * (pow(ux[idx], 2) + pow(uy[idx], 2) + pow(uz[idx], 2)) / cssq;

        for (int l = 0; l < fpoints; l++) {
            udotc = (ux[idx] * cix[l] + uy[idx] * ciy[l] + uz[idx] * ciz[l]) / cssq;
            feq = w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc, 2) - uu));
            HeF = 0.5 * (w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc, 2) - uu)))
                    * ((cix[l] - ux[idx]) * ffx[idx] + 
                        (ciy[l] - uy[idx]) * ffy[idx] + 
                        (ciz[l] - uz[idx]) * ffz[idx] 
                    ) / (rho[idx] * cssq);
            float fneq = (cix[l] * cix[l] - cssq) * pxx[idx] + 
                        (ciy[l] * ciy[l] - cssq) * pyy[idx] + 
                        (ciz[l] * ciz[l] - cssq) * pzz[idx] + 
                        2 * cix[l] * ciy[l] * pxy[idx] + 
                        2 * cix[l] * ciz[l] * pxz[idx] + 
                        2 * ciy[l] * ciz[l] * pyz[idx]; // float fneq ~= fneq[l]
            f[F_IDX(i + static_cast<int>(cix[l]),
                    j + static_cast<int>(ciy[l]),
                    k + static_cast<int>(ciz[l]),
                    l)] = feq + (1 - omega) * (w[l] / (2 * pow(cssq, 2))) * fneq + HeF;
        }

        for (int l = 0; l < gpoints; l++) {
            udotc = (ux[idx] * cix[l] + uy[idx] * ciy[l] + uz[idx] * ciz[l]) / cssq;
            feq = w_g[l] * phi[idx] * (1 + udotc);
            Hi = sharp_c * phi[idx] * (1 - phi[idx]) * (cix[l] * normx[idx] + ciy[l] * normy[idx] + ciz[l] * normz[idx]); 
            g[F_IDX(i,j,k,l)] = feq + w_g[l] * Hi;
        }

    }
}

__global__ void streamingCalc(
    float *g, const float *cix, const float *ciy, const float *ciz,
    int nx, int ny, int nz, int gpoints
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define F_IDX(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        for (int l = 0; l < gpoints; l++) {
            g[F_IDX(i,j,k,l)] = g[F_IDX(i + static_cast<int>(cix[l]),
                                        j + static_cast<int>(ciy[l]),
                                        k + static_cast<int>(ciz[l]),
                                        l)];
        }
    }
}

__global__ void fgBoundaryCalc(
    float *f, float *g, float *rho, float *phi, float *w, float *w_g,
    const float *cix, const float *ciy, const float *ciz,
    int fpoints, int gpoints, int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + nx * (j + ny * k);
    #define F_IDX(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == ny-1) {
        for (int l = 0; l < fpoints; l++) {
            if (i + static_cast<int>(cix[l]) >= 0 && j + static_cast<int>(ciy[l]) >= 0 && k + static_cast<int>(ciz[l]) >= 0) {
                f[F_IDX(i + static_cast<int>(cix[l]),
                        j + static_cast<int>(ciy[l]),
                        k + static_cast<int>(ciz[l]),
                        l)] = rho[idx] * w[l];
            }
        }
        for (int l = 0; l < gpoints; l++) {
            if (i + static_cast<int>(cix[l]) >= 0 && j + static_cast<int>(ciy[l]) >= 0 && k + static_cast<int>(ciz[l]) >= 0) {
                g[F_IDX(i + static_cast<int>(cix[l]),
                        j + static_cast<int>(ciy[l]),
                        k + static_cast<int>(ciz[l]),
                        l)] = phi[idx] * w_g[l];
            }
        }
    }
}

__global__ void boundaryConditions(
    float *phi, int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))

    if (j == 0) {
        phi[IDX3D(i,j,k)] = phi[IDX3D(i,j+1,k)]; 
    }
    if (j == ny - 1) {
        phi[IDX3D(i,j,k)] = phi[IDX3D(i,j-1,k)]; 
    }
    if (k == 0) {
        phi[IDX3D(i,j,k)] = phi[IDX3D(i,j,k+1)]; 
    }
    if (k == nz - 1) {
        phi[IDX3D(i,j,k)] = phi[IDX3D(i,j,k-1)]; 
    }
    if (i == 0) {
        phi[IDX3D(i,j,k)] = phi[IDX3D(i+1,j,k)];
    }
    if (i == nx - 1) {
        phi[IDX3D(i,j,k)] = phi[IDX3D(i-1,j,k)];
    }
}

