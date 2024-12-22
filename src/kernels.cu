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

    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))
    #define IDX4D(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        phi[IDX3D(i,j,k)] = g[IDX4D(i,j,k,0)] + g[IDX4D(i,j,k,1)] + g[IDX4D(i,j,k,2)] +
                            g[IDX4D(i,j,k,3)] + g[IDX4D(i,j,k,4)] + g[IDX4D(i,j,k,5)] +
                            g[IDX4D(i,j,k,6)] + g[IDX4D(i,j,k,7)] + g[IDX4D(i,j,k,8)] +
                            g[IDX4D(i,j,k,9)] + g[IDX4D(i,j,k,10)] + g[IDX4D(i,j,k,11)] +
                            g[IDX4D(i,j,k,12)] + g[IDX4D(i,j,k,13)] + g[IDX4D(i,j,k,14)];
    }

}

__global__ void gradCalc(
    float *phi, float *mod_grad, float *normx, float *normy,
    float *normz, float *indicator, float *w, const float *cix,
    const float *ciy, const float *ciz, int fpoints,
    int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {

        float grad_fix = 0, grad_fiy = 0, grad_fiz = 0;
        for (int l = 0; l < fpoints; ++l) {
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
        __syncthreads()
        mod_grad[IDX3D(i,j,k)] = sqrt(pow(grad_fix,2) + pow(grad_fiy,2) + pow(grad_fiz,2));
        normx[IDX3D(i,j,k)] = grad_fix / (mod_grad[IDX3D(i,j,k)] + 1e-9);
        normy[IDX3D(i,j,k)] = grad_fiy / (mod_grad[IDX3D(i,j,k)] + 1e-9);
        normz[IDX3D(i,j,k)] = grad_fiz / (mod_grad[IDX3D(i,j,k)] + 1e-9);
        indicator[IDX3D(i,j,k)] = sqrt(pow(grad_fix,2) + pow(grad_fiy,2) + pow(grad_fiz,2));

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

    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {

        curvature[IDX3D(i,j,k)] = 0;
        for (int l = 0; l < fpoints; ++l) {
            curvature[IDX3D(i,j,k)] -= 3 * w[l] *
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
        __syncthreads()
        ffx[IDX3D(i,j,k)] = sigma * curvature[IDX3D(i,j,k)] * normx[IDX3D(i,j,k)] * indicator[IDX3D(i,j,k)];
        ffy[IDX3D(i,j,k)] = sigma * curvature[IDX3D(i,j,k)] * normy[IDX3D(i,j,k)] * indicator[IDX3D(i,j,k)];
        ffz[IDX3D(i,j,k)] = sigma * curvature[IDX3D(i,j,k)] * normz[IDX3D(i,j,k)] * indicator[IDX3D(i,j,k)];

    }

}

__global__ void momentiCalc(
    float *ux, float *uy, float *uz, float *rho,
    float *ffx, float *ffy, float *ffz, float *w, float *f,
    const float *cix, const float *ciy, const float *ciz,
    float *pxx, float *pyy, float *pzz,
    float *pxy, float *pxz, float *pyz,
    float cssq, int nx, int ny, int nz,
    int fpoints, float *fneq
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))
    #define IDX4D(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))
    
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {

        ux[IDX3D(i,j,k)] = (
            f[IDX4D(i,j,k,1)] - f[IDX4D(i,j,k,2)] + f[IDX4D(i,j,k,7)] - f[IDX4D(i,j,k,8)] + f[IDX4D(i,j,k,9)] - 
            f[IDX4D(i,j,k,10)] + f[IDX4D(i,j,k,13)] - f[IDX4D(i,j,k,14)] + f[IDX4D(i,j,k,15)] - f[IDX4D(i,j,k,16)] 
        ) / rho[IDX3D(i,j,k)] + ffx[IDX3D(i,j,k)] * 0.5 / rho[IDX3D(i,j,k)];
        uy[IDX3D(i,j,k)] = (
            f[IDX4D(i,j,k,3)] - f[IDX4D(i,j,k,4)] + f[IDX4D(i,j,k,7)] - f[IDX4D(i,j,k,8)] + f[IDX4D(i,j,k,11)] - 
            f[IDX4D(i,j,k,12)] - f[IDX4D(i,j,k,13)] + f[IDX4D(i,j,k,14)] + f[IDX4D(i,j,k,17)] - f[IDX4D(i,j,k,18)] 
        ) / rho[IDX3D(i,j,k)] + ffy[IDX3D(i,j,k)] * 0.5 / rho[IDX3D(i,j,k)];
        uz[IDX3D(i,j,k)] = (
            f[IDX4D(i,j,k,5)] - f[IDX4D(i,j,k,6)] + f[IDX4D(i,j,k,9)] - f[IDX4D(i,j,k,10)] + f[IDX4D(i,j,k,11)] - 
            f[IDX4D(i,j,k,12)] - f[IDX4D(i,j,k,15)] + f[IDX4D(i,j,k,16)] - f[IDX4D(i,j,k,17)] + f[IDX4D(i,j,k,18)]
        ) / rho[IDX3D(i,j,k)] + ffz[IDX3D(i,j,k)] * 0.5 / rho[IDX3D(i,j,k)];
        __syncthreads()
        float uu = 0.5 * (pow(ux[IDX3D(i,j,k)],2) + pow(uy[IDX3D(i,j,k)],2) + pow(uz[IDX3D(i,j,k)],2)) / cssq;
        rho[IDX3D(i,j,k)] = f[IDX4D(i,j,k,0)] + f[IDX4D(i,j,k,1)] + f[IDX4D(i,j,k,2)] +
                            f[IDX4D(i,j,k,3)] + f[IDX4D(i,j,k,4)] + f[IDX4D(i,j,k,5)] +
                            f[IDX4D(i,j,k,6)] + f[IDX4D(i,j,k,7)] + f[IDX4D(i,j,k,8)] +
                            f[IDX4D(i,j,k,9)] + f[IDX4D(i,j,k,10)] + f[IDX4D(i,j,k,11)] +
                            f[IDX4D(i,j,k,12)] + f[IDX4D(i,j,k,13)] + f[IDX4D(i,j,k,14)] +
                            f[IDX4D(i,j,k,15)] + f[IDX4D(i,j,k,16)] + f[IDX4D(i,j,k,17)] +
                            f[IDX4D(i,j,k,18)]; 
        __syncthreads()
        for (int l = 0; l < fpoints; ++l) {
            float udotc = (ux[IDX3D(i,j,k)] * cix[l] + uy[IDX3D(i,j,k)] * ciy[l] + uz[IDX3D(i,j,k)] * ciz[l]) / cssq;
            float HeF = (w[l] * (rho[IDX3D(i,j,k)] + rho[IDX3D(i,j,k)] * (udotc + 0.5 * pow(udotc,2) - uu)))
                    * ((cix[l] - ux[IDX3D(i,j,k)]) * ffx[IDX3D(i,j,k)] +
                        (ciy[l] - uy[IDX3D(i,j,k)]) * ffy[IDX3D(i,j,k)] +
                        (ciz[l] - uz[IDX3D(i,j,k)]) * ffz[IDX3D(i,j,k)]
                    ) / (rho[IDX3D(i,j,k)] * cssq);
            float feq = w[l] * (rho[IDX3D(i,j,k)] + rho[IDX3D(i,j,k)] * (udotc + 0.5 * pow(udotc,2) - uu)) - 0.5 * HeF;
            fneq[l] = f[IDX4D(i,j,k,l)] - feq;
        }
        __syncthreads()
        pxx[IDX3D(i,j,k)] = fneq[1] + fneq[2] + fneq[7] + fneq[8] + fneq[9] + fneq[10] + fneq[13] + fneq[14] + fneq[15] + fneq[16];
        pyy[IDX3D(i,j,k)] = fneq[3] + fneq[4] + fneq[7] + fneq[8] + fneq[11] + fneq[12] + fneq[13] + fneq[14] + fneq[17] + fneq[18];
        pzz[IDX3D(i,j,k)] = fneq[5] + fneq[6] + fneq[9] + fneq[10] + fneq[11] + fneq[12] + fneq[15] + fneq[16] + fneq[17] + fneq[18];
        pxy[IDX3D(i,j,k)] = fneq[7] + fneq[8] - fneq[13] - fneq[14];
        pxz[IDX3D(i,j,k)] = fneq[9] + fneq[10] - fneq[15] - fneq[16];
        pyz[IDX3D(i,j,k)] = fneq[11] + fneq[12] - fneq[17] - fneq[18];

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
    int nx, int ny, int nz
) {     
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))
    #define IDX4D(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {

        float uu = 0.5 * (pow(ux[IDX3D(i,j,k)],2) + pow(uy[IDX3D(i,j,k)],2) + pow(uz[IDX3D(i,j,k)],2)) / cssq;
        __syncthreads()
        for (int l = 0; l < fpoints; ++l) {
            float udotc = (ux[IDX3D(i,j,k)] * cix[l] + uy[IDX3D(i,j,k)] * ciy[l] + uz[IDX3D(i,j,k)] * ciz[l]) / cssq;
            float feq = w[l] * (rho[IDX3D(i,j,k)] + rho[IDX3D(i,j,k)] * (udotc + 0.5 * pow(udotc, 2) - uu));
            float HeF = 0.5 * (w[l] * (rho[IDX3D(i,j,k)] + rho[IDX3D(i,j,k)] * (udotc + 0.5 * pow(udotc, 2) - uu)))
                    * ((cix[l] - ux[IDX3D(i,j,k)]) * ffx[IDX3D(i,j,k)] +
                        (ciy[l] - uy[IDX3D(i,j,k)]) * ffy[IDX3D(i,j,k)] +
                        (ciz[l] - uz[IDX3D(i,j,k)]) * ffz[IDX3D(i,j,k)]
                    ) / (rho[IDX3D(i,j,k)] * cssq);
            float singlefneq = (cix[l] * cix[l] - cssq) * pxx[IDX3D(i,j,k)] +
                        (ciy[l] * ciy[l] - cssq) * pyy[IDX3D(i,j,k)] +
                        (ciz[l] * ciz[l] - cssq) * pzz[IDX3D(i,j,k)] +
                        2 * cix[l] * ciy[l] * pxy[IDX3D(i,j,k)] +
                        2 * cix[l] * ciz[l] * pxz[IDX3D(i,j,k)] +
                        2 * ciy[l] * ciz[l] * pyz[IDX3D(i,j,k)];
            f[IDX4D(i + static_cast<int>(cix[l]),
                    j + static_cast<int>(ciy[l]),
                    k + static_cast<int>(ciz[l]),
                    l)] = feq + (1 - omega) * (w[l] / (2 * pow(cssq, 2))) * singlefneq + HeF;
        }
        __syncthreads()
        for (int l = 0; l < gpoints; ++l) {
            float udotc = (ux[IDX3D(i,j,k)] * cix[l] + uy[IDX3D(i,j,k)] * ciy[l] + uz[IDX3D(i,j,k)] * ciz[l]) / cssq;
            float feq = w_g[l] * phi[IDX3D(i,j,k)] * (1 + udotc);
            float Hi = sharp_c * phi[IDX3D(i,j,k)] * (1 - phi[IDX3D(i,j,k)]) *
                (cix[l] * normx[IDX3D(i,j,k)] +
                 ciy[l] * normy[IDX3D(i,j,k)] +
                 ciz[l] * normz[IDX3D(i,j,k)]);
            g[IDX4D(i,j,k,l)] = feq + w_g[l] * Hi;
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

    #define IDX4D(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    for (int l = 0; l < gpoints; ++l) {
        g[IDX4D(i,j,k,l)] = g[IDX4D(i + static_cast<int>(cix[l]),
                                    j + static_cast<int>(ciy[l]),
                                    k + static_cast<int>(ciz[l]),
                                    l)];
    }
}

__global__ void boundaryConditions(
    float *f, float *g, float *rho, float *phi, float *w, float *w_g,
    const float *cix, const float *ciy, const float *ciz,
    int fpoints, int gpoints, int nx, int ny, int nz
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))
    #define IDX4D(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))

    if (i == 0 || i == nx-1 || j == 0 || j == ny-1 || k == 0 || k == ny-1) {
        for (int l = 0; l < fpoints; ++l) {
            if (i + static_cast<int>(cix[l]) >= 0 && j + static_cast<int>(ciy[l]) >= 0 && k + static_cast<int>(ciz[l]) >= 0) {
                f[IDX4D(i + static_cast<int>(cix[l]),
                        j + static_cast<int>(ciy[l]),
                        k + static_cast<int>(ciz[l]),
                        l)] = rho[IDX3D(i,j,k)] * w[l];
            }
        }
        for (int l = 0; l < gpoints; ++l) {
            if (i + static_cast<int>(cix[l]) >= 0 && j + static_cast<int>(ciy[l]) >= 0 && k + static_cast<int>(ciz[l]) >= 0) {
                g[IDX4D(i + static_cast<int>(cix[l]),
                        j + static_cast<int>(ciy[l]),
                        k + static_cast<int>(ciz[l]),
                        l)] = phi[IDX3D(i,j,k)] * w_g[l];
            }
        }
    }
    __syncthreads()
    phi[IDX3D(i,j,0)] = phi[IDX3D(i,j,1)];
    phi[IDX3D(i,j,nz-1)] = phi[IDX3D(i,j,nz-2)];
    phi[IDX3D(i,0,k)] = phi[IDX3D(i,1,k)];
    phi[IDX3D(i,ny-1,k)] = phi[IDX3D(i,ny-2,k)];
    phi[IDX3D(0,j,k)] = phi[IDX3D(1,j,k)];
    phi[IDX3D(nx-1,j,k)] = phi[IDX3D(nx-2,j,k)];

}
