#include "kernels.cuh"
#include "constants.cuh"
#include <math.h>

__global__ void gpuMomCollisionStreamBCS(
    float *f, float *g, float *phi, float *rho, const float *w, const float *w_g,
    const float *cix, const float *ciy, const float *ciz,
    float *mod_grad, float *normx, float *normy, float *normz, float *indicator,
    float *curvature, float *ffx, float *ffy, float *ffz, 
    float *ux, float *uy, float *uz,
    float *pxx, float *pyy, float *pzz, float *pxy, float *pxz, float *pyz,
    int nx, int ny, int nz, int fpoints, int gpoints, 
    float sigma, float cssq, float omega, float sharp_c, int nsteps
) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + nx * (j + ny * k);
    #define F_IDX(i,j,k,l) ((i) + nx * ((j) + ny * ((k) + nz * (l))))
    #define IDX3D(i,j,k) ((i) + nx * ((j) + ny * (k)))

    float fneq[19];

    for (int t = 0; t < nsteps; t++) {

        if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
            for (int l = 0; l < gpoints; l++) {
                phi[idx] += g[F_IDX(i, j, k, l)];
            }            
        }

        if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
            float grad_fix = 0, grad_fiy = 0, grad_fiz = 0;
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

            float uu = 0.5 * (pow(ux[idx],2) + pow(uy[idx],2) + pow(uz[idx],2)) / cssq;

            for (int l = 0; l < fpoints; l++) {
                rho[idx] += f[F_IDX(i, j, k, l)];
            }   

            for (int l = 0; l < fpoints; l++) {
                float udotc = (ux[idx] * cix[l] + uy[idx] * ciy[l] + uz[idx] * ciz[l]) / cssq;
                float HeF = (w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc,2) - uu)))
                        * ((cix[l] - ux[idx]) * ffx[idx] + 
                            (ciy[l] - uy[idx]) * ffy[idx] + 
                            (ciz[l] - uz[idx]) * ffz[idx] 
                        ) / (rho[idx] * cssq);
                float feq = w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc,2) - uu)) - 0.5 * HeF;
                fneq[l] = f[F_IDX(i,j,k,l)] - feq;
            }

            pxx[idx] = fneq[2] + fneq[3] + fneq[8] + fneq[9] + fneq[10] + fneq[11] + fneq[14] + fneq[15] + fneq[16] + fneq[17];
            pyy[idx] = fneq[4] + fneq[5] + fneq[8] + fneq[9] + fneq[12] + fneq[13] + fneq[14] + fneq[15] + fneq[18] + fneq[19];
            pzz[idx] = fneq[6] + fneq[7] + fneq[10] + fneq[11] + fneq[12] + fneq[13] + fneq[16] + fneq[17] + fneq[18] + fneq[19];
            pxy[idx] = fneq[8] + fneq[9] - fneq[14] - fneq[15];
            pxz[idx] = fneq[10] + fneq[11] - fneq[16] - fneq[17];
            pyz[idx] = fneq[12] + fneq[13] - fneq[18] - fneq[19];
        }

        if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {

            float uu = 0.5 * (pow(ux[idx], 2) + pow(uy[idx], 2) + pow(uz[idx], 2)) / cssq;

            for (int l = 0; l < fpoints; l++) {
                float udotc = (ux[idx] * cix[l] + uy[idx] * ciy[l] + uz[idx] * ciz[l]) / cssq;
                float feq = w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc, 2) - uu));
                float HeF = 0.5 * (w[l] * (rho[idx] + rho[idx] * (udotc + 0.5 * pow(udotc, 2) - uu)))
                        * ((cix[l] - ux[idx]) * ffx[idx] + 
                            (ciy[l] - uy[idx]) * ffy[idx] + 
                            (ciz[l] - uz[idx]) * ffz[idx] 
                        ) / (rho[idx] * cssq);
                float fneq = (cix[l] * cix[l] - cssq) * pxx[idx] + 
                            (ciy[l] * ciy[l] - cssq) * pyy[idx] + 
                            (ciz[l] * ciz[l] - cssq) * pzz[idx] + 
                            2 * cix[l] * ciy[l] * pxy[idx] + 
                            2 * cix[l] * ciz[l] * pxz[idx] + 
                            2 * ciy[l] * ciz[l] * pyz[idx];
                f[F_IDX(i + static_cast<int>(cix[l]),
                        j + static_cast<int>(ciy[l]),
                        k + static_cast<int>(ciz[l]),
                        l)] = feq + (1 - omega) * (w[l] / (2 * pow(cssq, 2))) * fneq + HeF;
            }

            for (int l = 0; l < gpoints; l++) {
                float udotc = (ux[idx] * cix[l] + uy[idx] * ciy[l] + uz[idx] * ciz[l]) / cssq;
                float feq = w_g[l] * phi[idx] * (1 + udotc);
                float Hi = sharp_c * phi[idx] * (1 - phi[idx]) * (cix[l] * normx[idx] + ciy[l] * normy[idx] + ciz[l] * normz[idx]); 
                g[F_IDX(i,j,k,l)] = feq + w_g[l] * Hi;
            }

        }

        if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
            for (int l = 0; l < gpoints; l++) {
                g[F_IDX(i,j,k,l)] = g[F_IDX(i + static_cast<int>(cix[l]),
                                            j + static_cast<int>(ciy[l]),
                                            k + static_cast<int>(ciz[l]),
                                            l)];
            }
        }

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

        if (j == 0) {
            phi[IDX3D(i, j, k)] = phi[IDX3D(i, j + 1, k)]; // phi(:,1,:) = phi(:,2,:);
        }
        if (j == ny - 1) {
            phi[IDX3D(i, j, k)] = phi[IDX3D(i, j - 1, k)]; // phi(:,ny,:) = phi(:,ny-1,:);
        }
        if (k == 0) {
            phi[IDX3D(i, j, k)] = phi[IDX3D(i, j, k + 1)]; // phi(:,:,1) = phi(:,:,2);
        }
        if (k == nz - 1) {
            phi[IDX3D(i, j, k)] = phi[IDX3D(i, j, k - 1)]; // phi(:,:,nz) = phi(:,:,nz-1);
        }
        if (i == 0) {
            phi[IDX3D(i, j, k)] = phi[IDX3D(i + 1, j, k)]; // phi(1,:,:) = phi(2,:,:);
        }
        if (i == nx - 1) {
            phi[IDX3D(i, j, k)] = phi[IDX3D(i - 1, j, k)]; // phi(nx,:,:) = phi(nx-1,:,:);
        }

    } // end of time loop
}