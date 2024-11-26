#include "kernels.cuh"
#include "utils.cuh"
#include "constants.cuh"
#include "errorDef.cuh"

int main() {
    initializeConstants();

    std::vector<float> phi(nx * ny * nz, 0.0f);
    std::vector<float> rho(nx * ny * nz, 1.0f); 
    const std::vector<float> w = {
        1.0f / 3.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
        1.0f / 18.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
        1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
    };
    const std::vector<float> w_g = {
        2.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
        1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f,
        1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f
    };
    std::vector<float> f(nx * ny * nz * fpoints, 0.0f);
    std::vector<float> g(nx * ny * nz * gpoints, 0.0f);
    
    computeInitialCPU(phi, rho, w, w_g, f, g, nx, ny, nz, fpoints, gpoints);

    checkCudaErrors(cudaMemcpy(d_phi, phi.data(), nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rho, rho.data(), nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w, w.data(), fpoints * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w_g, w_g.data(), gpoints * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_f, f.data(), nx * ny * nz * fpoints * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_g, g.data(), nx * ny * nz * gpoints * sizeof(float), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    gpuMomCollisionStreamBCS<<<numBlocks, threadsPerBlock>>> (
        d_f, d_g, d_phi, d_rho, d_w, d_w_g, d_cix, d_ciy, d_ciz, 
        d_mod_grad, d_normx, d_normy, d_normz, d_indicator,
        d_curvature, d_ffx, d_ffy, d_ffz, d_ux, d_uy, d_uz,
        d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz,
        nx, ny, nz, fpoints, gpoints, 
        sigma, cssq, omega, sharp_c, nsteps
    );
    getLastCudaError("CUDA Error after kernel launch");

    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Error after cudaDeviceSynchronize");

    float *pointers[] = {d_f, d_g, d_phi, d_rho, d_w, d_w_g, d_cix, d_ciy, d_ciz, 
                     d_mod_grad, d_normx, d_normy, d_normz, d_indicator,
                     d_curvature, d_ffx, d_ffy, d_ffz, d_ux, d_uy, d_uz,
                     d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz};
    freeMemory(pointers, 27);  

    return 0;
}
