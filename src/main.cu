#include "kernels.cuh"
#include "utils.cuh"
#include "constants.cuh"
#include "errorDef.cuh"
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

int main() {
    initializeConstants();

    // initialize storage vars
    int stamp = 100;
    std::vector<float> phi_host(nx * ny * nz);
    std::vector<float> ux_host(nx * ny * nz);
    std::vector<float> uy_host(nx * ny * nz);
    std::vector<float> uz_host(nx * ny * nz);
    std::string output_dir = "../bin/simulation/";

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

    for (int t = 0; t < nsteps; t++) {

        std::cout << "Passo " << t << " de " << nsteps << " iniciado..." << std::endl;

        checkCudaErrors(cudaMemcpy(phi_host.data(), d_phi, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(ux_host.data(), d_ux, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(uy_host.data(), d_uy, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(uz_host.data(), d_uz, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

        phiCalc<<<numBlocks, threadsPerBlock>>> (
            d_phi, d_g, gpoints, nx, ny, nz
        );
        checkCudaErrors(cudaDeviceSynchronize());

        gradCalc<<<numBlocks, threadsPerBlock>>> (
            d_phi, d_mod_grad, d_normx, d_normy, d_normz, 
            d_indicator, d_w, d_cix, d_ciy, d_ciz, 
            fpoints, nx, ny, nz
        );
        checkCudaErrors(cudaDeviceSynchronize());

        curvatureCalc<<<numBlocks, threadsPerBlock>>> (
            d_curvature, d_indicator, d_w,
            d_cix, d_ciy, d_ciz,
            d_normx, d_normy, d_normz, 
            d_ffx, d_ffy, d_ffz, sigma,
            fpoints, nx, ny, nz
        );
        checkCudaErrors(cudaDeviceSynchronize());

        momentsCalc<<<numBlocks, threadsPerBlock>>> (
            d_ux, d_uy, d_uz, d_rho,
            d_ffx, d_ffy, d_ffz, d_w, d_f,
            d_cix, d_ciy, d_ciz,
            d_pxx, d_pyy, d_pzz,
            d_pxy, d_pxz, d_pyz,
            cssq, nx, ny, nz,
            fpoints
        );
        checkCudaErrors(cudaDeviceSynchronize());

        collisionCalc<<<numBlocks, threadsPerBlock>>> (
            d_ux, d_uy, d_uz, d_w, d_w_g,
            d_cix, d_ciy, d_ciz,
            d_normx, d_normy, d_normz,
            d_ffx, d_ffy, d_ffz,
            d_rho, d_phi, d_f, d_g, 
            d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz, 
            cssq, omega, sharp_c, fpoints, gpoints,
            nx, ny, nz
        );
        checkCudaErrors(cudaDeviceSynchronize());

        streamingCalc<<<numBlocks, threadsPerBlock>>> (
            d_g, d_cix, d_ciy, d_ciz, nx, ny, nz, gpoints
        );
        checkCudaErrors(cudaDeviceSynchronize());

        fgBoundaryCalc<<<numBlocks, threadsPerBlock>>> (
            d_f, d_g, d_rho, d_phi, d_w, d_w_g,
            d_cix, d_ciy, d_ciz, fpoints, gpoints, nx, ny, nz
        );
        checkCudaErrors(cudaDeviceSynchronize());

        boundaryConditions<<<numBlocks, threadsPerBlock>>> (
            d_phi, nx, ny, nz
        );
        checkCudaErrors(cudaDeviceSynchronize());

        // save data
        if (t % stamp == 0) {
            std::ofstream file_phi(output_dir + "phi_" + std::to_string(t) + ".bin", std::ios::binary);
            std::ofstream file_ux(output_dir + "ux_" + std::to_string(t) + ".bin", std::ios::binary);
            std::ofstream file_uy(output_dir + "uy_" + std::to_string(t) + ".bin", std::ios::binary);
            std::ofstream file_uz(output_dir + "uz_" + std::to_string(t) + ".bin", std::ios::binary);

            file_phi.write(reinterpret_cast<const char*>(phi_host.data()), phi_host.size() * sizeof(float));
            file_ux.write(reinterpret_cast<const char*>(ux_host.data()), ux_host.size() * sizeof(float));
            file_uy.write(reinterpret_cast<const char*>(uy_host.data()), uy_host.size() * sizeof(float));
            file_uz.write(reinterpret_cast<const char*>(uz_host.data()), uz_host.size() * sizeof(float));

            file_phi.close();
            file_ux.close();
            file_uy.close();
            file_uz.close();

            std::cout << "Passo " << t << ": Dados salvos em " << output_dir << std::endl;
        }
        
    }

    float *pointers[] = {d_f, d_g, d_phi, d_rho, d_w, d_w_g, d_cix, d_ciy, d_ciz, 
                     d_mod_grad, d_normx, d_normy, d_normz, d_indicator,
                     d_curvature, d_ffx, d_ffy, d_ffz, d_ux, d_uy, d_uz,
                     d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz};
    freeMemory(pointers, 27);  

    return 0;
}
