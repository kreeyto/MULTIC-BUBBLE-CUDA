#include "kernels.cuh"
#include "utils.cuh"
#include "var.cuh"
#include "errorDef.cuh"
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "precision.cuh"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Erro: Uso: " << argv[0] << " <fluid_model> <phase_model> <id>" << std::endl;
        return 1;
    }

    std::string fluid_model = argv[1];
    std::string phase_model = argv[2];
    std::string id = argv[3];

    std::string base_dir = "/home/breno/Desktop/Bubble-GPU/bin/";
    std::string model_dir = base_dir + fluid_model + "_" + phase_model + "/";
    std::string sim_dir = model_dir + id + "/";

    int ret = system(("mkdir -p " + sim_dir).c_str());
    (void)ret; 

    std::string info_file = sim_dir + id + "_info.txt";

    initializeVars();

    int stamp = 1, nsteps = 50;
    std::vector<dfloat> phi_host(nx * ny * nz, 0.0f);

    generateSimulationInfoFile(info_file, nx, ny, nz, stamp, nsteps, tau, id, fluid_model);

    std::vector<dfloat> f(nx * ny * nz * fpoints, 0.0f);
    std::vector<dfloat> g(nx * ny * nz * gpoints, 0.0f);
    std::vector<dfloat> phi(nx * ny * nz, 0.0f);
    std::vector<dfloat> rho(nx * ny * nz, 1.0f); 
    
    #ifdef FD3Q19
        const std::vector<dfloat> w = {
            1.0f / 3.0f, 
            1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
            1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
        };
    #elif defined(FD3Q27)
        const std::vector <dfloat> w = {
            8.0f / 27.0f,
            2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 
            1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 
            1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f
        };
    #endif

    #ifdef PD3Q15
        const std::vector<dfloat> w_g = {
            2.0f / 9.0f, 
            1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
            1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f, 1.0f / 72.0f
        };
    #elif defined(PD3Q19)
        const std::vector <dfloat> w_g = {
            1.0f / 3.0f, 
            1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
            1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
        };
    #elif defined(PD3Q27)
        const std::vector <dfloat> w_g = {
            8.0f / 27.0f,
            2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 2.0f / 27.0f, 
            1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 1.0f / 54.0f, 
            1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f, 1.0f / 216.0f
        };
    #endif

    computeInitialCPU(phi, rho, w, w_g, f, g, nx, ny, nz, fpoints, gpoints, res);
    dfloat max_val = 0.0f;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                dfloat val = phi[i + nx*(j + ny*k)];
                if (val > max_val) max_val = val;
            }
        }
    }
    std::cout << "Valor maximo inicial de phi = " << max_val << std::endl;
    checkCudaErrors(cudaMemcpy(d_f, f.data(), nx * ny * nz * fpoints * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_g, g.data(), nx * ny * nz * gpoints * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_phi, phi.data(), nx * ny * nz * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rho, rho.data(), nx * ny * nz * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w, w.data(), fpoints * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w_g, w_g.data(), gpoints * sizeof(dfloat), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    for (int t = 0; t < nsteps; ++t) {

        std::cout << "Passo " << t << " de " << nsteps << " iniciado..." << std::endl;

        phiCalc<<<numBlocks, threadsPerBlock>>> (
            d_phi, d_g, nx, ny, nz
        );
        getLastCudaError("Erro no kernel phiCalc");
        cudaDeviceSynchronize();

        gradCalc<<<numBlocks, threadsPerBlock>>> (
            d_phi, d_mod_grad, d_normx, d_normy, d_normz, 
            d_indicator, d_w, d_cix, d_ciy, d_ciz, 
            fpoints, nx, ny, nz
        );
        getLastCudaError("Erro no kernel gradCalc");
        cudaDeviceSynchronize();

        curvatureCalc<<<numBlocks, threadsPerBlock>>> (
            d_curvature, d_indicator, d_w,
            d_cix, d_ciy, d_ciz,
            d_normx, d_normy, d_normz, 
            d_ffx, d_ffy, d_ffz, sigma,
            fpoints, nx, ny, nz
        );
        getLastCudaError("Erro no kernel curvatureCalc");
        cudaDeviceSynchronize();

        momentiCalc<<<numBlocks, threadsPerBlock>>> (
            d_ux, d_uy, d_uz, d_rho,
            d_ffx, d_ffy, d_ffz, d_w, d_f,
            d_cix, d_ciy, d_ciz, 
            d_pxx, d_pyy, d_pzz,
            d_pxy, d_pxz, d_pyz,
            cssq, nx, ny, nz,
            fpoints, d_fneq
        );
        getLastCudaError("Erro no kernel momentiCalc");
        cudaDeviceSynchronize();

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
        getLastCudaError("Erro no kernel collisionCalc");
        cudaDeviceSynchronize();

        streamingCalc<<<numBlocks, threadsPerBlock>>> (
            d_g, d_cix, d_ciy, d_ciz, nx, ny, nz, gpoints
        );
        getLastCudaError("Erro no kernel streamingCalc");
        cudaDeviceSynchronize();

        boundaryConditions<<<numBlocks, threadsPerBlock>>> (
            d_f, d_g, d_rho, d_phi, d_w, d_w_g,
            d_cix, d_ciy, d_ciz, fpoints, gpoints, nx, ny, nz
        );
        getLastCudaError("Erro no kernel boundaryConditions");
        cudaDeviceSynchronize();

        if (t % stamp == 0) {

            std::ostringstream filename_phi;
            checkCudaErrors(cudaMemcpy(phi_host.data(), d_phi, nx * ny * nz * sizeof(dfloat), cudaMemcpyDeviceToHost));
            filename_phi << sim_dir << id << "_phi" << std::setw(6) << std::setfill('0') << t << ".bin";
            std::ofstream file_phi(filename_phi.str(), std::ios::binary);
            file_phi.write(reinterpret_cast<const char*>(phi_host.data()), phi_host.size() * sizeof(dfloat));
            file_phi.close();

            std::cout << "Passo " << t << ": Dados salvos em " << sim_dir << std::endl;

        }
        
    }

    dfloat *pointers[] = {d_f, d_g, d_phi, d_rho, d_w, d_w_g, d_cix, d_ciy, d_ciz, 
                     d_mod_grad, d_normx, d_normy, d_normz, d_indicator,
                     d_curvature, d_ffx, d_ffy, d_ffz, d_ux, d_uy, d_uz,
                     d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz, d_fneq
                     };
    freeMemory(pointers, 28);  

    return 0;
}
