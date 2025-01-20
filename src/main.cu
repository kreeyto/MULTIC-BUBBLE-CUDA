#include "kernels.cuh"
#include "auxFunctions.cuh"
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
    std::string base_dir;   
    #ifdef _WIN32
        base_dir = "..\\";
    #else
        base_dir = "../";
    #endif
    std::string model_dir = base_dir + fluid_model + "_" + phase_model + "/";
    std::string sim_dir = model_dir + id + "/";
    #ifdef _WIN32
        std::string mkdir_command = "mkdir \"" + sim_dir + "\"";
    #else
        std::string mkdir_command = "mkdir -p \"" + sim_dir + "\"";
    #endif
    int ret = system(mkdir_command.c_str());
    (void)ret; 
    std::string info_file = sim_dir + id + "_info.txt";

    // ========================= //
    int stamp = 1, nsteps = 10;
    // ========================= //
    initializeVars();
    generateSimulationInfoFile(info_file, nx, ny, nz, stamp, nsteps, tau, id, fluid_model);

    std::vector<dfloat> f(nx * ny * nz * fpoints, 0.0);
    std::vector<dfloat> g(nx * ny * nz * gpoints, 0.0);
    std::vector<dfloat> phi(nx * ny * nz, 0.0);
    std::vector<dfloat> rho(nx * ny * nz, 1.0); 

    // =========================== FLUID WEIGHTS =========================== //
    const std::vector<dfloat> w = {
        1.0 / 3.0, 
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
    // ===================================================================== //

    // ======================== PHASE FIELD WEIGHTS ======================== //
    const std::vector<dfloat> w_g = {
        2.0 / 9.0, 
        1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
        1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0
    };
    // ===================================================================== //

    computeInitialCPU(phi, rho, w, w_g, f, g, nx, ny, nz, fpoints, gpoints, res);
    checkCudaErrors(cudaMemcpy(d_f, f.data(), nx * ny * nz * fpoints * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_g, g.data(), nx * ny * nz * gpoints * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_phi, phi.data(), nx * ny * nz * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_rho, rho.data(), nx * ny * nz * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w, w.data(), fpoints * sizeof(dfloat), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_w_g, w_g.data(), gpoints * sizeof(dfloat), cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    std::vector<dfloat> phi_host(nx * ny * nz);
    std::vector<dfloat> rho_host(nx * ny * nz);

    for (int t = 0; t < nsteps; ++t) {

        std::cout << "Passo " << t << " de " << nsteps << " iniciado..." << std::endl;

        phiCalc<<<numBlocks, threadsPerBlock>>>(
            d_phi, d_g, 
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        gradCalc<<<numBlocks, threadsPerBlock>>>(
            d_phi, d_w, d_cix,
            d_grad_fix, d_grad_fiy, d_grad_fiz,
            d_ciy, d_ciz, fpoints,
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        normCalc<<<numBlocks, threadsPerBlock>>>(
            d_grad_fix, d_grad_fiy, d_grad_fiz,
            d_mod_grad, d_normx, d_normy, d_normz,
            d_indicator, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        curvatureCalc<<<numBlocks, threadsPerBlock>>>(
            d_curvature, d_indicator, d_w,
            d_cix, d_ciy, d_ciz,
            d_normx, d_normy, d_normz,
            d_ffx, d_ffy, d_ffz, sigma,
            fpoints, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        forceCalc<<<numBlocks, threadsPerBlock>>>(
            d_ffx, d_ffy, d_ffz,
            sigma, d_curvature, d_indicator,
            d_normx, d_normy, d_normz,
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        macroCalc<<<numBlocks, threadsPerBlock>>>(
            d_ux, d_uy, d_uz, d_f,
            d_ffx, d_ffy, d_ffz, d_rho,
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        uuCalc<<<numBlocks, threadsPerBlock>>>(
            d_ux, d_uy, d_uz, d_uu,
            cssq, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        rhoCalc<<<numBlocks, threadsPerBlock>>>(
            d_rho, d_f, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        momentiCalc<<<numBlocks, threadsPerBlock>>>(
            d_ux, d_uy, d_uz, d_w,
            d_cix, d_ciy, d_ciz,
            d_ffx, d_ffy, d_ffz,
            d_uu, d_rho, d_fneq,
            d_f, cssq, nx, ny, nz, 
            fpoints
        );
        cudaDeviceSynchronize();
        
        tensorCalc<<<numBlocks, threadsPerBlock>>>(
            d_pxx, d_pyy, d_pzz,
            d_pxy, d_pxz, d_pyz,
            nx, ny, nz, d_fneq
        );
        cudaDeviceSynchronize();
        
        uuCalc<<<numBlocks, threadsPerBlock>>>(
            d_ux, d_uy, d_uz, d_uu,
            cssq, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        fCalc<<<numBlocks, threadsPerBlock>>>(
            d_ux, d_uy, d_uz,
            d_cix, d_ciy, d_ciz,
            d_w, d_rho, d_uu,
            d_ffx, d_ffy, d_ffz,
            d_pxx, d_pyy, d_pzz,
            d_pxy, d_pxz, d_pyz,
            d_f, omega, cssq,
            fpoints, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        gCalc<<<numBlocks, threadsPerBlock>>>(
            d_ux, d_uy, d_uz,
            d_cix, d_ciy, d_ciz,
            d_w_g, d_phi, d_g,
            d_normx, d_normy, d_normz,
            cssq, gpoints, nx, ny, nz,
            sharp_c
        );
        cudaDeviceSynchronize();
        
        streamingCalc<<<numBlocks, threadsPerBlock>>>(
            d_g, d_cix, d_ciy, d_ciz,
            nx, ny, nz, gpoints
        );
        cudaDeviceSynchronize();
        
        fgBoundary<<<numBlocks, threadsPerBlock>>>(
            d_f, d_g, d_rho, d_phi, d_w, d_w_g,
            d_cix, d_ciy, d_ciz,
            fpoints, gpoints, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        boundaryConditions<<<numBlocks, threadsPerBlock>>>(
            d_phi, nx, ny, nz
        );
        cudaDeviceSynchronize();

        if (t % stamp == 0) {
            
            // phi 
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
                          d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz, d_fneq,
                          d_grad_fix, d_grad_fiy, d_grad_fiz, d_uu
                        };
    freeMemory(pointers, 32);  

    return 0;
}
