#include "kernels.cuh"
#include "auxFunctions.cuh"
#include "var.cuh"
#include "errorDef.cuh"
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

#include "precision.h"

int main(int argc, char* argv[]) {
    auto start_time = chrono::high_resolution_clock::now();
    if (argc < 4) {
        cerr << "Erro: Uso: " << argv[0] << " F<fluid velocity set> P<phase field velocity set> <id>" << endl;
        return 1;
    }
    string fluid_model = argv[1];
    string phase_model = argv[2];
    string id = argv[3];

    string base_dir;   
    #ifdef _WIN32
        base_dir = "..\\";
    #else
        base_dir = "../";
    #endif
    string model_dir = base_dir + "bin/" + fluid_model + "_" + phase_model + "/";
    string sim_dir = model_dir + id + "/";
    #ifdef _WIN32
        string mkdir_command = "mkdir \"" + sim_dir + "\"";
    #else
        string mkdir_command = "mkdir -p \"" + sim_dir + "\"";
    #endif
    int ret = system(mkdir_command.c_str());
    (void)ret;

    // ============================================================================================================================================================= //

    // ========================= //
    int stamp = 10, nsteps = 100;
    // ========================= //
    initializeVars();

    string info_file = sim_dir + id + "_info.txt";
    dfloat h_tau;
    cudaMemcpyFromSymbol(&h_tau, TAU, sizeof(dfloat), 0, cudaMemcpyDeviceToHost);
    generateSimulationInfoFile(info_file, nx, ny, nz, stamp, nsteps, h_tau, id, fluid_model);

    dim3 threadsPerBlock(8,8,8);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // ================== INIT ================== //
    initPhase<<<numBlocks, threadsPerBlock>>> (
        d_phi, res, nx, ny, nz
    );
    cudaDeviceSynchronize();

    initDist<<<numBlocks, threadsPerBlock>>> (
        d_rho, d_phi, d_f, d_g, nx, ny, nz
    );
    cudaDeviceSynchronize();
    // ========================================= //

    vector<dfloat> phi_host(nx * ny * nz);
    //vector<dfloat> ux_host(nx * ny * nz);
    //vector<dfloat> uy_host(nx * ny * nz);
    //vector<dfloat> uz_host(nx * ny * nz);

    for (int t = 0; t < nsteps; ++t) {
        cout << "Passo " << t << " de " << nsteps << " iniciado..." << endl;

        phiCalc<<<numBlocks, threadsPerBlock>>> (
            d_phi, d_g, nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        gradCalc<<<numBlocks, threadsPerBlock>>> (
            d_phi, d_mod_grad, d_normx, d_normy, d_normz, 
            d_indicator, 
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        curvatureCalc<<<numBlocks, threadsPerBlock>>> (
            d_curvature, d_indicator,
            d_normx, d_normy, d_normz, 
            d_ffx, d_ffy, d_ffz,
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        momentiCalc<<<numBlocks, threadsPerBlock>>> (
            d_ux, d_uy, d_uz, d_rho,
            d_ffx, d_ffy, d_ffz, d_f,
            d_pxx, d_pyy, d_pzz,
            d_pxy, d_pxz, d_pyz,
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        
        collisionCalc<<<numBlocks, threadsPerBlock>>> (
            d_ux, d_uy, d_uz, 
            d_normx, d_normy, d_normz,
            d_ffx, d_ffy, d_ffz,
            d_rho, d_phi, d_g, 
            d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz, 
            nx, ny, nz, d_f_coll 
        );
        cudaDeviceSynchronize();

        streamingCalcNew<<<numBlocks, threadsPerBlock>>> (
            d_f_coll, 
            nx, ny, nz, d_f 
        ); 
        cudaDeviceSynchronize();
        
        streamingCalc<<<numBlocks, threadsPerBlock>>> (
            d_g, d_g_out, 
            nx, ny, nz
        );
        cudaDeviceSynchronize();

        fgBoundary_f<<<numBlocks, threadsPerBlock>>> (
            d_f, d_rho, 
            nx, ny, nz
        );
        cudaDeviceSynchronize();

        fgBoundary_g<<<numBlocks, threadsPerBlock>>> (
            d_g_out, d_phi, 
            nx, ny, nz
        );
        cudaDeviceSynchronize();
        cudaMemcpy(d_g, d_g_out, nx * ny * nz * GPOINTS * sizeof(dfloat), cudaMemcpyDeviceToDevice);
        
        boundaryConditions_z<<<numBlocks, threadsPerBlock>>> (
            d_phi, nx, ny, nz
        );
        cudaDeviceSynchronize();

        boundaryConditions_y<<<numBlocks, threadsPerBlock>>> (
            d_phi, nx, ny, nz
        );
        cudaDeviceSynchronize();

        if (t % stamp == 0) {

            checkCudaErrors(cudaMemcpy(phi_host.data(), d_phi, nx * ny * nz * sizeof(dfloat), cudaMemcpyDeviceToHost));
            ostringstream filename_phi_bin;
            filename_phi_bin << sim_dir << id << "_phi" << setw(6) << setfill('0') << t << ".bin";
            ofstream file_phi_bin(filename_phi_bin.str(), ios::binary);
            file_phi_bin.write(reinterpret_cast<const char*>(phi_host.data()), phi_host.size() * sizeof(dfloat));
            file_phi_bin.close();

            cout << "Passo " << t << ": Dados salvos em " << sim_dir << endl;
        }
    }

    dfloat *pointers[] = {d_f, d_g, d_phi, d_rho, 
                          d_mod_grad, d_normx, d_normy, d_normz, d_indicator,
                          d_curvature, d_ffx, d_ffy, d_ffz, d_ux, d_uy, d_uz,
                          d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz, d_f_coll, d_g_out
                        };
    freeMemory(pointers, 24);  

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed_time = end_time - start_time;
    cout << "Tempo total de execução: " << elapsed_time.count() << " segundos" << endl;

    return 0;
}
