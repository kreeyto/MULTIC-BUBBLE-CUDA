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
    if (argc < 5) {
        cerr << "Erro: Uso: " << argv[0] << " F<fluid velocity set> P<phase field velocity set> <id> <save_binary>" << endl;
        return 1;
    }
    string fluid_model = argv[1];
    string phase_model = argv[2];
    string id = argv[3];
    bool save_binary = stoi(argv[4]);
    bool debug_mode = (argc > 5) ? stoi(argv[5]) : 0; 

    string base_dir;   
    #ifdef _WIN32
        base_dir = "..\\";
    #else
        base_dir = "../";
    #endif
    string model_dir = base_dir + "bin/" + fluid_model + "_" + phase_model + "/";
    string sim_dir = model_dir + id + "/";
    string matlab_dir = base_dir + "matlabFiles/" + fluid_model + "_" + phase_model + "/" + id + "/";

    if (!debug_mode) { 
        if (save_binary) {
            #ifdef _WIN32
                string mkdir_command = "mkdir \"" + sim_dir + "\"";
            #else
                string mkdir_command = "mkdir -p \"" + sim_dir + "\"";
            #endif
            int ret = system(mkdir_command.c_str());
            (void)ret;
        } else {
            #ifdef _WIN32
                string mkdir_command = "mkdir \"" + matlab_dir + "\"";
            #else
                string mkdir_command = "mkdir -p \"" + matlab_dir + "\"";
            #endif
            int ret = system(mkdir_command.c_str());
            (void)ret;
        }
    }

    // ========================= //
    int stamp = 1, nsteps = 10;
    // ========================= //
    initializeVars();

    if (!debug_mode && save_binary) {
        string info_file = sim_dir + id + "_info.txt";
        generateSimulationInfoFile(info_file, nx, ny, nz, stamp, nsteps, tau, id, fluid_model);
    }

    vector<dfloat> f(nx * ny * nz * fpoints, 0.0);
    vector<dfloat> g(nx * ny * nz * gpoints, 0.0);
    vector<dfloat> phi(nx * ny * nz, 0.0);
    vector<dfloat> rho(nx * ny * nz, 1.0); 

    // =========================== FLUID WEIGHTS =========================== //
    const vector<dfloat> w = {
        1.0 / 3.0, 
        1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
        1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
    };
    // ===================================================================== //

    // ======================== PHASE FIELD WEIGHTS ======================== //
    const vector<dfloat> w_g = {
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

    vector<dfloat> phi_host(nx * ny * nz);

    for (int t = 0; t < nsteps; ++t) {

        cout << "Passo " << t << " de " << nsteps << " iniciado..." << endl;

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

        if (!debug_mode && t % stamp == 0) {
            checkCudaErrors(cudaMemcpy(phi_host.data(), d_phi, nx * ny * nz * sizeof(dfloat), cudaMemcpyDeviceToHost));
            
            if (save_binary) {
                ostringstream filename_phi_bin;
                filename_phi_bin << sim_dir << id << "_phi" << setw(6) << setfill('0') << t << ".bin";
                ofstream file_phi_bin(filename_phi_bin.str(), ios::binary);
                file_phi_bin.write(reinterpret_cast<const char*>(phi_host.data()), phi_host.size() * sizeof(dfloat));
                file_phi_bin.close();
            } else {
                ostringstream filename_phi_txt;
                filename_phi_txt << matlab_dir << id << "_phi" << t << ".txt";
                ofstream file_phi_txt(filename_phi_txt.str());
                if (file_phi_txt.is_open()) {
                    for (int z = 0; z < nz; ++z) {
                        for (int y = 0; y < ny; ++y) {
                            for (int x = 0; x < nx; ++x) {
                                int index = x + nx * (y + ny * z);
                                file_phi_txt << phi_host[index] << " ";
                            }
                            file_phi_txt << "\n";
                        }
                        file_phi_txt << "\n";
                    }
                    file_phi_txt.close();
                }
            }

            cout << "Passo " << t << ": Dados salvos em " << (save_binary ? sim_dir : matlab_dir) << endl;
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
