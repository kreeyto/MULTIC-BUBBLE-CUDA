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

#include "precision.cuh"

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
    int stamp = 100, nsteps = 1000;
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
            d_phi, nx, ny, nz
        ); 
        getLastCudaError("Erro ao lançar initPhase");
        checkCudaErrors(cudaDeviceSynchronize());

        initDist<<<numBlocks, threadsPerBlock>>> (
            d_rho, d_phi, d_f, d_g, nx, ny, nz
        ); 
        getLastCudaError("Erro ao lançar initDist");
        checkCudaErrors(cudaDeviceSynchronize());

    // ========================================= //

    vector<dfloat> phi_host(nx * ny * nz);
    vector<dfloat> ux_host(nx * ny * nz);
    vector<dfloat> uy_host(nx * ny * nz);
    vector<dfloat> uz_host(nx * ny * nz);

    for (int t = 0; t <= nsteps; ++t) {
        cout << "Passo " << t << " de " << nsteps << " iniciado..." << endl;



        // ================= PHASE FIELD ================= //

            phiCalc<<<numBlocks, threadsPerBlock>>> (
                d_phi, d_g, nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar phiCalc");
            checkCudaErrors(cudaDeviceSynchronize());

        // =============================================== // 
        


        // ===================== NORMALS ===================== //

            gradCalc<<<numBlocks, threadsPerBlock>>> (
                d_phi, d_mod_grad, d_normx, d_normy, d_normz, 
                d_indicator, 
                nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar gradCalc");
            checkCudaErrors(cudaDeviceSynchronize());

        // =================================================== // 

        

        // ==================== CURVATURE ==================== //

            curvatureCalc<<<numBlocks, threadsPerBlock>>> (
                d_curvature, d_indicator,
                d_normx, d_normy, d_normz, 
                d_ffx, d_ffy, d_ffz,
                nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar curvatureCalc");
            checkCudaErrors(cudaDeviceSynchronize());

        // =================================================== //   


        
        // ===================== MOMENTI ===================== //

            momentiCalc<<<numBlocks, threadsPerBlock>>> (
                d_ux, d_uy, d_uz, d_rho,
                d_ffx, d_ffy, d_ffz, d_f,
                d_pxx, d_pyy, d_pzz,
                d_pxy, d_pxz, d_pyz,
                nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar momentiCalc");
            checkCudaErrors(cudaDeviceSynchronize());

        // ================================================== //   



        // ==================== COLLISION ==================== //

            collisionCalc<<<numBlocks, threadsPerBlock>>> (
                d_ux, d_uy, d_uz, 
                d_normx, d_normy, d_normz,
                d_ffx, d_ffy, d_ffz,
                d_rho, d_phi, d_g, 
                d_pxx, d_pyy, d_pzz, d_pxy, d_pxz, d_pyz, 
                nx, ny, nz, d_f_coll
            ); 
            getLastCudaError("Erro ao lançar collisionCalc");
            checkCudaErrors(cudaDeviceSynchronize());
            
            streamingColl<<<numBlocks, threadsPerBlock>>> (
                d_f, d_f_coll, 
                nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar streamingCalcNew");
            checkCudaErrors(cudaDeviceSynchronize());

        // ================================================== //    



        // =================== STREAMING =================== //

            streamingCalc<<<numBlocks, threadsPerBlock>>> (
                d_g, d_g_out, 
                nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar streamingCalc");
            checkCudaErrors(cudaDeviceSynchronize());
            cudaMemcpy(d_g, d_g_out, nx * ny * nz * GPOINTS * sizeof(dfloat), cudaMemcpyDeviceToDevice);

        // ================================================= //



        // ========================================== DISTRIBUTION ========================================== //

            fgBoundary_f<<<numBlocks, threadsPerBlock>>> (
                d_f, d_rho,
                nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar fgBoundary_f");
            checkCudaErrors(cudaDeviceSynchronize());

            fgBoundary_g<<<numBlocks, threadsPerBlock>>> (
                d_g, d_phi,
                nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar fgBoundary_g");
            checkCudaErrors(cudaDeviceSynchronize());

        // ================================================================================================= //

        
        
        // ======================= BOUNDARY ======================= //

            boundaryConditionsZ<<<numBlocks, threadsPerBlock>>> (
                d_phi, nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar boundaryConditionsZ");
            checkCudaErrors(cudaDeviceSynchronize());

            boundaryConditionsY<<<numBlocks, threadsPerBlock>>> (
                d_phi, nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar boundaryConditionsZ");
            checkCudaErrors(cudaDeviceSynchronize());

            boundaryConditionsX<<<numBlocks, threadsPerBlock>>> (
                d_phi, nx, ny, nz
            ); 
            getLastCudaError("Erro ao lançar boundaryConditionsZ");
            checkCudaErrors(cudaDeviceSynchronize());

        // ======================================================== //



        if (t % stamp == 0) {

            copyAndSaveToBinary(d_phi, nx * ny * nz, sim_dir, id, t, "phi");
            copyAndSaveToBinary(d_ux, nx * ny * nz, sim_dir, id, t, "ux");
            copyAndSaveToBinary(d_uy, nx * ny * nz, sim_dir, id, t, "uy");
            copyAndSaveToBinary(d_uz, nx * ny * nz, sim_dir, id, t, "uz");

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
