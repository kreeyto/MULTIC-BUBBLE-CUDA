#include "utils.cuh"
#include "var.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>

void freeMemory(float **pointers, int count) {
    for (int i = 0; i < count; ++i) {
        if (pointers[i] != nullptr) {
            cudaFree(pointers[i]);
        }
    }
}

void computeInitialCPU(
    std::vector<float> &phi, std::vector<float> &rho, const std::vector<float> &w, const std::vector<float> &w_g, 
    std::vector<float> &f, std::vector<float> &g, int nx, int ny, int nz, int fpoints, int gpoints, float res
) {

    auto F_IDX = [&](int i, int j, int k, int l) {
        return i + nx * (j + ny * (k + nz * l));
    };

    for (int k = 1; k < nz-1; ++k) {
        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                int idx = i + nx * (j + ny * k);
                float Ri = std::sqrt((i - nx / 2.0f) * (i - nx / 2.0f) / 4.0f +
                                        (j - ny / 2.0f) * (j - ny / 2.0f) +
                                        (k - nz / 2.0f) * (k - nz / 2.0f));
                phi[idx] = 0.5f + 0.5f * std::tanh(2.0f * (20 * res - Ri) / (3.0f * res));
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                for (int l = 0; l < fpoints; ++l) {
                    f[F_IDX(i, j, k, l)] = w[l] * rho[idx];
                }
                for (int l = 0; l < gpoints; ++l) {
                    g[F_IDX(i, j, k, l)] = w_g[l] * phi[idx];
                }
            }
        }
    }

}

void generateSimulationInfoFile(const std::string& filepath, int nx, int ny, int nz, int stamp, int nsteps, float tau) {
    try {
        std::ofstream file(filepath);

        if (!file.is_open()) {
            std::cerr << "Erro ao abrir o arquivo: " << filepath << std::endl;
            return;
        }

        file << "---------------------------- SIMULATION INFORMATION ----------------------------\n"
             << "                           Simulation ID: 000\n"
             << "                           Velocity set: D3Q19\n"
             << "                           Precision: float\n"
             << "                           NX: " << nx << '\n'
             << "                           NY: " << ny << '\n'
             << "                           NZ: " << nz << '\n'
             << "                           NZ_TOTAL: " << nz << '\n'
             << "                           Tau: " << tau << '\n'
             << "                           Umax: 5.000000e-02\n"
             << "                           FX: 0.000000e+00\n"
             << "                           FY: 0.000000e+00\n"
             << "                           FZ: 0.000000e+00\n"
             << "                           Save steps: " << stamp << '\n'
             << "                           Nsteps: " << nsteps << '\n'
             << "                           MLUPS: 1.187970e+01\n"
             << "--------------------------------------------------------------------------------\n\n"
             << "------------------------------ BOUNDARY CONDITIONS -----------------------------\n"
             << "                           BC mode: Moment Based \n"
             << "                           BC type: testBC\n"
             << "--------------------------------------------------------------------------------\n";

        file.close();
        std::cout << "Arquivo de informações da simulação criado em: " << filepath << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erro ao gerar o arquivo de informações: " << e.what() << std::endl;
    }
}

