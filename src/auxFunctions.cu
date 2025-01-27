#include "auxFunctions.cuh"
#include "var.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include "errorDef.cuh"

#include "precision.cuh"

void freeMemory(dfloat **pointers, int count) {
    for (int i = 0; i < count; ++i) {
        if (pointers[i] != nullptr) {
            cudaFree(pointers[i]);
        }
    }
}

void computeInitialCPU(
    vector<dfloat> &phi, vector<dfloat> &rho, const vector<dfloat> &w, const vector<dfloat> &w_g, 
    vector<dfloat> &f, vector<dfloat> &g, int nx, int ny, int nz, int fpoints, int gpoints, dfloat res
) {

    auto IDX3D = [&](int i, int j, int k) {
        return ((i) + nx * ((j) + ny * (k)));
    };
    auto IDX4D = [&](int i, int j, int k, int l) {
        return ((i) + nx * ((j) + ny * ((k) + nz * (l))));
    };

    for (int k = 1; k < nz-1; ++k) {
        for (int j = 1; j < ny-1; ++j) {
            for (int i = 1; i < nx-1; ++i) {
                dfloat Ri = sqrt((i - nx / 2.0) * (i - nx / 2.0) / 4.0 +
                                        (j - ny / 2.0) * (j - ny / 2.0) +
                                        (k - nz / 2.0) * (k - nz / 2.0));
                phi[IDX3D(i,j,k)] = 0.5 + 0.5 * tanh(2.0 * (20 - Ri) / (3.0 * res));
            }
        }
    }

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                for (int l = 0; l < fpoints; ++l) {
                    f[IDX4D(i,j,k,l)] = w[l] * rho[IDX3D(i,j,k)];
                }
                for (int l = 0; l < gpoints; ++l) {
                    g[IDX4D(i,j,k,l)] = w_g[l] * phi[IDX3D(i,j,k)];
                }
            }
        }
    }

}

void generateSimulationInfoFile(
    const string& filepath, const int nx, const int ny, const int nz, const int stamp, const int nsteps, const dfloat tau, 
    const string& sim_id, const string& fluid_model
) {
    try {
        ofstream file(filepath);

        if (!file.is_open()) {
            cerr << "Erro ao abrir o arquivo: " << filepath << endl;
            return;
        }

        file << "---------------------------- SIMULATION INFORMATION ----------------------------\n"
             << "                           Simulation ID: " << sim_id << '\n'
             << "                           Velocity set: " << fluid_model << '\n'
             << "                           Precision: " << PRECISION_TYPE << '\n'
             << "                           NX: " << nx << '\n'
             << "                           NY: " << ny << '\n'
             << "                           NZ: " << nz << '\n'
             << "                           NZ_TOTAL: " << nz << '\n'
             << "                           Tau: " << tau << '\n'
             << "                           Umax: 0.000000e+00\n"
             << "                           FX: 0.000000e+00\n"
             << "                           FY: 0.000000e+00\n"
             << "                           FZ: 0.000000e+00\n"
             << "                           Save steps: " << stamp << '\n'
             << "                           Nsteps: " << nsteps << '\n'
             << "                           MLUPS: 1.187970e+01\n"
             << "--------------------------------------------------------------------------------\n";

        file.close();
        cout << "Arquivo de informações da simulação criado em: " << filepath << endl;
    } catch (const exception& e) {
        cerr << "Erro ao gerar o arquivo de informações: " << e.what() << endl;
    }
}
