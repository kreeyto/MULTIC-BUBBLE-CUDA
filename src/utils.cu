#include "utils.cuh"
#include "constants.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

void freeMemory(float **pointers, int count) {
    for (int i = 0; i < count; ++i) {
        if (pointers[i] != nullptr) {
            cudaFree(pointers[i]);
        }
    }
}

void computeInitialCPU(
    std::vector<float> &phi, std::vector<float> &rho, const std::vector<float> &w, const std::vector<float> &w_g, 
    std::vector<float> &f, std::vector<float> &g, int nx, int ny, int nz, int fpoints, int gpoints
) {
    auto F_IDX = [&](int i, int j, int k, int l) {
        return i + nx * (j + ny * (k + nz * l));
    };

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx = i + nx * (j + ny * k);
                if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1) {
                    float Ri = std::sqrt((i - nx / 2.0f) * (i - nx / 2.0f) / 4.0f +
                                         (j - ny / 2.0f) * (j - ny / 2.0f) +
                                         (k - nz / 2.0f) * (k - nz / 2.0f));
                    phi[idx] = 0.5f + 0.5f * std::tanh(2.0f * (20 - Ri) / 3.0f);
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
}
