#ifndef UTILS_CUH
#define UTILS_CUH
#include <vector>

void initializeConstants();
void freeMemory(float **pointers, int count);
void computeInitialCPU(
    std::vector<float> &phi, std::vector<float> &rho, const std::vector<float> &w, const std::vector<float> &w_g, 
    std::vector<float> &f, std::vector<float> &g, int nx, int ny, int nz, int fpoints, int gpoints
);

#endif
