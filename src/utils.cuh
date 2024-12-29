#ifndef UTILS_CUH
#define UTILS_CUH
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

#include "precision.cuh"

void initializeVars();

void freeMemory(dfloat **pointers, int count);

void computeInitialCPU(
    std::vector<dfloat> &phi, std::vector<dfloat> &rho, const std::vector<dfloat> &w, const std::vector<dfloat> &w_g, 
    std::vector<dfloat> &f, std::vector<dfloat> &g, int nx, int ny, int nz, int fpoints, int gpoints, dfloat res
);

void generateSimulationInfoFile(const std::string& filepath, int nx, int ny, int nz, int stamp, int nsteps, dfloat tau, 
    const std::string& sim_id, const std::string& fluid_model);

#endif
