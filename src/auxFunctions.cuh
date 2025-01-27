#ifndef AUXFUNCTIONS_CUH
#define AUXFUNCTIONS_CUH
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

#include "precision.cuh"

void initializeVars();

void freeMemory(dfloat **pointers, int count);

void computeInitialCPU(
    vector<dfloat> &phi, vector<dfloat> &rho, const vector<dfloat> &w, const vector<dfloat> &w_g, 
    vector<dfloat> &f, vector<dfloat> &g, int nx, int ny, int nz, int fpoints, int gpoints, dfloat res
);

void generateSimulationInfoFile(const string& filepath, int nx, int ny, int nz, int stamp, int nsteps, dfloat tau, 
    const string& sim_id, const string& fluid_model);

#endif
