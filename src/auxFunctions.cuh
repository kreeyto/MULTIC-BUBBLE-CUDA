#ifndef AUXFUNCTIONS_CUH
#define AUXFUNCTIONS_CUH
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

#include "precision.cuh"

void freeMemory(dfloat **pointers, int count);

void generateSimulationInfoFile(const string& filepath, int nx, int ny, int nz, int stamp, int nsteps, dfloat tau, 
    const string& sim_id, const string& fluid_model);

#endif
