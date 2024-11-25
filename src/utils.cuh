#ifndef UTILS_CUH
#define UTILS_CUH

void initializeConstants();
void allocateMemory(float *&h_rho, float *&h_phi, float *&d_rho, float *&d_phi);
void freeMemory(float *h_rho, float *h_phi, float *d_rho, float *d_phi);
void copyToDevice(const float *h_rho, const float *h_phi, float *d_rho, float *d_phi);
void initializeHostArrays(float *h_rho, float *h_phi);

#endif
