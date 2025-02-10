// precision.cuh
#ifndef PRECISION_H
#define PRECISION_H

using namespace std;
#define SINGLE_PRECISION

#ifdef SINGLE_PRECISION
    typedef float dfloat;      // single precision
    #define PRECISION_TYPE "float"
#endif

#ifdef DOUBLE_PRECISION
    typedef double dfloat;     // double precision
    #define PRECISION_TYPE "double"
#endif

#ifdef FD3Q19
    #define FPOINTS 19
#endif
#ifdef PD3Q15
    #define GPOINTS 15
#endif

#endif // PRECISION_H
