// precision.h
#ifndef PRECISION_CUH
#define PRECISION_CUH

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
#elif defined(FD3Q27)
    #define FPOINTS 27
#endif
#ifdef PD3Q15
    #define GPOINTS 15
#elif defined(PD3Q19)
    #define GPOINTS 19
#endif

#endif // PRECISION_CUH
