// precision.h
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

#endif // PRECISION_H
