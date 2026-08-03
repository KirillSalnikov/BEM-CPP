#ifndef BEM_HOST_CUDA_STUB_CUCOMPLEX_H
#define BEM_HOST_CUDA_STUB_CUCOMPLEX_H

typedef struct {
    float x;
    float y;
} cuFloatComplex;

typedef struct {
    double x;
    double y;
} cuDoubleComplex;

inline cuFloatComplex make_cuFloatComplex(float real, float imag)
{
    cuFloatComplex value = {real, imag};
    return value;
}

inline cuDoubleComplex make_cuDoubleComplex(double real, double imag)
{
    cuDoubleComplex value = {real, imag};
    return value;
}

#endif
