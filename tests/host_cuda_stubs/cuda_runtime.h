#ifndef BEM_HOST_CUDA_STUB_RUNTIME_H
#define BEM_HOST_CUDA_STUB_RUNTIME_H

#include <cstddef>

typedef struct {
    float x;
    float y;
} float2;

typedef struct {
    double x;
    double y;
} double2;

inline float2 make_float2(float x, float y)
{
    float2 value = {x, y};
    return value;
}

inline double2 make_double2(double x, double y)
{
    double2 value = {x, y};
    return value;
}

typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;

inline const char* cudaGetErrorString(cudaError_t)
{
    return "CUDA runtime is unavailable in this host-only test";
}

#endif
