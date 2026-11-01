#ifndef BEM_HOST_CUDA_STUB_CUFFT_H
#define BEM_HOST_CUDA_STUB_CUFFT_H

#include "cuComplex.h"

// Types only: host metadata fixtures include pfft.h but never execute cuFFT.
// GPU targets must use the actual CUDA toolkit headers and libraries.
typedef int cufftHandle;
typedef cuFloatComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;

#endif
