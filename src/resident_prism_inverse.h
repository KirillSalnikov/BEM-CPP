#pragma once
#include "host_symmetric_inverse_layout.h"
#include "orbit_inverse_gpu.h"
class PrismCirculantInverse;

// The complete full-D6h local/FFT blend. Uses the existing factorizations and
// mapped FFT symbols; no new numerical approximation. Non-reentrant; all
// enqueued operations use cudaStreamPerThread. The borrowed FFT must outlive
// this instance and may not be rebuilt or used concurrently.
class ResidentPrismInverse {
public:
    using Complex=std::complex<double>;
    ResidentPrismInverse();
    ~ResidentPrismInverse();
    ResidentPrismInverse(const ResidentPrismInverse&)=delete;
    ResidentPrismInverse& operator=(const ResidentPrismInverse&)=delete;
    // Budget counts this object's explicit allocations only. Existing FFT
    // banks/mapping and CUDA/cuBLAS internals are reported separately.
    void init(const bem_orbit_device::Layout& local,const bem_orbit_device::SymmetricLayout& symmetry,
              PrismCirculantInverse& fft,size_t memory_budget_bytes);
    void enqueue_device(const Complex* input,Complex* output,double blend=.5);
    void apply(const Complex* input,Complex* output,double blend=.5);
    Complex* device_input();
    Complex* device_output();
    size_t storage_bytes()const;
    int system_dofs()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
