#pragma once
#include "host_orbit_inverse_layout.h"
#include <memory>

// Resident FP64 orbit LU inverse. All 12 group RHS share each factorization.
// Calls use the calling thread's CUDA per-thread default stream. Not reentrant.
class OrbitInverseGPU {
public:
    using Complex=bem_orbit_device::Complex;
    OrbitInverseGPU();
    ~OrbitInverseGPU();
    OrbitInverseGPU(const OrbitInverseGPU&)=delete;
    OrbitInverseGPU& operator=(const OrbitInverseGPU&)=delete;
    void init(const bem_orbit_device::Layout& layout,size_t memory_budget_bytes);
    // Input/output already reside on the device. Enqueues gather, pivots,
    // two batched triangular solves and deterministic scatter, without host
    // copies, host synchronization or explicit per-call workspace allocation.
    void enqueue_device(const Complex* input,Complex* output);
    void apply(const Complex* input,Complex* output);
    Complex* device_input();
    Complex* device_output();
    size_t storage_bytes()const; // Explicit arrays; excludes CUDA/cuBLAS internals.
    int system_dofs()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
