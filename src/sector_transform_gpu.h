#pragma once
#include "host_sector_transform_layout.h"
#include <memory>
class SectorTransformGPU {
public:
    using Complex=bem_sector_device::Complex;
    SectorTransformGPU();
    ~SectorTransformGPU();
    SectorTransformGPU(const SectorTransformGPU&)=delete;
    SectorTransformGPU& operator=(const SectorTransformGPU&)=delete;
    void init(const bem_sector_device::Layout& layout,size_t budget_bytes);
    // Disjoint device vectors. Pure kernels on cudaStreamPerThread, with no
    // vector workspace, allocations, host copies or host synchronization.
    void enqueue(const Complex* input,Complex* output,bool encode,bool equation);
    size_t storage_bytes()const;
    int system_dofs()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
