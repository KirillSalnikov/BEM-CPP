#pragma once
#include "sector_transform_gpu.h"
class MullerFmmOperator;
class ResidentPrismInverse;
// Fused physical-current chain for the existing host sector GMRES. Arnoldi
// vectors stay in host RAM, but neither physical currents nor the operator
// result are staged through host memory between these four operations.
class ResidentSectorOperator {
public:
    using Complex=bem_sector_device::Complex;
    ResidentSectorOperator();
    ~ResidentSectorOperator();
    ResidentSectorOperator(const ResidentSectorOperator&)=delete;
    ResidentSectorOperator& operator=(const ResidentSectorOperator&)=delete;
    // Both borrowed backends must outlive this object; they cannot be used
    // concurrently. Operator must have complete device assembly/correction.
    void init(MullerFmmOperator& op,ResidentPrismInverse& inverse,
              const bem_sector_device::Layout& transform,size_t budget_bytes);
    void enqueue(const Complex* encoded_input,Complex* encoded_output,double blend);
    void apply(const Complex* encoded_input,Complex* encoded_output,double blend);
    Complex* device_input();
    Complex* device_output();
    size_t storage_bytes()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
