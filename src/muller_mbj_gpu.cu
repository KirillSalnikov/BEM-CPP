#include "muller_mbj.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

void cuda_check(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}

__device__ __forceinline__ double2 complex_subtract_product(
    double2 value, double2 left, double2 right)
{
    return make_double2(
        value.x - (left.x * right.x - left.y * right.y),
        value.y - (left.x * right.y + left.y * right.x));
}

__device__ __forceinline__ double2 complex_divide(
    double2 numerator, double2 denominator)
{
    const double inverse =
        1.0 / (denominator.x * denominator.x +
               denominator.y * denominator.y);
    return make_double2(
        (numerator.x * denominator.x +
         numerator.y * denominator.y) * inverse,
        (numerator.y * denominator.x -
         numerator.x * denominator.y) * inverse);
}

__global__ void muller_mbj_pair_kernel(
    const double2* rhs_x,
    const double2* rhs_y,
    double2* solution_x,
    double2* solution_y,
    const int* block_offsets,
    const int* lu_offsets,
    const int* block_dofs,
    const int* block_pivots,
    const double2* block_lu,
    const int* core_begin,
    const int* core_end,
    int block_count)
{
    const int block_index = blockIdx.x;
    if (block_index >= block_count)
        return;
    const int begin = block_offsets[block_index];
    const int end = block_offsets[block_index + 1];
    const int dimension = end - begin;
    const int lu_begin = lu_offsets[block_index];
    extern __shared__ double2 shared[];
    double2* local_x = shared;
    double2* local_y = shared + dimension;

    for (int row = threadIdx.x; row < dimension; row += blockDim.x) {
        const int dof = block_dofs[begin + row];
        local_x[row] = rhs_x[dof];
        local_y[row] = rhs_y[dof];
    }
    __syncthreads();

    for (int column = 0; column < dimension; column++) {
        if (threadIdx.x == 0) {
            const int pivot = block_pivots[begin + column];
            if (pivot != column) {
                const double2 swap_x = local_x[column];
                const double2 swap_y = local_y[column];
                local_x[column] = local_x[pivot];
                local_y[column] = local_y[pivot];
                local_x[pivot] = swap_x;
                local_y[pivot] = swap_y;
            }
        }
        __syncthreads();
    }

    for (int column = 0; column < dimension; column++) {
        const double2 value_x = local_x[column];
        const double2 value_y = local_y[column];
        for (int row = column + 1 + threadIdx.x;
             row < dimension; row += blockDim.x) {
            const double2 multiplier =
                block_lu[lu_begin + row * dimension + column];
            local_x[row] = complex_subtract_product(
                local_x[row], multiplier, value_x);
            local_y[row] = complex_subtract_product(
                local_y[row], multiplier, value_y);
        }
        __syncthreads();
    }

    for (int column = dimension - 1; column >= 0; column--) {
        if (threadIdx.x == 0) {
            const double2 diagonal =
                block_lu[lu_begin + column * dimension + column];
            local_x[column] =
                complex_divide(local_x[column], diagonal);
            local_y[column] =
                complex_divide(local_y[column], diagonal);
        }
        __syncthreads();
        const double2 value_x = local_x[column];
        const double2 value_y = local_y[column];
        for (int row = threadIdx.x; row < column; row += blockDim.x) {
            const double2 upper =
                block_lu[lu_begin + row * dimension + column];
            local_x[row] = complex_subtract_product(
                local_x[row], upper, value_x);
            local_y[row] = complex_subtract_product(
                local_y[row], upper, value_y);
        }
        __syncthreads();
    }

    const int output_begin = core_begin[block_index];
    const int output_end = core_end[block_index];
    for (int row = output_begin + threadIdx.x;
         row < output_end; row += blockDim.x) {
        const int dof = block_dofs[begin + row];
        solution_x[dof] = local_x[row];
        if (solution_y != solution_x)
            solution_y[dof] = local_y[row];
    }
}

} // namespace

void MullerMbjPreconditioner::upload_device()
{
    cleanup_device();
    if (system_dofs <= 0 || blocks.empty() || stores_inverse ||
        coarse_rank != 0) {
        return;
    }

    std::vector<int> offsets(blocks.size() + 1, 0);
    std::vector<int> lu_offsets(blocks.size() + 1, 0);
    std::vector<int> dofs;
    std::vector<int> pivots;
    std::vector<int> core_begin(blocks.size());
    std::vector<int> core_end(blocks.size());
    std::vector<std::complex<double>> lu;
    for (size_t index = 0; index < blocks.size(); index++) {
        const MullerMbjBlock& block = blocks[index];
        const int dimension = static_cast<int>(block.dofs.size());
        if (dimension <= 0 ||
            block.pivots.size() != block.dofs.size() ||
            block.lu.size() !=
                static_cast<size_t>(dimension) * dimension) {
            throw std::runtime_error(
                "invalid Muller MBJ block for GPU upload");
        }
        offsets[index + 1] = offsets[index] + dimension;
        lu_offsets[index + 1] =
            lu_offsets[index] + dimension * dimension;
        device_max_dimension =
            std::max(device_max_dimension, dimension);
        dofs.insert(dofs.end(), block.dofs.begin(), block.dofs.end());
        pivots.insert(
            pivots.end(), block.pivots.begin(), block.pivots.end());
        lu.insert(lu.end(), block.lu.begin(), block.lu.end());
        core_begin[index] = block.core_dof_begin;
        core_end[index] = block.core_dof_end;
    }

    const auto allocate_copy = [](
        void** device, const void* host, size_t bytes,
        const char* operation) {
        cuda_check(cudaMalloc(device, bytes), operation);
        cuda_check(
            cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice),
            operation);
    };
    allocate_copy(
        &d_block_offsets, offsets.data(),
        offsets.size() * sizeof(int), "upload Muller MBJ offsets");
    allocate_copy(
        &d_block_lu_offsets, lu_offsets.data(),
        lu_offsets.size() * sizeof(int), "upload Muller MBJ LU offsets");
    allocate_copy(
        &d_block_dofs, dofs.data(),
        dofs.size() * sizeof(int), "upload Muller MBJ dofs");
    allocate_copy(
        &d_block_pivots, pivots.data(),
        pivots.size() * sizeof(int), "upload Muller MBJ pivots");
    allocate_copy(
        &d_block_lu, lu.data(),
        lu.size() * sizeof(std::complex<double>),
        "upload Muller MBJ factors");
    allocate_copy(
        &d_block_core_begin, core_begin.data(),
        core_begin.size() * sizeof(int), "upload Muller MBJ core starts");
    allocate_copy(
        &d_block_core_end, core_end.data(),
        core_end.size() * sizeof(int), "upload Muller MBJ core ends");
    device_block_count = static_cast<int>(blocks.size());
    device_ready = true;
    std::printf(
        "  [MBJ-GPU] uploaded %d blocks, maximum dimension %d\n",
        device_block_count, device_max_dimension);
    std::fflush(stdout);

    const char* validate = std::getenv("BEM_MULLER_MBJ_VALIDATE_GPU");
    if (validate != nullptr && std::string(validate) != "0") {
        std::vector<std::complex<double>> rhs_x(system_dofs);
        std::vector<std::complex<double>> rhs_y(system_dofs);
        for (int index = 0; index < system_dofs; index++) {
            rhs_x[index] = std::complex<double>(
                std::sin(0.013 * index), std::cos(0.017 * index));
            rhs_y[index] = std::complex<double>(
                std::cos(0.019 * index), -std::sin(0.023 * index));
        }
        std::vector<std::complex<double>> cpu_x(system_dofs);
        std::vector<std::complex<double>> cpu_y(system_dofs);
        std::vector<std::complex<double>> gpu_x(system_dofs);
        std::vector<std::complex<double>> gpu_y(system_dofs);
        apply(rhs_x.data(), cpu_x.data());
        apply(rhs_y.data(), cpu_y.data());
        void* device_rhs_x = nullptr;
        void* device_rhs_y = nullptr;
        void* device_solution_x = nullptr;
        void* device_solution_y = nullptr;
        const size_t bytes =
            static_cast<size_t>(system_dofs) * sizeof(double2);
        cuda_check(
            cudaMalloc(&device_rhs_x, bytes),
            "allocate MBJ validation RHS X");
        cuda_check(
            cudaMalloc(&device_rhs_y, bytes),
            "allocate MBJ validation RHS Y");
        cuda_check(
            cudaMalloc(&device_solution_x, bytes),
            "allocate MBJ validation output X");
        cuda_check(
            cudaMalloc(&device_solution_y, bytes),
            "allocate MBJ validation output Y");
        cuda_check(
            cudaMemcpy(
                device_rhs_x, rhs_x.data(), bytes,
                cudaMemcpyHostToDevice),
            "upload MBJ validation RHS X");
        cuda_check(
            cudaMemcpy(
                device_rhs_y, rhs_y.data(), bytes,
                cudaMemcpyHostToDevice),
            "upload MBJ validation RHS Y");
        apply_device_complex_pair(
            device_rhs_x, device_rhs_y,
            device_solution_x, device_solution_y);
        cuda_check(
            cudaMemcpy(
                gpu_x.data(), device_solution_x, bytes,
                cudaMemcpyDeviceToHost),
            "download MBJ validation output X");
        cuda_check(
            cudaMemcpy(
                gpu_y.data(), device_solution_y, bytes,
                cudaMemcpyDeviceToHost),
            "download MBJ validation output Y");
        cudaFree(device_rhs_x);
        cudaFree(device_rhs_y);
        cudaFree(device_solution_x);
        cudaFree(device_solution_y);
        double error_squared = 0.0;
        double reference_squared = 0.0;
        for (int index = 0; index < system_dofs; index++) {
            error_squared +=
                std::norm(gpu_x[index] - cpu_x[index]) +
                std::norm(gpu_y[index] - cpu_y[index]);
            reference_squared +=
                std::norm(cpu_x[index]) + std::norm(cpu_y[index]);
        }
        const double relative_error = std::sqrt(
            error_squared / std::max(reference_squared, 1.0e-300));
        std::printf(
            "  [MBJ-GPU] CPU/GPU application relative error %.3e\n",
            relative_error);
        std::fflush(stdout);
        if (!(relative_error <= 2.0e-12))
            throw std::runtime_error(
                "Muller MBJ CPU/GPU validation failed");
    }
}

void MullerMbjPreconditioner::cleanup_device()
{
    void** pointers[] = {
        &d_block_offsets,
        &d_block_lu_offsets,
        &d_block_dofs,
        &d_block_pivots,
        &d_block_lu,
        &d_block_core_begin,
        &d_block_core_end
    };
    for (void** pointer : pointers) {
        if (*pointer != nullptr)
            cudaFree(*pointer);
        *pointer = nullptr;
    }
    device_block_count = 0;
    device_max_dimension = 0;
    device_ready = false;
}

bool MullerMbjPreconditioner::device_apply_available() const
{
    return device_ready && device_block_count > 0 &&
        device_max_dimension > 0 && coarse_rank == 0 &&
        !stores_inverse;
}

void MullerMbjPreconditioner::apply_device_complex_pair(
    const void* device_rhs_x,
    const void* device_rhs_y,
    void* device_solution_x,
    void* device_solution_y) const
{
    if (!device_apply_available())
        throw std::runtime_error("Muller MBJ GPU factors are unavailable");
    cuda_check(
        cudaMemsetAsync(
            device_solution_x, 0,
            static_cast<size_t>(system_dofs) * sizeof(double2)),
        "clear Muller MBJ output X");
    cuda_check(
        cudaMemsetAsync(
            device_solution_y, 0,
            static_cast<size_t>(system_dofs) * sizeof(double2)),
        "clear Muller MBJ output Y");
    const int threads = 128;
    const size_t shared_bytes =
        static_cast<size_t>(2 * device_max_dimension) * sizeof(double2);
    muller_mbj_pair_kernel<<<device_block_count, threads, shared_bytes>>>(
        static_cast<const double2*>(device_rhs_x),
        static_cast<const double2*>(device_rhs_y),
        static_cast<double2*>(device_solution_x),
        static_cast<double2*>(device_solution_y),
        static_cast<const int*>(d_block_offsets),
        static_cast<const int*>(d_block_lu_offsets),
        static_cast<const int*>(d_block_dofs),
        static_cast<const int*>(d_block_pivots),
        static_cast<const double2*>(d_block_lu),
        static_cast<const int*>(d_block_core_begin),
        static_cast<const int*>(d_block_core_end),
        device_block_count);
    cuda_check(cudaGetLastError(), "apply paired Muller MBJ");
}

void MullerMbjPreconditioner::apply_device_complex(
    const void* device_rhs,
    void* device_solution) const
{
    apply_device_complex_pair(
        device_rhs, device_rhs, device_solution, device_solution);
}
