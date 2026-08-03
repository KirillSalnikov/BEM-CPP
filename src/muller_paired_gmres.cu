#include "muller_paired_gmres.h"

#include "muller_fmm.h"
#include "muller_mbj.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cdouble = std::complex<double>;

bool env_flag_enabled(const char* name, bool default_value)
{
    const char* value = std::getenv(name);
    if (value == nullptr)
        return default_value;
    return std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "off") != 0;
}

void cuda_check(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess)
        return;
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status));
}

void cublas_check(cublasStatus_t status, const char* operation)
{
    if (status == CUBLAS_STATUS_SUCCESS)
        return;
    throw std::runtime_error(
        std::string(operation) + " failed with cuBLAS status " +
        std::to_string(static_cast<int>(status)));
}

struct DeviceBuffer {
    void* pointer = nullptr;

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t bytes)
    {
        cuda_check(cudaMalloc(&pointer, bytes), "allocate paired GMRES buffer");
    }
    ~DeviceBuffer()
    {
        if (pointer != nullptr)
            cudaFree(pointer);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    double2* complex_data()
    {
        return static_cast<double2*>(pointer);
    }
};

struct CublasHandle {
    cublasHandle_t value = nullptr;

    CublasHandle()
    {
        cublas_check(cublasCreate(&value), "create paired GMRES cuBLAS handle");
        cublas_check(
            cublasSetPointerMode(value, CUBLAS_POINTER_MODE_HOST),
            "set paired GMRES cuBLAS pointer mode");
    }
    ~CublasHandle()
    {
        if (value != nullptr)
            cublasDestroy(value);
    }
};

cuDoubleComplex* as_cublas(double2* pointer)
{
    return reinterpret_cast<cuDoubleComplex*>(pointer);
}

const cuDoubleComplex* as_cublas(const double2* pointer)
{
    return reinterpret_cast<const cuDoubleComplex*>(pointer);
}

cuDoubleComplex as_cublas(cdouble value)
{
    return make_cuDoubleComplex(value.real(), value.imag());
}

double vector_norm(
    cublasHandle_t handle, int count, const double2* vector)
{
    double result = 0.0;
    cublas_check(
        cublasDznrm2(handle, count, as_cublas(vector), 1, &result),
        "paired GMRES norm");
    return result;
}

void copy_vector(
    cublasHandle_t handle, int count,
    const double2* source, double2* destination)
{
    cublas_check(
        cublasZcopy(
            handle, count, as_cublas(source), 1,
            as_cublas(destination), 1),
        "paired GMRES vector copy");
}

void axpy_vector(
    cublasHandle_t handle, int count, cdouble alpha,
    const double2* source, double2* destination)
{
    const cuDoubleComplex coefficient = as_cublas(alpha);
    cublas_check(
        cublasZaxpy(
            handle, count, &coefficient,
            as_cublas(source), 1,
            as_cublas(destination), 1),
        "paired GMRES vector update");
}

void scale_copy(
    cublasHandle_t handle, int count, double inverse_norm,
    const double2* source, double2* destination)
{
    copy_vector(handle, count, source, destination);
    const cuDoubleComplex scale =
        make_cuDoubleComplex(inverse_norm, 0.0);
    cublas_check(
        cublasZscal(
            handle, count, &scale,
            as_cublas(destination), 1),
        "paired GMRES basis normalization");
}

void orthogonalize(
    cublasHandle_t handle,
    int vector_size,
    int basis_count,
    const double2* basis,
    double2* work,
    double2* device_coefficients,
    std::vector<cdouble>& host_coefficients)
{
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    const cuDoubleComplex minus_one = make_cuDoubleComplex(-1.0, 0.0);
    cublas_check(
        cublasZgemv(
            handle, CUBLAS_OP_C,
            vector_size, basis_count,
            &one, as_cublas(basis), vector_size,
            as_cublas(work), 1,
            &zero, as_cublas(device_coefficients), 1),
        "paired GMRES block inner products");
    cublas_check(
        cublasZgemv(
            handle, CUBLAS_OP_N,
            vector_size, basis_count,
            &minus_one, as_cublas(basis), vector_size,
            as_cublas(device_coefficients), 1,
            &one, as_cublas(work), 1),
        "paired GMRES block orthogonalization");
    host_coefficients.resize(basis_count);
    cuda_check(
        cudaMemcpy(
            host_coefficients.data(), device_coefficients,
            static_cast<size_t>(basis_count) * sizeof(double2),
            cudaMemcpyDeviceToHost),
        "download paired GMRES inner products");
}

void update_projected_column(
    std::vector<cdouble>& hessenberg,
    std::vector<cdouble>& cosine,
    std::vector<cdouble>& sine,
    std::vector<cdouble>& projected,
    int leading_dimension,
    int column)
{
    for (int row = 0; row < column; row++) {
        const cdouble first =
            hessenberg[static_cast<size_t>(row) *
                leading_dimension + column];
        const cdouble second =
            hessenberg[static_cast<size_t>(row + 1) *
                leading_dimension + column];
        hessenberg[static_cast<size_t>(row) *
            leading_dimension + column] =
            std::conj(cosine[row]) * first +
            std::conj(sine[row]) * second;
        hessenberg[static_cast<size_t>(row + 1) *
            leading_dimension + column] =
            -sine[row] * first + cosine[row] * second;
    }
    const cdouble first =
        hessenberg[static_cast<size_t>(column) *
            leading_dimension + column];
    const cdouble second =
        hessenberg[static_cast<size_t>(column + 1) *
            leading_dimension + column];
    const double denominator =
        std::sqrt(std::norm(first) + std::norm(second));
    cosine[column] = denominator > 1.0e-30
        ? first / denominator : cdouble(1.0);
    sine[column] = denominator > 1.0e-30
        ? second / denominator : cdouble(0.0);
    hessenberg[static_cast<size_t>(column) *
        leading_dimension + column] =
        std::conj(cosine[column]) * first +
        std::conj(sine[column]) * second;
    hessenberg[static_cast<size_t>(column + 1) *
        leading_dimension + column] = 0.0;
    projected[column + 1] =
        -sine[column] * projected[column];
    projected[column] =
        std::conj(cosine[column]) * projected[column];
}

void solve_projected(
    const std::vector<cdouble>& hessenberg,
    const std::vector<cdouble>& projected,
    int leading_dimension,
    int columns,
    std::vector<cdouble>& coefficients)
{
    coefficients.assign(columns, cdouble(0.0));
    for (int row = columns - 1; row >= 0; row--) {
        cdouble value = projected[row];
        for (int column = row + 1; column < columns; column++) {
            value -=
                hessenberg[static_cast<size_t>(row) *
                    leading_dimension + column] *
                coefficients[column];
        }
        const cdouble diagonal =
            hessenberg[static_cast<size_t>(row) *
                leading_dimension + row];
        if (std::abs(diagonal) <= 1.0e-30)
            throw std::runtime_error(
                "paired GMRES projected system is singular");
        coefficients[row] = value / diagonal;
    }
}

void combine_basis(
    cublasHandle_t handle,
    int vector_size,
    int columns,
    const double2* basis,
    const std::vector<cdouble>& coefficients,
    double2* device_coefficients,
    double2* output)
{
    cuda_check(
        cudaMemcpy(
            device_coefficients, coefficients.data(),
            static_cast<size_t>(columns) * sizeof(double2),
            cudaMemcpyHostToDevice),
        "upload paired GMRES update coefficients");
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    cublas_check(
        cublasZgemv(
            handle, CUBLAS_OP_N,
            vector_size, columns,
            &one, as_cublas(basis), vector_size,
            as_cublas(device_coefficients), 1,
            &zero, as_cublas(output), 1),
        "paired GMRES basis update");
}

int memory_limited_restart(int requested, int n)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cuda_check(
        cudaMemGetInfo(&free_bytes, &total_bytes),
        "query paired GMRES GPU memory");
    const size_t reserve =
        std::min<size_t>(
            static_cast<size_t>(2048) << 20,
            free_bytes / 3);
    const size_t fixed_vectors =
        static_cast<size_t>(14) * n * sizeof(double2);
    const size_t per_column =
        static_cast<size_t>(2) * n * sizeof(double2);
    if (free_bytes <= reserve + fixed_vectors + 2 * per_column)
        throw std::runtime_error(
            "insufficient GPU memory for paired Muller GMRES");
    const size_t available = free_bytes - reserve - fixed_vectors;
    const int maximum = static_cast<int>(
        std::min<size_t>(
            static_cast<size_t>(requested),
            available / per_column - 1));
    return std::max(2, maximum);
}

} // namespace

MullerPairedGmresResult solve_muller_paired_gmres_device(
    MullerFmmOperator& op,
    const MullerMbjPreconditioner& preconditioner,
    const cdouble* rhs_x,
    const cdouble* rhs_y,
    cdouble* solution_x,
    cdouble* solution_y,
    int restart,
    double tolerance,
    int maximum_iterations,
    bool verbose)
{
    if (!op.device_matvec_available())
        throw std::runtime_error(
            "paired Muller GMRES requires a device FMM matvec");
    if (!preconditioner.device_apply_available())
        throw std::runtime_error(
            "paired Muller GMRES requires GPU-resident MBJ");
    if (restart < 2 || maximum_iterations < 1 || tolerance <= 0.0)
        throw std::invalid_argument("invalid paired Muller GMRES options");

    const auto start = std::chrono::steady_clock::now();
    const int n = op.system_dofs;
    const int effective_restart =
        memory_limited_restart(
            std::min(restart, maximum_iterations), n);
    if (verbose && effective_restart < restart) {
        std::printf(
            "  [Muller paired GPU GMRES] restart %d -> %d "
            "to retain a 2 GiB GPU reserve\n",
            restart, effective_restart);
    }
    restart = effective_restart;
    const size_t vector_bytes =
        static_cast<size_t>(n) * sizeof(double2);
    const size_t basis_bytes =
        static_cast<size_t>(restart + 1) * vector_bytes;

    CublasHandle handle;
    DeviceBuffer b_x(vector_bytes), b_y(vector_bytes);
    DeviceBuffer x_x(vector_bytes), x_y(vector_bytes);
    DeviceBuffer r_x(vector_bytes), r_y(vector_bytes);
    DeviceBuffer w_x(vector_bytes), w_y(vector_bytes);
    DeviceBuffer z_x(vector_bytes), z_y(vector_bytes);
    DeviceBuffer update_x(vector_bytes), update_y(vector_bytes);
    DeviceBuffer basis_x(basis_bytes), basis_y(basis_bytes);
    DeviceBuffer coefficients_x(
        static_cast<size_t>(restart + 1) * sizeof(double2));
    DeviceBuffer coefficients_y(
        static_cast<size_t>(restart + 1) * sizeof(double2));

    cuda_check(
        cudaMemcpy(
            b_x.pointer, rhs_x, vector_bytes, cudaMemcpyHostToDevice),
        "upload paired Muller RHS X");
    cuda_check(
        cudaMemcpy(
            b_y.pointer, rhs_y, vector_bytes, cudaMemcpyHostToDevice),
        "upload paired Muller RHS Y");
    cuda_check(
        cudaMemcpy(
            x_x.pointer, solution_x, vector_bytes,
            cudaMemcpyHostToDevice),
        "upload paired Muller initial guess X");
    cuda_check(
        cudaMemcpy(
            x_y.pointer, solution_y, vector_bytes,
            cudaMemcpyHostToDevice),
        "upload paired Muller initial guess Y");

    const double rhs_norm_x = std::max(
        vector_norm(handle.value, n, b_x.complex_data()), 1.0e-300);
    const double rhs_norm_y = std::max(
        vector_norm(handle.value, n, b_y.complex_data()), 1.0e-300);
    const bool warm =
        vector_norm(handle.value, n, x_x.complex_data()) > 1.0e-30 ||
        vector_norm(handle.value, n, x_y.complex_data()) > 1.0e-30;

    const bool strict_residual = env_flag_enabled(
        "BEM_MIXED_ITERATIVE_REFINEMENT", false);
    const auto apply_for_residual = [&](const void* input_x,
                                        const void* input_y,
                                        void* output_x,
                                        void* output_y) {
        if (strict_residual)
            op.matvec_batch2_device_strict(
                input_x, input_y, output_x, output_y);
        else
            op.matvec_batch2_device(
                input_x, input_y, output_x, output_y);
    };

    MullerPairedGmresResult result;
    if (warm) {
        apply_for_residual(
            x_x.pointer, x_y.pointer, w_x.pointer, w_y.pointer);
        result.operator_evaluations++;
        copy_vector(
            handle.value, n, b_x.complex_data(), r_x.complex_data());
        copy_vector(
            handle.value, n, b_y.complex_data(), r_y.complex_data());
        axpy_vector(
            handle.value, n, cdouble(-1.0),
            w_x.complex_data(), r_x.complex_data());
        axpy_vector(
            handle.value, n, cdouble(-1.0),
            w_y.complex_data(), r_y.complex_data());
    } else {
        copy_vector(
            handle.value, n, b_x.complex_data(), r_x.complex_data());
        copy_vector(
            handle.value, n, b_y.complex_data(), r_y.complex_data());
    }

    double residual_norm_x =
        vector_norm(handle.value, n, r_x.complex_data());
    double residual_norm_y =
        vector_norm(handle.value, n, r_y.complex_data());
    result.initial_residual_x = residual_norm_x / rhs_norm_x;
    result.initial_residual_y = residual_norm_y / rhs_norm_y;
    result.final_residual_x = result.initial_residual_x;
    result.final_residual_y = result.initial_residual_y;
    if (verbose) {
        std::printf(
            "  [Muller paired GPU GMRES] start: X %.3e, Y %.3e%s%s\n",
            result.initial_residual_x,
            result.initial_residual_y,
            warm ? " (warm)" : "",
            strict_residual ? " [FP64 residual refinement]" : "");
        std::fflush(stdout);
    }

    std::vector<cdouble> hessenberg_x(
        static_cast<size_t>(restart + 1) * restart);
    std::vector<cdouble> hessenberg_y(
        static_cast<size_t>(restart + 1) * restart);
    std::vector<cdouble> cosine_x(restart), sine_x(restart);
    std::vector<cdouble> cosine_y(restart), sine_y(restart);
    std::vector<cdouble> projected_x(restart + 1);
    std::vector<cdouble> projected_y(restart + 1);
    std::vector<cdouble> inner_x, inner_y;
    std::vector<cdouble> coefficients_host_x;
    std::vector<cdouble> coefficients_host_y;
    // A mixed-precision correction must be solved substantially more tightly
    // than the FP64 residual target; otherwise operator roundoff can make the
    // outer refinement stagnate above that target.
    const double projected_tolerance =
        tolerance * (strict_residual ? 0.05 : 0.9);

    while (result.iterations < maximum_iterations &&
           (result.final_residual_x >= tolerance ||
            result.final_residual_y >= tolerance)) {
        std::fill(
            hessenberg_x.begin(), hessenberg_x.end(), cdouble(0.0));
        std::fill(
            hessenberg_y.begin(), hessenberg_y.end(), cdouble(0.0));
        std::fill(projected_x.begin(), projected_x.end(), cdouble(0.0));
        std::fill(projected_y.begin(), projected_y.end(), cdouble(0.0));
        scale_copy(
            handle.value, n, 1.0 / residual_norm_x,
            r_x.complex_data(), basis_x.complex_data());
        scale_copy(
            handle.value, n, 1.0 / residual_norm_y,
            r_y.complex_data(), basis_y.complex_data());
        projected_x[0] = residual_norm_x;
        projected_y[0] = residual_norm_y;

        const int cycle_limit = std::min(
            restart, maximum_iterations - result.iterations);
        int cycle_iterations = 0;
        for (int column = 0; column < cycle_limit; column++) {
            double2* vector_x =
                basis_x.complex_data() + static_cast<size_t>(column) * n;
            double2* vector_y =
                basis_y.complex_data() + static_cast<size_t>(column) * n;
            preconditioner.apply_device_complex_pair(
                vector_x, vector_y, z_x.pointer, z_y.pointer);
            op.matvec_batch2_device(
                z_x.pointer, z_y.pointer, w_x.pointer, w_y.pointer);
            result.operator_evaluations++;

            for (int pass = 0; pass < 2; pass++) {
                orthogonalize(
                    handle.value, n, column + 1,
                    basis_x.complex_data(), w_x.complex_data(),
                    coefficients_x.complex_data(), inner_x);
                orthogonalize(
                    handle.value, n, column + 1,
                    basis_y.complex_data(), w_y.complex_data(),
                    coefficients_y.complex_data(), inner_y);
                for (int row = 0; row <= column; row++) {
                    hessenberg_x[
                        static_cast<size_t>(row) * restart + column] +=
                        inner_x[row];
                    hessenberg_y[
                        static_cast<size_t>(row) * restart + column] +=
                        inner_y[row];
                }
            }
            const double next_norm_x =
                vector_norm(handle.value, n, w_x.complex_data());
            const double next_norm_y =
                vector_norm(handle.value, n, w_y.complex_data());
            hessenberg_x[
                static_cast<size_t>(column + 1) * restart + column] =
                next_norm_x;
            hessenberg_y[
                static_cast<size_t>(column + 1) * restart + column] =
                next_norm_y;
            if (next_norm_x > 1.0e-30) {
                scale_copy(
                    handle.value, n, 1.0 / next_norm_x,
                    w_x.complex_data(),
                    basis_x.complex_data() +
                        static_cast<size_t>(column + 1) * n);
            }
            if (next_norm_y > 1.0e-30) {
                scale_copy(
                    handle.value, n, 1.0 / next_norm_y,
                    w_y.complex_data(),
                    basis_y.complex_data() +
                        static_cast<size_t>(column + 1) * n);
            }
            update_projected_column(
                hessenberg_x, cosine_x, sine_x, projected_x,
                restart, column);
            update_projected_column(
                hessenberg_y, cosine_y, sine_y, projected_y,
                restart, column);
            cycle_iterations = column + 1;
            result.iterations++;
            const double projected_relative_x =
                std::abs(projected_x[column + 1]) / rhs_norm_x;
            const double projected_relative_y =
                std::abs(projected_y[column + 1]) / rhs_norm_y;
            if (verbose &&
                (result.iterations == 1 ||
                 result.iterations % 25 == 0)) {
                std::printf(
                    "  [Muller paired GPU GMRES %d] projected: "
                    "X %.3e, Y %.3e\n",
                    result.iterations,
                    projected_relative_x,
                    projected_relative_y);
                std::fflush(stdout);
            }
            if (projected_relative_x < projected_tolerance &&
                projected_relative_y < projected_tolerance) {
                break;
            }
        }

        solve_projected(
            hessenberg_x, projected_x, restart,
            cycle_iterations, coefficients_host_x);
        solve_projected(
            hessenberg_y, projected_y, restart,
            cycle_iterations, coefficients_host_y);
        combine_basis(
            handle.value, n, cycle_iterations,
            basis_x.complex_data(), coefficients_host_x,
            coefficients_x.complex_data(), update_x.complex_data());
        combine_basis(
            handle.value, n, cycle_iterations,
            basis_y.complex_data(), coefficients_host_y,
            coefficients_y.complex_data(), update_y.complex_data());
        preconditioner.apply_device_complex_pair(
            update_x.pointer, update_y.pointer,
            z_x.pointer, z_y.pointer);
        axpy_vector(
            handle.value, n, cdouble(1.0),
            z_x.complex_data(), x_x.complex_data());
        axpy_vector(
            handle.value, n, cdouble(1.0),
            z_y.complex_data(), x_y.complex_data());

        apply_for_residual(
            x_x.pointer, x_y.pointer, w_x.pointer, w_y.pointer);
        result.operator_evaluations++;
        copy_vector(
            handle.value, n, b_x.complex_data(), r_x.complex_data());
        copy_vector(
            handle.value, n, b_y.complex_data(), r_y.complex_data());
        axpy_vector(
            handle.value, n, cdouble(-1.0),
            w_x.complex_data(), r_x.complex_data());
        axpy_vector(
            handle.value, n, cdouble(-1.0),
            w_y.complex_data(), r_y.complex_data());
        residual_norm_x =
            vector_norm(handle.value, n, r_x.complex_data());
        residual_norm_y =
            vector_norm(handle.value, n, r_y.complex_data());
        result.final_residual_x = residual_norm_x / rhs_norm_x;
        result.final_residual_y = residual_norm_y / rhs_norm_y;
        if (verbose) {
            std::printf(
                "  [Muller paired GPU GMRES %d] true residual: "
                "X %.3e, Y %.3e\n",
                result.iterations,
                result.final_residual_x,
                result.final_residual_y);
            std::fflush(stdout);
        }
    }

    result.converged_x = result.final_residual_x < tolerance;
    result.converged_y = result.final_residual_y < tolerance;
    cuda_check(
        cudaMemcpy(
            solution_x, x_x.pointer, vector_bytes,
            cudaMemcpyDeviceToHost),
        "download paired Muller solution X");
    cuda_check(
        cudaMemcpy(
            solution_y, x_y.pointer, vector_bytes,
            cudaMemcpyDeviceToHost),
        "download paired Muller solution Y");
    result.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (verbose) {
        std::printf(
            "  [Muller paired GPU GMRES] %s in %d iterations, "
            "%d paired operator evaluations, %.3fs\n",
            result.converged_x && result.converged_y
                ? "converged" : "not converged",
            result.iterations, result.operator_evaluations,
            result.seconds);
        std::fflush(stdout);
    }
    return result;
}
