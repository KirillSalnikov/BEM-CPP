#include "muller_fmm_gpu.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr int kMaximumBasisFunctions = 12;

void cuda_check(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess)
        return;
    char message[512];
    std::snprintf(
        message, sizeof(message), "%s: %s",
        operation, cudaGetErrorString(status));
    throw std::runtime_error(message);
}

__device__ double2 complex_add(double2 a, double2 b)
{
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ double2 complex_sub(double2 a, double2 b)
{
    return make_double2(a.x - b.x, a.y - b.y);
}

__device__ double2 complex_mul(double2 a, double2 b)
{
    return make_double2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x);
}

__device__ double2 complex_scale(double2 a, double scale)
{
    return make_double2(a.x * scale, a.y * scale);
}

__device__ double2 load_split(
    const double* real, const double* imaginary, int index)
{
    return make_double2(real[index], imaginary[index]);
}

__device__ void atomic_add_complex(
    double* real, double* imaginary, int index, double2 value)
{
    atomicAdd(real + index, value.x);
    atomicAdd(imaginary + index, value.y);
}

__global__ void project_charges_kernel(
    const double2* input,
    int input_offset,
    const int* counts,
    const int* dofs,
    const double* values,
    const double* weights,
    int point_count,
    double* charge_x_re,
    double* charge_x_im,
    double* charge_y_re,
    double* charge_y_im,
    double* charge_z_re,
    double* charge_z_im)
{
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count)
        return;
    double2 current[3] = {
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0)
    };
    for (int local = 0; local < counts[point]; local++) {
        const int flat = point * kMaximumBasisFunctions + local;
        const double2 coefficient = input[input_offset + dofs[flat]];
        for (int axis = 0; axis < 3; axis++)
            current[axis] = complex_add(
                current[axis],
                complex_scale(
                    coefficient,
                    values[3 * flat + axis]));
    }
    const double weight = weights[point];
    charge_x_re[point] = weight * current[0].x;
    charge_x_im[point] = weight * current[0].y;
    charge_y_re[point] = weight * current[1].x;
    charge_y_im[point] = weight * current[1].y;
    charge_z_re[point] = weight * current[2].x;
    charge_z_im[point] = weight * current[2].y;
}

__global__ void mass_kernel(
    const double2* input,
    int input_offset,
    const int* counts,
    const int* dofs,
    const double* values,
    const double* weights,
    int point_count,
    int output_offset,
    double* output_re,
    double* output_im)
{
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count)
        return;
    double2 current[3] = {
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0)
    };
    const int count = counts[point];
    for (int local = 0; local < count; local++) {
        const int flat = point * kMaximumBasisFunctions + local;
        const double2 coefficient = input[input_offset + dofs[flat]];
        for (int axis = 0; axis < 3; axis++)
            current[axis] = complex_add(
                current[axis],
                complex_scale(
                    coefficient,
                    values[3 * flat + axis]));
    }
    for (int local = 0; local < count; local++) {
        const int flat = point * kMaximumBasisFunctions + local;
        double2 tested = make_double2(0.0, 0.0);
        for (int axis = 0; axis < 3; axis++)
            tested = complex_add(
                tested,
                complex_scale(
                    current[axis],
                    values[3 * flat + axis]));
        atomic_add_complex(
            output_re, output_im,
            output_offset + dofs[flat],
            complex_scale(tested, weights[point]));
    }
}

__global__ void project_farfield_currents_kernel(
    const double2* input,
    int current_dofs,
    const int* counts,
    const int* dofs,
    const double* values,
    const double* weights,
    int point_count,
    double* jx_re,
    double* jx_im,
    double* jy_re,
    double* jy_im,
    double* jz_re,
    double* jz_im,
    double* mx_re,
    double* mx_im,
    double* my_re,
    double* my_im,
    double* mz_re,
    double* mz_im)
{
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count)
        return;
    double2 current_j[3] = {
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0)
    };
    double2 current_m[3] = {
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0),
        make_double2(0.0, 0.0)
    };
    for (int local = 0; local < counts[point]; local++) {
        const int flat = point * kMaximumBasisFunctions + local;
        const int dof = dofs[flat];
        const double2 coefficient_j = input[dof];
        const double2 coefficient_m = input[current_dofs + dof];
        for (int axis = 0; axis < 3; axis++) {
            const double basis = values[3 * flat + axis];
            current_j[axis] = complex_add(
                current_j[axis],
                complex_scale(coefficient_j, basis));
            current_m[axis] = complex_add(
                current_m[axis],
                complex_scale(coefficient_m, basis));
        }
    }
    const double weight = weights[point];
    double* j_re[3] = {jx_re, jy_re, jz_re};
    double* j_im[3] = {jx_im, jy_im, jz_im};
    double* m_re[3] = {mx_re, my_re, mz_re};
    double* m_im[3] = {mx_im, my_im, mz_im};
    for (int axis = 0; axis < 3; axis++) {
        j_re[axis][point] = weight * current_j[axis].x;
        j_im[axis][point] = weight * current_j[axis].y;
        m_re[axis][point] = weight * current_m[axis].x;
        m_im[axis][point] = weight * current_m[axis].y;
    }
}

__global__ void farfield_kernel(
    const double* positions,
    int point_count,
    const double* directions,
    int direction_count,
    const double* jx_re,
    const double* jx_im,
    const double* jy_re,
    const double* jy_im,
    const double* jz_re,
    const double* jz_im,
    const double* mx_re,
    const double* mx_im,
    const double* my_re,
    const double* my_im,
    const double* mz_re,
    const double* mz_im,
    double k_real,
    double k_imaginary,
    double2* output)
{
    const int direction = blockIdx.x;
    if (direction >= direction_count)
        return;
    const double rx = directions[3 * direction];
    const double ry = directions[3 * direction + 1];
    const double rz = directions[3 * direction + 2];
    const double* current_re[6] = {
        jx_re, jy_re, jz_re, mx_re, my_re, mz_re
    };
    const double* current_im[6] = {
        jx_im, jy_im, jz_im, mx_im, my_im, mz_im
    };
    double2 sums[6];
    for (int component = 0; component < 6; component++)
        sums[component] = make_double2(0.0, 0.0);
    for (int point = threadIdx.x;
         point < point_count; point += blockDim.x) {
        const double dot =
            rx * positions[3 * point] +
            ry * positions[3 * point + 1] +
            rz * positions[3 * point + 2];
        double sine;
        double cosine;
        sincos(-k_real * dot, &sine, &cosine);
        const double growth = exp(k_imaginary * dot);
        const double2 phase =
            make_double2(growth * cosine, growth * sine);
        for (int component = 0; component < 6; component++) {
            sums[component] = complex_add(
                sums[component],
                complex_mul(
                    phase,
                    make_double2(
                        current_re[component][point],
                        current_im[component][point])));
        }
    }

    __shared__ double2 shared[6][256];
    for (int component = 0; component < 6; component++)
        shared[component][threadIdx.x] = sums[component];
    __syncthreads();
    for (int stride = blockDim.x / 2;
         stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            for (int component = 0; component < 6; component++) {
                shared[component][threadIdx.x] = complex_add(
                    shared[component][threadIdx.x],
                    shared[component][threadIdx.x + stride]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x != 0)
        return;

    const double2 r_dot_j = complex_add(
        complex_add(
            complex_scale(shared[0][0], rx),
            complex_scale(shared[1][0], ry)),
        complex_scale(shared[2][0], rz));
    const double2 j_perpendicular[3] = {
        complex_sub(shared[0][0], complex_scale(r_dot_j, rx)),
        complex_sub(shared[1][0], complex_scale(r_dot_j, ry)),
        complex_sub(shared[2][0], complex_scale(r_dot_j, rz))
    };
    const double2 r_cross_m[3] = {
        complex_sub(
            complex_scale(shared[5][0], ry),
            complex_scale(shared[4][0], rz)),
        complex_sub(
            complex_scale(shared[3][0], rz),
            complex_scale(shared[5][0], rx)),
        complex_sub(
            complex_scale(shared[4][0], rx),
            complex_scale(shared[3][0], ry))
    };
    const double2 prefactor =
        make_double2(k_imaginary * INV4PI, -k_real * INV4PI);
    for (int axis = 0; axis < 3; axis++) {
        output[3 * direction + axis] = complex_mul(
            prefactor,
            complex_sub(j_perpendicular[axis], r_cross_m[axis]));
    }
}

__global__ void farfield_pair_kernel(
    const double* positions,
    int point_count,
    const double* directions,
    int direction_count,
    const double* x_jx_re,
    const double* x_jx_im,
    const double* x_jy_re,
    const double* x_jy_im,
    const double* x_jz_re,
    const double* x_jz_im,
    const double* x_mx_re,
    const double* x_mx_im,
    const double* x_my_re,
    const double* x_my_im,
    const double* x_mz_re,
    const double* x_mz_im,
    const double* y_jx_re,
    const double* y_jx_im,
    const double* y_jy_re,
    const double* y_jy_im,
    const double* y_jz_re,
    const double* y_jz_im,
    const double* y_mx_re,
    const double* y_mx_im,
    const double* y_my_re,
    const double* y_my_im,
    const double* y_mz_re,
    const double* y_mz_im,
    double k_real,
    double k_imaginary,
    double2* output)
{
    const int direction = blockIdx.x;
    if (direction >= direction_count)
        return;
    const double rx = directions[3 * direction];
    const double ry = directions[3 * direction + 1];
    const double rz = directions[3 * direction + 2];
    const double* current_re[12] = {
        x_jx_re, x_jy_re, x_jz_re, x_mx_re, x_my_re, x_mz_re,
        y_jx_re, y_jy_re, y_jz_re, y_mx_re, y_my_re, y_mz_re
    };
    const double* current_im[12] = {
        x_jx_im, x_jy_im, x_jz_im, x_mx_im, x_my_im, x_mz_im,
        y_jx_im, y_jy_im, y_jz_im, y_mx_im, y_my_im, y_mz_im
    };
    double2 sums[12];
    for (int component = 0; component < 12; component++)
        sums[component] = make_double2(0.0, 0.0);
    for (int point = threadIdx.x;
         point < point_count; point += blockDim.x) {
        const double dot =
            rx * positions[3 * point] +
            ry * positions[3 * point + 1] +
            rz * positions[3 * point + 2];
        double sine;
        double cosine;
        sincos(-k_real * dot, &sine, &cosine);
        const double growth = exp(k_imaginary * dot);
        const double2 phase =
            make_double2(growth * cosine, growth * sine);
        for (int component = 0; component < 12; component++) {
            sums[component] = complex_add(
                sums[component],
                complex_mul(
                    phase,
                    make_double2(
                        current_re[component][point],
                        current_im[component][point])));
        }
    }

    __shared__ double2 shared[12][128];
    for (int component = 0; component < 12; component++)
        shared[component][threadIdx.x] = sums[component];
    __syncthreads();
    for (int stride = blockDim.x / 2;
         stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            for (int component = 0; component < 12; component++) {
                shared[component][threadIdx.x] = complex_add(
                    shared[component][threadIdx.x],
                    shared[component][threadIdx.x + stride]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x != 0)
        return;

    const double2 prefactor =
        make_double2(k_imaginary * INV4PI, -k_real * INV4PI);
    for (int polarization = 0; polarization < 2; polarization++) {
        const int offset = 6 * polarization;
        const double2 r_dot_j = complex_add(
            complex_add(
                complex_scale(shared[offset][0], rx),
                complex_scale(shared[offset + 1][0], ry)),
            complex_scale(shared[offset + 2][0], rz));
        const double2 j_perpendicular[3] = {
            complex_sub(
                shared[offset][0], complex_scale(r_dot_j, rx)),
            complex_sub(
                shared[offset + 1][0], complex_scale(r_dot_j, ry)),
            complex_sub(
                shared[offset + 2][0], complex_scale(r_dot_j, rz))
        };
        const double2 r_cross_m[3] = {
            complex_sub(
                complex_scale(shared[offset + 5][0], ry),
                complex_scale(shared[offset + 4][0], rz)),
            complex_sub(
                complex_scale(shared[offset + 3][0], rz),
                complex_scale(shared[offset + 5][0], rx)),
            complex_sub(
                complex_scale(shared[offset + 4][0], rx),
                complex_scale(shared[offset + 3][0], ry))
        };
        for (int axis = 0; axis < 3; axis++) {
            output[
                3 * (polarization * direction_count + direction) + axis] =
                complex_mul(
                    prefactor,
                    complex_sub(
                        j_perpendicular[axis], r_cross_m[axis]));
        }
    }
}

__global__ void assemble_media_kernel(
    const int* counts,
    const int* dofs,
    const double* values,
    const double* normals,
    const double* weights,
    int point_count,
    const double* exterior_curl_re,
    const double* exterior_curl_im,
    const double* exterior_hessian_re,
    const double* exterior_hessian_im,
    const double* interior_curl_re,
    const double* interior_curl_im,
    const double* interior_hessian_re,
    const double* interior_hessian_im,
    double2 epsilon_exterior,
    double2 epsilon_interior,
    double2 mu_exterior,
    double2 mu_interior,
    int output_offset,
    double* k1_re,
    double* k1_im,
    double* k2_epsilon_re,
    double* k2_epsilon_im,
    double* k2_mu_re,
    double* k2_mu_im)
{
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count)
        return;

    double2 k1_vector[3];
    double2 curl_epsilon[3];
    double2 curl_mu[3];
    for (int component = 0; component < 3; component++) {
        const int index = 3 * point + component;
        const double2 exterior_hessian = load_split(
            exterior_hessian_re, exterior_hessian_im, index);
        const double2 interior_hessian = load_split(
            interior_hessian_re, interior_hessian_im, index);
        const double2 exterior_curl = load_split(
            exterior_curl_re, exterior_curl_im, index);
        const double2 interior_curl = load_split(
            interior_curl_re, interior_curl_im, index);
        k1_vector[component] =
            complex_sub(exterior_hessian, interior_hessian);
        curl_epsilon[component] = complex_sub(
            complex_mul(epsilon_exterior, exterior_curl),
            complex_mul(epsilon_interior, interior_curl));
        curl_mu[component] = complex_sub(
            complex_mul(mu_exterior, exterior_curl),
            complex_mul(mu_interior, interior_curl));
    }

    const double nx = normals[3 * point];
    const double ny = normals[3 * point + 1];
    const double nz = normals[3 * point + 2];
    const double weight = weights[point];
    for (int local = 0; local < counts[point]; local++) {
        const int flat = point * kMaximumBasisFunctions + local;
        const double tx = values[3 * flat];
        const double ty = values[3 * flat + 1];
        const double tz = values[3 * flat + 2];
        const double rotated[3] = {
            ty * nz - tz * ny,
            tz * nx - tx * nz,
            tx * ny - ty * nx
        };
        double2 k1_value = make_double2(0.0, 0.0);
        for (int axis = 0; axis < 3; axis++)
            k1_value = complex_add(
                k1_value,
                complex_scale(k1_vector[axis], rotated[axis]));

        const double curl_coefficients[3] = {
            tx * ny - nx * ty,
            tx * nz - nx * tz,
            ty * nz - ny * tz
        };
        double2 k2_epsilon_value = make_double2(0.0, 0.0);
        double2 k2_mu_value = make_double2(0.0, 0.0);
        for (int component = 0; component < 3; component++) {
            k2_epsilon_value = complex_add(
                k2_epsilon_value,
                complex_scale(
                    curl_epsilon[component],
                    curl_coefficients[component]));
            k2_mu_value = complex_add(
                k2_mu_value,
                complex_scale(
                    curl_mu[component],
                    curl_coefficients[component]));
        }
        const int row = output_offset + dofs[flat];
        atomic_add_complex(
            k1_re, k1_im, row,
            complex_scale(k1_value, weight));
        atomic_add_complex(
            k2_epsilon_re, k2_epsilon_im, row,
            complex_scale(k2_epsilon_value, weight));
        atomic_add_complex(
            k2_mu_re, k2_mu_im, row,
            complex_scale(k2_mu_value, weight));
    }
}

__global__ void correction_kernel(
    const double2* input,
    int input_offset,
    const int* row_offsets,
    const int* columns,
    const double2* correction_k1,
    const double2* correction_k2_epsilon,
    const double2* correction_k2_mu,
    int current_dofs,
    int output_offset,
    double* k1_re,
    double* k1_im,
    double* k2_epsilon_re,
    double* k2_epsilon_im,
    double* k2_mu_re,
    double* k2_mu_im)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= current_dofs)
        return;
    double2 sum_k1 = make_double2(0.0, 0.0);
    double2 sum_k2_epsilon = make_double2(0.0, 0.0);
    double2 sum_k2_mu = make_double2(0.0, 0.0);
    for (int index = row_offsets[row];
         index < row_offsets[row + 1]; index++) {
        const double2 value = input[input_offset + columns[index]];
        sum_k1 = complex_add(
            sum_k1, complex_mul(correction_k1[index], value));
        sum_k2_epsilon = complex_add(
            sum_k2_epsilon,
            complex_mul(correction_k2_epsilon[index], value));
        sum_k2_mu = complex_add(
            sum_k2_mu,
            complex_mul(correction_k2_mu[index], value));
    }
    const int output = output_offset + row;
    k1_re[output] += sum_k1.x;
    k1_im[output] += sum_k1.y;
    k2_epsilon_re[output] += sum_k2_epsilon.x;
    k2_epsilon_im[output] += sum_k2_epsilon.y;
    k2_mu_re[output] += sum_k2_mu.x;
    k2_mu_im[output] += sum_k2_mu.y;
}

__global__ void combine_muller_kernel(
    int current_dofs,
    const double* mass_re,
    const double* mass_im,
    const double* k1_re,
    const double* k1_im,
    const double* k2_epsilon_re,
    const double* k2_epsilon_im,
    const double* k2_mu_re,
    const double* k2_mu_im,
    double2 imaginary_over_k,
    double2 half_epsilon_sum,
    double2 half_mu_sum,
    double2* output)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= current_dofs)
        return;
    const int j = row;
    const int m = current_dofs + row;
    const double2 k1_j = load_split(k1_re, k1_im, j);
    const double2 k1_m = load_split(k1_re, k1_im, m);
    const double2 mass_j = load_split(mass_re, mass_im, j);
    const double2 mass_m = load_split(mass_re, mass_im, m);
    const double2 k2e_m =
        load_split(k2_epsilon_re, k2_epsilon_im, m);
    const double2 k2m_j = load_split(k2_mu_re, k2_mu_im, j);
    output[row] = complex_add(
        complex_add(
            complex_mul(imaginary_over_k, k1_j),
            complex_mul(half_epsilon_sum, mass_m)),
        k2e_m);
    output[current_dofs + row] = complex_sub(
        complex_add(
            complex_mul(half_mu_sum, mass_j),
            k2m_j),
        complex_mul(imaginary_over_k, k1_m));
}

template <typename T>
void allocate_and_upload(
    T** device,
    const std::vector<T>& host,
    const char* operation)
{
    if (host.empty()) {
        *device = nullptr;
        return;
    }
    cuda_check(
        cudaMalloc(
            reinterpret_cast<void**>(device),
            host.size() * sizeof(T)),
        operation);
    cuda_check(
        cudaMemcpy(
            *device, host.data(), host.size() * sizeof(T),
            cudaMemcpyHostToDevice),
        operation);
}

double2 to_double2(std::complex<double> value)
{
    return make_double2(value.real(), value.imag());
}

} // namespace

void MullerGpuAssembly::init(
    int current_dofs_value,
    const std::vector<int>& regular_counts,
    const std::vector<int>& regular_dofs,
    const std::vector<double>& regular_values,
    const std::vector<double>& regular_normals,
    const std::vector<double>& regular_weights,
    const std::vector<int>& mass_counts,
    const std::vector<int>& mass_dofs,
    const std::vector<double>& mass_values,
    const std::vector<double>& mass_positions,
    const std::vector<double>& mass_weights,
    const std::vector<int>& correction_row_offsets,
    const std::vector<MullerGpuCorrectionValue>& correction_entries)
{
    cleanup();
    if (current_dofs_value <= 0 ||
        regular_counts.empty() || mass_counts.empty())
        throw std::invalid_argument("invalid Muller GPU assembly dimensions");
    if (regular_dofs.size() !=
            regular_counts.size() * kMaximumBasisFunctions ||
        regular_values.size() != regular_dofs.size() * 3 ||
        regular_normals.size() != regular_counts.size() * 3 ||
        regular_weights.size() != regular_counts.size() ||
        mass_dofs.size() !=
            mass_counts.size() * kMaximumBasisFunctions ||
        mass_values.size() != mass_dofs.size() * 3 ||
        mass_positions.size() != mass_counts.size() * 3 ||
        mass_weights.size() != mass_counts.size() ||
        correction_row_offsets.size() !=
            static_cast<size_t>(current_dofs_value + 1))
        throw std::invalid_argument("inconsistent Muller GPU assembly data");

    current_dofs = current_dofs_value;
    regular_points = static_cast<int>(regular_counts.size());
    mass_points = static_cast<int>(mass_counts.size());
    correction_count = static_cast<int>(correction_entries.size());

    allocate_and_upload(
        &d_regular_counts, regular_counts, "upload regular counts");
    allocate_and_upload(
        &d_regular_dofs, regular_dofs, "upload regular dofs");
    allocate_and_upload(
        &d_regular_values, regular_values, "upload regular basis");
    allocate_and_upload(
        &d_regular_normals, regular_normals, "upload regular normals");
    allocate_and_upload(
        &d_regular_weights, regular_weights, "upload regular weights");
    allocate_and_upload(&d_mass_counts, mass_counts, "upload mass counts");
    allocate_and_upload(&d_mass_dofs, mass_dofs, "upload mass dofs");
    allocate_and_upload(&d_mass_values, mass_values, "upload mass basis");
    allocate_and_upload(
        &d_mass_positions, mass_positions, "upload mass positions");
    allocate_and_upload(&d_mass_weights, mass_weights, "upload mass weights");
    allocate_and_upload(
        &d_correction_row_offsets, correction_row_offsets,
        "upload correction row offsets");

    std::vector<int> correction_columns(correction_count);
    std::vector<double2> correction_k1(correction_count);
    std::vector<double2> correction_k2_epsilon(correction_count);
    std::vector<double2> correction_k2_mu(correction_count);
    for (int index = 0; index < correction_count; index++) {
        correction_columns[index] = correction_entries[index].column;
        correction_k1[index] = to_double2(correction_entries[index].k1);
        correction_k2_epsilon[index] =
            to_double2(correction_entries[index].k2_epsilon);
        correction_k2_mu[index] =
            to_double2(correction_entries[index].k2_mu);
    }
    allocate_and_upload(
        &d_correction_columns, correction_columns,
        "upload correction columns");
    double2* correction_k1_device = nullptr;
    double2* correction_k2_epsilon_device = nullptr;
    double2* correction_k2_mu_device = nullptr;
    allocate_and_upload(
        &correction_k1_device, correction_k1,
        "upload correction k1");
    allocate_and_upload(
        &correction_k2_epsilon_device, correction_k2_epsilon,
        "upload correction k2 epsilon");
    allocate_and_upload(
        &correction_k2_mu_device, correction_k2_mu,
        "upload correction k2 mu");
    d_correction_k1 = correction_k1_device;
    d_correction_k2_epsilon = correction_k2_epsilon_device;
    d_correction_k2_mu = correction_k2_mu_device;

    cuda_check(
        cudaMalloc(&d_input, 2 * current_dofs * sizeof(double2)),
        "allocate Muller input");
    cuda_check(
        cudaMalloc(&d_output, 2 * current_dofs * sizeof(double2)),
        "allocate Muller output");
    for (int component = 0; component < 3; component++) {
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_charge_re[component]),
                regular_points * sizeof(double)),
            "allocate Muller charge real");
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_charge_im[component]),
                regular_points * sizeof(double)),
            "allocate Muller charge imaginary");
    }
    const size_t current_bytes =
        static_cast<size_t>(2 * current_dofs) * sizeof(double);
    double** arrays[] = {
        &d_mass_re, &d_mass_im, &d_k1_re, &d_k1_im,
        &d_k2_epsilon_re, &d_k2_epsilon_im,
        &d_k2_mu_re, &d_k2_mu_im
    };
    for (double** array : arrays)
        cuda_check(
            cudaMalloc(reinterpret_cast<void**>(array), current_bytes),
            "allocate Muller operator work array");
    for (int polarization = 0; polarization < 2; polarization++) {
        for (int component = 0; component < 6; component++) {
            cuda_check(
                cudaMalloc(
                    reinterpret_cast<void**>(
                        &d_farfield_current_re[polarization][component]),
                    static_cast<size_t>(mass_points) * sizeof(double)),
                "allocate farfield current real");
            cuda_check(
                cudaMalloc(
                    reinterpret_cast<void**>(
                        &d_farfield_current_im[polarization][component]),
                    static_cast<size_t>(mass_points) * sizeof(double)),
                "allocate farfield current imaginary");
        }
    }
    initialized = true;
}

void MullerGpuAssembly::upload_system_input(
    const std::complex<double>* input)
{
    if (!initialized)
        throw std::runtime_error("Muller GPU assembly is not initialized");
    static_assert(
        sizeof(std::complex<double>) == sizeof(double2),
        "std::complex<double> must match CUDA double2");
    cuda_check(
        cudaMemcpy(
            d_input, input,
            static_cast<size_t>(2 * current_dofs) * sizeof(double2),
            cudaMemcpyHostToDevice),
        "upload Muller system vector");
    const size_t bytes =
        static_cast<size_t>(2 * current_dofs) * sizeof(double);
    double* arrays[] = {
        d_mass_re, d_mass_im, d_k1_re, d_k1_im,
        d_k2_epsilon_re, d_k2_epsilon_im,
        d_k2_mu_re, d_k2_mu_im
    };
    for (double* array : arrays)
        cuda_check(cudaMemset(array, 0, bytes), "clear Muller work array");
}

void MullerGpuAssembly::project_charges_and_mass(
    int input_offset, int slot)
{
    if (!initialized || slot < 0 || slot > 1)
        throw std::invalid_argument("invalid Muller GPU assembly slot");
    const int block = 256;
    const int regular_grid = (regular_points + block - 1) / block;
    project_charges_kernel<<<regular_grid, block>>>(
        static_cast<const double2*>(d_input),
        input_offset,
        d_regular_counts,
        d_regular_dofs,
        d_regular_values,
        d_regular_weights,
        regular_points,
        d_charge_re[0], d_charge_im[0],
        d_charge_re[1], d_charge_im[1],
        d_charge_re[2], d_charge_im[2]);
    cuda_check(cudaGetLastError(), "project Muller charges");
    const int mass_grid = (mass_points + block - 1) / block;
    mass_kernel<<<mass_grid, block>>>(
        static_cast<const double2*>(d_input),
        input_offset,
        d_mass_counts,
        d_mass_dofs,
        d_mass_values,
        d_mass_weights,
        mass_points,
        slot * current_dofs,
        d_mass_re,
        d_mass_im);
    cuda_check(cudaGetLastError(), "assemble Muller mass");
    cuda_check(
        cudaStreamSynchronize(0),
        "synchronize Muller projection");
}

const double* MullerGpuAssembly::charge_re(int component) const
{
    return component >= 0 && component < 3
        ? d_charge_re[component] : nullptr;
}

const double* MullerGpuAssembly::charge_im(int component) const
{
    return component >= 0 && component < 3
        ? d_charge_im[component] : nullptr;
}

void MullerGpuAssembly::assemble_media_and_correction(
    const HelmholtzFMM& exterior,
    const HelmholtzFMM& interior,
    std::complex<double> epsilon_exterior,
    std::complex<double> epsilon_interior,
    std::complex<double> mu_exterior,
    std::complex<double> mu_interior,
    int input_offset,
    int slot)
{
    if (!initialized || slot < 0 || slot > 1)
        throw std::invalid_argument("invalid Muller GPU assembly slot");
    const int block = 256;
    const int regular_grid = (regular_points + block - 1) / block;
    assemble_media_kernel<<<regular_grid, block>>>(
        d_regular_counts,
        d_regular_dofs,
        d_regular_values,
        d_regular_normals,
        d_regular_weights,
        regular_points,
        exterior.d_grad_re,
        exterior.d_grad_im,
        exterior.d_hess_re,
        exterior.d_hess_im,
        interior.d_grad_re,
        interior.d_grad_im,
        interior.d_hess_re,
        interior.d_hess_im,
        to_double2(epsilon_exterior),
        to_double2(epsilon_interior),
        to_double2(mu_exterior),
        to_double2(mu_interior),
        slot * current_dofs,
        d_k1_re,
        d_k1_im,
        d_k2_epsilon_re,
        d_k2_epsilon_im,
        d_k2_mu_re,
        d_k2_mu_im);
    cuda_check(cudaGetLastError(), "assemble Muller media");

    const int row_grid = (current_dofs + block - 1) / block;
    correction_kernel<<<row_grid, block>>>(
        static_cast<const double2*>(d_input),
        input_offset,
        d_correction_row_offsets,
        d_correction_columns,
        static_cast<const double2*>(d_correction_k1),
        static_cast<const double2*>(d_correction_k2_epsilon),
        static_cast<const double2*>(d_correction_k2_mu),
        current_dofs,
        slot * current_dofs,
        d_k1_re,
        d_k1_im,
        d_k2_epsilon_re,
        d_k2_epsilon_im,
        d_k2_mu_re,
        d_k2_mu_im);
    cuda_check(cudaGetLastError(), "apply Muller near correction");
    cuda_check(
        cudaStreamSynchronize(0),
        "synchronize Muller GPU assembly");
}

void MullerGpuAssembly::combine_and_download(
    std::complex<double> k_exterior,
    std::complex<double> epsilon_exterior,
    std::complex<double> epsilon_interior,
    std::complex<double> mu_exterior,
    std::complex<double> mu_interior,
    std::complex<double>* output)
{
    if (!initialized)
        throw std::runtime_error("Muller GPU assembly is not initialized");
    const int block = 256;
    const int grid = (current_dofs + block - 1) / block;
    const std::complex<double> imaginary_over_k =
        std::complex<double>(0.0, 1.0) / k_exterior;
    combine_muller_kernel<<<grid, block>>>(
        current_dofs,
        d_mass_re,
        d_mass_im,
        d_k1_re,
        d_k1_im,
        d_k2_epsilon_re,
        d_k2_epsilon_im,
        d_k2_mu_re,
        d_k2_mu_im,
        to_double2(imaginary_over_k),
        to_double2(0.5 * (epsilon_interior + epsilon_exterior)),
        to_double2(0.5 * (mu_interior + mu_exterior)),
        static_cast<double2*>(d_output));
    cuda_check(cudaGetLastError(), "combine Muller system action");
    cuda_check(
        cudaMemcpy(
            output, d_output,
            static_cast<size_t>(2 * current_dofs) * sizeof(double2),
            cudaMemcpyDeviceToHost),
        "download Muller system action");
}

void MullerGpuAssembly::farfield(
    const std::complex<double>* solution,
    std::complex<double> k_exterior,
    const std::vector<Vec3>& directions,
    std::vector<std::complex<double>>& field)
{
    if (!initialized)
        throw std::runtime_error("Muller GPU assembly is not initialized");
    if (directions.empty()) {
        field.clear();
        return;
    }
    cuda_check(
        cudaMemcpy(
            d_input, solution,
            static_cast<size_t>(2 * current_dofs) * sizeof(double2),
            cudaMemcpyHostToDevice),
        "upload Muller farfield solution");
    const int block = 256;
    const int point_grid = (mass_points + block - 1) / block;
    project_farfield_currents_kernel<<<point_grid, block>>>(
        static_cast<const double2*>(d_input),
        current_dofs,
        d_mass_counts,
        d_mass_dofs,
        d_mass_values,
        d_mass_weights,
        mass_points,
        d_farfield_current_re[0][0],
        d_farfield_current_im[0][0],
        d_farfield_current_re[0][1],
        d_farfield_current_im[0][1],
        d_farfield_current_re[0][2],
        d_farfield_current_im[0][2],
        d_farfield_current_re[0][3],
        d_farfield_current_im[0][3],
        d_farfield_current_re[0][4],
        d_farfield_current_im[0][4],
        d_farfield_current_re[0][5],
        d_farfield_current_im[0][5]);
    cuda_check(cudaGetLastError(), "project Muller farfield currents");

    const int direction_count = static_cast<int>(directions.size());
    if (direction_count > farfield_capacity) {
        cudaFree(d_farfield_directions);
        cudaFree(d_farfield_output);
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_farfield_directions),
                static_cast<size_t>(3 * direction_count) * sizeof(double)),
            "allocate farfield directions");
        cuda_check(
            cudaMalloc(
                &d_farfield_output,
                    static_cast<size_t>(6 * direction_count) *
                    sizeof(double2)),
            "allocate farfield output");
        farfield_capacity = direction_count;
    }
    std::vector<double> flat_directions(
        static_cast<size_t>(3 * direction_count));
    for (int index = 0; index < direction_count; index++) {
        flat_directions[3 * index] = directions[index].x;
        flat_directions[3 * index + 1] = directions[index].y;
        flat_directions[3 * index + 2] = directions[index].z;
    }
    cuda_check(
        cudaMemcpy(
            d_farfield_directions,
            flat_directions.data(),
            flat_directions.size() * sizeof(double),
            cudaMemcpyHostToDevice),
        "upload farfield directions");
    farfield_kernel<<<direction_count, block>>>(
        d_mass_positions,
        mass_points,
        d_farfield_directions,
        direction_count,
        d_farfield_current_re[0][0],
        d_farfield_current_im[0][0],
        d_farfield_current_re[0][1],
        d_farfield_current_im[0][1],
        d_farfield_current_re[0][2],
        d_farfield_current_im[0][2],
        d_farfield_current_re[0][3],
        d_farfield_current_im[0][3],
        d_farfield_current_re[0][4],
        d_farfield_current_im[0][4],
        d_farfield_current_re[0][5],
        d_farfield_current_im[0][5],
        k_exterior.real(),
        k_exterior.imag(),
        static_cast<double2*>(d_farfield_output));
    cuda_check(cudaGetLastError(), "evaluate Muller farfield");
    field.resize(static_cast<size_t>(3 * direction_count));
    cuda_check(
        cudaMemcpy(
            field.data(),
            d_farfield_output,
            field.size() * sizeof(double2),
            cudaMemcpyDeviceToHost),
        "download Muller farfield");
}

void MullerGpuAssembly::farfield_pair(
    const std::complex<double>* solution_x,
    const std::complex<double>* solution_y,
    std::complex<double> k_exterior,
    const std::vector<Vec3>& directions,
    std::vector<std::complex<double>>& field_x,
    std::vector<std::complex<double>>& field_y)
{
    if (!initialized)
        throw std::runtime_error("Muller GPU assembly is not initialized");
    if (directions.empty()) {
        field_x.clear();
        field_y.clear();
        return;
    }
    const int block = 256;
    const int point_grid = (mass_points + block - 1) / block;
    const std::complex<double>* solutions[2] = {solution_x, solution_y};
    for (int polarization = 0; polarization < 2; polarization++) {
        cuda_check(
            cudaMemcpy(
                d_input, solutions[polarization],
                static_cast<size_t>(2 * current_dofs) * sizeof(double2),
                cudaMemcpyHostToDevice),
            "upload Muller paired farfield solution");
        project_farfield_currents_kernel<<<point_grid, block>>>(
            static_cast<const double2*>(d_input),
            current_dofs,
            d_mass_counts,
            d_mass_dofs,
            d_mass_values,
            d_mass_weights,
            mass_points,
            d_farfield_current_re[polarization][0],
            d_farfield_current_im[polarization][0],
            d_farfield_current_re[polarization][1],
            d_farfield_current_im[polarization][1],
            d_farfield_current_re[polarization][2],
            d_farfield_current_im[polarization][2],
            d_farfield_current_re[polarization][3],
            d_farfield_current_im[polarization][3],
            d_farfield_current_re[polarization][4],
            d_farfield_current_im[polarization][4],
            d_farfield_current_re[polarization][5],
            d_farfield_current_im[polarization][5]);
        cuda_check(
            cudaGetLastError(),
            "project paired Muller farfield currents");
    }

    const int direction_count = static_cast<int>(directions.size());
    if (direction_count > farfield_capacity) {
        cudaFree(d_farfield_directions);
        cudaFree(d_farfield_output);
        cuda_check(
            cudaMalloc(
                reinterpret_cast<void**>(&d_farfield_directions),
                static_cast<size_t>(3 * direction_count) * sizeof(double)),
            "allocate paired farfield directions");
        cuda_check(
            cudaMalloc(
                &d_farfield_output,
                static_cast<size_t>(6 * direction_count) *
                    sizeof(double2)),
            "allocate paired farfield output");
        farfield_capacity = direction_count;
    }
    std::vector<double> flat_directions(
        static_cast<size_t>(3 * direction_count));
    for (int index = 0; index < direction_count; index++) {
        flat_directions[3 * index] = directions[index].x;
        flat_directions[3 * index + 1] = directions[index].y;
        flat_directions[3 * index + 2] = directions[index].z;
    }
    cuda_check(
        cudaMemcpy(
            d_farfield_directions,
            flat_directions.data(),
            flat_directions.size() * sizeof(double),
            cudaMemcpyHostToDevice),
        "upload paired farfield directions");
    const int reduction_block = 128;
    farfield_pair_kernel<<<direction_count, reduction_block>>>(
        d_mass_positions,
        mass_points,
        d_farfield_directions,
        direction_count,
        d_farfield_current_re[0][0],
        d_farfield_current_im[0][0],
        d_farfield_current_re[0][1],
        d_farfield_current_im[0][1],
        d_farfield_current_re[0][2],
        d_farfield_current_im[0][2],
        d_farfield_current_re[0][3],
        d_farfield_current_im[0][3],
        d_farfield_current_re[0][4],
        d_farfield_current_im[0][4],
        d_farfield_current_re[0][5],
        d_farfield_current_im[0][5],
        d_farfield_current_re[1][0],
        d_farfield_current_im[1][0],
        d_farfield_current_re[1][1],
        d_farfield_current_im[1][1],
        d_farfield_current_re[1][2],
        d_farfield_current_im[1][2],
        d_farfield_current_re[1][3],
        d_farfield_current_im[1][3],
        d_farfield_current_re[1][4],
        d_farfield_current_im[1][4],
        d_farfield_current_re[1][5],
        d_farfield_current_im[1][5],
        k_exterior.real(),
        k_exterior.imag(),
        static_cast<double2*>(d_farfield_output));
    cuda_check(cudaGetLastError(), "evaluate paired Muller farfield");

    std::vector<std::complex<double>> combined(
        static_cast<size_t>(6 * direction_count));
    cuda_check(
        cudaMemcpy(
            combined.data(),
            d_farfield_output,
            combined.size() * sizeof(double2),
            cudaMemcpyDeviceToHost),
        "download paired Muller farfield");
    const size_t field_size = static_cast<size_t>(3 * direction_count);
    field_x.assign(combined.begin(), combined.begin() + field_size);
    field_y.assign(combined.begin() + field_size, combined.end());
}

void MullerGpuAssembly::cleanup()
{
    if (!initialized && d_input == nullptr)
        return;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_regular_counts);
    cudaFree(d_regular_dofs);
    cudaFree(d_regular_values);
    cudaFree(d_regular_normals);
    cudaFree(d_regular_weights);
    cudaFree(d_mass_counts);
    cudaFree(d_mass_dofs);
    cudaFree(d_mass_values);
    cudaFree(d_mass_positions);
    cudaFree(d_mass_weights);
    cudaFree(d_correction_row_offsets);
    cudaFree(d_correction_columns);
    cudaFree(d_correction_k1);
    cudaFree(d_correction_k2_epsilon);
    cudaFree(d_correction_k2_mu);
    for (int component = 0; component < 3; component++) {
        cudaFree(d_charge_re[component]);
        cudaFree(d_charge_im[component]);
        d_charge_re[component] = nullptr;
        d_charge_im[component] = nullptr;
    }
    cudaFree(d_mass_re);
    cudaFree(d_mass_im);
    cudaFree(d_k1_re);
    cudaFree(d_k1_im);
    cudaFree(d_k2_epsilon_re);
    cudaFree(d_k2_epsilon_im);
    cudaFree(d_k2_mu_re);
    cudaFree(d_k2_mu_im);
    for (int polarization = 0; polarization < 2; polarization++) {
        for (int component = 0; component < 6; component++) {
            cudaFree(d_farfield_current_re[polarization][component]);
            cudaFree(d_farfield_current_im[polarization][component]);
            d_farfield_current_re[polarization][component] = nullptr;
            d_farfield_current_im[polarization][component] = nullptr;
        }
    }
    cudaFree(d_farfield_directions);
    cudaFree(d_farfield_output);
    d_input = nullptr;
    d_output = nullptr;
    d_regular_counts = nullptr;
    d_regular_dofs = nullptr;
    d_regular_values = nullptr;
    d_regular_normals = nullptr;
    d_regular_weights = nullptr;
    d_mass_counts = nullptr;
    d_mass_dofs = nullptr;
    d_mass_values = nullptr;
    d_mass_positions = nullptr;
    d_mass_weights = nullptr;
    d_correction_row_offsets = nullptr;
    d_correction_columns = nullptr;
    d_correction_k1 = nullptr;
    d_correction_k2_epsilon = nullptr;
    d_correction_k2_mu = nullptr;
    d_mass_re = nullptr;
    d_mass_im = nullptr;
    d_k1_re = nullptr;
    d_k1_im = nullptr;
    d_k2_epsilon_re = nullptr;
    d_k2_epsilon_im = nullptr;
    d_k2_mu_re = nullptr;
    d_k2_mu_im = nullptr;
    d_farfield_directions = nullptr;
    d_farfield_output = nullptr;
    farfield_capacity = 0;
    correction_count = 0;
    current_dofs = 0;
    regular_points = 0;
    mass_points = 0;
    initialized = false;
}
