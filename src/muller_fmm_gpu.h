#ifndef BEM_MULLER_FMM_GPU_H
#define BEM_MULLER_FMM_GPU_H

#include "fmm.h"

#include <complex>
#include <vector>

struct MullerGpuCorrectionValue {
    int column = -1;
    std::complex<double> k1 = 0.0;
    std::complex<double> k2_epsilon = 0.0;
    std::complex<double> k2_mu = 0.0;
};

class MullerGpuAssembly {
public:
    bool initialized = false;
    int current_dofs = 0;
    int regular_points = 0;
    int mass_points = 0;

    void init(
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
        const std::vector<MullerGpuCorrectionValue>& correction_entries);

    void upload_system_input(const std::complex<double>* input);
    void project_charges_and_mass(int input_offset, int slot);

    const double* charge_re(int component) const;
    const double* charge_im(int component) const;

    void assemble_media_and_correction(
        const HelmholtzFMM& exterior,
        const HelmholtzFMM& interior,
        std::complex<double> epsilon_exterior,
        std::complex<double> epsilon_interior,
        std::complex<double> mu_exterior,
        std::complex<double> mu_interior,
        int input_offset,
        int slot);

    void combine_and_download(
        std::complex<double> k_exterior,
        std::complex<double> epsilon_exterior,
        std::complex<double> epsilon_interior,
        std::complex<double> mu_exterior,
        std::complex<double> mu_interior,
        std::complex<double>* output);

    void farfield(
        const std::complex<double>* solution,
        std::complex<double> k_exterior,
        const std::vector<Vec3>& directions,
        std::vector<std::complex<double>>& field);

    void farfield_pair(
        const std::complex<double>* solution_x,
        const std::complex<double>* solution_y,
        std::complex<double> k_exterior,
        const std::vector<Vec3>& directions,
        std::vector<std::complex<double>>& field_x,
        std::vector<std::complex<double>>& field_y);

    void cleanup();

private:
    void* d_input = nullptr;
    void* d_output = nullptr;
    int* d_regular_counts = nullptr;
    int* d_regular_dofs = nullptr;
    double* d_regular_values = nullptr;
    double* d_regular_normals = nullptr;
    double* d_regular_weights = nullptr;
    int* d_mass_counts = nullptr;
    int* d_mass_dofs = nullptr;
    double* d_mass_values = nullptr;
    double* d_mass_positions = nullptr;
    double* d_mass_weights = nullptr;
    int* d_correction_row_offsets = nullptr;
    int* d_correction_columns = nullptr;
    void* d_correction_k1 = nullptr;
    void* d_correction_k2_epsilon = nullptr;
    void* d_correction_k2_mu = nullptr;
    int correction_count = 0;
    double* d_charge_re[3] = {nullptr, nullptr, nullptr};
    double* d_charge_im[3] = {nullptr, nullptr, nullptr};
    double* d_mass_re = nullptr;
    double* d_mass_im = nullptr;
    double* d_k1_re = nullptr;
    double* d_k1_im = nullptr;
    double* d_k2_epsilon_re = nullptr;
    double* d_k2_epsilon_im = nullptr;
    double* d_k2_mu_re = nullptr;
    double* d_k2_mu_im = nullptr;
    double* d_farfield_current_re[2][6] = {
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}
    };
    double* d_farfield_current_im[2][6] = {
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr}
    };
    double* d_farfield_directions = nullptr;
    void* d_farfield_output = nullptr;
    int farfield_capacity = 0;
};

#endif
