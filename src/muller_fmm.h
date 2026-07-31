#ifndef BEM_MULLER_FMM_H
#define BEM_MULLER_FMM_H

#include "fmm.h"
#include "muller_fmm_gpu.h"
#include "muller_nodal.h"
#ifndef BEM_FMM_ONLY
#include "pfft.h"
#endif

#include <array>
#include <complex>
#include <string>
#include <vector>

struct MullerNearCorrectionEntry {
    int column = -1;
    std::complex<double> k1 = 0.0;
    std::complex<double> k2_epsilon = 0.0;
    std::complex<double> k2_mu = 0.0;
};

struct MullerNearCorrection {
    std::vector<int> row_offsets;
    std::vector<MullerNearCorrectionEntry> entries;
};

struct MullerFmmQuadraturePoint {
    int element = -1;
    MullerFrameSample sample;
    double weight = 0.0;
};

struct MullerFmmOperator {
    MullerP2Mesh mesh;
    int current_dofs = 0;
    int system_dofs = 0;
    int quadrature_order = 7;
    std::complex<double> k_exterior;
    std::complex<double> k_interior;
    std::complex<double> epsilon_exterior = 1.0;
    std::complex<double> epsilon_interior;
    std::complex<double> mu_exterior = 1.0;
    std::complex<double> mu_interior = 1.0;
    double geometry_setup_seconds = 0.0;
    double near_correction_setup_seconds = 0.0;
    double fmm_engine_setup_seconds = 0.0;
    int near_correction_colors = 0;
    int near_correction_pairs = 0;
    int near_correction_unique_templates = 0;
    bool near_correction_template_reuse = true;
    bool near_correction_cache_hit = false;
    std::string near_correction_cache_path;
    bool use_pfft = false;
    bool gpu_operator_assembly = false;
    bool gpu_operator_assembly_requested = false;
    bool fmm_near_fp32 = false;
    int fmm_near_radius = 3;
    int pfft_interpolation_order = 2;
    double pfft_correction_radius_cells = 2.0;
    double pfft_grid_safety = 0.96;

    std::vector<MullerFmmQuadraturePoint> quadrature;
    std::vector<MullerFmmQuadraturePoint> mass_quadrature;
    MullerNearCorrection correction;
    HelmholtzFMM fmm_exterior;
    HelmholtzFMM fmm_interior;
#ifndef BEM_FMM_ONLY
    HelmholtzPFFT pfft_exterior;
    HelmholtzPFFT pfft_interior;
#endif

    void init(
        const Mesh& linear_mesh,
        std::complex<double> k_exterior_value,
        std::complex<double> refractive_index,
        bool project_edge_nodes_to_sphere,
        int quadrature_order_value = 7,
        int duffy_order = 4,
        int fmm_digits = 5,
        int max_leaf = 64,
        bool use_pfft_value = false,
        int pfft_order = 2,
        double pfft_correction_radius = 2.0,
        double pfft_grid_safety_value = 0.96,
        const char* correction_cache_path = nullptr,
        int fmm_near_radius_value = 3,
        bool near_template_reuse_value = true);

    void init(
        const Mesh& linear_mesh,
        std::complex<double> k_exterior_value,
        std::complex<double> refractive_index,
        const MullerP2BuildOptions& build_options,
        int quadrature_order_value = 7,
        int duffy_order = 4,
        int fmm_digits = 5,
        int max_leaf = 64,
        bool use_pfft_value = false,
        int pfft_order = 2,
        double pfft_correction_radius = 2.0,
        double pfft_grid_safety_value = 0.96,
        const char* correction_cache_path = nullptr,
        int fmm_near_radius_value = 3,
        bool near_template_reuse_value = true);

    void matvec(
        const std::complex<double>* input,
        std::complex<double>* output);

    // Device-resident action used by the paired orientation solver. The
    // pointers contain system_dofs CUDA double2 values.
    void matvec_device(
        const void* device_input,
        void* device_output);
    void matvec_batch2_device(
        const void* device_input_x,
        const void* device_input_y,
        void* device_output_x,
        void* device_output_y);
    void matvec_batch2_device_strict(
        const void* device_input_x,
        const void* device_input_y,
        void* device_output_x,
        void* device_output_y);
    bool device_matvec_available() const;

    void farfield(
        const std::complex<double>* solution,
        const std::vector<Vec3>& directions,
        std::vector<std::complex<double>>& field);

    void farfield_pair(
        const std::complex<double>* solution_x,
        const std::complex<double>* solution_y,
        const std::vector<Vec3>& directions,
        std::vector<std::complex<double>>& field_x,
        std::vector<std::complex<double>>& field_y);

    // Build exact-reference FMM engines after pFFT setup. Geometry, Duffy
    // correction and MBJ-compatible local blocks remain unchanged. Keeping
    // pFFT permits it to act as an inner solver for flexible GMRES.
    double switch_pfft_to_fmm(
        int fmm_digits, int max_leaf, bool keep_pfft = false);

    void select_pfft_backend();
    void select_fmm_backend();
    void set_fmm_near_fp32(bool enabled);

    // O(Q^2) validation action with the same quadrature/correction split.
    void matvec_direct_reference(
        const std::complex<double>* input,
        std::complex<double>* output);

    void cleanup();

    const char* backend_name() const {
        return use_pfft ? "pfft" : "fmm";
    }

private:
    std::array<std::vector<std::complex<double>>, 3> charges;
    std::array<std::vector<std::complex<double>>, 3> gradient_exterior;
    std::array<std::vector<std::complex<double>>, 3> gradient_interior;
    std::array<std::vector<std::complex<double>>, 3> hessian_exterior;
    std::array<std::vector<std::complex<double>>, 3> hessian_interior;
    std::vector<std::complex<double>> curl_exterior;
    std::vector<std::complex<double>> curl_interior;
    std::vector<std::complex<double>> hessian_action_exterior;
    std::vector<std::complex<double>> hessian_action_interior;
    std::vector<std::complex<double>> mass_work;
    std::vector<std::complex<double>> k1_work;
    std::vector<std::complex<double>> k2_epsilon_work;
    std::vector<std::complex<double>> k2_mu_work;
    std::vector<std::vector<int>> assembly_colors;
    MullerGpuAssembly gpu_assembly;
    int regular_points_per_element = 0;
    int mass_points_per_element = 0;

    void apply_current_operators(
        const std::complex<double>* coefficients,
        std::vector<std::complex<double>>& mass,
        std::vector<std::complex<double>>& k1,
        std::vector<std::complex<double>>& k2_epsilon,
        std::vector<std::complex<double>>& k2_mu);

    void apply_current_operators_direct(
        const std::complex<double>* coefficients,
        std::vector<std::complex<double>>& mass,
        std::vector<std::complex<double>>& k1,
        std::vector<std::complex<double>>& k2_epsilon,
        std::vector<std::complex<double>>& k2_mu);

    void apply_mass(
        const std::complex<double>* coefficients,
        std::vector<std::complex<double>>& mass) const;

    void prepare_gpu_assembly();
    void apply_current_operators_gpu(int input_offset, int slot);
    void apply_current_operator_pair_gpu(bool strict = false);
    void apply_current_operator_quad_gpu();
};

// Assemble one exact Galerkin principal block in paired-current
// [J_0,J_1,M_0,M_1] ordering. A group is a P2 node for the nodal basis
// and a topological edge for BDM1.
std::vector<std::complex<double>> assemble_muller_nodal_block(
    const MullerFmmOperator& op,
    const std::vector<int>& dof_groups,
    const std::vector<int>* support_elements = nullptr);

#endif
