#pragma once
// Experimental full nonmagnetic dielectric Dirac transmission formulation.
// Helsing/Karlsson/Rosen, arXiv:2007.12260v4, equations (23)-(35).
// Flat P0/P1/P2/P3 panels, collocation (P0) or reciprocal Galerkin quadrature,
// analytically regularized self gradient. Dense reference storage or sparse
// exact-minus-quadrature corrections for an external eight-charge FMM.
// This is a separate discretization, NOT a replacement for a Muller residual.
#include "mesh.h"
#include <array>
#include <complex>
#include <vector>

namespace bem_dirac {
using C = std::complex<double>;
using V = std::vector<C>;
using Eight = std::array<C,8>;
struct Panel {
    std::array<Vec3,3> vertex;
    Vec3 center,normal,tangent,second;
    double area,diameter;
};
struct Kernel { C s; std::array<C,3> d; }; // ik Phi and source-gradient Phi
struct Scaling { Eight p,np,n,pp; };
Scaling scaling(C index);
std::array<C,8> kernel_action(const Kernel&,const Panel&,const C* local_density);

class Transmission {
    std::vector<Panel> panels_;
    std::array<std::vector<Kernel>,2> kernels_;
    // Eight real components per row, stored separately for contiguous SIMD loads.
    std::array<std::vector<float>,2> kernels32_;
    C wave_[2];
    C index_;
    Scaling scale_;
    int near_order_,far_order_,target_order_,basis_count_;
    std::vector<std::pair<double,double>> near_gauss_;
    double near_radius_;
    bool correction_only_;
    int near_subdiv_;
    // Full coefficient row -> stored representative row; empty for full cache.
    std::vector<int> dense_row_map_;
    std::vector<size_t> near_rows_;
    std::vector<int> near_columns_;
    // Optional pair -> translation-equivalent block; never changes quadrature.
    std::vector<size_t> near_patterns_;
    std::array<Kernel,10> integrate(const Panel&,const Vec3&,C,bool)const;
    std::array<Kernel,100> pair_integrals(int,int,int,const std::vector<std::array<double,3>>&,const std::vector<std::array<double,3>>&)const;
    std::array<Kernel,100> reciprocal_pair(int,int,int,const std::vector<std::array<double,3>>&,const std::vector<std::array<double,3>>&)const;
    V to_charges(const C*,const Eight*)const;
    void cauchy_charges(int,const C*,C*)const;
    void cauchy_charges_simd(int,const C*,C*,bool)const;
public:
    Transmission(const Mesh&,double k,C index,int near_order=14,int far_order=13,
                 double near_radius=2.,size_t max_bytes=size_t(8)<<30,int target_order=1,int degree=0,
                 bool correction_only=false,int near_subdiv=0,bool defer_dense=false);
    int size()const{return int(panels_.size())*basis_count_*8;}
    int basis_count()const{return basis_count_;}
    const std::vector<Panel>& panels()const{return panels_;}
    size_t kernel_bytes()const{return (kernels_[0].size()+kernels_[1].size())*sizeof(Kernel)+(kernels32_[0].size()+kernels32_[1].size())*sizeof(float)+(near_rows_.size()+near_patterns_.size())*sizeof(size_t)+near_columns_.size()*sizeof(int);}
    size_t near_pair_count()const{return near_columns_.size();}
    size_t near_pattern_count()const{return kernels_[0].size()/(basis_count_*basis_count_);}
    const Scaling& coefficients()const{return scale_;}
    void cauchy(int medium,const C* density,C* result)const;
    void apply(const C* density,C* result)const;
    // Same FP64 matrix with explicitly vectorized Cartesian contractions.
    void apply_simd(const C* density,C* result,bool mixed_storage=false)const;
    // RHS-major dense FP64 action. Source tiles remain cached across RHS.
    void apply_batch(int count,const C* density,C* result,int tile=256)const;
    void apply_rows(int count,const C* density,int row_count,const int* rows,C* result,int tile=256)const;
    void build_rows(int panel_count,const int* panel_ids,size_t max_bytes);
    // Sparse exact-minus-quadrature correction on representative panels only.
    void build_near_rows(int panel_count,const int* panel_ids,size_t max_bytes,bool share_translations=false);
    void near_correction_rows(int medium,const C*density,int panel_count,const int*panel_ids,C*result)const;
    void self_blocks_direct(int panel_count,const int*panel_ids,C*result)const;
    // Direct bounded tile for a coarse matrix; no dense pair cache required.
    void direct_transformed_columns(int target_count,const int*targets,int source_count,const int*sources,
        const double*basis,const double*components,C*result)const;
    // Independent exact panel blocks for sparse coarse assembly; no pair cache.
    void direct_panel_pairs(int count,const int*targets,const int*sources,C*result)const;
    void direct_transformed_columns_basis(int target_count,const int*targets,int source_count,const int*sources,
        int coarse_basis,const double*basis,const double*components,C*result)const;
    void apply_rows_reference(const C* density,int row_count,const int* rows,C* result)const;
    // Independent FP64 audit: vectorize across RHS, retain source summation
    // order. Works for arbitrary full-space densities, not only modal vectors.
    void apply_rows_reference_batch(int count,const C* density,int row_count,const int* rows,C* result)const;
    void apply_direct_panels(const C* density,int panel_count,const int* panel_ids,C* result)const;
    void self_blocks_selected(int panel_count,const int* panel_ids,C* result)const;
    // Exact scalar submatrix, coefficient row/column indices; 8 components per
    // coefficient. Row-major output. Works on stored representative rows.
    void submatrix(int row_count,const int* rows,int column_count,const int* columns,C* result)const;
    void transformed_columns(int row_count,const int* rows,int source_count,const int* panels,
        int coarse_basis,const double* basis,const double* components,C* result)const;
    void prepare_mixed_cache(size_t additional_budget);
    // Row-major 8*basis_count square diagonal blocks, one per panel.
    void self_blocks(C* result)const;
    void diagonal_blocks(int panels_per_block,C* result)const;
    void rhs(const C* incident_local,C* result)const;
    void incident(const Vec3&direction,const Vec3&polar,C*result)const;
    void project_fields(const C*sampled_EH,C*result)const;
    // Sparse exact-minus-13-point-quadrature correction for an external FMM.
    // Full cauchy/apply deliberately reject correction-only instances.
    void near_correction(int medium,const C*density,C*result)const;
    void quadrature_charges(const C*density,C*charges)const;
    void project_cauchy(const C*sampled_cartesian_eight,C*result)const;
    void project_cauchy_rows(const C*sampled_cartesian_eight,int count,const int*ids,C*result)const;
    // Cartesian output [scalar,Hx,Hy,Hz,pseudoscalar,Ex,Ey,Ez].
    // Interior H is rescaled to exterior impedance units.
    void field(int medium,const C* density,const std::vector<Vec3>& targets,C* result)const;
    void farfield(const C* density,const std::vector<Vec3>& directions,C* result)const;
};
}
