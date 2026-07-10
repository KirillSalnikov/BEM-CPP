#ifndef BEM_OUTPUT_H
#define BEM_OUTPUT_H

#include <complex>
#include <cstdio>
#include <stdbool.h>

// Write Mueller matrix and metadata to JSON file.
// M: [16 * ntheta] array, layout M[(i*4+j)*ntheta + t]
// theta: [ntheta] array in radians
void write_json(const char* filename,
                const double* M, const double* theta, int ntheta,
                double ka, double n_re, double n_im, int refinements,
                const char* shape, const char* obj_file,
                double prism_aspect, int edge_refine,
                int n_alpha, int n_beta, int n_gamma, int alpha_avg,
                int orient_start, int orient_count, int orient_total,
                double orientation_weight_sum,
                long long gmres_matvecs,
                int gmres_converged_systems, int gmres_nonconverged_systems,
                int gmres_stagnation_stops, int gmres_numerical_breakdowns,
                int gmres_restored_best_iterates,
                int gmres_max_cycle_exhaustions,
                double gmres_max_final_relres,
                int fmm_digits, int max_leaf, int gmres_restart,
                double gmres_tol, int gmres_max_cycles,
                int requested_fmm_digits, double requested_gmres_tol,
                bool fmm_digits_cli_set, bool gmres_tol_cli_set,
                bool accuracy_policy_adjusted,
                const char* random_orientation_projection,
                const char* farfield_mode,
                const char* solver_backend, const char* solver_profile,
                const char* krylov_solver,
                const char* requested_system_kind, const char* system_kind,
                bool system_canonicalized,
                int quad_order, double unknown_m_scale,
                std::complex<double> row_h_scale,
                double int_op_sign, double k_identity,
                bool preconditioner_enabled, bool schwarz_preconditioner,
                bool device_gmres,
                const char* preconditioner_reason,
                int mesh_vertices, int mesh_triangles, int mesh_skinny_triangles,
                double mesh_min_angle_deg, double mesh_max_aspect_ratio,
                int mesh_feature_edges_30deg,
                double mesh_feature_edge_fraction,
                double mesh_max_dihedral_deg,
                double mesh_mean_feature_dihedral_deg,
                double mesh_max_adjacent_area_ratio,
                bool mesh_near_touch_checked,
                double mesh_near_touch_ratio,
                int mesh_near_touch_pairs,
                int mesh_self_panel_count,
                int mesh_edge_adjacent_pair_count,
                int mesh_vertex_adjacent_pair_count,
                int mesh_near_disjoint_pair_count,
                int mesh_taylor_duffy_candidate_count,
                int mesh_recommended_min_quad_order,
                const char* mesh_recommended_strategy,
                const char* mesh_recommended_action,
                bool mesh_voxel_surface_like,
                bool mesh_requires_remesh,
                int edge_refine_requested, int edge_refine_applied,
                bool edge_refine_uniform_fallback,
                bool mesh_quality_pass,
                double time_assembly, double time_solve, double time_farfield,
                double time_total);

#endif
