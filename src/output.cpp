#include "output.h"
#include <cstdio>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>

static bool ensure_parent_dir(const char* filename)
{
    const char* slash = strrchr(filename, '/');
    if (!slash)
        return true;

    size_t len = (size_t)(slash - filename);
    if (len == 0)
        return true;

    char path[4096];
    if (len >= sizeof(path)) {
        fprintf(stderr, "Output path is too long: %s\n", filename);
        return false;
    }
    memcpy(path, filename, len);
    path[len] = '\0';

    for (char* p = path + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = '\0';
        if (mkdir(path, 0775) != 0 && errno != EEXIST) {
            fprintf(stderr, "Cannot create output directory %s: %s\n", path, strerror(errno));
            return false;
        }
        *p = '/';
    }
    if (mkdir(path, 0775) != 0 && errno != EEXIST) {
        fprintf(stderr, "Cannot create output directory %s: %s\n", path, strerror(errno));
        return false;
    }
    return true;
}

static void write_json_string(FILE* f, const char* value)
{
    fputc('"', f);
    if (value) {
        for (const unsigned char* p = (const unsigned char*)value; *p; ++p) {
            switch (*p) {
            case '"': fputs("\\\"", f); break;
            case '\\': fputs("\\\\", f); break;
            case '\b': fputs("\\b", f); break;
            case '\f': fputs("\\f", f); break;
            case '\n': fputs("\\n", f); break;
            case '\r': fputs("\\r", f); break;
            case '\t': fputs("\\t", f); break;
            default:
                if (*p < 0x20)
                    fprintf(f, "\\u%04x", (unsigned int)*p);
                else
                    fputc(*p, f);
                break;
            }
        }
    }
    fputc('"', f);
}

static void write_json_key_string(FILE* f, const char* key, const char* value, bool comma)
{
    fprintf(f, "    \"%s\": ", key);
    write_json_string(f, value ? value : "");
    fprintf(f, "%s\n", comma ? "," : "");
}

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
                double time_total)
{
    ensure_parent_dir(filename);
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", filename);
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"ka\": %.6f,\n", ka);
    fprintf(f, "  \"ri\": [%.6f, %.6f],\n", n_re, n_im);
    fprintf(f, "  \"refinements\": %d,\n", refinements);
    fprintf(f, "  \"shape\": ");
    write_json_string(f, shape ? shape : "unknown");
    fprintf(f, ",\n");
    if (obj_file) {
        fprintf(f, "  \"obj_file\": ");
        write_json_string(f, obj_file);
        fprintf(f, ",\n");
    } else {
        fprintf(f, "  \"obj_file\": null,\n");
    }
    fprintf(f, "  \"prism_aspect\": %.17g,\n", prism_aspect);
    fprintf(f, "  \"edge_refine\": %d,\n", edge_refine);
    fprintf(f, "  \"orient\": [%d, %d, %d],\n", n_alpha, n_beta, n_gamma);
    fprintf(f, "  \"alpha_avg\": %d,\n", alpha_avg);
    fprintf(f, "  \"orient_start\": %d,\n", orient_start);
    fprintf(f, "  \"orient_count\": %d,\n", orient_count);
    fprintf(f, "  \"orient_total\": %d,\n", orient_total);
    fprintf(f, "  \"orientation_weight_sum\": %.17g,\n", orientation_weight_sum);
    fprintf(f, "  \"random_orientation_projection\": ");
    write_json_string(f, random_orientation_projection ? random_orientation_projection : "unknown");
    fprintf(f, ",\n");
    fprintf(f, "  \"gmres_matvecs\": %lld,\n", gmres_matvecs);
    fprintf(f, "  \"gmres_converged_systems\": %d,\n", gmres_converged_systems);
    fprintf(f, "  \"gmres_nonconverged_systems\": %d,\n", gmres_nonconverged_systems);
    fprintf(f, "  \"gmres_stagnation_stops\": %d,\n", gmres_stagnation_stops);
    fprintf(f, "  \"gmres_numerical_breakdowns\": %d,\n", gmres_numerical_breakdowns);
    fprintf(f, "  \"gmres_restored_best_iterates\": %d,\n", gmres_restored_best_iterates);
    fprintf(f, "  \"gmres_max_cycle_exhaustions\": %d,\n", gmres_max_cycle_exhaustions);
    fprintf(f, "  \"gmres_max_final_relres\": %.17g,\n", gmres_max_final_relres);
    fprintf(f, "  \"fmm_digits\": %d,\n", fmm_digits);
    fprintf(f, "  \"max_leaf\": %d,\n", max_leaf);
    fprintf(f, "  \"gmres_restart\": %d,\n", gmres_restart);
    fprintf(f, "  \"gmres_tol\": %.17g,\n", gmres_tol);
    fprintf(f, "  \"gmres_max_cycles\": %d,\n", gmres_max_cycles);
    fprintf(f, "  \"requested_fmm_digits\": %d,\n", requested_fmm_digits);
    fprintf(f, "  \"requested_gmres_tol\": %.17g,\n", requested_gmres_tol);
    fprintf(f, "  \"fmm_digits_cli_set\": %s,\n", fmm_digits_cli_set ? "true" : "false");
    fprintf(f, "  \"gmres_tol_cli_set\": %s,\n", gmres_tol_cli_set ? "true" : "false");
    fprintf(f, "  \"accuracy_policy_adjusted\": %s,\n", accuracy_policy_adjusted ? "true" : "false");
    fprintf(f, "  \"solver_backend\": ");
    write_json_string(f, solver_backend ? solver_backend : "unknown");
    fprintf(f, ",\n");
    fprintf(f, "  \"solver_profile\": ");
    write_json_string(f, solver_profile ? solver_profile : "unknown");
    fprintf(f, ",\n");
    fprintf(f, "  \"krylov_solver\": ");
    write_json_string(f, krylov_solver ? krylov_solver : "gmres");
    fprintf(f, ",\n");
    fprintf(f, "  \"requested_system\": ");
    write_json_string(f, requested_system_kind ? requested_system_kind : "unknown");
    fprintf(f, ",\n");
    fprintf(f, "  \"system\": ");
    write_json_string(f, system_kind ? system_kind : "unknown");
    fprintf(f, ",\n");
    fprintf(f, "  \"device_gmres\": %s,\n", device_gmres ? "true" : "false");
    fprintf(f, "  \"preconditioner_enabled\": %s,\n", preconditioner_enabled ? "true" : "false");
    fprintf(f, "  \"method\": {\n");
    write_json_key_string(f, "solver_backend", solver_backend ? solver_backend : "unknown", true);
    write_json_key_string(f, "solver_profile", solver_profile ? solver_profile : "unknown", true);
    write_json_key_string(f, "krylov_solver", krylov_solver ? krylov_solver : "gmres", true);
    write_json_key_string(f, "requested_system", requested_system_kind ? requested_system_kind : "unknown", true);
    write_json_key_string(f, "system", system_kind ? system_kind : "unknown", true);
    fprintf(f, "    \"system_canonicalized\": %s,\n",
            system_canonicalized ? "true" : "false");
    fprintf(f, "    \"gmres_true_residual_checked\": %s,\n",
            gmres_matvecs > 0 ? "true" : "false");
    fprintf(f, "    \"quad_order\": %d,\n", quad_order);
    fprintf(f, "    \"unknown_m_scale\": %.17g,\n", unknown_m_scale);
    fprintf(f, "    \"row_h_scale\": %.17g,\n", row_h_scale.real());
    fprintf(f, "    \"row_h_scale_imag\": %.17g,\n", row_h_scale.imag());
    fprintf(f, "    \"row_h_scale_complex\": [%.17g, %.17g],\n",
            row_h_scale.real(), row_h_scale.imag());
    fprintf(f, "    \"interior_operator_sign\": %.17g,\n", int_op_sign);
    fprintf(f, "    \"k_identity_jump\": %.17g,\n", k_identity);
    fprintf(f, "    \"farfield_phase_sign\": %.17g,\n",
            std::getenv("BEM_FF_PHASE_SIGN") ? std::atof(std::getenv("BEM_FF_PHASE_SIGN")) : -1.0);
    fprintf(f, "    \"farfield_j_scale\": %.17g,\n",
            std::getenv("BEM_FF_J_SCALE") ? std::atof(std::getenv("BEM_FF_J_SCALE")) : 1.0);
    fprintf(f, "    \"farfield_m_sign\": %.17g,\n",
            std::getenv("BEM_FF_M_SIGN") ? std::atof(std::getenv("BEM_FF_M_SIGN")) : -1.0);
    fprintf(f, "    \"preconditioner_enabled\": %s,\n",
            preconditioner_enabled ? "true" : "false");
    fprintf(f, "    \"schwarz_preconditioner\": %s,\n",
            schwarz_preconditioner ? "true" : "false");
    fprintf(f, "    \"device_gmres\": %s,\n",
            device_gmres ? "true" : "false");
    write_json_key_string(f, "preconditioner_reason", preconditioner_reason ? preconditioner_reason : "unknown", true);
    write_json_key_string(f, "farfield_mode", farfield_mode ? farfield_mode : "unknown", false);
    fprintf(f, "  },\n");
    fprintf(f, "  \"mesh\": {\n");
    fprintf(f, "    \"vertices\": %d,\n", mesh_vertices);
    fprintf(f, "    \"triangles\": %d,\n", mesh_triangles);
    fprintf(f, "    \"skinny_triangles\": %d,\n", mesh_skinny_triangles);
    fprintf(f, "    \"min_angle_deg\": %.17g,\n", mesh_min_angle_deg);
    fprintf(f, "    \"max_aspect_ratio\": %.17g,\n", mesh_max_aspect_ratio);
    fprintf(f, "    \"feature_edges_30deg\": %d,\n", mesh_feature_edges_30deg);
    fprintf(f, "    \"feature_edge_fraction\": %.17g,\n", mesh_feature_edge_fraction);
    fprintf(f, "    \"max_dihedral_deg\": %.17g,\n", mesh_max_dihedral_deg);
    fprintf(f, "    \"mean_feature_dihedral_deg\": %.17g,\n", mesh_mean_feature_dihedral_deg);
    fprintf(f, "    \"max_adjacent_area_ratio\": %.17g,\n", mesh_max_adjacent_area_ratio);
    fprintf(f, "    \"near_touch_checked\": %s,\n",
            mesh_near_touch_checked ? "true" : "false");
    fprintf(f, "    \"near_touch_ratio\": %.17g,\n", mesh_near_touch_ratio);
    fprintf(f, "    \"near_touch_pairs\": %d,\n", mesh_near_touch_pairs);
    fprintf(f, "    \"self_panel_count\": %d,\n", mesh_self_panel_count);
    fprintf(f, "    \"edge_adjacent_pair_count\": %d,\n", mesh_edge_adjacent_pair_count);
    fprintf(f, "    \"vertex_adjacent_pair_count\": %d,\n", mesh_vertex_adjacent_pair_count);
    fprintf(f, "    \"near_disjoint_pair_count\": %d,\n", mesh_near_disjoint_pair_count);
    fprintf(f, "    \"taylor_duffy_candidate_count\": %d,\n", mesh_taylor_duffy_candidate_count);
    fprintf(f, "    \"recommended_min_quad_order\": %d,\n", mesh_recommended_min_quad_order);
    write_json_key_string(f, "recommended_mesh_strategy", mesh_recommended_strategy, true);
    write_json_key_string(f, "recommended_mesh_action", mesh_recommended_action, true);
    fprintf(f, "    \"voxel_surface_like\": %s,\n",
            mesh_voxel_surface_like ? "true" : "false");
    fprintf(f, "    \"requires_remesh\": %s,\n", mesh_requires_remesh ? "true" : "false");
    fprintf(f, "    \"edge_refine_requested\": %d,\n", edge_refine_requested);
    fprintf(f, "    \"edge_refine_applied\": %d,\n", edge_refine_applied);
    fprintf(f, "    \"edge_refine_uniform_fallback\": %s,\n",
            edge_refine_uniform_fallback ? "true" : "false");
    fprintf(f, "    \"quality_gate_pass\": %s\n",
            mesh_quality_pass ? "true" : "false");
    fprintf(f, "  },\n");
    if (orient_count > 0)
        fprintf(f, "  \"gmres_matvecs_per_orientation\": %.8g,\n",
                (double)gmres_matvecs / (double)orient_count);
    else
        fprintf(f, "  \"gmres_matvecs_per_orientation\": 0,\n");
    fprintf(f, "  \"ntheta\": %d,\n", ntheta);

    // Timing
    fprintf(f, "  \"timing\": {\n");
    fprintf(f, "    \"assembly_s\": %.6g,\n", time_assembly);
    fprintf(f, "    \"solve_s\": %.6g,\n", time_solve);
    fprintf(f, "    \"farfield_s\": %.6g,\n", time_farfield);
    fprintf(f, "    \"total_s\": %.6g\n", time_total);
    fprintf(f, "  },\n");

    // Theta array (degrees)
    fprintf(f, "  \"theta\": [");
    for (int t = 0; t < ntheta; t++) {
        fprintf(f, "%.4f", theta[t] * 180.0 / M_PI);
        if (t < ntheta - 1) fprintf(f, ", ");
    }
    fprintf(f, "],\n");

    // Mueller matrix: M[i][j][theta]
    fprintf(f, "  \"mueller\": [\n");
    for (int i = 0; i < 4; i++) {
        fprintf(f, "    [\n");
        for (int j = 0; j < 4; j++) {
            fprintf(f, "      [");
            for (int t = 0; t < ntheta; t++) {
                fprintf(f, "%.8e", M[(i*4+j)*ntheta + t]);
                if (t < ntheta - 1) fprintf(f, ", ");
            }
            fprintf(f, "]");
            if (j < 3) fprintf(f, ",");
            fprintf(f, "\n");
        }
        fprintf(f, "    ]");
        if (i < 3) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");

    fclose(f);
    printf("Results written to %s\n", filename);
}
