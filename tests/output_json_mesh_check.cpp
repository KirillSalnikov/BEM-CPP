#include "output.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void require_contains(const std::string& text, const char* needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << "missing JSON field: " << needle << "\n";
        std::exit(1);
    }
}

int main()
{
    const char* path = "/tmp/bem_output_json_mesh_check.json";
    const int ntheta = 3;
    double theta[ntheta] = {0.0, 0.5, 1.0};
    std::vector<double> M(16 * ntheta, 0.0);
    M[0] = 1.0;

    write_json(path, M.data(), theta, ntheta,
               5.0, 1.3116, 0.0, 2,
               "hex_prism", nullptr, 1.5, 1,
               1, 1, 1, 0,
               0, 1, 1, 1.0,
               12, 2, 0, 0, 0, 0, 0, 1e-3,
               4, 96, 200, 1e-3, 80,
               "not_applicable", "gpu_geometry_direct",
               "FMM", "hex_guarded", "muller2-balanced", "balanced", true,
               7, 1.3116, std::complex<double>(0.7624, -0.001), 1.0, 0.0,
               false, false, "small_nonsphere",
               1058, 2112, 0, 44.70465569859524, 1.39,
               72, 90.0, 90.0, 1.25,
               true, 1.11, 0,
               2112, 3168, 0, 0, 5280,
               7, "edge_aware_refinement",
               "keep conforming edge-aware refinement near sharp dihedral edges",
               false,
               1, 0, true,
               true,
               0.1, 0.2, 0.3, 0.6);

    std::ifstream in(path);
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    require_contains(text, "\"mesh\"");
    require_contains(text, "\"method\"");
    require_contains(text, "\"shape\": \"hex_prism\"");
    require_contains(text, "\"obj_file\": null");
    require_contains(text, "\"prism_aspect\": 1.5");
    require_contains(text, "\"edge_refine\": 1");
    require_contains(text, "\"solver_backend\": \"FMM\"");
    require_contains(text, "\"solver_profile\": \"hex_guarded\"");
    require_contains(text, "\"quad_order\": 7");
    require_contains(text, "\"gmres_max_cycles\": 80");
    require_contains(text, "\"gmres_restored_best_iterates\": 0");
    require_contains(text, "\"gmres_max_cycle_exhaustions\": 0");
    require_contains(text, "\"requested_system\": \"muller2-balanced\"");
    require_contains(text, "\"system\": \"balanced\"");
    require_contains(text, "\"system_canonicalized\": true");
    require_contains(text, "\"row_h_scale\": 0.76239999999999997");
    require_contains(text, "\"row_h_scale_imag\": -0.001");
    require_contains(text, "\"row_h_scale_complex\": [0.76239999999999997, -0.001]");
    require_contains(text, "\"preconditioner_enabled\": false");
    require_contains(text, "\"preconditioner_reason\": \"small_nonsphere\"");
    require_contains(text, "\"farfield_mode\": \"gpu_geometry_direct\"");
    require_contains(text, "\"vertices\": 1058");
    require_contains(text, "\"triangles\": 2112");
    require_contains(text, "\"skinny_triangles\": 0");
    require_contains(text, "\"feature_edges_30deg\": 72");
    require_contains(text, "\"max_dihedral_deg\": 90");
    require_contains(text, "\"mean_feature_dihedral_deg\": 90");
    require_contains(text, "\"max_adjacent_area_ratio\": 1.25");
    require_contains(text, "\"near_touch_checked\": true");
    require_contains(text, "\"near_touch_ratio\": 1.1100000000000001");
    require_contains(text, "\"self_panel_count\": 2112");
    require_contains(text, "\"edge_adjacent_pair_count\": 3168");
    require_contains(text, "\"vertex_adjacent_pair_count\": 0");
    require_contains(text, "\"near_disjoint_pair_count\": 0");
    require_contains(text, "\"taylor_duffy_candidate_count\": 5280");
    require_contains(text, "\"recommended_min_quad_order\": 7");
    require_contains(text, "\"recommended_mesh_strategy\": \"edge_aware_refinement\"");
    require_contains(text, "\"recommended_mesh_action\": \"keep conforming edge-aware refinement near sharp dihedral edges\"");
    require_contains(text, "\"requires_remesh\": false");
    require_contains(text, "\"edge_refine_requested\": 1");
    require_contains(text, "\"edge_refine_applied\": 0");
    require_contains(text, "\"edge_refine_uniform_fallback\": true");
    require_contains(text, "\"quality_gate_pass\": true");

    std::cout << "output json mesh metadata: ok\n";
    return 0;
}
