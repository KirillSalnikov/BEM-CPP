#include "mesh.h"

#include <cassert>
#include <iostream>

static Mesh tetrahedron()
{
    Mesh m;
    m.verts = {
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.0, 0.0, 1.0),
        Vec3(0.0, 0.0, 0.0),
    };
    m.tris = {
        0, 1, 2,
        0, 3, 1,
        0, 2, 3,
        1, 3, 2,
    };
    return m;
}

static Mesh near_touching_nonadjacent_panels()
{
    Mesh m;
    m.verts = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 1.0, 0.0),
        Vec3(0.02, 0.02, 0.01),
        Vec3(1.02, 0.02, 0.01),
        Vec3(0.02, 1.02, 0.01),
    };
    m.tris = {
        0, 1, 2,
        3, 5, 4,
    };
    return m;
}

int main()
{
    Mesh good = tetrahedron();
    MeshQualityReport good_q = analyze_mesh_quality(good);
    assert(good_q.near_touch_checked);
    assert(good_q.near_touch_pairs == 0);
    assert(good_q.near_touch_ratio >= 0.0);
    assert(good_q.self_panel_count == good.nt());
    assert(good_q.edge_adjacent_pair_count == 6);
    assert(good_q.vertex_adjacent_pair_count == 0);
    assert(good_q.near_disjoint_pair_count == 0);
    assert(good_q.taylor_duffy_candidate_count == good.nt() + 6);
    assert(good_q.manifold_edges > 0);
    assert(good_q.max_dihedral_deg > 0.0);

    Mesh bad = near_touching_nonadjacent_panels();
    MeshQualityReport bad_q = analyze_mesh_quality(bad);
    assert(bad_q.near_touch_checked);
    assert(bad_q.near_touch_pairs > 0);
    assert(bad_q.near_touch_ratio < 0.35);
    assert(bad_q.self_panel_count == bad.nt());
    assert(bad_q.edge_adjacent_pair_count == 0);
    assert(bad_q.vertex_adjacent_pair_count == 0);
    assert(bad_q.near_disjoint_pair_count == bad_q.near_touch_pairs);
    assert(bad_q.taylor_duffy_candidate_count == bad.nt() + bad_q.near_disjoint_pair_count);
    assert(!bad_q.pass_default_gate);

    Mesh sphere = icosphere(1.0, 2);
    MeshQualityReport sphere_q = analyze_mesh_quality(sphere);
    assert(sphere_q.feature_edges_30deg == 0);
    assert(sphere_q.max_dihedral_deg < 30.0);

    Mesh prism = regular_prism(6, 1.5, 1, 1.0, 0);
    MeshQualityReport prism_q = analyze_mesh_quality(prism);
    assert(prism_q.pass_default_gate);
    assert(prism_q.feature_edges_30deg > 0);
    assert(prism_q.edge_adjacent_pair_count == prism_q.manifold_edges);
    assert(prism_q.taylor_duffy_candidate_count > prism_q.self_panel_count);
    assert(prism_q.max_dihedral_deg > 60.0);
    assert(prism_q.mean_feature_dihedral_deg > 60.0);

    std::cout << "mesh quality check: ok\n";
    return 0;
}
