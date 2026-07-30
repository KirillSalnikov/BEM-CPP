#include "mesh.h"

#include <cassert>
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <set>
#include <tuple>

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

static Mesh many_voxel_edges_box()
{
    Mesh m;
    // A small closed cubical box triangulated by face grids.  It is a clean
    // manifold, but its surface is dominated by right-angle feature edges,
    // which should trigger the voxel-edge high quadrature guard.
    const int n = 4;
    std::map<std::tuple<int,int,int>, int> vertex_ids;
    auto vid = [&](double x, double y, double z) -> int {
        int ix = (int)(x * n);
        int iy = (int)(y * n);
        int iz = (int)(z * n);
        auto key = std::make_tuple(ix, iy, iz);
        std::map<std::tuple<int,int,int>, int>::const_iterator it = vertex_ids.find(key);
        if (it != vertex_ids.end())
            return it->second;
        m.verts.push_back(Vec3(x, y, z));
        int id = (int)m.verts.size() - 1;
        vertex_ids[key] = id;
        return id;
    };
    auto add_quad = [&](int a, int b, int c, int d) {
        m.tris.push_back(a); m.tris.push_back(b); m.tris.push_back(c);
        m.tris.push_back(a); m.tris.push_back(c); m.tris.push_back(d);
    };
    for (int face = 0; face < 6; face++) {
        int ids[n + 1][n + 1];
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n; j++) {
                double u = -1.0 + 2.0 * i / n;
                double v = -1.0 + 2.0 * j / n;
                if (face == 0) ids[i][j] = vid( 1.0, u, v);
                if (face == 1) ids[i][j] = vid(-1.0, v, u);
                if (face == 2) ids[i][j] = vid(u,  1.0, v);
                if (face == 3) ids[i][j] = vid(v, -1.0, u);
                if (face == 4) ids[i][j] = vid(u, v,  1.0);
                if (face == 5) ids[i][j] = vid(v, u, -1.0);
            }
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                add_quad(ids[i][j], ids[i + 1][j], ids[i + 1][j + 1], ids[i][j + 1]);
    }
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
    assert(!prism_q.voxel_surface_like);
    assert(prism_q.recommended_min_quad_order == 7);
    assert(prism_q.edge_adjacent_pair_count == prism_q.manifold_edges);
    assert(prism_q.taylor_duffy_candidate_count > prism_q.self_panel_count);
    assert(prism_q.max_dihedral_deg > 60.0);
    assert(prism_q.mean_feature_dihedral_deg > 60.0);

    Mesh voxel = many_voxel_edges_box();
    MeshQualityReport voxel_q = analyze_mesh_quality(voxel);
    assert(voxel_q.closed);
    assert(voxel_q.voxel_surface_like);
    assert(voxel_q.recommended_min_quad_order == 13);

    const int cube_refinement = 3;
    const int cube_cells = 1 << cube_refinement;
    Mesh cube = structured_cube(cube_refinement, 1.0);
    MeshQualityReport cube_q = analyze_mesh_quality(cube);
    assert(cube.nv() == 6 * cube_cells * cube_cells + 2);
    assert(cube.nt() == 12 * cube_cells * cube_cells);
    assert(cube_q.unique_edges == 18 * cube_cells * cube_cells);
    assert(cube_q.feature_edges_30deg == 12 * cube_cells);
    assert(cube_q.closed);
    assert(cube_q.outward_winding);
    assert(cube_q.boundary_edges == 0);
    assert(cube_q.nonmanifold_edges == 0);
    assert(cube_q.degenerate_triangles == 0);
    assert(cube_q.skinny_triangles == 0);
    assert(cube_q.min_angle_deg > 44.9);
    assert(cube_q.max_adjacent_area_ratio < 1.0000001);
    assert(cube_q.pass_default_gate);
    assert(std::abs(cube_q.signed_volume - 4.0 * M_PI / 3.0) < 1e-12);

    std::map<std::tuple<long long, long long, long long>, int> cube_vertices;
    const double position_scale = 1e12;
    auto position_key = [&](const Vec3& point) {
        return std::make_tuple(
            std::llround(position_scale * point.x),
            std::llround(position_scale * point.y),
            std::llround(position_scale * point.z));
    };
    for (int vertex = 0; vertex < cube.nv(); vertex++)
        cube_vertices[position_key(cube.verts[vertex])] = vertex;
    std::set<std::array<int, 3>> cube_triangles;
    for (int triangle = 0; triangle < cube.nt(); triangle++) {
        std::array<int, 3> vertices = {{
            cube.tris[3 * triangle],
            cube.tris[3 * triangle + 1],
            cube.tris[3 * triangle + 2]}};
        std::sort(vertices.begin(), vertices.end());
        cube_triangles.insert(vertices);
    }
    for (int triangle = 0; triangle < cube.nt(); triangle++) {
        std::array<int, 3> rotated;
        for (int local = 0; local < 3; local++) {
            const Vec3& point =
                cube.verts[cube.tris[3 * triangle + local]];
            const Vec3 rotated_point(-point.y, point.x, point.z);
            const auto found = cube_vertices.find(position_key(rotated_point));
            assert(found != cube_vertices.end());
            rotated[local] = found->second;
        }
        std::sort(rotated.begin(), rotated.end());
        assert(cube_triangles.find(rotated) != cube_triangles.end());
    }

    std::cout << "mesh quality check: ok\n";
    return 0;
}
