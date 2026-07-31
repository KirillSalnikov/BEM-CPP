#ifndef BEM_MESH_H
#define BEM_MESH_H

#include "types.h"
#include <vector>
#include <string>

struct Mesh {
    std::vector<Vec3> verts;
    std::vector<int> tris;   // flat: [v0,v1,v2, v0,v1,v2, ...], size = 3*ntri
    int edge_refine_requested = 0;
    int edge_refine_applied = 0;
    bool edge_refine_uniform_fallback = false;
    int nv() const { return (int)verts.size(); }
    int nt() const { return (int)tris.size() / 3; }

    // Triangle vertex access
    void tri_verts(int ti, Vec3& v0, Vec3& v1, Vec3& v2) const {
        v0 = verts[tris[3*ti]]; v1 = verts[tris[3*ti+1]]; v2 = verts[tris[3*ti+2]];
    }
    double tri_area(int ti) const {
        Vec3 v0, v1, v2; tri_verts(ti, v0, v1, v2);
        return 0.5 * (v1-v0).cross(v2-v0).norm();
    }
};

struct MeshQualityReport {
    int vertices = 0;
    int triangles = 0;
    int unique_edges = 0;
    int manifold_edges = 0;
    int boundary_edges = 0;
    int nonmanifold_edges = 0;
    int feature_edges_30deg = 0;
    int degenerate_triangles = 0;
    int skinny_triangles = 0;
    double signed_volume = 0.0;
    double min_area = 0.0;
    double max_area = 0.0;
    double mean_area = 0.0;
    double min_edge = 0.0;
    double max_edge = 0.0;
    double mean_edge = 0.0;
    double min_angle_deg = 0.0;
    double p01_angle_deg = 0.0;
    double p05_angle_deg = 0.0;
    double median_angle_deg = 0.0;
    double max_edge_ratio = 0.0;
    double max_aspect_ratio = 0.0;
    double max_dihedral_deg = 0.0;
    double mean_feature_dihedral_deg = 0.0;
    double max_adjacent_area_ratio = 0.0;
    double near_touch_ratio = 1e300;
    int near_touch_pairs = 0;
    int self_panel_count = 0;
    int edge_adjacent_pair_count = 0;
    int vertex_adjacent_pair_count = 0;
    int near_disjoint_pair_count = 0;
    int taylor_duffy_candidate_count = 0;
    int recommended_min_quad_order = 4;
    double feature_edge_fraction = 0.0;
    int edge_refine_requested = 0;
    int edge_refine_applied = 0;
    bool edge_refine_uniform_fallback = false;
    bool near_touch_checked = false;
    bool closed = false;
    bool outward_winding = false;
    bool voxel_surface_like = false;
    bool requires_remesh = false;
    bool pass_default_gate = false;
    std::string verdict;
    std::string recommended_mesh_strategy;
    std::string recommended_mesh_action;
};

// Generate icosphere with given radius and refinement level
Mesh icosphere(double radius, int refinements);

// Generate a regular right prism scaled to a requested equal-volume sphere radius.
// aspect = h / Dx, where Dx follows ADDA's prism definition.
// edge_refine applies conforming local midpoint refinement near sharp prism
// edges while preserving the uniform base mesh quality.
Mesh regular_prism(int sides, double aspect, int refinements, double equiv_radius,
                   int edge_refine = 0,
                   bool mirror_symmetric_sides = false);

// Generate a cube with a conforming tensor grid on all six faces.
// Each face has (2^refinements)^2 square cells split into triangles.
Mesh structured_cube(int refinements, double equiv_radius);

// Load and prepare arbitrary closed OBJ meshes.
Mesh load_obj(const char* filename);
bool write_mesh_obj(const char* filename, const Mesh& mesh);
Mesh subdivide_flat(const Mesh& m);
double refine_feature_edges(
    Mesh& mesh, double feature_angle_degrees, int passes);
double mesh_volume(const Mesh& m);
double normalize_mesh(Mesh& m);
double mesh_dmax(const Mesh& m);
MeshQualityReport analyze_mesh_quality(const Mesh& m,
                                       double min_angle_warn_deg = 20.0,
                                       double max_aspect_warn = 12.0);
void print_mesh_quality_report(const MeshQualityReport& q);
bool write_mesh_quality_json(const char* path, const MeshQualityReport& q,
                             const char* shape, double ka, int ref_or_subdiv,
                             int quad_order);

#endif // BEM_MESH_H
