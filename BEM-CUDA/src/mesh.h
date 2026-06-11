#ifndef BEM_MESH_H
#define BEM_MESH_H

#include "types.h"
#include <vector>

struct Mesh {
    std::vector<Vec3> verts;
    std::vector<int> tris;   // flat: [v0,v1,v2, v0,v1,v2, ...], size = 3*ntri
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

// Generate icosphere with given radius and refinement level
Mesh icosphere(double radius, int refinements);

// Generate a regular right prism scaled to a requested equal-volume sphere radius.
// aspect = h / Dx, where Dx follows ADDA's prism definition.
// edge_refine applies conforming local midpoint refinement near sharp prism
// edges while preserving the uniform base mesh quality.
Mesh regular_prism(int sides, double aspect, int refinements, double equiv_radius,
                   int edge_refine = 0);

// Load and prepare arbitrary closed OBJ meshes.
Mesh load_obj(const char* filename);
Mesh subdivide_flat(const Mesh& m);
double mesh_volume(const Mesh& m);
double normalize_mesh(Mesh& m);
double mesh_dmax(const Mesh& m);

#endif // BEM_MESH_H
