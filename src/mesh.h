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

// Generate hexagonal prism with unit equivalent-sphere radius.
// aspect_ratio = H/Dx (height over inscribed diameter, same convention as ADDA).
// refinements controls mesh density (~24*4^ref triangles).
Mesh hex_prism(double aspect_ratio, int refinements);

// Load mesh from Wavefront OBJ file
Mesh load_obj(const char* filename);

// Subdivide all triangles (midpoint subdivision, no projection)
Mesh subdivide_flat(const Mesh& m);

#endif // BEM_MESH_H
