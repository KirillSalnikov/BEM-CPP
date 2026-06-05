#include "rwg.h"
#include <map>
#include <utility>

RWG build_rwg(const Mesh& mesh) {
    // Edge → list of (triangle_index, opposite_vertex_index)
    typedef std::pair<int,int> Edge;
    std::map<Edge, std::vector<std::pair<int,int>>> edge_tris;

    int ntri = mesh.nt();
    for (int ti = 0; ti < ntri; ti++) {
        for (int j = 0; j < 3; j++) {
            int v0 = mesh.tris[3*ti + j];
            int v1 = mesh.tris[3*ti + (j+1)%3];
            int vopp = mesh.tris[3*ti + (j+2)%3];
            Edge e(std::min(v0,v1), std::max(v0,v1));
            edge_tris[e].push_back({ti, vopp});
        }
    }

    RWG rwg;
    for (auto& kv : edge_tris) {
        if (kv.second.size() == 2) {
            int tp = kv.second[0].first;
            int vp = kv.second[0].second;
            int tm = kv.second[1].first;
            int vm = kv.second[1].second;

            rwg.tri_p.push_back(tp);
            rwg.tri_m.push_back(tm);
            rwg.free_p.push_back(mesh.verts[vp]);
            rwg.free_m.push_back(mesh.verts[vm]);

            Vec3 va = mesh.verts[kv.first.first];
            Vec3 vb = mesh.verts[kv.first.second];
            rwg.length.push_back((vb - va).norm());
            rwg.area_p.push_back(mesh.tri_area(tp));
            rwg.area_m.push_back(mesh.tri_area(tm));
        }
    }
    rwg.N = (int)rwg.tri_p.size();
    return rwg;
}
