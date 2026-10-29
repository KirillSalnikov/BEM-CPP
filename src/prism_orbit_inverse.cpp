#include "prism_orbit_inverse.h"
#include "muller_fmm.h"
void PrismOrbitInverse::build(const MullerFmmOperator& op,const bem_tmatrix::PrismGroupMaps& group,int nodes) {
    std::vector<std::vector<int>> support(op.mesh.topology_edge_count);
    for(int e=0;e<int(op.mesh.elements.size());++e)for(int edge:op.mesh.elements[e].topology_edges)support[edge].push_back(e);
    build(op.mesh,group,nodes,[&](const std::vector<int>& selected) {
        std::vector<int> elements;for(int e:selected)elements.insert(elements.end(),support[e].begin(),support[e].end());
        std::sort(elements.begin(),elements.end());elements.erase(std::unique(elements.begin(),elements.end()),elements.end());
        return assemble_muller_nodal_block(op,selected,&elements);
    });
}
