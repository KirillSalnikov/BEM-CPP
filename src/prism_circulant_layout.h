#ifndef BEM_PRISM_CIRCULANT_LAYOUT_H
#define BEM_PRISM_CIRCULANT_LAYOUT_H
#include "muller_nodal.h"
#include <vector>

struct PrismCirculantPatch {
    int kind=0,nx=0,ny=0,parity=1;
    Vec3 origin,u,v,normal;
    // channel=(current*3+edge_type)*2+moment; channel-major cell arrays.
    // Missing slots are zero extended; interface DOFs use the local inverse.
    std::vector<int> dof;
    std::vector<signed char> rhs_sign,solution_sign;
    int slot(int channel,int i,int j)const{return channel*nx*ny+i*ny+j;}
};

struct PrismCirculantLayout {
    std::vector<PrismCirculantPatch> patches;
    int system_dofs=0,covered_dofs=0,interface_dofs=0;
    double basis_mapping_error=0;
    // Supports the regular, unrefined-edge D6h prism; aspect is recovered
    // from the supplied geometry, not assumed to be one.
    void build(const MullerP2Mesh& mesh,int refinement);
};
#endif
