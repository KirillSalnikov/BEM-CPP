#pragma once
#include "planar_inverse_kernel.h"

namespace bem_osrc {
struct PrismKernels {
    InverseKernel side,cap;
    int side_nx=0,side_ny=0,cap_nx=0,cap_ny=0;
    double maximum_condition()const{return std::max(side.maximum_frobenius_condition,cap.maximum_frobenius_condition);}
    double maximum_defect()const{return std::max(side.maximum_inverse_defect,cap.maximum_inverse_defect);}
    size_t kernel_bytes()const{return (side.values.size()+cap.values.size())*sizeof(C);}
};
// Return inverse kernels in precisely the channel/sign convention accepted
// by PrismCirculantInverse. The finite-prism operator and edge blocks stay exact.
inline PrismKernels make_prism_kernels(const MullerP2Mesh& mesh,int ref,double ka,C index,double damping) {
    if(ref<2 || !std::isfinite(ka) || ka<=0)throw std::invalid_argument("planar prism kernel geometry/ka");
    PrismCirculantLayout layout;layout.build(mesh,ref);PrismKernels result;
    for(int kind=0;kind<2;++kind) {
        const auto& patch=layout.patches.at(kind?24:0);
        PlanarBdmSymbol stencil(mesh,patch,16);
        auto kernel=make_inverse_kernel(stencil,patch.nx,patch.ny,ka,index,damping,1);
        if(!std::isfinite(kernel.maximum_frobenius_condition) || !std::isfinite(kernel.maximum_inverse_defect) ||
                kernel.maximum_inverse_defect>1e-9)throw std::runtime_error("planar prism kernel inverse failed");
        if(kind) {result.cap=std::move(kernel);result.cap_nx=patch.nx;result.cap_ny=patch.ny;}
        else {result.side=std::move(kernel);result.side_nx=patch.nx;result.side_ny=patch.ny;}
    }
    return result;
}
}
