#include "muller_fmm.h"
#include "muller_dense.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// No CUDA context is created by this independent dense/block integration test.
void HelmholtzFMM::cleanup() { if (d_multi_re || d_local_re || d_charges_re) std::abort(); initialized=false; }
struct PrismFaceFFT::Impl {};
PrismFaceFFT::~PrismFaceFFT() { if (p) std::abort(); }

int main() try {
    using C=std::complex<double>;
    double worst=0.0;
    for (bool bdm : {false,true}) for (double ka : {0.001,1.0,10.0}) {
        MullerP2BuildOptions options;
        options.edge_mode=bdm?MullerEdgeMode::HDivBdm1:MullerEdgeMode::Smooth;
        options.project_edge_nodes_to_sphere=!bdm;
        Mesh mesh;
        mesh.verts={Vec3(1,0,0),Vec3(-1,0,0),Vec3(0,1,0),Vec3(0,-1,0),Vec3(0,0,1),Vec3(0,0,-1)};
        mesh.tris={4,0,2, 4,2,1, 4,1,3, 4,3,0, 5,2,0, 5,1,2, 5,3,1, 5,0,3};
        const C index(1.3116,ka==1.0?0.1:0.0);
        const auto dense=assemble_muller_nodal_dense(mesh,C(ka),index,options,7,4);
        MullerFmmOperator op;
        op.mesh=dense.mesh;op.current_dofs=dense.current_dofs;op.system_dofs=dense.system_dofs;
        op.k_exterior=ka;op.k_interior=ka*index;op.epsilon_interior=index*index;
        const std::vector<int> groups{3,0,op.current_dofs/2-1,5};
        const auto block=assemble_muller_nodal_block(op,groups);
        std::vector<int> dofs;
        for(int g:groups)for(int d:{2*g,2*g+1,op.current_dofs+2*g,op.current_dofs+2*g+1})dofs.push_back(d);
        double error2=0.0,norm2=0.0;
        for(size_t i=0;i<dofs.size();++i)for(size_t j=0;j<dofs.size();++j){
            const C expected=dense.matrix[size_t(dofs[i])*op.system_dofs+dofs[j]];
            error2+=std::norm(block[i*dofs.size()+j]-expected);norm2+=std::norm(expected);
        }
        const double error=std::sqrt(error2/norm2);
        if(!std::isfinite(error)||error>1e-12)throw std::runtime_error("local block differs from independent dense Galerkin assembly");
        worst=std::max(worst,error);
        std::printf("%s ka=%g local block relative L2 %.3g\n",bdm?"BDM1":"curved P2",ka,error);
    }
    std::printf("six independent dense/local block controls passed; worst L2 %.3g\n",worst);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
