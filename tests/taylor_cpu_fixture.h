#pragma once
#include "muller_fmm.h"
#include "prism_face_fft.h"
#include "quadrature.h"
#include <numeric>
// Exact near-only CPU geometry for small matrix-free integration controls.
// This fixture intentionally does not stand in for native far-tree validation.
void HelmholtzFMM::cleanup(){if(d_multi_re||d_local_re||d_charges_re)std::abort();initialized=false;}
struct PrismFaceFFT::Impl{};
PrismFaceFFT::~PrismFaceFFT(){if(p)std::abort();}
PrismFaceFFT::HostView PrismFaceFFT::host_view()const{throw std::runtime_error("GPU face view in pure CPU fixture");}
inline void initialize_taylor_cpu_fixture(MullerFmmOperator& op,int ref,double ka){
    using C=std::complex<double>;MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
    op.current_dofs=op.mesh.current_dofs();op.system_dofs=op.mesh.system_dofs();
    op.k_exterior=ka;op.k_interior=ka*C(1.3116);op.epsilon_interior=C(1.3116)*C(1.3116);
    for(int order:{7,13}){const auto rule=tri_quadrature(order);auto& out=order==7?op.quadrature:op.mass_quadrature;
        for(size_t t=0;t<op.mesh.elements.size();++t)for(int q=0;q<rule.npts;++q){MullerFmmQuadraturePoint p;p.element=t;
            p.sample=evaluate_muller_frame(op.mesh,t,rule.pts[q][0],rule.pts[q][1]);p.weight=.5*rule.wts[q]*p.sample.jacobian;out.push_back(p);}}
    for(auto* f:{&op.fmm_exterior,&op.fmm_interior}){f->initialized=true;f->k=f==&op.fmm_exterior?op.k_exterior:op.k_interior;
        f->Ns=f->Nt=op.quadrature.size();f->n_nodes=1;f->L=1;f->p=0;f->n_unique_transfers=0;f->p2p_enabled=true;
        f->minimum_m2l_level=1;f->maximum_m2l_level=0;f->tree.max_level=0;f->tree.nodes.resize(1);f->tree.nodes[0]={};
        f->tree.level_nodes={{0}};f->h_leaf_indices={0};f->h_src_id_offsets=f->h_tgt_id_offsets={0,f->Ns};
        f->h_src_ids_flat.resize(f->Ns);f->h_tgt_ids_flat.resize(f->Nt);
        std::iota(f->h_src_ids_flat.begin(),f->h_src_ids_flat.end(),0);std::iota(f->h_tgt_ids_flat.begin(),f->h_tgt_ids_flat.end(),0);
        f->squad.dirs={1,0,0};f->squad.weights={1};}
}
