#pragma once
#include "maxwell_exterior_projector.h"
#include "muller_fmm.h"

// Reuse the validated exterior FMM bands, keeping their scalar point arrays.
// This path needs BEM_FMM_MULTILEVEL_KEEP_POINT_WORKSPACE=1 at operator init.
class MaxwellExteriorFmm {
    using C=std::complex<double>;
    MullerFmmOperator& op_;
    std::vector<C> p_,q_,g_,h_;
public:
    MaxwellExteriorFmm(MullerFmmOperator& op,const MaxwellExteriorProjector& projector):op_(op){
        const auto& points=projector.points();
        if(points.size()!=op.quadrature.size())throw std::invalid_argument("projector/FMM quadrature mismatch");
        for(size_t i=0;i<points.size();++i)if((points[i].position-op.quadrature[i].sample.position).norm()>1e-12)
            throw std::invalid_argument("projector/FMM point ordering mismatch");
        for(auto* band:{&op.fmm_exterior,op.banded_fmm?&op.fmm_exterior_coarse:nullptr,op.banded_fmm_middle?&op.fmm_exterior_middle:nullptr})
            if(band&&(!band->d_complex_tmp2||!band->d_grad2_re||!band->d_tgt_pts))
                throw std::invalid_argument("exterior projection needs BEM_FMM_MULTILEVEL_KEEP_POINT_WORKSPACE=1 at init");
        p_.resize(points.size());q_.resize(points.size());g_.resize(3*points.size());h_.resize(g_.size());
    }
    void operator()(const C*a,const C*b,C*p,C*g,C*q,C*h){
        op_.fmm_exterior.evaluate_pot_grad_batch2(a,b,p,g,q,h);
        for(auto* band:{op_.banded_fmm?&op_.fmm_exterior_coarse:nullptr,op_.banded_fmm_middle?&op_.fmm_exterior_middle:nullptr})if(band){
            band->evaluate_pot_grad_batch2(a,b,p_.data(),g_.data(),q_.data(),h_.data());
            for(size_t i=0;i<p_.size();++i){p[i]+=p_[i];q[i]+=q_[i];}
            for(size_t i=0;i<g_.size();++i){g[i]+=g_[i];h[i]+=h_[i];}}
    }
};
