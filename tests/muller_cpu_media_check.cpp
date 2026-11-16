#include "muller_fmm_cpu.h"
#include "muller_taylor_correction.h"
#include "tmatrix_symmetry_warm_start.h"
#include "physical_face_cpu_fixture.h"
#include <cstdio>

// Physical BDM surface and Duffy corrections, CPU-only one-leaf metadata.
// This verifies the production CPU dispatch/assembly, not native GPU trees.
using C=std::complex<double>;using V=std::vector<C>;
static double error(const V&a,const V&b){double e=0,n=0;for(size_t j=0;j<a.size();++j){e+=std::norm(a[j]-b[j]);n+=std::norm(b[j]);}return std::sqrt(e/std::max(n,1e-300));}
int main()try{
#ifndef BEM_CPU_HAS_AVX512_P2P
    std::printf("SKIP CPU media pairing requires AVX512/glibc\n");return 0;
#else
    setenv("BEM_CPU_AVX512","1",1);double worst=0;int cases=0;
    for(C index:{C(1.3116),C(1.3116,.1)}){
        MullerFmmOperator op;initialize_physical_face_fixture(op,1,20);
        op.prism_fft_exterior.reset();op.prism_fft_interior.reset();
        op.k_interior=op.k_exterior*index;op.epsilon_interior=index*index;op.fmm_interior.k=op.k_interior;
        bem_taylor::SparseCorrection<0> correction(op.mesh,20,index,7,6);
        op.correction.row_offsets=correction.offsets;
        for(const auto& e:correction.entries){MullerNearCorrectionEntry v;v.column=e.column;v.k1=C(0,-20)*e.value.diagonal[0];
            v.k2_epsilon=e.value.epsilon[0];v.k2_mu=e.value.mu[0];op.correction.entries.push_back(v);}
        bem_cpu::MullerAction original(op,0),paired(op,0,true,false,false,32,false,true);
        if(paired.fused_media_near_pairs!=1)throw std::runtime_error("pair dispatch missing");
        const size_t estimate=bem_cpu::MullerAction::estimated_workspace_bytes(op,0,true,false,false,32,true);
        if(estimate<paired.workspace_bytes())throw std::runtime_error("paired workspace underestimated");
        const int n=op.system_dofs;V x(n),a(n),b(n);for(int j=0;j<n;++j)x[j]=C(std::sin(.37*j),std::cos(.71*j));
        original.apply(x.data(),a.data());paired.apply(x.data(),b.data());worst=std::max(worst,error(b,a));++cases;
        V alias=x;paired.apply(alias.data(),alias.data());worst=std::max(worst,error(alias,a));++cases;
        V zero(n),y(n);paired.apply(zero.data(),y.data());for(C c:y)if(c!=C(0))throw std::runtime_error("stale paired workspace");++cases;
        // Reused actions must agree on both reflection transforms and rotation,
        // even when finite Duffy rules have a small absolute symmetry defect.
        bem_tmatrix::D6hOrbit orbit;orbit.build(op.mesh);V transformed;
        for(int sym=0;sym<3;++sym){
            if(sym<2)orbit.transform(x,transformed,sym==1,false);
            else {std::vector<int> im,sign;bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,im,sign);transformed.resize(n);
                for(int j=0;j<n;++j)transformed[im[j]]=double(sign[j])*x[j];}
            original.apply(transformed.data(),a.data());paired.apply(transformed.data(),b.data());worst=std::max(worst,error(b,a));++cases;
        }
        // An ordinal/topology mismatch must reject pairing, not silently drop
        // one medium's near interactions. Scalar-only requests also reject.
        std::swap(op.fmm_interior.h_src_ids_flat[0],op.fmm_interior.h_src_ids_flat[1]);
        bool refused=false;try{bem_cpu::MullerAction invalid(op,0,true,false,false,32,false,true);}catch(const std::invalid_argument&){refused=true;}
        if(!refused)throw std::runtime_error("mismatched near list accepted");
        ++cases;
        std::swap(op.fmm_interior.h_src_ids_flat[0],op.fmm_interior.h_src_ids_flat[1]);
        setenv("BEM_CPU_AVX512","0",1);refused=false;try{bem_cpu::MullerAction invalid(op,0,true,false,false,32,false,true);}catch(const std::invalid_argument&){refused=true;}
        if(!refused)throw std::runtime_error("nonvector pairing accepted");
        setenv("BEM_CPU_AVX512","1",1);++cases;
    }
    if(!std::isfinite(worst)||worst>1e-11)throw std::runtime_error("physical paired CPU action regression");
    std::printf("PASS CPU media pairing: cases=%d relative_error=%.17g; zero, alias, symmetry, memory, rejection\n",cases,worst);
#endif
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"CPU media pairing: %s\n",e.what());return 1;}
