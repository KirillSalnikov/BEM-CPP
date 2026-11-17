#include "fmm.h"
#include "gpu_select.h"
#include <cuda_runtime.h>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
using C=std::complex<double>;
thread_local std::string error;
void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Plan {
    HelmholtzFMM f;
    std::unique_ptr<HelmholtzFMM> middle,coarse;
    std::vector<std::unique_ptr<HelmholtzFMM>> upper;
    std::vector<double> real,imag;
    size_t allocation=0;
    bool scalar=false;
    ~Plan() {
        f.cleanup();
        for(auto& band:upper)band->cleanup();
        if(middle)middle->cleanup();
        if(coarse)coarse->cleanup();
    }
};
}
extern "C" {
const char* bem_gpu_error(){return error.c_str();}
void* bem_gpu_create(const double* source,int ns,const double* target,int nt,
                    double kr,double ki,int digits,int leaf,int depth) {
    try {
        if(ns<1||nt<1||digits<2||digits>10||leaf<8)throw std::invalid_argument("GPU FMM dimensions/order");
        size_t before,total,after;checked(cudaMemGetInfo(&before,&total));
        std::unique_ptr<Plan> p(new Plan);
        if(depth) {
            if(depth<3||depth>7||depth>Octree::configured_depth_limit())throw std::invalid_argument("multilevel depth must be 3..7 and within BEM_FMM_MAX_DEPTH");
            p->scalar=true;
            double lo[3]={source[0],source[1],source[2]},hi[3]={source[0],source[1],source[2]};
            for(int i=0;i<ns+nt;++i)for(int a=0;a<3;++a) {
                const double v=i<ns?source[3*i+a]:target[3*(i-ns)+a];
                lo[a]=std::min(lo[a],v);hi[a]=std::max(hi[a],v);
            }
            const double side=1.001*std::max({hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]});
            const int requested_depth=depth;
            // Plane-wave Helmholtz translations lose digits when k*h is too
            // small. A shallower near partition avoids low-frequency breakdown;
            // depths <=2 with radius3 are entirely direct interactions.
            // The tested ka80/ref7 geometry has k*h=1.477. Preserve its
            // accurate depth7 hierarchy; reject smaller cells below 1.4.
            // This guard is empirical, not a replacement for CPU audits.
            while(depth>1 && std::abs(C(kr,ki))*side/(1<<depth)<1.4)--depth;
            if(depth!=requested_depth)std::printf("  [Scalar FMM] low-frequency depth %d -> %d\n",requested_depth,depth);
            auto occupancy=[&](int d){const long long boxes=1LL<<(3*d);return int((ns+nt+boxes-1)/boxes);};
            p->f.configure_interaction_band(depth<=2?1:depth,depth,true);p->f.configure_order_reference_depth(depth);
            p->f.init(target,nt,source,ns,C(kr,ki),digits,occupancy(depth),3,true,false,true);
            // This bridge only requests scalar potential/gradient. Batch4
            // compatibility needs a zero fourth local field, but never its
            // multipoles, Hessians, or the generic complex upload workspace.
            if(bem_env_flag_enabled("BEM_SCALAR_GPU_COMPACT",true)) {
                auto release=[](double*& v){checked(cudaFree(v));v=nullptr;};
                release(p->f.d_multi4_re);release(p->f.d_multi4_im);
                release(p->f.d_hess_re);release(p->f.d_hess_im);
                release(p->f.d_hess2_re);release(p->f.d_hess2_im);
                release(p->f.d_hess3_re);release(p->f.d_hess3_im);
                release(p->f.d_hess4_re);release(p->f.d_hess4_im);
                checked(cudaFree(p->f.d_complex_tmp1));p->f.d_complex_tmp1=nullptr;
                checked(cudaFree(p->f.d_complex_tmp2));p->f.d_complex_tmp2=nullptr;
            }
            if(depth>5 && bem_env_flag_enabled("BEM_SCALAR_GPU_FULL_BANDS",true)) {
                for(int level=depth-1;level>=3;--level) {
                    std::unique_ptr<HelmholtzFMM> band(new HelmholtzFMM);
                    band->configure_interaction_band(level==3?1:level,level,false);
                    band->configure_order_reference_depth(level==3?2:level);
                    band->init(target,nt,source,ns,C(kr,ki),digits,occupancy(level),3,false,false,true);
                    if(band->tree.max_level!=level)throw std::runtime_error("unexpected upper band depth");
                    p->upper.push_back(std::move(band));
                }
            } else if(depth>=3) {
                p->middle.reset(new HelmholtzFMM);p->coarse.reset(new HelmholtzFMM);
                p->middle->configure_interaction_band(depth-1,depth-1,false);p->middle->configure_order_reference_depth(depth);
                p->coarse->configure_interaction_band(1,depth-2,false);p->coarse->configure_order_reference_depth(std::max(1,depth-3));
                p->middle->init(target,nt,source,ns,C(kr,ki),digits,occupancy(depth-1),3,false,false,true);
                p->coarse->init(target,nt,source,ns,C(kr,ki),digits,occupancy(depth-2),3,false,false,true);
                if(p->middle->tree.max_level!=depth-1||p->coarse->tree.max_level!=depth-2)
                    throw std::runtime_error("unexpected upper hierarchy depth");
            }
            if(p->f.tree.max_level!=depth)
                throw std::runtime_error("unexpected GPU FMM hierarchy depth");
        } else p->f.init(target,nt,source,ns,C(kr,ki),digits,leaf,1,true,false,false);
        checked(cudaDeviceSynchronize());checked(cudaMemGetInfo(&after,&total));p->allocation=before-after;
        p->real.resize(std::max(ns,3*nt));p->imag.resize(p->real.size());
        return p.release();
    }catch(const std::exception& e){error=e.what();return nullptr;}
}
void bem_gpu_destroy(void* p){delete static_cast<Plan*>(p);}
int bem_gpu_info(void* ptr,long long* out) {
    if(!ptr||!out)return 1;Plan& p=*static_cast<Plan*>(ptr);
    out[0]=p.f.Ns;out[1]=p.f.Nt;out[2]=p.f.n_nodes;out[3]=p.f.L;out[4]=p.f.p;out[5]=p.allocation;out[6]=p.f.tree.max_level;
    return 0;
}
int bem_gpu_apply(void* ptr,const C* charge,int nd,C* pot,C* grad) {
    try {
        if(!ptr||!charge||!pot||!grad||nd<1||nd>4)throw std::invalid_argument("GPU FMM packet");
        Plan& p=*static_cast<Plan*>(ptr);auto& f=p.f;
        size_t before,after,total;checked(cudaMemGetInfo(&before,&total));
        if(p.scalar && nd>3)throw std::invalid_argument("multilevel bridge takes at most three channels");
        double* qr[]={f.d_charges_re,f.d_charges2_re,f.d_charges3_re,f.d_charges4_re};
        double* qi[]={f.d_charges_im,f.d_charges2_im,f.d_charges3_im,f.d_charges4_im};
        double* pr[]={f.d_result_re,f.d_result2_re,f.d_result3_re,f.d_result4_re};
        double* pi[]={f.d_result_im,f.d_result2_im,f.d_result3_im,f.d_result4_im};
        double* gr[]={f.d_grad_re,f.d_grad2_re,f.d_grad3_re,f.d_grad4_re};
        double* gi[]={f.d_grad_im,f.d_grad2_im,f.d_grad3_im,f.d_grad4_im};
        for(int d=0;d<4;++d){
            if(d>=nd){checked(cudaMemset(qr[d],0,size_t(f.Ns)*sizeof(double)));checked(cudaMemset(qi[d],0,size_t(f.Ns)*sizeof(double)));continue;}
            for(int s=0;s<f.Ns;++s){p.real[s]=charge[size_t(s)*nd+d].real();p.imag[s]=charge[size_t(s)*nd+d].imag();}
            checked(cudaMemcpy(qr[d],p.real.data(),size_t(f.Ns)*sizeof(double),cudaMemcpyHostToDevice));
            checked(cudaMemcpy(qi[d],p.imag.data(),size_t(f.Ns)*sizeof(double),cudaMemcpyHostToDevice));
        }
        if(!p.upper.empty()) {
            std::vector<HelmholtzFMM*> upper;
            for(auto& band:p.upper)upper.push_back(band.get());
            f.evaluate_multilevel_pot_grad_batch3_uploaded(upper);
        } else if(p.coarse)f.evaluate_multilevel_pot_grad_batch3_uploaded(p.middle.get(),*p.coarse);
        else if(p.scalar)f.evaluate_multilevel_pot_grad_batch3_uploaded(std::vector<HelmholtzFMM*>{});
        else f.evaluate_pot_grad_batch4_uploaded();
        checked(cudaDeviceSynchronize());
        checked(cudaMemGetInfo(&after,&total));
        if(before>=after)p.allocation+=before-after;
        else p.allocation-=std::min(p.allocation,after-before);
        for(int d=0;d<nd;++d){
            checked(cudaMemcpy(p.real.data(),pr[d],size_t(f.Nt)*sizeof(double),cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(p.imag.data(),pi[d],size_t(f.Nt)*sizeof(double),cudaMemcpyDeviceToHost));
            for(int t=0;t<f.Nt;++t)pot[size_t(t)*nd+d]=C(p.real[t],p.imag[t]);
            checked(cudaMemcpy(p.real.data(),gr[d],size_t(f.Nt)*3*sizeof(double),cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(p.imag.data(),gi[d],size_t(f.Nt)*3*sizeof(double),cudaMemcpyDeviceToHost));
            for(int t=0;t<f.Nt;++t)for(int a=0;a<3;++a)grad[(size_t(t)*3+a)*nd+d]=C(p.real[3*t+a],p.imag[3*t+a]);
        }
        return 0;
    }catch(const std::exception& e){error=e.what();return 1;}
}
}
