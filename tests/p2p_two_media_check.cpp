// Runtime CUDA regression: original separate media vs fused media, including
// cancellation, signed/complex current weights, self exclusion and empty leaves.
#include "p2p.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <complex>
#include <numeric>
#include <random>
#include <cstdio>
#define CUDA_CHECK(x) do {auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));} while(0)
#include "p2p_test_buffers.h"
using C=std::complex<double>;
int main()try {
    const int ns=173,nt=137;std::mt19937 rng(841107);std::uniform_real_distribution<double> random(-1,1);
    std::vector<double> src(3*ns),tgt(3*nt),charge(12*ns);
    for(auto& v:src)v=10000+random(rng);for(auto& v:tgt)v=10000+random(rng);for(auto& v:charge)v=random(rng);
    for(int d=0;d<3;++d){tgt[d]=src[d];tgt[3+d]=src[3+d];}tgt[3]+=1e-5;
    std::vector<int> si(ns),ti(nt),flat;std::iota(si.begin(),si.end(),0);std::iota(ti.begin(),ti.end(),0);
    for(int l=0;l<3;++l)flat.insert(flat.end(),si.begin(),si.end());
    Buffer<double>S(src),T(tgt),Q(charge),O(std::vector<double>(24*nt));
    Buffer<int>SI(si),TI(ti),SO(std::vector<int>{0,ns/2,ns-5,ns}),TO(std::vector<int>{0,0,nt-3,nt}),
        NO(std::vector<int>{0,2,4,6}),NL(std::vector<int>{1,2,0,2,0,1}),FO(std::vector<int>{0,ns,2*ns,3*ns}),FI(flat);
    setenv("BEM_FMM_P2P_SHARED_SOURCES","1",1);setenv("BEM_FMM_P2P_STRICT_REAL","1",1);
    setenv("BEM_FMM_P2P_STRICT_FACTORED","0",1);setenv("BEM_FMM_P2P_DOUBLE_SINGLE","0",1);
    setenv("BEM_FMM_P2P_THREADS","128",1);setenv("BEM_FMM_P2P_LEAF_SPLIT","1",1);
    const auto old=[&](C k){O.zero();for(int j=0;j<2;++j) {
        double* q=Q.p+6*j*ns;double* o=O.p+12*j*nt;
        launch_p2p_vector_actions_batch3_leaf(T.p,S.p,q,q+ns,q+2*ns,q+3*ns,q+4*ns,q+5*ns,
            TO.p,TI.p,SO.p,SI.p,NO.p,NL.p,FO.p,FI.p,3,k.real(),k.imag(),o,o+3*nt,o+6*nt,o+9*nt,false);
        }return O.read();};
    int cases=0;double worst[3]={},combined_worst[3]={};
    for(double k:{0.,40.,4000.,14000.})for(double absorption:{0.,.1}) {
        const C ri(1.3116,absorption),ke(k),ki=k*ri;const auto e=old(ke),i=old(ki);
        for(bool magnetic:{false,true}){const C ce=1.,ci=magnetic?ri*ri:C(1.);
        std::vector<double> ref(24*nt),scale(24*nt);
        for(int j=0;j<2;++j)for(int action=0;action<2;++action)for(int v=0;v<3*nt;++v){
            int r=j*12*nt+action*6*nt+v,im=r+3*nt;
            C a(e[r],e[im]),b(i[r],i[im]);if(action==0){a*=ce;b*=ci;}
            C value=a-b;ref[r]=value.real();ref[im]=value.imag();scale[r]=scale[im]=std::abs(a)+std::abs(b);
        }
        for(int precision:{0,1,2})for(int threads:{64,128,256})for(int split:{1,4,32}) {
            O.zero();for(int j=0;j<2;++j){double* q=Q.p+6*j*ns;double* o=O.p+12*j*nt;
                launch_p2p_two_media_leaf(T.p,S.p,q,q+ns,q+2*ns,q+3*ns,q+4*ns,q+5*ns,TO.p,TI.p,FO.p,FI.p,3,
                    ke.real(),ke.imag(),ki.real(),ki.imag(),ce.real(),ce.imag(),ci.real(),ci.imag(),
                    o,o+3*nt,o+6*nt,o+9*nt,precision,threads,split);}
            const auto a=O.read();double scaled=0,relative=0;
            for(int j=0;j<2;++j)for(int action=0;action<2;++action)for(int t=0;t<nt;++t){
                double error=0,norm=0,s=0;for(int ri=0;ri<2;++ri)for(int d=0;d<3;++d){int v=j*12*nt+action*6*nt+ri*3*nt+3*t+d;
                    if(!std::isfinite(a[v]))throw std::runtime_error("nonfinite fused action");
                    error+=(a[v]-ref[v])*(a[v]-ref[v]);norm+=ref[v]*ref[v];s+=scale[v]*scale[v];}
                scaled=std::max(scaled,std::sqrt(error/std::max(s,1e-300)));
                // Near-cancellation has a separate conditioning-aware gate.
                if(t!=1 && norm>1e-20*s)relative=std::max(relative,std::sqrt(error/norm));
            }
            worst[precision]=std::max(worst[precision],scaled);combined_worst[precision]=std::max(combined_worst[precision],relative);
            if(scaled>(precision?3e-9:2e-11)||relative>(precision?2e-7:2e-9)){
                std::fprintf(stderr,"FAIL k=%g im=%g magnetic=%d precision=%d threads=%d split=%d scaled=%.6g relative=%.6g\n",k,absorption,magnetic,precision,threads,split,scaled,relative);
                return 1;}
            ++cases;
        }}
    }
    Q.zero();O.zero();launch_p2p_two_media_leaf(T.p,S.p,Q.p,Q.p+ns,Q.p+2*ns,Q.p+3*ns,Q.p+4*ns,Q.p+5*ns,
        TO.p,TI.p,FO.p,FI.p,3,20,0,26.232,0,1,0,1,0,O.p,O.p+3*nt,O.p+6*nt,O.p+9*nt,2,64,32);
    for(double v:O.read())if(v!=0)throw std::runtime_error("zero charges changed output");
    std::printf("PASS %d cases; scaled FP64/DS/comp %.5e %.5e %.5e; combined %.5e %.5e %.5e; zero exact\n",cases,worst[0],worst[1],worst[2],combined_worst[0],combined_worst[1],combined_worst[2]);
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
