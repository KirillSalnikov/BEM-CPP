// Check opt-in DS dispatch against FP64, per target and vector action.
#include "../src/p2p.cu"
#include "p2p_test_buffers.h"
#include <numeric>
#include <random>
#include <stdexcept>

int main() try {
    const int ns=173, nt=137;
    std::mt19937 generator(841107);
    std::uniform_real_distribution<double> random(-1,1);
    std::vector<double> src(3*ns), tgt(3*nt), charge(12*ns);
    for (auto& v:src) v=random(generator)+10000.;
    for (auto& v:tgt) v=random(generator)+10000.;
    for (auto& v:charge) v=random(generator);
    // Self exclusion, close separation, offset coordinates, partial tiles.
    for (int d=0;d<3;++d) {tgt[d]=src[d]; tgt[3+d]=src[3+d];}
    tgt[3]+=1e-5;
    std::vector<int> si(ns),ti(nt);
    std::iota(si.begin(),si.end(),0); std::iota(ti.begin(),ti.end(),0);
    std::vector<int> flat;
    for(int leaf=0;leaf<3;++leaf) flat.insert(flat.end(),si.begin(),si.end());
    Buffer<double> S(src),T(tgt),Q(charge),O(std::vector<double>(24*nt));
    Buffer<int> SI(si),TI(ti),SO(std::vector<int>{0,ns/2,ns-5,ns}),
        TO(std::vector<int>{0,0,nt-3,nt}),NO(std::vector<int>{0,2,4,6}),
        NL(std::vector<int>{1,2,0,2,0,1}),FO(std::vector<int>{0,ns,2*ns,3*ns}),FI(flat);
    setenv("BEM_FMM_P2P_STRICT_REAL","1",1);
    setenv("BEM_FMM_P2P_STRICT_FACTORED","0",1);
    setenv("BEM_FMM_P2P_SHARED_SOURCES","1",1);
    unsetenv("BEM_FMM_P2P_DS_THREADS");
    unsetenv("BEM_FMM_P2P_DS_LEAF_SPLIT");
    auto run=[&](bool pair,bool ds,bool flat_lists,bool mixed,double kr,double ki) {
        setenv("BEM_FMM_P2P_DOUBLE_SINGLE",ds?"1":"0",1); O.zero();
        double* q[12]; for(int j=0;j<12;++j) q[j]=Q.p+j*ns;
        double* o[8]; for(int j=0;j<8;++j) o[j]=O.p+j*3*nt;
        if(pair) launch_p2p_vector_actions_pair_batch3_leaf(T.p,S.p,
            q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],
            TO.p,TI.p,SO.p,SI.p,NO.p,NL.p,flat_lists?FO.p:nullptr,flat_lists?FI.p:nullptr,
            3,kr,ki,o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7],nullptr,nullptr,nullptr,ns,mixed);
        else for(int j=0;j<2;++j) launch_p2p_vector_actions_batch3_leaf(T.p,S.p,
            q[6*j],q[6*j+1],q[6*j+2],q[6*j+3],q[6*j+4],q[6*j+5],
            TO.p,TI.p,SO.p,SI.p,NO.p,NL.p,flat_lists?FO.p:nullptr,flat_lists?FI.p:nullptr,
            3,kr,ki,o[4*j],o[4*j+1],o[4*j+2],o[4*j+3],mixed);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        auto values=O.read();
        for(auto v:values) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite P2P action");
        return values;
    };
    double worst=0, fallback_worst=0; int cases=0;
    for(int factored:{0,1}) for(int threads:{64,128,256}) for(int split:{1,4,32}) {
        setenv("BEM_FMM_P2P_DS_FACTORED",factored?"1":"0",1);
        setenv("BEM_FMM_P2P_THREADS",std::to_string(threads).c_str(),1);
        setenv("BEM_FMM_P2P_PAIR_THREADS",std::to_string(threads).c_str(),1);
        setenv("BEM_FMM_P2P_LEAF_SPLIT",std::to_string(split).c_str(),1);
        for(double kr:{0.,40.,4000.,14000.}) for(bool pair:{false,true}) {
            auto reference=run(pair,false,true,false,kr,0.);
            auto candidate=run(pair,true,true,false,kr,0.);
            double error=difference(candidate,reference); worst=std::max(worst,error);
            if(error>2e-9) throw std::runtime_error("DS action mismatch: "+std::to_string(error));
            ++cases;
        }
        // Unsupported geometry/media and ordinary mixed iterations must retain
        // the existing implementation, even when the DS environment flag is on.
        for(bool pair:{false,true}) for(int fallback=0;fallback<3;++fallback) {
            bool flat_lists=fallback!=0, mixed=fallback==2;
            double ki=fallback==1?.4:0.;
            auto a=run(pair,false,flat_lists,mixed,40.,ki);
            auto b=run(pair,true,flat_lists,mixed,40.,ki);
            double error=difference(b,a); fallback_worst=std::max(fallback_worst,error);
            if(error>2e-12) throw std::runtime_error("DS fallback changed an unsupported action");
            ++cases;
        }
    }
    // Independent DS settings must not change the ordinary FP64/mixed path;
    // invalid settings fall back and oversized blocks are capped.
    for(int threads:{64,128,256,512,1024,17}) for(int split:{1,32,128,17}) {
        setenv("BEM_FMM_P2P_THREADS","128",1);
        setenv("BEM_FMM_P2P_LEAF_SPLIT","4",1);
        setenv("BEM_FMM_P2P_DS_THREADS",std::to_string(threads).c_str(),1);
        setenv("BEM_FMM_P2P_DS_LEAF_SPLIT",std::to_string(split).c_str(),1);
        auto reference=run(false,false,true,false,40.,0.);
        auto candidate=run(false,true,true,false,40.,0.);
        double error=difference(candidate,reference);worst=std::max(worst,error);
        if(error>2e-9)throw std::runtime_error("DS independent launch setting mismatch");
        ++cases;
    }
    unsetenv("BEM_FMM_P2P_DS_THREADS");
    unsetenv("BEM_FMM_P2P_DS_LEAF_SPLIT");
    // Oversized mixed-kernel launch settings must be capped for DS. Compare
    // against a valid FP64 launch, independently of its register requirements.
    for(int threads:{512,1024}) for(bool pair:{false,true}) {
        setenv("BEM_FMM_P2P_THREADS","128",1);
        setenv("BEM_FMM_P2P_PAIR_THREADS","128",1);
        auto reference=run(pair,false,true,false,40.,0.);
        setenv("BEM_FMM_P2P_THREADS",std::to_string(threads).c_str(),1);
        setenv("BEM_FMM_P2P_PAIR_THREADS",std::to_string(threads).c_str(),1);
        auto candidate=run(pair,true,true,false,40.,0.);
        double error=difference(candidate,reference); worst=std::max(worst,error);
        if(error>2e-9) throw std::runtime_error("DS launch cap mismatch");
        ++cases;
    }
    Q.zero();
    auto zero=run(true,true,true,false,40.,0.);
    for(auto value:zero) if(value!=0.) throw std::runtime_error("Zero charges are not zero");
    std::printf("PASS %d cases; worst DS per-target error %.4e; fallback %.4e; zero exact\n",
                cases,worst,fallback_worst);
} catch(const std::exception& error) {
    std::fprintf(stderr,"%s\n",error.what()); return 1;
}
