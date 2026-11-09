// Exercise actual launch dispatch, including absorbing fallback and split lists.
#include "../src/p2p.cu"
#include <numeric>
#include <random>
#include <stdexcept>

#include "p2p_test_buffers.h"
int main()try{
    const int ns=67,nt=41;std::mt19937 gen(7811);std::uniform_real_distribution<double> random(-1,1);
    std::vector<double> src(3*ns),tgt(3*nt),charge(12*ns);
    for(auto&v:src)v=random(gen);for(auto&v:tgt)v=random(gen);for(auto&v:charge)v=random(gen);
    // Self exclusion and a small nonzero separation exercise both distance cases.
    for(int d=0;d<3;++d){tgt[d]=src[d];tgt[3+d]=src[3+d];}tgt[3]+=1e-5;
    std::vector<int>si(ns),ti(nt);std::iota(si.begin(),si.end(),0);std::iota(ti.begin(),ti.end(),0);
    Buffer<double>S(src),T(tgt),Q(charge),O(std::vector<double>(24*nt));
    std::vector<int>flat_ids;for(int leaf=0;leaf<3;++leaf)flat_ids.insert(flat_ids.end(),si.begin(),si.end());
    Buffer<int>SI(si),TI(ti),SO(std::vector<int>{0,ns/2,ns-5,ns}),
        TO(std::vector<int>{0,0,13,nt}),NO(std::vector<int>{0,2,4,6}),
        NL(std::vector<int>{1,2,0,2,0,1}),FO(std::vector<int>{0,ns,2*ns,3*ns}),FI(flat_ids);
    double worst=0;int cases=0;
    setenv("BEM_FMM_P2P_THREADS","128",1);setenv("BEM_FMM_P2P_PAIR_THREADS","128",1);
    auto run=[&](bool pair,int variant,bool flat,bool mixed,double kr,double ki){
        setenv("BEM_FMM_P2P_STRICT_REAL",variant==1?"1":"0",1);
        setenv("BEM_FMM_P2P_STRICT_FACTORED",variant==2?"1":"0",1);O.zero();
        double* q[12];for(int i=0;i<12;++i)q[i]=Q.p+i*ns;
        double* out[8];for(int i=0;i<8;++i)out[i]=O.p+i*3*nt;
        if(pair)launch_p2p_vector_actions_pair_batch3_leaf(T.p,S.p,
            q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11],
            TO.p,TI.p,SO.p,SI.p,NO.p,NL.p,flat?FO.p:nullptr,flat?FI.p:nullptr,
            3,kr,ki,out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7],
            nullptr,nullptr,nullptr,ns,mixed);
        else for(int j=0;j<2;++j)launch_p2p_vector_actions_batch3_leaf(T.p,S.p,
            q[6*j],q[6*j+1],q[6*j+2],q[6*j+3],q[6*j+4],q[6*j+5],
            TO.p,TI.p,SO.p,SI.p,NO.p,NL.p,flat?FO.p:nullptr,flat?FI.p:nullptr,
            3,kr,ki,out[4*j],out[4*j+1],out[4*j+2],out[4*j+3],mixed);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());return O.read();
    };
    for(int split:{1,8})for(bool flat:{false,true})for(bool mixed:{false,true})
    for(double kr:{0.,10.,100.})for(double ki:{0.,1e-15,10.}){
        setenv("BEM_FMM_P2P_LEAF_SPLIT",std::to_string(split).c_str(),1);
        auto single=run(false,0,flat,mixed,kr,ki);
        for(bool pair:{false,true})for(int variant:{1,2}){
            auto base=run(pair,0,flat,mixed,kr,ki),candidate=run(pair,variant,flat,mixed,kr,ki);
            const double e=difference(candidate,base),combined=difference(candidate,single);
            if(!std::isfinite(e)||e>2e-12||!std::isfinite(combined)||combined>(mixed?2e-5:2e-12))
                throw std::runtime_error("strict-real or paired action mismatch");
            worst=std::max(worst,e);++cases;
        }
    }
    std::printf("PASS %d dispatch cases; max candidate/control per-target vector relative L2 %.3e\n",cases,worst);
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
