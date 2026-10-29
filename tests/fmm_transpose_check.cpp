#include "fmm.h"
#include <random>
#include <cstdio>
#include <stdexcept>
int main()try {
    using C=std::complex<double>;const int ns=137,nt=101;
    std::mt19937 rng(3891);std::uniform_real_distribution<double> uniform(-1,1);
    std::vector<double> src(3*ns),tgt(3*nt);for(auto& x:src)x=uniform(rng);for(auto& x:tgt)x=uniform(rng);
    for(int a=0;a<3;++a)tgt[a]=src[9+a]; // singular self exclusion with unequal IDs
    std::vector<C> q[3],result[3],c(3*nt),h(3*nt),cs(3*nt),hs(3*nt);
    for(int a=0;a<3;++a){q[a].resize(ns);result[a].resize(ns);for(auto& z:q[a])z=C(uniform(rng),uniform(rng));}
    for(auto& z:cs)z=C(uniform(rng),uniform(rng));
    for(auto& z:hs)z=C(uniform(rng),uniform(rng));
    for(double absorption:{0.,.2}) {
        HelmholtzFMM f;f.init(tgt.data(),nt,src.data(),ns,C(5,absorption),4,8,1,false,false,true);
        for(bool mixed:{false,true}) {
            f.near_field_fp32=mixed;
            f.evaluate_vector_actions_batch3(q[0].data(),q[1].data(),q[2].data(),c.data(),h.data());
            f.evaluate_vector_actions_transpose(cs.data(),hs.data(),result[0].data(),result[1].data(),result[2].data());
            C lhs=0,rhs=0;double yn=0,fn=0;
            for(int j=0;j<3*nt;++j){lhs+=cs[j]*c[j]+hs[j]*h[j];yn+=std::norm(cs[j])+std::norm(hs[j]);fn+=std::norm(c[j])+std::norm(h[j]);}
            for(int a=0;a<3;++a)for(int j=0;j<ns;++j)rhs+=result[a][j]*q[a][j];
            const double error=std::abs(lhs-rhs)/std::sqrt(yn*fn);
            std::printf("unequal points: absorption=%.2f mixed=%d transpose_error=%.3e\n",absorption,mixed,error);
            if(error>(mixed?2e-5:1e-12))throw std::runtime_error("FMM rectangular transpose failed");
        }
        f.cleanup();
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
