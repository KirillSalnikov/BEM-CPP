#include "muller_fmm_cpu.h"
#include <cstdio>

// This fixture builds only host metadata and never allocates a CUDA buffer.
// Keep the arithmetic/map test runnable without a CUDA context or GPU lock.
void HelmholtzFMM::cleanup(){
    if(d_multi_re || d_local_re || d_charges_re)std::abort();
    initialized=false;
}

int main()try {
    using C=bem_cpu::C;
    const std::vector<Vec3> points{{.1,.2,0},{.3,-.1,.7},{-.2,.3,.4},{.8,.1,-.2},{-.4,-.1,.2}};
    std::vector<bem_cpu::Six> charges(points.size());
    for(size_t p=0;p<points.size();++p)for(int j=0;j<6;++j)charges[p][j]=C(.2*p-.03*j,.1*j+.07*p);
    double worst=0;
    for(int mode:{0,1,2})for(int avx:{0,1}){
        const bool far=mode==1,identity=mode==2;
        std::vector<int> sources=identity?std::vector<int>{0,1,2,3,4}:std::vector<int>{3,0,4};
        std::vector<int> targets=identity?sources:std::vector<int>{2,1};
        setenv("BEM_CPU_AVX512",avx?"1":"0",1);
        HelmholtzFMM f;f.initialized=true;f.Ns=sources.size();f.Nt=targets.size();f.L=1;f.k=C(3,.2);
        f.p2p_enabled=!far;f.n_nodes=far?2:1;f.minimum_m2l_level=1;f.tree.max_level=1;
        f.tree.nodes.resize(f.n_nodes);f.h_leaf_indices={0};
        if(far)f.h_leaf_indices.push_back(1);
        for(int i=0;i<f.n_nodes;++i){f.tree.nodes[i]={};f.tree.nodes[i].index=i;f.tree.nodes[i].level=1;}
        f.h_src_id_offsets=far?std::vector<int>{0,3,3}:std::vector<int>{0,3};
        f.h_tgt_id_offsets=far?std::vector<int>{0,0,2}:std::vector<int>{0,2};
        f.h_src_ids_flat={2,0,1};f.h_tgt_ids_flat={1,0};
        if(identity){f.h_src_id_offsets={0,5};f.h_tgt_id_offsets={0,5};f.h_src_ids_flat={4,0,3,1,2};f.h_tgt_ids_flat={3,1,4,0,2};}
        f.squad.dirs={1,0,0};f.squad.weights={1};
        if(far){f.m2l_batches.resize(2);for(auto& b:f.m2l_batches)b.n_pairs=0;
            auto& b=f.m2l_batches[1];b.n_pairs=1;b.tgt_idx={1};b.src_idx={0};b.transfer_idx={0};f.transfer_cache={C(.7,.3)};}
        for(size_t budget:{size_t(0),size_t(1024)}){
            bem_cpu::FmmBand band(f,points,budget,nullptr,identity?nullptr:&sources,identity?nullptr:&targets);
            std::vector<bem_cpu::Twelve> actual(points.size()),expected(points.size());band.add(charges,actual);
            for(int t:targets)for(int s:sources){
                const Vec3 r=points[t]-points[s];const double radius=std::sqrt(r.norm2());
                if(radius<1e-14)continue;
                const C ik=C(0,1)*f.k,green=std::exp(ik*radius)/(4*std::acos(-1.)*radius);
                const C d1=green*(ik-1./radius),d2=green*(-f.k*f.k-2.*ik/radius+2./(radius*radius));
                C grad[3],h[3][3],trace=0;const double xyz[3]={r.x,r.y,r.z};
                for(int i=0;i<3;++i){grad[i]=d1*xyz[i]/radius;for(int j=0;j<3;++j)h[i][j]=(d2-d1/radius)*xyz[i]*xyz[j]/(radius*radius)+(i==j?d1/radius:C(0));trace+=h[i][i];}
                if(far){
                    const C weight=ik/(16*std::acos(-1.)*std::acos(-1.))*f.transfer_cache[0]*std::exp(ik*(points[t].x-points[s].x));
                    grad[0]=ik*weight;grad[1]=grad[2]=0;trace=-f.k*f.k*weight;
                    for(int i=0;i<3;++i)for(int j=0;j<3;++j)h[i][j]=(i==0&&j==0)?trace:C(0);
                }
                for(int current=0;current<2;++current){const C* q=charges[s].data()+3*current;C* out=expected[t].data()+6*current;
                    out[0]+=grad[1]*q[0]-grad[0]*q[1];out[1]+=grad[2]*q[0]-grad[0]*q[2];out[2]+=grad[2]*q[1]-grad[1]*q[2];
                    for(int i=0;i<3;++i){for(int j=0;j<3;++j)out[3+i]+=h[i][j]*q[j];out[3+i]-=trace*q[i];}
                }
            }
            double a=0,b=0;for(size_t i=0;i<points.size();++i)for(int j=0;j<12;++j){a+=std::norm(actual[i][j]-expected[i][j]);b+=std::norm(expected[i][j]);}
            const double error=std::sqrt(a/b);worst=std::max(worst,error);
            if(!std::isfinite(error)||error>2e-12)throw std::runtime_error("rectangular CPU FMM point map");
        }
    }
    std::printf("PASS rectangular maps, permuted leaf IDs, near/far, phase cache, AVX512/scalar; worst %.3e\n",worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
