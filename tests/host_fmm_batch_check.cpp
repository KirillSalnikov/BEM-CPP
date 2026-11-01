#include "host_fmm_batch.h"
#include <cstdio>
#include <numeric>

// CPU-only metadata fixture, with no CUDA context or buffers.
void HelmholtzFMM::cleanup(){if(d_multi_re||d_local_re||d_charges_re)std::abort();initialized=false;}
using C=bem_cpu::C;using Six=bem_cpu::Six;using Twelve=bem_cpu::Twelve;
static double error(const std::vector<Twelve>& a,const std::vector<Twelve>& b){
    double e=0,n=0;for(size_t i=0;i<a.size();++i)for(int c=0;c<12;++c){e+=std::norm(a[i][c]-b[i][c]);n+=std::norm(b[i][c]);}
    const double r=std::sqrt(e/std::max(n,1e-300));if(!std::isfinite(r))throw std::runtime_error("nonfinite batch result");return r;
}
int main()try{
    const std::vector<Vec3> points{{-2.1,-1.,.1},{-1.9,-.9,-.1},{-2.1,1.1,.2},{-1.8,1.,.3},
                                   {2.1,-1.,-.1},{1.9,-.9,-.3},{2.1,1.1,.2},{1.8,1.,.1}};
    double worst=0;int cases=0;
    for(bool far:{false,true})for(int avx:{0,1})for(double absorption:{0.,.2}){
        setenv("BEM_CPU_AVX512",avx?"1":"0",1);
        HelmholtzFMM f;f.initialized=true;f.k=C(3,absorption);f.squad.init(3);f.L=f.squad.L;
        f.p2p_enabled=!far;f.n_nodes=far?7:1;f.minimum_m2l_level=1;f.tree.max_level=far?2:1;
        f.tree.nodes.resize(f.n_nodes);f.tree.level_nodes.resize(3);
        for(int i=0;i<f.n_nodes;++i){auto& node=f.tree.nodes[i];node={};node.index=i;node.level=far?(i==0?0:(i<3?1:2)):1;f.tree.level_nodes[node.level].push_back(i);}
        std::vector<int> source{3,0,2,1},target{6,4,7,5};
        if(!far){source.resize(points.size());std::iota(source.begin(),source.end(),0);target=source;}
        f.Ns=source.size();f.Nt=target.size();
        if(far){
            const double centers[7][3]={{0,0,0},{-2,0,0},{2,0,0},{-2,-1,0},{-2,1,.2},{2,-1,-.2},{2,1,.1}};
            for(int i=0;i<7;++i)for(int d=0;d<3;++d)f.tree.nodes[i].center[d]=centers[i][d];
            f.h_leaf_indices={3,4,5,6};f.h_src_id_offsets={0,2,4,4,4};f.h_tgt_id_offsets={0,0,0,2,4};
            f.h_src_ids_flat={1,3,2,0};f.h_tgt_ids_flat={1,3,0,2};
            f.m2m_data.resize(3);f.l2l_data.resize(3);
            for(int child=3;child<7;++child){const int parent=child<5?1:2;
                f.m2m_data[2].pairs.push_back({parent,child});f.l2l_data[2].pairs.push_back({parent,child});
                for(int l=0;l<f.L;++l){double dot=0;for(int d=0;d<3;++d)dot+=(centers[child][d]-centers[parent][d])*f.squad.dirs[3*l+d];
                    f.m2m_data[2].shifts.push_back(std::exp(C(0,-1)*f.k*dot));f.l2l_data[2].shifts.push_back(std::exp(C(0,1)*f.k*dot));}
            }
            f.m2l_batches.resize(3);for(auto& batch:f.m2l_batches)batch.n_pairs=0;
            auto& m=f.m2l_batches[1];m.n_pairs=1;m.src_idx={1};m.tgt_idx={2};m.transfer_idx={0};
            for(int l=0;l<f.L;++l)f.transfer_cache.push_back(C(.7+.02*l,.3-.01*l));
        }else{
            f.h_leaf_indices={0};f.h_src_id_offsets={0,8};f.h_tgt_id_offsets={0,8};f.h_src_ids_flat=source;f.h_tgt_ids_flat=target;
        }
        for(size_t budget:{size_t(0),size_t(1<<20)})for(int batches:{1,2,6,8})for(int tile:{1,7,32,64}){
            bem_cpu::FmmBand serial(f,points,budget,nullptr,&source,&target);
            bem_cpu::FmmBand metadata(f,points,budget,nullptr,&source,&target,true);
            bem_cpu::FmmBandBatch batch(metadata,batches,tile);
            std::vector<std::vector<Six>> charges(batches,std::vector<Six>(points.size()));
            std::vector<std::vector<Twelve>> actual(batches,std::vector<Twelve>(points.size())),expected=actual,oracle=actual;
            for(int b=0;b<batches;++b)for(size_t p=0;p<points.size();++p)for(int c=0;c<6;++c)
                charges[b][p][c]=b==batches-1&&batches>1?C(0):C(std::sin(.13*(p+1)*(b+1)+.17*c),std::cos(.19*(p+1)-.11*c+.2*b));
            for(int b=0;b<batches;++b)serial.add(charges[b],expected[b]);
            batch.add(charges,actual);
            if(far && batch.phase_evaluations!=(budget?0:points.size()*f.L))throw std::runtime_error("phase reuse count");
            // Independent finite angular kernel (including both inter-level
            // shifts) or direct Green derivatives, evaluated per ordered pair.
            for(int b=0;b<batches;++b)for(int t:target)for(int s:source){
                if(far){
                    for(int l=0;l<f.L;++l){const double x=f.squad.dirs[3*l],y=f.squad.dirs[3*l+1],z=f.squad.dirs[3*l+2],norm=x*x+y*y+z*z;
                        const Vec3 r=points[t]-Vec3(2,0,0)-points[s]+Vec3(-2,0,0);
                        const C q=f.squad.weights[l]*f.transfer_cache[l]*std::exp(C(0,1)*f.k*(r.x*x+r.y*y+r.z*z));
                        const C pref=C(0,1)*f.k/(16*M_PI*M_PI),curl=pref*C(0,1)*f.k,action=-pref*f.k*f.k;
                        for(int c=0;c<2;++c){const C qx=q*charges[b][s][3*c],qy=q*charges[b][s][3*c+1],qz=q*charges[b][s][3*c+2],dot=x*qx+y*qy+z*qz;
                            auto& v=oracle[b][t];v[6*c]+=curl*(y*qx-x*qy);v[6*c+1]+=curl*(z*qx-x*qz);v[6*c+2]+=curl*(z*qy-y*qz);
                            v[6*c+3]+=action*(x*dot-norm*qx);v[6*c+4]+=action*(y*dot-norm*qy);v[6*c+5]+=action*(z*dot-norm*qz);}
                    }
                }else{
                    const Vec3 r=points[t]-points[s];const double r2=r.norm2();if(r2<1e-28)continue;
                    const double radius=std::sqrt(r2),xyz[3]={r.x,r.y,r.z};const C ik=C(0,1)*f.k,g=std::exp(ik*radius)/(4*M_PI*radius);
                    const C d1=g*(ik-1./radius),d2=g*(-f.k*f.k-2.*ik/radius+2./r2);C grad[3],h[3][3],trace=0;
                    for(int i=0;i<3;++i){grad[i]=d1*xyz[i]/radius;for(int j=0;j<3;++j)h[i][j]=(d2-d1/radius)*xyz[i]*xyz[j]/r2+(i==j?d1/radius:C(0));trace+=h[i][i];}
                    for(int c=0;c<2;++c){const C* q=charges[b][s].data()+3*c;C* v=oracle[b][t].data()+6*c;
                        v[0]+=grad[1]*q[0]-grad[0]*q[1];v[1]+=grad[2]*q[0]-grad[0]*q[2];v[2]+=grad[2]*q[1]-grad[1]*q[2];
                        for(int i=0;i<3;++i){for(int j=0;j<3;++j)v[3+i]+=h[i][j]*q[j];v[3+i]-=trace*q[i];}}
                }
            }
            for(int b=0;b<batches;++b){worst=std::max(worst,error(actual[b],expected[b]));worst=std::max(worst,error(actual[b],oracle[b]));}
            // add is additive and must not leak prior work into a second RHS.
            for(auto& q:charges)for(auto& p:q)p=Six{};
            const auto saved=actual;batch.add(charges,actual);
            for(int b=0;b<batches;++b)if(error(actual[b],saved[b])!=0)throw std::runtime_error("batch stale/zero field");
            ++cases;
        }
    }
    if(worst>2e-12)throw std::runtime_error("CPU batch disagrees with independent kernel");
    std::printf("{\"passed\":true,\"cases\":%d,\"independent_kernel_error\":%.17g}\n",cases,worst);
}catch(const std::exception& e){std::fprintf(stderr,"CPU FMM batch: %s\n",e.what());return 1;}
