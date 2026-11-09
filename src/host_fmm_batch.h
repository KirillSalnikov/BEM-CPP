#ifndef BEM_HOST_FMM_BATCH_H
#define BEM_HOST_FMM_BATCH_H
#include "host_fmm_band.h"
#include "host_shared_near_avx512.h"
#include <limits>

namespace bem_cpu {
// Same prepared FMM action for several independent currents. Exact angular
// nodes, shifts and translations are retained. Direction tiles bound memory;
// point phases are computed once per tile and reused across all currents.
// No interpolation between angular grids or geometric approximation is made.
class FmmBandBatch {
    FmmBand& band_;
    const int batches_,tile_;
    const bool shared_near_;
    std::vector<Six> multi_,local_;
    std::vector<C> phase_;
    size_t index(int node,int l,int batch)const{return (size_t(node)*tile_+l)*batches_+batch;}
    C phase(int p,int l,int begin)const {
        return band_.phase.empty()?phase_[size_t(p)*tile_+l]:band_.phase[size_t(p)*band_.L+begin+l];
    }
public:
    std::array<double,6> times{};
    double phase_seconds=0;
    size_t phase_evaluations=0;
    FmmBandBatch(FmmBand& band,int batches,int tile=32,bool shared_near=false)
        :band_(band),batches_(batches),tile_(tile),shared_near_(shared_near) {
        if(batches<1||batches>8||tile<1||tile>256)throw std::invalid_argument("CPU FMM batch dimensions");
        if(shared_near_ && band_.points.size()>size_t(std::numeric_limits<int>::max()/12))
            throw std::invalid_argument("CPU shared near current gather index overflow");
        if(band_.far){
            multi_.resize(size_t(band_.f.n_nodes)*tile_*batches_);local_.resize(multi_.size());
            if(band_.phase.empty())phase_.resize(band_.points.size()*tile_);
        }
    }
    bool shared_avx512_near()const{return shared_near_ && band_.near && band_.vector_near;}
    size_t workspace_bytes()const{return (multi_.size()+local_.size())*sizeof(Six)+phase_.size()*sizeof(C);}
    void add(const std::vector<std::vector<Six>>& charges,std::vector<std::vector<Twelve>>& fields) {
        auto& b=band_;const auto& f=b.f;
        if(charges.size()!=size_t(batches_)||fields.size()!=size_t(batches_))throw std::invalid_argument("CPU FMM batch count");
        for(int j=0;j<batches_;++j)if(charges[j].size()!=b.points.size()||fields[j].size()!=b.points.size())
            throw std::invalid_argument("CPU FMM batch field dimensions");
        times.fill(0);phase_seconds=0;phase_evaluations=0;
        for(int begin=0;b.far && begin<b.L;begin+=tile_){
            const int count=std::min(tile_,b.L-begin);auto start=Clock::now();
            if(b.phase.empty()){
#pragma omp parallel for schedule(static)
                for(size_t p=0;p<b.points.size();++p)for(int l=0;l<count;++l)
                    phase_[p*tile_+l]=b.outgoing_phase(int(p),begin+l);
                phase_evaluations+=b.points.size()*count;
            }
            phase_seconds+=elapsed(start);start=Clock::now();
            std::fill(multi_.begin(),multi_.end(),Six{});std::fill(local_.begin(),local_.end(),Six{});
#pragma omp parallel for schedule(dynamic,1)
            for(size_t leaf=0;leaf<f.h_leaf_indices.size();++leaf){
                const int node=f.h_leaf_indices[leaf];
                for(int j=f.h_src_id_offsets[leaf];j<f.h_src_id_offsets[leaf+1];++j){
                    const int p=b.source_ids[f.h_src_ids_flat[j]];
                    for(int l=0;l<count;++l){const C e=phase(p,l,begin);
                        for(int batch=0;batch<batches_;++batch){auto& sum=multi_[index(node,l,batch)];
                            for(int c=0;c<6;++c)sum[c]+=e*charges[batch][p][c];}
                    }
                }
            }
            times[0]+=elapsed(start);start=Clock::now();
            for(int level=b.depth-1;level>=std::max(1,b.minimum);--level){
                const auto& nodes=f.tree.level_nodes[level];
#pragma omp parallel for schedule(dynamic,1)
                for(size_t j=0;j<nodes.size();++j){const int node=nodes[j];
                    for(const auto& edge:b.up[node])for(int l=0;l<count;++l){const C shift=edge.shift[begin+l];
                        for(int batch=0;batch<batches_;++batch){auto& to=multi_[index(node,l,batch)];const auto& from=multi_[index(edge.node,l,batch)];
                            for(int c=0;c<6;++c)to[c]+=shift*from[c];}
                    }
                }
            }
            times[1]+=elapsed(start);start=Clock::now();
#pragma omp parallel for schedule(dynamic,1)
            for(int node=0;node<f.n_nodes;++node)for(const auto& edge:b.interaction[node])for(int l=0;l<count;++l){
                const C transfer=edge.values[begin+l];
                for(int batch=0;batch<batches_;++batch){auto& to=local_[index(node,l,batch)];const auto& from=multi_[index(edge.source,l,batch)];
                    for(int c=0;c<6;++c)to[c]+=transfer*from[c];}
            }
            times[2]+=elapsed(start);start=Clock::now();
            for(int level=std::max(2,b.minimum+1);level<=b.depth;++level){
                const auto& data=f.l2l_data[level];
#pragma omp parallel for schedule(dynamic,1)
                for(size_t j=0;j<data.pairs.size();++j){const auto& edge=data.pairs[j];
                    for(int l=0;l<count;++l){const C shift=data.shifts[j*b.L+begin+l];
                        for(int batch=0;batch<batches_;++batch){auto& to=local_[index(edge.child,l,batch)];const auto& from=local_[index(edge.parent,l,batch)];
                            for(int c=0;c<6;++c)to[c]+=shift*from[c];}
                    }
                }
            }
            times[3]+=elapsed(start);start=Clock::now();
            const C ik=C(0,1)*f.k,prefactor=ik/(16*M_PI*M_PI);
            const C curl_scale=prefactor*ik,action_scale=-prefactor*f.k*f.k;
#pragma omp parallel for schedule(dynamic,16)
            for(size_t t=0;t<b.target_ids.size();++t){const int p=b.target_ids[t];Twelve sum[8]{};
                for(int l=0;l<count;++l){const int direction=begin+l;
                    const double x=f.squad.dirs[3*direction],y=f.squad.dirs[3*direction+1],z=f.squad.dirs[3*direction+2],norm=x*x+y*y+z*z;
                    const C outgoing=phase(p,l,begin);
                    const C e=f.squad.weights[direction]*(f.k.imag()==0?std::conj(outgoing):C(1)/outgoing);
                    for(int batch=0;batch<batches_;++batch)for(int current=0;current<2;++current){
                        const auto& v=local_[index(b.point_leaf[p],l,batch)];const C qx=e*v[3*current],qy=e*v[3*current+1],qz=e*v[3*current+2],dot=x*qx+y*qy+z*qz;
                        auto& s=sum[batch];s[6*current]+=y*qx-x*qy;s[6*current+1]+=z*qx-x*qz;s[6*current+2]+=z*qy-y*qz;
                        s[6*current+3]+=x*dot-norm*qx;s[6*current+4]+=y*dot-norm*qy;s[6*current+5]+=z*dot-norm*qz;
                    }
                }
                for(int batch=0;batch<batches_;++batch)for(int current=0;current<2;++current)for(int c=0;c<3;++c){
                    fields[batch][p][6*current+c]+=curl_scale*sum[batch][6*current+c];
                    fields[batch][p][6*current+3+c]+=action_scale*sum[batch][6*current+3+c];
                }
            }
            times[4]+=elapsed(start);
        }
        if(!b.near)return;
        auto start=Clock::now();
#ifdef BEM_CPU_HAS_AVX512_P2P
        if(b.vector_near){
            if(shared_near_){
                const Six* inputs[8];for(int j=0;j<batches_;++j)inputs[j]=charges[j].data();
#pragma omp parallel for schedule(dynamic,16)
                for(size_t t=0;t<b.target_ids.size();++t){const int p=b.target_ids[t];Twelve* outputs[8];
                    for(int j=0;j<batches_;++j)outputs[j]=&fields[j][p];
                    avx512::near_field_batch(b.near_soa,b.near_sources[b.point_leaf[p]],b.points[p],f.k,inputs,outputs,batches_);
                }
                times[5]=elapsed(start);return;
            }
            // Original separately evaluated AVX512 path remains the default.
            for(int batch=0;batch<batches_;++batch){
#pragma omp parallel for schedule(static)
                for(size_t p=0;p<b.points.size();++p)for(int c=0;c<6;++c){
                    b.near_soa[3+2*c][p]=charges[batch][p][c].real();b.near_soa[4+2*c][p]=charges[batch][p][c].imag();}
#pragma omp parallel for schedule(dynamic,16)
                for(size_t t=0;t<b.target_ids.size();++t){const int p=b.target_ids[t];
                    avx512::near_field(b.near_soa,b.near_sources[b.point_leaf[p]],b.points[p],f.k,fields[batch][p]);}
            }
            times[5]=elapsed(start);return;
        }
#endif
        const C ik=C(0,1)*f.k,k2=f.k*f.k;
#pragma omp parallel for schedule(dynamic,16)
        for(size_t t=0;t<b.target_ids.size();++t){const int p=b.target_ids[t];Twelve sum[8]{};
            for(int s:b.near_sources[b.point_leaf[p]]){
                const Vec3 r=b.points[p]-b.points[s];const double r2=r.norm2();if(r2<1e-28)continue;
                const double ir=1/std::sqrt(r2),distance=r2*ir,x=r.x*ir,y=r.y*ir,z=r.z*ir;
                const C green=std::exp(ik*distance)*(INV4PI*ir),gradient=green*(ik-ir);
                const C a=green*(3*ir*ir-3.*ik*ir-k2),v=green*(k2+ik*ir-ir*ir);
                for(int batch=0;batch<batches_;++batch)for(int current=0;current<2;++current){
                    const auto& q=charges[batch][s];const C qx=q[3*current],qy=q[3*current+1],qz=q[3*current+2],dot=x*qx+y*qy+z*qz;
                    auto& out=sum[batch];out[6*current]+=gradient*(y*qx-x*qy);out[6*current+1]+=gradient*(z*qx-x*qz);out[6*current+2]+=gradient*(z*qy-y*qz);
                    out[6*current+3]+=a*x*dot+v*qx;out[6*current+4]+=a*y*dot+v*qy;out[6*current+5]+=a*z*dot+v*qz;
                }
            }
            for(int batch=0;batch<batches_;++batch)for(int c=0;c<12;++c)fields[batch][p][c]+=sum[batch][c];
        }
        times[5]=elapsed(start);
    }
};
}
#endif
