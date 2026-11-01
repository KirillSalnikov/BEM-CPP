#ifndef BEM_HOST_FMM_BAND_H
#define BEM_HOST_FMM_BAND_H

#include "muller_fmm.h"
#include "muller_fmm_cpu_avx512.h"
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>

// Experimental independent FP64 host action. No CUDA calls, shared workspaces,
// or changes to the GPU operator's precision flags. The prepared operator must
// outlive this object; its geometry/trees/correction must not be rebuilt while
// either action runs. One CPU action at a time per instance. Different angular
// grids are evaluated separately; equal grids can share exact tree shifts.
// There is no angular interpolation in this CPU reference.
// Used by the opt-in T-worker hybrid gate; GPU work remains on its owner thread.
namespace bem_cpu {
using C = std::complex<double>;
using Six = std::array<C, 6>;
using Twelve = std::array<C, 12>;
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}

class FmmBand {
    friend class FmmBandBatch;
    const HelmholtzFMM& f;
    const std::vector<Vec3>& points;
    int L, depth, minimum;
    bool near, far;
    struct Edge { int node; const C* shift; };
    struct Transfer { int source; const C* values; };
    std::vector<std::vector<Edge>> up;
    std::vector<std::vector<Transfer>> interaction;
    std::vector<int> point_leaf;
    // Local FMM point IDs may describe two distinct subsets of one host mesh.
    // The original coincident-point path uses identity maps.
    std::vector<int> source_ids,target_ids;
    std::vector<std::vector<int>> near_sources;
    std::vector<Six> multi, local;
    std::vector<C> phase;
    std::array<std::vector<double>,15> near_soa;
    bool vector_near=false;

    C outgoing_phase(int p, int l) const {
        if (!phase.empty()) return phase[size_t(p)*L+l];
        const auto& node=f.tree.nodes[point_leaf[p]];
        const Vec3 r=points[p]-Vec3(node.center[0],node.center[1],node.center[2]);
        const double dot=r.x*f.squad.dirs[3*l]+r.y*f.squad.dirs[3*l+1]+r.z*f.squad.dirs[3*l+2];
        return std::exp(C(0,-1)*f.k*dot);
    }

public:
    // P2M, M2M, M2L, L2L, L2P, P2P, in seconds for the last action.
    std::array<double,6> times{};
    bool uses_avx512_near() const {return vector_near;}
    size_t phase_bytes() const { return phase.size()*sizeof(C); }
    size_t workspace_bytes() const {
        size_t bytes=(multi.size()+local.size())*sizeof(Six)+phase_bytes();
        for(const auto& a:near_soa)bytes+=a.size()*sizeof(double);
        return bytes;
    }
    FmmBand(const HelmholtzFMM& engine, const std::vector<Vec3>& pts,
            size_t phase_budget=64*1024*1024,const HelmholtzFMM* upper=nullptr,
            const std::vector<int>* source_map=nullptr,const std::vector<int>* target_map=nullptr,
            bool defer_single_workspace=false)
        : f(engine),points(pts),L(f.L),depth(f.tree.max_level),
          minimum(f.minimum_m2l_level),near(f.p2p_enabled),far(false) {
        if (!f.initialized || bool(source_map)!=bool(target_map))
            throw std::runtime_error("CPU FMM requires prepared trees and both point maps");
        if(source_map){source_ids=*source_map;target_ids=*target_map;}
        else {for(size_t i=0;i<points.size();++i)source_ids.push_back(int(i));target_ids=source_ids;}
        if(f.Ns!=int(source_ids.size()) || f.Nt!=int(target_ids.size()))
            throw std::runtime_error("CPU FMM source/target map dimensions");
        for(const auto* map:{&source_ids,&target_ids}){
            std::vector<bool> seen(points.size(),false);
            for(int i:*map){
                if(i<0||i>=int(points.size())||seen[i])throw std::runtime_error("CPU FMM invalid/duplicate point map");
                seen[i]=true;
            }
        }
        up.resize(f.n_nodes);interaction.resize(f.n_nodes);
        for (const auto& level:f.m2l_batches)
            for (int j=0;j<level.n_pairs;++j) {
                interaction[level.tgt_idx[j]].push_back({level.src_idx[j],
                    f.transfer_cache.data()+size_t(level.transfer_idx[j])*L});
                far=true;
            }
        if(upper) {
            // Identical angular grids need only the ordinary exact M2M/L2L
            // shifts. Fold the upper band's transfers into the fine tree,
            // avoiding a second P2M/L2P and duplicate point-phase storage.
            if(upper->L!=L || upper->k!=f.k || upper->Nt!=f.Nt || upper->Ns!=f.Ns ||
               upper->squad.dirs!=f.squad.dirs || upper->squad.weights!=f.squad.weights ||
               upper->p2p_enabled || upper->maximum_m2l_level>=minimum)
                throw std::runtime_error("incompatible CPU equal-angle band merge");
            using Key=std::tuple<int,int,int,int>;
            std::map<Key,int> index;
            for(const auto& node:f.tree.nodes)index[Key(node.level,node.gx,node.gy,node.gz)]=node.index;
            std::vector<int> mapping(upper->n_nodes);
            for(const auto& node:upper->tree.nodes) {
                const auto found=index.find(Key(node.level,node.gx,node.gy,node.gz));
                if(found==index.end())throw std::runtime_error("CPU band trees do not nest");
                const auto& target=f.tree.nodes[found->second];
                for(int a=0;a<3;++a)if(target.center[a]!=node.center[a])
                    throw std::runtime_error("CPU band centers differ");
                if(target.half_size!=node.half_size)throw std::runtime_error("CPU band box sizes differ");
                mapping[node.index]=found->second;
            }
            for(const auto& level:upper->m2l_batches)
                for(int j=0;j<level.n_pairs;++j) {
                    interaction[mapping[level.tgt_idx[j]]].push_back({mapping[level.src_idx[j]],
                        upper->transfer_cache.data()+size_t(level.transfer_idx[j])*L});
                    far=true;
                }
            minimum=std::min(minimum,upper->minimum_m2l_level);
        }
        for (const auto& level:f.m2m_data)
            for (size_t j=0;j<level.pairs.size();++j)
                up[level.pairs[j].parent].push_back({level.pairs[j].child,level.shifts.data()+j*L});
        point_leaf.assign(points.size(),-1);
        std::vector<int> ordinal(f.n_nodes,-1);
        for (size_t j=0;j<f.h_leaf_indices.size();++j) {
            ordinal[f.h_leaf_indices[j]]=int(j);
            const auto assign=[&](int point){
                if(point_leaf[point]>=0 && point_leaf[point]!=f.h_leaf_indices[j])
                    throw std::runtime_error("CPU FMM source/target leaf geometry mismatch");
                point_leaf[point]=f.h_leaf_indices[j];
            };
            for(int t=f.h_tgt_id_offsets[j];t<f.h_tgt_id_offsets[j+1];++t)assign(target_ids.at(f.h_tgt_ids_flat[t]));
            for(int s=f.h_src_id_offsets[j];s<f.h_src_id_offsets[j+1];++s)assign(source_ids.at(f.h_src_ids_flat[s]));
        }
        for (int leaf:point_leaf) if(leaf<0)throw std::runtime_error("missing CPU FMM target");
        if (near) {
#ifdef BEM_CPU_HAS_AVX512_P2P
            const char* setting=std::getenv("BEM_CPU_AVX512");
            vector_near=setting && std::atoi(setting)!=0 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl");
            if(vector_near) {
                for(auto& a:near_soa)a.resize(points.size());
                for(size_t p=0;p<points.size();++p) {
                    near_soa[0][p]=points[p].x;near_soa[1][p]=points[p].y;near_soa[2][p]=points[p].z;
                }
            }
#endif
            near_sources.resize(f.n_nodes);
            for (int ni:f.h_leaf_indices) {
                const auto& node=f.tree.nodes[ni];
                auto append=[&](int source_node) {
                    const int j=ordinal[source_node];
                    if(j<0)throw std::runtime_error("nonleaf CPU near source");
                    for(int s=f.h_src_id_offsets[j];s<f.h_src_id_offsets[j+1];++s)
                        near_sources[ni].push_back(source_ids[f.h_src_ids_flat[s]]);
                };
                append(ni);
                for(int j=0;j<node.near_count;++j)append(f.tree.near_list[node.near_start+j]);
            }
        }
        if (far) {
            if(!defer_single_workspace){multi.resize(size_t(f.n_nodes)*L);local.resize(multi.size());}
            if(points.size()*size_t(L)*sizeof(C)<=phase_budget) {
                // Compute before resize: outgoing_phase otherwise reads the cache.
                std::vector<C> values(points.size()*size_t(L));
#pragma omp parallel for schedule(static)
                for(size_t p=0;p<points.size();++p)
                    for(int l=0;l<L;++l)values[p*L+l]=outgoing_phase(int(p),l);
                phase.swap(values);
            }
        }
    }

    void add(const std::vector<Six>& charges,std::vector<Twelve>& fields) {
        if(charges.size()!=points.size() || fields.size()!=points.size())
            throw std::invalid_argument("CPU FMM charge/field dimensions");
        times.fill(0);
        auto start=Clock::now();
        if(far) {
            multi.resize(size_t(f.n_nodes)*L);local.resize(multi.size());
            std::fill(multi.begin(),multi.end(),Six{});
            std::fill(local.begin(),local.end(),Six{});
            const int tiles=(L+31)/32;
#pragma omp parallel for schedule(dynamic,1)
            for(int task=0;task<int(f.h_leaf_indices.size())*tiles;++task) {
                const int leaf=task/tiles,begin=(task%tiles)*32;
                const size_t base=size_t(f.h_leaf_indices[leaf])*L;
                Six sum[32]{};
                // Read adjacent phase entries together; the alternative
                // direction-major walk strides through a multi-GB cache.
                for(int j=f.h_src_id_offsets[leaf];j<f.h_src_id_offsets[leaf+1];++j) {
                    const int p=source_ids[f.h_src_ids_flat[j]];
                    for(int l=begin;l<std::min(begin+32,L);++l) {
                        const C e=outgoing_phase(p,l);
                        for(int c=0;c<6;++c)sum[l-begin][c]+=e*charges[p][c];
                    }
                }
                for(int l=begin;l<std::min(begin+32,L);++l)multi[base+l]=sum[l-begin];
            }
            times[0]=elapsed(start);start=Clock::now();
            for(int level=depth-1;level>=std::max(1,minimum);--level) {
                const auto& nodes=f.tree.level_nodes[level];
#pragma omp parallel for schedule(dynamic,1)
                for(size_t j=0;j<nodes.size();++j) {
                    const int node=nodes[j];
                    for(const auto& edge:up[node])
                        for(int l=0;l<L;++l)
                            for(int c=0;c<6;++c)
                                multi[size_t(node)*L+l][c]+=edge.shift[l]*multi[size_t(edge.node)*L+l][c];
                }
            }
            times[1]=elapsed(start);start=Clock::now();
#pragma omp parallel for schedule(dynamic,1)
            for(int node=0;node<f.n_nodes;++node)
                for(const auto& edge:interaction[node]) {
                    const C* transfer=edge.values;
                    for(int l=0;l<L;++l)
                        for(int c=0;c<6;++c)
                            local[size_t(node)*L+l][c]+=transfer[l]*multi[size_t(edge.source)*L+l][c];
                }
            times[2]=elapsed(start);start=Clock::now();
            for(int level=std::max(2,minimum+1);level<=depth;++level) {
                const auto& data=f.l2l_data[level];
#pragma omp parallel for schedule(dynamic,1)
                for(size_t j=0;j<data.pairs.size();++j) {
                    const auto& edge=data.pairs[j];const C* shift=data.shifts.data()+j*L;
                    for(int l=0;l<L;++l)
                        for(int c=0;c<6;++c)
                            local[size_t(edge.child)*L+l][c]+=shift[l]*local[size_t(edge.parent)*L+l][c];
                }
            }
            times[3]=elapsed(start);start=Clock::now();
            const C ik=C(0,1)*f.k,prefactor=ik/(16*M_PI*M_PI);
            const C curl_scale=prefactor*ik,action_scale=-prefactor*f.k*f.k;
#pragma omp parallel for schedule(dynamic,16)
            for(size_t t=0;t<target_ids.size();++t) {
                const int p=target_ids[t];
                Twelve sum{};const size_t base=size_t(point_leaf[p])*L;
                for(int l=0;l<L;++l) {
                    const double x=f.squad.dirs[3*l],y=f.squad.dirs[3*l+1],z=f.squad.dirs[3*l+2];
                    const double norm=x*x+y*y+z*z;
                    const C outgoing=outgoing_phase(int(p),l);
                    const C e=f.squad.weights[l]*(f.k.imag()==0 ? std::conj(outgoing) : C(1)/outgoing);
                    for(int j=0;j<2;++j) {
                        const C qx=e*local[base+l][3*j],qy=e*local[base+l][3*j+1],qz=e*local[base+l][3*j+2];
                        const C dot=x*qx+y*qy+z*qz;
                        sum[6*j]+=y*qx-x*qy;sum[6*j+1]+=z*qx-x*qz;sum[6*j+2]+=z*qy-y*qz;
                        sum[6*j+3]+=x*dot-norm*qx;sum[6*j+4]+=y*dot-norm*qy;sum[6*j+5]+=z*dot-norm*qz;
                    }
                }
                for(int j=0;j<2;++j)for(int c=0;c<3;++c) {
                    fields[p][6*j+c]+=curl_scale*sum[6*j+c];
                    fields[p][6*j+3+c]+=action_scale*sum[6*j+3+c];
                }
            }
            times[4]=elapsed(start);start=Clock::now();
        }
        if(near) {
#ifdef BEM_CPU_HAS_AVX512_P2P
            if(vector_near) {
#pragma omp parallel for schedule(static)
                for(size_t p=0;p<points.size();++p)
                    for(int c=0;c<6;++c) {
                        near_soa[3+2*c][p]=charges[p][c].real();
                        near_soa[4+2*c][p]=charges[p][c].imag();
                    }
#pragma omp parallel for schedule(dynamic,16)
                for(size_t t=0;t<target_ids.size();++t){
                    const int p=target_ids[t];
                    avx512::near_field(near_soa,near_sources[point_leaf[p]],points[p],f.k,fields[p]);
                }
                times[5]=elapsed(start);
                return;
            }
#endif
            const C ik=C(0,1)*f.k,k2=f.k*f.k;
#pragma omp parallel for schedule(dynamic,16)
            for(size_t t=0;t<target_ids.size();++t) {
                const int p=target_ids[t];
                Twelve sum{};
                for(int s:near_sources[point_leaf[p]]) {
                    const Vec3 r=points[p]-points[s];const double r2=r.norm2();
                    if(r2<1e-28)continue;
                    const double ir=1/std::sqrt(r2),distance=r2*ir;
                    const double x=r.x*ir,y=r.y*ir,z=r.z*ir;
                    const C green=std::exp(ik*distance)*(INV4PI*ir);
                    const C gradient=green*(ik-ir);
                    const C a=green*(3*ir*ir-3.*ik*ir-k2);
                    const C b=green*(k2+ik*ir-ir*ir);
                    for(int j=0;j<2;++j) {
                        const C qx=charges[s][3*j],qy=charges[s][3*j+1],qz=charges[s][3*j+2];
                        const C dot=x*qx+y*qy+z*qz;
                        sum[6*j]+=gradient*(y*qx-x*qy);
                        sum[6*j+1]+=gradient*(z*qx-x*qz);
                        sum[6*j+2]+=gradient*(z*qy-y*qz);
                        sum[6*j+3]+=a*x*dot+b*qx;
                        sum[6*j+4]+=a*y*dot+b*qy;
                        sum[6*j+5]+=a*z*dot+b*qz;
                    }
                }
                for(int c=0;c<12;++c)fields[p][c]+=sum[c];
            }
            times[5]=elapsed(start);
        }
    }
};

} // namespace bem_cpu
#endif
