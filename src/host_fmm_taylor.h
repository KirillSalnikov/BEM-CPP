#pragma once
#include "host_fmm_band.h"
#include "helmholtz_taylor.h"

namespace bem_cpu {
// Reference matrix-free AWE action on the prepared fixed FMM discretization.
// The full history is propagated through every stage, including the k^2/k^3
// L2P factors and exact near field. CPU-owned buffers; no CUDA calls.
// No angular-band merging or persistent source-history cache in this version.
template<size_t N> class TaylorFmmBand {
public:
    using Charge=std::array<Six,N+1>;
    using Field=std::array<Twelve,N+1>;
private:
    using Series=std::array<C,N+1>;
    const HelmholtzFMM& f;
    const std::vector<Vec3>& points;
    C slope;
    int L;
    std::vector<int> leaf,source,target;
    std::vector<std::vector<int>> near_sources;
    struct Interaction{int source,transfer;};
    std::vector<std::vector<Interaction>> interactions;
    std::vector<std::vector<int>> children;
    std::vector<Series> transfer;
    std::vector<Charge> multi,local;
    static Series phase(C value,C rate){
        Series a{};a[0]=std::exp(value);for(size_t j=1;j<=N;++j)a[j]=a[j-1]*rate/double(j);return a;
    }
    Series translation(int child,int parent,int l,int sign)const{
        double dot=0;for(int a=0;a<3;++a)dot+=(f.tree.nodes[child].center[a]-f.tree.nodes[parent].center[a])*f.squad.dirs[3*l+a];
        return phase(C(0,sign)*f.k*dot,C(0,sign)*slope*dot);
    }
    Series point_phase(int p,int l,int sign)const{
        const auto& node=f.tree.nodes[leaf[p]];const Vec3 r=points[p]-Vec3(node.center[0],node.center[1],node.center[2]);
        const double dot=r.x*f.squad.dirs[3*l]+r.y*f.squad.dirs[3*l+1]+r.z*f.squad.dirs[3*l+2];
        return phase(C(0,sign)*f.k*dot,C(0,sign)*slope*dot);
    }
    static void accumulate(Charge& to,const Series& kernel,const Charge& from,size_t output_degree,size_t source_degree){
        for(size_t j=0;j<=output_degree;++j)for(size_t q=j>source_degree?j-source_degree:0;q<=j;++q)
            for(int c=0;c<6;++c)to[j][c]+=kernel[q]*from[j-q][c];
    }
public:
    TaylorFmmBand(const HelmholtzFMM& prepared,const std::vector<Vec3>& pts,C k_slope,
                  const std::vector<int>* source_map=nullptr,const std::vector<int>* target_map=nullptr)
        :f(prepared),points(pts),slope(k_slope),L(f.L){
        static_assert(N<=16,"FMM Taylor order");
        if(!f.initialized||L<1||bool(source_map)!=bool(target_map)||!std::isfinite(slope.real())||!std::isfinite(slope.imag()))
            throw std::invalid_argument("FMM Taylor metadata");
        if(source_map){source=*source_map;target=*target_map;}
        else{for(size_t i=0;i<pts.size();++i)source.push_back(int(i));target=source;}
        if(int(source.size())!=f.Ns||int(target.size())!=f.Nt)throw std::invalid_argument("FMM Taylor point maps");
        for(const auto* map:{&source,&target}){std::vector<bool> seen(pts.size());
            for(int p:*map){if(p<0||p>=int(pts.size())||seen[p])throw std::invalid_argument("FMM Taylor duplicate/invalid point");seen[p]=true;}}
        leaf.assign(pts.size(),-1);std::vector<int> ordinal(f.n_nodes,-1);
        for(size_t j=0;j<f.h_leaf_indices.size();++j){const int node=f.h_leaf_indices[j];ordinal[node]=int(j);
            auto assign=[&](int p){if(leaf[p]!=-1&&leaf[p]!=node)throw std::invalid_argument("FMM Taylor leaf map");leaf[p]=node;};
            for(int s=f.h_src_id_offsets[j];s<f.h_src_id_offsets[j+1];++s)assign(source.at(f.h_src_ids_flat[s]));
            for(int t=f.h_tgt_id_offsets[j];t<f.h_tgt_id_offsets[j+1];++t)assign(target.at(f.h_tgt_ids_flat[t]));
        }
        for(int p:leaf)if(p<0)throw std::invalid_argument("FMM Taylor uncovered point");
        children.resize(f.n_nodes);interactions.resize(f.n_nodes);near_sources.resize(f.n_nodes);
        for(const auto& level:f.m2m_data)for(auto e:level.pairs)children[e.parent].push_back(e.child);
        transfer.resize(size_t(f.n_unique_transfers)*L);std::vector<bool> ready(f.n_unique_transfers);
        for(const auto& level:f.m2l_batches)for(int j=0;j<level.n_pairs;++j){
            const int t=level.tgt_idx[j],s=level.src_idx[j],id=level.transfer_idx[j];interactions[t].push_back({s,id});
            if(ready.at(id))continue;
            double d[3],r2=0;for(int a=0;a<3;++a){d[a]=f.tree.nodes[t].center[a]-f.tree.nodes[s].center[a];r2+=d[a]*d[a];}
            const double r=std::sqrt(r2);const auto radial=bem_frequency::m2l_radial<N>(bem_frequency::C(f.k),bem_frequency::C(slope),r,f.p);
            for(int l=0;l<L;++l){double cosine=0;for(int a=0;a<3;++a)cosine+=d[a]/r*f.squad.dirs[3*l+a];
                const auto jet=bem_frequency::m2l_direction<N>(radial,cosine);
                for(size_t q=0;q<=N;++q)transfer[size_t(id)*L+l][q]=C(jet[q]);
            }ready[id]=true;
        }
        // Check a whole symbol norm: individual transfer entries can vanish.
        for(size_t id=0;id<ready.size();++id)if(ready[id]){double a=0,b=0;
            for(int l=0;l<L;++l){a+=std::norm(transfer[id*L+l][0]-f.transfer_cache.at(id*L+l));b+=std::norm(f.transfer_cache[id*L+l]);}
            if(!std::isfinite(a+b)||a>1e-16*std::max(b,1e-280))throw std::runtime_error("FMM Taylor central transfer differs from prepared operator");
        }
        if(f.p2p_enabled)for(int node:f.h_leaf_indices){
            auto append=[&](int s){const int j=ordinal.at(s);if(j<0)throw std::invalid_argument("FMM Taylor near nonleaf");
                for(int q=f.h_src_id_offsets[j];q<f.h_src_id_offsets[j+1];++q)near_sources[node].push_back(source.at(f.h_src_ids_flat[q]));};
            append(node);const auto& v=f.tree.nodes[node];for(int q=0;q<v.near_count;++q)append(f.tree.near_list[v.near_start+q]);
        }
        if(!transfer.empty()){multi.resize(size_t(f.n_nodes)*L);local.resize(multi.size());}
    }
    size_t workspace_bytes()const{return (multi.size()+local.size())*sizeof(Charge)+transfer.size()*sizeof(Series);}
    void add(const std::vector<Charge>& charge,std::vector<Field>& field,size_t input_degree=N,size_t output_degree=N){
        if(charge.size()!=points.size()||field.size()!=points.size())throw std::invalid_argument("FMM Taylor field dimensions");
        if(input_degree>output_degree||output_degree>N)throw std::invalid_argument("FMM Taylor history prefix");
        if(!multi.empty()){
            std::fill(multi.begin(),multi.end(),Charge{});std::fill(local.begin(),local.end(),Charge{});
#pragma omp parallel for schedule(dynamic,1)
            for(size_t j=0;j<f.h_leaf_indices.size();++j){const int node=f.h_leaf_indices[j];
                for(int s=f.h_src_id_offsets[j];s<f.h_src_id_offsets[j+1];++s){const int p=source[f.h_src_ids_flat[s]];
                    for(int l=0;l<L;++l)accumulate(multi[size_t(node)*L+l],point_phase(p,l,-1),charge[p],output_degree,input_degree);}
            }
            for(int level=f.tree.max_level-1;level>=std::max(1,f.minimum_m2l_level);--level){
                const auto& nodes=f.tree.level_nodes[level];
#pragma omp parallel for schedule(dynamic,1)
                for(size_t j=0;j<nodes.size();++j){const int node=nodes[j];for(int child:children[node])for(int l=0;l<L;++l)
                    accumulate(multi[size_t(node)*L+l],translation(child,node,l,-1),multi[size_t(child)*L+l],output_degree,output_degree);}
            }
#pragma omp parallel for schedule(dynamic,1)
            for(int node=0;node<f.n_nodes;++node)for(auto e:interactions[node])for(int l=0;l<L;++l)
                accumulate(local[size_t(node)*L+l],transfer[size_t(e.transfer)*L+l],multi[size_t(e.source)*L+l],output_degree,output_degree);
            for(int level=std::max(2,f.minimum_m2l_level+1);level<=f.tree.max_level;++level){const auto& data=f.l2l_data[level];
#pragma omp parallel for schedule(dynamic,1)
                for(size_t j=0;j<data.pairs.size();++j){const auto e=data.pairs[j];for(int l=0;l<L;++l)
                    accumulate(local[size_t(e.child)*L+l],translation(e.child,e.parent,l,1),local[size_t(e.parent)*L+l],output_degree,output_degree);}
            }
            const auto k=bem_frequency::variable<N>(bem_frequency::C(f.k),bem_frequency::C(slope));
            const auto square=bem_frequency::product<N>(k,k),cube=bem_frequency::product<N>(square,k);
            const double denominator=16*M_PI*M_PI;
#pragma omp parallel for schedule(dynamic,1)
            for(size_t t=0;t<target.size();++t){const int p=target[t];Field sum{};
                for(int l=0;l<L;++l){const double x=f.squad.dirs[3*l],y=f.squad.dirs[3*l+1],z=f.squad.dirs[3*l+2],norm=x*x+y*y+z*z;
                    Charge value{};auto e=point_phase(p,l,1);for(auto& q:e)q*=f.squad.weights[l];accumulate(value,e,local[size_t(leaf[p])*L+l],output_degree,output_degree);
                    for(size_t j=0;j<=output_degree;++j)for(int c=0;c<2;++c){const C qx=value[j][3*c],qy=value[j][3*c+1],qz=value[j][3*c+2],dot=x*qx+y*qy+z*qz;
                        sum[j][6*c]+=y*qx-x*qy;sum[j][6*c+1]+=z*qx-x*qz;sum[j][6*c+2]+=z*qy-y*qz;
                        sum[j][6*c+3]+=x*dot-norm*qx;sum[j][6*c+4]+=y*dot-norm*qy;sum[j][6*c+5]+=z*dot-norm*qz;}
                }
                for(size_t j=0;j<=output_degree;++j)for(size_t q=0;q<=j;++q)for(int c=0;c<2;++c)for(int a=0;a<3;++a){
                    field[p][j][6*c+a]-=C(square[q])/denominator*sum[j-q][6*c+a];
                    field[p][j][6*c+3+a]-=C(0,1)*C(cube[q])/denominator*sum[j-q][6*c+3+a];}
            }
        }
        if(f.p2p_enabled){
#pragma omp parallel for schedule(dynamic,1)
            for(size_t t=0;t<target.size();++t){const int p=target[t];
                for(int s:near_sources[leaf[p]]){const Vec3 d=points[p]-points[s];const double r2=d.norm2();if(r2<1e-28)continue;
                    const double r=std::sqrt(r2),ir=1/r,x=d.x*ir,y=d.y*ir,z=d.z*ir;
                    const C ik=C(0,1)*f.k,is=C(0,1)*slope;
                    auto green=phase(ik*r,is*r);for(auto& q:green)q*=INV4PI*ir;
                    Series gradient{},a{},b{};
                    for(size_t j=0;j<=output_degree;++j){const C previous=j?green[j-1]:C(0),previous2=j>1?green[j-2]:C(0);
                        gradient[j]=green[j]*(ik-ir)+previous*is;
                        a[j]=green[j]*(3*ir*ir-3.*ik*ir-f.k*f.k)+previous*(-3.*is*ir-2.*f.k*slope)-previous2*slope*slope;
                        b[j]=green[j]*(f.k*f.k+ik*ir-ir*ir)+previous*(2.*f.k*slope+is*ir)+previous2*slope*slope;
                    }
                    for(size_t j=0;j<=output_degree;++j)for(size_t q=j>input_degree?j-input_degree:0;q<=j;++q)for(int c=0;c<2;++c){const auto& v=charge[s][j-q];
                        const C qx=v[3*c],qy=v[3*c+1],qz=v[3*c+2],dot=x*qx+y*qy+z*qz;
                        field[p][j][6*c]+=gradient[q]*(y*qx-x*qy);field[p][j][6*c+1]+=gradient[q]*(z*qx-x*qz);field[p][j][6*c+2]+=gradient[q]*(z*qy-y*qz);
                        field[p][j][6*c+3]+=a[q]*x*dot+b[q]*qx;field[p][j][6*c+4]+=a[q]*y*dot+b[q]*qy;field[p][j][6*c+5]+=a[q]*z*dot+b[q]*qz;
                    }
                }
            }
        }
    }
};
}
