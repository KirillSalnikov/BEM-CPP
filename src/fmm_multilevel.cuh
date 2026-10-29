// Included by fmm.cu. Strict residual actions always use FP64 transfers.
#include "sphere_interpolation.h"
#include <map>
#include <tuple>

namespace {
// Uniform octrees reuse the same child-to-parent displacement. Keep exact
// double-valued displacement keys (rather than rounding to octants), so even
// asymmetric/root-offset trees retain their original translation phases.
__global__ void multilevel_cache_shifts(const double* offsets,
    const double* dirs, const double* weights, double kr, double ki,
    double2* up, double2* down, int directions, int count) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count*directions)return;
    const int shift=index/directions,l=index%directions;
    double dot=0;
    for(int d=0;d<3;++d)dot+=dirs[3*l+d]*offsets[3*shift+d];
    double s,c;sincos(-kr*dot,&s,&c);
    const double a=ki==0?1.:exp(ki*dot);
    up[index]=make_double2(a*c,a*s);
    sincos(kr*dot,&s,&c);
    const double b=weights[l]*(ki==0?1.:exp(-ki*dot));
    down[index]=make_double2(b*c,b*s);
}
struct MultilevelArrays {
    double *r[3], *i[3];
};
MultilevelArrays multilevel_arrays(HelmholtzFMM& f, bool local) {
    MultilevelArrays a;
    a.r[0]=local?f.d_local_re:f.d_multi_re; a.i[0]=local?f.d_local_im:f.d_multi_im;
    a.r[1]=local?f.d_local2_re:f.d_multi2_re; a.i[1]=local?f.d_local2_im:f.d_multi2_im;
    a.r[2]=local?f.d_local3_re:f.d_multi3_re; a.i[2]=local?f.d_local3_im:f.d_multi3_im;
    return a;
}
struct MultilevelLink {
    HelmholtzFMM* coarse;
    int ntf, npf, ntc, npc, leaves;
    int chunk;
    double *te=nullptr, *to=nullptr, *pe=nullptr, *po=nullptr;
    int *parents=nullptr;
    int *shift_ids=nullptr;
    int *active_up=nullptr,*active_down=nullptr;
    int up_count=0,down_count=0;
    double2 *up_phase=nullptr, *down_phase=nullptr;
    double2 *even=nullptr, *odd=nullptr, *shifted=nullptr;
    MultilevelLink(HelmholtzFMM& f, HelmholtzFMM& c) : coarse(&c) {
        chunk=std::max(32,std::min(2048,bem_env_int("BEM_FMM_MULTILEVEL_CHUNK",64)));
        if (f.tree.max_level != c.tree.max_level+1 || f.k != c.k ||
            f.Ns != c.Ns || f.Nt != c.Nt || !f.batch3_device_allocated ||
            !c.batch3_device_allocated || c.p2p_enabled)
            throw std::runtime_error("incompatible multilevel FMM bands");
        for (int d=0;d<3;++d)
            if (f.tree.nodes[0].center[d] != c.tree.nodes[0].center[d])
                throw std::runtime_error("multilevel FMM root centers differ");
        if (f.tree.nodes[0].half_size != c.tree.nodes[0].half_size)
            throw std::runtime_error("multilevel FMM root sizes differ");
        SphereInterpolation interpolation(f.squad,c.squad);
        ntf=interpolation.ntf; npf=interpolation.npf;
        ntc=interpolation.ntc; npc=interpolation.npc;
        leaves=int(f.leaf_info.size());
        up_count=down_count=leaves;
        if(bem_env_flag_enabled("BEM_FMM_PRUNE_EMPTY",true)) {
            auto compact=[&](bool targets,int*& device,int& count) {
                std::vector<int> active;
                for(int i=0;i<leaves;++i)
                    if(targets?f.leaf_info[i].tgt_count>0:f.leaf_info[i].src_count>0)active.push_back(i);
                count=int(active.size());
                if(count && count!=leaves) {
                    CUDA_CHECK(cudaMalloc(&device,count*sizeof(int)));
                    CUDA_CHECK(cudaMemcpy(device,active.data(),count*sizeof(int),cudaMemcpyHostToDevice));
                }
            };
            compact(false,active_up,up_count);compact(true,active_down,down_count);
        }
        auto upload=[](double*& dst,const std::vector<double>& src) {
            CUDA_CHECK(cudaMalloc(&dst,src.size()*sizeof(double)));
            CUDA_CHECK(cudaMemcpy(dst,src.data(),src.size()*sizeof(double),cudaMemcpyHostToDevice));
        };
        upload(te,interpolation.theta_even); upload(to,interpolation.theta_odd);
        upload(pe,interpolation.phi_even); upload(po,interpolation.phi_odd);
        std::map<std::tuple<int,int,int>,int> nodes;
        for (int n : c.tree.leaves) {
            const auto& v=c.tree.nodes[n]; nodes[std::make_tuple(v.gx,v.gy,v.gz)]=n;
        }
        std::vector<int> mapping(leaves);
        for (int i=0;i<leaves;++i) {
            const auto& v=f.tree.nodes[f.leaf_info[i].node_idx];
            auto it=nodes.find(std::make_tuple(v.gx/2,v.gy/2,v.gz/2));
            if (it==nodes.end()) throw std::runtime_error("multilevel FMM parent missing");
            mapping[i]=it->second;
        }
        CUDA_CHECK(cudaMalloc(&parents,leaves*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(parents,mapping.data(),leaves*sizeof(int),cudaMemcpyHostToDevice));
        if(bem_env_flag_enabled("BEM_FMM_MULTILEVEL_SHIFT_CACHE")) {
            std::map<std::tuple<double,double,double>,int> unique;
            std::vector<double> offsets;
            std::vector<int> ids(leaves);
            for(int i=0;i<leaves;++i) {
                const auto& child=f.tree.nodes[f.leaf_info[i].node_idx].center;
                const auto& parent=c.tree.nodes[mapping[i]].center;
                const auto key=std::make_tuple(child[0]-parent[0],child[1]-parent[1],child[2]-parent[2]);
                auto inserted=unique.emplace(key,int(unique.size()));
                ids[i]=inserted.first->second;
                if(inserted.second) {
                    offsets.push_back(std::get<0>(key));offsets.push_back(std::get<1>(key));offsets.push_back(std::get<2>(key));
                }
            }
            double* device_offsets=nullptr;
            upload(device_offsets,offsets);
            CUDA_CHECK(cudaMalloc(&shift_ids,leaves*sizeof(int)));
            CUDA_CHECK(cudaMemcpy(shift_ids,ids.data(),leaves*sizeof(int),cudaMemcpyHostToDevice));
            const size_t phase_count=unique.size()*c.L;
            CUDA_CHECK(cudaMalloc(&up_phase,phase_count*sizeof(double2)));
            CUDA_CHECK(cudaMalloc(&down_phase,phase_count*sizeof(double2)));
            multilevel_cache_shifts<<<(phase_count+255)/256,256>>>(device_offsets,c.d_dirs_cached,
                c.d_weights_cached,f.k.real(),f.k.imag(),up_phase,down_phase,c.L,int(unique.size()));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaFree(device_offsets));
            std::printf("  [MLFMA] FP64 shift cache: %zu exact displacements, %.3f MiB\n",
                unique.size(),double(phase_count*2*sizeof(double2)+leaves*sizeof(int))/(1024*1024));
        }
        const size_t bytes=size_t(chunk)*3*ntf*npc*sizeof(double2);
        CUDA_CHECK(cudaMalloc(&even,bytes)); CUDA_CHECK(cudaMalloc(&odd,bytes));
        CUDA_CHECK(cudaMalloc(&shifted,size_t(chunk)*3*ntc*npc*sizeof(double2)));
        std::printf("  [MLFMA] level %d -> %d, p %d -> %d, %d leaves; chunk=%d, FP64 storage\n",
                    f.tree.max_level,c.tree.max_level,f.p,c.p,leaves,chunk);
    }
    ~MultilevelLink() {
        cudaFree(te);cudaFree(to);cudaFree(pe);cudaFree(po);cudaFree(parents);
        cudaFree(even);cudaFree(odd);cudaFree(shifted);
        cudaFree(shift_ids);cudaFree(up_phase);cudaFree(down_phase);
        cudaFree(active_up);cudaFree(active_down);
    }
};

// Intermediate sums already have Real precision. Mixed actions can retain
// them as float2 instead of widening to double2 and narrowing on every read.
// The down-shift product still evaluates in FP64 before the single cast that
// the original down-theta kernel also performed. Strict actions use double2.
template <typename Temporary>
__device__ __forceinline__ Temporary multilevel_temporary(double real, double imag);
template <>
__device__ __forceinline__ double2 multilevel_temporary<double2>(double real, double imag) {
    return make_double2(real, imag);
}
template <>
__device__ __forceinline__ float2 multilevel_temporary<float2>(double real, double imag) {
    return make_float2(float(real), float(imag));
}

// Phi interpolation for all three vector components. A thread computes one
// (leaf, component, fine theta, coarse phi) value, separately for each parity.
template <typename Real, typename Temporary>
__global__ void multilevel_up_phi(MultilevelArrays in,const int* nodes,
    const double* pe,const double* po,Temporary* even,Temporary* odd,
    int ntf,int npf,int npc,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count*3*ntf*npc)return;
    const int ip=id%npc, it=(id/npc)%ntf, comp=(id/(npc*ntf))%3;
    const int leaf=first+id/(3*ntf*npc);
    const int node=nodes[active?active[leaf]:leaf];
    const size_t base=size_t(node)*ntf*npf+it*npf;
    Real er=0,ei=0,orr=0,oi=0;
    for(int j=0;j<npf;++j) {
        const Real r=Real(in.r[comp][base+j]),v=Real(in.i[comp][base+j]);
        const Real a=Real(pe[ip*npf+j]),b=Real(po[ip*npf+j]);
        er+=a*r;ei+=a*v;orr+=b*r;oi+=b*v;
    }
    even[id]=multilevel_temporary<Temporary>(er,ei);odd[id]=multilevel_temporary<Temporary>(orr,oi);
}
template <typename Real, typename Temporary>
__global__ void multilevel_up_theta(MultilevelArrays out,const int* nodes,const int* parents,
    const double* fc,const double* cc,const double* dirs,const double* te,const double* to,
    const Temporary* even,const Temporary* odd,const int* shift_ids,const double2* phase_cache,int ntf,int ntc,int npc,
    double kr,double ki,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count*3*ntc*npc)return;
    const int ip=id%npc,it=(id/npc)%ntc,comp=(id/(ntc*npc))%3;
    const int leaf=id/(3*ntc*npc),index=active?active[first+leaf]:first+leaf;
    const int node=nodes[index],parent=parents[index];
    const int l=it*npc+ip;
    Real r=0,v=0;
    for(int j=0;j<ntf;++j) {
        const size_t index=((size_t(leaf)*3+comp)*ntf+j)*npc+ip;
        const Real a=Real(te[it*ntf+j]),b=Real(to[it*ntf+j]);
        r+=a*Real(even[index].x)+b*Real(odd[index].x);
        v+=a*Real(even[index].y)+b*Real(odd[index].y);
    }
    double2 phase;
    if(phase_cache)phase=phase_cache[size_t(shift_ids[index])*ntc*npc+l];
    else {
        double dot=0;
        for(int d=0;d<3;++d)dot+=dirs[3*l+d]*(fc[3*node+d]-cc[3*parent+d]);
        double s,c;sincos(-kr*dot,&s,&c);const double amp=ki==0?1.:exp(ki*dot);
        phase=make_double2(amp*c,amp*s);
    }
    const size_t dst=size_t(parent)*ntc*npc+l;
    atomicAdd(out.r[comp]+dst,phase.x*r-phase.y*v);
    atomicAdd(out.i[comp]+dst,phase.y*r+phase.x*v);
}

// Downward transfer is W_f^{-1} I^T W_c after the positive translation phase.
// This is the quadrature-weighted transpose, NOT ordinary interpolation of
// local coefficients. That distinction preserves the source/target duality.
template <typename Temporary>
__global__ void multilevel_down_shift(MultilevelArrays in,const int* nodes,const int* parents,
    const double* fc,const double* cc,const double* dirs,const double* weights,
    Temporary* shifted,const int* shift_ids,const double2* phase_cache,int ntc,int npc,double kr,double ki,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count*3*ntc*npc)return;
    const int l=id%(ntc*npc),comp=(id/(ntc*npc))%3;
    const int leaf=id/(3*ntc*npc),active_index=active?active[first+leaf]:first+leaf;
    const int node=nodes[active_index],parent=parents[active_index];
    double2 phase;
    if(phase_cache)phase=phase_cache[size_t(shift_ids[active_index])*ntc*npc+l];
    else {
        double dot=0;
        for(int d=0;d<3;++d)dot+=dirs[3*l+d]*(fc[3*node+d]-cc[3*parent+d]);
        double s,c;sincos(kr*dot,&s,&c);
        const double amp=weights[l]*(ki==0?1.:exp(-ki*dot));
        phase=make_double2(amp*c,amp*s);
    }
    const size_t index=size_t(parent)*ntc*npc+l;
    const double r=in.r[comp][index],v=in.i[comp][index];
    shifted[id]=multilevel_temporary<Temporary>(phase.x*r-phase.y*v,phase.y*r+phase.x*v);
}
template <typename Real, typename Temporary>
__global__ void multilevel_down_theta(const Temporary* shifted,
    const double* te,const double* to,Temporary* even,Temporary* odd,
    int ntf,int ntc,int npc,int count) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count*3*ntf*npc)return;
    const int ip=id%npc,it=(id/npc)%ntf,batch=id/(ntf*npc);
    Real er=0,ei=0,orr=0,oi=0;
    for(int j=0;j<ntc;++j) {
        const Temporary v=shifted[(size_t(batch)*ntc+j)*npc+ip];
        const Real a=Real(te[j*ntf+it]),b=Real(to[j*ntf+it]);
        er+=a*Real(v.x);ei+=a*Real(v.y);orr+=b*Real(v.x);oi+=b*Real(v.y);
    }
    even[id]=multilevel_temporary<Temporary>(er,ei);odd[id]=multilevel_temporary<Temporary>(orr,oi);
}
template <typename Real, typename Temporary>
__global__ void multilevel_down_phi(MultilevelArrays out,const int* nodes,const double* weights,
    const double* pe,const double* po,const Temporary* even,const Temporary* odd,
    int ntf,int npf,int npc,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count*3*ntf*npf)return;
    const int ip=id%npf,it=(id/npf)%ntf,comp=(id/(ntf*npf))%3;
    const int leaf=id/(3*ntf*npf),index=active?active[first+leaf]:first+leaf,node=nodes[index];
    Real r=0,v=0;
    for(int j=0;j<npc;++j) {
        const size_t index=((size_t(leaf)*3+comp)*ntf+it)*npc+j;
        const Real a=Real(pe[j*npf+ip]),b=Real(po[j*npf+ip]);
        r+=a*Real(even[index].x)+b*Real(odd[index].x);
        v+=a*Real(even[index].y)+b*Real(odd[index].y);
    }
    const size_t dst=size_t(node)*ntf*npf+it*npf+ip;
    out.r[comp][dst]+=r/weights[it*npf+ip];
    out.i[comp][dst]+=v/weights[it*npf+ip];
}
// Phi grids have an even number of nodes. Their even interpolation kernel
// is unchanged by a pi shift, while the odd kernel changes sign. Fold the
// source pair once and retain only half of the temporary azimuths. The
// downward kernels implement the exact weighted transpose of this folding.
template <typename Real,typename Temporary>
__global__ void multilevel_up_phi_parity(MultilevelArrays in,const int* nodes,
    const double* pe,const double* po,Temporary* even,Temporary* odd,
    int ntf,int npf,int npc,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x,half=npc/2;
    if(id>=count*3*ntf*half)return;
    const int ip=id%half,it=(id/half)%ntf,comp=(id/(half*ntf))%3;
    const int leaf=first+id/(3*ntf*half),node=nodes[active?active[leaf]:leaf];
    const size_t base=size_t(node)*ntf*npf+it*npf;
    Real er=0,ei=0,orr=0,oi=0;
    for(int j=0;j<npf/2;++j) {
        const Real r=Real(in.r[comp][base+j]),v=Real(in.i[comp][base+j]);
        const Real r2=Real(in.r[comp][base+j+npf/2]),v2=Real(in.i[comp][base+j+npf/2]);
        const Real a=Real(pe[ip*npf+j]),b=Real(po[ip*npf+j]);
        er+=a*(r+r2);ei+=a*(v+v2);orr+=b*(r-r2);oi+=b*(v-v2);
    }
    even[id]=multilevel_temporary<Temporary>(er,ei);odd[id]=multilevel_temporary<Temporary>(orr,oi);
}
template <typename Real,typename Temporary>
__global__ void multilevel_up_theta_parity(MultilevelArrays out,const int* nodes,const int* parents,
    const double* fc,const double* cc,const double* dirs,const double* te,const double* to,
    const Temporary* even,const Temporary* odd,const int* shift_ids,const double2* phase_cache,
    int ntf,int ntc,int npc,double kr,double ki,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x,half=npc/2;
    if(id>=count*3*ntc*half)return;
    const int ip=id%half,it=(id/half)%ntc,comp=(id/(ntc*half))%3;
    const int leaf=id/(3*ntc*half),index=active?active[first+leaf]:first+leaf;
    const int node=nodes[index],parent=parents[index];
    Real er=0,ei=0,orr=0,oi=0;
    for(int j=0;j<ntf;++j) {
        const size_t at=((size_t(leaf)*3+comp)*ntf+j)*half+ip;
        const Real a=Real(te[it*ntf+j]),b=Real(to[it*ntf+j]);
        er+=a*Real(even[at].x);ei+=a*Real(even[at].y);
        orr+=b*Real(odd[at].x);oi+=b*Real(odd[at].y);
    }
    for(int side=0;side<2;++side) {
        const int l=it*npc+ip+side*half;
        const Real r=side?er-orr:er+orr,v=side?ei-oi:ei+oi;
        double2 phase;
        if(phase_cache)phase=phase_cache[size_t(shift_ids[index])*ntc*npc+l];
        else {
            double dot=0;for(int d=0;d<3;++d)dot+=dirs[3*l+d]*(fc[3*node+d]-cc[3*parent+d]);
            double s,c;sincos(-kr*dot,&s,&c);const double amp=ki==0?1.:exp(ki*dot);
            phase=make_double2(amp*c,amp*s);
        }
        const size_t dst=size_t(parent)*ntc*npc+l;
        atomicAdd(out.r[comp]+dst,phase.x*r-phase.y*v);atomicAdd(out.i[comp]+dst,phase.y*r+phase.x*v);
    }
}
template <typename Real,typename Temporary>
__global__ void multilevel_down_theta_parity(const Temporary* shifted,const double* te,const double* to,
    Temporary* even,Temporary* odd,int ntf,int ntc,int npc,int count) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x,half=npc/2;
    if(id>=count*3*ntf*half)return;
    const int ip=id%half,it=(id/half)%ntf,batch=id/(ntf*half);
    Real er=0,ei=0,orr=0,oi=0;
    for(int j=0;j<ntc;++j) {
        const size_t index=(size_t(batch)*ntc+j)*npc+ip;
        const Temporary v=shifted[index],w=shifted[index+half];
        const Real a=Real(te[j*ntf+it]),b=Real(to[j*ntf+it]);
        er+=a*(Real(v.x)+Real(w.x));ei+=a*(Real(v.y)+Real(w.y));
        orr+=b*(Real(v.x)-Real(w.x));oi+=b*(Real(v.y)-Real(w.y));
    }
    even[id]=multilevel_temporary<Temporary>(er,ei);odd[id]=multilevel_temporary<Temporary>(orr,oi);
}
template <typename Real,typename Temporary>
__global__ void multilevel_down_phi_parity(MultilevelArrays out,const int* nodes,const double* weights,
    const double* pe,const double* po,const Temporary* even,const Temporary* odd,
    int ntf,int npf,int npc,int first,int count,const int* active) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x,half=npf/2;
    if(id>=count*3*ntf*half)return;
    const int ip=id%half,it=(id/half)%ntf,comp=(id/(ntf*half))%3;
    const int leaf=id/(3*ntf*half),index=active?active[first+leaf]:first+leaf,node=nodes[index];
    Real er=0,ei=0,orr=0,oi=0;
    for(int j=0;j<npc/2;++j) {
        const size_t at=((size_t(leaf)*3+comp)*ntf+it)*(npc/2)+j;
        const Real a=Real(pe[j*npf+ip]),b=Real(po[j*npf+ip]);
        er+=a*Real(even[at].x);ei+=a*Real(even[at].y);
        orr+=b*Real(odd[at].x);oi+=b*Real(odd[at].y);
    }
    for(int side=0;side<2;++side) {
        const int l=it*npf+ip+side*half;const size_t dst=size_t(node)*ntf*npf+l;
        out.r[comp][dst]+=(side?er-orr:er+orr)/weights[l];
        out.i[comp][dst]+=(side?ei-oi:ei+oi)/weights[l];
    }
}
template <typename Real, typename Temporary>
void multilevel_transfer_impl(HelmholtzFMM& f,HelmholtzFMM& c,bool down) {
    if(!f.multilevel_link)f.multilevel_link=new MultilevelLink(f,c);
    auto& a=*static_cast<MultilevelLink*>(f.multilevel_link);
    // Reuse the FP64-sized buffers: no allocation or precision-dependent
    // workspace lifetime is added when GMRES switches to a strict action.
    auto even = reinterpret_cast<Temporary*>(a.even);
    auto odd = reinterpret_cast<Temporary*>(a.odd);
    auto shifted = reinterpret_cast<Temporary*>(a.shifted);
    if(a.coarse!=&c)throw std::runtime_error("multilevel FMM coarse engine changed");
    if(!down) {
        auto out=multilevel_arrays(c,false);
        for(int v=0;v<3;++v) {
            CUDA_CHECK(cudaMemsetAsync(out.r[v],0,size_t(c.n_nodes)*c.L*sizeof(double)));
            CUDA_CHECK(cudaMemsetAsync(out.i[v],0,size_t(c.n_nodes)*c.L*sizeof(double)));
        }
    }
    const int active_count=down?a.down_count:a.up_count;
    const int* active=down?a.active_down:a.active_up;
    const bool parity=bem_env_flag_enabled("BEM_FMM_MULTILEVEL_PARITY",true) && a.npf%2==0 && a.npc%2==0;
    for(int first=0;first<active_count;first+=a.chunk) {
        const int count=std::min(a.chunk,active_count-first);
        const int temporary=count*3*a.ntf*a.npc;
        if(parity && !down) {
            multilevel_up_phi_parity<Real,Temporary><<<(temporary/2+255)/256,256>>>(multilevel_arrays(f,false),f.d_leaf_idx_cached,
                a.pe,a.po,even,odd,a.ntf,a.npf,a.npc,first,count,active);
            const int total=count*3*c.L/2;
            multilevel_up_theta_parity<Real,Temporary><<<(total+255)/256,256>>>(multilevel_arrays(c,false),f.d_leaf_idx_cached,a.parents,
                f.d_node_centers_cached,c.d_node_centers_cached,c.d_dirs_cached,a.te,a.to,even,odd,
                a.shift_ids,a.up_phase,a.ntf,a.ntc,a.npc,f.k.real(),f.k.imag(),first,count,active);
        } else if(parity) {
            const int coarse_total=count*3*c.L;
            multilevel_down_shift<Temporary><<<(coarse_total+255)/256,256>>>(multilevel_arrays(c,true),f.d_leaf_idx_cached,a.parents,
                f.d_node_centers_cached,c.d_node_centers_cached,c.d_dirs_cached,c.d_weights_cached,
                shifted,a.shift_ids,a.down_phase,a.ntc,a.npc,f.k.real(),f.k.imag(),first,count,active);
            multilevel_down_theta_parity<Real,Temporary><<<(temporary/2+255)/256,256>>>(shifted,
                a.te,a.to,even,odd,a.ntf,a.ntc,a.npc,count);
            const int total=count*3*f.L/2;
            multilevel_down_phi_parity<Real,Temporary><<<(total+255)/256,256>>>(multilevel_arrays(f,true),f.d_leaf_idx_cached,f.d_weights_cached,
                a.pe,a.po,even,odd,a.ntf,a.npf,a.npc,first,count,active);
        } else if(!down) {
            multilevel_up_phi<Real,Temporary><<<(temporary+255)/256,256>>>(multilevel_arrays(f,false),f.d_leaf_idx_cached,
                a.pe,a.po,even,odd,a.ntf,a.npf,a.npc,first,count,active);
            const int total=count*3*c.L;
            multilevel_up_theta<Real,Temporary><<<(total+255)/256,256>>>(multilevel_arrays(c,false),f.d_leaf_idx_cached,a.parents,
                f.d_node_centers_cached,c.d_node_centers_cached,c.d_dirs_cached,a.te,a.to,even,odd,
                a.shift_ids,a.up_phase,a.ntf,a.ntc,a.npc,f.k.real(),f.k.imag(),first,count,active);
        } else {
            const int coarse_total=count*3*c.L;
            multilevel_down_shift<Temporary><<<(coarse_total+255)/256,256>>>(multilevel_arrays(c,true),f.d_leaf_idx_cached,a.parents,
                f.d_node_centers_cached,c.d_node_centers_cached,c.d_dirs_cached,c.d_weights_cached,
                shifted,a.shift_ids,a.down_phase,a.ntc,a.npc,f.k.real(),f.k.imag(),first,count,active);
            multilevel_down_theta<Real,Temporary><<<(temporary+255)/256,256>>>(shifted,
                a.te,a.to,even,odd,a.ntf,a.ntc,a.npc,count);
            const int total=count*3*f.L;
            multilevel_down_phi<Real,Temporary><<<(total+255)/256,256>>>(multilevel_arrays(f,true),f.d_leaf_idx_cached,f.d_weights_cached,
                a.pe,a.po,even,odd,a.ntf,a.npf,a.npc,first,count,active);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
void multilevel_transfer(HelmholtzFMM& f,HelmholtzFMM& c,bool down) {
    if(f.near_field_fp32 && bem_env_flag_enabled("BEM_FMM_MULTILEVEL_INTERPOLATION_FP32")) {
        if (bem_env_flag_enabled("BEM_FMM_MULTILEVEL_COMPACT_TEMP"))
            multilevel_transfer_impl<float,float2>(f,c,down);
        else multilevel_transfer_impl<float,double2>(f,c,down);
    } else multilevel_transfer_impl<double,double2>(f,c,down);
}
void cleanup_multilevel_link(HelmholtzFMM& f) {
    delete static_cast<MultilevelLink*>(f.multilevel_link); f.multilevel_link=nullptr;
}
} // namespace

void HelmholtzFMM::evaluate_multilevel_far_uploaded(HelmholtzFMM* middle,HelmholtzFMM& coarse) {
    evaluate_batch3_far_uploaded();
    HelmholtzFMM& next=middle?*middle:coarse;
    multilevel_transfer(*this,next,false);
    FmmCudaTimeline::mark("up-interpolate", tree.max_level);
    next.evaluate_batch3_far_uploaded(true);
    if(middle) {
        multilevel_transfer(next,coarse,false);
        FmmCudaTimeline::mark("up-interpolate", next.tree.max_level);
        coarse.evaluate_batch3_far_uploaded(true);
        multilevel_transfer(next,coarse,true);
        FmmCudaTimeline::mark("down-interpolate", next.tree.max_level);
    }
    multilevel_transfer(*this,next,true);
    FmmCudaTimeline::mark("down-interpolate", tree.max_level);
}
