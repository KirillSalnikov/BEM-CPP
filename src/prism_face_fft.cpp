#include "prism_face_fft.h"
#include "regular_face_fft.h"
#include "quadrature.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

namespace {
using V=std::array<double,3>;
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V add(V a,V b){return {{a[0]+b[0],a[1]+b[1],a[2]+b[2]}};}
V scale(V a,double s){return {{a[0]*s,a[1]*s,a[2]*s}};}
V cross(V a,V b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
void cuda_ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("prism FFT: ")+cudaGetErrorString(e));}
class FaceTimeline {
    std::vector<std::pair<cudaEvent_t,int>> events;
    bool enabled=false;
public:
    FaceTimeline(){const char* flag=std::getenv("BEM_FACE_FFT_PROFILE");enabled=flag && std::string(flag)!="0";mark(-1);}
    void mark(int stage) {
        if(!enabled)return;
        cudaEvent_t event;cuda_ok(cudaEventCreate(&event));events.emplace_back(event,stage);
        cuda_ok(cudaEventRecord(event));
    }
    void finish(double k,bool mixed) {
        if(!enabled)return;
        cuda_ok(cudaEventSynchronize(events.back().first));
        double totals[4]={};
        for(size_t i=1;i<events.size();++i){float ms=0;cuda_ok(cudaEventElapsedTime(&ms,events[i-1].first,events[i].first));totals[events[i].second]+=ms;}
        std::printf("  [Prism FFT profile k=%.6g mixed=%d] self=%.3fms cross-FMM=%.3fms near-cache=%.3fms gather=%.3fms\n",
            k,int(mixed),totals[0],totals[1],totals[2],totals[3]);
    }
    ~FaceTimeline(){for(auto event:events)cudaEventDestroy(event.first);}
};
struct Layout {
    int nx=0,ny=0,channels=0;
    V u,v;
    std::vector<double> offsets;
    std::vector<std::array<double,2>> fractional;
    void offsets_from_triangles(const std::vector<std::array<double,6>>& triangles) {
        const auto rule=tri_quadrature(7);
        for(const auto& t:triangles)for(int q=0;q<7;++q) {
            const double a=rule.pts[q][0],b=rule.pts[q][1];
            const double x=(1-a-b)*t[0]+a*t[2]+b*t[4],y=(1-a-b)*t[1]+a*t[3]+b*t[5];
            fractional.push_back({{x,y}});
            for(int d=0;d<3;++d)offsets.push_back(x*u[d]+y*v[d]);
        }
        channels=int(fractional.size());
    }
};
struct NearBlock {
    int target_begin,target_count,source_begin,source_count;
    unsigned long long pair_begin;
};
struct TemporaryGpu {
    std::vector<void*> allocations;
    void* allocate(size_t bytes) {void* value=nullptr;cuda_ok(cudaMalloc(&value,std::max(size_t(1),bytes)));allocations.push_back(value);return value;}
    void release(void* value) {allocations.erase(std::find(allocations.begin(),allocations.end(),value));}
    ~TemporaryGpu(){for(void* value:allocations)cudaFree(value);}
};
struct NearCache {
    NearBlock* blocks=nullptr;
    int* source_ids=nullptr;
    double2* coefficients=nullptr;
    unsigned int* radial_ids=nullptr;
    double* squared_radii=nullptr;
    std::shared_ptr<const NearCache> geometry_owner;
    int count=0;
    size_t pairs=0,radials=0,bytes=0;
    ~NearCache(){cudaFree(coefficients);if(!geometry_owner){cudaFree(blocks);cudaFree(source_ids);cudaFree(radial_ids);cudaFree(squared_radii);}}
    bool init(HelmholtzFMM& f,size_t budget,std::shared_ptr<const NearCache> geometry={});
};
struct Face {
    V origin,e1,e2,e3;
    std::vector<int> global,slot,other,cross_global;
    int engine_face=0;
    std::array<double,9> rotation{{1,0,0,0,1,0,0,0,1}};
    std::unique_ptr<HelmholtzFMM> fine,middle,coarse;
    std::vector<cdouble> packed,curl,hessian;
    int *d_global=nullptr,*d_slot=nullptr,*d_other=nullptr,*d_cross_global=nullptr;
    std::shared_ptr<NearCache> near{new NearCache};
    ~Face(){cudaFree(d_global);cudaFree(d_slot);cudaFree(d_other);cudaFree(d_cross_global);}
};
// Map quadrature points under proper D6 rotations. Keeping the original
// canonical ordering permits one cross-face hierarchy per face orbit.
struct PointKeyHash {
    size_t operator()(const std::array<long long,3>& a)const {
        size_t h=0;for(auto v:a)h^=std::hash<long long>{}(v)+0x9e3779b9+(h<<6)+(h>>2);return h;
    }
};
class RotatedPointMap {
    const std::vector<double>& xyz;
    double scale_factor,tolerance2;
    std::unordered_map<std::array<long long,3>,int,PointKeyHash> index;
    std::array<long long,3> key(const V& x)const {
        return {{std::llround(x[0]*scale_factor),std::llround(x[1]*scale_factor),std::llround(x[2]*scale_factor)}};
    }
public:
    explicit RotatedPointMap(const std::vector<double>& points):xyz(points) {
        double extent=1.;for(double x:xyz)extent=std::max(extent,std::abs(x));
        scale_factor=1e8/extent;tolerance2=std::pow(1e-10*extent,2);index.reserve(xyz.size()/3);
        for(size_t i=0;i<xyz.size()/3;++i)
            if(!index.emplace(key({{xyz[3*i],xyz[3*i+1],xyz[3*i+2]}}),int(i)).second)
                throw std::runtime_error("ambiguous prism FFT rotation point map");
    }
    int image(int i,const std::array<double,9>& r)const {
        V y{{0,0,0}};for(int d=0;d<3;++d)for(int k=0;k<3;++k)y[d]+=r[3*d+k]*xyz[3*i+k];
        const auto bucket=key(y);
        auto match=[&](const std::array<long long,3>& b) {
            const auto found=index.find(b);if(found==index.end())return -1;
            double error=0;for(int d=0;d<3;++d)error+=std::pow(xyz[3*found->second+d]-y[d],2);
            return error<tolerance2?found->second:-1;
        };
        int target=match(bucket);if(target>=0)return target;
        for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z) {
            target=match({{bucket[0]+x,bucket[1]+y,bucket[2]+z}});if(target>=0)return target;
        }
        throw std::runtime_error("prism FFT quadrature is not invariant under D6 rotations");
    }
};
struct Inputs {const double* re[3];const double* im[3];};
struct Outputs {double *cr,*ci,*hr,*hi;};
struct Frame {double axes[9];};
__device__ void radial_coefficients(double r2,double kr,double ki,double2& alpha,double2& beta) {
    alpha=make_double2(0,0);beta=make_double2(0,0);
    if(r2<1e-24)return;
    const double inv=rsqrt(r2),radius=r2*inv,inv2=inv*inv;
    double sn,cs;sincos(kr*radius,&sn,&cs);
    const double amplitude=0.07957747154594767*inv*(ki==0?1.:exp(-ki*radius));
    const double gr=amplitude*cs,gi=amplitude*sn;
    const double ar=(-ki-inv)*inv,ai=kr*inv;
    alpha=make_double2(gr*ar-gi*ai,gr*ai+gi*ar);
    const double br=3*inv2+3*ki*inv-kr*kr+ki*ki,bi=-3*kr*inv-2*kr*ki;
    beta=make_double2((gr*br-gi*bi)*inv2,(gr*bi+gi*br)*inv2);
}
// Only two complex radial coefficients are needed. With r=t-s,
// alpha=G'/|r|, beta=(G''-G'/|r|)/|r|^2, the Hessian contraction is
// beta * (r*(r dot J)-|r|^2*J) - 2*alpha*J. All cached values are FP64.
__global__ void prepare_near_cache(const NearBlock* blocks,const int* target_ids,const int* source_ids,
    const double* target,const double* source,double kr,double ki,double2* coefficients) {
    const NearBlock b=blocks[blockIdx.x];
    const size_t count=size_t(b.target_count)*b.source_count;
    for(size_t j=size_t(blockIdx.y)*blockDim.x+threadIdx.x;j<count;j+=size_t(gridDim.y)*blockDim.x) {
        const int ti=target_ids[b.target_begin+j/b.source_count],si=source_ids[b.source_begin+j%b.source_count];
        const double x=target[3*ti]-source[3*si],y=target[3*ti+1]-source[3*si+1],z=target[3*ti+2]-source[3*si+2];
        double2 alpha,beta;radial_coefficients(x*x+y*y+z*z,kr,ki,alpha,beta);
        const size_t index=b.pair_begin+j;
        coefficients[2*index]=alpha;coefficients[2*index+1]=beta;
    }
}
__global__ void prepare_near_keys(const NearBlock* blocks,const int* target_ids,const int* source_ids,
    const double* target,const double* source,unsigned long long* keys,unsigned int* order) {
    const NearBlock b=blocks[blockIdx.x];const size_t count=size_t(b.target_count)*b.source_count;
    for(size_t j=size_t(blockIdx.y)*blockDim.x+threadIdx.x;j<count;j+=size_t(gridDim.y)*blockDim.x) {
        const int ti=target_ids[b.target_begin+j/b.source_count],si=source_ids[b.source_begin+j%b.source_count];
        const double x=target[3*ti]-source[3*si],y=target[3*ti+1]-source[3*si+1],z=target[3*ti+2]-source[3*si+2];
        const size_t index=b.pair_begin+j;
        keys[index]=__double_as_longlong(x*x+y*y+z*z);order[index]=unsigned(index);
    }
}
__global__ void near_key_flags(const unsigned long long* keys,unsigned int* groups,size_t n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)groups[i]=(i==0 || keys[i]!=keys[i-1])?1:0;
}
__global__ void build_near_table(const unsigned long long* keys,const unsigned int* order,
    const unsigned int* groups,unsigned int* ids,double* radii,double2* coefficients,size_t n,double kr,double ki) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    const unsigned int radial=groups[i]-1;ids[order[i]]=radial;
    if(i && keys[i]==keys[i-1])return;
    const double r2=__longlong_as_double(keys[i]);double2 alpha,beta;
    radial_coefficients(r2,kr,ki,alpha,beta);radii[radial]=r2;
    coefficients[2*radial]=alpha;coefficients[2*radial+1]=beta;
}
__global__ void prepare_shared_near_cache(const double* radii,double2* coefficients,size_t n,double kr,double ki) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    double2 alpha,beta;radial_coefficients(radii[i],kr,ki,alpha,beta);
    coefficients[2*i]=alpha;coefficients[2*i+1]=beta;
}
template<class Real>
__global__ void apply_near_cache(const NearBlock* blocks,const int* target_ids,const int* source_ids,
    const double* target,const double* source,const double2* coefficients,const unsigned int* ids,Inputs q,Outputs out) {
    const NearBlock b=blocks[blockIdx.x];const int lane=threadIdx.x%32;
    const int warp=blockIdx.y*(blockDim.x/32)+threadIdx.x/32;
    const int stride=gridDim.y*(blockDim.x/32);
    for(int ti0=warp;ti0<b.target_count;ti0+=stride) {
        const int ti=target_ids[b.target_begin+ti0];
        Real cr[3]={},ci[3]={},hr[3]={},hi[3]={};
        for(int j=lane;j<b.source_count;j+=32) {
            const int si=source_ids[b.source_begin+j];
            const size_t pair=b.pair_begin+size_t(ti0)*b.source_count+j;
            const size_t index=ids?ids[pair]:pair;
            const double2 aa=coefficients[2*index],bb=coefficients[2*index+1];
            const Real ar=Real(aa.x),ai=Real(aa.y),br=Real(bb.x),bi=Real(bb.y);
            Real r[3],qr[3],qi[3],gr[3],gi[3],dotr=0,doti=0,r2=0;
            for(int d=0;d<3;++d) {
                r[d]=Real(target[3*ti+d]-source[3*si+d]);qr[d]=Real(q.re[d][si]);qi[d]=Real(q.im[d][si]);
                dotr+=r[d]*qr[d];doti+=r[d]*qi[d];r2+=r[d]*r[d];
                gr[d]=ar*qr[d]-ai*qi[d];gi[d]=ar*qi[d]+ai*qr[d];
            }
            cr[0]+=r[1]*gr[0]-r[0]*gr[1];ci[0]+=r[1]*gi[0]-r[0]*gi[1];
            cr[1]+=r[2]*gr[0]-r[0]*gr[2];ci[1]+=r[2]*gi[0]-r[0]*gi[2];
            cr[2]+=r[2]*gr[1]-r[1]*gr[2];ci[2]+=r[2]*gi[1]-r[1]*gi[2];
            for(int d=0;d<3;++d) {
                const Real vr=r[d]*dotr-r2*qr[d],vi=r[d]*doti-r2*qi[d];
                hr[d]+=br*vr-bi*vi-2*gr[d];hi[d]+=br*vi+bi*vr-2*gi[d];
            }
        }
        for(int offset=16;offset;offset/=2)for(int d=0;d<3;++d) {
            cr[d]+=__shfl_down_sync(0xffffffff,cr[d],offset);ci[d]+=__shfl_down_sync(0xffffffff,ci[d],offset);
            hr[d]+=__shfl_down_sync(0xffffffff,hr[d],offset);hi[d]+=__shfl_down_sync(0xffffffff,hi[d],offset);
        }
        if(lane==0)for(int d=0;d<3;++d) {
            out.cr[3*ti+d]+=double(cr[d]);out.ci[3*ti+d]+=double(ci[d]);
            out.hr[3*ti+d]+=double(hr[d]);out.hi[3*ti+d]+=double(hi[d]);
        }
    }
}

bool NearCache::init(HelmholtzFMM& f,size_t budget,std::shared_ptr<const NearCache> geometry) {
    double reserve_gib=4.;
    if(const char* value=std::getenv("BEM_FACE_FFT_NEAR_RESERVE_GIB")) {
        reserve_gib=std::stod(value);
        if(!std::isfinite(reserve_gib) || reserve_gib<2. || reserve_gib>20.)
            throw std::runtime_error("BEM_FACE_FFT_NEAR_RESERVE_GIB must be between 2 and 20");
    }
    const size_t reserve=size_t(reserve_gib*std::pow(2.,30));
    size_t free=0,capacity=0;cuda_ok(cudaMemGetInfo(&free,&capacity));
    if(geometry && geometry->count && geometry->squared_radii) {
        const size_t needed=geometry->radials*2*sizeof(double2);
        if(needed>budget || needed+reserve>free) {
            std::printf("  [Prism FFT near cache] shared geometry fallback: %.3f GiB coefficients, %.3f GiB free\n",needed/std::pow(2.,30),free/std::pow(2.,30));
            return false;
        }
        geometry_owner=geometry;blocks=geometry->blocks;source_ids=geometry->source_ids;
        radial_ids=geometry->radial_ids;squared_radii=geometry->squared_radii;
        pairs=geometry->pairs;radials=geometry->radials;
        cuda_ok(cudaMalloc(&coefficients,needed));
        prepare_shared_near_cache<<<(radials+255)/256,256>>>(squared_radii,coefficients,radials,f.k.real(),f.k.imag());
        cuda_ok(cudaGetLastError());cuda_ok(cudaStreamSynchronize(0));
        count=geometry->count;bytes=needed;
        std::printf("  [Prism FFT near cache] shared geometry: %zu pairs -> %zu radii, %.3f GiB new FP64 coefficients\n",pairs,radials,bytes/std::pow(2.,30));
        return true;
    }
    std::vector<NearBlock> host_blocks;std::vector<int> host_sources;
    std::vector<int> ordinal(f.n_nodes,-1);
    for(size_t i=0;i<f.leaf_info.size();++i)ordinal[f.leaf_info[i].node_idx]=int(i);
    size_t total=0;
    for(size_t leaf=0;leaf<f.leaf_info.size();++leaf) {
        const int nt=f.h_tgt_id_offsets[leaf+1]-f.h_tgt_id_offsets[leaf];if(!nt)continue;
        NearBlock b={f.h_tgt_id_offsets[leaf],nt,int(host_sources.size()),0,total};
        const auto append=[&](int i){host_sources.insert(host_sources.end(),
            f.h_src_ids_flat.begin()+f.h_src_id_offsets[i],f.h_src_ids_flat.begin()+f.h_src_id_offsets[i+1]);};
        append(int(leaf));const auto& node=f.tree.nodes[f.leaf_info[leaf].node_idx];
        for(int j=0;j<node.near_count;++j) {
            const int i=ordinal[f.tree.near_list[node.near_start+j]];
            if(i<0)throw std::runtime_error("near cache requires uniform leaf neighbors");
            append(i);
        }
        b.source_count=int(host_sources.size())-b.source_begin;
        if(!b.source_count)continue;
        total+=size_t(nt)*b.source_count;host_blocks.push_back(b);
    }
    const size_t metadata=host_sources.size()*sizeof(int)+host_blocks.size()*sizeof(NearBlock);
    size_t needed=2*total*sizeof(double2)+metadata;
    const char* dedup_flag=std::getenv("BEM_FACE_FFT_NEAR_CACHE_DEDUP");
    const bool deduplicate=!dedup_flag || std::string(dedup_flag)!="0";
    cub::DoubleBuffer<unsigned long long> keys(nullptr,nullptr);
    cub::DoubleBuffer<unsigned int> order(nullptr,nullptr);
    size_t sort_workspace=0,scan_workspace=0;
    if(deduplicate && total && total<=size_t(std::numeric_limits<int>::max())) {
        cuda_ok(cub::DeviceRadixSort::SortPairs(nullptr,sort_workspace,keys,order,int(total)));
        cuda_ok(cub::DeviceScan::InclusiveSum(nullptr,scan_workspace,static_cast<unsigned int*>(nullptr),static_cast<unsigned int*>(nullptr),int(total)));
    }
    const size_t workspace=std::max(sort_workspace,scan_workspace);
    const size_t peak=deduplicate?68*total+metadata+workspace:needed;
    if(!total || total>size_t(std::numeric_limits<int>::max()) ||
       (deduplicate && total*sizeof(unsigned int)+metadata>budget) ||
       (!deduplicate && needed>budget) || peak+2ULL*1024*1024*1024>free) {
        std::printf("  [Prism FFT near cache] direct fallback: %.3f GiB required, %.3f GiB budget\n",needed/std::pow(2.,30),budget/std::pow(2.,30));
        return false;
    }
    cuda_ok(cudaMalloc(&blocks,host_blocks.size()*sizeof(NearBlock)));
    cuda_ok(cudaMalloc(&source_ids,host_sources.size()*sizeof(int)));
    cuda_ok(cudaMemcpy(blocks,host_blocks.data(),host_blocks.size()*sizeof(NearBlock),cudaMemcpyHostToDevice));
    cuda_ok(cudaMemcpy(source_ids,host_sources.data(),host_sources.size()*sizeof(int),cudaMemcpyHostToDevice));
    radials=total;
    TemporaryGpu temp;
    if(deduplicate) {
        keys=cub::DoubleBuffer<unsigned long long>(static_cast<unsigned long long*>(temp.allocate(total*8)),
                                                   static_cast<unsigned long long*>(temp.allocate(total*8)));
        order=cub::DoubleBuffer<unsigned int>(static_cast<unsigned int*>(temp.allocate(total*4)),
                                             static_cast<unsigned int*>(temp.allocate(total*4)));
        auto* groups=static_cast<unsigned int*>(temp.allocate(total*4));void* scratch=temp.allocate(workspace);
        prepare_near_keys<<<dim3(int(host_blocks.size()),16),256>>>(blocks,f.d_tgt_ids_cached,source_ids,
            f.d_tgt_pts,f.d_src_pts,keys.Current(),order.Current());cuda_ok(cudaGetLastError());
        cuda_ok(cub::DeviceRadixSort::SortPairs(scratch,sort_workspace,keys,order,int(total)));
        near_key_flags<<<(total+255)/256,256>>>(keys.Current(),groups,total);cuda_ok(cudaGetLastError());
        cuda_ok(cub::DeviceScan::InclusiveSum(scratch,scan_workspace,groups,groups,int(total)));
        unsigned int unique=0;cuda_ok(cudaMemcpy(&unique,groups+total-1,sizeof(unique),cudaMemcpyDeviceToHost));
        radials=unique;needed=total*sizeof(unsigned int)+radials*(2*sizeof(double2)+sizeof(double))+metadata;
        if(needed>budget || needed+reserve>free) {
            cudaFree(blocks);cudaFree(source_ids);blocks=nullptr;source_ids=nullptr;
            std::printf("  [Prism FFT near cache] deduplicated fallback: %zu pairs -> %zu radii, %.3f GiB, budget %.3f GiB\n",
                total,radials,needed/std::pow(2.,30),budget/std::pow(2.,30));
            return false;
        }
        cuda_ok(cudaMalloc(&coefficients,radials*2*sizeof(double2)));
        cuda_ok(cudaMalloc(&squared_radii,radials*sizeof(double)));
        radial_ids=order.Alternate();temp.release(radial_ids);
        build_near_table<<<(total+255)/256,256>>>(keys.Current(),order.Current(),groups,radial_ids,
            squared_radii,coefficients,total,f.k.real(),f.k.imag());cuda_ok(cudaGetLastError());
    }else{
        cuda_ok(cudaMalloc(&coefficients,2*total*sizeof(double2)));
        prepare_near_cache<<<dim3(int(host_blocks.size()),16),256>>>(blocks,f.d_tgt_ids_cached,source_ids,
            f.d_tgt_pts,f.d_src_pts,f.k.real(),f.k.imag(),coefficients);
    }
    cuda_ok(cudaGetLastError());cuda_ok(cudaStreamSynchronize(0));
    count=int(host_blocks.size());pairs=total;bytes=needed;
    std::printf("  [Prism FFT near cache] %zu pairs -> %zu radii, %.3f GiB, exact FP64 coefficients\n",pairs,radials,bytes/std::pow(2.,30));
    return true;
}
// Reverse the *cached* radial operator, rather than recomputing a potentially
// different near approximation. Each warp sums targets for one source; the
// atomic accumulation combines the original near blocks sharing that source.
template<class Real>
__global__ void apply_near_cache_transpose(const NearBlock* blocks,const int* target_ids,const int* source_ids,
    const double* target,const double* source,const double2* coefficients,const unsigned int* ids,
    Outputs seed,double* out,int ns) {
    const NearBlock b=blocks[blockIdx.x];const int lane=threadIdx.x%32;
    const int warp=blockIdx.y*(blockDim.x/32)+threadIdx.x/32,stride=gridDim.y*(blockDim.x/32);
    for(int si0=warp;si0<b.source_count;si0+=stride) {
        const int si=source_ids[b.source_begin+si0];Real qr[3]={},qi[3]={};
        for(int j=lane;j<b.target_count;j+=32) {
            const int ti=target_ids[b.target_begin+j];
            const size_t pair=b.pair_begin+size_t(j)*b.source_count+si0,index=ids?ids[pair]:pair;
            const double2 aa=coefficients[2*index],bb=coefficients[2*index+1];
            const Real ar=Real(aa.x),ai=Real(aa.y),br=Real(bb.x),bi=Real(bb.y);
            Real r[3],cr[3],ci[3],hr[3],hi[3],dotr=0,doti=0,r2=0;
            for(int d=0;d<3;++d) {
                r[d]=Real(target[3*ti+d]-source[3*si+d]);
                cr[d]=Real(seed.cr[3*ti+d]);ci[d]=Real(seed.ci[3*ti+d]);
                hr[d]=Real(seed.hr[3*ti+d]);hi[d]=Real(seed.hi[3*ti+d]);
                dotr+=r[d]*hr[d];doti+=r[d]*hi[d];r2+=r[d]*r[d];
            }
            const Real vr[3]={r[1]*cr[0]+r[2]*cr[1],-r[0]*cr[0]+r[2]*cr[2],-r[0]*cr[1]-r[1]*cr[2]};
            const Real vi[3]={r[1]*ci[0]+r[2]*ci[1],-r[0]*ci[0]+r[2]*ci[2],-r[0]*ci[1]-r[1]*ci[2]};
            for(int d=0;d<3;++d) {
                const Real wr=r[d]*dotr-r2*hr[d],wi=r[d]*doti-r2*hi[d];
                qr[d]+=ar*(vr[d]-2*hr[d])-ai*(vi[d]-2*hi[d])+br*wr-bi*wi;
                qi[d]+=ar*(vi[d]-2*hi[d])+ai*(vr[d]-2*hr[d])+br*wi+bi*wr;
            }
        }
        for(int offset=16;offset;offset/=2)for(int d=0;d<3;++d) {
            qr[d]+=__shfl_down_sync(0xffffffff,qr[d],offset);qi[d]+=__shfl_down_sync(0xffffffff,qi[d],offset);
        }
        if(lane==0)for(int d=0;d<3;++d) {
            atomicAdd(out+size_t(2*d)*ns+si,double(qr[d]));
            atomicAdd(out+size_t(2*d+1)*ns+si,double(qi[d]));
        }
    }
}
__global__ void scatter_face(int n,const int* ids,const int* slots,int cells,
                            Frame frame,Inputs q,double2* out) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    const int i=ids[j],slot=slots[j];
    for(int d=0;d<3;++d) {
        double re=0,im=0;
        for(int k=0;k<3;++k){re+=frame.axes[3*d+k]*q.re[k][i];im+=frame.axes[3*d+k]*q.im[k][i];}
        out[size_t(slot/cells*3+d)*cells+slot%cells]=make_double2(re,im);
    }
}
__global__ void gather_face(int n,const int* ids,const int* slots,int cells,
                           Frame frame,const double2* in,Outputs out) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    const int i=ids[j],slot=slots[j];double2 value[6],curl[3];
    for(int d=0;d<6;++d)value[d]=in[size_t(slot/cells*6+d)*cells+slot%cells];
    for(int d=0;d<3;++d) {
        const double a=frame.axes[d],b=frame.axes[3+d],c=frame.axes[6+d];
        curl[d]=make_double2(-a*value[2].x+b*value[1].x-c*value[0].x,
                            -a*value[2].y+b*value[1].y-c*value[0].y);
        out.hr[3*i+d]+=a*value[3].x+b*value[4].x+c*value[5].x;
        out.hi[3*i+d]+=a*value[3].y+b*value[4].y+c*value[5].y;
    }
    out.cr[3*i]-=curl[2].x;out.ci[3*i]-=curl[2].y;
    out.cr[3*i+1]+=curl[1].x;out.ci[3*i+1]+=curl[1].y;
    out.cr[3*i+2]-=curl[0].x;out.ci[3*i+2]-=curl[0].y;
}
__global__ void gather_cross_charges(int n,const int* ids,Frame rotation,Inputs q,double* out) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    const int i=ids[j];
    for(int d=0;d<3;++d) {
        double re=0,im=0;for(int k=0;k<3;++k){re+=rotation.axes[3*k+d]*q.re[k][i];im+=rotation.axes[3*k+d]*q.im[k][i];}
        out[size_t(2*d)*n+j]=re;out[size_t(2*d+1)*n+j]=im;
    }
}
__global__ void add_cross(int n,const int* ids,Frame rotation,Outputs from,Outputs out) {
    const int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;
    const int i=ids[j];
    // FMM stores antisymmetric derivatives as (xy,xz,yz), not Cartesian
    // curl components: curl=(-yz,xz,-xy). Rotate through that signed
    // permutation. All geometric maps have determinant +1.
    for(int d=0;d<3;++d)for(int k=0;k<3;++k) {
        const double a=rotation.axes[3*d+k];
        const double curl_a=(d==1?1.:-1.)*(k==1?1.:-1.)*rotation.axes[3*(2-d)+(2-k)];
        out.cr[3*i+d]+=curl_a*from.cr[3*j+k];out.ci[3*i+d]+=curl_a*from.ci[3*j+k];
        out.hr[3*i+d]+=a*from.hr[3*j+k];out.hi[3*i+d]+=a*from.hi[3*j+k];
    }
}
}

struct PrismFaceFFT::Impl {
    int count=0;
    int refinement=0;
    double aspect=1.;
    std::vector<double> geometry;
    Layout side,cap;
    RegularFaceFFT side_fft,cap_fft;
    Face faces[8];
    std::vector<cdouble> fft_input,fft_output;
    std::vector<cdouble> charges[3],curl,hessian;
    std::vector<double> real,imag;
    bool target_faces=true;
    bool reuse_cross=false;
    double *d_packed=nullptr,*d_host_charges=nullptr;
    HelmholtzFMM view; // Only a non-owning view of the four buffers below.
    ~Impl(){cudaFree(view.d_grad_re);cudaFree(view.d_grad_im);cudaFree(view.d_hess_re);cudaFree(view.d_hess_im);
        cudaFree(d_packed);cudaFree(d_host_charges);}
};
PrismFaceFFT::PrismFaceFFT():p(new Impl){}
PrismFaceFFT::~PrismFaceFFT()=default;
size_t PrismFaceFFT::fft_storage_bytes()const{return p->side_fft.storage_bytes()+p->cap_fft.storage_bytes();}
size_t PrismFaceFFT::cached_near_pairs()const{size_t n=0;for(const auto& f:p->faces)if(f.fine)n+=f.near->pairs;return n;}
size_t PrismFaceFFT::cached_near_radials()const{size_t n=0;for(const auto& f:p->faces)if(f.fine)n+=f.near->radials;return n;}
const HelmholtzFMM& PrismFaceFFT::result_view()const{return p->view;}
PrismFaceFFT::HostView PrismFaceFFT::host_view()const {
    if(!p->faces[0].fine)throw std::runtime_error("prism FFT host view needs prepared geometry");
    HostView view;view.k=p->faces[0].fine->k;view.target_faces=p->target_faces;
    view.side={p->side.nx,p->side.ny,p->side.u,p->side.v,&p->side.offsets};
    view.cap={p->cap.nx,p->cap.ny,p->cap.u,p->cap.v,&p->cap.offsets};
    for(int f=0;f<8;++f){
        const auto& face=p->faces[f];const auto& engine=p->faces[face.engine_face];auto& out=view.faces[f];
        for(int d=0;d<3;++d){out.frame[d]=face.e1[d];out.frame[3+d]=face.e2[d];out.frame[6+d]=face.e3[d];}
        out.rotation=face.rotation;out.global=&face.global;out.slot=&face.slot;out.other=&face.other;out.cross_global=&face.cross_global;
        out.engine=face.engine_face;out.fine=engine.fine.get();out.middle=engine.middle.get();out.coarse=engine.coarse.get();
    }
    return view;
}

void PrismFaceFFT::init(const std::vector<double>& xyz,int ref,cdouble k,int digits,const PrismFaceFFT* shared_geometry,double aspect) {
    if(ref<1||ref>7||xyz.empty()||xyz.size()%3||!std::isfinite(aspect)||aspect<=0)
        throw std::runtime_error("invalid regular prism FFT input");
    if(shared_geometry==this)throw std::runtime_error("FFT geometry cannot refer to itself during init");
    p.reset(new Impl);p->count=int(xyz.size()/3);
    p->geometry=xyz;p->refinement=ref;p->aspect=aspect;
    if(const char* flag=std::getenv("BEM_FACE_FFT_TARGET_FACES"))p->target_faces=std::string(flag)!="0";
    if(const char* flag=std::getenv("BEM_FACE_FFT_REUSE_CROSS"))p->reuse_cross=std::string(flag)!="0";
    if(shared_geometry && (shared_geometry->p->geometry!=xyz || shared_geometry->p->refinement!=ref ||
                          shared_geometry->p->aspect!=aspect || shared_geometry->p->target_faces!=p->target_faces ||
                          shared_geometry->p->reuse_cross!=p->reuse_cross))
        throw std::runtime_error("FFT near cache sharing requires identical ordered geometry");
    const int seg=1<<ref;int nz=std::max(seg,int(std::ceil(std::sqrt(3.)*aspect*seg)));if(nz%2)++nz;
    const int expected=7*12*seg*(seg+nz);
    if(p->count!=expected)throw std::runtime_error("prism FFT needs the unrefined-edge D6h mesh and quadrature-7");
    const double pi=std::acos(-1.),width=std::cbrt(8*pi/(3*std::sqrt(3.)*aspect));
    const double height=aspect*width,radius=width/std::sqrt(3.);
    const double h=radius/seg,zstep=height/nz;
    p->side.nx=seg;p->side.ny=nz;p->side.u={{h,0,0}};p->side.v={{0,zstep,0}};
    p->side.offsets_from_triangles({{{0,0,1,0,1,1}},{{0,0,1,1,0,1}},{{0,0,1,0,0,1}},{{1,0,1,1,0,1}}});
    p->cap.nx=2*seg;p->cap.ny=2*seg;p->cap.u={{h,0,0}};p->cap.v={{.5*h,std::sqrt(3.)*.5*h,0}};
    p->cap.offsets_from_triangles({{{0,0,1,0,0,1}},{{1,0,1,1,0,1}}});
    V vertices[6];for(int f=0;f<6;++f){double a=(2*f+1)*pi/6;vertices[f]={{radius*std::cos(a),radius*std::sin(a),0}};}
    for(int f=0;f<8;++f) {
        Face& face=p->faces[f];
        if(f<6) {
            face.origin=vertices[f];face.origin[2]=-.5*height;
            face.e1=scale(add(vertices[(f+1)%6],scale(vertices[f],-1)),1/radius);
            face.e2={{0,0,1}};face.e3=cross(face.e1,face.e2);
        }else{
            face.e1=scale(vertices[0],1/radius);face.e2={{-face.e1[1],face.e1[0],0}};face.e3={{0,0,1}};
            face.origin=scale(add(vertices[0],vertices[1]),-1);face.origin[2]=(f==6?.5:-.5)*height;
        }
    }
    std::vector<int> point_face(p->count,-1);
    for(int i=0;i<p->count;++i) {
        const V position={{xyz[3*i],xyz[3*i+1],xyz[3*i+2]}};
        int f=-1;
        if(std::abs(position[2]-.5*height)<1e-9)f=6;
        else if(std::abs(position[2]+.5*height)<1e-9)f=7;
        else for(int j=0;j<6;++j)if(std::abs(dot(add(position,scale(p->faces[j].origin,-1)),p->faces[j].e3))<1e-9){f=j;break;}
        if(f<0)throw std::runtime_error("point is not on the expected prism faces");
        Face& face=p->faces[f];const Layout& l=f<6?p->side:p->cap;
        const V delta=add(position,scale(face.origin,-1));
        const double x=dot(delta,face.e1),y=dot(delta,face.e2);
        const double b=y/l.v[1],a=(x-b*l.v[0])/l.u[0];
        int slot=-1;
        for(int ch=0;ch<l.channels;++ch) {
            const int ix=int(std::llround(a-l.fractional[ch][0])),iy=int(std::llround(b-l.fractional[ch][1]));
            if(ix>=0&&ix<l.nx&&iy>=0&&iy<l.ny&&std::abs(a-l.fractional[ch][0]-ix)<1e-8&&std::abs(b-l.fractional[ch][1]-iy)<1e-8) {
                slot=ch*l.nx*l.ny+ix*l.ny+iy;break;
            }
        }
        if(slot<0)throw std::runtime_error("quadrature point does not match any prism lattice channel");
        face.global.push_back(i);face.slot.push_back(slot);point_face[i]=f;
    }
    for(int f=0;f<8;++f) {
        auto slots=p->faces[f].slot;std::sort(slots.begin(),slots.end());
        if(std::adjacent_find(slots.begin(),slots.end())!=slots.end())throw std::runtime_error("duplicate FFT quadrature slot");
    }
    p->side_fft.init(p->side.nx,p->side.ny,p->side.u.data(),p->side.v.data(),p->side.offsets,p->side.offsets,k);
    p->cap_fft.init(p->cap.nx,p->cap.ny,p->cap.u.data(),p->cap.v.data(),p->cap.offsets,p->cap.offsets,k);
    const int depth=std::min(6,std::max(3,ref));
    auto leaf=[&](int d){const long long boxes=1LL<<(3*d);return std::max(1,int((p->count+boxes-1)/boxes));};
    std::unique_ptr<RotatedPointMap> rotation_map;
    if(p->reuse_cross)rotation_map.reset(new RotatedPointMap(xyz));
    for(int f=0;f<8;++f) {
        Face& face=p->faces[f];
        face.engine_face=p->reuse_cross?(f<6?0:6):f;
        if(face.engine_face!=f) {
            const Face& canonical=p->faces[face.engine_face];
            if(f<6) {
                const double c=std::cos(f*pi/3),s=std::sin(f*pi/3);
                face.rotation={{c,-s,0,s,c,0,0,0,1}};
            }else face.rotation={{1,0,0,0,-1,0,0,0,-1}};
            for(int i:canonical.global)face.cross_global.push_back(rotation_map->image(i,face.rotation));
            for(int i:canonical.other)face.other.push_back(rotation_map->image(i,face.rotation));
            auto actual=face.cross_global,expected=face.global;
            std::sort(actual.begin(),actual.end());std::sort(expected.begin(),expected.end());
            if(actual!=expected)throw std::runtime_error("prism FFT rotation does not map the target face bijectively");
            continue;
        }
        face.cross_global=face.global;
        std::vector<double> source,target;
        for(int i=0;i<p->count;++i)if(point_face[i]!=f)face.other.push_back(i);
        const auto& source_ids=p->target_faces?face.other:face.global;
        const auto& target_ids=p->target_faces?face.global:face.other;
        for(int i:source_ids)for(int d=0;d<3;++d)source.push_back(xyz[3*i+d]);
        for(int i:target_ids)for(int d=0;d<3;++d)target.push_back(xyz[3*i+d]);
        face.fine.reset(new HelmholtzFMM);face.middle.reset(new HelmholtzFMM);face.coarse.reset(new HelmholtzFMM);
        face.fine->configure_interaction_band(depth,depth,true);face.fine->configure_order_reference_depth(depth);
        face.middle->configure_interaction_band(depth-1,depth-1,false);face.middle->configure_order_reference_depth(depth);
        face.coarse->configure_interaction_band(1,depth-2,false);face.coarse->configure_order_reference_depth(std::max(1,depth-3));
        face.fine->init(target.data(),int(target_ids.size()),source.data(),int(source_ids.size()),k,digits,leaf(depth),3,false,false,true);
        face.middle->init(target.data(),int(target_ids.size()),source.data(),int(source_ids.size()),k,digits,leaf(depth-1),3,false,false,true);
        face.coarse->init(target.data(),int(target_ids.size()),source.data(),int(source_ids.size()),k,digits,leaf(depth-2),3,false,false,true);
        if(face.fine->tree.max_level!=depth||face.middle->tree.max_level!=depth-1||face.coarse->tree.max_level!=depth-2)
            throw std::runtime_error("unexpected prism cross-face hierarchy depth");
    }
    rotation_map.reset();
    for(auto& face:p->faces) {
        auto upload=[](int*& to,const std::vector<int>& from){cuda_ok(cudaMalloc(&to,from.size()*sizeof(int)));
            cuda_ok(cudaMemcpy(to,from.data(),from.size()*sizeof(int),cudaMemcpyHostToDevice));};
        upload(face.d_global,face.global);upload(face.d_slot,face.slot);upload(face.d_other,face.other);upload(face.d_cross_global,face.cross_global);
    }
    for(auto& q:p->charges)q.resize(p->count);
    p->curl.resize(size_t(p->count)*3);p->hessian.resize(size_t(p->count)*3);p->real.resize(size_t(p->count)*3);p->imag.resize(size_t(p->count)*3);
    cuda_ok(cudaMalloc(&p->view.d_grad_re,p->count*3ULL*sizeof(double)));cuda_ok(cudaMalloc(&p->view.d_grad_im,p->count*3ULL*sizeof(double)));
    cuda_ok(cudaMalloc(&p->view.d_hess_re,p->count*3ULL*sizeof(double)));cuda_ok(cudaMalloc(&p->view.d_hess_im,p->count*3ULL*sizeof(double)));
    cuda_ok(cudaMalloc(&p->d_packed,p->count*6ULL*sizeof(double)));
    cuda_ok(cudaMalloc(&p->d_host_charges,p->count*6ULL*sizeof(double)));
    const char* near_flag=std::getenv("BEM_FACE_FFT_NEAR_CACHE");
    if(near_flag && std::string(near_flag)!="0") {
        double gib=2.;if(const char* size=std::getenv("BEM_FACE_FFT_NEAR_CACHE_GIB"))gib=std::atof(size);
        if(!std::isfinite(gib)||gib<0||gib>20)throw std::runtime_error("near cache budget must be 0..20 GiB per medium");
        size_t budget=size_t(gib*std::pow(2.,30));
        for(int f=0;f<8;++f) {
            auto& face=p->faces[f];std::shared_ptr<const NearCache> geometry;
            if(face.engine_face!=f)continue;
            if(shared_geometry) {
                const auto& other=shared_geometry->p->faces[f];
                if(face.fine->h_src_ids_flat!=other.fine->h_src_ids_flat ||
                   face.fine->h_tgt_ids_flat!=other.fine->h_tgt_ids_flat ||
                   face.fine->h_src_id_offsets!=other.fine->h_src_id_offsets ||
                   face.fine->h_tgt_id_offsets!=other.fine->h_tgt_id_offsets ||
                   face.fine->tree.near_list!=other.fine->tree.near_list)
                    throw std::runtime_error("FFT shared near cache hierarchy mismatch");
                geometry=other.near;
            }
            if(face.near->init(*face.fine,budget,geometry))budget-=face.near->bytes;
        }
    }
    std::printf("  [Prism FFT] device-resident self FFT + cross-face FMM: %d points, target-faces=%d, cross-hierarchies=%d, FFT %.3f GiB\n",
            p->count,int(p->target_faces),p->reuse_cross?2:8,fft_storage_bytes()/std::pow(2.,30));
}

void PrismFaceFFT::evaluate(const cdouble* qx,const cdouble* qy,const cdouble* qz,cdouble* curl,cdouble* hessian,bool mixed) {
    const cdouble* q[3]={qx,qy,qz};const double* re[3];const double* im[3];
    for(int d=0;d<3;++d) {
        #pragma omp parallel for schedule(static)
        for(int i=0;i<p->count;++i){p->real[i]=q[d][i].real();p->imag[i]=q[d][i].imag();}
        re[d]=p->d_host_charges+size_t(2*d)*p->count;im[d]=p->d_host_charges+size_t(2*d+1)*p->count;
        cuda_ok(cudaMemcpy(const_cast<double*>(re[d]),p->real.data(),p->count*sizeof(double),cudaMemcpyHostToDevice));
        cuda_ok(cudaMemcpy(const_cast<double*>(im[d]),p->imag.data(),p->count*sizeof(double),cudaMemcpyHostToDevice));
    }
    evaluate_device(re,im,mixed);
    auto download=[&](const double* re,const double* im,cdouble* out) {
        cuda_ok(cudaMemcpy(p->real.data(),re,p->count*3ULL*sizeof(double),cudaMemcpyDeviceToHost));
        cuda_ok(cudaMemcpy(p->imag.data(),im,p->count*3ULL*sizeof(double),cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for(int i=0;i<3*p->count;++i)out[i]=cdouble(p->real[i],p->imag[i]);
    };
    download(p->view.d_grad_re,p->view.d_grad_im,curl);download(p->view.d_hess_re,p->view.d_hess_im,hessian);
}

void PrismFaceFFT::evaluate_device(const double* const real[3],const double* const imag[3],bool mixed) {
    FaceTimeline timeline;
    Inputs q;for(int d=0;d<3;++d){q.re[d]=real[d];q.im[d]=imag[d];}
    Outputs out={p->view.d_grad_re,p->view.d_grad_im,p->view.d_hess_re,p->view.d_hess_im};
    for(double* buffer:{out.cr,out.ci,out.hr,out.hi})cuda_ok(cudaMemsetAsync(buffer,0,p->count*3ULL*sizeof(double)));
    for(int f=0;f<8;++f) {
        Face& face=p->faces[f];const Layout& l=f<6?p->side:p->cap;
        Face& engine=p->faces[face.engine_face];Frame rotation;std::copy(face.rotation.begin(),face.rotation.end(),rotation.axes);
        RegularFaceFFT& fft=f<6?p->side_fft:p->cap_fft;const int cells=l.nx*l.ny,n=int(face.global.size());
        Frame frame;const V axes[3]={face.e1,face.e2,face.e3};
        for(int d=0;d<3;++d)for(int k=0;k<3;++k)frame.axes[3*d+k]=axes[d][k];
        cuda_ok(cudaMemsetAsync(fft.device_input(),0,size_t(cells)*l.channels*3*sizeof(double2)));
        scatter_face<<<(n+255)/256,256>>>(n,face.d_global,face.d_slot,cells,frame,q,
            reinterpret_cast<double2*>(fft.device_input()));cuda_ok(cudaGetLastError());
        fft.evaluate_device();
        gather_face<<<(n+255)/256,256>>>(n,face.d_global,face.d_slot,cells,frame,
            reinterpret_cast<const double2*>(fft.device_output()),out);cuda_ok(cudaGetLastError());
        timeline.mark(0);
        engine.fine->near_field_fp32=engine.middle->near_field_fp32=engine.coarse->near_field_fp32=mixed;
        const int ns=p->target_faces?int(face.other.size()):n;
        const int nt=p->target_faces?n:int(face.other.size());
        const int* source_ids=p->target_faces?face.d_other:face.d_cross_global;
        const int* target_ids=p->target_faces?face.d_cross_global:face.d_other;
        gather_cross_charges<<<(ns+255)/256,256>>>(ns,source_ids,rotation,q,p->d_packed);cuda_ok(cudaGetLastError());
        double* packed=p->d_packed;
        engine.fine->evaluate_vector_actions_batch3_device(packed,packed+ns,packed+2ULL*ns,
            packed+3ULL*ns,packed+4ULL*ns,packed+5ULL*ns,engine.middle.get(),engine.coarse.get(),engine.near->count==0);
        timeline.mark(1);
        Outputs from={engine.fine->d_grad_re,engine.fine->d_grad_im,engine.fine->d_hess_re,engine.fine->d_hess_im};
        if(engine.near->count) {
            Inputs charges;for(int d=0;d<3;++d){charges.re[d]=packed+size_t(2*d)*ns;charges.im[d]=packed+size_t(2*d+1)*ns;}
            if(mixed)apply_near_cache<float><<<dim3(engine.near->count,4),128>>>(
                engine.near->blocks,engine.fine->d_tgt_ids_cached,engine.near->source_ids,engine.fine->d_tgt_pts,
                engine.fine->d_src_pts,engine.near->coefficients,engine.near->radial_ids,charges,from);
            else apply_near_cache<double><<<dim3(engine.near->count,4),128>>>(
                engine.near->blocks,engine.fine->d_tgt_ids_cached,engine.near->source_ids,engine.fine->d_tgt_pts,
                engine.fine->d_src_pts,engine.near->coefficients,engine.near->radial_ids,charges,from);
            cuda_ok(cudaGetLastError());
        }
        timeline.mark(2);
        add_cross<<<(nt+255)/256,256>>>(nt,target_ids,rotation,from,out);cuda_ok(cudaGetLastError());
        timeline.mark(3);
    }
    cuda_ok(cudaStreamSynchronize(0));
    timeline.finish(p->faces[0].fine->k.real(),mixed);
}

void PrismFaceFFT::evaluate_transpose(const cdouble* curl_seed,const cdouble* hessian_seed,
    cdouble* qx,cdouble* qy,cdouble* qz,bool mixed) {
    if(!p->faces[0].fine || !curl_seed || !hessian_seed || !qx || !qy || !qz)
        throw std::runtime_error("prism FFT transpose needs initialized buffers");
    const std::vector<cdouble> cs(curl_seed,curl_seed+3ULL*p->count),hs(hessian_seed,hessian_seed+3ULL*p->count);
    cdouble* out[3]={qx,qy,qz};for(auto q:out)std::fill(q,q+p->count,cdouble(0));
    for(int f=0;f<8;++f) {
        const Face& face=p->faces[f];const Layout& l=f<6?p->side:p->cap;
        const int cells=l.nx*l.ny,n=int(face.global.size());
        const V axes[3]={face.e1,face.e2,face.e3};
        p->fft_output.assign(size_t(l.channels)*6*cells,cdouble(0));
        p->fft_input.resize(size_t(l.channels)*3*cells);
        // Reverse gather_face: signed curl permutation, then the transpose
        // of its local-to-global rotation. Zero inactive lattice slots.
        for(int j=0;j<n;++j) {
            const int i=face.global[j],slot=face.slot[j];
            for(int k=0;k<3;++k) {
                cdouble c=0,h=0;
                for(int d=0;d<3;++d) {
                    const double sign=(d==1?1.:-1.)*(k==1?1.:-1.);
                    c+=sign*axes[2-k][2-d]*cs[3*i+d];h+=axes[k][d]*hs[3*i+d];
                }
                p->fft_output[size_t(slot/cells*6+k)*cells+slot%cells]=c;
                p->fft_output[size_t(slot/cells*6+k+3)*cells+slot%cells]=h;
            }
        }
        auto& fft=f<6?p->side_fft:p->cap_fft;
        fft.evaluate_transpose(p->fft_output.data(),p->fft_input.data());
        for(int j=0;j<n;++j) {
            const int i=face.global[j],slot=face.slot[j];
            for(int d=0;d<3;++d)for(int k=0;k<3;++k)
                out[d][i]+=axes[k][d]*p->fft_input[size_t(slot/cells*3+k)*cells+slot%cells];
        }
        Face& engine=p->faces[face.engine_face];
        engine.fine->near_field_fp32=engine.middle->near_field_fp32=engine.coarse->near_field_fp32=mixed;
        const auto& source=p->target_faces?face.other:face.cross_global;
        const auto& target=p->target_faces?face.cross_global:face.other;
        const int ns=int(source.size()),nt=int(target.size());
        p->curl.assign(size_t(nt)*3,cdouble(0));p->hessian.assign(size_t(nt)*3,cdouble(0));
        for(int j=0;j<nt;++j)for(int k=0;k<3;++k)for(int d=0;d<3;++d) {
            const double sign=(d==1?1.:-1.)*(k==1?1.:-1.);
            p->curl[3*j+k]+=sign*face.rotation[3*(2-d)+(2-k)]*cs[3*target[j]+d];
            p->hessian[3*j+k]+=face.rotation[3*d+k]*hs[3*target[j]+d];
        }
        for(auto& q:p->charges)q.resize(ns);
        engine.fine->evaluate_vector_actions_transpose(p->curl.data(),p->hessian.data(),
            p->charges[0].data(),p->charges[1].data(),p->charges[2].data(),
            engine.middle.get(),engine.coarse.get(),engine.near->count==0);
        if(engine.near->count) {
            cuda_ok(cudaMemset(p->d_packed,0,size_t(ns)*6*sizeof(double)));
            Outputs seed={engine.fine->d_grad_re,engine.fine->d_grad_im,engine.fine->d_hess_re,engine.fine->d_hess_im};
            if(mixed)apply_near_cache_transpose<float><<<dim3(engine.near->count,4),128>>>(
                engine.near->blocks,engine.fine->d_tgt_ids_cached,engine.near->source_ids,
                engine.fine->d_tgt_pts,engine.fine->d_src_pts,engine.near->coefficients,engine.near->radial_ids,seed,p->d_packed,ns);
            else apply_near_cache_transpose<double><<<dim3(engine.near->count,4),128>>>(
                engine.near->blocks,engine.fine->d_tgt_ids_cached,engine.near->source_ids,
                engine.fine->d_tgt_pts,engine.fine->d_src_pts,engine.near->coefficients,engine.near->radial_ids,seed,p->d_packed,ns);
            cuda_ok(cudaGetLastError());
            for(int d=0;d<3;++d) {
                cuda_ok(cudaMemcpy(p->real.data(),p->d_packed+size_t(2*d)*ns,size_t(ns)*sizeof(double),cudaMemcpyDeviceToHost));
                cuda_ok(cudaMemcpy(p->imag.data(),p->d_packed+size_t(2*d+1)*ns,size_t(ns)*sizeof(double),cudaMemcpyDeviceToHost));
                for(int j=0;j<ns;++j)p->charges[d][j]+=cdouble(p->real[j],p->imag[j]);
            }
        }
        for(int j=0;j<ns;++j)for(int d=0;d<3;++d)for(int k=0;k<3;++k)
            out[d][source[j]]+=face.rotation[3*d+k]*p->charges[k][j];
    }
}
