#include "p2p.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <typename T> struct Device {
    T* p=nullptr;
    explicit Device(const std::vector<T>& a) {
        check(cudaMalloc(&p,a.size()*sizeof(T)));
        check(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(p); }
};

int main() try {
    setenv("BEM_FMM_P2P_LOCAL_COORDINATES","0",1);
    // Empty leaves, incomplete tiles, coincident points, and more targets
    // than threads exercise every barrier/tile boundary of the new kernel.
    const std::vector<int> target_offsets={0,0,1,18,147,660,1691};
    const std::vector<int> near_offsets={0,0,1,130,387,900,1157};
    const int leaves=6,ns=719,nt=target_offsets.back(),stride=3*nt;
    std::mt19937 rng(53); std::uniform_real_distribution<double> random(-1.,1.);
    std::vector<double> src(3*ns),tgt(3*nt),q(6*ns),initial(4*stride);
    for(auto& v:src)v=random(rng);
    for(auto& v:tgt)v=random(rng);
    for(auto& v:q)v=random(rng);
    for(auto& v:initial)v=random(rng)*.01;
    for(int t=0;t<nt;t+=97) for(int d=0;d<3;++d)tgt[3*t+d]=src[3*(t%ns)+d];
    std::vector<int> target_ids(nt),source_ids(ns),flat_ids(near_offsets.back());
    std::iota(target_ids.begin(),target_ids.end(),0);
    std::iota(source_ids.begin(),source_ids.end(),0);
    for(size_t i=0;i<flat_ids.size();++i)flat_ids[i]=int(i%ns);
    std::vector<int> source_offsets(leaves+1),empty_offsets(leaves+1,0),dummy(1,0);
    for(int i=0;i<=leaves;++i)source_offsets[i]=ns*i/leaves;
    Device<double> ds(src),dt(tgt),dq(q),out(initial);
    Device<int> dto(target_offsets),dti(target_ids),dso(source_offsets),dsi(source_ids);
    Device<int> dno(near_offsets),dni(flat_ids),deo(empty_offsets),dummy_ids(dummy);
    double max_mixed=0,max_strict=0; int cases=0;
    for(int variant: {0,1,2}) for(bool mixed: {false,true}) for(double absorption: {0.,.35})
    for(int threads: {64,512}) for(int split: {1,8,32}) for(bool flat: {false,true}) {
        setenv("BEM_FMM_P2P_THREADS",std::to_string(threads).c_str(),1);
        setenv("BEM_FMM_P2P_LEAF_SPLIT",std::to_string(split).c_str(),1);
        setenv("BEM_FMM_P2P_SHARED_REAL",variant>=1?"1":"0",1);
        setenv("BEM_FMM_P2P_SHARED_FAST_TRIG",variant==2?"1":"0",1);
        std::vector<double> control(initial.size()),actual(initial.size());
        for(int enabled: {0,1}) {
            setenv("BEM_FMM_P2P_SHARED_SOURCES",enabled?"1":"0",1);
            check(cudaMemcpy(out.p,initial.data(),initial.size()*sizeof(double),cudaMemcpyHostToDevice));
            launch_p2p_vector_actions_batch3_leaf(dt.p,ds.p,
                dq.p,dq.p+ns,dq.p+2*ns,dq.p+3*ns,dq.p+4*ns,dq.p+5*ns,
                dto.p,dti.p,dso.p,dsi.p,deo.p,dummy_ids.p,
                flat?dno.p:nullptr,flat?dni.p:nullptr,leaves,7.2,absorption,
                out.p,out.p+stride,out.p+2*stride,out.p+3*stride,mixed);
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            auto& result=enabled?actual:control;
            check(cudaMemcpy(result.data(),out.p,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
        }
        double difference=0,norm=0;
        for(size_t i=0;i<actual.size();++i) {
            if(!std::isfinite(actual[i]))throw std::runtime_error("non-finite P2P output");
            difference+=(actual[i]-control[i])*(actual[i]-control[i]);norm+=control[i]*control[i];
        }
        const double error=std::sqrt(difference/std::max(norm,1e-300));
        if(error>(mixed?1e-6:1e-13))throw std::runtime_error("shared P2P differs from control");
        if(mixed)max_mixed=std::max(max_mixed,error);else max_strict=std::max(max_strict,error);
        ++cases;
    }
    std::printf("P2P shared sources: %d cases passed; mixed error %.3e, FP64 error %.3e\n",cases,max_mixed,max_strict);
    // A small cloud far from the origin exposes rounding-before-subtraction.
    // Compare with the FP64 kernel on the SAME double coordinates, including
    // exact coincidences, empty leaves and partial target/source tiles.
    double local_max=0;int local_cases=0;
    for(double translation:{0.,10000.,-10000.}) {
        std::vector<double> shifted_src(src),shifted_tgt(tgt);
        for(auto& value:shifted_src)value=value*.001+translation;
        for(auto& value:shifted_tgt)value=value*.001+translation;
        check(cudaMemcpy(ds.p,shifted_src.data(),src.size()*sizeof(double),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dt.p,shifted_tgt.data(),tgt.size()*sizeof(double),cudaMemcpyHostToDevice));
        for(double absorption:{0.,.35})for(int threads:{64,128,256})for(int split:{1,8,32}) {
            setenv("BEM_FMM_P2P_LOCAL_COORDINATES","1",1);
            setenv("BEM_FMM_P2P_SHARED_SOURCES","1",1);
            setenv("BEM_FMM_P2P_SHARED_REAL","1",1);
            setenv("BEM_FMM_P2P_SHARED_FAST_TRIG","0",1);
            setenv("BEM_FMM_P2P_THREADS",std::to_string(threads).c_str(),1);
            setenv("BEM_FMM_P2P_LEAF_SPLIT",std::to_string(split).c_str(),1);
            std::vector<double> control(initial.size()),actual(initial.size());
            for(bool mixed:{false,true}) {
                check(cudaMemcpy(out.p,initial.data(),initial.size()*sizeof(double),cudaMemcpyHostToDevice));
                launch_p2p_vector_actions_batch3_leaf(dt.p,ds.p,
                    dq.p,dq.p+ns,dq.p+2*ns,dq.p+3*ns,dq.p+4*ns,dq.p+5*ns,
                    dto.p,dti.p,dso.p,dsi.p,deo.p,dummy_ids.p,dno.p,dni.p,leaves,7.2,absorption,
                    out.p,out.p+stride,out.p+2*stride,out.p+3*stride,mixed);
                check(cudaGetLastError());check(cudaDeviceSynchronize());
                auto& result=mixed?actual:control;
                check(cudaMemcpy(result.data(),out.p,result.size()*sizeof(double),cudaMemcpyDeviceToHost));
            }
            double difference=0,norm=0;
            for(size_t i=0;i<actual.size();++i) {
                if(!std::isfinite(actual[i]))throw std::runtime_error("non-finite local P2P output");
                difference+=std::pow(actual[i]-control[i],2);norm+=control[i]*control[i];
            }
            const double error=std::sqrt(difference/norm);
            if(error>2e-5)throw std::runtime_error("local P2P differs from FP64: "+std::to_string(error));
            local_max=std::max(local_max,error);++local_cases;
        }
    }
    std::printf("Local-coordinate P2P: %d translated-cloud cases passed; FP64 relative error %.3e\n",local_cases,local_max);
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
