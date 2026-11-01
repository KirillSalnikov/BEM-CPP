#include "muller_fmm_cpu.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <sched.h>
#include <sstream>

// Optional disjoint CPU sets for the host checker and CUDA submission team.
// Set every member of an existing OpenMP pool, not only its master thread.
static void configure_threads(int count,const char* affinity_name) {
    omp_set_num_threads(count);
    const char* value=std::getenv(affinity_name);
    if(!value)return;
    cpu_set_t mask;CPU_ZERO(&mask);
    std::stringstream input(value);std::string item;
    while(std::getline(input,item,',')) {
        const auto dash=item.find('-');
        const int first=std::stoi(item.substr(0,dash));
        const int last=dash==std::string::npos?first:std::stoi(item.substr(dash+1));
        if(first<0||last<first||last>=CPU_SETSIZE)throw std::runtime_error("invalid CPU affinity range");
        for(int cpu=first;cpu<=last;++cpu)CPU_SET(cpu,&mask);
    }
    if(!CPU_COUNT(&mask))throw std::runtime_error("empty CPU affinity");
    std::atomic<int> failed(0);
#pragma omp parallel
    {if(sched_setaffinity(0,sizeof(mask),&mask)!=0)failed.store(1);}
    if(failed.load())throw std::runtime_error("cannot set CPU affinity");
}

using C=std::complex<double>;
using Clock=std::chrono::steady_clock;
static double error(const std::vector<C>& a,const std::vector<C>& b) {
    long double num=0,den=0;
    if(a.size()!=b.size())throw std::runtime_error("mismatched validation vectors");
    for(size_t j=0;j<a.size();++j) {
        if(!std::isfinite(a[j].real())||!std::isfinite(a[j].imag())||
           !std::isfinite(b[j].real())||!std::isfinite(b[j].imag()))
            throw std::runtime_error("nonfinite validation vector");
        num+=std::norm(a[j]-b[j]);den+=std::norm(b[j]);
    }
    if(den==0)return num==0?0:std::numeric_limits<double>::infinity();
    return std::sqrt(double(num/den));
}
static double flag(const char* name,double fallback) {
    const char* value=std::getenv(name);return value?std::stod(value):fallback;
}
int main(int argc,char** argv)try {
    if(argc!=4)throw std::runtime_error("usage: muller_cpu_overlap_probe REF KA REPORT.json");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),aspect=10./7.;
    const int threads=int(flag("BEM_CPU_THREADS",16)),gpu_threads=int(flag("BEM_CPU_GPU_THREADS",4)),repeats=int(flag("BEM_CPU_REPEATS",3));
    if(ref<0||ref>5||ka<=0||threads<1||threads>32||repeats<1||repeats>20)
        throw std::runtime_error("invalid diagnostic limits");
    omp_set_dynamic(0);configure_threads(threads,"BEM_CPU_AFFINITY");
    setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);
    setenv("BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1",1);
    setenv("BEM_FMM_PAIR_CURRENTS","0",1);
    // Retain the independent-band GPU path for a numerical implementation check.
    setenv("BEM_FMM_MULTILEVEL","0",1);
    setenv("BEM_FMM_CONCURRENT_MEDIA","0",1);
    setenv("BEM_FACE_FFT_STRICT_ONLY","1",1);
    const Mesh geometry=regular_prism(6,aspect,ref,1.,0,false,true);
    // ref=0 has mirror-preserving centre fans; count actual triangles.
    const int points=7*int(geometry.tris.size()/3);
    const int depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
    setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
    setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    MullerFmmOperator op;MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const C index(1.3116,flag("BEM_CPU_RI_IM",0));
    auto setup=Clock::now();
    op.init(geometry,C(ka),index,options,7,int(flag("BEM_CPU_DUFFY",6)),5,leaf(depth));
    const double gpu_setup=bem_cpu::elapsed(setup);setup=Clock::now();
    bem_cpu::MullerAction cpu(op,size_t(flag("BEM_CPU_PHASE_MB",64)*1024*1024),flag("BEM_CPU_MERGE_BANDS",1)!=0);
    const double cpu_setup=bem_cpu::elapsed(setup);
    std::printf("CPU prepared: dofs=%d Q=%zu extra_workspace=%.3f GiB setup=%.3fs\n",op.system_dofs,op.quadrature.size(),cpu.workspace_bytes()/1073741824.,cpu_setup);
    std::vector<C> x(op.system_dofs),y(x.size()),legacy(x.size()),strict(x.size()),mixed(x.size()),other(x.size());
    std::mt19937 rng(9281);std::normal_distribution<double> normal;
    for(auto& q:x)q=C(normal(rng),normal(rng));
    auto gpu=[&](bool full,std::vector<C>& out) {
        configure_threads(gpu_threads,"BEM_CPU_GPU_AFFINITY");
        const auto t=Clock::now();
        if(full)op.matvec_strict(x.data(),out.data());else op.matvec(x.data(),out.data());
        return bem_cpu::elapsed(t);
    };
    op.set_fmm_near_fp32(true);gpu(true,legacy);
    configure_threads(threads,"BEM_CPU_AFFINITY");
    auto start=Clock::now();cpu.apply(x.data(),y.data());
    const double first_cpu=bem_cpu::elapsed(start),implementation_error=error(y,legacy);
    std::printf("Independent CPU %.6fs vs independent-band GPU: relative error %.9e\n",first_cpu,implementation_error);
    if(!std::isfinite(implementation_error)||implementation_error>2e-9)
        throw std::runtime_error("CPU action failed independent-band GPU reference");
    double direct_error=-1;
    if(flag("BEM_CPU_DIRECT",0)!=0) {
        if(ref>1)throw std::runtime_error("direct validation limited to ref <= 1");
        op.matvec_direct_reference(x.data(),other.data());direct_error=error(y,other);
        std::printf("CPU vs direct quadrature error %.9e\n",direct_error);
        if(!std::isfinite(direct_error)||direct_error>2e-5)throw std::runtime_error("direct CPU validation failed");
    }
    setenv("BEM_FMM_MULTILEVEL","1",1);setenv("BEM_FMM_CONCURRENT_MEDIA","1",1);gpu(true,strict);
    const double multilevel_error=error(y,strict);
    std::vector<double> mlfma_strict_s;for(int j=0;j<3;++j)mlfma_strict_s.push_back(gpu(true,strict));
    std::printf("CPU vs production MLFMA error %.9e\n",multilevel_error);
    bool fft=false;
    if(ref>0 && flag("BEM_CPU_FFT",1)!=0) {
        op.enable_prism_face_fft(ref,aspect);gpu(true,strict);fft=true;
        std::printf("CPU vs face-FFT strict error %.9e\n",error(y,strict));
    }
    const double fft_error=error(y,strict);
    gpu(false,mixed);
    // Final full residual agreement must also be examined on solved currents;
    // a random-vector action comparison alone does not certify a 2e-6 solve.
    std::vector<double> cpu_s,gpu_strict_s,gpu_mixed_s,serial_segment_s,overlap_s,overlap_cpu_s,overlap_gpu_s;
    double concurrent_error=0,mixed_concurrent_error=0,inplace_error=0;
    const int gpu_count=std::max(1,int(flag("BEM_CPU_GPU_ACTIONS",4)));
    for(int repeat=0;repeat<repeats;++repeat) {
        gpu_strict_s.push_back(gpu(true,strict));gpu_mixed_s.push_back(gpu(false,mixed));
        start=Clock::now();gpu(true,strict);
        for(int j=0;j<gpu_count;++j)gpu(false,mixed);
        serial_segment_s.push_back(bem_cpu::elapsed(start));
        configure_threads(threads,"BEM_CPU_AFFINITY");start=Clock::now();cpu.apply(x.data(),y.data());cpu_s.push_back(bem_cpu::elapsed(start));
        // Copies belong to the submitted candidate, including while GPU works.
        const auto snapshot=x;
        std::atomic<bool> ready(false),go(false);
        auto worker=std::async(std::launch::async,[&] {
            configure_threads(threads,"BEM_CPU_AFFINITY");ready.store(true);
            while(!go.load())std::this_thread::yield();
            const auto t=Clock::now();cpu.apply(snapshot.data(),other.data());return bem_cpu::elapsed(t);
        });
        while(!ready.load())std::this_thread::yield();
        start=Clock::now();go.store(true);
        const auto gt=Clock::now();
        for(int j=0;j<gpu_count;++j)gpu(false,mixed);
        overlap_gpu_s.push_back(bem_cpu::elapsed(gt));
        overlap_cpu_s.push_back(worker.get());overlap_s.push_back(bem_cpu::elapsed(start));
        concurrent_error=std::max(concurrent_error,error(other,y));
        gpu(false,other);mixed_concurrent_error=std::max(mixed_concurrent_error,error(other,mixed));
        std::printf("repeat %d CPU=%.6fs GPUstrict=%.6fs mixed=%.6fs serial=%.6fs overlapped=%.6fs CPUtask=%.6fs GPUtasks=%.6fs\n",
            repeat,cpu_s.back(),gpu_strict_s.back(),gpu_mixed_s.back(),serial_segment_s.back(),overlap_s.back(),overlap_cpu_s.back(),overlap_gpu_s.back());
    }
    configure_threads(threads,"BEM_CPU_AFFINITY");other=x;cpu.apply(other.data(),other.data());inplace_error=error(other,y);
    // A second distinct vector catches accidental reuse of cached current/fields.
    for(auto& q:x)q=C(normal(rng),normal(rng));
    gpu(true,strict);configure_threads(threads,"BEM_CPU_AFFINITY");cpu.apply(x.data(),other.data());
    const double second_error=error(other,strict);
    std::vector<double> solved_cpu_residual,solved_gpu_residual,solved_image_difference;
    if(const char* path=std::getenv("BEM_CPU_VALIDATION_FILE")) {
        std::ifstream file(path,std::ios::binary);int header[2]{};
        file.read(reinterpret_cast<char*>(header),sizeof(header));
        if(!file || header[0]!=op.system_dofs || header[1]<1 || header[1]>16)
            throw std::runtime_error("invalid saved-current validation file");
        std::vector<C> rhs(x.size());
        for(int j=0;j<header[1];++j) {
            file.read(reinterpret_cast<char*>(x.data()),x.size()*sizeof(C));
            file.read(reinterpret_cast<char*>(rhs.data()),rhs.size()*sizeof(C));
            if(!file)throw std::runtime_error("truncated saved-current validation file");
            gpu(true,strict);configure_threads(threads,"BEM_CPU_AFFINITY");cpu.apply(x.data(),other.data());
            solved_gpu_residual.push_back(error(strict,rhs));solved_cpu_residual.push_back(error(other,rhs));
            // Difference relative to the original RHS, the actual acceptance scale.
            for(size_t k=0;k<x.size();++k)y[k]=rhs[k]+other[k]-strict[k];
            solved_image_difference.push_back(error(y,rhs));
            std::printf("saved solution %d CPU residual %.9e GPU residual %.9e image/RHS difference %.9e\n",j,
                solved_cpu_residual.back(),solved_gpu_residual.back(),solved_image_difference.back());
        }
    }
    std::ofstream out(argv[3]);out<<std::setprecision(17);
    out<<"{\"scope\":\"full CPU FP64 BEM action and overlap diagnostic; no T solve\",\"ref\":"<<ref<<",\"ka\":"<<ka
       <<",\"ri_real\":1.3116,\"ri_imag\":"<<index.imag()<<",\"aspect\":"<<aspect<<",\"threads\":"<<threads
       <<",\"dofs\":"<<op.system_dofs<<",\"quadrature_points\":"<<points<<",\"depth\":"<<depth
       <<",\"GPU_setup_s\":"<<gpu_setup<<",\"CPU_setup_s\":"<<cpu_setup
       <<",\"CPU_workspace_bytes_excluding_shared_operator_and_indices\":"<<cpu.workspace_bytes()
       <<",\"CPU_merged_equal_angular_bands\":"<<cpu.merged_angular_bands
       <<",\"CPU_AVX512_near_bands\":"<<cpu.avx512_near_bands
       <<",\"independent_band_relative_error\":"<<implementation_error<<",\"direct_quadrature_relative_error\":"<<direct_error<<",\"MLFMA_relative_error\":"<<multilevel_error
       <<",\"face_FFT_enabled\":"<<(fft?"true":"false")<<",\"strict_relative_error\":"<<fft_error
       <<",\"second_vector_strict_relative_error\":"<<second_error<<",\"concurrent_CPU_error\":"<<concurrent_error
       <<",\"concurrent_GPU_error\":"<<mixed_concurrent_error<<",\"inplace_error\":"<<inplace_error
       <<",\"mixed_GPU_actions_per_CPU_check\":"<<gpu_count
       <<",\"last_project_s\":"<<cpu.project_s<<",\"last_fields_s\":"<<cpu.fields_s<<",\"last_test_s\":"<<cpu.test_s;
    auto array=[&](const char* name,const std::vector<double>& values) {
        out<<",\""<<name<<"\":[";for(size_t j=0;j<values.size();++j)out<<(j?",":"")<<values[j];out<<"]";
    };
    array("CPU_s",cpu_s);array("GPU_strict_s",gpu_strict_s);array("GPU_mixed_s",gpu_mixed_s);
    array("GPU_MLFMA_strict_s",mlfma_strict_s);
    array("serial_GPU_strict_then_mixed_segment_s",serial_segment_s);
    array("saved_solution_CPU_residual",solved_cpu_residual);array("saved_solution_GPU_residual",solved_gpu_residual);
    array("saved_solution_CPU_GPU_difference_over_RHS",solved_image_difference);
    array("overlap_s",overlap_s);array("overlap_CPU_s",overlap_cpu_s);array("overlap_GPU_s",overlap_gpu_s);
    array("last_P2M_M2M_M2L_L2L_L2P_P2P_s",std::vector<double>(cpu.fmm_times.begin(),cpu.fmm_times.end()));
    out<<"}\n";
    if(concurrent_error>2e-12||mixed_concurrent_error>5e-6||inplace_error>2e-12)
        throw std::runtime_error("concurrency/in-place check failed");
    op.cleanup();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
