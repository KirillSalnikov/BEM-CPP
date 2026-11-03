// Sustained application-operator throughput. Geometry/FMM preparation is
// excluded explicitly; this is not a full BEM solve or a T-matrix benchmark.
#include "muller_fmm.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using C=std::complex<double>;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
double epoch(){return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();}

struct DeviceVector {
    void* pointer=nullptr;
    explicit DeviceVector(size_t count){CUDA_CHECK(cudaMalloc(&pointer,count*sizeof(C)));}
    ~DeviceVector(){if(pointer)cudaFree(pointer);}
};

double relative(const std::vector<C>& a,const std::vector<C>& b,double scale=1.){
    double delta=0.,norm=0.;
    for(size_t i=0;i<a.size();++i){delta+=std::norm(a[i]-scale*b[i]);norm+=std::norm(scale*b[i]);}
    const double result=std::sqrt(delta/std::max(norm,1e-300));
    if(!std::isfinite(result))throw std::runtime_error("Nonfinite operator result");
    return result;
}

int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=7)throw std::runtime_error("Usage: bem_gpu_thermal REF KA ITERATIONS WARMUP_SECONDS REPEATS OUTPUT.json");
    const int ref=std::stoi(argv[1]),iterations=std::stoi(argv[3]),repeats=std::stoi(argv[5]);
    const double ka=std::stod(argv[2]),warmup=std::stod(argv[4]);
    if(ref<3||ref>5||ka<=0||!std::isfinite(ka)||iterations<1||warmup<0||!std::isfinite(warmup)||repeats<1)
        throw std::invalid_argument("Invalid workload parameters");
    const bool score_candidate=warmup>=120&&repeats>=3;
    const auto setup_start=Clock::now();
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),C(ka),C(1.3116),options,7,4,5,64,false,2,0.,.96,nullptr,3);
    op.set_fmm_near_fp32(true);
    if(!op.device_matvec_available())throw std::runtime_error("A fully device-resident FMM operator is required");
    const int n=op.system_dofs;
    std::vector<C> input(n),double_input(n),reference(n),actual(n),strict(n);
    for(int i=0;i<n;++i){input[i]=C(std::sin(.013*(i+1)),std::cos(.017*(i+1)))/std::sqrt(double(n));double_input[i]=2.*input[i];}
    DeviceVector x(n),twice(n),zero(n),y(n),strict_y(n);
    CUDA_CHECK(cudaMemcpy(x.pointer,input.data(),n*sizeof(C),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(twice.pointer,double_input.data(),n*sizeof(C),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(zero.pointer,0,n*sizeof(C)));
    op.matvec(input.data(),reference.data());
    op.matvec_device(x.pointer,y.pointer);
    CUDA_CHECK(cudaMemcpy(actual.data(),y.pointer,n*sizeof(C),cudaMemcpyDeviceToHost));
    const double host_device_error=relative(actual,reference);
    op.matvec_batch2_device_strict(x.pointer,x.pointer,strict_y.pointer,strict_y.pointer);
    CUDA_CHECK(cudaMemcpy(strict.data(),strict_y.pointer,n*sizeof(C),cudaMemcpyDeviceToHost));
    const double mixed_strict_error=relative(actual,strict);
    // The first host action can populate FP32 phase caches. Freeze the
    // warmed resident action as the reference for repeated-action/linearity
    // checks, while retaining its comparison with the strict FP64 result.
    op.matvec_device(x.pointer,y.pointer);
    CUDA_CHECK(cudaMemcpy(reference.data(),y.pointer,n*sizeof(C),cudaMemcpyDeviceToHost));
    op.matvec_device(twice.pointer,y.pointer);
    CUDA_CHECK(cudaMemcpy(actual.data(),y.pointer,n*sizeof(C),cudaMemcpyDeviceToHost));
    const double linearity_error=relative(actual,reference,2.);
    op.matvec_device(zero.pointer,y.pointer);
    CUDA_CHECK(cudaMemcpy(actual.data(),y.pointer,n*sizeof(C),cudaMemcpyDeviceToHost));
    double zero_max=0.;for(C value:actual)zero_max=std::max(zero_max,std::abs(value));
    std::cout<<"VALIDATION host/device="<<host_device_error
             <<" mixed/strict="<<mixed_strict_error<<" linearity="<<linearity_error<<std::endl;
    if(host_device_error>2e-6||mixed_strict_error>2e-4||linearity_error>2e-6||zero_max!=0.)
        throw std::runtime_error("Initial operator validation failed");
    CUDA_CHECK(cudaDeviceSynchronize());
    const double setup=elapsed(setup_start);
    size_t free_bytes=0,total_bytes=0;CUDA_CHECK(cudaMemGetInfo(&free_bytes,&total_bytes));
    std::cout<<"PREPARED dofs="<<n<<" setup="<<setup<<"s strict_relative="<<mixed_strict_error<<std::endl;
    auto action=[&](int i){op.matvec_device((i&1)?twice.pointer:x.pointer,y.pointer);};
    const auto heat=Clock::now();const double heat_epoch=epoch();size_t warmup_calls=0;
    while(elapsed(heat)<warmup){action(int(warmup_calls++));}
    CUDA_CHECK(cudaDeviceSynchronize());const double heat_seconds=elapsed(heat);
    struct Trial {double start,seconds,cuda_span,error;};std::vector<Trial> trials;
    auto save=[&](bool passed){
        std::ofstream out(argv[6]);if(!out)throw std::runtime_error("Cannot write report");
        out<<std::setprecision(17)<<"{\"schema\":1,\"kind\":\"prepared_device_muller_operator\",\"passed\":"<<(passed?"true":"false")
           <<",\"score_candidate\":"<<(score_candidate?"true":"false")<<",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"ri\":1.3116,\"aspect\":1.4285714285714286"
           <<",\"system_dofs\":"<<n<<",\"mesh_triangles\":"<<op.mesh.elements.size()<<",\"quadrature_order\":7,\"duffy_order\":4,\"fmm_digits\":5,\"max_leaf\":64,\"near_radius\":3"
           <<",\"iterations_per_repeat\":"<<iterations<<",\"setup_excluded_s\":"<<setup<<",\"device_used_bytes_after_setup\":"<<total_bytes-free_bytes
           <<",\"warmup_start_unix_s\":"<<heat_epoch<<",\"warmup_s\":"<<heat_seconds<<",\"warmup_calls\":"<<warmup_calls
           <<",\"host_device_relative_error\":"<<host_device_error<<",\"mixed_strict_relative_error\":"<<mixed_strict_error<<",\"linearity_relative_error\":"<<linearity_error<<",\"zero_input_output_max\":"<<zero_max
           <<",\"cuda_span_note\":\"Event span includes host submission gaps and concurrent-media synchronization; not pure kernel-active time\",\"trials\":[";
        for(size_t j=0;j<trials.size();++j){const auto& t=trials[j];out<<(j?",":"")<<"{\"start_unix_s\":"<<t.start<<",\"wall_s\":"<<t.seconds<<",\"cuda_span_s\":"<<t.cuda_span<<",\"output_relative_error\":"<<t.error<<",\"matvecs\":"<<iterations<<"}";}
        out<<"]}\n";
    };
    for(int repeat=0;repeat<repeats;++repeat){
        cudaEvent_t begin,end;CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaDeviceSynchronize());const auto start=Clock::now();const double start_epoch=epoch();
        CUDA_CHECK(cudaEventRecord(begin));
        for(int i=0;i<iterations;++i)action(i);
        CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaDeviceSynchronize());
        const double seconds=elapsed(start);float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
        CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaMemcpy(actual.data(),y.pointer,n*sizeof(C),cudaMemcpyDeviceToHost));
        const double error=relative(actual,reference,((iterations-1)&1)?2.:1.);
        if(error>2e-6)throw std::runtime_error("Repeated FP32 operator output exceeded its numerical gate");
        trials.push_back({start_epoch,seconds,double(ms)/1000,error});save(false);
        std::cout<<"TRIAL "<<repeat+1<<" matvecs="<<iterations<<" wall="<<seconds<<"s relative="<<error<<std::endl;
    }
    save(true);op.cleanup();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
