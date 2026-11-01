#include "host_regular_face_fft.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>
using C=std::complex<double>;using V=std::vector<C>;using Clock=std::chrono::steady_clock;
int main(int argc,char** argv)try{
    if(argc!=2)throw std::invalid_argument("REPORT.json");
    constexpr int degree=4,channels=4;
    std::ofstream report(argv[1]);report<<std::setprecision(17)<<"{\"shared_machine\":true,\"degree\":4,\"cases\":[";
    for(int case_id=0;case_id<2;++case_id){const int nx=case_id?24:8,ny=case_id?40:12;
        const size_t ni=size_t(channels)*6*nx*ny,no=2*ni;
        bem_cpu::RegularFaceAction action(nx,ny,{{.02,0}},{{.01,.025}},{{{0,0}},{{.003,.007}},{{.011,.009}},{{.018,.016}}},C(40,.3),degree,C(1.3116,.1));
        std::vector<V> inputs(degree+1,V(ni)),a(degree+1,V(no)),b=a;
        std::mt19937 rng(918);std::normal_distribution<double> normal;
        const C* input[degree+1];C* output[degree+1];
        for(int j=0;j<=degree;++j){input[j]=inputs[j].data();output[j]=b[j].data();for(auto& z:inputs[j])z=C(normal(rng),normal(rng));}
        auto separate=[&](){for(int j=0;j<=degree;++j)action.evaluate_convolution(j,input,a[j].data());};
        auto together=[&](){action.evaluate_history(input,output);};
        separate();together();double error=0;
        for(int j=0;j<=degree;++j){double n=0,d=0;for(size_t i=0;i<no;++i){n+=std::norm(a[j][i]-b[j][i]);d+=std::norm(a[j][i]);}error=std::max(error,std::sqrt(n/d));}
        if(error>1e-12)throw std::runtime_error("history timing equivalence");
        std::vector<double> old_times,new_times;
        for(int round=0;round<8;++round)for(int pass=0;pass<2;++pass){const bool batch=(round+pass)%2;const auto start=Clock::now();
            if(batch)together();else separate();
            (batch?new_times:old_times).push_back(std::chrono::duration<double>(Clock::now()-start).count());}
        auto median=[](std::vector<double> x){std::sort(x.begin(),x.end());return .5*(x[3]+x[4]);};
        report<<(case_id?",":"")<<"{\"nx\":"<<nx<<",\"ny\":"<<ny<<",\"channels\":4,\"error\":"<<error<<",\"workspace_bytes\":"<<action.workspace_bytes()
              <<",\"separate_median_s\":"<<median(old_times)<<",\"history_median_s\":"<<median(new_times)
              <<",\"separate_forward_FFT_count\":"<<channels*6*(degree+1)*(degree+2)/2<<",\"history_forward_FFT_count\":"<<channels*6*(degree+1)
              <<",\"inverse_FFT_count_each\":"<<channels*12*(degree+1)<<",\"separate_samples_s\":[";
        for(size_t j=0;j<old_times.size();++j)report<<(j?",":"")<<old_times[j];report<<"],\"history_samples_s\":[";
        for(size_t j=0;j<new_times.size();++j)report<<(j?",":"")<<new_times[j];report<<"]}";
    }
    report<<"],\"production_speedup_proven\":false}\n";
    std::puts("PASS alternating spectral-history timing and equivalence");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
