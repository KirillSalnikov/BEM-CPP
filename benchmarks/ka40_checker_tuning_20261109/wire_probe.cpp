#include "external_fft_action.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>

int main(int argc,char** argv)try {
    if(argc!=3)return 2;
    const size_t n=1142784;
    ExternalFFTAction action(argv[1],n,6,40.,{1.3116,0.},1.6495721976846451,7,6);
    using C=std::complex<double>;
    std::vector<C> input(n), reference(n), result(n);
    std::ifstream stream(argv[2],std::ios::binary);
    for(int vector=0;vector<6;++vector) {
        stream.read(reinterpret_cast<char*>(input.data()),n*sizeof(C));
        if(!stream)throw std::runtime_error("Missing probe currents");
        for(int trial=0;trial<4;++trial) {
            const bool compact=trial==1 || trial==2;
            setenv("BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32",compact?"1":"0",1);
            auto start=std::chrono::steady_clock::now();
            action.apply(input.data(),result.data());
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if(trial==0)reference=result;
            long double numerator=0,denominator=0;
            for(size_t i=0;i<n;++i){numerator+=std::norm(result[i]-reference[i]);denominator+=std::norm(reference[i]);}
            const double error=std::sqrt(double(numerator/denominator));
            std::printf("{\"vector\":%d,\"trial\":%d,\"compact\":%s,\"seconds\":%.9f,\"relative_difference\":%.15g}\n",
                        vector,trial,compact?"true":"false",seconds,error);std::fflush(stdout);
            if(!std::isfinite(error)||error>1e-7)throw std::runtime_error("Compact transport changes FFT action");
        }
    }
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
