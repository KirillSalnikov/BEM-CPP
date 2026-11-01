#include "host_spectral_inverse.h"
#include <cmath>
#include <iostream>
int main()try {
    using C=std::complex<double>;
    if(!bem_cpu::FftwRuntime::get().available()){std::cout<<"SKIP FFTW runtime unavailable\n";return 0;}
    double worst=0;const double pi=std::acos(-1.);
    for(auto dimensions:{std::pair<int,int>(4,4),std::pair<int,int>(3,5)}) {
        const int nx=dimensions.first,ny=dimensions.second,g=nx*ny,c=3,b=2;
        std::vector<C> symbol(c*c*g),input(b*c*g),observed(input.size()),expected(input.size()),hat(input.size());
        for(size_t i=0;i<symbol.size();++i)symbol[i]=C(std::cos(.13*i),std::sin(.21*i))+.01*double(i);
        for(size_t i=0;i<input.size();++i)input[i]=C(std::sin(.17*i),std::cos(.31*i));
        bem_cpu::SpectralInverse action(nx,ny,c,b,symbol);
        for(int adj=0;adj<2;++adj) {
            for(int j=0;j<b*c;++j)for(int f=0;f<g;++f) {
                C sum=0;for(int x=0;x<g;++x)sum+=input[j*g+x]*std::exp(C(0,-2*pi*((f/ny)*double(x/ny)/nx+(f%ny)*double(x%ny)/ny)));
                hat[j*g+f]=sum;
            }
            for(int batch=0;batch<b;++batch)for(int row=0;row<c;++row)for(int x=0;x<g;++x) {
                C sum=0;for(int f=0;f<g;++f) {
                    C value=0;for(int col=0;col<c;++col) {
                        C a=symbol[(adj?col*c+row:row*c+col)*g+f];if(adj)a=std::conj(a);
                        value+=a*hat[(batch*c+col)*g+f];
                    }
                    sum+=value*std::exp(C(0,2*pi*((f/ny)*double(x/ny)/nx+(f%ny)*double(x%ny)/ny)))/double(g);
                }
                expected[(batch*c+row)*g+x]=sum;
            }
            action.apply(input.data(),observed.data(),adj);
            double num=0,den=0;for(size_t j=0;j<input.size();++j){num+=std::norm(observed[j]-expected[j]);den+=std::norm(expected[j]);}
            const double error=std::sqrt(num/den);worst=std::max(worst,error);
            if(!std::isfinite(error)||error>2e-13)throw std::runtime_error("CPU FFT symbol/adjoint/normalization mismatch");
            auto alias=input;action.apply(alias.data(),alias.data(),adj);
            if(alias!=observed)throw std::runtime_error("CPU FFT aliasing");
        }
    }
    std::cout<<"PASS: CPU FFT matches explicit 2D DFT, complex channel coupling, adjoint, odd alignment, in-place; error "<<worst<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
