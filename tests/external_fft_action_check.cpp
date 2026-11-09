#include "external_fft_action.h"
#include <cmath>
#include <cstdio>
#include <vector>
int main(int argc,char**argv)try{
    if(argc!=2)return 2;
    ExternalFFTAction action(argv[1],9,6,40.,{1.3116,0.},1.6495721976846451,7,6);
    using C=std::complex<double>;std::vector<C>x(9),y(9);
    for(int rep=0;rep<4;++rep){
        setenv("BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32",rep>=2?"1":"0",1);
        for(int i=0;i<9;++i)x[i]=C(i+rep,-.5*i+rep);
        auto expected=x;
        for(auto&v:expected)v*=C(3,2);
        if(rep){y=x;action.apply(y.data(),y.data());}
        else action.apply(x.data(),y.data());
        for(int i=0;i<9;++i)if(std::abs(y[i]-expected[i])>1e-12)throw std::runtime_error("FFT transport action mismatch");
    }
    std::fill(x.begin(),x.end(),C());action.apply(x.data(),y.data());
    for(auto value:y)if(value!=C())throw std::runtime_error("FFT transport zero mismatch");
    if(action.calls()!=5)throw std::runtime_error("FFT request count");
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
