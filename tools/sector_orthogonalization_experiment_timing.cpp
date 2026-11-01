#include "sector_orthogonalization_experiment.h"
#include <cstdio>
using namespace bem_streaming;
int main()try {
    const int n=258048,width=n/12;std::array<int,13> offsets;
    for(int s=0;s<=12;++s)offsets[s]=s*width;
    std::vector<Complex> answer(n),rhs(n),x(n);
    for(int i=0;i<n;++i)answer[i]=Complex(std::sin(.13*i),std::cos(.29*i));
    const Action action=[&](const Complex* a,Complex* b){
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i){int j=i%width;
            const Complex diagonal(.02+1.98*(j%997)/996.,.02*std::sin(j*.01));
            b[i]=diagonal*a[i]+(j+1<width?Complex(.003,.001)*a[i+1]:Complex(0));}
    };
    action(answer.data(),rhs.data());
    for(const char* mode:{"0","1","2","2","1","0"}) {
        setenv("BEM_SECTOR_ORTHOGONALIZATION_EXPERIMENT",mode,1);
        const auto r=sector_gmres_experiment(action,action,rhs.data(),x.data(),offsets,64,2e-6,500,false);
        if(r.residual>2e-6)throw std::runtime_error("CPU timing solve failed");
        std::printf("{\"method\":%s,\"iterations\":%d,\"seconds\":%.9g,\"orthogonalization_s\":%.9g,\"strict_residual\":%.9g}\n",mode,r.iterations,r.seconds,r.orthogonalization_seconds,r.residual);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
