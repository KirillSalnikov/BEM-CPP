#include "block_circulant_symmetry.h"
#include <cstdio>
#include <random>
using C=std::complex<double>;
using V=std::vector<C>;
static V convolution(const V& kernel,const V& x,int n,int c) {
    const int grid=n*n;V out(x.size());
    for(int a=0;a<c;++a)for(int b=0;b<c;++b)for(int i=0;i<n;++i)for(int j=0;j<n;++j)
        for(int u=0;u<n;++u)for(int v=0;v<n;++v)
            out[a*grid+i*n+j]+=kernel[(a*c+b)*grid+(i-u+n)%n*n+(j-v+n)%n]*x[b*grid+u*n+v];
    return out;
}
int main()try {
    std::mt19937 rng(1827);std::normal_distribution<double> normal;
    for(int n:{3,4,5}) {
        const int c=3,grid=n*n;
        auto r=CirculantSymmetry::identity(n,c),v=r;
        r.matrix={{0,-1,1,1}};r.channel={1,2,0};r.offset={{{0,1}},{{1,0}},{{-1,2}}};
        r.solution_sign={1,-1,-1};r.rhs_sign={-1,1,-1};
        v.matrix={{1,0,-1,-1}};v.channel={1,0,2};v.offset={{{1,0}},{{0,-1}},{{2,1}}};
        v.solution_sign={-1,-1,1};v.rhs_sign={1,1,-1};
        std::vector<CirculantSymmetry> maps={CirculantSymmetry::identity(n,c),r,v,compose(v,r)};
        V kernel(c*c*grid),x(c*grid),expected(x.size());
        for(auto& z:kernel)z=C(normal(rng),normal(rng));for(auto& z:x)z=C(normal(rng),normal(rng));
        for(const auto& g:maps) {
            V pull(x.size());
            for(int ch=0;ch<c;++ch)for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
                const auto p=g.point(ch,i,j);pull[ch*grid+i*n+j]=double(g.rhs_sign[ch])*x[g.channel[ch]*grid+p[0]*n+p[1]];
            }
            const auto y=convolution(kernel,pull,n,c);
            for(int ch=0;ch<c;++ch)for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
                const auto p=g.point(ch,i,j);expected[g.channel[ch]*grid+p[0]*n+p[1]]+=double(g.solution_sign[ch])/maps.size()*y[ch*grid+i*n+j];
            }
        }
        const auto result=convolution(average_inverse_kernel(kernel,maps),x,n,c);
        double error=0,norm=0;for(size_t i=0;i<x.size();++i){error+=std::norm(result[i]-expected[i]);norm+=std::norm(expected[i]);}
        std::printf("n=%d explicit_group_vs_kernel_error=%.3e\n",n,std::sqrt(error/norm));
        if(std::sqrt(error/norm)>2e-14)throw std::runtime_error("inverse kernel symmetry failed");
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
