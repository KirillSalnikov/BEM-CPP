#include "global_coarse_inverse.h"
#include <cstdio>
#include <vector>
#include <stdexcept>
using C=std::complex<double>;
using V=std::vector<C>;
double error(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}return std::sqrt(x/std::max(y,1e-300));}
int main()try {
    const int n=43,r=8;
    const auto a=[&](const C* x,C* y){for(int i=0;i<n;++i)y[i]=C(2+.02*i,.03)*x[i]+C(.17,.08)*x[(i+1)%n]+C(-.1,.02)*x[(i+13)%n];};
    const auto b=[&](const C* x,C* y){for(int i=0;i<n;++i)y[i]=.3*x[i]+C(.02,.03)*x[(i+3)%n];};
    const auto generate=[&](int j,C* x){std::fill(x,x+n,C(0));if(j<r){x[j]=1;x[j+10]=C(.2,.13);x[j+22]=C(-.17,.23);}};
    GlobalCoarseInverse inverse;inverse.build(n,r+1,generate,a,b);
    if(inverse.available_rank()!=r)throw std::runtime_error("dependent basis candidate retained");
    V u(n),au(n),actual(n),x(n),y(n),px(n),py(n),sum(n),expected(n);
    double maximum=0;
    for(int rank:{4,8}) {
        inverse.set_rank(rank);
        for(int j=0;j<rank;++j){generate(j,u.data());a(u.data(),au.data());inverse.apply(au.data(),actual.data());maximum=std::max(maximum,error(actual,u));}
    }
    for(int i=0;i<n;++i){x[i]=C(std::sin(.3*i),std::cos(.17*i));y[i]=C(std::cos(.43*i),std::sin(.71*i));sum[i]=C(.7,.2)*x[i]+C(-.1,.4)*y[i];}
    inverse.apply(x.data(),px.data());inverse.apply(y.data(),py.data());inverse.apply(sum.data(),actual.data());
    for(int i=0;i<n;++i)expected[i]=C(.7,.2)*px[i]+C(-.1,.4)*py[i];
    const double linear=error(actual,expected);
    actual=x;inverse.apply(actual.data(),actual.data());const double alias=error(actual,px);
    inverse.set_rank(0);inverse.apply(x.data(),actual.data());b(x.data(),expected.data());const double local=error(actual,expected);
    std::printf("coarse exactness=%.3e linearity=%.3e alias=%.3e rank_zero=%.3e\n",maximum,linear,alias,local);
    if(std::max(std::max(maximum,linear),std::max(alias,local))>1e-12)throw std::runtime_error("coarse inverse identity failed");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
