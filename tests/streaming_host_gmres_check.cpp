#include "streaming_host_gmres.h"
#include <cstdio>
int main() {
    using namespace bem_streaming;
    const int n=17;
    std::vector<Complex> matrix(n*n),b(n),x(n),expected(n);
    for(int i=0;i<n;++i) {
        expected[i]=Complex(std::sin(.7*i),std::cos(.3*i));
        for(int j=0;j<n;++j) matrix[i*n+j]=i==j?Complex(2.+.1*i,.2):Complex(.03*std::sin(i+2*j),.02*std::cos(2*i+j));
    }
    const Action exact=[&](const Complex* x,Complex* y) {
        for(int i=0;i<n;++i) {y[i]=0;for(int j=0;j<n;++j)y[i]+=matrix[i*n+j]*x[j];}
    };
    const Action mixed=[&](const Complex* x,Complex* y) {
        exact(x,y);for(int i=0;i<n;++i)y[i]+=1e-4*Complex(.3,.2)*x[i];
    };
    const Action inverse=[&](const Complex* x,Complex* y) {for(int i=0;i<n;++i)y[i]=x[i]/matrix[i*n+i];};
    exact(expected.data(),b.data());
    const auto result=gmres(mixed,exact,inverse,b.data(),x.data(),n,4,1e-10,100,false);
    for(int i=0;i<n;++i)x[i]-=expected[i];
    const double error=norm(x.data(),n)/norm(expected.data(),n);
    std::printf("mixed host GMRES: %d iterations, strict residual %.3e, error %.3e\n",result.iterations,result.residual,error);
    return result.residual<1e-10 && error<1e-10?0:1;
}
