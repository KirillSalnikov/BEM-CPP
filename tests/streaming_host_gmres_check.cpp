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
    for(int adaptive: {0,1}) {
        setenv("BEM_MIXED_ADAPTIVE_INNER",adaptive?"1":"0",1);
        for(int warm: {0,1,2}) {
            for(int i=0;i<n;++i)x[i]=warm==2?expected[i]:.9*expected[i];
            const auto result=gmres(mixed,exact,inverse,b.data(),x.data(),n,4,1e-10,100,false,warm!=0);
            for(int i=0;i<n;++i)x[i]-=expected[i];
            const double error=norm(x.data(),n)/norm(expected.data(),n);
            std::printf("mixed host GMRES adaptive=%d warm=%d: %d iterations, strict %.3e, error %.3e\n",
                        adaptive,warm,result.iterations,result.residual,error);
            if(result.residual>=1e-10 || error>=1e-10 || result.strict_checks<1)return 1;
            if(warm==2 && (result.iterations!=0 || result.strict_checks!=1))return 1;
        }
    }
    std::fill(b.begin(),b.end(),Complex(0));
    std::fill(x.begin(),x.end(),Complex(3));
    const auto zero=gmres(mixed,exact,inverse,b.data(),x.data(),n,4,1e-10,100,false,true);
    return zero.residual==0 && zero.iterations==0 && norm(x.data(),n)==0?0:1;
}
