#include "streaming_qmr.h"
#include <cstdio>
using namespace bem_streaming;
int main()try {
    const int n=43;std::vector<Complex> matrix(n*n),truth(n),b(n),x(n),last(n),tmp(n);
    for(int i=0;i<n;++i){truth[i]=Complex(std::sin(.3*i),std::cos(.7*i));matrix[i*n+i]=Complex(2+.03*i,.2);
        matrix[i*n+(i+1)%n]=Complex(.8,.4);matrix[i*n+(i+9)%n]=Complex(-.1,.2);}
    const Action exact=[&](const Complex* in,Complex* out){for(int i=0;i<n;++i){out[i]=0;for(int j=0;j<n;++j)out[i]+=matrix[i*n+j]*in[j];}};
    const Action inverse=[&](const Complex* in,Complex* out){for(int i=0;i<n;++i)out[i]=in[i]/Complex(1+.03*i,.1*i);};
    const Action inverseH=[&](const Complex* in,Complex* out){for(int i=0;i<n;++i)out[i]=in[i]/std::conj(Complex(1+.03*i,.1*i));};
    exact(truth.data(),b.data());
    for(bool mixed:{false,true})for(int period:{16,128})for(int warm:{0,1,2}) {
        int f=0,h=0,c=0;
        const Action a=[&](const Complex* in,Complex* out){++f;exact(in,out);if(mixed)for(int i=0;i<n;++i)out[i]+=Complex(2e-4,-3e-4)*in[i];};
        const Action ah=[&](const Complex* in,Complex* out){++h;for(int i=0;i<n;++i){out[i]=0;for(int j=0;j<n;++j)out[i]+=std::conj(matrix[j*n+i])*in[j];
            if(mixed)out[i]+=Complex(2e-4,3e-4)*in[i];}};
        const Action strict=[&](const Complex* in,Complex* out){++c;std::copy(in,in+n,last.begin());exact(in,out);};
        for(int i=0;i<n;++i)x[i]=(warm==2?1.:.6)*truth[i];
        auto r=qmr(a,ah,strict,inverse,inverseH,b.data(),x.data(),n,1e-11,800,false,period,warm!=0);
        if(r.iterations!=f+h || r.forward_actions!=f || r.adjoint_actions!=h || r.strict_checks!=c || r.residual>1e-11 || x!=last)return 1;
        for(int i=0;i<n;++i)tmp[i]=x[i]-truth[i];if(norm(tmp.data(),n)/norm(truth.data(),n)>1e-10)return 2;
        if(warm==2 && (f || h || c!=1))return 3;
        std::printf("QMR mixed=%d period=%d warm=%d forward=%d adjoint=%d strict=%d residual=%.3e\n",mixed,period,warm,f,h,c,r.residual);
    }
    const Action zero=[&](const Complex*,Complex* out){std::fill(out,out+n,Complex(0));};
    auto broken=qmr(zero,zero,exact,inverse,inverseH,b.data(),x.data(),n,1e-11,100,false);
    if(broken.residual<=1e-11 || broken.iterations>3)return 4;
    for(int limit:{2,3,4,7}) {auto r=qmr(exact,exact,exact,inverse,inverseH,b.data(),x.data(),n,1e-30,limit,false);
        if(r.iterations>limit || !r.strict_checks)return 5;}
    std::fill(b.begin(),b.end(),Complex(0));auto r=qmr(exact,exact,exact,inverse,inverseH,b.data(),x.data(),n,1e-11,8,false);
    if(r.iterations || r.residual || norm(x.data(),n))return 6;
    std::puts("Complex QMR and fixed-right adjoint preconditioner tests passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 9;}
