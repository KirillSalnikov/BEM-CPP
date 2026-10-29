#include "streaming_tfqmr.h"
#include <cstdio>
using namespace bem_streaming;
int main()try {
    const int n=47;std::vector<Complex> truth(n),b(n),x(n),last(n),work(n);
    for(int i=0;i<n;++i)truth[i]=Complex(std::sin(.17*i),std::cos(.31*i));
    const Action exact=[&](const Complex* a,Complex* out){for(int i=0;i<n;++i)
        out[i]=Complex(1.+.04*i,.1*std::cos(i))*a[i]+Complex(.7,.3)*a[(i+1)%n]-.15*a[(i+7)%n];};
    const Action mixed=[&](const Complex* a,Complex* out){exact(a,out);for(int i=0;i<n;++i)out[i]+=Complex(2e-4,-3e-4)*a[i];};
    const Action local=[&](const Complex* a,Complex* out){for(int i=0;i<n;++i)out[i]=a[i]/Complex(1.+.04*i,.1);};
    int calls=0,actions=0;
    const Action strict=[&](const Complex* a,Complex* out){++calls;std::copy(a,a+n,last.begin());exact(a,out);};
    exact(truth.data(),b.data());
    for(int period:{8,128})for(bool perturbed:{false,true})for(int warm:{0,1,2}) {
        calls=actions=0;for(int i=0;i<n;++i)x[i]=(warm==2?1.:.7)*truth[i];
        const Action action=[&](const Complex* a,Complex* out){++actions;(perturbed?mixed:exact)(a,out);};
        auto r=tfqmr(action,strict,local,b.data(),x.data(),n,1e-11,600,false,period,warm!=0);
        if(r.iterations!=actions || r.strict_checks!=calls || x!=last || r.residual>1e-11)return 1;
        exact(x.data(),work.data());for(int i=0;i<n;++i)work[i]=b[i]-work[i];
        if(std::abs(norm(work.data(),n)/norm(b.data(),n)-r.residual)>1e-14)return 2;
        for(int i=0;i<n;++i)work[i]=x[i]-truth[i];
        if(norm(work.data(),n)/norm(truth.data(),n)>2e-10)return 3;
        if(warm==2 && (actions || calls!=1))return 4;
        std::printf("period=%d mixed=%d warm=%d actions=%d strict=%d residual=%.3e\n",period,perturbed,warm,actions,calls,r.residual);
    }
    for(int limit:{1,2,3,7}) {
        calls=actions=0;const Action action=[&](const Complex* a,Complex* out){++actions;exact(a,out);};
        auto r=tfqmr(action,strict,local,b.data(),x.data(),n,1e-30,limit,false);
        if(actions!=limit || r.iterations!=limit || calls!=1 || x!=last)return 5;
    }
    const Action zero=[&](const Complex*,Complex* out){std::fill(out,out+n,Complex(0));};
    auto broken=tfqmr(zero,zero,local,b.data(),x.data(),n,1e-11,100,false);
    if(broken.residual<=1e-11 || broken.iterations>3 || !broken.strict_checks)return 6;
    std::fill(b.begin(),b.end(),Complex(0));
    auto trivial=tfqmr(exact,strict,local,b.data(),x.data(),n,1e-11,100,false);
    if(trivial.iterations || trivial.residual || norm(x.data(),n))return 7;
    std::printf("TFQMR complex nonnormal, mixed refinement, warm start, limits, breakdown and zero RHS passed\n");
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 9;}
