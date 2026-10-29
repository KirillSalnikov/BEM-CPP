#include "streaming_host_gmres.h"
#include "streaming_polynomial_inverse.h"
#include "streaming_idr.h"
#include <cstdio>
#include <random>
using namespace bem_streaming;
int main() {
    const int n=31;
    std::vector<Complex> diagonal(n),truth(n),b(n),x(n),u(n),v(n),pu(n),pv(n),sum(n),psum(n);
    for(int i=0;i<n;++i) {
        diagonal[i]=Complex(1.+.05*i,.15*std::sin(i));
        truth[i]=Complex(std::sin(.3*i),std::cos(.7*i));
        u[i]=Complex(std::cos(.4*i),std::sin(.6*i));
        v[i]=Complex(std::cos(.8*i),std::sin(.9*i));
    }
    // Nonnormal complex matrix with a second, deliberately perturbed inner operator.
    const Action exact=[&](const Complex* a,Complex* out){
        for(int i=0;i<n;++i)out[i]=diagonal[i]*a[i]+Complex(.4,.2)*a[(i+1)%n];
    };
    const Action mixed=[&](const Complex* a,Complex* out){
        exact(a,out);for(int i=0;i<n;++i)out[i]+=Complex(2e-4,-3e-4)*a[i];
    };
    const Action local=[&](const Complex* a,Complex* out){for(int i=0;i<n;++i)out[i]=a[i]/diagonal[i];};
    exact(truth.data(),b.data());
    ArnoldiInverse polynomial(n,6,mixed,local);
    polynomial.train(u.data());
    polynomial.apply(u.data(),pu.data());polynomial.apply(v.data(),pv.data());
    const Complex alpha(.3,.7);
    for(int i=0;i<n;++i)sum[i]=u[i]+alpha*v[i];
    polynomial.apply(sum.data(),psum.data());
    for(int i=0;i<n;++i)psum[i]-=pu[i]+alpha*pv[i];
    if(norm(psum.data(),n)>2e-11*norm(pu.data(),n))return 1;
    // Applying a frozen polynomial to its training vector reproduces one GMRES cycle.
    ArnoldiInverse one_cycle(n,6,mixed,local);
    one_cycle.solve(u.data(),x.data(),0);
    for(int i=0;i<n;++i)x[i]-=pu[i];
    if(norm(x.data(),n)>2e-11*norm(pu.data(),n))return 2;
    for(int interval:{1,4})for(int mode:{0,1,2}) {
        setenv("BEM_HOST_GMRES_STRICT_INTERVAL",interval==1?"1":"4",1);
        int calls=0;
        ArnoldiInverse inner(n,3,mixed,local);
        const Action inverse=[&](const Complex* a,Complex* out) {
            ++calls;
            if(mode==0)polynomial.apply(a,out);
            else if(mode==1)inner.solve(a,out,.15);
            else {local(a,out);for(int i=0;i<n;++i)out[i]*=calls%2?.4:1.8;}
        };
        const auto result=gmres(mixed,exact,inverse,b.data(),x.data(),n,4,1e-11,120,false,false,nullptr,true,mode!=0);
        for(int i=0;i<n;++i)x[i]-=truth[i];
        const double error=norm(x.data(),n)/norm(truth.data(),n);
        std::printf("flexible/polynomial interval=%d mode=%d steps=%d inverse_calls=%d residual=%.3e error=%.3e\n",
            interval,mode,result.iterations,calls,result.residual,error);
        if(result.residual>1e-11 || error>2e-11 || result.strict_checks<1)return 3;
        if(mode!=0 && calls!=result.iterations)return 4;
    }
    bool rejected=false;RecycleSpace space(n,2);
    try {gmres(mixed,exact,local,b.data(),x.data(),n,4,1e-10,20,false,false,&space,true,true);}
    catch(const std::invalid_argument&){rejected=true;}
    if(!rejected)return 5;
    // Happy breakdown / a degree-zero inverse polynomial.
    const Action scalar=[](const Complex* a,Complex* out){for(int i=0;i<31;++i)out[i]=Complex(2,.3)*a[i];};
    const Action identity=[](const Complex* a,Complex* out){std::copy(a,a+31,out);};
    ArnoldiInverse trivial(n,8,scalar,identity);trivial.train(u.data());trivial.apply(v.data(),x.data());
    for(int i=0;i<n;++i)x[i]-=v[i]/Complex(2,.3);
    if(trivial.degree()!=1 || norm(x.data(),n)>1e-12)return 6;
    for(int s:{1,4,8})for(int period:{8,128}) {
        const auto result=idr(mixed,exact,local,b.data(),x.data(),n,s,1e-11,400,false,period);
        for(int i=0;i<n;++i)x[i]-=truth[i];
        const double error=norm(x.data(),n)/norm(truth.data(),n);
        std::printf("IDR(%d) period=%d actions=%d residual=%.3e error=%.3e\n",s,period,result.iterations,result.residual,error);
        if(result.residual>1e-11 || error>2e-11 || result.iterations>400)return 7;
    }
    const auto limited=idr(mixed,exact,local,b.data(),x.data(),n,4,1e-14,7,false);
    exact(x.data(),sum.data());for(int i=0;i<n;++i)sum[i]=b[i]-sum[i];
    if(limited.iterations!=7 || limited.strict_checks!=1 ||
       std::abs(limited.residual-norm(sum.data(),n)/norm(b.data(),n))>1e-14)return 8;
    std::fill(b.begin(),b.end(),Complex(0));
    const auto zero=idr(mixed,exact,local,b.data(),x.data(),n,4,1e-11,7,false);
    if(zero.iterations || zero.residual || norm(x.data(),n))return 9;
    return 0;
}
