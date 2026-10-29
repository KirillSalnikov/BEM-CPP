#include "streaming_host_gmres.h"
#include <cstdio>

using namespace bem_streaming;
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
int main() try {
    const int n=41;
    std::vector<Complex> matrix(n*n,0.),expected(n),b(n),x(n),previous(n),last_verified(n),product(n);
    for(int i=0;i<n;++i) {
        expected[i]=Complex(std::sin(.31*i),std::cos(.27*i));
        matrix[i*n+i]=Complex(.9+.07*i,.05*i);
        if(i+1<n)matrix[i*n+i+1]=Complex(.4,.17);
        if(i>0)matrix[i*n+i-1]=Complex(-.12,.09);
    }
    const Action exact=[&](const Complex* v,Complex* w) {
        for(int i=0;i<n;++i) {w[i]=0.;for(int j=0;j<n;++j)w[i]+=matrix[i*n+j]*v[j];}
    };
    // A rounded action is not exactly linear, like the FP32 FMM inner path.
    const Action rounded=[&](const Complex* v,Complex* w) {
        for(int i=0;i<n;++i) {
            std::complex<float> sum=0.;
            for(int j=0;j<n;++j)sum+=std::complex<float>(matrix[i*n+j])*std::complex<float>(v[j]);
            w[i]=Complex(sum);
        }
    };
    const Action inverse=[&](const Complex* v,Complex* w){for(int i=0;i<n;++i)w[i]=v[i]/Complex(1.+.02*i,.1);};
    int calls=0;
    const Action strict=[&](const Complex* v,Complex* w) {
        ++calls;std::copy(v,v+n,last_verified.begin());exact(v,w);
    };
    const auto verify_return=[&](const HostResult& result) {
        require(result.strict_checks==calls && calls>0,"wrong strict action count");
        require(x==last_verified,"returned solution was changed after its last strict check");
        exact(x.data(),product.data());for(int i=0;i<n;++i)product[i]=b[i]-product[i];
        require(std::abs(norm(product.data(),n)/norm(b.data(),n)-result.residual)<1e-14,"returned residual was not strict");
    };
    exact(expected.data(),b.data());
    setenv("BEM_MIXED_ADAPTIVE_INNER","1",1);
    // A nonconverged exit in the middle of a restart must still be verified.
    for(int interval:{1,4}) {
        setenv("BEM_HOST_GMRES_STRICT_INTERVAL",std::to_string(interval).c_str(),1);calls=0;
        const auto result=gmres(exact,strict,inverse,b.data(),x.data(),n,2,1e-30,7,false);
        verify_return(result);
        require(result.iterations==7 && result.residual>1e-30,"iteration limit changed");
        if(interval==1) {previous=x;require(calls==4,"legacy checkpoint count changed");}
        else {
            require(calls==1 && result.deferred_strict_checks==3,"unnecessary strict restarts retained");
            for(int i=0;i<n;++i)product[i]=x[i]-previous[i];
            require(norm(product.data(),n)/norm(x.data(),n)<1e-12,"complex restart recurrence is wrong");
        }
        std::printf("limit exit: interval=%d iterations=%d strict=%d deferred=%d residual=%.3e\n",
            interval,result.iterations,calls,result.deferred_strict_checks,result.residual);
    }
    for(int interval:{1,2,4,8,16})for(bool mixed:{false,true})for(int warm:{0,1,2}) {
        setenv("BEM_HOST_GMRES_STRICT_INTERVAL",std::to_string(interval).c_str(),1);calls=0;
        for(int i=0;i<n;++i)x[i]=(warm==2?1.:.8)*expected[i];
        const auto result=gmres(mixed?rounded:exact,strict,inverse,b.data(),x.data(),n,3,1e-10,500,false,warm!=0);
        verify_return(result);
        require(result.residual<=1e-10,"deferred solve did not converge");
        if(warm==2)require(result.iterations==0 && calls==1,"exact warm start was iterated");
        std::printf("convergence: interval=%d mixed=%d warm=%d iterations=%d strict=%d deferred=%d residual=%.3e\n",
            interval,mixed,warm,result.iterations,calls,result.deferred_strict_checks,result.residual);
    }
    // A zero inner residual can still be a false convergence in mixed mode.
    const Action identity=[&](const Complex* v,Complex* w){std::copy(v,v+n,w);};
    const Action scaled=[&](const Complex* v,Complex* w){++calls;for(int i=0;i<n;++i)w[i]=Complex(1.01,.005)*v[i];};
    calls=0;
    const auto false_inner=gmres(identity,scaled,identity,b.data(),x.data(),n,3,1e-10,100,false);
    require(false_inner.residual<=1e-10 && calls>=5,"inner residual bypassed FP64 acceptance");
    for(const char* bad:{"0","17","4x"}) {
        setenv("BEM_HOST_GMRES_STRICT_INTERVAL",bad,1);bool rejected=false;
        try {gmres(exact,strict,inverse,b.data(),x.data(),n,3,1e-10,100,false);}
        catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"invalid strict interval accepted");
    }
    unsetenv("BEM_HOST_GMRES_STRICT_INTERVAL");
    std::printf("deferred strict checks: complex recurrence, FP32 refinement, warm starts and iteration limit passed\n");
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
