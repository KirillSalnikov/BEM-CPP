#ifndef BEM_STREAMING_TFQMR_H
#define BEM_STREAMING_TFQMR_H
#include "streaming_host_gmres.h"

namespace bem_streaming {
// Fixed-right-preconditioned TFQMR: B=A P, with physical updates P d.
// Freund, SIAM J. Sci. Comput. 14 (1993), 470-482. The quasi-residual bound
// is only a trigger: acceptance and every exit use the original strict A.
// FP64 residual replacement restarts the short recurrence. Iterations count
// actual mixed A actions, not the two-step TFQMR outer cycles.
inline HostResult tfqmr(const Action& action,const Action& strict,const Action& inverse,
    const Complex* b,Complex* x,int n,double tol,int limit,bool verbose=true,
    int verification_interval=128,bool use_initial_guess=false) {
    if(n<1 || !std::isfinite(tol) || tol<=0 || limit<1 || verification_interval<1)
        throw std::invalid_argument("invalid TFQMR dimensions");
    const auto start=std::chrono::steady_clock::now();HostResult result;
    const double bn=norm(b,n);
    if(!use_initial_guess)std::fill(x,x+n,Complex(0));
    if(bn==0){std::fill(x,x+n,Complex(0));result.residual=0;return result;}
    std::vector<Complex> r(b,b+n),shadow(n),w(n),y1(n),y2(n),u1(n),u2(n),
        z1(n),z2(n),v(n),d(n),work(n);
    int since_strict=0,breakdowns=0;
    const auto verify=[&]() {
        const auto t=std::chrono::steady_clock::now();strict(x,work.data());
        result.strict_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        ++result.strict_checks;since_strict=0;
#pragma omp parallel for
        for(int i=0;i<n;++i)r[i]=b[i]-work[i];
        result.residual=norm(r.data(),n)/bn;
        if(!std::isfinite(result.residual))throw std::runtime_error("nonfinite TFQMR strict residual");
        if(verbose)std::printf("  [TFQMR] full FP64 residual %.6e\n",result.residual);
    };
    const auto image=[&](const Complex* y,Complex* z,Complex* u) {
        auto t=std::chrono::steady_clock::now();inverse(y,z);
        result.inverse_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        t=std::chrono::steady_clock::now();action(z,u);
        result.action_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        ++result.iterations;++since_strict;
    };
    if(use_initial_guess)verify();
    while(result.iterations<limit && result.residual>tol && breakdowns<3) {
        double tau=norm(r.data(),n),theta=0.;Complex eta=0.,rho=tau;
        if(!(tau>0) || !std::isfinite(tau))break;
        w=r;y1=r;std::fill(d.begin(),d.end(),Complex(0));
#pragma omp parallel for
        for(int i=0;i<n;++i)shadow[i]=r[i]/tau;
        image(y1.data(),z1.data(),u1.data());v=u1;
        int steps=0;bool restart=false;
        while(!restart) {
            const Complex sigma=dot(shadow.data(),v.data(),n);
            if(std::abs(sigma)<1e-300 || !std::isfinite(std::abs(sigma))) {
                verify();++breakdowns;break;
            }
            const Complex alpha=rho/sigma;
            if(std::abs(alpha)<1e-300 || !std::isfinite(std::abs(alpha))) {
                verify();++breakdowns;break;
            }
#pragma omp parallel for
            for(int i=0;i<n;++i)y2[i]=y1[i]-alpha*v[i];
            for(int half=0;half<2;++half) {
                if(half)image(y2.data(),z2.data(),u2.data());
                const Complex* u=half?u2.data():u1.data();
                const Complex* z=half?z2.data():z1.data();
                const Complex coefficient=theta*theta*eta/alpha;
                if(!std::isfinite(std::abs(coefficient))) {verify();++breakdowns;restart=true;break;}
#pragma omp parallel for
                for(int i=0;i<n;++i){w[i]-=alpha*u[i];d[i]=z[i]+coefficient*d[i];}
                const double wn=norm(w.data(),n);
                if(!std::isfinite(wn) || tau<1e-300) {verify();++breakdowns;restart=true;break;}
                theta=wn/tau;const double c=tau/std::hypot(tau,wn);
                tau=wn*c;eta=c*c*alpha;
#pragma omp parallel for
                for(int i=0;i<n;++i)x[i]+=eta*d[i];
                ++steps;
                const double bound=tau*std::sqrt(double(steps+1))/bn;
                if(verbose)std::printf("  [TFQMR action %d] bound %.6e elapsed_s=%.3f\n",result.iterations,bound,
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
                if(!std::isfinite(bound) || bound<=tol*.5 || since_strict>=verification_interval || result.iterations>=limit) {
                    verify();restart=true;break;
                }
            }
            if(restart)break;
            const Complex next_rho=dot(shadow.data(),w.data(),n);
            if(std::abs(next_rho)<1e-300 || !std::isfinite(std::abs(next_rho))) {
                verify();++breakdowns;break;
            }
            const Complex beta=next_rho/rho;rho=next_rho;
#pragma omp parallel for
            for(int i=0;i<n;++i)y1[i]=w[i]+beta*y2[i];
            image(y1.data(),z1.data(),u1.data());
#pragma omp parallel for
            for(int i=0;i<n;++i)v[i]=u1[i]+beta*(u2[i]+beta*v[i]);
        }
    }
    if(since_strict)verify();
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return result;
}
}
#endif
