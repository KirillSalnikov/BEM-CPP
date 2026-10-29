#pragma once
#include "streaming_host_gmres.h"
namespace bem_streaming {
struct QmrResult:HostResult {int forward_actions=0,adjoint_actions=0;double adjoint_seconds=0;};
// Complex two-sided Lanczos QMR without look-ahead (SIAM Templates, Fig. 2.8).
// Apply to B=A P, B^H=P^H A^H; retain physical solution directions P p.
// Counter/limit count BOTH forward and adjoint A calls, not Lanczos cycles.
inline QmrResult qmr(const Action& action,const Action& adjoint,const Action& strict,
    const Action& inverse,const Action& inverse_adjoint,const Complex* b,Complex* x,
    int n,double tol,int limit,bool verbose=true,int verification_interval=128,bool warm=false) {
    if(n<1 || !std::isfinite(tol) || tol<=0 || limit<2 || verification_interval<2)
        throw std::invalid_argument("invalid QMR dimensions");
    const auto start=std::chrono::steady_clock::now();QmrResult out;
    const double bn=norm(b,n);if(!warm)std::fill(x,x+n,Complex(0));
    if(bn==0){std::fill(x,x+n,Complex(0));out.residual=0;return out;}
    std::vector<Complex> r(b,b+n),vt(n),wt(n),v(n),w(n),p(n),q(n),ap(n),z(n),tmp(n),d(n),s(n);
    int since=0,breakdowns=0;
    const auto check=[&](){const auto t=std::chrono::steady_clock::now();strict(x,tmp.data());
        out.strict_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();++out.strict_checks;since=0;
#pragma omp parallel for
        for(int j=0;j<n;++j)r[j]=b[j]-tmp[j];
        out.residual=norm(r.data(),n)/bn;
        if(!std::isfinite(out.residual))throw std::runtime_error("nonfinite QMR strict residual");
        if(verbose)std::printf("  [QMR] full FP64 residual %.6e\n",out.residual);
    };
    const auto precondition=[&](const Action& f,const Complex* in,Complex* result){const auto t=std::chrono::steady_clock::now();f(in,result);
        out.inverse_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();};
    if(warm)check();
    while(out.residual>tol && out.iterations+2<=limit && breakdowns<3) {
        vt=r;wt=r;double rho=norm(vt.data(),n),xi=rho,gamma=1,theta=0;
        Complex eta=-1,previous_epsilon=1;bool first=true;
        while(out.iterations+2<=limit) {
            if(rho<1e-300 || xi<1e-300 || !std::isfinite(rho+xi)){check();++breakdowns;break;}
#pragma omp parallel for
            for(int j=0;j<n;++j){v[j]=vt[j]/rho;w[j]=wt[j]/xi;}
            const Complex delta=dot(w.data(),v.data(),n);
            if(std::abs(delta)<1e-300 || !std::isfinite(std::abs(delta))){check();++breakdowns;break;}
            const Complex cp=xi*delta/previous_epsilon,cq=std::conj(rho*delta/previous_epsilon);
#pragma omp parallel for
            for(int j=0;j<n;++j){p[j]=first?v[j]:v[j]-cp*p[j];q[j]=first?w[j]:w[j]-cq*q[j];}
            precondition(inverse,p.data(),z.data());
            auto t=std::chrono::steady_clock::now();action(z.data(),ap.data());
            out.action_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
            ++out.forward_actions;++out.iterations;++since;
            const Complex epsilon=dot(q.data(),ap.data(),n),beta=epsilon/delta;
            if(std::abs(beta)<1e-300 || !std::isfinite(std::abs(beta))){check();++breakdowns;break;}
            t=std::chrono::steady_clock::now();adjoint(q.data(),tmp.data());
            out.adjoint_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
            ++out.adjoint_actions;++out.iterations;++since;
            precondition(inverse_adjoint,tmp.data(),wt.data());
#pragma omp parallel for
            for(int j=0;j<n;++j){vt[j]=ap[j]-beta*v[j];wt[j]-=std::conj(beta)*w[j];}
            const double next_rho=norm(vt.data(),n),next_xi=norm(wt.data(),n);
            const double next_theta=next_rho/(gamma*std::abs(beta)),next_gamma=1./std::hypot(1.,next_theta);
            const Complex next_eta=-eta*rho*next_gamma*next_gamma/(beta*gamma*gamma);
            const double carry=std::pow(theta*next_gamma,2);
            if(!std::isfinite(std::abs(next_eta)) || !std::isfinite(carry)){check();++breakdowns;break;}
#pragma omp parallel for
            for(int j=0;j<n;++j){d[j]=next_eta*z[j]+(first?Complex(0):carry*d[j]);s[j]=next_eta*ap[j]+(first?Complex(0):carry*s[j]);
                x[j]+=d[j];r[j]-=s[j];}
            eta=next_eta;theta=next_theta;gamma=next_gamma;rho=next_rho;xi=next_xi;previous_epsilon=epsilon;first=false;
            const double recursive=norm(r.data(),n)/bn;
            if(verbose)std::printf("  [QMR actions %d=%d+%d] recursive %.6e elapsed_s=%.3f\n",out.iterations,out.forward_actions,out.adjoint_actions,
                recursive,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            if(!std::isfinite(recursive) || recursive<=tol*.5 || since>=verification_interval || out.iterations+2>limit){check();break;}
        }
    }
    if(since || !out.strict_checks)check();
    out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return out;
}
}
