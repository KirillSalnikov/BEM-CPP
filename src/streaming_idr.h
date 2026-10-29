#ifndef BEM_STREAMING_IDR_H
#define BEM_STREAMING_IDR_H
#include "streaming_host_gmres.h"
#include <random>

namespace bem_streaming {
// Experimental matrix-free, fixed-right-preconditioned IDR(s)-biortho.
// Algorithm: van Gijzen & Sonneveld, ACM TOMS 38 (2011), Algorithm 913,
// doi:10.1145/2049662.2049667. Independent implementation of the recurrence.
// Every reported iteration is ONE operator action (not an s+1 action cycle).
// A true residual replaces the recursive residual at acceptance attempts,
// at the action limit, and periodically; replacement restarts the recurrence.
inline HostResult idr(const Action& action,const Action& strict,const Action& inverse,
    const Complex* b,Complex* x,int n,int shadow_size,double tol,int limit,
    bool verbose=true,int verification_interval=128) {
    if(n<1 || shadow_size<1 || shadow_size>std::min(n,16) || tol<=0 || limit<1 || verification_interval<1)
        throw std::invalid_argument("invalid IDR dimensions");
    const auto start=std::chrono::steady_clock::now();
    const int s=shadow_size;
    HostResult result;
    const double bn=norm(b,n);
    std::fill(x,x+n,Complex(0));
    if(bn==0){result.residual=0;return result;}
    std::vector<Complex> p(size_t(n)*s),u(size_t(n)*s),g(size_t(n)*s),m(s*s),
        f(s),c(s),r(b,b+n),v(n),z(n),w(n);
    std::mt19937 random(73129);std::normal_distribution<double> normal;
    for(int col=0;col<s;++col) {
        Complex* q=p.data()+size_t(col)*n;
        for(int i=0;i<n;++i)q[i]=col==0?r[i]:Complex(normal(random),normal(random));
        for(int pass=0;pass<2;++pass)for(int j=0;j<col;++j) {
            const Complex* previous=p.data()+size_t(j)*n;
            const Complex a=dot(previous,q,n);
#pragma omp parallel for
            for(int i=0;i<n;++i)q[i]-=a*previous[i];
        }
        const double length=norm(q,n);
        if(length<1e-300)throw std::runtime_error("IDR shadow breakdown");
#pragma omp parallel for
        for(int i=0;i<n;++i)q[i]/=length;
    }
    Complex omega=1.;int since_strict=0,breakdowns=0;
    const auto reset=[&]() {
        std::fill(u.begin(),u.end(),Complex(0));std::fill(g.begin(),g.end(),Complex(0));
        std::fill(m.begin(),m.end(),Complex(0));for(int i=0;i<s;++i)m[i*s+i]=1.;
        omega=1.;
    };
    const auto verify=[&]() {
        const auto t=std::chrono::steady_clock::now();strict(x,w.data());
        result.strict_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        ++result.strict_checks;since_strict=0;
#pragma omp parallel for
        for(int i=0;i<n;++i)r[i]=b[i]-w[i];
        result.residual=norm(r.data(),n)/bn;
        if(!std::isfinite(result.residual))throw std::runtime_error("nonfinite IDR strict residual");
        if(verbose)std::printf("  [IDR(%d)] full FP64 residual %.6e\n",s,result.residual);
        reset();
    };
    const auto precondition=[&](const Complex* a,Complex* out) {
        const auto t=std::chrono::steady_clock::now();inverse(a,out);
        result.inverse_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
    };
    const auto multiply=[&](const Complex* a,Complex* out) {
        const auto t=std::chrono::steady_clock::now();action(a,out);
        result.action_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
        ++result.iterations;++since_strict;
    };
    const auto checkpoint=[&]() {
        const double recursive=norm(r.data(),n)/bn;
        if(verbose)std::printf("  [IDR(%d) action %d] recursive %.6e\n",s,result.iterations,recursive);
        if(!std::isfinite(recursive) || recursive<=tol*.5 || since_strict>=verification_interval || result.iterations>=limit) {
            verify();return true;
        }
        return false;
    };
    reset();
    while(result.iterations<limit && result.residual>tol) {
        for(int i=0;i<s;++i)f[i]=dot(p.data()+size_t(i)*n,r.data(),n);
        bool restarted=false;
        for(int k=0;k<s && result.iterations<limit;++k) {
            // M is lower triangular by biorthogonality. Only its trailing
            // block is needed for a vector in the next Sonneveld space.
            bool bad=false;
            for(int i=k;i<s;++i) {
                Complex value=f[i];for(int j=k;j<i;++j)value-=m[i*s+j]*c[j];
                if(std::abs(m[i*s+i])<1e-300){bad=true;break;}
                c[i]=value/m[i*s+i];
            }
            if(bad){verify();++breakdowns;restarted=true;break;}
#pragma omp parallel for
            for(int i=0;i<n;++i) {
                Complex value=r[i];for(int j=k;j<s;++j)value-=g[size_t(j)*n+i]*c[j];v[i]=value;
            }
            precondition(v.data(),z.data());
#pragma omp parallel for
            for(int i=0;i<n;++i) {
                Complex value=omega*z[i];for(int j=k;j<s;++j)value+=u[size_t(j)*n+i]*c[j];z[i]=value;
            }
            Complex* gk=g.data()+size_t(k)*n;Complex* uk=u.data()+size_t(k)*n;
            multiply(z.data(),gk);
            for(int j=0;j<k;++j) {
                const Complex a=dot(p.data()+size_t(j)*n,gk,n)/m[j*s+j];
                const Complex* gj=g.data()+size_t(j)*n;const Complex* uj=u.data()+size_t(j)*n;
#pragma omp parallel for
                for(int i=0;i<n;++i){gk[i]-=a*gj[i];z[i]-=a*uj[i];}
            }
            std::copy(z.begin(),z.end(),uk);
            for(int j=k;j<s;++j)m[j*s+k]=dot(p.data()+size_t(j)*n,gk,n);
            if(std::abs(m[k*s+k])<1e-300){verify();++breakdowns;restarted=true;break;}
            const Complex beta=f[k]/m[k*s+k];
#pragma omp parallel for
            for(int i=0;i<n;++i){x[i]+=beta*uk[i];r[i]-=beta*gk[i];}
            if(checkpoint()){restarted=true;break;}
            for(int j=k+1;j<s;++j)f[j]-=beta*m[j*s+k];
        }
        if(breakdowns>=3)break;
        if(restarted)continue;
        precondition(r.data(),z.data());multiply(z.data(),v.data());
        const double vn=norm(v.data(),n),rn=norm(r.data(),n);
        const Complex a=dot(v.data(),r.data(),n);
        if(vn<1e-300 || std::abs(a)<1e-300){verify();++breakdowns;continue;}
        omega=a/(vn*vn);
        const double cosine=std::abs(a)/(vn*rn);
        if(cosine<.7)omega*=.7/cosine;
#pragma omp parallel for
        for(int i=0;i<n;++i){x[i]+=omega*z[i];r[i]-=omega*v[i];}
        checkpoint();
    }
    if(since_strict)verify();
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return result;
}
}
#endif
