#ifndef BEM_STREAMING_HOST_GMRES_H
#define BEM_STREAMING_HOST_GMRES_H
#include "streaming_recycle_space.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

namespace bem_streaming {
struct HostResult {
    int iterations=0, strict_checks=0, recycle_rank=0, recycle_fallbacks=0;
    int deferred_strict_checks=0;
    double recycle_seconds=0;
    double residual=1,seconds=0,action_seconds=0,inverse_seconds=0,orthogonalization_seconds=0,strict_seconds=0;
};
// Fixed right preconditioning retains only the Arnoldi basis. Flexible
// mode additionally retains each actual inverse image: an inner Krylov
// solve is nonlinear and must never be reapplied to the final combination.
inline HostResult gmres(const Action& action,const Action& strict,const Action& inverse,
                        const Complex* b,Complex* x,int n,int restart,double tol,int limit,
                        bool verbose=true,bool use_initial_guess=false,
                        RecycleSpace* recycle=nullptr,bool use_recycle=true,
                        bool flexible=false) {
    const auto start=std::chrono::steady_clock::now();
    restart=std::min(restart,limit);
    if(n<1||restart<1||limit<1||tol<=0) throw std::invalid_argument("invalid host GMRES dimensions");
    int strict_interval=1;
    if(const char* setting=std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")) {
        char* end=nullptr;const long value=std::strtol(setting,&end,10);
        if(end==setting || *end || value<1 || value>16)
            throw std::invalid_argument("BEM_HOST_GMRES_STRICT_INTERVAL must be in [1,16]");
        strict_interval=int(value);
    }
    if(recycle && recycle->size()!=n)throw std::invalid_argument("recycle space size mismatch");
    if(recycle && recycle->capacity()==0)recycle=nullptr;
    if(flexible && recycle)throw std::invalid_argument("flexible GMRES recycling is not implemented");
    std::vector<Complex> basis,h(size_t(restart+1)*restart),
        c(restart),s(restart),g(restart+1),y(restart),r(b,b+n),z(n),w(n),
        residual_coefficients(restart+1);
    if(!use_initial_guess) std::fill(x,x+n,Complex(0));
    HostResult result;
    int cycles_since_strict=0;
    std::vector<Complex> raw_h(recycle?h.size():0),couplings;
    const double bn=norm(b,n);
    if(bn==0) { std::fill(x,x+n,Complex(0)); result.residual=0; return result; }
    const auto check_strict=[&]() {
        const auto before_strict=std::chrono::steady_clock::now();
        strict(x,w.data());
        result.strict_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before_strict).count();
        ++result.strict_checks;
        cycles_since_strict=0;
#pragma omp parallel for
        for(int i=0;i<n;++i) r[i]=b[i]-w[i];
        result.residual=norm(r.data(),n)/bn;
        if(!std::isfinite(result.residual))throw std::runtime_error("nonfinite strict residual");
    };
    if(use_initial_guess) {
        check_strict();
        if(verbose) std::printf("  [T host GMRES] initial guess FP64 residual %.6e\n",result.residual);
        if(result.residual<=tol) {
            result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            return result;
        }
    }
    basis.resize(size_t(n)*(restart+1));
    std::vector<Complex> inverse_basis(flexible?size_t(n)*restart:0);
    const char* adaptive_setting=std::getenv("BEM_MIXED_ADAPTIVE_INNER");
    const bool adaptive=adaptive_setting && std::strcmp(adaptive_setting,"0")!=0;
    double projected_factor=adaptive?.5:.1;
    while(result.iterations<limit && result.residual>tol) {
        int projected_rank=recycle && use_recycle?recycle->rank():0;
        if(projected_rank) {
            const auto before_recycle=std::chrono::steady_clock::now();
            recycle->correct(x,r.data());
            result.recycle_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before_recycle).count();
            // The stored images approximate A_inner U, not A_strict U.
            // A tiny coarse residual therefore still needs a strict action.
            if(norm(r.data(),n)/bn<=tol*projected_factor) {
                check_strict();
                if(result.residual<=tol)break;
                use_recycle=false;projected_rank=0;++result.recycle_fallbacks;
            }
        }
        const double rn=norm(r.data(),n);
        std::fill(h.begin(),h.end(),Complex(0)); std::fill(g.begin(),g.end(),Complex(0));
        std::fill(raw_h.begin(),raw_h.end(),Complex(0));
        couplings.assign(size_t(projected_rank)*restart,Complex(0));
        g[0]=rn;
#pragma omp parallel for
        for(int i=0;i<n;++i) basis[i]=r[i]/std::max(rn,1e-300);
        int used=0;
        bool projected_target_reached=false;
        for(int col=0;col<restart && result.iterations<limit;++col) {
            const Complex* v=basis.data()+size_t(col)*n;
            const auto before_inverse=std::chrono::steady_clock::now();
            inverse(v,z.data());
            if(flexible)std::copy(z.begin(),z.end(),inverse_basis.data()+size_t(col)*n);
            const auto before_action=std::chrono::steady_clock::now();
            action(z.data(),w.data());
            const auto before_orthogonalization=std::chrono::steady_clock::now();
            result.inverse_seconds+=std::chrono::duration<double>(before_action-before_inverse).count();
            result.action_seconds+=std::chrono::duration<double>(before_orthogonalization-before_action).count();
            if(projected_rank)recycle->project(w.data(),col,restart,couplings);
            for(int pass=0;pass<2;++pass)
                for(int row=0;row<=col;++row) {
                    const Complex* previous=basis.data()+size_t(row)*n;
                    const Complex a=dot(previous,w.data(),n);
                    h[size_t(row)*restart+col]+=a;
#pragma omp parallel for
                    for(int i=0;i<n;++i) w[i]-=a*previous[i];
                }
            const double next=norm(w.data(),n);
            h[size_t(col+1)*restart+col]=next;
            if(recycle)for(int row=0;row<=col+1;++row)
                raw_h[size_t(row)*restart+col]=h[size_t(row)*restart+col];
#pragma omp parallel for
            for(int i=0;i<n;++i) basis[size_t(col+1)*n+i]=next>1e-300?w[i]/next:Complex(0);
            for(int row=0;row<col;++row) {
                const Complex a=h[size_t(row)*restart+col],b=h[size_t(row+1)*restart+col];
                h[size_t(row)*restart+col]=std::conj(c[row])*a+std::conj(s[row])*b;
                h[size_t(row+1)*restart+col]=-s[row]*a+c[row]*b;
            }
            const Complex a=h[size_t(col)*restart+col],b=h[size_t(col+1)*restart+col];
            const double d=std::sqrt(std::norm(a)+std::norm(b));
            c[col]=d>1e-300?a/d:Complex(1);s[col]=d>1e-300?b/d:Complex(0);
            h[size_t(col)*restart+col]=std::conj(c[col])*a+std::conj(s[col])*b;
            h[size_t(col+1)*restart+col]=0;
            g[col+1]=-s[col]*g[col];g[col]=std::conj(c[col])*g[col];
            ++result.iterations;used=col+1;
            const double projected=std::abs(g[col+1])/bn;
            result.orthogonalization_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before_orthogonalization).count();
            if(verbose) {
                const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::printf("  [T host GMRES %d] projected %.6e elapsed_s=%.3f\n",result.iterations,projected,elapsed);
                std::fflush(stdout);
            }
            projected_target_reached=projected<tol*projected_factor;
            if(projected_target_reached || next<1e-300) break;
        }
        for(int row=used-1;row>=0;--row) {
            Complex value=g[row];
            for(int col=row+1;col<used;++col) value-=h[size_t(row)*restart+col]*y[col];
            if(std::abs(h[size_t(row)*restart+row])<1e-300) throw std::runtime_error("singular Arnoldi projection");
            y[row]=value/h[size_t(row)*restart+row];
        }
        // Continue an inner correction through restart boundaries. Its
        // residual is V Q* e_m g_m, obtained by undoing the complex Givens
        // rotations. Only a subsequent full strict action can accept x.
        // At a restart boundary check even the outer tolerance, so deferral
        // cannot miss an already acceptable residual between tol/2 and tol.
        const bool verify_now=std::abs(g[used])/bn<=tol || result.iterations>=limit ||
            used<restart || ++cycles_since_strict>=strict_interval;
        if(!verify_now) {
            std::fill(residual_coefficients.begin(),residual_coefficients.end(),Complex(0));
            residual_coefficients[used]=g[used];
            for(int row=used-1;row>=0;--row) {
                const Complex a=residual_coefficients[row],b=residual_coefficients[row+1];
                residual_coefficients[row]=c[row]*a-std::conj(s[row])*b;
                residual_coefficients[row+1]=s[row]*a+std::conj(c[row])*b;
            }
        }
#pragma omp parallel for
        for(int i=0;i<n;++i) {
            Complex value=0,residual=0;
            for(int col=0;col<used;++col) {
                const Complex v=basis[size_t(col)*n+i];
                value+=(flexible?inverse_basis[size_t(col)*n+i]:v)*y[col];
                if(!verify_now)residual+=v*residual_coefficients[col];
            }
            w[i]=value;
            if(!verify_now)r[i]=residual+basis[size_t(used)*n+i]*residual_coefficients[used];
        }
        const auto before_update=std::chrono::steady_clock::now();
        if(flexible)std::copy(w.begin(),w.end(),z.begin());
        else inverse(w.data(),z.data());
        result.inverse_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before_update).count();
        if(projected_rank)recycle->subtract_coupled_update(z.data(),y.data(),used,couplings,restart,projected_rank);
#pragma omp parallel for
        for(int i=0;i<n;++i) x[i]+=z[i];
        if(verify_now)check_strict();
        else ++result.deferred_strict_checks;
        if(recycle) {
            const auto before_recycle=std::chrono::steady_clock::now();
            recycle->learn(basis,raw_h,restart,used,inverse,couplings,projected_rank);
            result.recycle_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before_recycle).count();
        }
        // Match the device solver's early acceptance attempt, always gated
        // by the original FP64 tolerance. A rejected attempt tightens the
        // next correction to the legacy host target.
        if(adaptive && verify_now && projected_target_reached && result.residual>tol)
            projected_factor=.1;
        if(verbose) {
            if(verify_now)std::printf("  [T host GMRES] full FP64 residual %.6e\n",result.residual);
            else std::printf("  [T host GMRES] inner residual %.6e; FP64 check deferred (%d/%d cycles)\n",
                norm(r.data(),n)/bn,cycles_since_strict,strict_interval);
            std::fflush(stdout);
        }
    }
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    result.recycle_rank=recycle?recycle->rank():0;
    if(verbose) std::printf("  [T host timings] action_s=%.3f inverse_s=%.3f orthogonalization_s=%.3f strict_s=%.3f strict_checks=%d deferred_checks=%d\n",
        result.action_seconds,result.inverse_seconds,result.orthogonalization_seconds,result.strict_seconds,result.strict_checks,result.deferred_strict_checks);
    if(verbose && recycle)std::printf("  [T recycle] rank=%d storage_gib=%.3f update_s=%.3f fallbacks=%d\n",
        result.recycle_rank,recycle->storage_gib(),result.recycle_seconds,result.recycle_fallbacks);
    return result;
}
}
#endif
