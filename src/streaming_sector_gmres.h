#ifndef BEM_STREAMING_SECTOR_GMRES_H
#define BEM_STREAMING_SECTOR_GMRES_H
#include "streaming_host_gmres.h"
#include <array>

namespace bem_streaming {
struct SectorResult:HostResult {std::array<int,12> sector_iterations{};};
// All sectors share one batched operator call, but retain independent Arnoldi
// coefficients and stopping tests. The action is already right-preconditioned
// in unitary symmetry coordinates. Small symmetry defects are corrected by
// full residual replacements; only the original strict action accepts a solve.
inline SectorResult sector_gmres(const Action& action,const Action& strict,
        const Complex* b,Complex* x,const std::array<int,13>& offsets,
        int restart,double tol,int limit,bool verbose=true,int strict_interval=4) {
    const auto start=std::chrono::steady_clock::now();const int n=offsets[12];
    if(n<1 || offsets[0]!=0 || restart<1 || limit<1 || tol<=0 || !std::isfinite(tol) || strict_interval<1)
        throw std::invalid_argument("sector GMRES dimensions");
    for(int s=0;s<12;++s)if(offsets[s]>offsets[s+1])throw std::invalid_argument("sector offsets");
    restart=std::min(restart,limit);SectorResult result;
    std::fill(x,x+n,Complex(0));const double bn=norm(b,n);
    if(bn==0){result.residual=0;return result;}
    std::vector<Complex> basis(size_t(n)*(restart+1)),r(b,b+n),w(n),action_input(n);
    struct State {std::vector<Complex> h,c,s,g,y,rc;int used=0;bool active=false;double projected=0;};
    std::array<State,12> states;
    for(auto& q:states){q.h.resize(size_t(restart+1)*restart);q.c.resize(restart);q.s.resize(restart);q.g.resize(restart+1);q.y.resize(restart);q.rc.resize(restart+1);}
    const double target=tol*bn/std::sqrt(12.);int cycles=0;
    while(result.iterations<limit && result.residual>tol) {
        for(int s=0;s<12;++s) {
            auto& q=states[s];const int lo=offsets[s],hi=offsets[s+1];
            const double rn=norm(r.data()+lo,hi-lo);q.used=0;q.active=rn>.5*target;q.projected=rn;
            std::fill(q.h.begin(),q.h.end(),Complex(0));std::fill(q.g.begin(),q.g.end(),Complex(0));q.g[0]=rn;
            for(int i=lo;i<hi;++i)basis[i]=q.active?r[i]/rn:Complex(0);
        }
        for(int col=0;col<restart && result.iterations<limit;++col) {
            bool any=false;for(const auto& q:states)any=any||q.active;if(!any)break;
            for(int s=0;s<12;++s)for(int i=offsets[s];i<offsets[s+1];++i)
                action_input[i]=states[s].active?basis[size_t(col)*n+i]:Complex(0);
            const auto before=std::chrono::steady_clock::now();action(action_input.data(),w.data());
            const auto after=std::chrono::steady_clock::now();result.action_seconds+=std::chrono::duration<double>(after-before).count();
            // Parallelize across sectors, not individual dot products; each
            // thread owns its Hessenberg block and a disjoint vector segment.
#pragma omp parallel for schedule(static)
            for(int s=0;s<12;++s) {
                auto& q=states[s];const int lo=offsets[s],hi=offsets[s+1];
                Complex* next_basis=basis.data()+size_t(col+1)*n;
                if(!q.active){std::fill(next_basis+lo,next_basis+hi,Complex(0));continue;}
                for(int pass=0;pass<2;++pass)for(int row=0;row<=col;++row) {
                    const Complex* v=basis.data()+size_t(row)*n;Complex a=0;
                    for(int i=lo;i<hi;++i)a+=std::conj(v[i])*w[i];
                    q.h[size_t(row)*restart+col]+=a;for(int i=lo;i<hi;++i)w[i]-=a*v[i];
                }
                double squared=0;for(int i=lo;i<hi;++i)squared+=std::norm(w[i]);const double next=std::sqrt(squared);
                q.h[size_t(col+1)*restart+col]=next;
                for(int i=lo;i<hi;++i)next_basis[i]=next>1e-300?w[i]/next:Complex(0);
                for(int row=0;row<col;++row) {
                    const Complex a=q.h[size_t(row)*restart+col],bb=q.h[size_t(row+1)*restart+col];
                    q.h[size_t(row)*restart+col]=std::conj(q.c[row])*a+std::conj(q.s[row])*bb;
                    q.h[size_t(row+1)*restart+col]=-q.s[row]*a+q.c[row]*bb;
                }
                const Complex a=q.h[size_t(col)*restart+col],bb=q.h[size_t(col+1)*restart+col];
                const double d=std::hypot(std::abs(a),std::abs(bb));
                q.c[col]=d>1e-300?a/d:Complex(1);q.s[col]=d>1e-300?bb/d:Complex(0);
                q.h[size_t(col)*restart+col]=std::conj(q.c[col])*a+std::conj(q.s[col])*bb;
                q.h[size_t(col+1)*restart+col]=0;
                q.g[col+1]=-q.s[col]*q.g[col];q.g[col]=std::conj(q.c[col])*q.g[col];
                q.used=col+1;q.projected=std::abs(q.g[col+1]);++result.sector_iterations[s];
                if(q.projected<.5*target || next<1e-300)q.active=false;
            }
            ++result.iterations;result.orthogonalization_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-after).count();
            double projected2=0;for(const auto& q:states)projected2+=q.projected*q.projected;
            if(verbose)std::printf("  [T sector GMRES %d] projected %.6e elapsed_s=%.3f\n",result.iterations,std::sqrt(projected2)/bn,
                std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            // Match the adaptive global GMRES acceptance attempt. Waiting
            // for every individual sector to hit its budget can oversolve
            // the complete packed RHS. The strict gate still decides.
            if(std::sqrt(projected2)<=.5*tol*bn)break;
        }
        double projected2=0;bool any_active=false;
        for(const auto& q:states){projected2+=q.projected*q.projected;any_active=any_active||q.active;}
        const bool verify=std::sqrt(projected2)<=tol*bn || result.iterations>=limit || ++cycles>=strict_interval || !any_active;
        for(int s=0;s<12;++s) {
            auto& q=states[s];const int used=q.used;
            for(int row=used-1;row>=0;--row) {
                Complex v=q.g[row];for(int col=row+1;col<used;++col)v-=q.h[size_t(row)*restart+col]*q.y[col];
                if(std::abs(q.h[size_t(row)*restart+row])<1e-300)throw std::runtime_error("singular sector Arnoldi projection");
                q.y[row]=v/q.h[size_t(row)*restart+row];
            }
            if(!verify) {
                std::fill(q.rc.begin(),q.rc.end(),Complex(0));q.rc[used]=q.g[used];
                for(int row=used-1;row>=0;--row){const Complex a=q.rc[row],bb=q.rc[row+1];q.rc[row]=q.c[row]*a-std::conj(q.s[row])*bb;q.rc[row+1]=q.s[row]*a+std::conj(q.c[row])*bb;}
            }
            for(int i=offsets[s];i<offsets[s+1];++i) {
                Complex update=0,residual=0;for(int col=0;col<used;++col){update+=basis[size_t(col)*n+i]*q.y[col];if(!verify)residual+=basis[size_t(col)*n+i]*q.rc[col];}
                x[i]+=update;if(!verify && used)r[i]=residual+basis[size_t(used)*n+i]*q.rc[used];
            }
        }
        if(verify) {
            const auto before=std::chrono::steady_clock::now();strict(x,w.data());
            result.strict_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();++result.strict_checks;cycles=0;
            for(int i=0;i<n;++i)r[i]=b[i]-w[i];
            result.residual=norm(r.data(),n)/bn;
            if(!std::isfinite(result.residual))throw std::runtime_error("nonfinite sector residual");
            if(verbose)std::printf("  [sector GMRES] full FP64 residual %.6e\n",result.residual);
        }else ++result.deferred_strict_checks;
    }
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return result;
}
}
#endif
