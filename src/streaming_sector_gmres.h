#ifndef BEM_STREAMING_SECTOR_GMRES_H
#define BEM_STREAMING_SECTOR_GMRES_H
#include "streaming_host_gmres.h"
#include <array>

namespace bem_streaming {
// Separate GCRO spaces for the irreducible symmetry sectors. The stored
// coordinates belong to the fixed, right-preconditioned operator, so a
// caller must discard this object whenever that operator or its basis changes.
class SectorRecycleSpace {
    std::array<int,13> offsets_;
    std::vector<RecycleSpace> spaces_;
public:
    SectorRecycleSpace(const std::array<int,13>& offsets,int capacity,unsigned mask=4095):offsets_(offsets) {
        if(offsets[0]!=0 || capacity<0 || capacity>64 || mask>4095)throw std::invalid_argument("sector recycle dimensions");
        spaces_.reserve(12);
        for(int s=0;s<12;++s) {
            const int n=offsets[s+1]-offsets[s];
            if(n<0)throw std::invalid_argument("sector recycle offsets");
            spaces_.emplace_back(std::max(1,n),n && (mask&(1u<<s))?capacity:0);
        }
    }
    bool matches(const std::array<int,13>& offsets)const{return offsets==offsets_;}
    RecycleSpace& sector(int s){return spaces_.at(s);}
    int merge_from(const SectorRecycleSpace& other){
        if(offsets_!=other.offsets_)throw std::invalid_argument("sector recycle merge offsets");
        int added=0;for(int s=0;s<12;++s)added+=spaces_[s].merge_from(other.spaces_[s]);
        return added;
    }
    int rank()const{int r=0;for(const auto& s:spaces_)r+=s.rank();return r;}
    double storage_gib()const{double b=0;for(const auto& s:spaces_)b+=s.storage_gib();return b;}
};
struct SectorResult:HostResult {std::array<int,12> sector_iterations{};};
// All sectors share one batched operator call, but retain independent Arnoldi
// coefficients and stopping tests. The action is already right-preconditioned
// in unitary symmetry coordinates. Small symmetry defects are corrected by
// full residual replacements; only the original strict action accepts a solve.
inline SectorResult sector_gmres(const Action& action,const Action& strict,
        const Complex* b,Complex* x,const std::array<int,13>& offsets,
        int restart,double tol,int limit,bool verbose=true,int strict_interval=4,
        SectorRecycleSpace* recycle=nullptr,double projected_factor=.5,double first_pass_factor=0) {
    const auto start=std::chrono::steady_clock::now();const int n=offsets[12];
    if(n<1 || offsets[0]!=0 || restart<1 || limit<1 || tol<=0 || !std::isfinite(tol) || strict_interval<1 ||
       !std::isfinite(projected_factor) || projected_factor<=0 || projected_factor>.5 ||
       !std::isfinite(first_pass_factor) || first_pass_factor<0 || first_pass_factor>64)
        throw std::invalid_argument("sector GMRES dimensions");
    for(int s=0;s<12;++s)if(offsets[s]>offsets[s+1])throw std::invalid_argument("sector offsets");
    if(recycle && !recycle->matches(offsets))throw std::invalid_argument("sector recycle layout mismatch");
    restart=std::min(restart,limit);SectorResult result;
    std::fill(x,x+n,Complex(0));const double bn=norm(b,n);
    if(bn==0){result.residual=0;return result;}
    std::vector<Complex> basis(size_t(n)*(restart+1)),r(b,b+n),w(n),action_input(n);
    struct State {std::vector<Complex> h,c,s,g,y,rc,raw_h,couplings;int used=0,rank=0;bool active=false;double projected=0;};
    std::array<State,12> states;
    for(auto& q:states){q.h.resize(size_t(restart+1)*restart);q.c.resize(restart);q.s.resize(restart);q.g.resize(restart+1);q.y.resize(restart);q.rc.resize(restart+1);if(recycle)q.raw_h.resize(q.h.size());}
    const double target=tol*bn/std::sqrt(12.);int cycles=0;bool use_recycle=true;
    double inner_factor=first_pass_factor>0?first_pass_factor:projected_factor;
    const auto check_strict=[&]() {
        const auto before=std::chrono::steady_clock::now();strict(x,w.data());
        result.strict_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();++result.strict_checks;cycles=0;
        for(int i=0;i<n;++i)r[i]=b[i]-w[i];
        result.residual=norm(r.data(),n)/bn;
        if(!std::isfinite(result.residual))throw std::runtime_error("nonfinite sector residual");
        // A coarse first pass is only an inner stopping rule. After the
        // first true residual, every correction uses the original target.
        inner_factor=projected_factor;
        if(verbose)std::printf("  [sector GMRES] full FP64 residual %.6e\n",result.residual);
    };
    while(result.iterations<limit && result.residual>tol) {
        const auto before_recycle=std::chrono::steady_clock::now();
        bool projected=false;
#pragma omp parallel for schedule(static) reduction(||:projected)
        for(int s=0;s<12;++s) {
            auto& q=states[s];q.rank=recycle && use_recycle?recycle->sector(s).rank():0;
            q.couplings.assign(size_t(q.rank)*restart,Complex(0));
            std::fill(q.raw_h.begin(),q.raw_h.end(),Complex(0));
            if(q.rank){recycle->sector(s).correct(x+offsets[s],r.data()+offsets[s]);projected=true;}
        }
        if(recycle)result.recycle_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before_recycle).count();
        // Approximate recycled images cannot accept a solve or make a
        // zero-iteration loop when mixed and strict actions differ.
        if(projected && norm(r.data(),n)<=tol*bn) {
            check_strict();
            if(result.residual<=tol)break;
            use_recycle=false;++result.recycle_fallbacks;
            for(auto& q:states){q.rank=0;q.couplings.clear();}
        }
        for(int s=0;s<12;++s) {
            auto& q=states[s];const int lo=offsets[s],hi=offsets[s+1];
            const double rn=norm(r.data()+lo,hi-lo);q.used=0;q.active=rn>inner_factor*target;q.projected=rn;
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
                if(q.rank)recycle->sector(s).project(w.data()+lo,col,restart,q.couplings);
                for(int pass=0;pass<2;++pass)for(int row=0;row<=col;++row) {
                    const Complex* v=basis.data()+size_t(row)*n;Complex a=0;
                    for(int i=lo;i<hi;++i)a+=std::conj(v[i])*w[i];
                    q.h[size_t(row)*restart+col]+=a;for(int i=lo;i<hi;++i)w[i]-=a*v[i];
                }
                double squared=0;for(int i=lo;i<hi;++i)squared+=std::norm(w[i]);const double next=std::sqrt(squared);
                q.h[size_t(col+1)*restart+col]=next;
                if(recycle)for(int row=0;row<=col+1;++row)q.raw_h[size_t(row)*restart+col]=q.h[size_t(row)*restart+col];
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
                if(q.projected<inner_factor*target || next<1e-300)q.active=false;
            }
            ++result.iterations;result.orthogonalization_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-after).count();
            double projected2=0;for(const auto& q:states)projected2+=q.projected*q.projected;
            if(verbose)std::printf("  [T sector GMRES %d] projected %.6e elapsed_s=%.3f\n",result.iterations,std::sqrt(projected2)/bn,
                std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
            // Match the adaptive global GMRES acceptance attempt. Waiting
            // for every individual sector to hit its budget can oversolve
            // the complete packed RHS. The strict gate still decides.
            if(std::sqrt(projected2)<=inner_factor*tol*bn)break;
        }
        double projected2=0;bool any_active=false;
        for(const auto& q:states){projected2+=q.projected*q.projected;any_active=any_active||q.active;}
        const bool verify=std::sqrt(projected2)<=std::max(1.,inner_factor)*tol*bn || result.iterations>=limit || ++cycles>=strict_interval || !any_active;
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
                w[i]=update;if(!verify && used)r[i]=residual+basis[size_t(used)*n+i]*q.rc[used];
            }
            if(q.rank)recycle->sector(s).subtract_coupled_update(w.data()+offsets[s],q.y.data(),used,q.couplings,restart,q.rank);
            for(int i=offsets[s];i<offsets[s+1];++i)x[i]+=w[i];
        }
        if(recycle) {
            const auto before=std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
            for(int s=0;s<12;++s) {
                auto& q=states[s];const int lo=offsets[s],count=offsets[s+1]-lo;
                const Action identity=[&](const Complex* a,Complex* out){std::copy(a,a+count,out);};
                recycle->sector(s).learn_strided(basis.data()+lo,n,q.raw_h,restart,q.used,identity,q.couplings,q.rank);
            }
            result.recycle_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
        }
        if(verify) {
            check_strict();
        }else ++result.deferred_strict_checks;
    }
    result.recycle_rank=recycle?recycle->rank():0;
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return result;
}
}
#endif
