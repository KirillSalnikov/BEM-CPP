#pragma once
#include "streaming_recycle_space.h"

namespace bem_streaming {
// A fixed least-squares correction in explicitly supplied symmetry coordinates.
// Each column satisfies A U = C; C has orthonormal columns in every sector.
// The caller supplies the distinct solution/equation coordinate transforms.
class ImageSpace {
    using V=std::vector<Complex>;
    std::vector<int> offsets_;
    std::vector<V> u_,c_;
    int n_,rank_=0;
    static Complex inner(const V& a,const V& b,int lo,int hi) {
        Complex z=0;for(int i=lo;i<hi;++i)z+=std::conj(a[i])*b[i];return z;
    }
public:
    explicit ImageSpace(std::vector<int> offsets):offsets_(std::move(offsets)),n_(offsets_.empty()?0:offsets_.back()) {
        if(offsets_.size()<2||offsets_[0]!=0||n_<1)throw std::invalid_argument("image-space dimensions");
        for(size_t i=1;i<offsets_.size();++i)if(offsets_[i]<offsets_[i-1])throw std::invalid_argument("image-space offsets");
    }
    int rank()const{return rank_;}
    size_t bytes()const{return 2*u_.size()*size_t(n_)*sizeof(Complex);}
    bool add(V u,V c) {
        if(int(u.size())!=n_||int(c.size())!=n_)throw std::invalid_argument("image-space column size");
        int added=0;
#pragma omp parallel for reduction(+:added) schedule(static)
        for(int s=0;s<int(offsets_.size())-1;++s){const int lo=offsets_[s],hi=offsets_[s+1];
            const double original=std::sqrt(std::max(0.,inner(c,c,lo,hi).real()));
            for(int pass=0;pass<2;++pass)for(size_t j=0;j<c_.size();++j){const Complex a=inner(c_[j],c,lo,hi);
                for(int i=lo;i<hi;++i){c[i]-=a*c_[j][i];u[i]-=a*u_[j][i];}}
            const double length=std::sqrt(std::max(0.,inner(c,c,lo,hi).real()));
            if(std::isfinite(length)&&length>1e-8*std::max(original,1e-300)){
                for(int i=lo;i<hi;++i){c[i]/=length;u[i]/=length;}++added;
            }else for(int i=lo;i<hi;++i)c[i]=u[i]=0.;
        }
        if(added){rank_+=added;u_.push_back(std::move(u));c_.push_back(std::move(c));return true;}
        return false;
    }
    void apply(const V& rhs,V& guess,V& removed)const {
        if(int(rhs.size())!=n_)throw std::invalid_argument("image-space RHS size");
        V residual=rhs;guess.assign(n_,0.);removed.resize(n_);
#pragma omp parallel for schedule(static)
        for(int s=0;s<int(offsets_.size())-1;++s){const int lo=offsets_[s],hi=offsets_[s+1];
            for(int pass=0;pass<2;++pass)for(size_t j=0;j<c_.size();++j){const Complex a=inner(c_[j],residual,lo,hi);
                for(int i=lo;i<hi;++i){guess[i]+=a*u_[j][i];residual[i]-=a*c_[j][i];}}
            for(int i=lo;i<hi;++i)removed[i]=rhs[i]-residual[i];
        }
    }
};
}
