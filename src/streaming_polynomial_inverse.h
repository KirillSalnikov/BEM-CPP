#ifndef BEM_STREAMING_POLYNOMIAL_INVERSE_H
#define BEM_STREAMING_POLYNOMIAL_INVERSE_H
#include "streaming_recycle_space.h"
#include <algorithm>
#include <limits>

namespace bem_streaming {
// Experimental, low-degree inverse of A_cheap with a fixed local right
// preconditioner M^-1. Training uses twice-orthogonalized Arnoldi on
// B=A_cheap M^-1. Reuse evaluates the polynomial in that Arnoldi basis,
// avoiding ill-conditioned monomial coefficients. Not a high-degree
// replacement for root ordering / stability control (Embree et al., 2025).
// Mutable workspaces: a single instance is deliberately not thread safe.
class ArnoldiInverse {
    int n_,capacity_,degree_=0;
    Action action_,local_;
    std::vector<Complex> basis_,h_,coefficients_,work_,image_,sum_;
    double fit(const Complex* rhs,double target) {
        const double scale=norm(rhs,n_);
        degree_=0;
        std::fill(h_.begin(),h_.end(),Complex(0));
        if(scale==0)return 0;
        if(!std::isfinite(scale))throw std::runtime_error("nonfinite Arnoldi training RHS");
        for(int i=0;i<n_;++i)basis_[i]=rhs[i]/scale;
        std::vector<Complex> qr(h_.size()),c(capacity_),s(capacity_),g(capacity_+1);
        g[0]=1;
        for(int col=0;col<capacity_;++col) {
            local_(basis_.data()+size_t(col)*n_,image_.data());
            action_(image_.data(),work_.data());
            const double before=norm(work_.data(),n_);
            for(int pass=0;pass<2;++pass)for(int row=0;row<=col;++row) {
                const Complex* v=basis_.data()+size_t(row)*n_;
                const Complex a=dot(v,work_.data(),n_);
                h_[size_t(row)*capacity_+col]+=a;
#pragma omp parallel for
                for(int i=0;i<n_;++i)work_[i]-=a*v[i];
            }
            const double next=norm(work_.data(),n_);
            if(!std::isfinite(next))throw std::runtime_error("nonfinite Arnoldi inverse basis");
            h_[size_t(col+1)*capacity_+col]=next;
            const bool breakdown=next<=64*std::numeric_limits<double>::epsilon()*std::max(before,1e-300);
            if(!breakdown && col+1<capacity_) {
#pragma omp parallel for
                for(int i=0;i<n_;++i)basis_[size_t(col+1)*n_+i]=work_[i]/next;
            }
            for(int row=0;row<=col+1;++row)qr[size_t(row)*capacity_+col]=h_[size_t(row)*capacity_+col];
            for(int row=0;row<col;++row) {
                const Complex a=qr[size_t(row)*capacity_+col],b=qr[size_t(row+1)*capacity_+col];
                qr[size_t(row)*capacity_+col]=std::conj(c[row])*a+std::conj(s[row])*b;
                qr[size_t(row+1)*capacity_+col]=-s[row]*a+c[row]*b;
            }
            const Complex a=qr[size_t(col)*capacity_+col],b=qr[size_t(col+1)*capacity_+col];
            const double d=std::hypot(std::abs(a),std::abs(b));
            if(d<1e-300)throw std::runtime_error("singular Arnoldi inverse projection");
            c[col]=a/d;s[col]=b/d;
            qr[size_t(col)*capacity_+col]=d;
            g[col+1]=-s[col]*g[col];g[col]=std::conj(c[col])*g[col];
            degree_=col+1;
            if(breakdown || std::abs(g[col+1])<=target)break;
        }
        for(int row=degree_-1;row>=0;--row) {
            Complex value=g[row];
            for(int col=row+1;col<degree_;++col)value-=qr[size_t(row)*capacity_+col]*coefficients_[col];
            coefficients_[row]=value/qr[size_t(row)*capacity_+row];
        }
        return scale;
    }
    void combine(double scale,Complex* output) {
#pragma omp parallel for
        for(int i=0;i<n_;++i) {
            Complex value=0;
            for(int j=0;j<degree_;++j)value+=coefficients_[j]*basis_[size_t(j)*n_+i];
            sum_[i]=scale*value;
        }
        local_(sum_.data(),output);
    }
public:
    ArnoldiInverse(int n,int steps,const Action& action,const Action& local):
        n_(n),capacity_(steps),action_(action),local_(local) {
        if(n<1 || steps<1 || steps>32)throw std::invalid_argument("Arnoldi inverse supports 1..32 steps");
        basis_.resize(size_t(n)*steps);h_.resize(size_t(steps+1)*steps);
        coefficients_.resize(steps);work_.resize(n);image_.resize(n);sum_.resize(n);
    }
    int degree() const {return degree_;}
    // Variable inverse: new minimum-residual polynomial for each RHS.
    // Requires flexible outer GMRES even with a fixed iteration limit.
    void solve(const Complex* rhs,Complex* output,double target=.1) {
        if(target<0 || target>=1)throw std::invalid_argument("invalid inner residual target");
        const double scale=fit(rhs,target);
        if(scale==0){std::fill(output,output+n_,Complex(0));return;}
        combine(scale,output);
    }
    void train(const Complex* rhs) {
        if(fit(rhs,0)==0)throw std::invalid_argument("zero polynomial training RHS");
    }
    // Fixed linear inverse, reusable across incident modes. A degree-d
    // Arnoldi solve gives an inverse polynomial of degree d-1 in B.
    void apply(const Complex* rhs,Complex* output) {
        if(degree_==0)throw std::logic_error("polynomial inverse not trained");
        std::copy(rhs,rhs+n_,basis_.begin());
        for(int col=0;col+1<degree_;++col) {
            local_(basis_.data()+size_t(col)*n_,image_.data());
            action_(image_.data(),work_.data());
#pragma omp parallel for
            for(int i=0;i<n_;++i) {
                Complex value=work_[i];
                for(int row=0;row<=col;++row)value-=h_[size_t(row)*capacity_+col]*basis_[size_t(row)*n_+i];
                basis_[size_t(col+1)*n_+i]=value/h_[size_t(col+1)*capacity_+col];
            }
        }
        combine(1,output);
    }
};
}
#endif
