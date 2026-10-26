#ifndef BEM_STREAMING_HOST_GMRES_H
#define BEM_STREAMING_HOST_GMRES_H
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

namespace bem_streaming {
using Complex=std::complex<double>;
using Action=std::function<void(const Complex*,Complex*)>;
struct HostResult { int iterations=0; double residual=1,seconds=0; };
inline Complex dot(const Complex* x,const Complex* y,int n) {
    double real=0,imaginary=0;
#pragma omp parallel for reduction(+:real,imaginary)
    for(int i=0;i<n;++i) {
        const Complex value=std::conj(x[i])*y[i];
        real+=value.real(); imaginary+=value.imag();
    }
    return Complex(real,imaginary);
}
inline double norm(const Complex* x,int n) { return std::sqrt(std::max(0.,dot(x,x,n).real())); }
// Fixed right preconditioning. Only the Arnoldi basis is retained; the
// linear MBJ inverse is applied once more to its final linear combination.
inline HostResult gmres(const Action& action,const Action& strict,const Action& inverse,
                        const Complex* b,Complex* x,int n,int restart,double tol,int limit,
                        bool verbose=true) {
    const auto start=std::chrono::steady_clock::now();
    restart=std::min(restart,limit);
    if(n<1||restart<1||limit<1||tol<=0) throw std::invalid_argument("invalid host GMRES dimensions");
    std::vector<Complex> basis(size_t(n)*(restart+1)),h(size_t(restart+1)*restart),
        c(restart),s(restart),g(restart+1),y(restart),r(b,b+n),z(n),w(n);
    std::fill(x,x+n,Complex(0));
    const double bn=std::max(norm(b,n),1e-300);
    HostResult result;
    while(result.iterations<limit && result.residual>tol) {
        const double rn=norm(r.data(),n);
        std::fill(h.begin(),h.end(),Complex(0)); std::fill(g.begin(),g.end(),Complex(0));
        g[0]=rn;
#pragma omp parallel for
        for(int i=0;i<n;++i) basis[i]=r[i]/std::max(rn,1e-300);
        int used=0;
        for(int col=0;col<restart && result.iterations<limit;++col) {
            const Complex* v=basis.data()+size_t(col)*n;
            inverse(v,z.data()); action(z.data(),w.data());
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
            if(verbose) std::printf("  [T host GMRES %d] projected %.6e\n",result.iterations,projected);
            if(projected<tol*.1 || next<1e-300) break;
        }
        for(int row=used-1;row>=0;--row) {
            Complex value=g[row];
            for(int col=row+1;col<used;++col) value-=h[size_t(row)*restart+col]*y[col];
            if(std::abs(h[size_t(row)*restart+row])<1e-300) throw std::runtime_error("singular Arnoldi projection");
            y[row]=value/h[size_t(row)*restart+row];
        }
#pragma omp parallel for
        for(int i=0;i<n;++i) {
            Complex value=0;
            for(int col=0;col<used;++col) value+=basis[size_t(col)*n+i]*y[col];
            w[i]=value;
        }
        inverse(w.data(),z.data());
#pragma omp parallel for
        for(int i=0;i<n;++i) x[i]+=z[i];
        strict(x,w.data());
#pragma omp parallel for
        for(int i=0;i<n;++i) r[i]=b[i]-w[i];
        result.residual=norm(r.data(),n)/bn;
        if(verbose) std::printf("  [T host GMRES] full FP64 residual %.6e\n",result.residual);
    }
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    return result;
}
}
#endif
