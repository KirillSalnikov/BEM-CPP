#pragma once
#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

// Signed affine channel permutation on a square periodic lattice. The maps
// act from source to target. Equation and current signs need not coincide.
struct CirculantSymmetry {
    int n=0;
    std::array<int,4> matrix{{1,0,0,1}};
    std::vector<int> channel,solution_sign,rhs_sign;
    std::vector<std::array<int,2>> offset;
    int mod(int x)const{return (x%n+n)%n;}
    std::array<int,2> point(int c,int i,int j)const {
        return {{mod(matrix[0]*i+matrix[1]*j+offset[c][0]),
                 mod(matrix[2]*i+matrix[3]*j+offset[c][1])}};
    }
    static CirculantSymmetry identity(int n,int c) {
        CirculantSymmetry a;a.n=n;a.channel.resize(c);a.offset.resize(c);
        a.solution_sign.assign(c,1);a.rhs_sign.assign(c,1);
        for(int j=0;j<c;++j)a.channel[j]=j;
        return a;
    }
    void validate()const {
        const int c=channel.size();
        if(n<1 || !c || offset.size()!=size_t(c) || solution_sign.size()!=size_t(c) || rhs_sign.size()!=size_t(c))
            throw std::invalid_argument("circulant symmetry dimensions");
        std::vector<bool> seen(size_t(c)*n*n);
        for(int ch=0;ch<c;++ch) {
            if(channel[ch]<0 || channel[ch]>=c || std::abs(solution_sign[ch])!=1 || std::abs(rhs_sign[ch])!=1)
                throw std::invalid_argument("circulant symmetry channel/sign");
            for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
                const auto p=point(ch,i,j);const int k=(channel[ch]*n+p[0])*n+p[1];
                if(seen[k])throw std::invalid_argument("nonbijective circulant symmetry");
                seen[k]=true;
            }
        }
    }
};

inline CirculantSymmetry compose(const CirculantSymmetry& a,const CirculantSymmetry& b) {
    if(a.n!=b.n || a.channel.size()!=b.channel.size())throw std::invalid_argument("circulant composition dimensions");
    auto out=CirculantSymmetry::identity(a.n,a.channel.size());
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)
        out.matrix[2*i+j]=a.mod(a.matrix[2*i]*b.matrix[j]+a.matrix[2*i+1]*b.matrix[2+j]);
    for(size_t c=0;c<a.channel.size();++c) {
        const int d=b.channel[c];out.channel[c]=a.channel[d];
        out.offset[c]=a.point(d,b.offset[c][0],b.offset[c][1]);
        out.solution_sign[c]=a.solution_sign[d]*b.solution_sign[c];
        out.rhs_sign[c]=a.rhs_sign[d]*b.rhs_sign[c];
    }
    return out;
}

// Average S_solution K S_equation^{-1}, AFTER inversion. Averaging A before
// inversion would define a different preconditioner. K is a spatial kernel.
inline std::vector<std::complex<double>> average_inverse_kernel(
    const std::vector<std::complex<double>>& kernel,const std::vector<CirculantSymmetry>& group) {
    if(group.empty())throw std::invalid_argument("empty circulant symmetry group");
    const int n=group[0].n,c=group[0].channel.size(),grid=n*n;
    if(kernel.size()!=size_t(c)*c*grid)throw std::invalid_argument("circulant kernel size");
    std::vector<std::complex<double>> out(kernel.size());
    for(const auto& g:group) {
        g.validate();
        if(g.n!=n || g.channel.size()!=size_t(c))throw std::invalid_argument("inconsistent circulant group");
        // Different channel pairs have disjoint destinations; the affine
        // spatial map is bijective, so accumulation needs no atomics.
#pragma omp parallel for schedule(static)
        for(int ab=0;ab<c*c;++ab) {
            const int a=ab/c,b=ab%c,target=g.channel[a]*c+g.channel[b];
            const double weight=double(g.solution_sign[a]*g.rhs_sign[b])/group.size();
            for(int i=0;i<n;++i)for(int j=0;j<n;++j) {
                const int x=g.mod(g.matrix[0]*i+g.matrix[1]*j+g.offset[a][0]-g.offset[b][0]);
                const int y=g.mod(g.matrix[2]*i+g.matrix[3]*j+g.offset[a][1]-g.offset[b][1]);
                out[size_t(target)*grid+x*n+y]+=weight*kernel[size_t(ab)*grid+i*n+j];
            }
        }
    }
    return out;
}
