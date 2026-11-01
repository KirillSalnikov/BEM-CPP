#pragma once
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

// Analytic frequency jets for a FIXED FMM tree, angular grid and truncation
// order. Changing any of those discretizations is a new expansion centre.
// Coefficients are derivatives divided by factorial, never finite differences.
namespace bem_frequency {
using C=std::complex<long double>;
template<size_t N> using Jet=std::array<C,N+1>;
template<size_t N> Jet<N> constant(C value){Jet<N> a{};a[0]=value;return a;}
template<size_t N> Jet<N> variable(C value,C slope){auto a=constant<N>(value);if(N)a[1]=slope;return a;}
template<size_t N> Jet<N> product(const Jet<N>& a,const Jet<N>& b){
    Jet<N> c{};for(size_t j=0;j<=N;++j)for(size_t q=0;q<=j;++q)c[j]+=a[q]*b[j-q];return c;
}
template<size_t N> Jet<N> reciprocal(const Jet<N>& a){
    if(std::abs(a[0])==0)throw std::invalid_argument("Taylor reciprocal at zero");
    Jet<N> c{};c[0]=C(1)/a[0];
    for(size_t j=1;j<=N;++j){for(size_t q=1;q<=j;++q)c[j]-=a[q]*c[j-q];c[j]/=a[0];}return c;
}
// exp(value+slope*delta), used by P2M/M2M/L2L/L2P as well as Hankel.
template<size_t N> Jet<N> phase(C value,C slope){
    Jet<N> c{};c[0]=std::exp(value);for(size_t j=1;j<=N;++j)c[j]=c[j-1]*slope/static_cast<long double>(j);return c;
}

template<size_t N> std::vector<Jet<N>> m2l_radial(C k,C slope,long double distance,int p){
    static_assert(N<=16,"frequency jet order limit");
    if(p<0||p>512||!std::isfinite(distance)||distance<=0||!std::isfinite(k.real())||
       !std::isfinite(k.imag())||std::abs(k)==0||!std::isfinite(slope.real())||!std::isfinite(slope.imag()))
        throw std::invalid_argument("FMM Taylor radial parameters");
    const C I(0,1);const auto inv=reciprocal<N>(variable<N>(k*distance,slope*distance));
    const auto exp=phase<N>(I*k*distance,I*slope*distance);
    std::vector<Jet<N>> h(p+1),radial(p+1);
    h[0]=product<N>(exp,inv);for(auto& z:h[0])z*=-I;
    if(p){auto a=inv;for(auto& z:a)z*=I;a[0]+=1;
        h[1]=product<N>(product<N>(exp,inv),a);for(auto& z:h[1])z=-z;}
    for(int l=1;l<p;++l){h[l+1]=product<N>(inv,h[l]);
        for(size_t j=0;j<=N;++j)h[l+1][j]=(2.L*l+1)*h[l+1][j]-h[l-1][j];}
    C power=1;for(int l=0;l<=p;++l){
        for(size_t j=0;j<=N;++j)radial[l][j]=(2.L*l+1)*power*h[l][j];
        power*=I;
    }
    return radial;
}
template<size_t N> Jet<N> m2l_direction(const std::vector<Jet<N>>& radial,long double cosine){
    if(radial.empty()||!std::isfinite(cosine)||std::abs(cosine)>1+1e-12L)
        throw std::invalid_argument("FMM Taylor direction");
    Jet<N> sum=radial[0];long double previous=1,current=cosine;
    if(radial.size()>1)for(size_t j=0;j<=N;++j)sum[j]+=current*radial[1][j];
    for(size_t l=2;l<radial.size();++l){
        const long double next=((2.L*l-1)*cosine*current-(l-1)*previous)/l;
        for(size_t j=0;j<=N;++j)sum[j]+=next*radial[l][j];
        previous=current;current=next;
    }
    return sum;
}
}
