#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

// Frequency moments for fixed nondispersive material and fixed geometry.
// Coefficient j is derivative/j!, with k=k0+delta. These point kernels can
// be integrated/scattered for sparse Duffy corrections; no dense Aq is needed.
// Match the production small-kr branch (including its degree-6 truncation).
// The switch is frozen at the central frequency; crossing it off-centre must
// be checked against the original operator, not treated as exact analytic continuation.
namespace bem_taylor {
using C=std::complex<double>;
static const C I(0,1);
static const double INV4PI=1/(4*std::acos(-1.));
inline double binomial(int n,int j) {
    if(j>n)return 0;
    double b=1;for(int a=1;a<=j;++a)b*=double(n-a+1)/a;return b;
}
template<size_t MaxOrder> struct Radial {
    using Series=std::array<C,MaxOrder+1>;
    Series diagonal{},along{},epsilon{},mu{};
};
template<size_t MaxOrder> Radial<MaxOrder> radial(double k,C m,double r,int order=MaxOrder) {
    static_assert(MaxOrder<=16,"Taylor order limit");
    using Series=typename Radial<MaxOrder>::Series;
    if(!std::isfinite(k)||k<=0||!std::isfinite(m.real())||!std::isfinite(m.imag())||
       !std::isfinite(r)||r<=0||order<0||order>int(MaxOrder))
        throw std::invalid_argument("Muller radial Taylor parameters");
    if(!(r>0))throw std::runtime_error("zero quadrature separation");
    Series phi{},hdiag{},huu{},ge{},gm{};
    for(int medium=0;medium<2;++medium) {
        const C s=medium?m:C(1),sign=medium?C(-1):C(1);
        const C chi=medium?m*m:C(1),z=I*s*r;
        Series e{};e[0]=std::exp(z*k);
        for(int j=1;j<=order;++j)e[j]=e[j-1]*z/double(j);
        for(int j=0;j<=order;++j) {
            const C em1=j?e[j-1]:C(0),em2=j>1?e[j-2]:C(0);
            phi[j]+=sign*s*s*INV4PI/r*(k*k*e[j]+2*k*em1+em2);
            const C f=INV4PI/(r*r*r)*((3.-3.*z*k-s*s*k*k*r*r)*e[j]+(-3.*z-2.*s*s*k*r*r)*em1-s*s*r*r*em2);
            const C g=INV4PI/(r*r*r)*((1.-z*k)*e[j]-z*em1);
            hdiag[j]-=sign*g;huu[j]+=sign*f;
            const C grad=INV4PI/r*((I*s*k-1./r)*e[j]+I*s*em1);
            ge[j]+=sign*chi*grad;gm[j]+=sign*grad;
        }
    }
    // Match the library's stable small-distance polynomial, including its
    // existing truncation, so this tests continuation of the same operator.
    if(std::max(1.,std::abs(m))*k*r<1e-2) {
        hdiag.fill(0);huu.fill(0);ge.fill(0);gm.fill(0);
        double factorial=1;
        for(int n=0;n<=6;++n) {
            if(n)factorial*=n;
            for(int j=0;j<=std::min(n,order);++j) {
                const C c=std::pow(I,n)*binomial(n,j)*std::pow(k,n-j)/factorial*INV4PI;
                if(n>=2) {
                    const C h=c*(1.-std::pow(m,n))*std::pow(r,n-3);
                    hdiag[j]+=h*double(n-1);huu[j]+=h*double((n-1)*(n-3));
                }
                if(n!=1) {
                    const C g=c*double(n-1)*std::pow(r,n-2);
                    ge[j]+=g*(1.-std::pow(m,n+2));gm[j]+=g*(1.-std::pow(m,n));
                }
            }
        }
    }
    Radial<MaxOrder> a;
    for(int j=0;j<=order;++j) {
        a.diagonal[j]=(I*(phi[j]+hdiag[j])-(j?a.diagonal[j-1]:C(0)))/k;
        a.along[j]=(I*huu[j]-(j?a.along[j-1]:C(0)))/k;
        a.epsilon[j]=ge[j];a.mu[j]=gm[j];
    }
    return a;
}
}
