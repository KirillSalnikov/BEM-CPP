#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

// Experimental planar two-medium symbol. Convention: exp(-i omega t), outgoing
// Green function exp(i k r)/(4 pi r), Fourier exp(-i xi.x), n=(0,0,1).
// The rotated test function in muller_nodal.cpp is test x n. Therefore its
// strong K1 action is R H, R v=n x v, not -R H.
namespace bem_osrc {
using C=std::complex<double>;
using Matrix2=std::array<C,4>;
using Matrix4=std::array<C,16>;
inline bool finite(C z){return std::isfinite(z.real())&&std::isfinite(z.imag());}
inline C outgoing_root(C k,double tangential_norm2){
    if(!finite(k)||k.real()<=0||k.imag()<0||!std::isfinite(tangential_norm2)||tangential_norm2<0)
        throw std::invalid_argument("planar symbol requires a passive positive-frequency medium");
    C q=std::sqrt(k*k-tangential_norm2);
    if(q.imag()<0 || (q.imag()==0 && q.real()<0))q=-q;
    return q;
}

// Hhat=(k^2 I-xi xi^T) i/(2 q) = i D/2. D has longitudinal
// eigenvalue q and transverse eigenvalue k^2/q. The same regularized k is
// used in both numerator and root, as required at glancing frequencies.
inline Matrix2 dyadic(C k,double x,double y){
    const double rho2=x*x+y*y;const C q=outgoing_root(k,rho2);
    if(std::abs(q)<=1e-14*std::abs(k))throw std::domain_error("unregularized glancing symbol");
    const C transverse=k*k/q;
    if(rho2==0)return {{k,0.,0.,k}};
    const C delta=q-transverse;
    return {{transverse+delta*x*x/rho2,delta*x*y/rho2,
             delta*x*y/rho2,transverse+delta*y*y/rho2}};
}

// The half-space MtE map of El Bouajaji et al. (2014), Eq.12, has eigenvalues
// (k/q)/Z along xi and (q/k)/Z perpendicular to xi. This helper is used to
// cross-check the Green-function-derived Muller block, not as a PEC shortcut.
inline Matrix2 mte(C k,C impedance,double x,double y){
    if(!finite(impedance)||std::abs(impedance)==0)throw std::invalid_argument("MtE impedance");
    const double rho2=x*x+y*y;const C q=outgoing_root(k,rho2);
    if(std::abs(q)<=1e-14*std::abs(k))throw std::domain_error("unregularized glancing MtE");
    const C along=k/(q*impedance),across=q/(k*impedance);
    if(rho2==0)return {{1./impedance,0.,0.,1./impedance}};
    const C delta=along-across;
    return {{across+delta*x*x/rho2,delta*x*y/rho2,delta*x*y/rho2,across+delta*y*y/rho2}};
}

inline Matrix2 diagonal_block(double k0,C exterior,C interior,double x,double y){
    if(!std::isfinite(k0)||k0<=0)throw std::invalid_argument("Muller frequency normalization");
    auto a=dyadic(exterior,x,y),b=dyadic(interior,x,y);
    for(int j=0;j<4;++j)a[j]-=b[j];
    // (i/k0) R (i/2)(D_e-D_i) = -R(D_e-D_i)/(2 k0).
    return {{a[2]/(2*k0),a[3]/(2*k0),-a[0]/(2*k0),-a[1]/(2*k0)}};
}

inline Matrix4 muller(double k0,C exterior,C interior,C epsilon_sum,C mu_sum,double x,double y){
    if(!finite(epsilon_sum)||!finite(mu_sum))throw std::invalid_argument("Muller material sums");
    const auto b=diagonal_block(k0,exterior,interior,x,y);Matrix4 a{};
    for(int i=0;i<2;++i)for(int j=0;j<2;++j){a[4*i+j]=b[2*i+j];a[4*(i+2)+j+2]=-b[2*i+j];}
    for(int i=0;i<2;++i){a[4*i+i+2]=epsilon_sum*.5;a[4*(i+2)+i]=mu_sum*.5;}
    return a;
}

inline Matrix4 inverse(double k0,C exterior,C interior,C epsilon_sum,C mu_sum,double x,double y){
    auto a=muller(k0,exterior,interior,epsilon_sum,mu_sum,x,y);
    const C denominator=a[0]*a[0]+a[1]*a[4]+epsilon_sum*mu_sum*.25;
    double scale=0;for(auto z:a)scale=std::max(scale,std::norm(z));
    if(std::abs(denominator)<=1e-13*std::max(scale,1e-300))throw std::domain_error("singular planar Muller symbol");
    for(auto& z:a)z/=denominator;
    return a;
}
}
