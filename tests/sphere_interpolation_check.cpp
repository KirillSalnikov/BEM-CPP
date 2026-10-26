#include "sphere_interpolation.h"
#include <complex>
#include <cstdio>
#include <vector>
using C=std::complex<double>;

static std::vector<C> interpolate(const SphereQuad& f,const SphereQuad& c,
    const SphereInterpolation& a,const std::vector<C>& x) {
    std::vector<C> e(a.ntf*a.npc),o(e.size()),y(c.L);
    for(int t=0;t<a.ntf;++t)for(int p=0;p<a.npc;++p)
        for(int j=0;j<a.npf;++j) {
            e[t*a.npc+p]+=a.phi_even[p*a.npf+j]*x[t*a.npf+j];
            o[t*a.npc+p]+=a.phi_odd[p*a.npf+j]*x[t*a.npf+j];
        }
    for(int t=0;t<a.ntc;++t)for(int p=0;p<a.npc;++p)
        for(int j=0;j<a.ntf;++j)
            y[t*a.npc+p]+=a.theta_even[t*a.ntf+j]*e[j*a.npc+p]
                           +a.theta_odd[t*a.ntf+j]*o[j*a.npc+p];
    return y;
}
int main() {
    double worst=0.;
    for(int order : {3,8,24,36}) {
        SphereQuad f,c;f.init(order);c.init(2*order+1);
        SphereInterpolation a(f,c);
        // Restrictions of Cartesian monomials span spherical polynomials.
        // Both azimuthal parities, extremal degree and coincident GL nodes.
        for(int m=0;m<=order;++m) for(int j : {0,order-m}) {
            std::vector<C> x(f.L);
            for(int n=0;n<f.L;++n)
                x[n]=std::pow(C(f.dirs[3*n],f.dirs[3*n+1]),m)*std::pow(f.dirs[3*n+2],j);
            auto y=interpolate(f,c,a,x);
            for(int n=0;n<c.L;++n) {
                C exact=std::pow(C(c.dirs[3*n],c.dirs[3*n+1]),m)*std::pow(c.dirs[3*n+2],j);
                worst=std::max(worst,std::abs(y[n]-exact));
            }
        }
    }
    std::printf("spherical polynomial interpolation max absolute error %.3e\n",worst);
    return worst<2e-12?0:1;
}
