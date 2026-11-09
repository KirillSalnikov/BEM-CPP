#include "p2p_radial_factors.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <stdexcept>

// Independent long-double derivatives, including the cancellation in
// 2*bG-aG and both radial and tangential vector actions. No CUDA required.
int main() try {
    using C = std::complex<long double>;
    const long double pi = std::acos(-1.L);
    std::mt19937_64 rng(17083);
    std::uniform_real_distribution<double> uniform(-1., 1.);
    double worst = 0;
    int cases = 0;
    for (int exponent = -11; exponent <= 8; ++exponent)
    for (double phase : {0., 1e-8, .1, 1., std::sqrt(3.), 3., 10., 100., 1000.})
    for (int trial = 0; trial < 32; ++trial) {
        const double radius = std::pow(10., exponent)*(1.5+.4*uniform(rng));
        const double k = phase/radius, inv = 1/radius;
        const double gr = std::cos(k*radius)/(4*double(pi)*radius);
        const double gi = std::sin(k*radius)/(4*double(pi)*radius);
        const double g_re = -gr*inv*inv-gi*k*inv;
        const double g_im = gr*k*inv-gi*inv*inv;
        double ar, ai, br, bi;
        bem_p2p::radial_from_gradient_real(g_re,g_im,gr,gi,k*k,ar,ai,br,bi);
        const long double r=radius, wave=k;
        const C green=std::exp(C(0,wave*r))/(4*pi*r);
        const C first=green*C(-1/r,wave);
        const C second=green*C(2/(r*r)-wave*wave,-2*wave/r);
        const C a=second-first/r, b=-first/r;
        // Scalar coefficients can have zero components; use complex norms.
        double error=double(std::max(std::abs(C(ar,ai)-a),std::abs(C(br,bi)-b)) /
                                 std::max(std::abs(a),std::abs(b)));
        long double u[3]={uniform(rng),uniform(rng),uniform(rng)};
        const long double len=std::sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
        C q[3],dot=0;
        for(int d=0;d<3;++d){u[d]/=len;q[d]=C(uniform(rng),uniform(rng));dot+=u[d]*q[d];}
        long double diff=0,norm=0;
        for(int d=0;d<3;++d){
            const C exact=u[d]*a*dot+(2.L*b-a)*q[d];
            const C value=u[d]*C(ar,ai)*dot+(2.L*C(br,bi)-C(ar,ai))*q[d];
            diff+=std::norm(value-exact);norm+=std::norm(exact);
        }
        error=std::max(error,double(std::sqrt(diff/norm)));
        if(!std::isfinite(error)||error>2e-12)throw std::runtime_error("radial identity accuracy");
        worst=std::max(worst,error);++cases;
    }
    std::printf("PASS %d coefficient/vector cases against long-double derivatives; max relative L2 %.3e\n",cases,worst);
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
