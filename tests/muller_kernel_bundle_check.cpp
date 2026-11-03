#include "muller_kernel_bundle.h"
#include <cstdio>
#include <vector>

int main() try {
    using C = std::complex<double>;
    double worst = 0.0;
    int cases = 0;
    for (C k : {C(0.01), C(1.0), C(20.0), C(3.0, 0.4)}) {
        for (C n : {C(1.0), C(1.0 + 1e-6), C(1.3116), C(1.5, 0.2)}) {
            for (double radius : {1e-8, 1e-5, 0.001, 0.00999, 0.01001, 0.1, 1.0, 10.0}) {
                const Vec3 r = Vec3(0.3, -0.4, std::sqrt(0.75)) * radius;
                const C e1(1.0, 0.1), e2 = n * n, m1(0.9, 0.02), m2(1.1, 0.03);
                const auto actual = bem_muller::evaluate_kernel_values(k, k*n, e1, e2, m1, m2, r);
                const auto h = muller_hessian_difference(k, k*n, r);
                const auto ge = muller_composite_gradient(e1, e2, k, k*n, r);
                const auto gm = muller_composite_gradient(m1, m2, k, k*n, r);
                auto check = [&](C a, C b) {
                    if (!std::isfinite(a.real()) || !std::isfinite(a.imag()))
                        throw std::runtime_error("nonfinite combined kernel");
                    const double error = std::abs(a-b) / std::max(1.0, std::abs(b));
                    worst = std::max(worst, error);
                    if (error > 1e-13)
                        throw std::runtime_error("combined kernel differs from scalar formulas");
                };
                for (int i=0;i<9;++i) check(actual.hessian[i], h[i]);
                for (int i=0;i<3;++i) {
                    check(actual.gradient_epsilon[i], ge[i]);
                    check(actual.gradient_mu[i], gm[i]);
                }
                const Vec3 test(0.2, -0.7, 0.3), source(-0.5, 0.1, 0.8);
                C contraction = actual.k1_scalar * test.dot(source);
                const double x[3]={test.x,test.y,test.z}, y[3]={source.x,source.y,source.z};
                for(int i=0;i<3;++i)for(int j=0;j<3;++j)
                    contraction += x[i]*actual.hessian[3*i+j]*y[j];
                check(contraction,muller_k1_kernel(k,k*n,r,test,source));
                ++cases;
            }
        }
    }
    bool rejected=false;
    try {bem_muller::evaluate_kernel_values(C(1),C(1.3),C(1),C(1.69),C(1),C(1),Vec3());}
    catch(const std::runtime_error&){rejected=true;}
    if(!rejected)return 2;
    std::printf("combined Muller kernels: %d cases, worst normalized difference %.3g\n",cases,worst);
    return 0;
} catch(const std::exception& e) {
    std::fprintf(stderr,"%s\n",e.what());return 1;
}
