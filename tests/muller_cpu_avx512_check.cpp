#include "muller_fmm_cpu_avx512.h"
#include <iostream>
#include <random>

int main() {
#ifdef BEM_CPU_HAS_AVX512_P2P
    using C=std::complex<double>;
    std::mt19937 rng(9273);std::normal_distribution<double> normal;
    std::array<std::vector<double>,15> source;
    for(auto& x:source)for(int i=0;i<43;++i)x.push_back(normal(rng));
    // Duplicate coordinates exercise the self exclusion, independently of ID.
    for(int c=0;c<3;++c)source[c][1]=source[c][0];
    double worst=0;
    for(C k:{C(0),C(40),C(40,8),C(3,-.1),C(1e-8,1e-9)})
        for(int count=0;count<=43;++count)
            for(int target_index:{0,17}) {
                const Vec3 target(source[0][target_index],source[1][target_index],source[2][target_index]);
                std::vector<int> ids;for(int i=0;i<count;++i)ids.push_back((i*17)%43);
                std::array<C,12> got{},ref{};
                bem_cpu::avx512::near_field(source,ids,target,k,got);
                for(int s:ids) {
                    const Vec3 r=target-Vec3(source[0][s],source[1][s],source[2][s]);
                    const double r2=r.norm2();if(r2<1e-28)continue;
                    const double ir=1/std::sqrt(r2),distance=r2*ir;
                    const double x=r.x*ir,y=r.y*ir,z=r.z*ir;
                    const C ik=C(0,1)*k,green=std::exp(ik*distance)*(INV4PI*ir);
                    const C gradient=green*(ik-ir),a=green*(3*ir*ir-3.*ik*ir-k*k),b=green*(k*k+ik*ir-ir*ir);
                    for(int j=0;j<2;++j) {
                        const C qx(source[3+6*j][s],source[4+6*j][s]);
                        const C qy(source[5+6*j][s],source[6+6*j][s]);
                        const C qz(source[7+6*j][s],source[8+6*j][s]);
                        const C dot=x*qx+y*qy+z*qz;
                        ref[6*j]+=gradient*(y*qx-x*qy);
                        ref[6*j+1]+=gradient*(z*qx-x*qz);
                        ref[6*j+2]+=gradient*(z*qy-y*qz);
                        ref[6*j+3]+=a*x*dot+b*qx;
                        ref[6*j+4]+=a*y*dot+b*qy;
                        ref[6*j+5]+=a*z*dot+b*qz;
                    }
                }
                double num=0,den=0;
                for(int c=0;c<12;++c) {
                    if(!std::isfinite(got[c].real())||!std::isfinite(got[c].imag()))return 2;
                    num+=std::norm(got[c]-ref[c]);den+=std::norm(ref[c]);
                }
                const double error=std::sqrt(num/std::max(den,1e-300));
                worst=std::max(worst,error);
                if(error>2e-12) {std::cerr<<"FAIL "<<k<<" "<<count<<" "<<error<<"\n";return 1;}
            }
    std::cout<<"PASS 440 tail/self/complex-wave-number cases; worst relative error "<<worst<<"\n";
#else
    std::cout<<"SKIP AVX512/glibc build unavailable\n";
#endif
}
