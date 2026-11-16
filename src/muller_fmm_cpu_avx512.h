#ifndef BEM_MULLER_FMM_CPU_AVX512_H
#define BEM_MULLER_FMM_CPU_AVX512_H
#include "muller_fmm.h"

// Optional host-only FP64 near-field kernel. The vector transcendental ABI is
// supplied by glibc libmvec; other platforms retain the scalar implementation.
// No fast-math flags or approximate reciprocals are used.
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__GLIBC__) && !defined(__CUDACC__)
#define BEM_CPU_HAS_AVX512_P2P 1
#include <immintrin.h>
extern "C" __m512d _ZGVeN8v_sin(__m512d);
extern "C" __m512d _ZGVeN8v_cos(__m512d);
extern "C" __m512d _ZGVeN8v_exp(__m512d);

namespace bem_cpu {
namespace avx512 {
struct Complex {
    __m512d re, im;
    Complex():re(_mm512_setzero_pd()),im(_mm512_setzero_pd()){}
    Complex(__m512d r,__m512d i):re(r),im(i){}
    Complex& operator+=(const Complex& b) {
        re=_mm512_add_pd(re,b.re);im=_mm512_add_pd(im,b.im);return *this;
    }
};
inline Complex operator+(const Complex& a,const Complex& b) {
    return {_mm512_add_pd(a.re,b.re),_mm512_add_pd(a.im,b.im)};
}
inline Complex operator-(const Complex& a,const Complex& b) {
    return {_mm512_sub_pd(a.re,b.re),_mm512_sub_pd(a.im,b.im)};
}
inline Complex operator*(const Complex& a,const Complex& b) {
    return {_mm512_sub_pd(_mm512_mul_pd(a.re,b.re),_mm512_mul_pd(a.im,b.im)),
            _mm512_add_pd(_mm512_mul_pd(a.re,b.im),_mm512_mul_pd(a.im,b.re))};
}
inline Complex operator*(__m512d a,const Complex& b) {
    return {_mm512_mul_pd(a,b.re),_mm512_mul_pd(a,b.im)};
}
inline __m512d constant(double x) {return _mm512_set1_pd(x);}

// SoA arrays contain coordinates followed by the real/imaginary parts of six
// source-current components. Source IDs are unchanged; padding is never read.
inline void near_field(const std::array<std::vector<double>,15>& source,
                       const std::vector<int>& ids,const Vec3& target,std::complex<double> k,
                       std::array<std::complex<double>,12>& result) {
    const __m512d one=constant(1),zero=constant(0),kr=constant(k.real()),ki=constant(k.imag());
    const std::complex<double> k2=k*k;
    Complex sums[12];
    for(size_t s=0;s<ids.size();s+=8) {
        const __mmask8 tail=__mmask8((1u<<std::min(size_t(8),ids.size()-s))-1u);
        const __m256i indices=_mm256_maskz_loadu_epi32(tail,ids.data()+s);
        auto gather=[&](int c) {
            return _mm512_mask_i32gather_pd(zero,tail,indices,source[c].data(),8);
        };
        const __m512d dx=_mm512_sub_pd(constant(target.x),gather(0));
        const __m512d dy=_mm512_sub_pd(constant(target.y),gather(1));
        const __m512d dz=_mm512_sub_pd(constant(target.z),gather(2));
        __m512d r2=_mm512_add_pd(_mm512_mul_pd(dx,dx),
                    _mm512_add_pd(_mm512_mul_pd(dy,dy),_mm512_mul_pd(dz,dz)));
        const __mmask8 active=tail&_mm512_cmp_pd_mask(r2,constant(1e-28),_CMP_GE_OQ);
        r2=_mm512_mask_mov_pd(one,active,r2);
        const __m512d ir=_mm512_div_pd(one,_mm512_sqrt_pd(r2));
        const __m512d r=_mm512_mul_pd(r2,ir),arg=_mm512_maskz_mul_pd(active,kr,r);
        const __m512d x=_mm512_mul_pd(dx,ir),y=_mm512_mul_pd(dy,ir),z=_mm512_mul_pd(dz,ir);
        __m512d scale=_mm512_maskz_mul_pd(active,constant(INV4PI),ir);
        if(k.imag()!=0)scale=_mm512_mul_pd(scale,_ZGVeN8v_exp(_mm512_maskz_mul_pd(active,constant(-k.imag()),r)));
        const Complex green(_mm512_mul_pd(scale,_ZGVeN8v_cos(arg)),
                            _mm512_mul_pd(scale,_ZGVeN8v_sin(arg)));
        const __m512d ir2=_mm512_mul_pd(ir,ir),kir=_mm512_mul_pd(ki,ir),krr=_mm512_mul_pd(kr,ir);
        const Complex gradient=green*Complex(_mm512_sub_pd(constant(-k.imag()),ir),kr);
        const Complex a=green*Complex(
            _mm512_sub_pd(_mm512_mul_pd(constant(3),_mm512_add_pd(ir2,kir)),constant(k2.real())),
            _mm512_sub_pd(_mm512_mul_pd(constant(-3),krr),constant(k2.imag())));
        const Complex b=green*Complex(
            _mm512_sub_pd(constant(k2.real()),_mm512_add_pd(kir,ir2)),
            _mm512_add_pd(constant(k2.imag()),krr));
        for(int j=0;j<2;++j) {
            const Complex qx(gather(3+6*j),gather(4+6*j));
            const Complex qy(gather(5+6*j),gather(6+6*j));
            const Complex qz(gather(7+6*j),gather(8+6*j));
            const Complex dot=x*qx+y*qy+z*qz;
            sums[6*j]+=gradient*(y*qx-x*qy);
            sums[6*j+1]+=gradient*(z*qx-x*qz);
            sums[6*j+2]+=gradient*(z*qy-y*qz);
            sums[6*j+3]+=x*(a*dot)+b*qx;
            sums[6*j+4]+=y*(a*dot)+b*qy;
            sums[6*j+5]+=z*(a*dot)+b*qz;
        }
    }
    for(int c=0;c<12;++c)result[c]+=std::complex<double>(_mm512_reduce_add_pd(sums[c].re),_mm512_reduce_add_pd(sums[c].im));
}

// Combined physical two-medium near action. Curl slots for J carry the mu
// combination, and those for M the epsilon combination; action slots carry
// exterior minus interior. Unlike two independent calls, coordinates and
// currents are gathered once, and each tensor contraction is performed once.
// The caller must verify identical source/target near lists for both media.
inline void near_field_two_media(const std::array<std::vector<double>,15>& source,
                       const std::vector<int>& ids,const Vec3& target,
                       std::complex<double> ke,std::complex<double> ki,
                       std::complex<double> mu_e,std::complex<double> mu_i,
                       std::complex<double> eps_e,std::complex<double> eps_i,
                       std::array<std::complex<double>,12>& result) {
    const __m512d one=constant(1),zero=constant(0);
    const auto broadcast=[](std::complex<double> z){return Complex(constant(z.real()),constant(z.imag()));};
    const Complex me=broadcast(mu_e),mi=broadcast(mu_i),ee=broadcast(eps_e),ei=broadcast(eps_i);
    Complex sums[12];
    for(size_t s=0;s<ids.size();s+=8) {
        const __mmask8 tail=__mmask8((1u<<std::min(size_t(8),ids.size()-s))-1u);
        const __m256i indices=_mm256_maskz_loadu_epi32(tail,ids.data()+s);
        auto gather=[&](int c){return _mm512_mask_i32gather_pd(zero,tail,indices,source[c].data(),8);};
        const __m512d dx=_mm512_sub_pd(constant(target.x),gather(0));
        const __m512d dy=_mm512_sub_pd(constant(target.y),gather(1));
        const __m512d dz=_mm512_sub_pd(constant(target.z),gather(2));
        __m512d r2=_mm512_add_pd(_mm512_mul_pd(dx,dx),_mm512_add_pd(_mm512_mul_pd(dy,dy),_mm512_mul_pd(dz,dz)));
        const __mmask8 active=tail&_mm512_cmp_pd_mask(r2,constant(1e-28),_CMP_GE_OQ);
        r2=_mm512_mask_mov_pd(one,active,r2);
        const __m512d ir=_mm512_div_pd(one,_mm512_sqrt_pd(r2)),r=_mm512_mul_pd(r2,ir),ir2=_mm512_mul_pd(ir,ir);
        const __m512d x=_mm512_mul_pd(dx,ir),y=_mm512_mul_pd(dy,ir),z=_mm512_mul_pd(dz,ir);
        Complex gradients[2],aa[2],bb[2];
        for(int medium=0;medium<2;++medium){
            const auto k=medium?ki:ke;const auto k2=k*k;
            const __m512d kr=constant(k.real()),kim=constant(k.imag());
            const __m512d arg=_mm512_maskz_mul_pd(active,kr,r);
            __m512d scale=_mm512_maskz_mul_pd(active,constant(INV4PI),ir);
            if(k.imag()!=0)scale=_mm512_mul_pd(scale,_ZGVeN8v_exp(_mm512_maskz_mul_pd(active,constant(-k.imag()),r)));
            const Complex green(_mm512_mul_pd(scale,_ZGVeN8v_cos(arg)),_mm512_mul_pd(scale,_ZGVeN8v_sin(arg)));
            const __m512d kir=_mm512_mul_pd(kim,ir),krr=_mm512_mul_pd(kr,ir);
            gradients[medium]=green*Complex(_mm512_sub_pd(constant(-k.imag()),ir),kr);
            aa[medium]=green*Complex(_mm512_sub_pd(_mm512_mul_pd(constant(3),_mm512_add_pd(ir2,kir)),constant(k2.real())),
                _mm512_sub_pd(_mm512_mul_pd(constant(-3),krr),constant(k2.imag())));
            bb[medium]=green*Complex(_mm512_sub_pd(constant(k2.real()),_mm512_add_pd(kir,ir2)),_mm512_add_pd(constant(k2.imag()),krr));
        }
        const Complex a=aa[0]-aa[1],b=bb[0]-bb[1];
        for(int j=0;j<2;++j){
            const Complex gradient=j?ee*gradients[0]-ei*gradients[1]:me*gradients[0]-mi*gradients[1];
            const Complex qx(gather(3+6*j),gather(4+6*j)),qy(gather(5+6*j),gather(6+6*j)),qz(gather(7+6*j),gather(8+6*j));
            const Complex dot=x*qx+y*qy+z*qz;
            sums[6*j]+=gradient*(y*qx-x*qy);sums[6*j+1]+=gradient*(z*qx-x*qz);sums[6*j+2]+=gradient*(z*qy-y*qz);
            sums[6*j+3]+=x*(a*dot)+b*qx;sums[6*j+4]+=y*(a*dot)+b*qy;sums[6*j+5]+=z*(a*dot)+b*qz;
        }
    }
    for(int c=0;c<12;++c)result[c]+=std::complex<double>(_mm512_reduce_add_pd(sums[c].re),_mm512_reduce_add_pd(sums[c].im));
}
} // namespace avx512
} // namespace bem_cpu
#endif
#endif
