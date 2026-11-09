#ifndef BEM_HOST_SHARED_NEAR_AVX512_H
#define BEM_HOST_SHARED_NEAR_AVX512_H
#include "muller_fmm_cpu_avx512.h"

#ifdef BEM_CPU_HAS_AVX512_P2P
namespace bem_cpu { namespace avx512 {
// One geometric kernel tile for several independent currents. The bounded
// stack tile avoids an O(points * batches) SoA copy and an O(pairs) cache.
// All six current components use the original FP64 Green derivatives.
inline void near_field_batch(const std::array<std::vector<double>,15>& coordinates,
                             const std::vector<int>& ids,const Vec3& target,
                             std::complex<double> k,
                             const std::array<std::complex<double>,6>* const* charges,
                             std::array<std::complex<double>,12>* const* results,int batches) {
    static_assert(sizeof(std::array<std::complex<double>,6>)==12*sizeof(double),"packed current layout");
    struct alignas(64) Kernel {
        __m512d x,y,z;
        Complex gradient,a,b;
        __m256i current_indices;
        __mmask8 tail;
    };
    const size_t block_count=16,source_tile=8*block_count;
    Kernel kernels[block_count];
    const __m512d one=constant(1),zero=constant(0),kr=constant(k.real()),ki=constant(k.imag());
    const std::complex<double> k2=k*k;
    for(size_t begin=0;begin<ids.size();begin+=source_tile) {
        const size_t blocks=(std::min(source_tile,ids.size()-begin)+7)/8;
        for(size_t block=0;block<blocks;++block) {
            auto& g=kernels[block];const size_t s=begin+8*block;
            g.tail=__mmask8((1u<<std::min(size_t(8),ids.size()-s))-1u);
            const __m256i indices=_mm256_maskz_loadu_epi32(g.tail,ids.data()+s);
            g.current_indices=_mm256_mullo_epi32(indices,_mm256_set1_epi32(12));
            auto gather=[&](int c){return _mm512_mask_i32gather_pd(zero,g.tail,indices,coordinates[c].data(),8);};
            const __m512d dx=_mm512_sub_pd(constant(target.x),gather(0));
            const __m512d dy=_mm512_sub_pd(constant(target.y),gather(1));
            const __m512d dz=_mm512_sub_pd(constant(target.z),gather(2));
            __m512d r2=_mm512_add_pd(_mm512_mul_pd(dx,dx),_mm512_add_pd(_mm512_mul_pd(dy,dy),_mm512_mul_pd(dz,dz)));
            const __mmask8 active=g.tail&_mm512_cmp_pd_mask(r2,constant(1e-28),_CMP_GE_OQ);
            r2=_mm512_mask_mov_pd(one,active,r2);
            const __m512d ir=_mm512_div_pd(one,_mm512_sqrt_pd(r2)),r=_mm512_mul_pd(r2,ir);
            const __m512d arg=_mm512_maskz_mul_pd(active,kr,r);
            g.x=_mm512_mul_pd(dx,ir);g.y=_mm512_mul_pd(dy,ir);g.z=_mm512_mul_pd(dz,ir);
            __m512d scale=_mm512_maskz_mul_pd(active,constant(INV4PI),ir);
            if(k.imag()!=0)scale=_mm512_mul_pd(scale,_ZGVeN8v_exp(_mm512_maskz_mul_pd(active,constant(-k.imag()),r)));
            const Complex green(_mm512_mul_pd(scale,_ZGVeN8v_cos(arg)),_mm512_mul_pd(scale,_ZGVeN8v_sin(arg)));
            const __m512d ir2=_mm512_mul_pd(ir,ir),kir=_mm512_mul_pd(ki,ir),krr=_mm512_mul_pd(kr,ir);
            g.gradient=green*Complex(_mm512_sub_pd(constant(-k.imag()),ir),kr);
            g.a=green*Complex(_mm512_sub_pd(_mm512_mul_pd(constant(3),_mm512_add_pd(ir2,kir)),constant(k2.real())),
                             _mm512_sub_pd(_mm512_mul_pd(constant(-3),krr),constant(k2.imag())));
            g.b=green*Complex(_mm512_sub_pd(constant(k2.real()),_mm512_add_pd(kir,ir2)),
                             _mm512_add_pd(constant(k2.imag()),krr));
        }
        for(int batch=0;batch<batches;++batch) {
            Complex sums[12];
            // std::complex specifies real/imaginary array access through T*.
            const double* source=reinterpret_cast<const double*>(charges[batch]);
            for(size_t block=0;block<blocks;++block) {
                const auto& g=kernels[block];
                auto gather=[&](int c){return _mm512_mask_i32gather_pd(zero,g.tail,g.current_indices,source+c,8);};
                for(int current=0;current<2;++current) {
                    const int j=6*current;
                    const Complex qx(gather(j),gather(j+1)),qy(gather(j+2),gather(j+3)),qz(gather(j+4),gather(j+5));
                    const Complex dot=g.x*qx+g.y*qy+g.z*qz;
                    sums[j]+=g.gradient*(g.y*qx-g.x*qy);
                    sums[j+1]+=g.gradient*(g.z*qx-g.x*qz);
                    sums[j+2]+=g.gradient*(g.z*qy-g.y*qz);
                    sums[j+3]+=g.x*(g.a*dot)+g.b*qx;
                    sums[j+4]+=g.y*(g.a*dot)+g.b*qy;
                    sums[j+5]+=g.z*(g.a*dot)+g.b*qz;
                }
            }
            for(int c=0;c<12;++c)(*results[batch])[c]+=std::complex<double>(_mm512_reduce_add_pd(sums[c].re),_mm512_reduce_add_pd(sums[c].im));
        }
    }
}
} }
#endif
#endif
