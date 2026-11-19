// CPU timing control for thin spectral panels. Convert FP16 lanes in registers.
#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <omp.h>

static inline float horizontal_sum(__m256 a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
    s = _mm_hadd_ps(s,s);
    return _mm_cvtss_f32(_mm_hadd_ps(s,s));
}

extern "C" void panel_dot(int nf,int rows,int cols,int rhs,const std::uint16_t* factor,
                          const float* transposed_input,float* output,int threads) {
#pragma omp parallel for collapse(2) schedule(static) num_threads(threads)
    for(int f=0;f<nf;++f)for(int r=0;r<rows;++r) {
        const std::uint16_t* a=factor+2LL*(f*rows+r)*cols;
        for(int h=0;h<rhs;++h) {
            const float* x=transposed_input+2LL*(f*rhs+h)*cols;
            __m256 real=_mm256_setzero_ps(),imag=_mm256_setzero_ps();
            int k=0;
            for(;k+4<=cols;k+=4) {
                __m256 w=_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a+2*k)));
                __m256 v=_mm256_loadu_ps(x+2*k);
                real=_mm256_fmadd_ps(w,v,real);
                imag=_mm256_fmadd_ps(w,_mm256_permute_ps(v,0xb1),imag);
            }
            // Alternating signs convert ar*xr + ai*xi into the complex real part.
            real=_mm256_xor_ps(real,_mm256_setr_ps(0.f,-0.f,0.f,-0.f,0.f,-0.f,0.f,-0.f));
            float re=horizontal_sum(real),im=horizontal_sum(imag);
            for(;k<cols;++k) {
                float ar=_cvtsh_ss(a[2*k]),ai=_cvtsh_ss(a[2*k+1]);
                re+=ar*x[2*k]-ai*x[2*k+1];im+=ar*x[2*k+1]+ai*x[2*k];
            }
            float* dest=output+2LL*((f*rows+r)*rhs+h);
            dest[0]=re;dest[1]=im;
        }
    }
}

extern "C" void packed_dot(int nf,int rows,int cols,int rhs,const std::uint16_t* symbol,
                           const float* scale,const float* input,float* out,int threads) {
    const __m256 odd_sign=_mm256_setr_ps(0.f,-0.f,0.f,-0.f,0.f,-0.f,0.f,-0.f);
#pragma omp parallel for schedule(static) num_threads(threads)
    for(int f=0;f<nf;++f)for(int row=0;row<rows;++row)for(int h=0;h<rhs;++h) {
        const auto* a=symbol+2LL*(3LL*f*rows+row)*cols;
        const auto* b=a+2LL*rows*cols;
        const auto* c=b+2LL*rows*cols;
        const float* j=input+4LL*(f*rhs+h)*cols;
        const float* m=j+2*cols;
        __m256 tr=_mm256_setzero_ps(),ti=tr,br=tr,bi=tr;
        int k=0;
        for(;k+4<=cols;k+=4) {
            auto load=[k](const std::uint16_t* w){
                return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+2*k)));
            };
            __m256 av=load(a),bv=load(b),cv=load(c);
            __m256 jv=_mm256_loadu_ps(j+2*k),mv=_mm256_loadu_ps(m+2*k);
            __m256 js=_mm256_permute_ps(jv,0xb1),ms=_mm256_permute_ps(mv,0xb1);
            tr=_mm256_fmadd_ps(bv,mv,_mm256_fmadd_ps(av,jv,tr));
            ti=_mm256_fmadd_ps(bv,ms,_mm256_fmadd_ps(av,js,ti));
            br=_mm256_fnmadd_ps(av,mv,_mm256_fmadd_ps(cv,jv,br));
            bi=_mm256_fnmadd_ps(av,ms,_mm256_fmadd_ps(cv,js,bi));
        }
        float ur=horizontal_sum(_mm256_xor_ps(tr,odd_sign)),ui=horizontal_sum(ti);
        float vr=horizontal_sum(_mm256_xor_ps(br,odd_sign)),vi=horizontal_sum(bi);
        for(;k<cols;++k) {
            float ar=_cvtsh_ss(a[2*k]),ai=_cvtsh_ss(a[2*k+1]);
            float br0=_cvtsh_ss(b[2*k]),bi0=_cvtsh_ss(b[2*k+1]);
            float cr=_cvtsh_ss(c[2*k]),ci=_cvtsh_ss(c[2*k+1]);
            float jr=j[2*k],ji=j[2*k+1],mr=m[2*k],mi=m[2*k+1];
            ur+=(ar*jr-ai*ji)+(br0*mr-bi0*mi);ui+=(ar*ji+ai*jr)+(br0*mi+bi0*mr);
            vr+=(cr*jr-ci*ji)-(ar*mr-ai*mi);vi+=(cr*ji+ci*jr)-(ar*mi+ai*mr);
        }
        float* top=out+2LL*((2LL*f*rows+row)*rhs+h);
        float* bot=top+2LL*rows*rhs;
        top[0]=ur*scale[f];top[1]=ui*scale[f];bot[0]=vr*scale[f];bot[1]=vi*scale[f];
    }
}
