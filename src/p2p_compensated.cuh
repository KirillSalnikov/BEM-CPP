#pragma once
#include <cuda_runtime.h>
#include <cmath>

// Opt-in accurate FP32 expansion for real-k P2P. Keep the original
// P2PDoubleSingle implementation available as an independent control.
struct P2PCompensated {
    float hi, lo;
    __device__ P2PCompensated() = default;
    __device__ P2PCompensated(int x):hi(float(x)),lo(0){}
    __device__ P2PCompensated(float x):hi(x),lo(0){}
    __device__ P2PCompensated(double x):hi(float(x)),lo(float(x-double(hi))){}
    __device__ explicit operator double() const {return double(hi)+double(lo);}
    __device__ static P2PCompensated parts(float h,float l) {
        P2PCompensated x;x.hi=h;x.lo=l;return x;
    }
};
__device__ __forceinline__ P2PCompensated cp_sum(float a,float b) {
    const float s=__fadd_rn(a,b),v=__fsub_rn(s,a);
    return P2PCompensated::parts(s,__fadd_rn(__fsub_rn(a,__fsub_rn(s,v)),__fsub_rn(b,v)));
}
__device__ __forceinline__ P2PCompensated cp_normalize(float h,float l) {
    const float s=__fadd_rn(h,l);
    return P2PCompensated::parts(s,__fsub_rn(l,__fsub_rn(s,h)));
}
__device__ __forceinline__ P2PCompensated operator+(P2PCompensated a,P2PCompensated b) {
    const auto s=cp_sum(a.hi,b.hi);
    return cp_normalize(s.hi,__fadd_rn(s.lo,__fadd_rn(a.lo,b.lo)));
}
__device__ __forceinline__ P2PCompensated operator-(P2PCompensated a) {
    return P2PCompensated::parts(-a.hi,-a.lo);
}
__device__ __forceinline__ P2PCompensated operator-(P2PCompensated a,P2PCompensated b) {return a+(-b);}
__device__ __forceinline__ P2PCompensated operator*(P2PCompensated a,P2PCompensated b) {
    const float p=__fmul_rn(a.hi,b.hi);
    float e=__fmaf_rn(a.hi,b.hi,-p);
    e=__fmaf_rn(a.hi,b.lo,e);e=__fmaf_rn(a.lo,b.hi,e);
    // A product residual stays small relative to p. Consumers recover it in
    // their sum; normalizing it here would duplicate that work.
    return P2PCompensated::parts(p,e);
}
__device__ __forceinline__ P2PCompensated cp_fma_raw(P2PCompensated a,P2PCompensated b,P2PCompensated c) {
    const float p=__fmul_rn(a.hi,b.hi);
    const auto s=cp_sum(p,c.hi);
    float e=__fmaf_rn(a.hi,b.hi,-p);
    e=__fmaf_rn(a.hi,b.lo,e);e=__fmaf_rn(a.lo,b.hi,e);
    return P2PCompensated::parts(s.hi,__fadd_rn(__fadd_rn(e,c.lo),s.lo));
}
__device__ __forceinline__ P2PCompensated cp_fma(P2PCompensated a,P2PCompensated b,P2PCompensated c) {
    const auto r=cp_fma_raw(a,b,c);return cp_normalize(r.hi,r.lo);
}
__device__ __forceinline__ P2PCompensated& operator+=(P2PCompensated& a,P2PCompensated b) {a=a+b;return a;}
__device__ __forceinline__ bool operator==(P2PCompensated a,P2PCompensated b) {return a.hi==b.hi&&a.lo==b.lo;}
__device__ __forceinline__ bool operator<(P2PCompensated a,P2PCompensated b) {return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}

__device__ __forceinline__ P2PCompensated p2p_inverse_sqrt(P2PCompensated x) {
    P2PCompensated y(rsqrtf(x.hi));
    // Evaluate the Newton residual with FMA product residuals. Unlike
    // 1-x*y*y in FP32, this keeps the bits lost in the nearly equal subtraction.
#pragma unroll
    for(int step=0;step<2;++step) {
        const float p=__fmul_rn(y.hi,y.hi);
        const float e=__fmaf_rn(2.f*y.hi,y.lo,__fmaf_rn(y.hi,y.hi,-p));
        float r=__fmaf_rn(-x.hi,p,1.f);
        r=__fmaf_rn(-x.hi,e,r);r=__fmaf_rn(-x.lo,p,r);
        y=cp_normalize(y.hi,__fmaf_rn(.5f*y.hi,r,y.lo));
    }
    return y;
}
__device__ __forceinline__ void cp_sincos(P2PCompensated x,P2PCompensated* sine,P2PCompensated* cosine) {
    if(fabsf(x.hi)>10000.f) {
        double s,c;sincos(double(x),&s,&c);*sine=P2PCompensated(s);*cosine=P2PCompensated(c);return;
    }
    const int quadrant=__float2int_rn(x.hi*0.63661977236758134308f);
    const auto r=cp_fma(P2PCompensated(-quadrant),P2PCompensated(1.57079632679489661923),x);
    const auto z=r*r;
    // On |r| <= pi/4, the FP32 tail first enters at r^9 (sine)
    // or r^10 (cosine). Its absolute rounding error is below 1e-13;
    // the leading terms and range reduction retain both components.
    float st=__fmaf_rn(float(-1./1307674368000.),z.hi,float(1./6227020800.));
    st=__fmaf_rn(st,z.hi,float(-1./39916800.));
    st=__fmaf_rn(st,z.hi,float(1./362880.));
    P2PCompensated sp(st);
    sp=cp_fma(z,sp,P2PCompensated(-1./5040.));
    sp=cp_fma(z,sp,P2PCompensated(1./120.));
    sp=cp_fma(z,sp,P2PCompensated(-1./6.));
    const auto s=cp_fma(r*z,sp,r);
    float ct=__fmaf_rn(float(-1./87178291200.),z.hi,float(1./479001600.));
    ct=__fmaf_rn(ct,z.hi,float(-1./3628800.));
    P2PCompensated cp(ct);
    cp=cp_fma(z,cp,P2PCompensated(1./40320.));
    cp=cp_fma(z,cp,P2PCompensated(-1./720.));
    cp=cp_fma(z,cp,P2PCompensated(1./24.));
    cp=cp_fma(z,cp,P2PCompensated(-.5));
    const auto c=cp_fma(z,cp,P2PCompensated(1));
    switch(quadrant&3) {
        case 0:*sine=s;*cosine=c;break;case 1:*sine=c;*cosine=-s;break;
        case 2:*sine=-s;*cosine=-c;break;default:*sine=-c;*cosine=s;
    }
}
__device__ inline P2PCompensated p2p_attenuation(P2PCompensated x) {return P2PCompensated(exp(double(x)));}
template<bool FastTrig> __device__ inline void p2p_pair_sine_cosine(P2PCompensated x,P2PCompensated* s,P2PCompensated* c) {cp_sincos(x,s,c);}
