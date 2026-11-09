#pragma once
#include <cuda_runtime.h>
#include <cmath>

// Experimental double-single arithmetic: two nonoverlapping FP32 components.
// Explicit RN operations keep the error-free transforms from being contracted.
struct P2PDoubleSingle {
    float hi,lo;
    __device__ P2PDoubleSingle()=default;
    __device__ P2PDoubleSingle(int x):hi(float(x)),lo(0){}
    __device__ P2PDoubleSingle(float x):hi(x),lo(0){}
    __device__ P2PDoubleSingle(double x):hi(float(x)),lo(float(x-double(hi))){}
    __device__ explicit operator double()const{return double(hi)+double(lo);}
    __device__ static P2PDoubleSingle parts(float h,float l){P2PDoubleSingle x;x.hi=h;x.lo=l;return x;}
};
__device__ __forceinline__ P2PDoubleSingle ds_sum(float a,float b){
    const float s=__fadd_rn(a,b),v=__fsub_rn(s,a);
    const float e=__fadd_rn(__fsub_rn(a,__fsub_rn(s,v)),__fsub_rn(b,v));
    return P2PDoubleSingle::parts(s,e);
}
__device__ __forceinline__ P2PDoubleSingle operator+(P2PDoubleSingle a,P2PDoubleSingle b){
    P2PDoubleSingle t=ds_sum(a.hi,b.hi);
    return ds_sum(t.hi,__fadd_rn(t.lo,__fadd_rn(a.lo,b.lo)));
}
__device__ __forceinline__ P2PDoubleSingle operator-(P2PDoubleSingle a){return P2PDoubleSingle::parts(-a.hi,-a.lo);}
__device__ __forceinline__ P2PDoubleSingle operator-(P2PDoubleSingle a,P2PDoubleSingle b){return a+(-b);}
__device__ __forceinline__ P2PDoubleSingle operator*(P2PDoubleSingle a,P2PDoubleSingle b){
    const float p=__fmul_rn(a.hi,b.hi);
    float e=__fmaf_rn(a.hi,b.hi,-p);
    e=__fmaf_rn(a.hi,b.lo,e);e=__fmaf_rn(a.lo,b.hi,e);e=__fmaf_rn(a.lo,b.lo,e);
    return ds_sum(p,e);
}
__device__ __forceinline__ P2PDoubleSingle& operator+=(P2PDoubleSingle& a,P2PDoubleSingle b){a=a+b;return a;}
__device__ __forceinline__ bool operator==(P2PDoubleSingle a,P2PDoubleSingle b){return a.hi==b.hi&&a.lo==b.lo;}
__device__ __forceinline__ bool operator<(P2PDoubleSingle a,P2PDoubleSingle b){return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}
__device__ __forceinline__ P2PDoubleSingle ds_rsqrt(P2PDoubleSingle x){
    P2PDoubleSingle y(rsqrtf(x.hi));
    y=y+y*(P2PDoubleSingle(1)-x*y*y)*P2PDoubleSingle(.5f);
    y=y+y*(P2PDoubleSingle(1)-x*y*y)*P2PDoubleSingle(.5f);
    return y;
}
__device__ __forceinline__ void ds_sincos(P2PDoubleSingle x,P2PDoubleSingle* sine,P2PDoubleSingle* cosine){
    // The tested ka=40 geometry uses small arguments. Keep a safe FP64
    // fallback for very large phases, where this range reduction is unsuitable.
    if(fabsf(x.hi)>10000.f){double s,c;sincos(double(x),&s,&c);*sine=P2PDoubleSingle(s);*cosine=P2PDoubleSingle(c);return;}
    const int quadrant=__float2int_rn(x.hi*0.63661977236758134308f);
    const P2PDoubleSingle r=x-P2PDoubleSingle(quadrant)*P2PDoubleSingle(1.57079632679489661923),z=r*r;
    P2PDoubleSingle sp(-1./1307674368000.);
    sp=P2PDoubleSingle(1./6227020800.)+z*sp;sp=P2PDoubleSingle(-1./39916800.)+z*sp;
    sp=P2PDoubleSingle(1./362880.)+z*sp;sp=P2PDoubleSingle(-1./5040.)+z*sp;
    sp=P2PDoubleSingle(1./120.)+z*sp;sp=P2PDoubleSingle(-1./6.)+z*sp;
    const P2PDoubleSingle s=r+r*z*sp;
    P2PDoubleSingle cp(-1./87178291200.);
    cp=P2PDoubleSingle(1./479001600.)+z*cp;cp=P2PDoubleSingle(-1./3628800.)+z*cp;
    cp=P2PDoubleSingle(1./40320.)+z*cp;cp=P2PDoubleSingle(-1./720.)+z*cp;
    cp=P2PDoubleSingle(1./24.)+z*cp;cp=P2PDoubleSingle(-.5)+z*cp;
    const P2PDoubleSingle c=P2PDoubleSingle(1)+z*cp;
    switch(quadrant&3){case 0:*sine=s;*cosine=c;break;case 1:*sine=c;*cosine=-s;break;case 2:*sine=-s;*cosine=-c;break;default:*sine=-c;*cosine=s;}
}
__device__ inline P2PDoubleSingle p2p_inverse_sqrt(P2PDoubleSingle x){return ds_rsqrt(x);}
// Absorbing media use an accurate FP64 exponential for now. Real-k kernels
// eliminate this branch. No speed claim is made for the absorbing case.
__device__ inline P2PDoubleSingle p2p_attenuation(P2PDoubleSingle x){return P2PDoubleSingle(exp(double(x)));}
template<bool FastTrig>__device__ inline void p2p_pair_sine_cosine(P2PDoubleSingle x,P2PDoubleSingle* s,P2PDoubleSingle* c){ds_sincos(x,s,c);}
