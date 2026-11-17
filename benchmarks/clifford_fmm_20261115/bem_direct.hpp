#ifndef BEM_FMM_DIRECT_HPP
#define BEM_FMM_DIRECT_HPP

// Included after the pinned FMM3D template-kernels.hpp. FP64 throughout.
// FMM3D already parallelizes target boxes. Each call uses only bounded stack
// workspace: 32 source pairs and at most 16 density channels, no nested OpenMP.
// The radial derivative H times charge is shared by all three Cartesian axes.
// This changes FP64 rounding, not the kernel, cutoff, quadrature or FMM eps.
// Range reduction/sign selection adapted from the pinned FMM3D/SCTL
// intrin-wrapper.hpp. Evaluate sin(x)/x and cos(x) as degree-6 polynomials
// in x*x on [-pi/4,pi/4], without the reciprocal-sqrt cosine reconstruction.
// Coefficients: 80-digit Chebyshev interpolation, rounded to FP64; see
// benchmarks/dirac_fmm_leaf_20261116/coefficients.py. No runtime dependency.
// Independent direct-sum and phase-quadrant tests guard the complete kernel.
namespace bem_cheby {
using namespace sctl;
  template <bool EXTRA, class VData> void polynomial_sincos(VData& sinx, VData& cosx, const VData& x) {
    using Real = typename VData::ScalarType;
    using Int = typename IntegerType<sizeof(Real)>::value;
    using IntVec = VecData<Int, VData::Size>;
    static_assert(TypeTraits<Real>::Type == DataType::Real, "Expected real type!");

    static constexpr Integer SigBits = TypeTraits<Real>::SigBits;
    static constexpr Real pi_over_2 = const_pi<Real>()/2;
    static constexpr Real neg_pi_over_2 = -const_pi<Real>()/2;
    static constexpr Real inv_pi_over_2 = 1 / pi_over_2;

    union {
      Int Cint = 0 + (((Int)1) << (SigBits - 1)) + ((SigBits + ((((Int)1)<<(sizeof(Real)*8 - SigBits - 2))-1)) << SigBits);
      Real Creal;
    };
    VData real_offset(set1_intrin<VData>(Creal));

    VData x_int(fma_intrin(x, set1_intrin<VData>(inv_pi_over_2), real_offset));
    VData x_(sub_intrin(x_int, real_offset)); // x_ <-- round(x*inv_pi_over_2)
    VData x1 = fma_intrin(x_, set1_intrin<VData>(neg_pi_over_2), x);

    VData x2=mul_intrin(x1,x1);
    VData s1=mul_intrin(x1,EvalPolynomial(x2,
      1.0,-0.16666666666666616,0.008333333333320363,
      -0.000198412698286503,2.755731337640013e-06,
      -2.5050716974102745e-08,1.5894736651849094e-10));
    VData c1=EvalPolynomial(x2,
      1.0,-0.4999999999999925,0.041666666666472306,
      -0.0013888888869978658,2.4801578538562985e-05,
      -2.755523388484742e-07,2.0630454379662294e-09);

    IntVec vec_zero(zero_intrin<IntVec>());
    auto xAnd1 = reinterpret_mask<Mask<VData>>(comp_intrin<ComparisonType::eq>(and_intrin(reinterpret_intrin<IntVec>(x_int), set1_intrin<IntVec>(1)), vec_zero));
    auto xAnd2 = reinterpret_mask<Mask<VData>>(comp_intrin<ComparisonType::eq>(and_intrin(reinterpret_intrin<IntVec>(x_int), set1_intrin<IntVec>(2)), vec_zero));

    VData s2(select_intrin(xAnd1, s1,                    c1 ));
    VData c2(select_intrin(xAnd1, c1, unary_minus_intrin(s1)));
    sinx = select_intrin(xAnd2, s2, unary_minus_intrin(s2));
    cosx = select_intrin(xAnd2, c2, unary_minus_intrin(c2));
  }
template<bool EXTRA,class V> void sincos(V& sn,V& cs,const V& x) {
 typename V::VData s,c;polynomial_sincos<EXTRA>(s,c,x.get());sn.set(s);cs.set(c);
}
}

namespace bem_fmm {
template<int VL, bool REAL_K> void blocked_direct(
 const int64_t* nd,const double* zk,const double* src,const double* q,
 const int64_t* ns,const double* tgt,const int64_t* nt,
 double* pot,double* grad,const double* threshold) {
  constexpr int BS=32;
  using Vec=sctl::Vec<double,VL>;
  const Vec threshold2=*threshold * *threshold;
  for(int64_t t=0;t<*nt;t+=VL) {
    alignas(64) double coord[3][VL]={};
    int count=std::min<int64_t>(VL,*nt-t);
    for(int a=0;a<3;++a)for(int lane=0;lane<count;++lane)coord[a][lane]=tgt[3*(t+lane)+a];
    Vec xt[3]={Vec::LoadAligned(coord[0]),Vec::LoadAligned(coord[1]),Vec::LoadAligned(coord[2])};
    Vec accum[16][4][2];
    for(int d=0;d<*nd;++d)for(int a=0;a<4;++a)for(int c=0;c<2;++c)accum[d][a][c]=Vec::Zero();
    for(int64_t b=0;b<*ns;b+=BS) {
      Vec kernel[BS][4][2];
      int bs=std::min<int64_t>(BS,*ns-b);
      for(int s=0;s<bs;++s) {
        Vec dx[3],r2=Vec::Zero();
        for(int a=0;a<3;++a){dx[a]=xt[a]-Vec::Load1(src+3*(b+s)+a);r2+=dx[a]*dx[a];}
        Vec ri=sctl::approx_rsqrt<-1>(r2,(r2>threshold2));
        Vec r=r2*ri,ri2=ri*ri;ri=ri*const_inv4pi<double>();
        Vec phase=zk[0]*r,attenuation=REAL_K?Vec::Zero():-zk[1]*r,sn,cs;
        bem_cheby::sincos<false>(sn,cs,phase);
        Vec ex=REAL_K?Vec(1.0):sctl::approx_exp<-1>(attenuation);
        Vec g0=cs*ex*ri,g1=sn*ex*ri;
        Vec h0=((attenuation-1.0)*g0-phase*g1)*ri2;
        Vec h1=(phase*g0+(attenuation-1.0)*g1)*ri2;
        kernel[s][0][0]=g0;kernel[s][0][1]=g1;
        kernel[s][1][0]=h0;kernel[s][1][1]=h1;
        kernel[s][2][0]=dx[0];kernel[s][2][1]=dx[1];kernel[s][3][0]=dx[2];
      }
      for(int d=0;d<*nd;++d) {
        Vec acc[4][2];
        for(int a=0;a<4;++a)for(int c=0;c<2;++c)acc[a][c]=accum[d][a][c];
        for(int s=0;s<bs;++s) {
          Vec qr=Vec::Load1(q+2*((b+s)* *nd+d)),qi=Vec::Load1(q+2*((b+s)* *nd+d)+1);
          acc[0][0]+=kernel[s][0][0]*qr-kernel[s][0][1]*qi;
          acc[0][1]+=kernel[s][0][1]*qr+kernel[s][0][0]*qi;
          Vec hr=kernel[s][1][0]*qr-kernel[s][1][1]*qi;
          Vec hi=kernel[s][1][1]*qr+kernel[s][1][0]*qi;
          acc[1][0]+=hr*kernel[s][2][0];acc[1][1]+=hi*kernel[s][2][0];
          acc[2][0]+=hr*kernel[s][2][1];acc[2][1]+=hi*kernel[s][2][1];
          acc[3][0]+=hr*kernel[s][3][0];acc[3][1]+=hi*kernel[s][3][0];
        }
        for(int a=0;a<4;++a)for(int c=0;c<2;++c)accum[d][a][c]=acc[a][c];
      }
    }
    alignas(64) double result[VL];
    for(int d=0;d<*nd;++d)for(int a=0;a<4;++a)for(int c=0;c<2;++c) {
      accum[d][a][c].StoreAligned(result);
      for(int lane=0;lane<count;++lane) {
        if(a==0)pot[2*((t+lane)* *nd+d)+c]+=result[lane];
        else grad[2*(((t+lane)*3+a-1)* *nd+d)+c]+=result[lane];
      }
    }
  }
}

// Return false for unsupported channel counts, preserving the upstream path.
template<int VL> bool direct(const int64_t* nd,const double* zk,
 const double* src,const double* q,const int64_t* ns,const double* tgt,
 const int64_t* nt,double* pot,double* grad,const double* threshold) {
  if(*nd<1 || *nd>16) return false;
  if(*ns<1 || *nt<1) return true;
  if(zk[1]==0) blocked_direct<VL,true>(nd,zk,src,q,ns,tgt,nt,pot,grad,threshold);
  else blocked_direct<VL,false>(nd,zk,src,q,ns,tgt,nt,pot,grad,threshold);
  return true;
}
}
#endif
