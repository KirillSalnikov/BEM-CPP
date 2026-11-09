#ifndef BEM_P2P_RADIAL_FACTORS_H
#define BEM_P2P_RADIAL_FACTORS_H

#ifdef __CUDACC__
#define BEM_P2P_RADIAL_INLINE __host__ __device__ __forceinline__
#else
#define BEM_P2P_RADIAL_INLINE inline
#endif

namespace bem_p2p {

// G = exp(ikr)/(4*pi*r), g = G'(r)/r. For real k:
// bG = -g, aG = G''(r)-G'(r)/r = -3*g-k^2*G.
// Reuse the gradient already needed for the curl. Coordinates, sin/cos,
// charges and all arithmetic remain double precision in the strict kernel.
template<class Real>
BEM_P2P_RADIAL_INLINE void radial_from_gradient_real(
    Real g_re, Real g_im, Real green_re, Real green_im, Real k_squared,
    Real& a_re, Real& a_im, Real& b_re, Real& b_im)
{
    a_re = -Real(3)*g_re - k_squared*green_re;
    a_im = -Real(3)*g_im - k_squared*green_im;
    b_re = -g_re;
    b_im = -g_im;
}

} // namespace bem_p2p
#undef BEM_P2P_RADIAL_INLINE
#endif
