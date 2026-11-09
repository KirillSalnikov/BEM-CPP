#ifndef BEM_PRISM_FACE_FFT_H
#define BEM_PRISM_FACE_FFT_H
#include "fmm.h"
#include <array>
#include <memory>
#include <vector>

// Complete prism point action: same-face lattice FFT plus cross-face FMM.
// Every ordered point pair belongs to exactly one of the two paths.
class PrismFaceFFT {
public:
    PrismFaceFFT();
    ~PrismFaceFFT();
    PrismFaceFFT(const PrismFaceFFT&)=delete;
    PrismFaceFFT& operator=(const PrismFaceFFT&)=delete;
    // A matching geometry can share exact near-distance indices across media.
    void init(const std::vector<double>& xyz,int refinement,cdouble k,int digits=5,
              const PrismFaceFFT* shared_geometry=nullptr,double aspect=1.);
    void evaluate(const cdouble* qx,const cdouble* qy,const cdouble* qz,
                  cdouble* curl,cdouble* hessian,bool mixed);
    // Transpose of this same backend, including rotated cross hierarchies and
    // cached near coefficients. Host-staged seeds/results; no dense matrix.
    void evaluate_transpose(const cdouble* curl_seed,const cdouble* hessian_seed,
                            cdouble* qx,cdouble* qy,cdouble* qz,bool mixed);
    void evaluate_device(const double* const real[3],const double* const imag[3],bool mixed);
    const HelmholtzFMM& result_view() const;
    size_t fft_storage_bytes() const;
    size_t cached_near_pairs() const;
    size_t cached_near_radials() const;
    // Immutable, borrowed CPU geometry. The prepared GPU backend must outlive
    // a consumer; no CUDA calls or shared work buffers are exposed here.
    struct HostView {
        struct Lattice {int nx,ny;std::array<double,3> u,v;const std::vector<double>* offsets;};
        struct Face {
            std::array<double,9> frame,rotation;
            const std::vector<int> *global,*slot,*other,*cross_global;
            int engine;
            const HelmholtzFMM *fine,*middle,*coarse;
        };
        cdouble k;bool target_faces;
        Lattice side,cap;std::array<Face,8> faces;
    };
    HostView host_view() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
#endif
