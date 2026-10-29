#ifndef BEM_PRISM_FACE_FFT_H
#define BEM_PRISM_FACE_FFT_H
#include "fmm.h"
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
              const PrismFaceFFT* shared_geometry=nullptr);
    void evaluate(const cdouble* qx,const cdouble* qy,const cdouble* qz,
                  cdouble* curl,cdouble* hessian,bool mixed);
    void evaluate_device(const double* const real[3],const double* const imag[3],bool mixed);
    const HelmholtzFMM& result_view() const;
    size_t fft_storage_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p;
};
#endif
