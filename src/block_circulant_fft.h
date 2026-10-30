#pragma once
#include <complex>
#include <memory>
#include <vector>

// Fixed inverse of a block-periodic 2D convolution. Channel-major host arrays,
// kernel[(row*channels+column)*nx*ny+dx*ny+dy]. No physical periodicity is assumed.
class BlockCirculantFFT {
public:
    using Complex=std::complex<double>;
    BlockCirculantFFT();
    ~BlockCirculantFFT();
    BlockCirculantFFT(const BlockCirculantFFT&)=delete;
    BlockCirculantFFT& operator=(const BlockCirculantFFT&)=delete;
    void init(int nx,int ny,int channels,int batches,const std::vector<Complex>& kernel,double shift=0);
    // Export/replace the inverse convolution, without reinverting it.
    std::vector<Complex> inverse_kernel()const;
    void init_inverse_kernel(int nx,int ny,int channels,int batches,const std::vector<Complex>& kernel);
    void apply(const Complex* input,Complex* output);
    void apply_adjoint(const Complex* input,Complex* output);
    void apply_transpose(const Complex* input,Complex* output);
    size_t storage_bytes() const;
    int fallback_count() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
