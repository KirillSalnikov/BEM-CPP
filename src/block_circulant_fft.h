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
    void prepare_batch2();
    void apply_batch2(const Complex* first,const Complex* second,Complex* first_result,Complex* second_result);
    void apply_adjoint(const Complex* input,Complex* output);
    void apply_transpose(const Complex* input,Complex* output);
    // Borrowed device workspaces, valid until reinitialization. enqueue_device
    // destroys input and leaves output on the calling thread's default stream.
    // No host transfers or synchronization; the object is not reentrant.
    size_t device_vector_size() const;
    Complex* device_input();
    const Complex* device_output() const;
    void enqueue_device(bool adjoint=false);
    bool prepare_cpu();
    void enable_cpu(bool enabled);
    size_t cpu_storage_bytes() const;
    size_t storage_bytes() const;
    int fallback_count() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// Sum of mapped FFT banks with one upload/download of the global vector.
// Each bank can represent one parity of a pair of opposite patches. Scatter
// maps must be injective within each term; terms execute in stream order.
class MappedBlockCirculantFFT {
public:
    using Complex=BlockCirculantFFT::Complex;
    struct Slot {
        int first=-1,second=-1;
        signed char rhs_first=1,rhs_second=1,solution_first=1,solution_second=1;
    };
    struct Term {
        BlockCirculantFFT* bank=nullptr; // borrowed; must outlive this object
        std::vector<Slot> slots;
        int pair_sign=0; // 0: one patch, +1/-1: normalized sum/difference
    };
    MappedBlockCirculantFFT();
    ~MappedBlockCirculantFFT();
    MappedBlockCirculantFFT(const MappedBlockCirculantFFT&)=delete;
    MappedBlockCirculantFFT& operator=(const MappedBlockCirculantFFT&)=delete;
    static void validate(int dofs,const std::vector<Term>& terms);
    void init(int dofs,const std::vector<Term>& terms);
    void apply(const Complex* input,Complex* output);
    size_t storage_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
