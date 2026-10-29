#pragma once
#include "block_circulant_fft.h"
#include "prism_circulant_layout.h"
#include <functional>

// Experimental fixed Strang circulant: sampled self-face kernels, shared by
// mirrored/rotated patches. Edges crossing a patch boundary retain local MBJ.
class PrismCirculantInverse {
public:
    using Complex=std::complex<double>;
    using Action=std::function<void(const Complex*,Complex*)>;
    PrismCirculantLayout layout;
    void build(const MullerP2Mesh& mesh,int ref,const Action& action,const Action& local,double shift=0);
    void apply(const Complex* rhs,Complex* out,double blend=1);
    // Pure FFT contribution S C R, without the local inverse or row blend.
    void apply_fft_part(const Complex* rhs,Complex* out,int kind=-1);
    void apply_adjoint(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend=1);
    void apply_transpose(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend=1);
    size_t storage_bytes()const{return banks[0].storage_bytes()+banks[1].storage_bytes();}
    int fallback_count()const{return banks[0].fallback_count()+banks[1].fallback_count();}
private:
    BlockCirculantFFT banks[2];
    Action local;
    std::vector<Complex> input[2],output[2];
};
