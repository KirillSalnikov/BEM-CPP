#pragma once
#include "block_circulant_fft.h"
#include "prism_circulant_layout.h"
#include "block_circulant_symmetry.h"
#include <functional>
#include <stdexcept>

// Experimental fixed Strang circulant: sampled self-face kernels, shared by
// mirrored/rotated patches. Edges crossing a patch boundary retain local MBJ.
class PrismCirculantInverse {
public:
    using Complex=std::complex<double>;
    using Action=std::function<void(const Complex*,Complex*)>;
    PrismCirculantLayout layout;
    void build(const MullerP2Mesh& mesh,int ref,const Action& action,const Action& local,double shift=0,double opposite_weight=0,int opposite_mask=3);
    // Experimental analytic-symbol path. Kernels already represent the inverse
    // convolution; no operator sampling or second inversion is performed.
    void build_from_inverse_kernels(const MullerP2Mesh& mesh,int ref,const Action& local,
                                    const std::vector<Complex>& side,const std::vector<Complex>& cap);
    bool coupled(int kind)const{return opposite_weight_>0 && (opposite_mask_&(1<<kind));}
    double opposite_weight()const{return opposite_weight_;}
    int opposite_mask()const{return opposite_mask_;}
    void apply(const Complex* rhs,Complex* out,double blend=1);
    // Pure FFT contribution S C R, without the local inverse or row blend.
    void apply_fft_part(const Complex* rhs,Complex* out,int kind=-1,bool symmetric_caps=false);
    void prepare_batch2(bool symmetric_caps=false);
    void apply_fft_part_batch2(const Complex* first,const Complex* second,Complex* first_result,Complex* second_result,
                              bool symmetric_caps=false);
    void prepare_symmetric_caps(const std::vector<CirculantSymmetry>& group);
    // Opt-in GPU gather/scatter path. Full FFT-part actions use one global
    // upload/download; local inverse and Krylov vectors still live on CPU.
    void prepare_device_mapping(bool symmetric_caps=false);
    void enable_device_mapping(bool enabled) {
        if(enabled && !device_mapping_)throw std::logic_error("device mapping not prepared");
        device_mapping_enabled_=enabled;
    }
    bool device_mapping_ready()const{return bool(device_mapping_);}
    size_t device_mapping_storage_bytes()const{return device_mapping_?device_mapping_->storage_bytes():0;}
    void apply_adjoint(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend=1);
    void apply_transpose(const Complex* rhs,Complex* out,const Action& local_adjoint,double blend=1);
    size_t storage_bytes()const{return banks[0].storage_bytes()+banks[1].storage_bytes()+minus[0].storage_bytes()+minus[1].storage_bytes()+symmetric_caps.storage_bytes()+symmetric_caps_minus.storage_bytes()+device_mapping_storage_bytes();}
    int fallback_count()const{return banks[0].fallback_count()+banks[1].fallback_count()+minus[0].fallback_count()+minus[1].fallback_count();}
    bool prepare_cpu() {
        for(auto* bank:{&banks[0],&banks[1],&minus[0],&minus[1],&symmetric_caps,&symmetric_caps_minus})
            if(!bank->prepare_cpu())return false;
        return true;
    }
    void enable_cpu(bool enabled) {
        for(auto* bank:{&banks[0],&banks[1],&minus[0],&minus[1],&symmetric_caps,&symmetric_caps_minus})bank->enable_cpu(enabled);
        cpu_enabled_=enabled;
    }
    size_t cpu_storage_bytes() const {
        return banks[0].cpu_storage_bytes()+banks[1].cpu_storage_bytes()+minus[0].cpu_storage_bytes()+minus[1].cpu_storage_bytes()+symmetric_caps.cpu_storage_bytes()+symmetric_caps_minus.cpu_storage_bytes();
    }
private:
    BlockCirculantFFT banks[2];
    BlockCirculantFFT minus[2],symmetric_caps,symmetric_caps_minus;
    double opposite_weight_=0;
    int opposite_mask_=3;
    bool symmetric_caps_ready=false;
    bool device_mapping_symmetric_=false,device_mapping_enabled_=false,cpu_enabled_=false;
    std::unique_ptr<MappedBlockCirculantFFT> device_mapping_;
    Action local;
    std::vector<Complex> input[2],output[2];
    std::vector<Complex> pair_input[2],pair_output[2];
    void add_fft(const Complex* rhs,Complex* out,int kind,double blend,bool adjoint=false,bool symmetric=false);
};
