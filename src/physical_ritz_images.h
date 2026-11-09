#pragma once
#include "prism_d6h_model_action.h"
#include "host_memory_estimate.h"
#include <cstdlib>
#include <cstring>
#include <memory>

namespace bem_ritz {
struct ImageConfig {
    std::string model="original";
    bool full_d6h=false;
    bool strict_action=false;
    static ImageConfig from_environment() {
        ImageConfig result;
        if(const char* value=std::getenv("BEM_TMATRIX_RITZ_IMAGE_MODEL"))result.model=value;
        if(result.model!="original" && result.model!="strict" && result.model!="d6h_mixed" && result.model!="d6h_strict")
            throw std::invalid_argument("BEM_TMATRIX_RITZ_IMAGE_MODEL must be original, strict, d6h_mixed or d6h_strict");
        result.full_d6h=result.model=="d6h_strict" || result.model=="d6h_mixed";
        result.strict_action=result.model=="strict" || result.model=="d6h_strict";
        if(result.model!="original") {
            const char* path=std::getenv("BEM_TMATRIX_RITZ_PRIOR");
            if(!path || !*path)throw std::invalid_argument("Ritz image model needs a physical prior file");
            for(const char* name:{"BEM_TMATRIX_SECTOR_GMRES","BEM_TMATRIX_D6H_MESH","BEM_TMATRIX_VERTICAL_PRECONDITIONER"}) {
                const char* value=std::getenv(name);
                if(!value || std::strcmp(value,"1"))throw std::invalid_argument(std::string("D6h Ritz image model requires explicit ")+name+"=1");
            }
        }
        return result;
    }
    size_t estimated_additional_bytes(size_t n)const {
        // Group maps:60*n bytes; model:53*n; construction maps/temporary
        // signs are not all live simultaneously. Reserve128*n plus1MiB.
        return full_d6h?bem_cpu::memory::add(bem_cpu::memory::mul(128,n),1024*1024):0;
    }
};

// Only constructs images for the approximate inverse. Never supplies the
// Krylov or final-residual action. Its extra maps/workspace die after setup.
class PhysicalImages {
    bem_tmatrix::PrismGroupMaps maps_;
    std::unique_ptr<bem_tmatrix::PrismD6hModelAction> model_;
    bem_streaming::Action original_;
    size_t original_calls_=0;
    bool strict_action_=false;
public:
    PhysicalImages(const MullerP2Mesh& mesh,const bem_tmatrix::D6hOrbit& orbit,const ImageConfig& config,
                       bem_streaming::Action mixed,bem_streaming::Action strict)
        :original_(config.strict_action?std::move(strict):std::move(mixed)),strict_action_(config.strict_action) {
        if(!original_)throw std::invalid_argument("missing original Ritz image action");
        if(config.full_d6h) {
            maps_.build(orbit);
            model_.reset(new bem_tmatrix::PrismD6hModelAction(mesh,maps_,original_));
        }
    }
    void apply(const std::complex<double>* input,std::complex<double>* output) {
        if(model_)model_->apply(input,output);
        else {++original_calls_;original_(input,output);}
    }
    size_t mixed_actions()const{return strict_action_?0:(model_?model_->original_actions():original_calls_);}
    size_t strict_actions()const{return strict_action_?(model_?model_->original_actions():original_calls_):0;}
    size_t storage_bytes()const {
        size_t bytes=model_?model_->storage_bytes():0;
        for(int g=0;g<12;++g)bytes+=maps_.image[g].size()*sizeof(int)+maps_.sign[g].size()*sizeof(int8_t);
        return bytes;
    }
};
}
