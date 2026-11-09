#ifndef BEM_HOST_CPU_CHECKER_CONFIG_H
#define BEM_HOST_CPU_CHECKER_CONFIG_H
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
namespace bem_cpu {
struct CpuCheckerConfig {
    std::string backend="fmm";
    bool face=false,batch=false,shared_near=false;
    int tile=32;
    static CpuCheckerConfig from_environment(const std::string& shape,int ref){
        CpuCheckerConfig config;
        if(const char* value=std::getenv("BEM_CPU_CHECKER_BACKEND"))config.backend=value;
        if(config.backend=="fmm")return config;
        if(config.backend!="face_fft"&&config.backend!="face_fft_batch"&&config.backend!="face_fft_shared")
            throw std::invalid_argument("BEM_CPU_CHECKER_BACKEND must be fmm, face_fft, face_fft_batch or face_fft_shared");
        config.face=true;config.batch=config.backend!="face_fft";config.shared_near=config.backend=="face_fft_shared";
        if(shape!="prism"||ref<1)throw std::invalid_argument("CPU face checker requires a refined regular prism");
        const auto require=[](const char* key){const char* value=std::getenv(key);
            if(!value||std::strcmp(value,"1"))throw std::invalid_argument(std::string("CPU face checker requires explicit ")+key+"=1");};
        for(const char* key:{"BEM_TMATRIX_HYBRID","BEM_TMATRIX_SECTOR_GMRES","BEM_TMATRIX_D6H_MESH",
                            "BEM_TMATRIX_FACE_FFT","BEM_FACE_FFT_STRICT_ONLY","BEM_FACE_FFT_TARGET_FACES"})require(key);
        if(const char* value=std::getenv("BEM_TMATRIX_CPU_STRICT"))
            if(std::strcmp(value,"1"))throw std::invalid_argument("CPU face checker requires CPU strict checks");
        if(config.batch)require("BEM_FACE_FFT_REUSE_CROSS");
        if(config.shared_near)if(const char* value=std::getenv("BEM_CPU_AVX512"))
            if(std::strcmp(value,"1"))throw std::invalid_argument("CPU shared checker requires AVX512");
        if(const char* value=std::getenv("BEM_CPU_FACE_FFT_TILE")){
            char* end=nullptr;const long tile=std::strtol(value,&end,10);
            if(end==value||*end||tile<1||tile>256)throw std::invalid_argument("BEM_CPU_FACE_FFT_TILE must be1..256");
            config.tile=int(tile);
        }
        return config;
    }
};
}
#endif
