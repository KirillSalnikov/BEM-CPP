#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include "muller_fmm_cpu_avx512.h"

namespace bem_cpu {
// Only the central A0 implementation changes. The Taylor derivatives,
// preconditioner, recycling and independent original residual stay intact.
struct TaylorCentralConfig {
    std::string backend="serial";
    bool batch=false,shared=false;
    int tile=32;
    static TaylorCentralConfig from_environment(){
        TaylorCentralConfig result;
        if(const char* value=std::getenv("BEM_TAYLOR_CENTRAL_BACKEND"))result.backend=value;
        if(result.backend!="serial"&&result.backend!="batch"&&result.backend!="shared")
            throw std::invalid_argument("BEM_TAYLOR_CENTRAL_BACKEND must be serial, batch or shared");
        result.batch=result.backend!="serial";result.shared=result.backend=="shared";
        if(const char* value=std::getenv("BEM_TAYLOR_CENTRAL_TILE")){
            char* end=nullptr;const long tile=std::strtol(value,&end,10);
            if(end==value||*end||tile<1||tile>256)throw std::invalid_argument("BEM_TAYLOR_CENTRAL_TILE must be1..256");
            result.tile=int(tile);
        }
        if(result.shared){
            const char* enabled=std::getenv("BEM_CPU_AVX512");
            if(!enabled||std::strcmp(enabled,"1"))throw std::invalid_argument("Taylor shared central action requires BEM_CPU_AVX512=1");
#ifdef BEM_CPU_HAS_AVX512_P2P
            if(!__builtin_cpu_supports("avx512f")||!__builtin_cpu_supports("avx512vl"))
                throw std::invalid_argument("Taylor shared central action requires AVX512 hardware");
#else
            throw std::invalid_argument("Taylor shared central action requires an AVX512 host build");
#endif
        }
        return result;
    }
};
}
