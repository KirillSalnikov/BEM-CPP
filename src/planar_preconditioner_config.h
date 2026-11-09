#pragma once
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace bem_osrc {
// Parse before opening the worker socket. A numerical symbol remains the
// default; experimental settings must never silently select another inverse.
struct PreconditionerConfig {
    std::string symbol="sampled";
    bool analytic=false;
    double damping=0;
    static double number(const char* name,double fallback) {
        const char* raw=std::getenv(name);if(!raw)return fallback;
        char* end=nullptr;const double value=std::strtod(raw,&end);
        if(end==raw || *end || !std::isfinite(value))
            throw std::invalid_argument(std::string("invalid ")+name);
        return value;
    }
    static PreconditionerConfig from_environment(const std::string& shape,int ref,double ka,std::complex<double> index) {
        PreconditionerConfig config;
        if(const char* value=std::getenv("BEM_CIRCULANT_SYMBOL"))config.symbol=value;
        if(config.symbol!="sampled" && config.symbol!="planar_bdm")
            throw std::invalid_argument("BEM_CIRCULANT_SYMBOL must be sampled or planar_bdm");
        if(config.symbol=="sampled") {
            if(std::getenv("BEM_CIRCULANT_PLANAR_DAMPING"))
                throw std::invalid_argument("planar damping requires BEM_CIRCULANT_SYMBOL=planar_bdm");
            return config;
        }
        config.analytic=true;config.damping=number("BEM_CIRCULANT_PLANAR_DAMPING",.01);
        if(!(config.damping>0 && config.damping<=1))throw std::invalid_argument("planar damping must be in (0,1]");
        if(shape!="prism" || ref<2 || !std::isfinite(ka) || ka<=0 || !std::isfinite(index.real()) ||
                !std::isfinite(index.imag()) || index.real()<=0 || index.imag()<0)
            throw std::invalid_argument("planar BDM inverse requires ref>=2 regular prism, positive ka and passive index");
        for(const char* name:{"BEM_TMATRIX_SECTOR_GMRES","BEM_TMATRIX_D6H_MESH","BEM_TMATRIX_VERTICAL_PRECONDITIONER"}) {
            const char* value=std::getenv(name);
            if(!value || std::strcmp(value,"1"))
                throw std::invalid_argument(std::string("planar BDM inverse requires explicit ")+name+"=1");
        }
        const double blend=number("BEM_TMATRIX_CIRCULANT_BLEND",0);
        if(!(blend>0 && blend<=1))throw std::invalid_argument("planar BDM inverse requires circulant blend in (0,1]");
        if(number("BEM_CIRCULANT_SHIFT",0)!=0 || number("BEM_CIRCULANT_OPPOSITE_WEIGHT",0)!=0)
            throw std::invalid_argument("planar BDM inverse does not implement sampled-symbol shift or opposite-face coupling");
        return config;
    }
};
}
