#include "host_cpu_checker_config.h"
#include "host_memory_estimate.h"
#include <cstdio>
#include <functional>
int main()try{
    using bem_cpu::CpuCheckerConfig;
    const char* requirements[]={"BEM_TMATRIX_HYBRID","BEM_TMATRIX_SECTOR_GMRES","BEM_TMATRIX_D6H_MESH",
        "BEM_TMATRIX_FACE_FFT","BEM_FACE_FFT_STRICT_ONLY","BEM_FACE_FFT_TARGET_FACES","BEM_FACE_FFT_REUSE_CROSS"};
    const auto reset=[&]{for(auto key:requirements)unsetenv(key);
        for(auto key:{"BEM_CPU_CHECKER_BACKEND","BEM_CPU_FACE_FFT_TILE","BEM_TMATRIX_CPU_STRICT","BEM_CPU_AVX512"})unsetenv(key);};
    reset();if(CpuCheckerConfig::from_environment("sphere",0).backend!="fmm")throw std::runtime_error("default backend changed");
    int accepted=1,rejected=0;
    const auto reject=[&](std::function<void()> action){bool refused=false;try{action();}catch(const std::exception&){refused=true;}
        if(!refused)throw std::runtime_error("invalid CPU checker configuration accepted");
        ++rejected;};
    for(auto mode:{"face_fft","face_fft_batch","face_fft_shared"}){
        reset();for(auto key:requirements)setenv(key,"1",1);setenv("BEM_CPU_CHECKER_BACKEND",mode,1);
        const auto c=CpuCheckerConfig::from_environment("prism",2);
        if(!c.face||c.batch!=(std::string(mode)!="face_fft")||c.shared_near!=(std::string(mode)=="face_fft_shared"))throw std::runtime_error("backend mode lost");
        ++accepted;
        for(auto key:requirements){if(!c.batch && std::string(key)=="BEM_FACE_FFT_REUSE_CROSS")continue;
            unsetenv(key);reject([&]{CpuCheckerConfig::from_environment("prism",2);});setenv(key,"1",1);}
        for(auto shape:{"sphere","tetra"})reject([&]{CpuCheckerConfig::from_environment(shape,2);});
        reject([&]{CpuCheckerConfig::from_environment("prism",0);});
        setenv("BEM_TMATRIX_CPU_STRICT","0",1);reject([&]{CpuCheckerConfig::from_environment("prism",2);});unsetenv("BEM_TMATRIX_CPU_STRICT");
        for(auto tile:{"","0","-1","257","32x"}){setenv("BEM_CPU_FACE_FFT_TILE",tile,1);reject([&]{CpuCheckerConfig::from_environment("prism",2);});}
        for(auto tile:{"1","32","256"}){setenv("BEM_CPU_FACE_FFT_TILE",tile,1);CpuCheckerConfig::from_environment("prism",2);++accepted;}
        if(c.shared_near){setenv("BEM_CPU_AVX512","0",1);reject([&]{CpuCheckerConfig::from_environment("prism",2);});}
    }
    reset();for(auto mode:{"","auto","shared","face_fft_sharedx"}){
        setenv("BEM_CPU_CHECKER_BACKEND",mode,1);reject([&]{CpuCheckerConfig::from_environment("prism",2);});}
    const size_t maximum=std::numeric_limits<size_t>::max();
    reject([&]{bem_cpu::memory::add(maximum,1);});reject([&]{bem_cpu::memory::mul(maximum,2);});
    if(bem_cpu::memory::mul(maximum,0)!=0||bem_cpu::memory::mul(maximum,1)!=maximum||bem_cpu::memory::add(maximum,0)!=maximum)
        throw std::runtime_error("valid workspace arithmetic rejected");
    std::printf("PASS CPU checker policy: accepted%d rejected%d; overflow controls passed\n",accepted,rejected);
}catch(const std::exception& e){std::fprintf(stderr,"CPU checker policy: %s\n",e.what());return 1;}
