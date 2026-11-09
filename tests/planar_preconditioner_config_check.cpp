#include "planar_preconditioner_config.h"
#include <cstdio>
#include <functional>
#include <limits>
int main()try {
    using Config=bem_osrc::PreconditionerConfig;int accepted=0,rejected=0;
    const char* flags[]={"BEM_TMATRIX_SECTOR_GMRES","BEM_TMATRIX_D6H_MESH","BEM_TMATRIX_VERTICAL_PRECONDITIONER"};
    const auto reset=[&]{for(auto flag:flags)unsetenv(flag);
        for(auto flag:{"BEM_CIRCULANT_SYMBOL","BEM_CIRCULANT_PLANAR_DAMPING","BEM_TMATRIX_CIRCULANT_BLEND",
                       "BEM_CIRCULANT_SHIFT","BEM_CIRCULANT_OPPOSITE_WEIGHT"})unsetenv(flag);};
    const auto valid=[&]{reset();setenv("BEM_CIRCULANT_SYMBOL","planar_bdm",1);
        for(auto flag:flags)setenv(flag,"1",1);setenv("BEM_TMATRIX_CIRCULANT_BLEND",".5",1);};
    const auto reject=[&](std::function<void()> action){bool refused=false;try{action();}catch(const std::exception&){refused=true;}
        if(!refused)throw std::runtime_error("invalid planar configuration accepted");++rejected;};
    reset();if(Config::from_environment("sphere",0,3.,1.3).analytic)throw std::runtime_error("default changed");++accepted;
    for(auto mode:{"","planar","auto","sampledx"}){reset();setenv("BEM_CIRCULANT_SYMBOL",mode,1);
        reject([&]{Config::from_environment("prism",2,3.,1.3);});}
    reset();setenv("BEM_CIRCULANT_PLANAR_DAMPING",".01",1);reject([&]{Config::from_environment("prism",2,3.,1.3);});
    valid();auto c=Config::from_environment("prism",2,3.,1.3);
    if(!c.analytic||c.damping!=.01)throw std::runtime_error("analytic default wrong");++accepted;
    for(auto flag:flags)for(const char* value:std::initializer_list<const char*>{nullptr,"0","2","true"}) {
        valid();if(value)setenv(flag,value,1);else unsetenv(flag);
        reject([&]{Config::from_environment("prism",2,3.,1.3);});
    }
    for(auto setting:{"BEM_CIRCULANT_PLANAR_DAMPING","BEM_TMATRIX_CIRCULANT_BLEND"}) {
        for(auto value:{"","0","-1","1.001","nan","inf",".01x"}) {
            valid();setenv(setting,value,1);reject([&]{Config::from_environment("prism",2,3.,1.3);});
        }
        for(auto value:{".001",".01",".5","1"}) {
            valid();setenv(setting,value,1);Config::from_environment("prism",2,3.,{1.3,.5});++accepted;
        }
    }
    for(auto setting:{"BEM_CIRCULANT_SHIFT","BEM_CIRCULANT_OPPOSITE_WEIGHT"})
        for(auto value:{".01","-1","nan","0x"}){valid();setenv(setting,value,1);reject([&]{Config::from_environment("prism",2,3.,1.3);});}
    valid();for(auto shape:{"sphere","tetra","mesh.obj"})reject([&]{Config::from_environment(shape,2,3.,1.3);});
    for(int ref:{0,1})reject([&]{Config::from_environment("prism",ref,3.,1.3);});
    for(double ka:{0.,-1.,std::numeric_limits<double>::infinity()})reject([&]{Config::from_environment("prism",2,ka,1.3);});
    for(auto index:{std::complex<double>(0),std::complex<double>(-1),std::complex<double>(1.3,-.1),
                    std::complex<double>(1.3,std::numeric_limits<double>::quiet_NaN())})
        reject([&]{Config::from_environment("prism",2,3.,index);});
    std::printf("PASS planar preconditioner policy accepted%d rejected%d\n",accepted,rejected);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
