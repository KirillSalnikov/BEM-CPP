#include "muller_fmm.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
using Complex=std::complex<double>;
using Clock=std::chrono::steady_clock;
int main(int argc,char** argv)try {
    if(argc!=4)throw std::runtime_error("Usage: muller_face_fft_check REF KA REPORT.json");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int ref=std::atoi(argv[1]);const double ka=std::atof(argv[2]);
    const char* compare_flag=std::getenv("BEM_FACE_FFT_CHECK_COMPARE_FLAG");
    if(compare_flag && std::string(compare_flag)!="BEM_FMM_MULTILEVEL_PARITY")throw std::runtime_error("unsupported comparison flag");
    if(ref<1||ref>6||ka<=0)throw std::runtime_error("invalid ref/ka");
    setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);setenv("BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1",1);
    setenv("BEM_FMM_MULTILEVEL","1",1);setenv("BEM_FMM_PAIR_CURRENTS","0",1);
    const int segments=1<<ref;int nz=int(std::ceil(std::sqrt(3.)*segments));nz+=nz%2;
    const int points=7*12*segments*(segments+nz);
    int depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    if(const char* flag=std::getenv("BEM_FACE_FFT_CHECK_DEPTH"))depth=std::max(3,std::min(6,std::atoi(flag)));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
    setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
    setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    MullerFmmOperator op;MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,1.,ref,1.,0,false,true),Complex(ka),Complex(1.3116),options,7,6,5,leaf(depth));
    std::vector<Complex> q(op.system_dofs),reference(q.size()),result(q.size());
    for(size_t i=0;i<q.size();++i)q[i]=Complex(std::sin(.173*(i+1)),std::cos(.127*(i+1)));
    auto timed=[&](bool strict,std::vector<Complex>& out) {
        const auto start=Clock::now();if(strict)op.matvec_strict(q.data(),out.data());else op.matvec(q.data(),out.data());
        return std::chrono::duration<double>(Clock::now()-start).count();
    };
    auto error=[&] {double a=0,b=0;for(size_t i=0;i<result.size();++i){a+=std::norm(result[i]-reference[i]);b+=std::norm(reference[i]);}return std::sqrt(a/b);};
    op.set_fmm_near_fp32(true);timed(true,reference);
    std::vector<double> baseline_mixed,baseline_strict;
    for(int i=0;i<3;++i){baseline_mixed.push_back(timed(false,result));baseline_strict.push_back(timed(true,reference));}
    const auto start=Clock::now();op.enable_prism_face_fft(ref);
    const double setup=std::chrono::duration<double>(Clock::now()-start).count();
    timed(true,result);const double strict_error=error();
    timed(false,result);const double mixed_error=error();
    std::vector<double> fft_mixed,fft_strict;
    for(int i=0;i<3;++i){fft_mixed.push_back(timed(false,result));fft_strict.push_back(timed(true,result));}
    // Keep the same prepared operator, input, and live GPU load for ABBA pairs.
    // The strict-only backend retains its original MLFMA hierarchies.
    std::vector<double> alternating_fmm,alternating_fft;
    if(op.prism_fft_strict_only && !compare_flag) {
        auto baseline=[&] {
            auto exterior=std::move(op.prism_fft_exterior),interior=std::move(op.prism_fft_interior);
            try {
                alternating_fmm.push_back(timed(true,result));
            }catch(...) {
                op.prism_fft_exterior=std::move(exterior);op.prism_fft_interior=std::move(interior);throw;
            }
            op.prism_fft_exterior=std::move(exterior);op.prism_fft_interior=std::move(interior);
        };
        for(int i=0;i<3;++i) {
            baseline();alternating_fft.push_back(timed(true,result));
            alternating_fft.push_back(timed(true,result));baseline();
        }
    }
    std::vector<double> variant_strict[2],variant_mixed[2];double variant_strict_error=0,variant_mixed_error=0;
    if(compare_flag) {
        const char* previous=std::getenv(compare_flag);const bool existed=previous;
        const std::string saved=previous?previous:"";
        std::vector<Complex> strict_reference(q.size()),mixed_reference(q.size());
        setenv(compare_flag,"0",1);timed(true,strict_reference);timed(false,mixed_reference);
        auto difference=[&](const std::vector<Complex>& a) {
            double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::norm(result[i]-a[i]);d+=std::norm(a[i]);}
            const double e=std::sqrt(n/d);if(!std::isfinite(e))throw std::runtime_error("nonfinite variant action");return e;
        };
        for(int enabled:{0,1,1,0,1,0,0,1}) {
            setenv(compare_flag,enabled?"1":"0",1);
            variant_strict[enabled].push_back(timed(true,result));variant_strict_error=std::max(variant_strict_error,difference(strict_reference));
            variant_mixed[enabled].push_back(timed(false,result));variant_mixed_error=std::max(variant_mixed_error,difference(mixed_reference));
        }
        if(existed)setenv(compare_flag,saved.c_str(),1);else unsetenv(compare_flag);
    }
    std::ofstream file(argv[3]);file<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<op.system_dofs
      <<",\"baseline_depth\":"<<depth<<",\"baseline\":\"three_band_MLFMA\""
      <<",\"backend\":\""<<op.backend_name()<<"\",\"setup_s\":"<<setup<<",\"strict_relative_error\":"<<strict_error<<",\"mixed_relative_error\":"<<mixed_error;
    auto array=[&](const char* key,const std::vector<double>& v){file<<",\""<<key<<"\":[";for(size_t i=0;i<v.size();++i)file<<(i?",":"")<<v[i];file<<"]";};
    array("FMM_mixed_s",baseline_mixed);array("FMM_strict_s",baseline_strict);array("hybrid_mixed_s",fft_mixed);array("hybrid_strict_s",fft_strict);
    array("alternating_FMM_strict_s",alternating_fmm);array("alternating_hybrid_strict_s",alternating_fft);
    if(compare_flag) {
        file<<",\"compare_flag\":\""<<compare_flag<<"\",\"variant_strict_relative_error\":"<<variant_strict_error
            <<",\"variant_mixed_relative_error\":"<<variant_mixed_error;
        array("variant_strict_off_s",variant_strict[0]);array("variant_strict_on_s",variant_strict[1]);
        array("variant_mixed_off_s",variant_mixed[0]);array("variant_mixed_on_s",variant_mixed[1]);
    }
    file<<"}\n";
    std::printf("FULL MULLER ref=%d ka=%g strict error %.8e mixed error %.8e\n",ref,ka,strict_error,mixed_error);
    op.cleanup();return std::isfinite(strict_error)&&std::isfinite(mixed_error)&&strict_error<2e-5&&mixed_error<5e-4&&variant_strict_error<2e-12&&variant_mixed_error<2e-5?0:2;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
