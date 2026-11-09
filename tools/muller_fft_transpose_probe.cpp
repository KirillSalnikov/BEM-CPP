// Full original Muller action: mass, both media, FFT/FMM, singular correction.
// Bilinear identities test the transpose of the actual forward backend, not
// agreement with a second approximate forward method.
#include "muller_fmm.h"
#include "prism_face_fft.h"
#include "tmatrix_symmetry_warm_start.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <sys/resource.h>
using C=std::complex<double>;using Vector=std::vector<C>;
using Clock=std::chrono::steady_clock;
double error(const Vector& x,const Vector& y){
    double a=0,b=0;for(size_t j=0;j<x.size();++j){a+=std::norm(x[j]-y[j]);b+=std::norm(y[j]);}
    return std::sqrt(a/std::max(b,1e-300));
}
int main(int argc,char** argv)try {
    if(argc!=6)throw std::invalid_argument("REF KA ASPECT RI_IMAG REPORT_JSON");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),aspect=std::stod(argv[3]),imag=std::stod(argv[4]);
    if(ref<1||ref>5||!std::isfinite(ka)||ka<=0||!std::isfinite(aspect)||aspect<=0||!std::isfinite(imag)||imag<0)
        throw std::invalid_argument("invalid geometry/material");
    setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);setenv("BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1",1);
    setenv("BEM_FMM_MULTILEVEL","1",1);setenv("BEM_FMM_PAIR_CURRENTS","0",1);
    const int segments=1<<ref;int nz=std::max(segments,int(std::ceil(std::sqrt(3.)*aspect*segments)));nz+=nz%2;
    const int points=7*12*segments*(segments+nz),depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
    setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
    setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;const auto start=Clock::now();
    op.init(regular_prism(6,aspect,ref,1.,0,false,true),C(ka),C(1.3116,imag),options,7,6,5,leaf(depth));
    op.enable_prism_face_fft(ref,aspect);
    const size_t pairs[2]={op.prism_fft_exterior->cached_near_pairs(),op.prism_fft_interior->cached_near_pairs()};
    const size_t radials[2]={op.prism_fft_exterior->cached_near_radials(),op.prism_fft_interior->cached_near_radials()};
    if(const char* flag=std::getenv("BEM_FACE_FFT_NEAR_CACHE"))if(std::string(flag)!="0" && (!pairs[0]||!pairs[1]))
        throw std::runtime_error("requested cache branch was not exercised");
    const double setup=std::chrono::duration<double>(Clock::now()-start).count();
    const int n=op.system_dofs;Vector x(n),seed(n),ax(n),aty(n),ahy(n);
    std::mt19937 rng(719332);std::normal_distribution<double> normal;
    double transpose[2]={},adjoint[2]={},forward_s[2]={},reverse_s[2]={},alias_error=0,forward_repeat=0;
    for(int mixed=0;mixed<2;++mixed)for(int trial=0;trial<2;++trial){
        op.set_fmm_near_fp32(mixed);
        for(int j=0;j<n;++j){x[j]=C(normal(rng),normal(rng));seed[j]=C(normal(rng),normal(rng));}
        const auto begin=Clock::now();op.matvec(x.data(),ax.data());
        forward_s[mixed]+=std::chrono::duration<double>(Clock::now()-begin).count();
        const auto before=Clock::now();op.matvec_transpose(seed.data(),aty.data());
        reverse_s[mixed]+=std::chrono::duration<double>(Clock::now()-before).count();
        op.matvec_adjoint(seed.data(),ahy.data());
        C left=0,right=0,left_h=0,right_h=0;double fn=0,sn=0;
        for(int j=0;j<n;++j){
            left+=seed[j]*ax[j];right+=aty[j]*x[j];left_h+=std::conj(seed[j])*ax[j];right_h+=std::conj(ahy[j])*x[j];
            fn+=std::norm(ax[j]);sn+=std::norm(seed[j]);
        }
        transpose[mixed]=std::max(transpose[mixed],std::abs(left-right)/std::sqrt(fn*sn));
        adjoint[mixed]=std::max(adjoint[mixed],std::abs(left_h-right_h)/std::sqrt(fn*sn));
        Vector repeat(n);op.matvec(x.data(),repeat.data());forward_repeat=std::max(forward_repeat,error(repeat,ax));
        std::printf("mixed=%d trial=%d transpose=%.17g adjoint=%.17g\n",mixed,trial,transpose[mixed],adjoint[mixed]);
    }
    // Strict wrappers must select FFT even when the mixed backend is FMM,
    // restore the prior precision state, and support overlapping buffers.
    op.set_fmm_near_fp32(true);op.matvec_adjoint_strict(seed.data(),ahy.data());
    Vector alias=seed;op.matvec_adjoint_strict(alias.data(),alias.data());alias_error=error(alias,ahy);
    if(!op.fmm_near_fp32)throw std::runtime_error("strict reverse did not restore precision");
    op.matvec_transpose_strict(seed.data(),aty.data());alias=seed;
    op.matvec_transpose_strict(alias.data(),alias.data());alias_error=std::max(alias_error,error(alias,aty));
    if(!op.fmm_near_fp32)throw std::runtime_error("strict transpose did not restore precision");
    bem_tmatrix::D6hOrbit group;group.build(op.mesh);
    std::vector<int> vertical,sign;group.vertical_image_map(op.mesh,vertical,sign);
    double symmetry[3]={};
    for(int g=0;g<3;++g){
        Vector transformed(n),expected(n),actual(n);
        if(g<2){group.transform(seed,transformed,g==1,true);group.transform(ahy,expected,g==1,false);}
        else for(int j=0;j<n;++j){transformed[vertical[j]]=-double(sign[j])*seed[j];expected[vertical[j]]=double(sign[j])*ahy[j];}
        op.matvec_adjoint_strict(transformed.data(),actual.data());symmetry[g]=error(actual,expected);
    }
    std::fill(seed.begin(),seed.end(),C(0));op.matvec_adjoint_strict(seed.data(),ahy.data());
    double zero=0;for(C z:ahy)zero=std::max(zero,std::abs(z));
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    const bool pass=std::isfinite(transpose[0]+transpose[1]+adjoint[0]+adjoint[1]+alias_error+forward_repeat+zero+
        symmetry[0]+symmetry[1]+symmetry[2]) &&
        transpose[0]<2e-12 && adjoint[0]<2e-12 && transpose[1]<2e-5 && adjoint[1]<2e-5 &&
        alias_error<2e-12 && forward_repeat<2e-5 && zero<1e-14 &&
        *std::max_element(symmetry,symmetry+3)<2e-5;
    std::ofstream report(argv[5]);report<<std::setprecision(17)<<"{\"passed\":"<<(pass?"true":"false")
        <<",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"aspect\":"<<aspect<<",\"ri_imag\":"<<imag<<",\"dofs\":"<<n
        <<",\"strict_only\":"<<(op.prism_fft_strict_only?"true":"false")<<",\"setup_s\":"<<setup
        <<",\"cached_near_pairs\":["<<pairs[0]<<','<<pairs[1]<<"],\"cached_near_radials\":["<<radials[0]<<','<<radials[1]<<']'
        <<",\"transpose_error\":["<<transpose[0]<<','<<transpose[1]<<"],\"adjoint_error\":["<<adjoint[0]<<','<<adjoint[1]
        <<"],\"forward_action_s\":["<<forward_s[0]/2<<','<<forward_s[1]/2<<"],\"transpose_action_s\":["<<reverse_s[0]/2<<','<<reverse_s[1]/2
        <<"],\"alias_error\":"<<alias_error<<",\"forward_repeat_error\":"<<forward_repeat<<",\"zero_output\":"<<zero
        <<",\"adjoint_D6h_generator_defects\":["<<symmetry[0]<<','<<symmetry[1]<<','<<symmetry[2]
        <<"],\"max_rss_bytes\":"<<size_t(usage.ru_maxrss)*1024<<",\"full_T_or_SO3_test\":false}\n";
    report.close();if(!report)throw std::runtime_error("report write failed");
    op.cleanup();return pass?0:2;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
