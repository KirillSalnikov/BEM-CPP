#include "muller_fmm.h"
#include "prism_symmetric_circulant.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>

using C=std::complex<double>;
using V=std::vector<C>;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static double error(const V& actual,const V& expected){
    double a=0,b=0;for(size_t i=0;i<actual.size();++i){a+=std::norm(actual[i]-expected[i]);b+=std::norm(expected[i]);}
    const double e=std::sqrt(a/std::max(1.,b));
    if(!std::isfinite(e))throw std::runtime_error("nonfinite batch result");return e;
}
int main(int argc,char** argv)try {
    if(argc!=4)throw std::invalid_argument("usage: check REF KA REPORT");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    const std::string profile=std::getenv("BEM_PROBE_CAPTURE")?std::getenv("BEM_PROBE_CAPTURE"):"";
    if(!profile.empty() && profile!="mixed" && profile!="inverse")throw std::invalid_argument("profile range");
    if(ref<2||ref>5||ka<=0)throw std::invalid_argument("ref/ka");
    setvbuf(stdout,nullptr,_IOLBF,0);
    setenv("BEM_MULLER_BATCH2_HOST","1",1);setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);
    setenv("BEM_MULLER_GPU_STRICT_ONLY","0",1);setenv("BEM_CIRCULANT_CPU_FFT","0",1);
    const int segments=1<<ref;int nz=std::max(segments,int(std::ceil(std::sqrt(3.)*(10./7)*segments)));nz+=nz%2;
    const int points=7*12*segments*(segments+nz);
    const int depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
    setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
    setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;struct Cleanup{MullerFmmOperator& op;~Cleanup(){op.cleanup();}} cleanup{op};
    const auto setup=Clock::now();op.init(regular_prism(6,10./7,ref,1.,0,false,true),C(ka),C(1.3116),options,7,12,5,leaf(depth));
    if(!op.host_batch2_available())throw std::runtime_error("batch unavailable");
    const double setup_s=elapsed(setup);const int n=op.system_dofs;
    V x(n),y(n),a(n),b(n),pa(n),pb(n),zero(n);
    std::mt19937 rng(4317);std::normal_distribution<double> normal;
    for(int i=0;i<n;++i){x[i]=C(normal(rng),normal(rng));y[i]=C(normal(rng),normal(rng));}
    double defects[2]={0,0};std::vector<double> action_single[2],action_pair[2];
    for(int mixed=0;mixed<2;++mixed){
        op.set_fmm_near_fp32(mixed);op.matvec(x.data(),a.data());
        for(const auto* second:{&y,&x,&zero}){
            op.matvec(second->data(),b.data());op.matvec_batch2(x.data(),second->data(),pa.data(),pb.data());
            defects[mixed]=std::max(defects[mixed],std::max(error(pa,a),error(pb,b)));
        }
        if(defects[mixed]>(mixed?2e-5:2e-12))throw std::runtime_error("FMM pair differs from serial actions");
        if(mixed && profile=="mixed")cudaProfilerStart();
        for(int paired:{0,1,1,0,1,0,0,1}){
            const auto start=Clock::now();
            if(paired)op.matvec_batch2(x.data(),y.data(),pa.data(),pb.data());
            else{op.matvec(x.data(),pa.data());op.matvec(y.data(),pb.data());}
            (paired?action_pair:action_single)[mixed].push_back(elapsed(start));
        }
        if(mixed && profile=="mixed")cudaProfilerStop();
    }
    op.set_fmm_near_fp32(true);
    bem_tmatrix::D6hOrbit group;group.build(op.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(group);
    PrismOrbitInverse local;local.build(op,maps,50);
    double inverse_defect=0,symmetry_defect=0,fft_setup_s=0;
    size_t pair_bytes=0;std::vector<double> inverse_single,inverse_pair;
    for(double opposite:{0.,.25}){
        const auto begin=Clock::now();PrismCirculantInverse circ;
        circ.build(op.mesh,ref,[&](const C* i,C* o){op.matvec(i,o);},[&](const C* i,C* o){local.apply(i,o);},0.,opposite,3);
        PrismSymmetricCirculant sym;sym.build(group,circ,local);
        if(!sym.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("D6h proof failed");
        const size_t before=circ.storage_bytes();sym.prepare_batch2();pair_bytes=std::max(pair_bytes,circ.storage_bytes()-before);
        fft_setup_s+=elapsed(begin);sym.apply_full_fft_symmetry(x.data(),a.data(),.5);
        for(const auto* second:{&y,&x,&zero}){
            sym.apply_full_fft_symmetry(second->data(),b.data(),.5);
            sym.apply_full_fft_symmetry_batch2(x.data(),second->data(),pa.data(),pb.data(),.5);
            inverse_defect=std::max(inverse_defect,std::max(error(pa,a),error(pb,b)));
        }
        V transformed(n),expected(n);
        for(bool mirror:{false,true}){
            group.transform(x,transformed,mirror,true);
            sym.apply_full_fft_symmetry_batch2(transformed.data(),y.data(),pa.data(),pb.data(),.5);
            group.transform(a,expected,mirror,false);symmetry_defect=std::max(symmetry_defect,error(pa,expected));
        }
        std::vector<int> target,sign;bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,target,sign);
        for(int i=0;i<n;++i){transformed[target[i]]=-double(sign[i])*x[i];expected[target[i]]=double(sign[i])*a[i];}
        sym.apply_full_fft_symmetry_batch2(transformed.data(),y.data(),pa.data(),pb.data(),.5);
        symmetry_defect=std::max(symmetry_defect,error(pa,expected));
        if(inverse_defect>2e-12||symmetry_defect>1e-8)throw std::runtime_error("FFT pair or D6h symmetry failed");
        if(opposite==0 && profile=="inverse")cudaProfilerStart();
        if(opposite==0)for(int paired:{0,1,1,0,1,0,0,1}){
            const auto start=Clock::now();
            if(paired)sym.apply_full_fft_symmetry_batch2(x.data(),y.data(),pa.data(),pb.data(),.5);
            else{sym.apply_full_fft_symmetry(x.data(),pa.data(),.5);sym.apply_full_fft_symmetry(y.data(),pb.data(),.5);}
            (paired?inverse_pair:inverse_single).push_back(elapsed(start));
        }
        if(opposite==0 && profile=="inverse")cudaProfilerStop();
    }
    std::ofstream report(argv[3]);report<<std::setprecision(17)<<"{\"passed\":true,\"ref\":"<<ref<<",\"ka\":"<<ka
        <<",\"dofs\":"<<n<<",\"setup_s\":"<<setup_s<<",\"fft_setup_s\":"<<fft_setup_s
        <<",\"host_correction\":"<<(op.gpu_host_correction?"true":"false")
        <<",\"strict_action_defect\":"<<defects[0]<<",\"mixed_action_defect\":"<<defects[1]
        <<",\"inverse_defect\":"<<inverse_defect<<",\"D6h_defect\":"<<symmetry_defect<<",\"extra_fft_pair_bytes\":"<<pair_bytes;
    auto array=[&](const char* key,const std::vector<double>& v){report<<",\""<<key<<"\":[";for(size_t j=0;j<v.size();++j)report<<(j?",":"")<<v[j];report<<"]";};
    array("strict_serial_two_s",action_single[0]);array("strict_pair_s",action_pair[0]);
    array("mixed_serial_two_s",action_single[1]);array("mixed_pair_s",action_pair[1]);
    array("inverse_serial_two_s",inverse_single);array("inverse_pair_s",inverse_pair);report<<"}\n";
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"batch check: %s\n",e.what());return 1;}
