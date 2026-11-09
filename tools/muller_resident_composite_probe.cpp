#include "muller_fmm.h"
#include "prism_symmetric_circulant.h"
#include "resident_prism_inverse.h"
#include "resident_sector_operator.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <fstream>
#include <iomanip>
using namespace bem_tmatrix;
using C=std::complex<double>;using V=std::vector<C>;using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
static void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
static double error(const V& a,const V& b){long double u=0,v=0;for(size_t i=0;i<a.size();++i){u+=std::norm(a[i]-b[i]);v+=std::norm(b[i]);}
    const double result=std::sqrt(double(u/std::max(v,1e-300L)));if(!std::isfinite(result))throw std::runtime_error("nonfinite composite error");return result;}
int main(int argc,char** argv)try {
    if(argc!=5)throw std::invalid_argument("usage: REF KA REPORT CAPTURE(none|host|device)");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);const std::string capture=argv[4];
    if(ref<2 || ref>5 || !std::isfinite(ka) || ka<=0 || (capture!="none" && capture!="host" && capture!="device"))throw std::invalid_argument("composite probe parameters");
    setvbuf(stdout,nullptr,_IOLBF,0);std::ofstream report(argv[3]);if(!report)throw std::runtime_error("report open");report<<std::setprecision(17);
    auto start=Clock::now();MullerFmmOperator op;struct Cleanup{MullerFmmOperator& op;~Cleanup(){op.cleanup();}} cleanup{op};
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,ref==2?12:6,5,64,false,2,0,.96,nullptr,3);
    op.set_fmm_near_fp32(true);op.enable_prism_face_fft(ref,10./7);const int n=op.system_dofs;
    if(!op.device_matvec_available())throw std::runtime_error("composite probe requires full GPU assembly/correction");
    D6hOrbit orbit;orbit.build(op.mesh);PrismGroupMaps maps;maps.build(orbit);PrismSymmetrySectors sectors;sectors.build(maps);
    PrismOrbitInverse local;local.build(op,maps,50);PrismCirculantInverse fft;
    fft.build(op.mesh,ref,[&](const C* a,C* b){op.matvec(a,b);},[&](const C* a,C* b){local.apply(a,b);});
    PrismSymmetricCirculant symmetric;symmetric.build(orbit,fft,local);if(!symmetric.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("D6h FFT proof");
    const double common_setup=seconds(start);start=Clock::now();fft.prepare_device_mapping(true);fft.enable_device_mapping(false);
    ResidentPrismInverse inverse;ResidentSectorOperator composite;
    {const auto transform=bem_sector_device::Layout::capture(sectors);
        inverse.init(bem_orbit_device::Layout::capture(local),bem_orbit_device::SymmetricLayout::capture(symmetric),fft,size_t(2)*1024*1024*1024);
        composite.init(op,inverse,transform,size_t(512)*1024*1024);}
    const double candidate_setup=seconds(start);start=Clock::now();V x(n),physical(n),current(n),value(n),expected(n),actual(n),once(n),twice(n),zero(n);
    const auto staged=[&](const C* a,C* b,bool local_GPU){
        sectors.decode(a,physical.data(),true);
        if(local_GPU)inverse.apply(physical.data(),current.data(),.5);else symmetric.apply_full_fft_symmetry(physical.data(),current.data(),.5);
        op.matvec(current.data(),value.data());sectors.encode(value.data(),b,true);
    };
    double defect=0,device_defect=0;
    for(int seed=0;seed<3;++seed){
        for(int i=0;i<n;++i)x[i]=C(std::sin((.021+.011*seed)*i),std::cos((.031+.017*seed)*i));
        staged(x.data(),expected.data(),false);composite.apply(x.data(),actual.data(),.5);defect=std::max(defect,error(actual,expected));
        staged(x.data(),once.data(),true);defect=std::max(defect,error(actual,once));
    }
    once=actual;actual=x;composite.apply(actual.data(),actual.data(),.5);device_defect=std::max(device_defect,error(actual,once));
    composite.apply(once.data(),twice.data(),.5);
    check(cudaMemcpyAsync(composite.device_input(),x.data(),n*sizeof(C),cudaMemcpyHostToDevice,cudaStreamPerThread));
    composite.enqueue(composite.device_input(),composite.device_output(),.5);composite.enqueue(composite.device_output(),composite.device_output(),.5);
    check(cudaMemcpyAsync(actual.data(),composite.device_output(),n*sizeof(C),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));device_defect=std::max(device_defect,error(actual,twice));
    composite.apply(zero.data(),actual.data(),.5);for(auto v:actual)if(v!=C(0))throw std::runtime_error("composite zero leakage");
    if(defect>2e-6 || device_defect>2e-12)throw std::runtime_error("composite physical action/device-chain gate");
    const double audits=seconds(start);
    check(cudaMemcpyAsync(composite.device_input(),x.data(),n*sizeof(C),cudaMemcpyHostToDevice,cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));std::vector<double> timings[4];
    const auto apply=[&](int mode){if(mode<2)staged(x.data(),actual.data(),mode==1);
        else if(mode==2)composite.apply(x.data(),actual.data(),.5);else composite.enqueue(composite.device_input(),composite.device_output(),.5);};
    for(int mode:{0,1,2,3,3,2,1,0,1,0,3,2}){start=Clock::now();for(int j=0;j<3;++j)apply(mode);check(cudaStreamSynchronize(cudaStreamPerThread));timings[mode].push_back(seconds(start)/3);}
    if(capture!="none"){
        check(cudaDeviceSynchronize());check(cudaProfilerStart());
        for(int j=0;j<8;++j)apply(capture=="device"?3:2);
        check(cudaStreamSynchronize(cudaStreamPerThread));check(cudaProfilerStop());
    }
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
    report<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<n<<",\"original_vs_resident_error\":"<<defect<<",\"device_chain_error\":"<<device_defect
          <<",\"common_setup_s\":"<<common_setup<<",\"candidate_setup_s\":"<<candidate_setup<<",\"audits_s\":"<<audits
          <<",\"inverse_explicit_bytes\":"<<inverse.storage_bytes()<<",\"composite_explicit_bytes\":"<<composite.storage_bytes()
          <<",\"FFT_bytes\":"<<fft.storage_bytes()<<",\"GPU_used_bytes_snapshot\":"<<total-free
          <<",\"assembly_per_thread_stream\":"<<(op.device_matvec_stream()==reinterpret_cast<void*>(cudaStreamPerThread)?"true":"false");
    const char* names[]={"baseline","staged_GPU_inverse","composite","device"};
    for(int mode=0;mode<4;++mode){report<<",\""<<names[mode]<<"_action_s\":[";
        for(size_t j=0;j<timings[mode].size();++j)report<<(j?",":"")<<timings[mode][j];
        report<<"]";}
    report<<",\"passed\":true}\n";
}catch(const std::exception& e){std::fprintf(stderr,"resident composite probe: %s\n",e.what());return 1;}
