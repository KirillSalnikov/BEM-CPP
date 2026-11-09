// P4 comparison using the original BEM operator, factors and sector solver.
// Kept separate from the production worker until native controls have passed.
#include "muller_fmm.h"
#include "muller_dense.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include "resident_prism_inverse.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <fstream>
#include <iomanip>
#include <memory>
using namespace bem_tmatrix;
using C=std::complex<double>;using V=std::vector<C>;using Action=bem_streaming::Action;using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
static double error(const V& a,const V& b){double x=0,y=0;
    for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}
    const double r=std::sqrt(x/std::max(y,1e-300));if(!std::isfinite(r))throw std::runtime_error("nonfinite error");return r;
}
static V rhs_file(const std::string& path,int n,int phase){std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(size_t(n)*sizeof(C)*2))throw std::runtime_error("packet RHS length");
    f.seekg(size_t(phase)*n*sizeof(C));V v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(C));
    if(!f)throw std::runtime_error("packet RHS read");return v;
}
int main(int argc,char** argv)try {
    if(argc<5)throw std::invalid_argument("usage: REF KA PACKET_ROOT REPORT [DEGREE...]");
    const int ref=std::stoi(argv[1]),ka=std::stoi(argv[2]);
    if(ref<2 || ref>5 || ka<1)throw std::invalid_argument("resident screen parameters");
    const int mask=std::getenv("BEM_RESIDENT_OPPOSITE_MASK")?std::stoi(std::getenv("BEM_RESIDENT_OPPOSITE_MASK")):0;
    if(mask<0 || mask>3)throw std::invalid_argument("opposite mask");
    setvbuf(stdout,nullptr,_IOLBF,0);std::ofstream log(argv[4]);if(!log)throw std::runtime_error("report open");log<<std::setprecision(17);
    auto before=Clock::now();MullerFmmOperator op;struct Cleanup{MullerFmmOperator& op;~Cleanup(){op.cleanup();}}cleanup{op};
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,ref==2?12:6,5,64,false,2,0,.96,nullptr,3);
    op.set_fmm_near_fp32(true);op.enable_prism_face_fft(ref,10./7);const int n=op.system_dofs;
    D6hOrbit orbit;orbit.build(op.mesh);PrismGroupMaps maps;maps.build(orbit);PrismSymmetrySectors sectors;sectors.build(maps);
    const Action action=[&](const C* a,C* b){op.matvec(a,b);},strict=[&](const C* a,C* b){op.matvec_strict(a,b);};
    PrismOrbitInverse local;local.build(op,maps,50);PrismCirculantInverse fft;
    fft.build(op.mesh,ref,action,[&](const C* a,C* b){local.apply(a,b);},0,mask?.25:0.,mask);
    PrismSymmetricCirculant inverse;inverse.build(orbit,fft,local);
    if(!inverse.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("FFT full D6h proof");
    const double common_setup=elapsed(before);before=Clock::now();ResidentPrismInverse resident;
    fft.prepare_device_mapping(true);const double mapping_setup=elapsed(before);before=Clock::now();
    {const auto layout=bem_orbit_device::Layout::capture(local);const auto sym=bem_orbit_device::SymmetricLayout::capture(inverse);
        resident.init(layout,sym,fft,size_t(2)*1024*1024*1024);}
    const double resident_setup=elapsed(before);const size_t resident_bytes=resident.storage_bytes();before=Clock::now();
    V test(n),image(n),transformed(n),actual(n),expected(n),zero(n),twice(n);
    double equivalence=0,defect=0;
    for(int seed=0;seed<3;++seed){
        for(int i=0;i<n;++i)test[i]=C(std::sin((.021+.011*seed)*i),std::cos((.031+.017*seed)*i));
        for(double blend:{0.,.25,.5,1.}){
            fft.enable_device_mapping(false);inverse.apply_full_fft_symmetry(test.data(),expected.data(),blend);
            resident.apply(test.data(),actual.data(),blend);equivalence=std::max(equivalence,error(actual,expected));
        }
    }
    resident.apply(test.data(),image.data(),.5);
    for(bool mirror:{false,true}){
        orbit.transform(test,transformed,mirror,true);resident.apply(transformed.data(),actual.data(),.5);
        orbit.transform(image,expected,mirror,false);defect=std::max(defect,error(actual,expected));
    }
    std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
    for(int j=0;j<n;++j){transformed[vertical[j]]=-double(sign[j])*test[j];expected[vertical[j]]=double(sign[j])*image[j];}
    resident.apply(transformed.data(),actual.data(),.5);defect=std::max(defect,error(actual,expected));
    actual=test;resident.apply(actual.data(),actual.data(),.5);equivalence=std::max(equivalence,error(actual,image));
    inverse.apply_full_fft_symmetry(image.data(),twice.data(),.5);
    check(cudaMemcpyAsync(resident.device_input(),test.data(),n*sizeof(C),cudaMemcpyHostToDevice,cudaStreamPerThread));
    resident.enqueue_device(resident.device_input(),resident.device_output(),.5);
    resident.enqueue_device(resident.device_output(),resident.device_output(),.5);
    check(cudaMemcpyAsync(actual.data(),resident.device_output(),n*sizeof(C),cudaMemcpyDeviceToHost,cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));equivalence=std::max(equivalence,error(actual,twice));
    resident.apply(zero.data(),actual.data());for(auto z:actual)if(z!=C(0))throw std::runtime_error("resident zero leakage");
    if(equivalence>2e-12 || defect>1e-8 || resident.storage_bytes()!=resident_bytes)throw std::runtime_error("resident action/D6h/workspace gate");
    const double resident_audit=elapsed(before);
    // Host actions include upload/download. Resident device timing includes
    // the final synchronization but excludes the single initial upload.
    std::vector<double> times[4];
    const auto apply=[&](int which,const C* x,C* y){
        if(which==2)resident.apply(x,y,.5);
        else {fft.enable_device_mapping(which==1);inverse.apply_full_fft_symmetry(x,y,.5);}
    };
    check(cudaMemcpyAsync(resident.device_input(),test.data(),n*sizeof(C),cudaMemcpyHostToDevice,cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
    for(int which:{0,1,2,3,3,2,1,0,1,0,3,2}){
        before=Clock::now();
        for(int j=0;j<8;++j){if(which==3)resident.enqueue_device(resident.device_input(),resident.device_output(),.5);else apply(which,test.data(),actual.data());}
        check(cudaStreamSynchronize(cudaStreamPerThread));times[which].push_back(elapsed(before)/8);
    }
    const std::string capture=std::getenv("BEM_RESIDENT_CAPTURE")?std::getenv("BEM_RESIDENT_CAPTURE"):"";
    if(!capture.empty()){
        int which=capture=="baseline"?0:capture=="mapped"?1:capture=="resident"?2:capture=="device"?3:-1;
        if(which<0)throw std::invalid_argument("capture kind");
        check(cudaDeviceSynchronize());check(cudaProfilerStart());
        for(int j=0;j<8;++j){if(which==3)resident.enqueue_device(resident.device_input(),resident.device_output(),.5);else apply(which,test.data(),actual.data());}
        check(cudaStreamSynchronize(cudaStreamPerThread));check(cudaProfilerStop());
    }
    log<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<n<<",\"common_setup_s\":"<<common_setup
       <<",\"D6h_defect\":"<<defect<<",\"resident_defect\":"<<equivalence<<",\"opposite_mask\":"<<mask
       <<",\"mapping_setup_s\":"<<mapping_setup<<",\"resident_setup_s\":"<<resident_setup<<",\"resident_audit_s\":"<<resident_audit
       <<",\"resident_explicit_bytes\":"<<resident_bytes<<",\"mapping_bytes\":"<<fft.device_mapping_storage_bytes()<<",\"FFT_bytes\":"<<fft.storage_bytes();
    for(int which=0;which<4;++which){const char* names[]={"baseline","mapped","resident","device"};log<<",\""<<names[which]<<"_inverse_s\":[";
        for(size_t j=0;j<times[which].size();++j)log<<(j?",":"")<<times[which][j];log<<"]";}
    log<<"}\n"<<std::flush;
    struct Variant{std::string name;int which;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;};std::vector<Variant> variants;
    for(const auto& name:{"baseline","mapped","resident","resident_repeat","baseline_repeat"}){
        Variant v;v.name=name;v.which=v.name=="mapped"?1:v.name.find("resident")==0?2:0;
        v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));
    }
    std::vector<Vec3> directions;for(int angle=0;angle<=180;angle+=3){const double theta=angle*std::acos(-1.)/180;
        directions.emplace_back(std::sin(theta)*std::cos(.27),std::sin(theta)*std::sin(.27),std::cos(theta));}
    for(int arg=5;arg<argc;++arg){const int degree=std::stoi(argv[arg]);
        const std::string path=std::string(argv[3])+"/packet_ref"+std::to_string(ref)+"_ka"+std::to_string(ka)+"_l"+std::to_string(degree)+"/packet.rhs.bin";
        const V rhs[2]={rhs_file(path,n,0),rhs_file(path,n,1)};V reference[2];
        for(auto& variant:variants){V x[2]={V(n),V(n)};double inverse_s=0;int inverse_calls=0;
            const Action measured=[&](const C* a,C* b){auto t=Clock::now();apply(variant.which,a,b);inverse_s+=elapsed(t);++inverse_calls;};
            before=Clock::now();auto a=solve_prism_sectors(action,strict,measured,sectors,rhs[0].data(),x[0].data(),64,2e-6,300,false,true,variant.recycle.get());
            bem_streaming::HostResult b;
            if(a.residual<=2e-6){auto warm=symmetry_guess(orbit,rhs[0],x[0],rhs[1],x[1]);b=solve_prism_sectors(action,strict,measured,sectors,rhs[1].data(),x[1].data(),64,2e-6,300,warm.fitted,true,variant.recycle.get());}
            const double solve_s=elapsed(before);before=Clock::now();double residual[2],field_error[2];
            for(int j=0;j<2;++j){strict(x[j].data(),image.data());residual[j]=error(image,rhs[j]);V field;
                muller_nodal_farfield(op.mesh,x[j].data(),x[j].data()+n/2,ka,directions,field,13);
                if(variant.name=="baseline")reference[j]=field;field_error[j]=error(field,reference[j]);}
            const bool passed=std::max(residual[0],residual[1])<=2e-6 && std::max(field_error[0],field_error[1])<=1e-4;
            log<<"{\"type\":\"packet\",\"variant\":\""<<variant.name<<"\",\"degree\":"<<degree<<",\"seconds\":"<<solve_s
               <<",\"validation_s\":"<<elapsed(before)<<",\"iterations\":["<<a.iterations<<","<<b.iterations<<"],\"inverse_s\":"<<inverse_s
               <<",\"inverse_calls\":"<<inverse_calls<<",\"strict_residual\":["<<residual[0]<<","<<residual[1]<<"],\"far_error\":["<<field_error[0]<<","<<field_error[1]
               <<"],\"passed\":"<<(passed?"true":"false")<<"}\n"<<std::flush;
            if(!passed)throw std::runtime_error("resident packet validation failed");
        }
    }
}catch(const std::exception& e){std::fprintf(stderr,"resident screen: %s\n",e.what());return 1;}
