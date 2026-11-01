// P4 experimental gather/scatter and whole packet check. No production toggle.
#include "muller_fmm.h"
#include "muller_dense.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include <cuda_profiler_api.h>
#include <fstream>
#include <iomanip>
#include <memory>
using namespace bem_tmatrix;
using C=std::complex<double>;using V=std::vector<C>;using Action=bem_streaming::Action;using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static double error(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}
    const double r=std::sqrt(x/std::max(y,1e-300));if(!std::isfinite(r))throw std::runtime_error("nonfinite error");return r;}
static V rhs_file(const std::string& path,int n,int phase){std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(size_t(n)*sizeof(C)*2))throw std::runtime_error("packet RHS length");
    f.seekg(size_t(phase)*n*sizeof(C));V v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(C));if(!f)throw std::runtime_error("packet RHS read");return v;}
int main(int argc,char** argv)try{
    if(argc<5)throw std::invalid_argument("usage: REF KA PACKET_ROOT REPORT [DEGREE...]");
    const int ref=std::stoi(argv[1]),ka=std::stoi(argv[2]);if(ref<2||ref>5||ka<1)throw std::invalid_argument("mapped screen parameters");
    const int mask=std::getenv("BEM_MAPPED_OPPOSITE_MASK")?std::stoi(std::getenv("BEM_MAPPED_OPPOSITE_MASK")):0;
    if(mask<0||mask>3)throw std::invalid_argument("opposite mask");
    setvbuf(stdout,nullptr,_IOLBF,0);std::ofstream log(argv[4]);if(!log)throw std::runtime_error("report open");log<<std::setprecision(17);
    auto before=Clock::now();MullerFmmOperator op;struct Cleanup{MullerFmmOperator& o;~Cleanup(){o.cleanup();}}cleanup{op};
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,6,5,64,false,2,0,.96,nullptr,3);
    op.set_fmm_near_fp32(true);op.enable_prism_face_fft(ref,10./7);const int n=op.system_dofs;
    D6hOrbit orbit;orbit.build(op.mesh);PrismGroupMaps maps;maps.build(orbit);PrismSymmetrySectors sectors;sectors.build(maps);
    const Action action=[&](const C* a,C* b){op.matvec(a,b);},strict=[&](const C* a,C* b){op.matvec_strict(a,b);};
    PrismOrbitInverse local;local.build(op,maps,50);PrismCirculantInverse fft;
    fft.build(op.mesh,ref,action,[&](const C* a,C* b){local.apply(a,b);},0,mask?.25:0.,mask);
    PrismSymmetricCirculant inverse;inverse.build(orbit,fft,local);
    if(!inverse.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("FFT full D6h proof");
    const double common_setup=elapsed(before);double mapping_setup=0,mapping_defect=0,defect=0;
    V test(n),image(n),transformed(n),actual(n),expected(n),zero(n);
    for(int i=0;i<n;++i)test[i]=C(std::sin(.021*i),std::cos(.031*i));
    for(bool symmetric:{false,true}){
        if(fft.device_mapping_ready())fft.enable_device_mapping(false);
        fft.apply_fft_part(test.data(),expected.data(),-1,symmetric);
        before=Clock::now();fft.prepare_device_mapping(symmetric);mapping_setup=elapsed(before);
        fft.apply_fft_part(test.data(),actual.data(),-1,symmetric);mapping_defect=std::max(mapping_defect,error(actual,expected));
        fft.apply_fft_part(zero.data(),actual.data(),-1,symmetric);
        for(auto z:actual)if(z!=C(0))throw std::runtime_error("mapped zero RHS leakage");
    }
    fft.enable_device_mapping(false);inverse.apply_full_fft_symmetry(test.data(),expected.data(),.5);
    fft.enable_device_mapping(true);inverse.apply_full_fft_symmetry(test.data(),image.data(),.5);
    mapping_defect=std::max(mapping_defect,error(image,expected));
    for(bool mirror:{false,true}){orbit.transform(test,transformed,mirror,true);inverse.apply_full_fft_symmetry(transformed.data(),actual.data(),.5);
        orbit.transform(image,expected,mirror,false);defect=std::max(defect,error(actual,expected));}
    std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
    for(int j=0;j<n;++j){transformed[vertical[j]]=-double(sign[j])*test[j];expected[vertical[j]]=double(sign[j])*image[j];}
    inverse.apply_full_fft_symmetry(transformed.data(),actual.data(),.5);defect=std::max(defect,error(actual,expected));
    if(mapping_defect>2e-12 || defect>1e-8)throw std::runtime_error("mapped inverse action/D6h gate");
    // Capture production host assembly too; input/output host actions are
    // profiled separately from the preconditioner so API waits are attributed.
    const std::string capture=std::getenv("BEM_MAPPED_CAPTURE")?std::getenv("BEM_MAPPED_CAPTURE"):"";
    if(!capture.empty() && capture!="action" && capture!="baseline" && capture!="mapped")throw std::invalid_argument("capture kind");
    action(test.data(),actual.data());
    if(capture=="action"){cudaProfilerStart();for(int j=0;j<8;++j)action(test.data(),actual.data());cudaProfilerStop();}
    std::vector<double> inverse_times[2];
    for(int which:{0,1,1,0,1,0,0,1}){
        fft.enable_device_mapping(which!=0);const bool profiling=capture==(which?"mapped":"baseline");
        if(profiling)cudaProfilerStart();before=Clock::now();
        for(int j=0;j<8;++j)inverse.apply_full_fft_symmetry(test.data(),actual.data(),.5);
        inverse_times[which].push_back(elapsed(before)/8);if(profiling)cudaProfilerStop();
    }
    log<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<n<<",\"common_setup_s\":"<<common_setup
       <<",\"D6h_defect\":"<<defect<<",\"mapping_defect\":"<<mapping_defect<<",\"opposite_mask\":"<<mask
       <<",\"mapping_setup_s\":"<<mapping_setup<<",\"mapping_bytes\":"<<fft.device_mapping_storage_bytes()<<",\"FFT_bytes\":"<<fft.storage_bytes();
    for(int which=0;which<2;++which){log<<",\""<<(which?"mapped_inverse_s":"baseline_inverse_s")<<"\":[";
        for(size_t j=0;j<inverse_times[which].size();++j)log<<(j?",":"")<<inverse_times[which][j];log<<"]";}log<<"}\n"<<std::flush;
    struct Variant{std::string name;bool mapped;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;};
    std::vector<Variant> variants;
    for(const auto& name:{"baseline","mapped","baseline_repeat"}){Variant v;v.name=name;v.mapped=v.name=="mapped";
        v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));}
    std::vector<Vec3> directions;for(int angle=0;angle<=180;angle+=3){const double theta=angle*std::acos(-1.)/180;
        directions.emplace_back(std::sin(theta)*std::cos(.27),std::sin(theta)*std::sin(.27),std::cos(theta));}
    for(int arg=5;arg<argc;++arg){const int degree=std::stoi(argv[arg]);
        const std::string path=std::string(argv[3])+"/packet_ref"+std::to_string(ref)+"_ka"+std::to_string(ka)+"_l"+std::to_string(degree)+"/packet.rhs.bin";
        const V rhs[2]={rhs_file(path,n,0),rhs_file(path,n,1)};V reference[2];
        for(auto& variant:variants){fft.enable_device_mapping(variant.mapped);V x[2]={V(n),V(n)};double inverse_s=0;int inverse_calls=0;
            const Action measured=[&](const C* a,C* b){auto t=Clock::now();inverse.apply_full_fft_symmetry(a,b,.5);inverse_s+=elapsed(t);++inverse_calls;};
            before=Clock::now();auto a=solve_prism_sectors(action,strict,measured,sectors,rhs[0].data(),x[0].data(),64,2e-6,300,false,true,variant.recycle.get());
            bem_streaming::HostResult b;
            if(a.residual<=2e-6){auto warm=symmetry_guess(orbit,rhs[0],x[0],rhs[1],x[1]);b=solve_prism_sectors(action,strict,measured,sectors,rhs[1].data(),x[1].data(),64,2e-6,300,warm.fitted,true,variant.recycle.get());}
            const double seconds=elapsed(before);before=Clock::now();double residual[2],field_error[2];
            for(int j=0;j<2;++j){strict(x[j].data(),image.data());residual[j]=error(image,rhs[j]);V field;
                muller_nodal_farfield(op.mesh,x[j].data(),x[j].data()+n/2,ka,directions,field,13);if(variant.name=="baseline")reference[j]=field;field_error[j]=error(field,reference[j]);}
            const bool passed=std::max(residual[0],residual[1])<=2e-6 && std::max(field_error[0],field_error[1])<=1e-4;
            log<<"{\"type\":\"packet\",\"variant\":\""<<variant.name<<"\",\"degree\":"<<degree<<",\"seconds\":"<<seconds
               <<",\"validation_s\":"<<elapsed(before)<<",\"iterations\":["<<a.iterations<<","<<b.iterations<<"],\"inverse_s\":"<<inverse_s
               <<",\"inverse_calls\":"<<inverse_calls<<",\"strict_residual\":["<<residual[0]<<","<<residual[1]<<"],\"far_error\":["<<field_error[0]<<","<<field_error[1]
               <<"],\"passed\":"<<(passed?"true":"false")<<"}\n"<<std::flush;
            if(!passed)throw std::runtime_error("packet validation gate failed");
        }
    }
}catch(const std::exception& e){std::fprintf(stderr,"mapped screen: %s\n",e.what());return 1;}
