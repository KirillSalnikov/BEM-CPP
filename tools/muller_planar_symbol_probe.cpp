#include "muller_fmm.h"
#include "muller_dense.h"
#include "planar_inverse_kernel.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include <fstream>
#include <iomanip>
#include <memory>
using namespace bem_tmatrix;
using C=std::complex<double>;using V=std::vector<C>;using Action=bem_streaming::Action;using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static double error(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}
    const double result=std::sqrt(x/std::max(y,1e-300));if(!std::isfinite(result))throw std::runtime_error("nonfinite error");return result;}
static V rhs_file(const std::string& path,int n,int phase){std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(size_t(n)*sizeof(C)*2))throw std::runtime_error("OSRC RHS length");
    f.seekg(size_t(phase)*n*sizeof(C));V v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(C));if(!f)throw std::runtime_error("OSRC RHS read");return v;}
int main(int argc,char** argv)try{
    if(argc<6)throw std::invalid_argument("usage: REF KA PACKET_ROOT REPORT DEGREE...");
    const int ref=std::stoi(argv[1]),ka=std::stoi(argv[2]);if(ref<2||ref>5||ka<1)throw std::invalid_argument("OSRC screen parameters");
    setvbuf(stdout,nullptr,_IOLBF,0);std::ofstream log(argv[4]);log<<std::setprecision(17);
    auto before=Clock::now();MullerFmmOperator op;struct Cleanup{MullerFmmOperator& o;~Cleanup(){o.cleanup();}}cleanup{op};
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,6,5,64,false,2,0,.96,nullptr,3);
    op.set_fmm_near_fp32(true);op.enable_prism_face_fft(ref,10./7);const int n=op.system_dofs;
    D6hOrbit orbit;orbit.build(op.mesh);PrismGroupMaps maps;maps.build(orbit);PrismSymmetrySectors sectors;sectors.build(maps);
    const Action action=[&](const C* a,C* b){op.matvec(a,b);},strict=[&](const C* a,C* b){op.matvec_strict(a,b);};
    PrismOrbitInverse local;local.build(op,maps,50);const double common_setup=elapsed(before);
    std::array<std::unique_ptr<PrismCirculantInverse>,4> fft;
    std::array<std::unique_ptr<PrismSymmetricCirculant>,4> inverse;
    double setup[4]{},condition[4]{},kernel_defect[4]{},damping[4]={0,.01,.05,.20};
    auto prepare=[&](int i){inverse[i].reset(new PrismSymmetricCirculant);inverse[i]->build(orbit,*fft[i],local);
        if(!inverse[i]->prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("OSRC FFT D6h layout proof");};
    before=Clock::now();fft[0].reset(new PrismCirculantInverse);
    fft[0]->build(op.mesh,ref,action,[&](const C* a,C* b){local.apply(a,b);});prepare(0);setup[0]=elapsed(before);
    before=Clock::now();PrismCirculantLayout layout;layout.build(op.mesh,ref);
    std::unique_ptr<bem_osrc::PlanarBdmSymbol> stencil[2];
    for(int kind=0;kind<2;++kind)stencil[kind].reset(new bem_osrc::PlanarBdmSymbol(op.mesh,layout.patches[kind?24:0]));
    const double stencil_setup=elapsed(before);
    for(int i=1;i<4;++i){before=Clock::now();bem_osrc::InverseKernel kernels[2];
        for(int kind=0;kind<2;++kind){const auto& patch=layout.patches[kind?24:0];
            kernels[kind]=bem_osrc::make_inverse_kernel(*stencil[kind],patch.nx,patch.ny,ka,C(1.3116),damping[i],1);
            condition[i]=std::max(condition[i],kernels[kind].maximum_frobenius_condition);
            kernel_defect[i]=std::max(kernel_defect[i],kernels[kind].maximum_inverse_defect);}
        fft[i].reset(new PrismCirculantInverse);
        fft[i]->build_from_inverse_kernels(op.mesh,ref,[&](const C* a,C* b){local.apply(a,b);},kernels[0].values,kernels[1].values);
        prepare(i);setup[i]=stencil_setup+elapsed(before);
    }
    for(auto& s:stencil)s.reset();
    struct Variant{std::string name;int index;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;bool failed=false;};
    std::vector<Variant> variants;
    const auto add=[&](const std::string& name,int i){Variant v;v.name=name;v.index=i;v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));};
    add("baseline",0);add("osrc001",1);add("osrc005",2);add("osrc020",3);add("baseline_repeat",0);
    V test(n),image(n),transformed(n),actual(n),expected(n);for(int i=0;i<n;++i)test[i]=C(std::sin(.021*i),std::cos(.031*i));
    std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(op.mesh,vertical,sign);double defect=0;
    for(int i=0;i<4;++i){inverse[i]->apply_full_fft_symmetry(test.data(),image.data(),.5);
        for(bool mirror:{false,true}){orbit.transform(test,transformed,mirror,true);inverse[i]->apply_full_fft_symmetry(transformed.data(),actual.data(),.5);orbit.transform(image,expected,mirror,false);defect=std::max(defect,error(actual,expected));}
        for(int j=0;j<n;++j){transformed[vertical[j]]=-double(sign[j])*test[j];expected[vertical[j]]=double(sign[j])*image[j];}
        inverse[i]->apply_full_fft_symmetry(transformed.data(),actual.data(),.5);defect=std::max(defect,error(actual,expected));}
    log<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<n<<",\"common_setup_s\":"<<common_setup
       <<",\"D6h_defect\":"<<defect<<",\"aliases\":1,\"gauss_order\":16,\"local_bytes\":"<<local.storage_bytes()<<",\"variants\":[";
    for(int i=0;i<4;++i)log<<(i?",":"")<<"{\"damping\":"<<damping[i]<<",\"setup_s\":"<<setup[i]<<",\"FFT_bytes\":"<<fft[i]->storage_bytes()
        <<",\"maximum_condition_F\":"<<condition[i]<<",\"inverse_defect\":"<<kernel_defect[i]<<"}";
    log<<"]}\n"<<std::flush;
    if(defect>1e-8)throw std::runtime_error("analytic FFT inverse breaks D6h");
    std::vector<Vec3> directions;for(int angle=0;angle<=180;angle+=3){const double theta=angle*std::acos(-1.)/180;directions.emplace_back(std::sin(theta)*std::cos(.27),std::sin(theta)*std::sin(.27),std::cos(theta));}
    for(int arg=5;arg<argc;++arg){const int degree=std::stoi(argv[arg]);
        const std::string path=std::string(argv[3])+"/packet_ref"+std::to_string(ref)+"_ka"+std::to_string(ka)+"_l"+std::to_string(degree)+"/packet.rhs.bin";
        const V rhs[2]={rhs_file(path,n,0),rhs_file(path,n,1)};V reference[2];
        for(auto& variant:variants){if(variant.failed)continue;V x[2]={V(n),V(n)};double inverse_s=0;int inverse_calls=0;
            const Action measured=[&](const C* a,C* b){auto t=Clock::now();inverse[variant.index]->apply_full_fft_symmetry(a,b,.5);inverse_s+=elapsed(t);++inverse_calls;};
            before=Clock::now();auto a=solve_prism_sectors(action,strict,measured,sectors,rhs[0].data(),x[0].data(),64,2e-6,300,false,true,variant.recycle.get());
            bem_streaming::HostResult b;
            if(a.residual<=2e-6){auto warm=symmetry_guess(orbit,rhs[0],x[0],rhs[1],x[1]);b=solve_prism_sectors(action,strict,measured,sectors,rhs[1].data(),x[1].data(),64,2e-6,300,warm.fitted,true,variant.recycle.get());}
            const double seconds=elapsed(before);before=Clock::now();double residual[2],field_error[2];
            for(int j=0;j<2;++j){strict(x[j].data(),image.data());residual[j]=error(image,rhs[j]);V field;
                muller_nodal_farfield(op.mesh,x[j].data(),x[j].data()+n/2,ka,directions,field,13);if(variant.name=="baseline")reference[j]=field;field_error[j]=error(field,reference[j]);}
            const bool passed=std::max(residual[0],residual[1])<=2e-6&&std::max(field_error[0],field_error[1])<=1e-4;
            log<<"{\"type\":\"packet\",\"variant\":\""<<variant.name<<"\",\"degree\":"<<degree<<",\"seconds\":"<<seconds
               <<",\"validation_s\":"<<elapsed(before)<<",\"iterations\":["<<a.iterations<<","<<b.iterations<<"],\"inverse_s\":"<<inverse_s
               <<",\"inverse_calls\":"<<inverse_calls<<",\"strict_residual\":["<<residual[0]<<","<<residual[1]<<"],\"far_error\":["<<field_error[0]<<","<<field_error[1]
               <<"],\"passed\":"<<(passed?"true":"false")<<"}\n"<<std::flush;
            if(!passed){variant.failed=true;if(variant.name.find("baseline")==0)throw std::runtime_error("baseline failed");}
        }
    }
}catch(const std::exception& e){std::fprintf(stderr,"OSRC screen: %s\n",e.what());return 1;}
