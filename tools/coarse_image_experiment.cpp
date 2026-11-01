// A fixed coarse image correction trained on l36 only. l38/l40 are held out.
#include "bdm_mesh_transfer.h"
#include "coarse_image_experiment.h"
#include "muller_fmm.h"
#include "prism_orbit_inverse.h"
#include "prism_sector_solver.h"
#include "prism_symmetric_circulant.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static Vector read(const std::string& path,int n,int column=0,bool pair=false){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(size_t(n)*sizeof(Complex)*(pair?2:1)))throw std::runtime_error("wrong file size: "+path);
    f.seekg(size_t(column)*n*sizeof(Complex));Vector v(n);
    if(!f.read(reinterpret_cast<char*>(v.data()),size_t(n)*sizeof(Complex)))throw std::runtime_error("read: "+path);
    return v;
}
static double error(const Vector& a,const Vector& b){Vector d=a;for(size_t i=0;i<a.size();++i)d[i]-=b[i];return bem_streaming::norm(d.data(),d.size())/std::max(bem_streaming::norm(b.data(),b.size()),1e-300);}
int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=6)throw std::invalid_argument("probe REF CACHE OUTPUT_JSONL TRAINING_DIRECTORY FINE_PACKET_DIRECTORY");
    const int ref=std::stoi(argv[1]);if(ref<2 || ref>6)throw std::invalid_argument("refinement");
    const std::string cache=argv[2],training=argv[4],packets=argv[5];
    const auto before=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;struct Cleanup{MullerFmmOperator& o;~Cleanup(){o.cleanup();}} cleanup{op};
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),40.,Complex(1.3116),options,7,6,5,64,false,2,0,.96,(cache+".near").c_str(),3);
    op.set_fmm_near_fp32(true);
    if(const char* flag=std::getenv("BEM_TMATRIX_FACE_FFT"))if(std::string(flag)!="0")op.enable_prism_face_fft(ref,10./7);
    D6hOrbit group;group.build(op.mesh);PrismGroupMaps maps;maps.build(group);PrismSymmetrySectors sectors;sectors.build(maps);
    PrismOrbitInverse local;local.build(op,maps,50);
    int actions=0,strict_actions=0;
    const bem_streaming::Action action=[&](const Complex* x,Complex* y){++actions;op.matvec(x,y);};
    const bem_streaming::Action strict=[&](const Complex* x,Complex* y){++strict_actions;op.matvec_strict(x,y);};
    PrismCirculantInverse circ;circ.build(op.mesh,ref,action,[&](const Complex* x,Complex* y){local.apply(x,y);},0.,0.,3);
    PrismSymmetricCirculant symmetric;symmetric.build(group,circ,local);
    if(!symmetric.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("full D6h FFT proof failed");
    const bem_streaming::Action original=[&](const Complex* x,Complex* y){symmetric.apply_full_fft_symmetry(x,y,.5);};
    const int n=op.system_dofs;const double setup_s=elapsed(before);
    BdmMeshTransfer transfer;
    const auto train_start=Clock::now();
    transfer.build(build_muller_p2_mesh(regular_prism(6,10./7,ref-1,1.,0,false,true),options),op.mesh);
    CoarseImageExperiment correction(op.mesh,sectors,original);
    Vector image(n),column(n);
    for(const std::string name:{"coarse_1e2","coarse_1e3"}){
        const auto coarse=read(training+"/"+name+"_l36_x.bin",transfer.source_size());
        transfer.apply(coarse.data(),column.data());action(column.data(),image.data());correction.add_d6h(column,image);
    }
    const double training_s=elapsed(train_start);
    const bem_streaming::Action amended=[&](const Complex* x,Complex* y){correction.apply(x,y);};
    // Check the newly assembled inverse, independently of its operator images.
    Vector test(n),value(n),rotated(n),actual(n),expected(n);
    for(int i=0;i<n;++i)test[i]=Complex(std::sin(.031*i),std::cos(.043*i));
    amended(test.data(),value.data());double defect=0;
    for(bool mirror:{false,true}){group.transform(test,rotated,mirror,true);amended(rotated.data(),actual.data());
        group.transform(value,expected,mirror,false);defect=std::max(defect,error(actual,expected));}
    std::vector<int> target,sign;D6hOrbit::vertical_image_map(op.mesh,target,sign);
    for(int i=0;i<n;++i)rotated[target[i]]=-double(sign[i])*test[i];
    amended(rotated.data(),actual.data());for(int i=0;i<n;++i)expected[target[i]]=double(sign[i])*value[i];defect=std::max(defect,error(actual,expected));
    if(defect>1e-8)throw std::runtime_error("full D6h coarse inverse failed");
    std::ofstream report(argv[3]);report<<std::setprecision(17);
    report<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"dofs\":"<<n<<",\"setup_s\":"<<setup_s
        <<",\"training_degree\":36,\"training_s\":"<<training_s<<",\"training_fine_actions\":2,\"rank\":"<<correction.rank()
        <<",\"bytes\":"<<correction.bytes()<<",\"D6h_defect\":"<<defect<<"}\n";report.flush();
    struct Variant{std::string name;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;};std::vector<Variant> variants;
    for(const std::string name:{"baseline","coarse_guess","coarse_inverse","baseline_repeat"}){
        Variant v;v.name=name;v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));}
    std::vector<Vec3> directions;
    for(int theta=0;theta<=180;theta+=3){const double t=theta*std::acos(-1.)/180;directions.emplace_back(std::sin(t)*std::cos(.27),std::sin(t)*std::sin(.27),std::cos(t));}
    for(int degree:{38,40}){
        const std::string prefix=packets+"/packet_ref"+std::to_string(ref)+"_ka40_l"+std::to_string(degree)+"/packet";
        Vector rhs[2]={read(prefix+".rhs.bin",n,0,true),read(prefix+".rhs.bin",n,1,true)},reference[2];
        for(auto& v:variants){const bool guess=v.name=="coarse_guess",inverse=v.name=="coarse_inverse";
            const bem_streaming::Action selected=inverse?amended:original;
            Vector x[2]={Vector(n),Vector(n)};const auto start=Clock::now();
            if(guess)correction.guess(rhs[0].data(),x[0].data());
            actions=strict_actions=0;
            const auto a=solve_prism_sectors(action,strict,selected,sectors,rhs[0].data(),x[0].data(),64,2e-6,300,guess,true,v.recycle.get());
            WarmStartResult warm;bem_streaming::HostResult b;
            if(a.residual<=2e-6){warm=symmetry_guess(group,rhs[0],x[0],rhs[1],x[1]);
                b=solve_prism_sectors(action,strict,selected,sectors,rhs[1].data(),x[1].data(),64,2e-6,300,warm.fitted,true,v.recycle.get());}
            const double solve_s=elapsed(start);const int solve_actions=actions,solve_strict=strict_actions;
            const auto check_start=Clock::now();double residual[2]={1.,1.};
            for(int j=0;j<2;++j){strict(x[j].data(),image.data());residual[j]=error(image,rhs[j]);}
            const double validation_s=elapsed(check_start);const bool converged=std::max(residual[0],residual[1])<=2e-6;
            double far_error[2]={0.,0.};
            if(converged)for(int j=0;j<2;++j){Vector far;op.farfield(x[j].data(),directions,far);
                if(v.name=="baseline")reference[j]=far;else far_error[j]=error(far,reference[j]);}
            if((v.name=="baseline" && !converged) || std::max(far_error[0],far_error[1])>1e-4)throw std::runtime_error("reference/field check failed");
            report<<"{\"type\":\"packet\",\"variant\":\""<<v.name<<"\",\"degree\":"<<degree
                <<",\"seconds\":"<<solve_s<<",\"validation_s\":"<<validation_s<<",\"iterations\":["<<a.iterations<<","<<b.iterations
                <<"],\"strict_residual\":["<<residual[0]<<","<<residual[1]<<"],\"far_error\":["<<far_error[0]<<","<<far_error[1]
                <<"],\"actions\":"<<solve_actions<<",\"strict_actions\":"<<solve_strict<<",\"converged\":"<<(converged?"true":"false")
                <<",\"warm_fitted\":"<<(warm.fitted?"true":"false")<<",\"recycle_rank\":"<<v.recycle->rank()<<"}\n";report.flush();
            std::printf("COARSE_IMAGE variant=%s degree=%d iterations=%d,%d seconds=%.3f residual=%.3e,%.3e\n",v.name.c_str(),degree,a.iterations,b.iterations,solve_s,residual[0],residual[1]);
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
