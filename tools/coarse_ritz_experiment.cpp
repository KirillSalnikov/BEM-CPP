// Transport eight coarse harmonic-Ritz directions; train l36, hold out l38/l40.
#include "bdm_mesh_transfer.h"
#include "coarse_image_experiment.h"
#include "coarse_ritz_experiment.h"
#include "muller_fmm.h"
#include "prism_orbit_inverse.h"
#include "prism_sector_solver.h"
#include "prism_symmetric_circulant.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static std::vector<int> packet_degrees(){
    const char* configured=std::getenv("BEM_COARSE_RITZ_DEGREES");
    if(!configured)return {38,40};
    std::vector<int> result;std::stringstream stream(configured);std::string value;
    while(std::getline(stream,value,',')){
        size_t used=0;const int degree=std::stoi(value,&used);
        if(used!=value.size() || degree<6 || degree>112 || degree==36 ||
           std::find(result.begin(),result.end(),degree)!=result.end())
            throw std::invalid_argument("held-out degrees must be distinct, in 6..112, excluding training degree 36");
        result.push_back(degree);
    }
    if(result.empty())throw std::invalid_argument("empty held-out degrees");
    return result;
}
static Vector read(const std::string& path,int n,int column=0,bool pair=false){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(size_t(n)*sizeof(Complex)*(pair?2:1)))throw std::runtime_error("wrong file size: "+path);
    f.seekg(size_t(column)*n*sizeof(Complex));Vector v(n);
    if(!f.read(reinterpret_cast<char*>(v.data()),size_t(n)*sizeof(Complex)))throw std::runtime_error("read: "+path);
    return v;
}
static double error(const Vector& a,const Vector& b){Vector d=a;for(size_t i=0;i<a.size();++i)d[i]-=b[i];return bem_streaming::norm(d.data(),d.size())/std::max(bem_streaming::norm(b.data(),b.size()),1e-300);}
static bem_streaming::HostResult solve_encoded(const bem_streaming::Action& action,const bem_streaming::Action& strict,
    const PrismSymmetrySectors& sectors,CoarseImageExperiment& coarse,const Vector& b,Vector& x,
    bem_streaming::SectorRecycleSpace* recycle,double& inverse_seconds,int& inverse_calls) {
    const auto start=Clock::now();const int n=sectors.size();Vector encoded(n),coordinates(n),physical_x(n),physical_y(n);
    sectors.encode(b.data(),encoded.data(),true);
    const auto apply_inverse=[&](const Complex* a,Complex* out){const auto t=Clock::now();coarse.apply_encoded(a,out);inverse_seconds+=elapsed(t);++inverse_calls;};
    const auto composite=[&](const Complex* a,Complex* out,bool full){apply_inverse(a,physical_x.data());
        (full?strict:action)(physical_x.data(),physical_y.data());sectors.encode(physical_y.data(),out,true);};
    const int interval=std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")?std::stoi(std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")):4;
    const double factor=std::getenv("BEM_SECTOR_PROJECTED_FACTOR")?std::stod(std::getenv("BEM_SECTOR_PROJECTED_FACTOR")):.5;
    const double first=std::getenv("BEM_SECTOR_FIRST_PASS_FACTOR")?std::stod(std::getenv("BEM_SECTOR_FIRST_PASS_FACTOR")):0;
    auto result=bem_streaming::sector_gmres([&](const Complex* a,Complex* out){composite(a,out,false);},
        [&](const Complex* a,Complex* out){composite(a,out,true);},encoded.data(),coordinates.data(),sectors.offsets,
        64,2e-6,300,true,interval,recycle,factor,first);
    apply_inverse(coordinates.data(),x.data());result.seconds=elapsed(start);return result;
}
int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=7)throw std::invalid_argument("probe train|fine REF CACHE OUTPUT_JSON TRAINING_DIRECTORY PACKET_DIRECTORY");
    const bool train=std::string(argv[1])=="train";if(!train && std::string(argv[1])!="fine")throw std::invalid_argument("stage");
    const int ref=std::stoi(argv[2]);if(ref<2 || ref>6)throw std::invalid_argument("refinement");
    const std::string cache=argv[3],training=argv[5],packets=argv[6];
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
    double local_seconds=0;int local_calls=0;
    const bem_streaming::Action original=[&](const Complex* x,Complex* y){const auto start=Clock::now();symmetric.apply_full_fft_symmetry(x,y,.5);local_seconds+=elapsed(start);++local_calls;};
    const int n=op.system_dofs;const double setup_s=elapsed(before);
    if(train) {
        const std::string prefix=packets+"/packet_ref"+std::to_string(ref)+"_from_ref"+std::to_string(ref+1)+"_l36/packet";
        const auto rhs=read(prefix+".rhs.bin",n,0,true);Vector solution(n);
        bem_streaming::SectorRecycleSpace recycle(sectors.offsets,16);
        const auto solved=solve_prism_sectors(action,strict,original,sectors,rhs.data(),solution.data(),64,1e-3,300,false,true,&recycle);
        if(solved.residual>1e-3)throw std::runtime_error("coarse Ritz training did not converge");
        const auto export_start=Clock::now();const std::string signature(64,'0');
        std::array<std::vector<Vector>,12> saved;
        for(int sector=0;sector<12;++sector){const auto path=training+"/sector_"+std::to_string(sector)+".cache";
            recycle.sector(sector).save(path,signature);
            saved[sector]=coarse_ritz_experiment::read_columns(path,sectors.offsets[sector+1]-sectors.offsets[sector],signature);
            if(saved[sector].size()<8)throw std::runtime_error("not enough coarse Ritz columns");
        }
        Vector encoded(n),physical_rhs(n),current(n);
        for(int j=0;j<8;++j){for(int sector=0;sector<12;++sector)
                std::copy(saved[sector][j].begin(),saved[sector][j].end(),encoded.begin()+sectors.offsets[sector]);
            sectors.decode(encoded.data(),physical_rhs.data(),true);original(physical_rhs.data(),current.data());
            const double scale=bem_streaming::norm(current.data(),n);
            if(!std::isfinite(scale) || scale<=0)throw std::runtime_error("invalid exported current");
            for(auto& value:current)value/=scale;
            std::ofstream f(training+"/ritz_"+std::to_string(j)+".bin",std::ios::binary);
            if(!f.write(reinterpret_cast<const char*>(current.data()),current.size()*sizeof(Complex)))throw std::runtime_error("Ritz current write");
        }
        std::ofstream report(argv[4]);report<<std::setprecision(17)
            <<"{\"type\":\"coarse_ritz_training\",\"degree\":36,\"ref\":"<<ref<<",\"dofs\":"<<n
            <<",\"setup_s\":"<<setup_s<<",\"solve_s\":"<<solved.seconds<<",\"export_s\":"<<elapsed(export_start)
            <<",\"iterations\":"<<solved.iterations<<",\"strict_residual\":"<<solved.residual
            <<",\"packed_columns\":8,\"right_preconditioned_coordinates_decoded_as_rhs\":true}\n";
        return 0;
    }
    BdmMeshTransfer transfer;
    const auto train_start=Clock::now();
    transfer.build(build_muller_p2_mesh(regular_prism(6,10./7,ref-1,1.,0,false,true),options),op.mesh);
    CoarseImageExperiment correction(op.mesh,sectors,original);
    Vector image(n),column(n);
    for(int j=0;j<8;++j){
        const auto coarse=read(training+"/ritz_"+std::to_string(j)+".bin",transfer.source_size());
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
    std::ofstream report(argv[4]);report<<std::setprecision(17);
    report<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"dofs\":"<<n<<",\"setup_s\":"<<setup_s
        <<",\"training_degree\":36,\"training_s\":"<<training_s<<",\"training_fine_actions\":8,\"rank\":"<<correction.rank()
        <<",\"bytes\":"<<correction.bytes()<<",\"D6h_defect\":"<<defect<<"}\n";report.flush();
    struct Variant{std::string name;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;};std::vector<Variant> variants;
    for(const std::string name:{"baseline","coarse_ritz8","baseline_repeat"}){
        Variant v;v.name=name;v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));}
    std::vector<Vec3> directions;
    for(int theta=0;theta<=180;theta+=3){const double t=theta*std::acos(-1.)/180;directions.emplace_back(std::sin(t)*std::cos(.27),std::sin(t)*std::sin(.27),std::cos(t));}
    for(int degree:packet_degrees()){
        const std::string prefix=packets+"/packet_ref"+std::to_string(ref)+"_ka40_l"+std::to_string(degree)+"/packet";
        Vector rhs[2]={read(prefix+".rhs.bin",n,0,true),read(prefix+".rhs.bin",n,1,true)},reference[2];
        for(auto& v:variants){const bool guess=false,inverse=v.name=="coarse_ritz8";
            const bem_streaming::Action selected=inverse?amended:original;
            double inverse_seconds=0;int inverse_calls=0;
            const bem_streaming::Action measured=[&](const Complex* a,Complex* out){const auto t=Clock::now();selected(a,out);inverse_seconds+=elapsed(t);++inverse_calls;};
            Vector x[2]={Vector(n),Vector(n)};const auto start=Clock::now();
            if(guess)correction.guess(rhs[0].data(),x[0].data());
            actions=strict_actions=0;local_seconds=0;local_calls=0;
            const auto a=inverse?solve_encoded(action,strict,sectors,correction,rhs[0],x[0],v.recycle.get(),inverse_seconds,inverse_calls):
                solve_prism_sectors(action,strict,measured,sectors,rhs[0].data(),x[0].data(),64,2e-6,300,guess,true,v.recycle.get());
            WarmStartResult warm;bem_streaming::HostResult b;
            if(a.residual<=2e-6){warm=symmetry_guess(group,rhs[0],x[0],rhs[1],x[1]);
                b=solve_prism_sectors(action,strict,measured,sectors,rhs[1].data(),x[1].data(),64,2e-6,300,warm.fitted,true,v.recycle.get());}
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
                <<"],\"inverse_s\":"<<inverse_seconds<<",\"inverse_calls\":"<<inverse_calls
                <<",\"local_inverse_s\":"<<local_seconds<<",\"local_inverse_calls\":"<<local_calls
                <<",\"actions\":"<<solve_actions<<",\"strict_actions\":"<<solve_strict<<",\"converged\":"<<(converged?"true":"false")
                <<",\"warm_fitted\":"<<(warm.fitted?"true":"false")<<",\"recycle_rank\":"<<v.recycle->rank()<<"}\n";report.flush();
            std::printf("COARSE_RITZ variant=%s degree=%d iterations=%d,%d seconds=%.3f residual=%.3e,%.3e\n",v.name.c_str(),degree,a.iterations,b.iterations,solve_s,residual[0],residual[1]);
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
