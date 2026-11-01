// Isolated ka40/ref5 FFT/FMM comparison. Never replaces the production worker.
#include "bdm_mesh_transfer.h"
#include "coarse_image_experiment.h"
#include "muller_fmm.h"
#include "prism_d6h_model_action.h"
#include "prism_orbit_inverse.h"
#include "prism_sector_solver.h"
#include "prism_symmetric_circulant.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sys/resource.h>
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static Vector read(const std::string& path,int n,int column=0,int count=1){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(size_t(n)*count*sizeof(Complex)))throw std::runtime_error("input size: "+path);
    f.seekg(size_t(column)*n*sizeof(Complex));Vector x(n);
    if(!f.read(reinterpret_cast<char*>(x.data()),x.size()*sizeof(Complex)))throw std::runtime_error("read: "+path);
    return x;
}
static double relative(const Vector& x,const Vector& y){Vector d=x;for(size_t i=0;i<d.size();++i)d[i]-=y[i];
    return bem_streaming::norm(d.data(),d.size())/std::max(bem_streaming::norm(y.data(),y.size()),1e-300);}
static bool below(double x,double gate){return std::isfinite(x) && x<=gate;}
static double rss_mib(){rusage r{};getrusage(RUSAGE_SELF,&r);return r.ru_maxrss/1024.;}
static bem_streaming::HostResult solve_encoded(const bem_streaming::Action& action,const bem_streaming::Action& strict,
    const PrismSymmetrySectors& sectors,CoarseImageExperiment& coarse,const Vector& rhs,Vector& x,
    bem_streaming::SectorRecycleSpace* recycle,double& inverse_seconds,int& inverse_calls){
    const auto start=Clock::now();const int n=sectors.size();Vector encoded(n),coordinates(n),physical_x(n),physical_y(n);
    sectors.encode(rhs.data(),encoded.data(),true);
    const auto inverse=[&](const Complex* a,Complex* out){const auto t=Clock::now();coarse.apply_encoded(a,out);inverse_seconds+=seconds(t);++inverse_calls;};
    const auto composite=[&](const Complex* a,Complex* out,bool full){inverse(a,physical_x.data());
        (full?strict:action)(physical_x.data(),physical_y.data());sectors.encode(physical_y.data(),out,true);};
    const int interval=std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")?std::stoi(std::getenv("BEM_HOST_GMRES_STRICT_INTERVAL")):4;
    const double factor=std::getenv("BEM_SECTOR_PROJECTED_FACTOR")?std::stod(std::getenv("BEM_SECTOR_PROJECTED_FACTOR")):.5;
    const double first=std::getenv("BEM_SECTOR_FIRST_PASS_FACTOR")?std::stod(std::getenv("BEM_SECTOR_FIRST_PASS_FACTOR")):0;
    auto result=bem_streaming::sector_gmres([&](const Complex* a,Complex* b){composite(a,b,false);},
        [&](const Complex* a,Complex* b){composite(a,b,true);},encoded.data(),coordinates.data(),sectors.offsets,
        64,2e-6,1000,true,interval,recycle,factor,first);
    inverse(coordinates.data(),x.data());result.seconds=seconds(start);return result;
}
int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=6)throw std::invalid_argument("CACHE TRAINING_DIRECTORY PACKET_DIRECTORY PHASE_COLUMNS OUTPUT_JSONL");
    const std::string cache=argv[1],training=argv[2],packets=argv[3];
    std::ofstream report(argv[5]);if(!report)throw std::runtime_error("report open");report<<std::setprecision(17);
    const auto start=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;struct Cleanup{MullerFmmOperator& op;~Cleanup(){op.cleanup();}} cleanup{op};
    op.init(regular_prism(6,10./7,5,1.,0,false,true),40.,Complex(1.3116),options,7,6,5,64,false,2,0,.96,(cache+".near").c_str(),3);
    op.set_fmm_near_fp32(true);op.enable_prism_face_fft(5,10./7);
    D6hOrbit group;group.build(op.mesh);PrismGroupMaps maps;maps.build(group);PrismSymmetrySectors sectors;sectors.build(maps);
    PrismOrbitInverse local;local.build(op,maps,50);
    int actions=0,strict_actions=0;
    const bem_streaming::Action action=[&](const Complex* x,Complex* y){++actions;op.matvec(x,y);};
    const bem_streaming::Action strict=[&](const Complex* x,Complex* y){++strict_actions;op.matvec_strict(x,y);};
    PrismCirculantInverse circ;circ.build(op.mesh,5,action,[&](const Complex* x,Complex* y){local.apply(x,y);},0.,0.,3);
    PrismSymmetricCirculant symmetric;symmetric.build(group,circ,local);
    if(!symmetric.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("full D6h FFT inverse proof failed");
    const bem_streaming::Action original=[&](const Complex* x,Complex* y){symmetric.apply_full_fft_symmetry(x,y,.5);};
    const int n=op.system_dofs;if(n!=258048)throw std::runtime_error("wrong working mesh size");
    const double common_s=seconds(start);const auto transfer_start=Clock::now();
    BdmMeshTransfer transfer;transfer.build(build_muller_p2_mesh(regular_prism(6,10./7,4,1.,0,false,true),options),op.mesh);
    std::vector<Vector> priors(8,Vector(n)),raw_images(8,Vector(n)),images(8,Vector(n));
    for(int j=0;j<8;++j){const auto source=read(training+"/ritz_"+std::to_string(j)+".bin",transfer.source_size());transfer.apply(source.data(),priors[j].data());}
    const double transfer_s=seconds(transfer_start);const auto raw_start=Clock::now();
    for(int j=0;j<8;++j)action(priors[j].data(),raw_images[j].data());
    const double raw_s=seconds(raw_start);const auto model_start=Clock::now();
    PrismD6hModelAction model(op.mesh,maps,strict);
    Vector test(n),value(n),transformed(n),actual(n),expected(n);
    for(int i=0;i<n;++i)test[i]=Complex(std::sin(.031*i),std::cos(.043*i));
    model.apply(test.data(),value.data());strict(test.data(),actual.data());
    const double model_probe_change=relative(value,actual);
    if(!std::isfinite(model_probe_change))throw std::runtime_error("nonfinite model probe");
    const double model_diagnostic_s=seconds(model_start);const auto ritz_start=Clock::now();
    for(int j=0;j<8;++j){model.apply(priors[j].data(),images[j].data());std::printf("P13 native Ritz image %d/8\n",j+1);}
    const double model_ritz_s=seconds(ritz_start);
    const int selected[]={0,1,10,11,20,21,22,23,4,5,14,15,16,17,18,19};
    std::vector<Vector> phase,phase_images;std::vector<double> phase_seconds;
    for(int index:selected){const auto before=Clock::now();phase.push_back(read(argv[4],n,index,24));phase_images.push_back(Vector(n));
        model.apply(phase.back().data(),phase_images.back().data());phase_seconds.push_back(seconds(before));
        std::printf("P13 native phase image %zu/16\n",phase.size());}
    struct Variant {std::string name;int extra;bool averaged;std::unique_ptr<CoarseImageExperiment> correction;
        std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;bem_streaming::Action apply;
        double preparation_s=0,defect=0;int preparation_mixed=0,preparation_strict=0;bool valid=true;};
    std::vector<Variant> variants;
    const std::vector<std::string> names={"local","P2_current","P2_D6h","P2_D6h_phase8","P2_D6h_phase16","P2_current_repeat","local_repeat"};
    std::vector<int> vertical,vertical_sign;D6hOrbit::vertical_image_map(op.mesh,vertical,vertical_sign);
    for(const auto& name:names){Variant v;v.name=name;v.extra=name=="P2_D6h_phase16"?16:name=="P2_D6h_phase8"?8:0;
        v.averaged=name.find("D6h")!=std::string::npos;const bool enriched=name.find("P2")==0;
        const auto before=Clock::now();
        if(enriched){v.correction.reset(new CoarseImageExperiment(op.mesh,sectors,original));
            for(int j=0;j<8;++j)v.correction->add_d6h(priors[j],v.averaged?images[j]:raw_images[j]);
            for(int j=0;j<v.extra;++j)v.correction->add_d6h(phase[j],phase_images[j]);
            auto* pointer=v.correction.get();v.apply=[pointer](const Complex* x,Complex* y){pointer->apply(x,y);};
            v.preparation_s=transfer_s+(v.averaged?model_diagnostic_s+model_ritz_s:raw_s);
            for(int j=0;j<v.extra;++j)v.preparation_s+=phase_seconds[j];
            v.preparation_mixed=v.averaged?0:8;v.preparation_strict=v.averaged?25+24*(8+v.extra):0;
        }else v.apply=original;
        v.apply(test.data(),value.data());
        for(int g=0;g<3;++g){
            if(g<2){group.transform(test,transformed,g==1,true);group.transform(value,expected,g==1,false);}
            else for(int i=0;i<n;++i){transformed[vertical[i]]=-double(vertical_sign[i])*test[i];expected[vertical[i]]=double(vertical_sign[i])*value[i];}
            v.apply(transformed.data(),actual.data());const double defect=relative(actual,expected);
            if(!std::isfinite(defect))throw std::runtime_error("nonfinite inverse symmetry check: "+v.name);
            v.defect=std::max(v.defect,defect);
        }
        v.preparation_s+=seconds(before);v.valid=below(v.defect,1e-8);
        if(!v.valid && !enriched)throw std::runtime_error("base FFT inverse fails D6h");
        v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));
        variants.push_back(std::move(v));
    }
    report<<"{\"type\":\"setup\",\"ka\":40,\"ref\":5,\"dofs\":"<<n<<",\"common_s\":"<<common_s
        <<",\"model_scratch_bytes\":"<<model.storage_bytes()<<",\"model_original_actions\":"<<model.original_actions()
        <<",\"model_probe_relative_change\":"<<model_probe_change<<",\"strict_operator_symmetrized\":false,\"peak_rss_mib\":"<<rss_mib()<<",\"variants\":[";
    for(size_t j=0;j<variants.size();++j){const auto& v=variants[j];if(j)report<<',';
        report<<"{\"name\":\""<<v.name<<"\",\"phase_columns\":"<<v.extra<<",\"rank\":"<<(v.correction?v.correction->rank():0)
            <<",\"correction_bytes\":"<<(v.correction?v.correction->bytes():0)<<",\"preparation_s\":"<<v.preparation_s
            <<",\"preparation_mixed_actions\":"<<v.preparation_mixed<<",\"preparation_strict_actions\":"<<v.preparation_strict
            <<",\"D6h_defect\":"<<v.defect<<",\"valid\":"<<(v.valid?"true":"false")<<'}';}
    report<<"]}\n";report.flush();
    std::vector<Vec3> directions;const double pi=std::acos(-1.);
    for(double phi:{.27,.27+pi/6,.27+pi/3})for(int theta=0;theta<=180;theta+=3){const double t=theta*pi/180;
        directions.emplace_back(std::sin(t)*std::cos(phi),std::sin(t)*std::sin(phi),std::cos(t));}
    for(int degree=20;degree<=64;degree+=2)if(degree!=36){
        const std::string prefix=packets+"/packet_ref5_ka40_l"+std::to_string(degree)+"/packet";
        const Vector rhs[2]={read(prefix+".rhs.bin",n,0,2),read(prefix+".rhs.bin",n,1,2)};Vector reference[2];
        for(auto& v:variants)if(v.valid){const auto packet_start=Clock::now();
            Vector x[2]={Vector(n),Vector(n)};actions=strict_actions=0;double inverse_s=0;int inverse_calls=0;
            const bem_streaming::Action measured=[&](const Complex* a,Complex* b){const auto t=Clock::now();v.apply(a,b);inverse_s+=seconds(t);++inverse_calls;};
            const auto solve_start=Clock::now();
            const auto first=v.correction?solve_encoded(action,strict,sectors,*v.correction,rhs[0],x[0],v.recycle.get(),inverse_s,inverse_calls):
                solve_prism_sectors(action,strict,measured,sectors,rhs[0].data(),x[0].data(),64,2e-6,1000,false,true,v.recycle.get());
            WarmStartResult warm;bem_streaming::HostResult second;
            if(below(first.residual,2e-6)){warm=symmetry_guess(group,rhs[0],x[0],rhs[1],x[1]);
                second=solve_prism_sectors(action,strict,measured,sectors,rhs[1].data(),x[1].data(),64,2e-6,1000,warm.fitted,true,v.recycle.get());}
            const double solve_s=seconds(solve_start);const int solve_mixed=actions,solve_strict=strict_actions;
            const auto check_start=Clock::now();double residual[2],far_error[2]={0,0};
            for(int j=0;j<2;++j){strict(x[j].data(),actual.data());residual[j]=relative(actual,rhs[j]);}
            const double check_s=seconds(check_start);const bool converged=below(residual[0],2e-6)&&below(residual[1],2e-6);
            const auto far_start=Clock::now();
            if(converged)for(int j=0;j<2;++j){Vector field;op.farfield(x[j].data(),directions,field);
                if(v.name=="local")reference[j]=field;else far_error[j]=relative(field,reference[j]);}
            const double far_s=seconds(far_start);
            report<<"{\"type\":\"packet\",\"variant\":\""<<v.name<<"\",\"degree\":"<<degree
                <<",\"packet_s\":"<<seconds(packet_start)<<",\"solve_s\":"<<solve_s<<",\"check_s\":"<<check_s<<",\"far_s\":"<<far_s
                <<",\"mixed_actions\":"<<solve_mixed<<",\"strict_actions_including_final\":"<<strict_actions
                <<",\"solve_strict_actions\":"<<solve_strict<<",\"inverse_s\":"<<inverse_s<<",\"inverse_calls\":"<<inverse_calls
                <<",\"iterations\":["<<first.iterations<<','<<second.iterations<<"],\"strict_residual\":["<<residual[0]<<','<<residual[1]
                <<"],\"far_L2_error\":["<<far_error[0]<<','<<far_error[1]<<"],\"far_directions\":"<<directions.size()
                <<",\"warm_fitted\":"<<(warm.fitted?"true":"false")<<",\"converged\":"<<(converged?"true":"false")
                <<",\"recycle_rank\":"<<v.recycle->rank()<<",\"peak_rss_mib\":"<<rss_mib()<<"}\n";report.flush();
            if(!converged || !below(far_error[0],1e-4) || !below(far_error[1],1e-4))throw std::runtime_error("strict RHS/far field gate failed: "+v.name);
            std::printf("P13 degree=%d variant=%s iterations=%d,%d packet_s=%.3f residual=%.3e,%.3e\n",degree,v.name.c_str(),first.iterations,second.iterations,seconds(packet_start),residual[0],residual[1]);
        }
    }
    report<<"{\"type\":\"complete\",\"all_packets_complete\":true,\"total_wall_s\":"<<seconds(start)<<",\"peak_rss_mib\":"<<rss_mib()<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
