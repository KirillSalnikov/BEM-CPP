// Two-level current initialization on real packed T-matrix right-hand sides.
// Coarse and fine stages are separate processes: no two GPU operators coexist.
#include "bdm_mesh_transfer.h"
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
static double elapsed(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}
static Vector read_vector(const std::string& path,int n,int column=0,bool pair=false) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()!=std::streamoff(size_t(n)*sizeof(Complex)*(pair?2:1)))
        throw std::runtime_error("wrong current/RHS file size: "+path);
    f.seekg(size_t(column)*n*sizeof(Complex));Vector v(n);
    if(!f.read(reinterpret_cast<char*>(v.data()),size_t(n)*sizeof(Complex)))throw std::runtime_error("read: "+path);
    return v;
}
static void write_vector(const std::string& path,const Vector& v) {
    std::ofstream f(path,std::ios::binary);
    if(!f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(Complex)))throw std::runtime_error("write: "+path);
}
static double relative_error(const Vector& x,const Vector& reference) {
    Vector d=x;for(size_t j=0;j<x.size();++j)d[j]-=reference[j];
    return bem_streaming::norm(d.data(),d.size())/std::max(bem_streaming::norm(reference.data(),reference.size()),1e-300);
}
int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=8)throw std::runtime_error("probe coarse|fine REF COARSE_REF CACHE OUTPUT COARSE_PACKET_DIR FINE_PACKET_DIR");
    const bool coarse=std::string(argv[1])=="coarse";
    if(!coarse && std::string(argv[1])!="fine")throw std::invalid_argument("stage");
    const int ref=std::stoi(argv[2]),coarse_ref=std::stoi(argv[3]);
    if(ref<1 || ref>7 || coarse_ref<1 || coarse_ref>=7 || (!coarse && coarse_ref>=ref))throw std::invalid_argument("refinement");
    const std::string cache=argv[4],out=argv[5],coarse_packets=argv[6],fine_packets=argv[7];
    const auto setup_start=Clock::now();
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;
    struct Cleanup {MullerFmmOperator& op;~Cleanup(){op.cleanup();}} cleanup{op};
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),40.,Complex(1.3116),options,
            7,6,5,64,false,2,0,.96,(cache+".near").c_str(),3);
    op.set_fmm_near_fp32(true);
    if(const char* flag=std::getenv("BEM_TMATRIX_FACE_FFT"))if(std::string(flag)!="0")op.enable_prism_face_fft(ref,10./7);
    D6hOrbit group;group.build(op.mesh);
    PrismGroupMaps maps;maps.build(group);
    PrismSymmetrySectors sectors;sectors.build(maps);
    PrismOrbitInverse local;local.build(op,maps,50);
    const bem_streaming::Action action=[&](const Complex* x,Complex* y){op.matvec(x,y);};
    const bem_streaming::Action strict=[&](const Complex* x,Complex* y){op.matvec_strict(x,y);};
    PrismCirculantInverse circ;
    circ.build(op.mesh,ref,action,[&](const Complex* x,Complex* y){local.apply(x,y);},0.,0.,3);
    PrismSymmetricCirculant symmetric;symmetric.build(group,circ,local);
    if(!symmetric.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("full D6h FFT symmetry failed");
    const bem_streaming::Action inverse=[&](const Complex* x,Complex* y){symmetric.apply_full_fft_symmetry(x,y,.5);};
    const double setup_s=elapsed(setup_start);
    BdmMeshTransfer transfer;double transfer_setup_s=0;
    if(!coarse){const auto start=Clock::now();
        transfer.build(build_muller_p2_mesh(regular_prism(6,10./7,coarse_ref,1.,0,false,true),options),op.mesh);
        transfer_setup_s=elapsed(start);
    }
    std::ofstream report(out+"/"+argv[1]+".jsonl");report<<std::setprecision(17);
    report<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"system_dofs\":"<<op.system_dofs
        <<",\"setup_s\":"<<setup_s<<",\"transfer_setup_s\":"<<transfer_setup_s
        <<",\"transfer_bytes\":"<<transfer.bytes()<<"}\n";report.flush();
    std::vector<Vec3> directions;
    for(int theta=0;theta<=180;++theta)for(double phi:{0.,.27,1.1}) {
        const double t=theta*std::acos(-1.)/180;
        directions.emplace_back(std::sin(t)*std::cos(phi),std::sin(t)*std::sin(phi),std::cos(t));
    }
    struct Variant {std::string name;double tol;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;};
    std::vector<Variant> variants;
    std::vector<std::string> names=coarse?std::vector<std::string>{"coarse_1e2","coarse_1e3"}:
        std::vector<std::string>{"baseline","coarse_1e2","coarse_1e3","baseline_repeat"};
    if(const char* selected=std::getenv("BEM_COARSE_PROBE_VARIANTS")) {
        names.clear();std::istringstream stream(selected);std::string name;
        while(std::getline(stream,name,',')) {
            if(name!="coarse_1e2" && name!="coarse_1e3" && (coarse || (name!="baseline" && name!="baseline_repeat")))
                throw std::invalid_argument("coarse probe variant");
            names.push_back(name);
        }
        if(names.empty() || (!coarse && names.front()!="baseline"))throw std::invalid_argument("fine variants need a baseline first");
    }
    for(const auto& name:names){Variant v;v.name=name;v.tol=coarse?(name=="coarse_1e2"?1e-2:1e-3):2e-6;
        v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));}
    const int n=op.system_dofs;
    for(int degree:{36,38}) {
        const std::string prefix=coarse?coarse_packets+"/packet_ref"+std::to_string(ref)+"_from_ref"+std::to_string(ref+1)+"_l"+std::to_string(degree)+"/packet":
            fine_packets+"/packet_ref"+std::to_string(ref)+"_ka40_l"+std::to_string(degree)+"/packet";
        Vector rhs[2]={read_vector(prefix+".rhs.bin",n,0,true),read_vector(prefix+".rhs.bin",n,1,true)};
        Vector reference[2],reference_far[2];
        for(auto& variant:variants) {
            const bool seeded=!coarse && variant.name.find("coarse_")==0;
            Vector solution[2]={Vector(n),Vector(n)};
            double seed_s=0,initial_residual=1.;
            if(seeded){const auto start=Clock::now();
                const auto old=read_vector(out+"/"+variant.name+"_l"+std::to_string(degree)+"_x.bin",transfer.source_size());
                transfer.apply(old.data(),solution[0].data());seed_s=elapsed(start);
            }
            // Initial residual is measured by the warm-start wrapper itself;
            // this extra measurement is diagnostic and timed separately.
            double diagnostic_s=0;
            if(seeded){const auto start=Clock::now();Vector image(n);strict(solution[0].data(),image.data());
                initial_residual=relative_error(image,rhs[0]);diagnostic_s=elapsed(start);}
            const auto start=Clock::now();
            auto x=solve_prism_sectors(action,strict,inverse,sectors,rhs[0].data(),solution[0].data(),64,variant.tol,1000,seeded,true,variant.recycle.get());
            WarmStartResult warm;bem_streaming::HostResult y;
            // A coarse current is only an initial guess. Generate the one
            // needed seed; the fine solve supplies and validates both codes.
            if(!coarse) {
                warm=symmetry_guess(group,rhs[0],solution[0],rhs[1],solution[1]);
                y=solve_prism_sectors(action,strict,inverse,sectors,rhs[1].data(),solution[1].data(),64,variant.tol,1000,warm.fitted,true,variant.recycle.get());
            }
            const double solve_s=elapsed(start);
            const auto validation_start=Clock::now();
            double residual[2]={x.residual,0.};
            if(coarse) {
                if(!std::isfinite(x.residual) || x.residual>variant.tol)throw std::runtime_error("coarse strict FP64 solve failed");
            }else {Vector image(n);
                for(int j=0;j<2;++j){strict(solution[j].data(),image.data());residual[j]=relative_error(image,rhs[j]);
                    if(!std::isfinite(residual[j]) || residual[j]>variant.tol)throw std::runtime_error("strict FP64 solve failed");}
            }
            const double validation_s=elapsed(validation_start);
            double error[2]={0,0},far_error[2]={0,0};
            if(coarse)write_vector(out+"/"+variant.name+"_l"+std::to_string(degree)+"_x.bin",solution[0]);
            else for(int j=0;j<2;++j){Vector far;op.farfield(solution[j].data(),directions,far);
                if(variant.name=="baseline"){reference[j]=solution[j];reference_far[j]=far;}
                else {error[j]=relative_error(solution[j],reference[j]);far_error[j]=relative_error(far,reference_far[j]);
                    if(far_error[j]>1e-4)throw std::runtime_error("fine far-field agreement failed");}
                write_vector(out+"/"+variant.name+"_l"+std::to_string(degree)+"_far"+std::to_string(j)+".bin",far);
            }
            report<<"{\"type\":\"packet\",\"variant\":\""<<variant.name<<"\",\"degree\":"<<degree
                <<",\"tol\":"<<variant.tol<<",\"seed_s\":"<<seed_s<<",\"initial_residual\":"<<initial_residual
                <<",\"diagnostic_s\":"<<diagnostic_s<<",\"solve_s\":"<<solve_s<<",\"validation_s\":"<<validation_s
                <<",\"iterations\":["<<x.iterations<<(coarse?"":","+std::to_string(y.iterations))<<"],\"strict_checks\":["<<x.strict_checks<<(coarse?"":","+std::to_string(y.strict_checks))
                <<"],\"codes_solved\":"<<(coarse?1:2)<<",\"residual\":["<<residual[0];
            if(!coarse)report<<","<<residual[1];
            report<<"],\"current_error\":["<<error[0]<<","<<error[1]
                <<"],\"far_error\":["<<far_error[0]<<","<<far_error[1]<<"],\"warm_fitted\":"<<(warm.fitted?"true":"false")
                <<",\"recycle_rank\":"<<variant.recycle->rank()<<"}\n";report.flush();
            std::printf("COARSE_TEST stage=%s variant=%s l=%d iterations=%d,%d time=%.3f residual=%.3e,%.3e seed_residual=%.3e\n",
                argv[1],variant.name.c_str(),degree,x.iterations,y.iterations,solve_s,residual[0],residual[1],initial_residual);
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
