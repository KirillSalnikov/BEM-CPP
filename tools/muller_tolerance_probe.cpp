// Bounded tolerance experiment on packed T RHS and a physical plane-wave pair.
// This does not construct a full large T matrix or modify production checkpoints.
#include "muller_fmm.h"
#include "muller_dense.h"
#include "prism_circulant_inverse.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <limits>
using C=std::complex<double>;
using V=std::vector<C>;
using Action=bem_streaming::Action;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double relative(const V& a,const V& b){
    if(a.size()!=b.size())throw std::runtime_error("validation dimensions");
    double numerator=0,denominator=0;
    for(size_t j=0;j<a.size();++j){numerator+=std::norm(a[j]-b[j]);denominator+=std::norm(b[j]);}
    const double result=std::sqrt(numerator/std::max(denominator,1e-300));
    if(!std::isfinite(result))throw std::runtime_error("nonfinite validation");
    return result;
}
int main(int argc,char** argv)try {
    if(argc!=5)throw std::runtime_error("REF KA PACKED_RHS OUTPUT_DIRECTORY");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),ri=1.3116,aspect=10./7;
    if(ref<1||ref>5||ka<=0)throw std::runtime_error("diagnostic size range");
    const std::string out=argv[4];std::ofstream log(out+"/results.jsonl");
    if(!log)throw std::runtime_error("output directory must exist");
    log<<std::setprecision(17);
    const auto setup=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;const auto mesh=regular_prism(6,aspect,ref,1.,0,false,true);
    op.init(mesh,ka,C(ri,0),options,7,6,5,64,false,2,0,.96,nullptr,3);op.set_fmm_near_fp32(true);
    if(std::getenv("BEM_TMATRIX_FACE_FFT")&&std::stoi(std::getenv("BEM_TMATRIX_FACE_FFT")))op.enable_prism_face_fft(ref,aspect);
    const int n=op.system_dofs;int mixed_calls=0,strict_calls=0;
    const Action action=[&](const C* x,C* y){++mixed_calls;op.matvec(x,y);};
    const Action strict=[&](const C* x,C* y){++strict_calls;op.matvec_strict(x,y);};
    bem_tmatrix::D6hOrbit orbit;orbit.build(op.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(orbit);
    bem_tmatrix::PrismSymmetrySectors sectors;sectors.build(maps);
    PrismOrbitInverse local;local.build(op,maps,50);
    PrismCirculantInverse circ;circ.build(op.mesh,ref,action,[&](const C* x,C* y){local.apply(x,y);},0,0);
    PrismSymmetricCirculant sym;sym.build(orbit,circ,local);
    if(!sym.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("FFT D6h symmetry failed");
    const Action inverse=[&](const C* x,C* y){sym.apply_full_fft_symmetry(x,y,.5);};
    log<<"{\"stage\":\"setup\",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"ri\":"<<ri<<",\"aspect\":"<<aspect
       <<",\"dofs\":"<<n<<",\"seconds\":"<<seconds(setup)<<",\"D6h\":true,\"preconditioner\":\"symmetric circulant blend 0.5 with orbit 50\"}\n"<<std::flush;
    std::ifstream input(argv[3],std::ios::binary);if(!input)throw std::runtime_error("RHS file missing");
    std::vector<V> rhs;
    while(true){V b(n);input.read(reinterpret_cast<char*>(b.data()),sizeof(C)*n);if(!input.gcount())break;
        if(input.gcount()!=std::streamsize(sizeof(C)*n))throw std::runtime_error("RHS dimension");
        rhs.push_back(std::move(b));}
    if(rhs.empty()||rhs.size()>16)throw std::runtime_error("RHS count");
    if(std::getenv("BEM_TOLERANCE_PLANES_ONLY"))rhs.clear();
    const int coded_count=rhs.size();
    const double incident_theta=std::getenv("BEM_TOLERANCE_INCIDENT_THETA")?std::stod(std::getenv("BEM_TOLERANCE_INCIDENT_THETA")):0;
    const double incident_phi=std::getenv("BEM_TOLERANCE_INCIDENT_PHI")?std::stod(std::getenv("BEM_TOLERANCE_INCIDENT_PHI")):0;
    const double it=incident_theta*M_PI/180,ip=incident_phi*M_PI/180;
    const Vec3 beam(std::sin(it)*std::cos(ip),std::sin(it)*std::sin(ip),std::cos(it));
    const Vec3 e1(std::cos(it)*std::cos(ip),std::cos(it)*std::sin(ip),-std::sin(it));
    const Vec3 e2(-std::sin(ip),std::cos(ip),0);
    rhs.push_back(muller_nodal_planewave_rhs(op.mesh,ka,e1,beam,13));
    rhs.push_back(muller_nodal_planewave_rhs(op.mesh,ka,e2,beam,13));
    std::vector<Vec3> directions;std::ofstream grid(out+"/directions.csv");
    grid<<"theta_deg,phi_deg,x,y,z,etx,ety,etz,epx,epy,epz\n"<<std::setprecision(17);
    for(double phi_deg:{0.,15.,30.,45.})for(int ti=0;ti<=720;++ti){
        const double t=ti*.25*M_PI/180,p=phi_deg*M_PI/180;
        directions.push_back(std::sin(t)*std::cos(p)*e1+std::sin(t)*std::sin(p)*e2+std::cos(t)*beam);
        const Vec3 et=std::cos(t)*std::cos(p)*e1+std::cos(t)*std::sin(p)*e2-std::sin(t)*beam;
        const Vec3 ep=-std::sin(p)*e1+std::cos(p)*e2;
        const auto& d=directions.back();grid<<ti*.25<<","<<phi_deg<<","<<d.x<<","<<d.y<<","<<d.z
            <<","<<et.x<<","<<et.y<<","<<et.z<<","<<ep.x<<","<<ep.y<<","<<ep.z<<"\n";
    }
    grid.close();V scratch(n);action(rhs[0].data(),scratch.data());strict(rhs[0].data(),scratch.data());
    std::vector<V> reference(rhs.size()),reference_far(rhs.size());
    std::vector<double> tolerances{2e-7,1e-2,1e-3,1e-4,1e-5,2e-6};
    if(const char* values=std::getenv("BEM_TOLERANCE_VALUES")) {
        tolerances.clear();std::stringstream stream(values);std::string token;
        while(std::getline(stream,token,','))tolerances.push_back(std::stod(token));
        if(tolerances.empty()||tolerances[0]>2e-6)throw std::runtime_error("first tolerance must be a tight reference");
    }
    const int recycle_rank=std::getenv("BEM_TOLERANCE_RECYCLE_RANK")?std::stoi(std::getenv("BEM_TOLERANCE_RECYCLE_RANK")):16;
    for(size_t trial=0;trial<tolerances.size();++trial){
        const double tol=tolerances[trial];
        // A fresh recycle space per tolerance prevents tight-reference leakage.
        bem_streaming::SectorRecycleSpace recycle(sectors.offsets,recycle_rank);
        for(size_t j=0;j<rhs.size();++j){
            V x(n),ax(n),far;const int before_m=mixed_calls,before_s=strict_calls;
            const auto result=bem_tmatrix::solve_prism_sectors(action,strict,inverse,sectors,rhs[j].data(),x.data(),64,tol,600,false,false,recycle_rank?&recycle:nullptr);
            const auto audit=Clock::now();strict(x.data(),ax.data());const double residual=relative(ax,rhs[j]);
            const double audit_s=seconds(audit);const auto radiation=Clock::now();op.farfield(x.data(),directions,far);
            const double far_s=seconds(radiation);
            if(reference[j].empty()){reference[j]=x;reference_far[j]=far;}
            V sx(n),sr(n),sb(n),sa(n);
            sectors.encode(x.data(),sx.data(),true);sectors.encode(reference[j].data(),sr.data(),true);
            sectors.encode(rhs[j].data(),sb.data(),true);sectors.encode(ax.data(),sa.data(),true);
            double sector_current_error=0,sector_residual=0;
            const double bn=bem_streaming::norm(sb.data(),n);
            for(int s=0;s<12;++s) {
                const int begin=sectors.offsets[s],count=sectors.offsets[s+1]-begin;
                if(count && bem_streaming::norm(sb.data()+begin,count)>bn*1e-8) {
                    V cx(sx.begin()+begin,sx.begin()+begin+count),cr(sr.begin()+begin,sr.begin()+begin+count);
                    V cb(sb.begin()+begin,sb.begin()+begin+count),ca(sa.begin()+begin,sa.begin()+begin+count);
                    sector_current_error=std::max(sector_current_error,relative(cx,cr));
                    sector_residual=std::max(sector_residual,relative(ca,cb));
                }
            }
            const std::string file="far_trial"+std::to_string(trial)+"_rhs"+std::to_string(j)+".bin";
            std::ofstream binary(out+"/"+file,std::ios::binary);binary.write(reinterpret_cast<const char*>(far.data()),far.size()*sizeof(C));
            log<<"{\"trial\":"<<trial<<",\"tolerance\":"<<tol<<",\"rhs\":"<<j
               <<",\"kind\":\""<<(j<size_t(coded_count)?"packed_T_code":"plane_wave_polarization")<<"\",\"coded_RHS_count\":"<<coded_count
               <<",\"incident_theta_deg\":"<<incident_theta<<",\"incident_phi_deg\":"<<incident_phi
               <<",\"iterations\":"<<result.iterations<<",\"solve_s\":"<<result.seconds<<",\"audit_s\":"<<audit_s<<",\"farfield_s\":"<<far_s
               <<",\"strict_residual\":"<<residual<<",\"solver_reported_residual\":"<<result.residual
               <<",\"current_relative_error\":"<<relative(x,reference[j])<<",\"far_field_relative_error\":"<<relative(far,reference_far[j])
               <<",\"maximum_active_sector_current_error\":"<<sector_current_error<<",\"maximum_active_sector_residual\":"<<sector_residual
               <<",\"mixed_calls\":"<<mixed_calls-before_m<<",\"strict_calls_including_audit\":"<<strict_calls-before_s
               <<",\"farfield_file\":\""<<file<<"\",\"accepted\":"<<(residual<=tol?"true":"false")<<"}\n"<<std::flush;
            std::printf("tolerance %.1e rhs %zu iterations %d solve %.3fs residual %.8e far error %.8e\n",tol,j,result.iterations,result.seconds,residual,relative(far,reference_far[j]));
            if(residual>tol)throw std::runtime_error("strict residual gate failed");
        }
    }
    op.cleanup();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
