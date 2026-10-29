// Bounded comparison of operator-informed global corrections. No T export.
#include "muller_fmm.h"
#include "muller_mbj.h"
#include "streaming_host_gmres.h"
#include "streaming_two_level.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace bem_streaming;
using Vector=std::vector<Complex>;
static int setting(const char* name,int fallback,int lower,int upper) {
    const int value=std::getenv(name)?std::stoi(std::getenv(name)):fallback;
    if(value<lower || value>upper)throw std::runtime_error(std::string("invalid ")+name);
    return value;
}
static void record(std::ofstream& out,const char* stage,const HostResult& r,int rank) {
    out<<std::setprecision(17)<<"{\"stage\":\""<<stage<<"\",\"iterations\":"<<r.iterations
       <<",\"rank\":"<<rank<<",\"seconds\":"<<r.seconds<<",\"strict_residual\":"<<r.residual
       <<",\"action_s\":"<<r.action_seconds<<",\"inverse_s\":"<<r.inverse_seconds
       <<",\"orthogonalization_s\":"<<r.orthogonalization_seconds<<",\"recycle_s\":"<<r.recycle_seconds
       <<",\"strict_s\":"<<r.strict_seconds<<",\"strict_checks\":"<<r.strict_checks<<"}\n";
    out.flush();
    std::printf("[global probe] %s: iterations=%d rank=%d strict=%.8e seconds=%.3f\n",stage,r.iterations,rank,r.residual,r.seconds);
}
int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=5)throw std::runtime_error("Usage: probe REF KA OPERATOR_CACHE OUTPUT_PREFIX");
    const int ref=std::stoi(argv[1]); const double ka=std::stod(argv[2]);
    const std::string cache=argv[3],out=argv[4];
    if(ref<1 || ref>7 || ka<=0)throw std::runtime_error("invalid geometry");
    const int rank=setting("BEM_GLOBAL_RANK",16,1,64);
    const int training_steps=setting("BEM_GLOBAL_TRAIN_STEPS",64,rank+1,160);
    const int test_steps=setting("BEM_GLOBAL_TEST_STEPS",32,8,240);
    const int leaf=setting("BEM_PROBE_MAX_LEAF",29,1,8192);
    const int nodes=setting("BEM_PROBE_MBJ_NODES",12,1,64);
    const double tolerance=std::getenv("BEM_PROBE_TOL")?std::stod(std::getenv("BEM_PROBE_TOL")):1e-5;
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;
    op.init(regular_prism(6,1.,ref,1.,0,false,true),ka,Complex(1.3,0),options,7,6,5,leaf,
            false,2,0,.96,(cache+".near").c_str(),3);
    op.set_fmm_near_fp32(true);
    MullerMbjPreconditioner mbj;
    mbj.build_cached(op,nodes,0,cache+".mbj");
    const int n=op.system_dofs;
    const Action mixed=[&](const Complex* v,Complex* w){op.matvec(v,w);};
    const Action exact=[&](const Complex* v,Complex* w){op.matvec_strict(v,w);};
    const Action local=[&](const Complex* v,Complex* w){mbj.apply(v,w);};
    auto incident=[&](double theta,double phi) {
        const Vec3 direction(std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi),std::cos(theta));
        const Vec3 polarization(std::cos(theta)*std::cos(phi),std::cos(theta)*std::sin(phi),-std::sin(theta));
        return muller_nodal_planewave_rhs(op.mesh,ka,polarization,direction,13);
    };
    const Vector train_rhs=incident(.63,.27),test_rhs=incident(.85,.63);
    Vector x(n),work(n);
    // Force lazy mixed caches/links to be ready before any solve timing.
    mixed(train_rhs.data(),work.data());
    RecycleSpace space(n,rank);
    std::ofstream report(out+".jsonl");
    if(!report)throw std::runtime_error("cannot write diagnostic report");
    report<<"{\"diagnostic_only\":true,\"ka\":"<<ka<<",\"ref\":"<<ref<<",\"system_dofs\":"<<n
          <<",\"training_steps\":"<<training_steps<<",\"test_steps\":"<<test_steps
          <<",\"capacity\":"<<rank<<",\"tolerance\":"<<tolerance<<"}\n";report.flush();
    std::puts("[global probe] train: cold oblique RHS, collect harmonic Ritz directions");
    const auto trained=gmres(mixed,exact,local,train_rhs.data(),x.data(),n,training_steps,tolerance,
                             training_steps,true,false,&space,false);
    record(report,"train",trained,space.rank());
    if(!space.validate())throw std::runtime_error("invalid trained coarse space");
    // Diagnostic checkpoint uses a supplied manifest identity; cannot be
    // loaded into production by the T worker's different builder signature.
    const char* signature=std::getenv("BEM_GLOBAL_SIGNATURE");
    if(signature && std::string(signature).size()==64)space.save(out+".recycle.bin",signature);
    std::puts("[global probe] control: different oblique RHS, local MBJ only");
    const auto control=gmres(mixed,exact,local,test_rhs.data(),x.data(),n,test_steps,tolerance,test_steps,true);
    record(report,"control",control,space.rank());
    // P = M^-1 (I-C C*) + U C*, A_inner U ~= C. C is orthonormal.
    // The space remains frozen for all iterations of this fixed right
    // preconditioner; strict acceptance continues to use the original A.
    TwoLevelInverse global_inverse(space,local);
    const Action two_level=[&](const Complex* rhs,Complex* solution){global_inverse(rhs,solution);};
    std::puts("[global probe] two_level: same test RHS and budget, frozen global correction");
    const auto coarse=gmres(mixed,exact,two_level,test_rhs.data(),x.data(),n,test_steps,tolerance,test_steps,true);
    record(report,"two_level",coarse,space.rank());
    std::puts("[global probe] gcro: same test RHS and initial space, projected recycling");
    const auto recycled=gmres(mixed,exact,local,test_rhs.data(),x.data(),n,test_steps,tolerance,test_steps,true,false,&space,true);
    record(report,"gcro",recycled,space.rank());
    if(signature && std::string(signature).size()==64)space.save(out+".final_recycle.bin",signature);
    std::ofstream solution(out+".solution.bin",std::ios::binary);
    solution.write(reinterpret_cast<const char*>(x.data()),size_t(n)*sizeof(Complex));
    op.cleanup(); return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
