// Isolated FP64 assembly comparison. No production worker or checkpoint edits.
#include "muller_fmm.h"
#include "muller_dense.h"
#include "tmatrix_symmetry_warm_start.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
using C=std::complex<double>;using V=std::vector<C>;using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static double relative(const V&a,const V&b){long double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}return std::sqrt(double(e/std::max(n,1.e-300L)));}
int main(int argc,char**argv)try{
    if(argc!=6)throw std::runtime_error("REF KA ASPECT CACHE_PREFIX RESULT_JSON");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),aspect=std::stod(argv[3]);
    if(ref<2||ref>6||ka<=0||aspect<=0)throw std::runtime_error("invalid geometry");
    omp_set_num_threads(4);
    setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);setenv("BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1",1);
    setenv("BEM_FMM_MULTILEVEL","1",1);setenv("BEM_MULLER_GPU_STRICT_ONLY","1",1);
    setenv("BEM_FMM_PAIR_CURRENTS","0",1);setenv("BEM_FMM_MULTILEVEL_PAIR","1",1);
    const int seg=1<<ref;int nz=std::max(seg,int(std::ceil(std::sqrt(3.)*aspect*seg)));nz+=nz%2;
    const int points=7*12*seg*(seg+nz),depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    const bool production_layout=std::getenv("BEM_STRICT_PROBE_PRODUCTION_LAYOUT")!=nullptr;
    if(!production_layout){
        setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
        setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
        setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    }
    const int max_leaf=production_layout?64:leaf(depth);
    const auto begin=Clock::now();MullerP2BuildOptions opt;opt.edge_mode=MullerEdgeMode::HDivBdm1;MullerFmmOperator op;
    struct Cleanup{MullerFmmOperator&op;~Cleanup(){op.cleanup();}}cleanup{op};
    op.init(regular_prism(6,aspect,ref,1.,0,false,true),C(ka),C(1.3116),opt,7,6,5,max_leaf,false,2,0,.96,(std::string(argv[4])+".near").c_str(),3);
    if(!op.gpu_operator_assembly)throw std::runtime_error("GPU assembly unavailable");
    const double setup=elapsed(begin);op.set_fmm_near_fp32(true);const int n=op.system_dofs;
    V inputs[3],baseline[3],output(n);std::mt19937 rng(7117);std::normal_distribution<double>random;
    for(int p=0;p<2;++p)inputs[p]=muller_nodal_planewave_rhs(op.mesh,ka,p?Vec3(0,1,0):Vec3(1,0,0),Vec3(0,0,1),13);
    inputs[2].resize(n);for(auto&z:inputs[2])z=C(random(rng),random(rng));
    const int input_count=std::getenv("BEM_STRICT_PROBE_ONE_INPUT")?1:3;
    op.gpu_operator_assembly=false;
    for(int p=0;p<input_count;++p){baseline[p].resize(n);op.matvec_strict(inputs[p].data(),baseline[p].data());}
    const auto action=[&](const V&x,V&y,int kind,int threads){
        omp_set_num_threads(threads);op.gpu_operator_assembly=(kind%3)!=0;
        setenv("BEM_FMM_MULTILEVEL_PAIR",kind%3==2?"1":"0",1);
        setenv("BEM_FMM_P2P_STRICT_REAL",kind>=3?"1":"0",1);
        op.matvec_strict(x.data(),y.data());
        if(!op.fmm_near_fp32)throw std::runtime_error("strict action did not restore precision");
    };
    struct Trial{const char*name;int kind,threads;};
    const bool short_probe=std::getenv("BEM_STRICT_PROBE_SHORT")!=nullptr;
    const std::vector<Trial> order=short_probe
        ? std::vector<Trial>{{"host16",0,16},{"gpu_pair_real",5,4},{"gpu_pair_real",5,4},{"host16",0,16}}
        : std::vector<Trial>{{"host16",0,16},{"host16_real",3,16},{"gpu_pair_real",5,4},
          {"gpu_pair",2,4},{"gpu_pair",2,4},{"gpu_pair_real",5,4},{"host16_real",3,16},{"host16",0,16}};
    std::ofstream report(argv[5]);report<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"aspect\":"<<aspect<<",\"dofs\":"<<n<<",\"setup_s\":"<<setup<<",\"production_layout\":"<<(production_layout?"true":"false")<<",\"fmm_max_leaf\":"<<max_leaf<<",\"inputs\":"<<input_count<<",\"trials\":[";
    for(size_t t=0;t<order.size();++t){const auto&v=order[t];action(inputs[0],output,v.kind,v.threads);
        report<<(t?",":"")<<"{\"name\":\""<<v.name<<"\",\"inputs\":[";
        for(int p=0;p<input_count;++p){const auto start=Clock::now();action(inputs[p],output,v.kind,v.threads);const double seconds=elapsed(start),error=relative(output,baseline[p]);
            if(!std::isfinite(error)||error>2e-12)throw std::runtime_error("FP64 assembly disagreement");
            report<<(p?",":"")<<"{\"seconds\":"<<seconds<<",\"relative_error\":"<<error<<"}";
            std::printf("%s input=%d seconds=%.6f error=%.3e\n",v.name,p,seconds,error);std::fflush(stdout);
        }report<<"]}";report.flush();
    }
    report<<"],\"strict_precision_preserved\":true,\"shared_workstation\":true,\"notes\":\"Inputs are plane-wave RHS vectors followed by a random current (only the first when inputs=1); same discrete operator. GPU assembly remains allocated during host controls. Pair shares electric/magnetic current geometry, not an approximate symmetry residual.\"}\n";
    if(!report)throw std::runtime_error("report write");
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
