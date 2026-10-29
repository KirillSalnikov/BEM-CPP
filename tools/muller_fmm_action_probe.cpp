// Bounded operator/solver diagnostics; never exports a production T matrix.
#include "muller_fmm.h"
#include "muller_mbj.h"
#include "streaming_host_gmres.h"
#include "tmatrix_symmetry_warm_start.h"
#include <chrono>
#include <complex>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <iomanip>

int main(int argc,char**argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=5)throw std::runtime_error("Usage: probe ref ka cache_prefix output_prefix");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    const std::string cache=argv[3],out=argv[4];
    const bool d6h=std::getenv("BEM_TMATRIX_D6H_MESH") &&
        std::string(std::getenv("BEM_TMATRIX_D6H_MESH"))!="0";
    const Mesh mesh=regular_prism(6,1.,ref,1.,0,false,d6h);
    MullerP2BuildOptions options;options.project_edge_nodes_to_sphere=false;
    options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto begin=std::chrono::steady_clock::now();
    MullerFmmOperator op;
    int leaf=std::max(64,int((2LL*mesh.nt()*7+32767)/32768));
    if(const char* value=std::getenv("BEM_PROBE_MAX_LEAF"))leaf=std::stoi(value);
    const bool strict=std::getenv("BEM_PROBE_STRICT") &&
        std::string(std::getenv("BEM_PROBE_STRICT"))!="0";
    const double absorption=std::getenv("BEM_PROBE_RI_IMAG")?std::stod(std::getenv("BEM_PROBE_RI_IMAG")):0.;
    const double refractive_index=std::getenv("BEM_PROBE_RI")?std::stod(std::getenv("BEM_PROBE_RI")):1.3;
    const int seed=std::getenv("BEM_PROBE_SEED")?std::stoi(std::getenv("BEM_PROBE_SEED")):0;
    const int duffy=std::getenv("BEM_TMATRIX_DUFFY_ORDER")?std::stoi(std::getenv("BEM_TMATRIX_DUFFY_ORDER")):4;
    const int radius=std::getenv("BEM_PROBE_NEAR_RADIUS")?std::stoi(std::getenv("BEM_PROBE_NEAR_RADIUS")):3;
    const int digits=std::getenv("BEM_PROBE_DIGITS")?std::stoi(std::getenv("BEM_PROBE_DIGITS")):5;
    op.init(mesh,std::complex<double>(ka,0),std::complex<double>(refractive_index,absorption),options,
            7,duffy,digits,leaf,false,2,0,.96,(cache+".near").c_str(),radius);
    op.set_fmm_near_fp32(true);
    const double setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::vector<std::complex<double>> x(op.system_dofs),y(op.system_dofs);
    for(int i=0;i<op.system_dofs;++i)x[i]={std::sin(.013*(i+1)),std::cos(.017*(i+1))};
    if(seed) {
        std::mt19937 rng(seed);std::uniform_real_distribution<double> uniform(-1.,1.);
        for(auto& value:x)value={uniform(rng),uniform(rng)};
    }
    const int actions=std::getenv("BEM_PROBE_ACTIONS")?std::max(1,std::min(10,std::stoi(std::getenv("BEM_PROBE_ACTIONS")))):1;
    const bool warm_mixed=std::getenv("BEM_PROBE_WARM_MIXED") &&
        std::string(std::getenv("BEM_PROBE_WARM_MIXED"))!="0";
    // Populate all mixed caches before testing the strict path, as happens
    // during residual replacement inside an actual solve.
    if(warm_mixed)op.matvec(x.data(),y.data());
    std::vector<double> timings;
    for(int repeat=0;repeat<actions;++repeat) {
        const auto action=std::chrono::steady_clock::now();
        if(strict)op.matvec_strict(x.data(),y.data());
        else op.matvec(x.data(),y.data());
        timings.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-action).count());
    }
    double seconds=0.;
    const int begin_steady=actions>1?1:0;
    for(int i=begin_steady;i<actions;++i)seconds+=timings[i]/(actions-begin_steady);
    std::ofstream binary(out+".bin",std::ios::binary);
    binary.write(reinterpret_cast<const char*>(y.data()),y.size()*sizeof(y[0]));binary.close();
    std::ofstream report(out+".json");
    report<<std::setprecision(17)<<"{\"diagnostic_only\":true,\"actions\":"<<actions<<",\"system_dofs\":"<<op.system_dofs
          <<",\"ka\":"<<ka<<",\"ref\":"<<ref<<",\"max_leaf\":"<<leaf
          <<",\"ri\":"<<refractive_index<<",\"ri_imag\":"<<absorption<<",\"seed\":"<<seed
          <<",\"d6h_mesh\":"<<(d6h?"true":"false")<<",\"duffy_order\":"<<duffy
          <<",\"near_radius\":"<<radius<<",\"digits\":"<<digits
          <<",\"strict\":"<<(strict?"true":"false")
          <<",\"warm_mixed\":"<<(warm_mixed?"true":"false")<<",\"setup_s\":"<<setup
          <<",\"matvec_s\":"<<seconds<<",\"action_seconds\":[";
    for(int i=0;i<actions;++i)report<<(i?",":"")<<timings[i];
    report<<"]";
    // Compare runtime-only kernel switches on the same initialized operator.
    // ABBA order limits drift from GPU temperature/background applications.
    if(const char* flag=std::getenv("BEM_PROBE_COMPARE_FLAG")) {
        const std::string name(flag);
        if(name!="BEM_FMM_MULTILEVEL_COMPACT_TEMP" &&
           name!="BEM_FMM_MULTILEVEL_PARITY" &&
           name!="BEM_FMM_MULTILEVEL_M2L_ROWS" &&
           name!="BEM_FMM_P2P_THREADS" &&
           name!="BEM_FMM_P2P_SHARED_REAL" && name!="BEM_FMM_P2P_SHARED_FAST_TRIG" &&
           name!="BEM_FMM_P2P_SHARED_SOURCES" && name!="combined")
            throw std::runtime_error("unsupported runtime comparison flag");
        const std::vector<std::string> flags=name=="combined"
            ? std::vector<std::string>{"BEM_FMM_MULTILEVEL_COMPACT_TEMP","BEM_FMM_P2P_SHARED_SOURCES"}
            : std::vector<std::string>{name};
        std::vector<std::string> previous;
        std::vector<bool> had_old;
        for(const auto& key:flags) {
            const char* old=std::getenv(key.c_str());
            had_old.push_back(old!=nullptr); previous.push_back(old?old:"");
        }
        const auto reference=y;
        const double denominator=std::max(bem_streaming::norm(reference.data(),op.system_dofs),1e-300);
        report<<",\"compare_flag\":\""<<name<<"\",\"comparison\":[";
        int trial=0;
        for(int enabled: {0,1,1,0}) {
            const char* setting=name=="BEM_FMM_P2P_THREADS"?(enabled?"128":"512"):(enabled?"1":"0");
            for(const auto& key:flags)setenv(key.c_str(),setting,1);
            const auto start=std::chrono::steady_clock::now();
            if(strict)op.matvec_strict(x.data(),y.data());
            else op.matvec(x.data(),y.data());
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            for(int i=0;i<op.system_dofs;++i)y[i]-=reference[i];
            const double error=bem_streaming::norm(y.data(),op.system_dofs)/denominator;
            report<<(trial++?",":"")<<"{\"enabled\":"<<enabled<<",\"seconds\":"<<elapsed
                  <<",\"setting\":\""<<setting<<"\",\"relative_to_initial\":"<<error<<"}";
            std::printf("[kernel compare] %s=%d seconds=%.6f relative_error=%.6e\n",flag,enabled,elapsed,error);
        }
        report<<"]";
        for(size_t i=0;i<flags.size();++i) {
            if(had_old[i])setenv(flags[i].c_str(),previous[i].c_str(),1);
            else unsetenv(flags[i].c_str());
        }
        y=reference;
    }
    if(std::getenv("BEM_PROBE_P2P_TUNE")) {
        const auto reference=y;
        const double denominator=std::max(bem_streaming::norm(reference.data(),op.system_dofs),1e-300);
        report<<",\"p2p_tuning\":[";
        bool first=true;
        for(int threads: {128,256,512})for(int split: {1,2,4,8,16,32}) {
            setenv("BEM_FMM_P2P_THREADS",std::to_string(threads).c_str(),1);
            setenv("BEM_FMM_P2P_LEAF_SPLIT",std::to_string(split).c_str(),1);
            const auto start=std::chrono::steady_clock::now();
            op.matvec(x.data(),y.data());
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            for(int i=0;i<op.system_dofs;++i)y[i]-=reference[i];
            const double error=bem_streaming::norm(y.data(),op.system_dofs)/denominator;
            report<<(first?"":",")<<"{\"threads\":"<<threads<<",\"split\":"<<split
                  <<",\"seconds\":"<<elapsed<<",\"relative_to_initial\":"<<error<<"}";
            first=false;
            std::printf("[P2P tune] threads=%d split=%d seconds=%.6f relative_error=%.6e\n",threads,split,elapsed,error);
        }
        report<<"]";
    }
    if(const char* nodes_text=std::getenv("BEM_PROBE_MBJ_NODES")) {
        MullerMbjPreconditioner mbj;
        const int nodes=std::stoi(nodes_text);
        const char* mbj_cache=std::getenv("BEM_PROBE_MBJ_CACHE");
        mbj.build_cached(op,nodes,0,mbj_cache?mbj_cache:out+".mbj");
        const int limit=std::getenv("BEM_PROBE_ITERS")?std::stoi(std::getenv("BEM_PROBE_ITERS")):200;
        const int restart=std::getenv("BEM_PROBE_RESTART")?std::stoi(std::getenv("BEM_PROBE_RESTART")):80;
        const double tolerance=std::getenv("BEM_PROBE_TOL")?std::stod(std::getenv("BEM_PROBE_TOL")):1e-5;
        const auto rhs=muller_nodal_planewave_rhs(op.mesh,ka,Vec3(1,0,0),Vec3(0,0,1),13);
        const bem_streaming::Action mixed=[&](const std::complex<double>* v,std::complex<double>* w){op.matvec(v,w);};
        const bem_streaming::Action exact=[&](const std::complex<double>* v,std::complex<double>* w){op.matvec_strict(v,w);};
        const bem_streaming::Action inverse=[&](const std::complex<double>* v,std::complex<double>* w){mbj.apply(v,w);};
        const auto result=bem_streaming::gmres(mixed,exact,inverse,rhs.data(),x.data(),op.system_dofs,restart,tolerance,limit);
        std::ofstream solution(out+".solution.bin",std::ios::binary);
        solution.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(x[0]));
        report<<",\"solve\":{\"mbj_nodes\":"<<nodes<<",\"mbj_setup_s\":"<<mbj.setup_seconds
              <<",\"mbj_storage_mb\":"<<mbj.storage_megabytes()<<",\"iterations\":"<<result.iterations
              <<",\"seconds\":"<<result.seconds<<",\"strict_residual\":"<<result.residual
              <<",\"strict_checks\":"<<result.strict_checks<<",\"deferred_strict_checks\":"<<result.deferred_strict_checks
              <<",\"action_s\":"<<result.action_seconds<<",\"strict_s\":"<<result.strict_seconds
              <<",\"tolerance\":"<<tolerance<<",\"converged\":"<<(result.residual<=tolerance?"true":"false")<<"}";
        if(std::getenv("BEM_PROBE_SYMMETRY_WARM_START") && result.residual<=tolerance) {
            bem_tmatrix::D6hOrbit orbit;orbit.build(op.mesh);
            const auto second=muller_nodal_planewave_rhs(op.mesh,ka,Vec3(0,1,0),Vec3(0,0,1),13);
            const auto warm=bem_tmatrix::symmetry_guess(orbit,rhs,x,second,y);
            const auto solve=bem_streaming::gmres(mixed,exact,inverse,second.data(),y.data(),op.system_dofs,
                restart,tolerance,limit,true,warm.fitted);
            report<<",\"second_solve\":{\"warm_start_fitted\":"<<(warm.fitted?"true":"false")
                  <<",\"warm_start_rhs_error\":"<<warm.relative_rhs_error<<",\"warm_start_s\":"<<warm.seconds
                  <<",\"iterations\":"<<solve.iterations<<",\"seconds\":"<<solve.seconds
                  <<",\"strict_checks\":"<<solve.strict_checks<<",\"deferred_strict_checks\":"<<solve.deferred_strict_checks
                  <<",\"strict_residual\":"<<solve.residual<<",\"converged\":"<<(solve.residual<=tolerance?"true":"false")<<"}";
            std::ofstream second_solution(out+".second_solution.bin",std::ios::binary);
            second_solution.write(reinterpret_cast<const char*>(y.data()),y.size()*sizeof(y[0]));
        }
    }
    report<<"}\n";
    std::printf("%d-action diagnostic completed: setup %.3fs, steady matvec %.3fs\n",actions,setup,seconds);
    op.cleanup();return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
