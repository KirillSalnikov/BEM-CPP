// One bounded, matrix-free action for profiling. This is not a physical solve.
#include "muller_fmm.h"
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

int main(int argc,char**argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=5)throw std::runtime_error("Usage: probe ref ka cache_prefix output_prefix");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    const std::string cache=argv[3],out=argv[4];
    const Mesh mesh=regular_prism(6,1.,ref,1.);
    MullerP2BuildOptions options;options.project_edge_nodes_to_sphere=false;
    options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto begin=std::chrono::steady_clock::now();
    MullerFmmOperator op;
    int leaf=std::max(64,int((2LL*mesh.nt()*7+32767)/32768));
    if(const char* value=std::getenv("BEM_PROBE_MAX_LEAF"))leaf=std::stoi(value);
    const bool strict=std::getenv("BEM_PROBE_STRICT") &&
        std::string(std::getenv("BEM_PROBE_STRICT"))!="0";
    const double absorption=std::getenv("BEM_PROBE_RI_IMAG")?std::stod(std::getenv("BEM_PROBE_RI_IMAG")):0.;
    const int seed=std::getenv("BEM_PROBE_SEED")?std::stoi(std::getenv("BEM_PROBE_SEED")):0;
    op.init(mesh,std::complex<double>(ka,0),std::complex<double>(1.3,absorption),options,
            7,4,5,leaf,false,2,0,.96,(cache+".near").c_str(),3);
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
    report<<"{\"diagnostic_only\":true,\"actions\":"<<actions<<",\"system_dofs\":"<<op.system_dofs
          <<",\"ka\":"<<ka<<",\"ref\":"<<ref<<",\"max_leaf\":"<<leaf
          <<",\"ri_imag\":"<<absorption<<",\"seed\":"<<seed
          <<",\"strict\":"<<(strict?"true":"false")
          <<",\"warm_mixed\":"<<(warm_mixed?"true":"false")<<",\"setup_s\":"<<setup
          <<",\"matvec_s\":"<<seconds<<",\"action_seconds\":[";
    for(int i=0;i<actions;++i)report<<(i?",":"")<<timings[i];
    report<<"]}\n";
    std::printf("%d-action diagnostic completed: setup %.3fs, steady matvec %.3fs\n",actions,setup,seconds);
    op.cleanup();return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
