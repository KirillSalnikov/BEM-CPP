// Same-operator ABBA screening of mixed-action kernels; never solves a T matrix.
#include "muller_fmm.h"
#include "physical_ritz_prior.h"
#include "host_thread_affinity.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Complex=std::complex<double>;
using Vector=std::vector<Complex>;
using Clock=std::chrono::steady_clock;
static double difference(const Vector& a,const Vector& b) {
    if(a.size()!=b.size())throw std::runtime_error("action dimensions");
    long double error=0,norm=0;
    for(size_t i=0;i<a.size();++i){error+=std::norm(a[i]-b[i]);norm+=std::norm(b[i]);}
    if(!(norm>0))throw std::runtime_error("zero action reference");
    const double value=std::sqrt(double(error/norm));
    if(!std::isfinite(value))throw std::runtime_error("nonfinite action difference");
    return value;
}
int main(int argc,char** argv)try {
    if(argc!=6)throw std::runtime_error("Usage: hotspot_probe ref ka prior.bin cache_prefix result.json");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    int repeats=1;
    if(const char* value=std::getenv("BEM_HOTSPOT_REPEATS")) {
        size_t used=0;repeats=std::stoi(value,&used);
        if(used!=std::string(value).size() || repeats<1 || repeats>20)throw std::runtime_error("invalid hotspot repeats");
    }
    const double aspect=10./7,ri=1.3116;const int leaf=19,duffy=6;
    bem_cpu::configure_threads(4,"BEM_CPU_GPU_AFFINITY");
    const Mesh mesh=regular_prism(6,aspect,ref,1.,0,false,true);
    MullerP2BuildOptions options;options.project_edge_nodes_to_sphere=false;
    options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto setup_start=Clock::now();MullerFmmOperator op;
    op.init(mesh,Complex(ka,0),Complex(ri,0),options,7,duffy,5,leaf,false,2,0,.96,
            (std::string(argv[4])+".near").c_str(),3);
    op.enable_prism_face_fft(ref,aspect);op.set_fmm_near_fp32(true);
    const double setup=std::chrono::duration<double>(Clock::now()-setup_start).count();
    // Independently compare these RHS quadrature positions with the frozen
    // T-worker fixture; matching sizes alone does not establish geometry.
    {
        std::ofstream positions(std::string(argv[5])+".positions.bin",std::ios::binary);
        for(const auto& q:op.mass_quadrature){
            const Vec3& p=q.sample.position;const double xyz[]={p.x,p.y,p.z};
            positions.write(reinterpret_cast<const char*>(xyz),sizeof(xyz));
        }
        positions.close();if(!positions)throw std::runtime_error("cannot write geometry positions");
    }
    bem_ritz::Parameters parameters;parameters.dofs=op.system_dofs;parameters.ref=ref;
    parameters.ka=ka;parameters.ri_real=ri;parameters.ri_imag=0;parameters.aspect=aspect;parameters.duffy=duffy;
    bem_ritz::Reader prior(argv[3],parameters);
    Vector inputs[3],mixed[3],strict[3],output(op.system_dofs);
    prior.read(0,inputs[0]);prior.read(1,inputs[1]);inputs[2].resize(op.system_dofs);
    std::mt19937 rng(1847);std::uniform_real_distribution<double> uniform(-1,1);
    for(auto& v:inputs[2])v=Complex(uniform(rng),uniform(rng));
    const char* names[]={"BEM_FMM_MULTILEVEL_L2P_WARP","BEM_FMM_P2P_SHARED_FAST_TRIG",
                         "BEM_FMM_P2P_LOCAL_COORDINATES","BEM_FMM_M2L_DIRECTION_BLOCKS"};
    const auto settings=[&](int warp,int trig,int local,int directions){
        const int values[]={warp,trig,local,directions};
        for(int i=0;i<4;++i)setenv(names[i],values[i]?"1":"0",1);
    };
    settings(0,0,0,1);
    for(int i=0;i<3;++i){
        mixed[i].resize(op.system_dofs);strict[i].resize(op.system_dofs);
        op.matvec(inputs[i].data(),mixed[i].data());
        op.matvec_strict(inputs[i].data(),strict[i].data());
    }
    struct Variant {const char* name;int warp,trig,local,directions;};
    const Variant variants[]={{"l2p_warp",1,0,0,1},{"p2p_fast_trig",0,1,0,1},
        {"p2p_local",0,0,1,1},{"p2p_fast_local",0,1,1,1},
        {"m2l_unsplit",0,0,0,0},{"l2p_warp_p2p_fast_local",1,1,1,1},{"l2p_warp_p2p_fast",1,1,0,1}};
    std::vector<std::string> selected;
    if(const char* value=std::getenv("BEM_HOTSPOT_VARIANTS")) {
        std::istringstream tokens(value);std::string token;
        while(std::getline(tokens,token,',')){
            bool known=false;for(const auto& v:variants)known=known || token==v.name;
            if(!known)throw std::runtime_error("unknown hotspot variant");selected.push_back(token);
        }
        if(selected.empty())throw std::runtime_error("empty hotspot variant selection");
    }
    std::ofstream report(argv[5]);report<<std::setprecision(17);
    report<<"{\"diagnostic_only\":true,\"d6h_mesh\":true,\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<op.system_dofs
          <<",\"ri_real\":"<<ri<<",\"aspect\":"<<aspect<<",\"duffy\":"<<duffy<<",\"max_leaf\":"<<leaf
          <<",\"setup_s\":"<<setup<<",\"repeats_per_input\":"<<repeats
          <<",\"input_types\":[\"physical_prior0\",\"physical_prior1\",\"random1847\"]"
          <<",\"baseline_mixed_to_strict\":[";
    for(int i=0;i<3;++i)report<<(i?",":"")<<difference(mixed[i],strict[i]);
    report<<"],\"variants\":[";
    bool first_variant=true;
    for(size_t v=0;v<sizeof(variants)/sizeof(variants[0]);++v) {
        const auto& variant=variants[v];
        if(!selected.empty() && std::find(selected.begin(),selected.end(),variant.name)==selected.end())continue;
        report<<(first_variant?"":",")<<"{\"name\":\""<<variant.name<<"\",\"settings\":{";first_variant=false;
        const int values[]={variant.warp,variant.trig,variant.local,variant.directions};
        for(int i=0;i<4;++i)report<<(i?",":"")<<"\""<<names[i]<<"\":\""<<values[i]<<"\"";
        report<<"},\"trials\":[";
        for(int trial=0;trial<4;++trial) {
            const bool candidate=trial==1||trial==2;
            if(candidate)settings(variant.warp,variant.trig,variant.local,variant.directions);
            else settings(0,0,0,1);
            // Each setting has an untimed warmup, including the control.
            op.matvec(inputs[0].data(),output.data());
            report<<(trial?",":"")<<"{\"candidate\":"<<(candidate?"true":"false")<<",\"inputs\":[";
            for(int i=0;i<3;++i) {
                const auto before=Clock::now();
                for(int repeat=0;repeat<repeats;++repeat)op.matvec(inputs[i].data(),output.data());
                const double seconds=std::chrono::duration<double>(Clock::now()-before).count()/repeats;
                const double delta=difference(output,mixed[i]),exact=difference(output,strict[i]);
                report<<(i?",":"")<<"{\"seconds\":"<<seconds<<",\"difference_from_baseline\":"<<delta
                      <<",\"difference_from_strict\":"<<exact<<"}";
                std::printf("[hotspot] %s candidate=%d input=%d seconds=%.6f delta=%.3e strict=%.3e\n",
                            variant.name,int(candidate),i,seconds,delta,exact);
            }
            report<<"]}";report.flush();
        }
        settings(variant.warp,variant.trig,variant.local,variant.directions);
        report<<"],\"strict_drift\":[";
        for(int i=0;i<3;++i){
            op.matvec_strict(inputs[i].data(),output.data());
            report<<(i?",":"")<<difference(output,strict[i]);
        }
        report<<"]}";report.flush();
    }
    settings(0,0,0,1);report<<"]}\n";report.close();
    if(!report)throw std::runtime_error("cannot write hotspot report");
    op.cleanup();return 0;
}catch(const std::exception& e){std::fprintf(stderr,"hotspot probe: %s\n",e.what());return 1;}
