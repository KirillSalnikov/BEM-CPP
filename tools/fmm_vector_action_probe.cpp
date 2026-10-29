// One physical medium on the full surface quadrature. This bounded diagnostic
// fits beside desktop applications; it does not build a BEM solve or T matrix.
#include "fmm.h"
#include "muller_nodal.h"
#include "quadrature.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if(argc!=5) throw std::runtime_error("Usage: probe REF K_REAL K_IMAG OUTPUT.json");
    const int ref=std::stoi(argv[1]);
    if(ref<4 || ref>7) throw std::runtime_error("ref must be 4..7");
    const cdouble k(std::stod(argv[2]),std::stod(argv[3]));
    const bool strict=std::getenv("BEM_PROBE_STRICT") && std::string(std::getenv("BEM_PROBE_STRICT"))!="0";
    const char* flag=strict?"BEM_FMM_MULTILEVEL_M2L_ROWS":"BEM_FMM_MULTILEVEL_COMPACT_TEMP";
    if(const char* requested=std::getenv("BEM_PROBE_COMPARE_FLAG")) flag=requested;
    if(std::string(flag)!="BEM_FMM_MULTILEVEL_COMPACT_TEMP" &&
       std::string(flag)!="BEM_FMM_MULTILEVEL_M2L_ROWS" &&
       std::string(flag)!="BEM_FMM_P2P_SHARED_REAL" && std::string(flag)!="BEM_FMM_P2P_SHARED_FAST_TRIG" &&
       std::string(flag)!="BEM_FMM_P2P_SHARED_SOURCES" && std::string(flag)!="combined" &&
       std::string(flag)!="strict_threads")
        throw std::runtime_error("unsupported runtime comparison flag");
    const auto begin=std::chrono::steady_clock::now();
    const Mesh linear=regular_prism(6,1.,ref,1.,0,false,true);
    MullerP2BuildOptions options; options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(linear,options);
    const TriQuad rule=tri_quadrature(7);
    const int points=int(mesh.elements.size())*rule.npts;
    std::vector<double> xyz(size_t(points)*3);
    std::vector<cdouble> q(size_t(points)*3);
    #pragma omp parallel for schedule(static)
    for(int e=0;e<int(mesh.elements.size());++e) for(int j=0;j<rule.npts;++j) {
        const int i=e*rule.npts+j;
        const auto frame=evaluate_muller_frame(mesh,e,rule.pts[j][0],rule.pts[j][1]);
        xyz[3*i]=frame.position.x; xyz[3*i+1]=frame.position.y; xyz[3*i+2]=frame.position.z;
        for(int d=0;d<3;++d)
            q[size_t(d)*points+i]=.5*rule.wts[j]*frame.jacobian*
                cdouble(std::sin(.013*(i+1)*(d+1)),std::cos(.017*(i+1)*(d+2)));
    }
    // Select the same uniform depths 6/5/4 as the production ka=112 profile.
    const int fine_leaf=std::max(1,int((2LL*points+262143)/262144));
    const int middle_leaf=std::max(1,int((2LL*points+32767)/32768));
    const int coarse_leaf=std::max(1,int((2LL*points+4095)/4096));
    HelmholtzFMM fine, middle, coarse;
    fine.configure_interaction_band(6,6,true); fine.configure_order_reference_depth(6);
    middle.configure_interaction_band(5,5,false); middle.configure_order_reference_depth(6);
    coarse.configure_interaction_band(1,4,false); coarse.configure_order_reference_depth(3);
    fine.init(xyz.data(),points,xyz.data(),points,k,5,fine_leaf,3,false,false,true);
    coarse.init(xyz.data(),points,xyz.data(),points,k,5,coarse_leaf,3,false,false,true);
    middle.init(xyz.data(),points,xyz.data(),points,k,5,middle_leaf,3,false,false,true);
    fine.near_field_fp32=middle.near_field_fp32=coarse.near_field_fp32=!strict;
    if(fine.tree.max_level!=6 || middle.tree.max_level!=5 || coarse.tree.max_level!=4)
        throw std::runtime_error("diagnostic requires uniform depths 6/5/4");
    const double setup=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::vector<cdouble> curl(size_t(points)*3), action(curl.size());
    auto evaluate=[&] {
        const auto start=std::chrono::steady_clock::now();
        fine.evaluate_vector_actions_batch3(q.data(),q.data()+points,q.data()+2LL*points,
                                            curl.data(),action.data(),&middle,&coarse);
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    };
    const bool combined=std::string(flag)=="combined";
    auto select=[&](bool enabled) {
        if(std::string(flag)=="strict_threads") {
            setenv("BEM_FMM_P2P_THREADS",enabled?"128":"512",1);
        } else if(combined) {
            setenv("BEM_FMM_MULTILEVEL_COMPACT_TEMP",enabled?"1":"0",1);
            setenv("BEM_FMM_P2P_SHARED_SOURCES",enabled?"1":"0",1);
        } else setenv(flag,enabled?"1":"0",1);
    };
    select(false); evaluate();
    const auto reference_curl=curl, reference_action=action;
    auto relative=[](const std::vector<cdouble>& a,const std::vector<cdouble>& b) {
        double delta=0,norm=0;
        #pragma omp parallel for reduction(+:delta,norm) schedule(static)
        for(size_t i=0;i<a.size();++i) { delta+=std::norm(a[i]-b[i]); norm+=std::norm(b[i]); }
        return std::sqrt(delta/std::max(norm,1e-300));
    };
    std::ofstream report(argv[4]);
    report<<std::setprecision(17)<<"{\"diagnostic_only\":true,\"single_medium\":true,\"ref\":"<<ref
          <<",\"k_real\":"<<k.real()<<",\"k_imag\":"<<k.imag()<<",\"points\":"<<points
          <<",\"triangles\":"<<linear.nt()<<",\"system_dofs\":"<<mesh.system_dofs()
          <<",\"strict\":"<<(strict?"true":"false")<<",\"setup_s\":"<<setup
          <<",\"compare_flag\":\""<<flag<<"\",\"comparison\":[";
    int trial=0; bool valid=true;
    for(int enabled: {0,1,1,0}) {
        select(enabled);
        const double elapsed=evaluate();
        const double ce=relative(curl,reference_curl), ae=relative(action,reference_action);
        valid=valid && std::isfinite(ce) && std::isfinite(ae) && ce<1e-10 && ae<1e-10;
        report<<(trial++?",":"")<<"{\"enabled\":"<<enabled<<",\"seconds\":"<<elapsed
              <<",\"curl_relative_error\":"<<ce<<",\"action_relative_error\":"<<ae<<"}";
        std::printf("[single medium] %s=%d seconds=%.6f curl_error=%.6e action_error=%.6e\n",flag,enabled,elapsed,ce,ae);
    }
    report<<"],\"valid\":"<<(valid?"true":"false")<<"}\n";
    fine.cleanup(); middle.cleanup(); coarse.cleanup();
    return valid?0:2;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
