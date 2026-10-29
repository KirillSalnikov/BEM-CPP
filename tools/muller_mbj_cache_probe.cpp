// Build prism MBJ caches on the CPU without allocating an FMM on the GPU.
#include "muller_fmm.h"
#include "muller_mbj.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <sys/resource.h>

int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=6) throw std::runtime_error("Usage: mbj_probe ref ka nodes cache_path report_path");
    const int ref=std::stoi(argv[1]),nodes=std::stoi(argv[3]);
    const double ka=std::stod(argv[2]);
    const bool d6h=std::getenv("BEM_TMATRIX_D6H_MESH") &&
        std::string(std::getenv("BEM_TMATRIX_D6H_MESH"))!="0";
    MullerP2BuildOptions options;
    options.project_edge_nodes_to_sphere=false;
    options.edge_mode=MullerEdgeMode::HDivBdm1;
    // MBJ assembly only reads mesh, material and the regular quadrature.
    // It uses its own fixed singular rule, independent of FMM workspaces.
    MullerFmmOperator op;
    op.mesh=build_muller_p2_mesh(regular_prism(6,1.,ref,1.,0,false,d6h),options);
    op.current_dofs=op.mesh.current_dofs();op.system_dofs=op.mesh.system_dofs();
    op.k_exterior=ka;op.k_interior=ka*1.3;op.epsilon_interior=1.3*1.3;
    MullerMbjPreconditioner mbj;
    mbj.build_cached(op,nodes,0,argv[4]);
    rusage usage={};getrusage(RUSAGE_SELF,&usage);
    std::ofstream report(argv[5]);
    report<<std::setprecision(17)<<"{\"ka\":"<<ka<<",\"ref\":"<<ref
          <<",\"d6h_mesh\":"<<(d6h?"true":"false")<<",\"system_dofs\":"<<op.system_dofs
          <<",\"nodes\":"<<nodes<<",\"setup_s\":"<<mbj.setup_seconds
          <<",\"assembly_s\":"<<mbj.assembly_seconds<<",\"factorization_s\":"<<mbj.factorization_seconds
          <<",\"storage_mb\":"<<mbj.storage_megabytes()<<",\"peak_rss_mb\":"<<usage.ru_maxrss/1024.
          <<",\"cache_hit\":"<<(mbj.cache_hit?"true":"false")<<"}\n";
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
