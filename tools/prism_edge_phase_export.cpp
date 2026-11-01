#include "prism_edge_phase_candidates.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace bem_edge_phase;
int main(int argc,char** argv)try {
    if(argc!=5)throw std::invalid_argument("REF KA RAY_FILE PREFIX");
    const int ref=std::stoi(argv[1]);const double k=std::stod(argv[2]);const std::string prefix=argv[4];
    if(ref<0||ref>6)throw std::invalid_argument("refinement");
    const auto before=std::chrono::steady_clock::now();MullerP2BuildOptions o;o.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),o);
    MbsSurfaceSeed rays(argv[3]);const auto selected=candidates(mesh,rays,k);TraceProjector projector(mesh);
    std::ofstream data(prefix+".bin",std::ios::binary),meta(prefix+".json");
    if(!data||!meta)throw std::runtime_error("phase output open");
    meta<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<k<<",\"dofs\":"<<mesh.system_dofs()
        <<",\"count\":"<<selected.size()<<",\"projector_bytes\":"<<projector.bytes()<<",\"candidates\":[";
    double worst=0;
    for(size_t j=0;j<selected.size();++j){const auto& candidate=selected[j];const HertzTrace field(candidate.spec);
        auto column=projector.project(field,candidate.polarization);const auto refined=projector.project(field,candidate.polarization,2);
        double delta=0,norm=0;for(size_t i=0;i<column.size();++i){delta+=std::norm(column[i]-refined[i]);norm+=std::norm(refined[i]);}
        if(!std::isfinite(norm)||norm<=1e-280)throw std::runtime_error("empty phase basis column");
        const double error=std::sqrt(delta/norm);worst=std::max(worst,error);
        if(error>2e-9)throw std::runtime_error("phase edge quadrature control failed");
        for(auto& z:column)z/=std::sqrt(norm);
        if(!data.write(reinterpret_cast<const char*>(column.data()),column.size()*sizeof(C)))throw std::runtime_error("phase current write");
        const auto& s=candidate.spec;
        if(j)meta<<',';
        meta<<"{\"source\":\""<<candidate.source<<"\",\"ray_depth\":"<<candidate.ray_depth<<",\"polarization\":"<<candidate.polarization
            <<",\"family\":\""<<(s.family==Family::EdgeCone?"edge_cone":"shadow_plane")<<"\",\"medium_index\":"<<s.index
            <<",\"radial_wave_number_real\":"<<field.radial_wave_number().real()<<",\"radial_wave_number_imag\":"<<field.radial_wave_number().imag()
            <<",\"ray_strength\":"<<candidate.ray_strength<<",\"path_at_center\":"<<s.path<<",\"core\":"<<s.core<<",\"width\":"<<s.width
            <<",\"half_length\":"<<s.half_length<<",\"quadrature_error\":"<<error<<",\"raw_norm\":"<<std::sqrt(norm)<<'}';
    }
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
    meta<<"],\"elapsed_s_including_projection_control\":"<<seconds<<",\"quadrature_error\":"<<worst<<"}\n";
    std::cout<<"phase candidates="<<selected.size()<<" dofs="<<mesh.system_dofs()<<" seconds="<<seconds<<" quadrature_error="<<worst<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
