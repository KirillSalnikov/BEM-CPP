// Combine verified physical coarse currents and fixed phase-basis columns.
// No application of A, no dense matrix, no GPU and no reuse of encoded Krylov vectors.
#include "physical_ritz_prior.h"
#include "muller_nodal.h"
#include <chrono>
#include <iomanip>
#include <iostream>
int main(int argc,char** argv)try {
    if(argc!=9)throw std::invalid_argument("REF KA DUFFY BASE_PRIOR PHASE_FILE EXTRA OUT_FILE REPORT");
    const int ref=std::stoi(argv[1]),extra=std::stoi(argv[6]);const double ka=std::stod(argv[2]);
    if(ref<2 || ref>6 || (extra!=8 && extra!=16))throw std::invalid_argument("phase prior refinement/extra count");
    if(std::ifstream(argv[7]).good() || std::ifstream(argv[8]).good())throw std::runtime_error("phase prior output exists");
    const auto start=std::chrono::steady_clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
    bem_ritz::Parameters p;p.dofs=mesh.system_dofs();p.columns=8;p.ref=ref;p.ka=ka;p.ri_real=1.3116;
    p.ri_imag=0;p.aspect=10./7;p.duffy=std::stoi(argv[3]);bem_ritz::Reader base(argv[4],p);
    std::ifstream phase(argv[5],std::ios::binary|std::ios::ate);
    if(!phase || phase.tellg()!=std::streamoff(24*p.dofs*sizeof(bem_ritz::Complex)))throw std::runtime_error("phase file size");
    const int selected[]={0,1,10,11,20,21,22,23,4,5,14,15,16,17,18,19};
    p.columns+=extra;double max_norm_error=0;
    bem_ritz::write(argv[7],p,[&](size_t j,bem_ritz::Vector& column){
        if(j<8)base.read(j,column);
        else {
            phase.seekg(size_t(selected[j-8])*p.dofs*sizeof(bem_ritz::Complex));
            if(!phase.read(reinterpret_cast<char*>(column.data()),column.size()*sizeof(bem_ritz::Complex)))throw std::runtime_error("phase read");
            long double norm=0;for(auto z:column)norm+=std::norm(z);
            const double error=std::abs(std::sqrt(double(norm))-1.);
            if(!std::isfinite(error)||error>2e-9)throw std::runtime_error("phase normalization gate");
            max_norm_error=std::max(max_norm_error,error);
        }
    });
    bem_ritz::Reader verified(argv[7],p);bem_ritz::Vector actual(p.dofs),expected(p.dofs);
    double maximum_difference=0;
    for(size_t j=0;j<p.columns;++j){verified.read(j,actual);
        if(j<8)base.read(j,expected);
        else {phase.seekg(size_t(selected[j-8])*p.dofs*sizeof(bem_ritz::Complex));
            if(!phase.read(reinterpret_cast<char*>(expected.data()),expected.size()*sizeof(bem_ritz::Complex)))throw std::runtime_error("phase verify read");}
        for(size_t i=0;i<p.dofs;++i)maximum_difference=std::max(maximum_difference,std::abs(actual[i]-expected[i]));
    }
    if(maximum_difference!=0)throw std::runtime_error("phase prior write changed a physical current");
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream report(argv[8]);report<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<p.dofs
        <<",\"ri\":1.3116,\"ri_imag\":0,\"aspect\":"<<p.aspect<<",\"duffy\":"<<p.duffy
        <<",\"base_columns\":8,\"phase_columns\":"<<extra<<",\"columns\":"<<p.columns<<",\"phase_indices\":[";
    for(int j=0;j<extra;++j)report<<(j?",":"")<<selected[j];
    report<<"],\"maximum_phase_norm_error\":"<<max_norm_error<<",\"roundtrip_maximum_difference\":"<<maximum_difference
        <<",\"total_s\":"<<seconds<<",\"GPU_used\":false}\n";report.close();if(!report)throw std::runtime_error("phase prior report write");
    std::cout<<"PASS physical phase prior columns="<<p.columns<<" dofs="<<p.dofs<<" seconds="<<seconds<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
