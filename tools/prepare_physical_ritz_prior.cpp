// Sparse BDM transfer of already decoded physical Ritz currents, CPU only.
#include "bdm_mesh_transfer.h"
#include "physical_ritz_prior.h"
#include "tmatrix_symmetry_warm_start.h"
#include <chrono>
#include <iomanip>
#include <iostream>
using Vector=bem_ritz::Vector;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
static double relative(const Vector& a,const Vector& b){
    long double delta=0,norm=0;
    for(size_t i=0;i<a.size();++i){delta+=std::norm(a[i]-b[i]);norm+=std::norm(b[i]);}
    return std::sqrt(double(delta/std::max(norm,1e-300L)));
}
int main(int argc,char** argv)try {
    if(argc!=11)throw std::invalid_argument("SRC_REF TARGET_REF KA RI_RE RI_IM ASPECT DUFFY TRAINING_DIR OUT_FILE REPORT_JSON");
    const int source_ref=std::stoi(argv[1]),target_ref=std::stoi(argv[2]);
    if(source_ref<0 || target_ref<2 || target_ref>6 || source_ref>=target_ref)throw std::invalid_argument("BDM source/target refinement");
    bem_ritz::Parameters p;p.columns=8;p.ref=target_ref;p.ka=std::stod(argv[3]);p.ri_real=std::stod(argv[4]);
    p.ri_imag=std::stod(argv[5]);p.aspect=std::stod(argv[6]);p.duffy=std::stoi(argv[7]);p.dofs=1;bem_ritz::validate(p);
    if(std::ifstream(argv[9]).good() || std::ifstream(argv[10]).good())throw std::runtime_error("prior preparation output exists");
    const auto start=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto source=build_muller_p2_mesh(regular_prism(6,p.aspect,source_ref,1.,0,false,true),options);
    const auto target=build_muller_p2_mesh(regular_prism(6,p.aspect,target_ref,1.,0,false,true),options);
    p.dofs=target.system_dofs();const double geometry_s=seconds(start);const auto transfer_start=Clock::now();
    BdmMeshTransfer transfer;transfer.build(source,target);const double transfer_setup_s=seconds(transfer_start);
    const auto audit_start=Clock::now();bem_tmatrix::D6hOrbit coarse,fine;coarse.build(source);fine.build(target);
    Vector original(source.system_dofs()),base(p.dofs),rotated,transferred(p.dofs),expected;
    for(size_t i=0;i<original.size();++i)original[i]=bem_ritz::Complex(std::sin(.131*i),std::cos(.27*i));
    transfer.apply(original.data(),base.data());double defects[3]={};
    for(int g=0;g<3;++g){
        if(g<2){coarse.transform(original,rotated,g==1,false);fine.transform(base,expected,g==1,false);}
        else {
            std::vector<int> index,sign;bem_tmatrix::D6hOrbit::vertical_image_map(source,index,sign);rotated.resize(original.size());
            for(size_t i=0;i<original.size();++i)rotated[index[i]]=double(sign[i])*original[i];
            bem_tmatrix::D6hOrbit::vertical_image_map(target,index,sign);expected.resize(base.size());
            for(size_t i=0;i<base.size();++i)expected[index[i]]=double(sign[i])*base[i];
        }
        transfer.apply(rotated.data(),transferred.data());defects[g]=relative(transferred,expected);
        if(!std::isfinite(defects[g]) || defects[g]>2e-12)throw std::runtime_error("BDM physical-prior D6h transfer check");
    }
    const double audit_s=seconds(audit_start);const auto application_start=Clock::now();
    bem_ritz::write(argv[9],p,[&](size_t j,Vector& column){
        const auto path=std::string(argv[8])+"/ritz_"+std::to_string(j)+".bin";
        std::ifstream in(path,std::ios::binary|std::ios::ate);
        if(!in || in.tellg()!=std::streamoff(original.size()*sizeof(bem_ritz::Complex)))throw std::runtime_error("wrong source current size: "+path);
        in.seekg(0);in.read(reinterpret_cast<char*>(original.data()),original.size()*sizeof(bem_ritz::Complex));
        for(auto z:original)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("source current not finite");
        transfer.apply(original.data(),column.data());
    });
    const double application_s=seconds(application_start);bem_ritz::Reader check(argv[9],p);
    std::ofstream report(argv[10]);report<<std::setprecision(17)
        <<"{\"source_ref\":"<<source_ref<<",\"target_ref\":"<<target_ref<<",\"source_dofs\":"<<source.system_dofs()
        <<",\"target_dofs\":"<<target.system_dofs()<<",\"columns\":8,\"ka\":"<<p.ka<<",\"ri_real\":"<<p.ri_real
        <<",\"ri_imag\":"<<p.ri_imag<<",\"aspect\":"<<p.aspect<<",\"quad\":7,\"duffy\":"<<p.duffy
        <<",\"geometry_s\":"<<geometry_s<<",\"transfer_setup_s\":"<<transfer_setup_s<<",\"transfer_audit_s\":"<<audit_s
        <<",\"transfer_and_write_s\":"<<application_s<<",\"total_s\":"<<seconds(start)<<",\"sparse_transfer_bytes\":"<<transfer.bytes()
        <<",\"D6h_transfer_defects\":["<<defects[0]<<","<<defects[1]<<","<<defects[2]<<"],\"GPU_used\":false}\n";
    report.close();if(!report)throw std::runtime_error("prior transfer report write");
    std::cout<<"PASS physical Ritz transfer "<<source.system_dofs()<<" -> "<<target.system_dofs()<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
