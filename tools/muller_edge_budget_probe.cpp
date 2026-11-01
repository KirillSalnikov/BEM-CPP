// Long sequence with independent recycling and a fixed inverse-memory budget.
#include "muller_fmm.h"
#include "muller_dense.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include "prism_edge_correction.h"
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
using namespace bem_tmatrix;
using C=std::complex<double>;using V=std::vector<C>;using Action=bem_streaming::Action;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static double error(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}
    const double e=std::sqrt(x/std::max(y,1e-300));if(!std::isfinite(e))throw std::runtime_error("nonfinite error");return e;}
static V read_rhs(const std::string& path,int n,int phase){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(size_t(n)*sizeof(C)*2))throw std::runtime_error("RHS size: "+path);
    f.seekg(size_t(phase)*n*sizeof(C));V v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(C));
    if(!f)throw std::runtime_error("RHS read");return v;
}
int main(int argc,char** argv)try {
    if(argc!=5)throw std::invalid_argument("usage: REF KA PACKET_ROOT REPORT.jsonl");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    if(ref<2||ref>5||ka!=40)throw std::invalid_argument("this packet sequence requires ka40 and ref2..5");
    setvbuf(stdout,nullptr,_IOLBF,0);
    std::ofstream log(argv[4]);log<<std::setprecision(17);
    const auto start=Clock::now();MullerFmmOperator op;struct Cleanup{MullerFmmOperator& o;~Cleanup(){o.cleanup();}} cleanup{op};
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,6,5,64,false,2,0,.96,nullptr,3);
    op.set_fmm_near_fp32(true);op.enable_prism_face_fft(ref,10./7);
    const int n=op.system_dofs;D6hOrbit orbit;orbit.build(op.mesh);PrismGroupMaps maps;maps.build(orbit);
    PrismSymmetrySectors sectors;sectors.build(maps);
    const Action action=[&](const C* a,C* b){op.matvec(a,b);},strict=[&](const C* a,C* b){op.matvec_strict(a,b);};
    PrismOrbitInverse local50,local100,edge_local;local50.build(op,maps,50);
    PrismCirculantInverse circ;circ.build(op.mesh,ref,action,[&](const C* a,C* b){local50.apply(a,b);});
    PrismSymmetricCirculant sym50,sym100,edge_sym;
    const auto prepare=[&](PrismSymmetricCirculant& sym,PrismOrbitInverse& local){sym.build(orbit,circ,local);
        if(!sym.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("full D6h FFT proof");};
    prepare(sym50,local50);const double common_setup=elapsed(start);
    auto before=Clock::now();local100.build(op,maps,100);prepare(sym100,local100);const double uniform_setup=elapsed(before);
    const size_t budget=local100.storage_bytes();
    if(budget!=PrismOrbitInverse::estimated_storage_bytes(maps,n,100))throw std::runtime_error("uniform storage estimate");
    std::vector<std::vector<int>> support(op.mesh.topology_edge_count);
    for(size_t t=0;t<op.mesh.elements.size();++t)for(int e:op.mesh.elements[t].topology_edges)support[e].push_back(t);
    before=Clock::now();PrismEdgeCorrection edge;
    edge.build(op.mesh,maps,-1,[&](const std::vector<int>& selected){std::vector<int> elements;
        for(int e:selected)elements.insert(elements.end(),support[e].begin(),support[e].end());
        std::sort(elements.begin(),elements.end());elements.erase(std::unique(elements.begin(),elements.end()),elements.end());
        return assemble_muller_nodal_block(op,selected,&elements);});
    const size_t edge_extra=edge.bytes()+size_t(n)*sizeof(C);
    int nodes=0;
    for(int trial=1;trial<=100;++trial)if(edge_extra+PrismOrbitInverse::estimated_storage_bytes(maps,n,trial)<=budget)nodes=trial;
    if(!nodes){log<<"{\"type\":\"rejected_memory\",\"budget_bytes\":"<<budget<<",\"edge_bytes\":"<<edge_extra<<"}\n";return 0;}
    edge_local.build(op,maps,nodes);prepare(edge_sym,edge_local);const double edge_setup=elapsed(before);
    const size_t candidate_bytes=edge_extra+edge_local.storage_bytes();
    if(candidate_bytes>budget)throw std::runtime_error("edge inverse exceeds common memory cap");
    V correction(n);
    const auto edge_action=[&](const C* a,C* b,double weight){edge_sym.apply_full_fft_symmetry(a,b,.5);edge.apply(a,correction.data());
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)if(edge.covers(i))b[i]=(1-weight)*b[i]+weight*correction[i];};
    struct Variant {std::string name;Action inverse;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;bool failed=false;double setup;size_t bytes;};
    std::vector<Variant> variants;
    const auto add=[&](const std::string& name,Action inverse,double setup,size_t bytes){
        Variant v;v.name=name;v.inverse=std::move(inverse);v.setup=setup;v.bytes=bytes;
        v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));};
    add("baseline50",[&](const C* a,C* b){sym50.apply_full_fft_symmetry(a,b,.5);},0,local50.storage_bytes());
    add("uniform100",[&](const C* a,C* b){sym100.apply_full_fft_symmetry(a,b,.5);},uniform_setup,budget);
    add("edge25",[&](const C* a,C* b){edge_action(a,b,.25);},edge_setup,candidate_bytes);
    add("edge50",[&](const C* a,C* b){edge_action(a,b,.5);},edge_setup,candidate_bytes);
    add("baseline50_repeat",variants[0].inverse,0,local50.storage_bytes());
    V test(n),value(n),transformed(n),actual(n),expected(n);for(int i=0;i<n;++i)test[i]=C(std::sin(.021*i),std::cos(.031*i));
    std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
    double defect=0;
    for(auto& variant:variants){variant.inverse(test.data(),value.data());
        for(bool mirror:{false,true}){orbit.transform(test,transformed,mirror,true);variant.inverse(transformed.data(),actual.data());
            orbit.transform(value,expected,mirror,false);defect=std::max(defect,error(actual,expected));}
        for(int i=0;i<n;++i){transformed[vertical[i]]=-double(sign[i])*test[i];expected[vertical[i]]=double(sign[i])*value[i];}
        variant.inverse(transformed.data(),actual.data());defect=std::max(defect,error(actual,expected));
    }
    if(defect>1e-8)throw std::runtime_error("budgeted inverse D6h defect");
    log<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<n<<",\"common_setup_s\":"<<common_setup
       <<",\"uniform100_setup_s\":"<<uniform_setup<<",\"edge_setup_s\":"<<edge_setup<<",\"edge_local_nodes\":"<<nodes
       <<",\"baseline50_bytes\":"<<local50.storage_bytes()<<",\"uniform100_budget_bytes\":"<<budget<<",\"edge_candidate_bytes\":"<<candidate_bytes
       <<",\"edge_rings\":-1,\"edge_block_max_dofs\":"<<edge.max_block_dofs<<",\"D6h_defect\":"<<defect<<"}\n"<<std::flush;
    std::vector<Vec3> directions;for(int t=0;t<=180;t+=3){const double theta=t*std::acos(-1.)/180;directions.emplace_back(std::sin(theta)*std::cos(.27),std::sin(theta)*std::sin(.27),std::cos(theta));}
    const int degrees[]={24,28,32,38,42,46,50,54,58,62};
    for(int degree:degrees){
        const std::string path=std::string(argv[3])+"/packet_ref"+std::to_string(ref)+"_ka40_l"+std::to_string(degree)+"/packet.rhs.bin";
        V rhs[2]={read_rhs(path,n,0),read_rhs(path,n,1)},reference[2];
        for(auto& variant:variants){if(variant.failed)continue;
            V x[2]={V(n),V(n)},image(n),field;double inverse_s=0;int inverse_calls=0;
            const Action measured=[&](const C* a,C* b){const auto t=Clock::now();variant.inverse(a,b);inverse_s+=elapsed(t);++inverse_calls;};
            before=Clock::now();
            const auto a=solve_prism_sectors(action,strict,measured,sectors,rhs[0].data(),x[0].data(),64,2e-6,300,false,true,variant.recycle.get());
            bem_streaming::HostResult b;
            if(a.residual<=2e-6){auto warm=symmetry_guess(orbit,rhs[0],x[0],rhs[1],x[1]);
                b=solve_prism_sectors(action,strict,measured,sectors,rhs[1].data(),x[1].data(),64,2e-6,300,warm.fitted,true,variant.recycle.get());}
            const double solve_s=elapsed(before);before=Clock::now();double residual[2],far_error[2];
            for(int j=0;j<2;++j){strict(x[j].data(),image.data());residual[j]=error(image,rhs[j]);
                muller_nodal_farfield(op.mesh,x[j].data(),x[j].data()+n/2,ka,directions,field,13);
                if(variant.name=="baseline50")reference[j]=field;
                far_error[j]=error(field,reference[j]);}
            const bool passed=std::max(residual[0],residual[1])<=2e-6 && std::max(far_error[0],far_error[1])<=1e-4;
            log<<"{\"type\":\"packet\",\"variant\":\""<<variant.name<<"\",\"degree\":"<<degree<<",\"seconds\":"<<solve_s
               <<",\"validation_s\":"<<elapsed(before)<<",\"iterations\":["<<a.iterations<<","<<b.iterations<<"],\"inverse_s\":"<<inverse_s
               <<",\"inverse_calls\":"<<inverse_calls<<",\"strict_residual\":["<<residual[0]<<","<<residual[1]<<"],\"far_error\":["<<far_error[0]<<","<<far_error[1]
               <<"],\"passed\":"<<(passed?"true":"false")<<"}\n"<<std::flush;
            if(!passed){variant.failed=true;if(variant.name.find("baseline50")==0)throw std::runtime_error("baseline failed");}
        }
    }
}catch(const std::exception& e){std::fprintf(stderr,"edge budget: %s\n",e.what());return 1;}
