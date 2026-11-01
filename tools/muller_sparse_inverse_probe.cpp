#include "muller_fmm.h"
#include "muller_dense.h"
#include "prism_sparse_inverse.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include <fstream>
#include <iomanip>
#include <memory>
#include <atomic>
using namespace bem_tmatrix;
using C=std::complex<double>;using V=std::vector<C>;using Action=bem_streaming::Action;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static double error(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}
    const double result=std::sqrt(x/std::max(y,1e-300));if(!std::isfinite(result))throw std::runtime_error("nonfinite error");return result;}
static V read_rhs(const std::string& path,int n,int phase){std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(size_t(n)*sizeof(C)*2))throw std::runtime_error("SPAI RHS length");
    f.seekg(size_t(phase)*n*sizeof(C));V v(n);f.read(reinterpret_cast<char*>(v.data()),n*sizeof(C));if(!f)throw std::runtime_error("SPAI RHS read");return v;}
int main(int argc,char** argv)try{
    if(argc<7)throw std::invalid_argument("usage: REF KA SUPPORT_EDGES PACKET_ROOT REPORT DEGREE...");
    const int ref=std::stoi(argv[1]),ka=std::stoi(argv[2]),support=std::stoi(argv[3]);
    if(ref<2||ref>5||ka<1||support<1||support>64)throw std::invalid_argument("SPAI screen parameters");
    setvbuf(stdout,nullptr,_IOLBF,0);std::ofstream log(argv[5]);log<<std::setprecision(17);
    auto before=Clock::now();MullerFmmOperator op;struct Cleanup{MullerFmmOperator& o;~Cleanup(){o.cleanup();}}cleanup{op};
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,6,5,64,false,2,0,.96,nullptr,3);
    op.set_fmm_near_fp32(true);op.enable_prism_face_fft(ref,10./7);const int n=op.system_dofs;
    D6hOrbit orbit;orbit.build(op.mesh);PrismGroupMaps maps;maps.build(orbit);PrismSymmetrySectors sectors;sectors.build(maps);
    const Action action=[&](const C* a,C* b){op.matvec(a,b);},strict=[&](const C* a,C* b){op.matvec_strict(a,b);};
    std::vector<std::vector<int>> element_support(op.mesh.topology_edge_count);
    for(size_t t=0;t<op.mesh.elements.size();++t)for(int e:op.mesh.elements[t].topology_edges)element_support[e].push_back(t);
    const auto assemble=[&](const std::vector<int>& ids){std::vector<int> elements;
        for(int e:ids)elements.insert(elements.end(),element_support[e].begin(),element_support[e].end());
        std::sort(elements.begin(),elements.end());elements.erase(std::unique(elements.begin(),elements.end()),elements.end());return assemble_muller_nodal_block(op,ids,&elements);};
    std::vector<int> old_block(op.mesh.topology_edge_count,-1);
    PrismOrbitInverse local;local.build(op.mesh,maps,50,[&](const std::vector<int>& ids){
        for(int e:ids)for(int g=0;g<12;++g)old_block[maps.image[g][2*e]/2]=ids.front();
        return assemble(ids);});
    PrismCirculantInverse fft;
    fft.build(op.mesh,ref,action,[&](const C* a,C* b){local.apply(a,b);});
    PrismSymmetricCirculant inverse;inverse.build(orbit,fft,local);if(!inverse.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("FFT D6h proof");
    const double common_setup=elapsed(before);before=Clock::now();
    std::atomic<size_t> outside_entries{0},crossing_stencils{0};
    bem_spai::PrismSparseInverse sparse;sparse.build(op.mesh,maps,support,2*support,[&](const std::vector<int>& ids){
        size_t outside=0;for(int j=0;j<std::min(support,int(ids.size()));++j)outside+=old_block[ids[j]]!=old_block[ids[0]];
        outside_entries+=outside;crossing_stencils+=outside>0;return assemble(ids);});
    const double sparse_setup=elapsed(before);V scratch(n);
    struct Variant{std::string name;Action inverse;std::unique_ptr<bem_streaming::SectorRecycleSpace> recycle;bool failed=false;};
    std::vector<Variant> variants;const auto add=[&](std::string name,double weight){Variant v;v.name=name;
        v.inverse=[&,weight](const C* rhs,C* out){if(weight==0){inverse.apply_full_fft_symmetry(rhs,out,.5);return;}
            if(weight==1){sparse.apply(rhs,out);return;}
            inverse.apply_full_fft_symmetry(rhs,out,.5);sparse.apply(rhs,scratch.data());
#pragma omp parallel for schedule(static)
            for(int i=0;i<n;++i)out[i]=(1-weight)*out[i]+weight*scratch[i];};
        v.recycle.reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));variants.push_back(std::move(v));};
    add("baseline",0);add("spai25",.25);add("spai50",.5);add("spai_only",1);add("baseline_repeat",0);
    V test(n),image(n),transformed(n),actual(n),expected(n);for(int i=0;i<n;++i)test[i]=C(std::sin(.021*i),std::cos(.031*i));
    std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(op.mesh,vertical,sign);double defect=0;
    for(auto& variant:variants){variant.inverse(test.data(),image.data());
        for(bool mirror:{false,true}){orbit.transform(test,transformed,mirror,true);variant.inverse(transformed.data(),actual.data());orbit.transform(image,expected,mirror,false);defect=std::max(defect,error(actual,expected));}
        for(int i=0;i<n;++i){transformed[vertical[i]]=-double(sign[i])*test[i];expected[vertical[i]]=double(sign[i])*image[i];}
        variant.inverse(transformed.data(),actual.data());defect=std::max(defect,error(actual,expected));}
    if(defect>1e-8)throw std::runtime_error("SPAI D6h defect");
    log<<"{\"type\":\"setup\",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<n<<",\"support_edges\":"<<support
       <<",\"test_edges\":"<<2*support<<",\"common_setup_s\":"<<common_setup<<",\"sparse_setup_s\":"<<sparse_setup
       <<",\"sparse_bytes\":"<<sparse.storage_bytes()<<",\"blend_scratch_bytes\":"<<n*sizeof(C)<<",\"local_bytes\":"<<local.storage_bytes()
       <<",\"patches\":"<<sparse.patches()<<",\"maximum_fit_defect\":"<<sparse.maximum_fit_defect
       <<",\"seed_stencils_crossing_old_blocks\":"<<crossing_stencils.load()<<",\"seed_support_edges_outside_old_blocks\":"<<outside_entries.load()
       <<",\"rank_deficient_patches\":"<<sparse.rank_deficient_patches<<",\"D6h_defect\":"<<defect<<"}\n"<<std::flush;
    if(sparse.rank_deficient_patches)throw std::runtime_error("rank-deficient SPAI screen: inspect before solving");
    if(!outside_entries)throw std::runtime_error("SPAI stencil did not extend beyond old blocks");
    std::vector<Vec3> directions;for(int angle=0;angle<=180;angle+=3){const double theta=angle*std::acos(-1.)/180;directions.emplace_back(std::sin(theta)*std::cos(.27),std::sin(theta)*std::sin(.27),std::cos(theta));}
    for(int arg=6;arg<argc;++arg){const int degree=std::stoi(argv[arg]);
        const std::string path=std::string(argv[4])+"/packet_ref"+std::to_string(ref)+"_ka"+std::to_string(ka)+"_l"+std::to_string(degree)+"/packet.rhs.bin";
        const V rhs[2]={read_rhs(path,n,0),read_rhs(path,n,1)};V reference[2];
        for(auto& variant:variants){if(variant.failed)continue;V x[2]={V(n),V(n)};double inverse_s=0;int inverse_calls=0;
            const Action measured=[&](const C* a,C* b){auto t=Clock::now();variant.inverse(a,b);inverse_s+=elapsed(t);++inverse_calls;};
            before=Clock::now();auto a=solve_prism_sectors(action,strict,measured,sectors,rhs[0].data(),x[0].data(),64,2e-6,300,false,true,variant.recycle.get());
            bem_streaming::HostResult b;
            if(a.residual<=2e-6){auto warm=symmetry_guess(orbit,rhs[0],x[0],rhs[1],x[1]);b=solve_prism_sectors(action,strict,measured,sectors,rhs[1].data(),x[1].data(),64,2e-6,300,warm.fitted,true,variant.recycle.get());}
            const double seconds=elapsed(before);before=Clock::now();double residual[2],field_error[2];
            for(int j=0;j<2;++j){strict(x[j].data(),image.data());residual[j]=error(image,rhs[j]);
                V field;muller_nodal_farfield(op.mesh,x[j].data(),x[j].data()+n/2,ka,directions,field,13);if(variant.name=="baseline")reference[j]=field;field_error[j]=error(field,reference[j]);}
            const bool passed=std::max(residual[0],residual[1])<=2e-6&&std::max(field_error[0],field_error[1])<=1e-4;
            log<<"{\"type\":\"packet\",\"variant\":\""<<variant.name<<"\",\"degree\":"<<degree<<",\"seconds\":"<<seconds
               <<",\"validation_s\":"<<elapsed(before)<<",\"iterations\":["<<a.iterations<<","<<b.iterations<<"],\"inverse_s\":"<<inverse_s
               <<",\"inverse_calls\":"<<inverse_calls<<",\"strict_residual\":["<<residual[0]<<","<<residual[1]<<"],\"far_error\":["<<field_error[0]<<","<<field_error[1]
               <<"],\"passed\":"<<(passed?"true":"false")<<"}\n"<<std::flush;
            if(!passed){variant.failed=true;if(variant.name.find("baseline")==0)throw std::runtime_error("baseline failed");}
        }
    }
}catch(const std::exception& e){std::fprintf(stderr,"SPAI screen: %s\n",e.what());return 1;}
