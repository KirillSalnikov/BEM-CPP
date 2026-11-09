// Persistent native P12 action/solve bridge. All matrices remain implicit;
// selected constraints are ORIGINAL Muller equations, never model equations.
#include "muller_fmm.h"
#include "prism_face_fft.h"
#include "selected_operator_rows.h"
#include "coarse_image_experiment.h"
#include "prism_d6h_model_action.h"
#include "prism_orbit_inverse.h"
#include "prism_sector_solver.h"
#include "prism_symmetric_circulant.h"
#include "quadrature.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
template<class T>std::vector<T> load(const std::string& path,size_t multiple,size_t max_blocks=1){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!multiple||!f||f.tellg()<=0||f.tellg()%std::streamoff(multiple*sizeof(T))||
       f.tellg()>std::streamoff(max_blocks*multiple*sizeof(T)))throw std::runtime_error("input size: "+path);
    const size_t count=size_t(f.tellg())/sizeof(T);std::vector<T> out(count);f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(out.data()),count*sizeof(T)))throw std::runtime_error("read: "+path);
    return out;
}
template<class T>void save(const std::string& path,const std::vector<T>& data){
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));f.close();
    if(!f)throw std::runtime_error("write: "+path);
}
double relative(const Vector& a,const Vector& b){
    double s=0,t=0;for(size_t i=0;i<a.size();++i){s+=std::norm(a[i]-b[i]);t+=std::norm(b[i]);}return std::sqrt(s/std::max(t,1e-300));
}
int main(int argc,char** argv)try {
    if(argc!=10)throw std::invalid_argument("REF KA ASPECT RI_IMAG DUFFY PRIOR_BIN ROWS_BIN CACHE DIRECTORY");
    const int ref=std::stoi(argv[1]),duffy=std::stoi(argv[5]);
    const double ka=std::stod(argv[2]),aspect=std::stod(argv[3]),imag=std::stod(argv[4]);
    if(ref<2||ref>6||duffy<6||duffy>32||!std::isfinite(ka)||ka<=0||!std::isfinite(aspect)||aspect<=0||!std::isfinite(imag)||imag<0)
        throw std::invalid_argument("invalid native P12 geometry/material/quadrature");
    const std::string directory=argv[9],cache=argv[8];
    // Native operator logs go to stderr; the original stdout is an exclusive
    // line protocol. No profiling or progress line can masquerade as a reply.
    const int descriptor=dup(STDOUT_FILENO);if(descriptor<0)throw std::runtime_error("protocol dup");
    struct FileCloser{void operator()(FILE* f)const{if(f)fclose(f);}};
    std::unique_ptr<FILE,FileCloser> reply(fdopen(descriptor,"w"));
    if(!reply)throw std::runtime_error("protocol fdopen");
    fflush(stdout);if(dup2(STDERR_FILENO,STDOUT_FILENO)<0)throw std::runtime_error("log redirect");
    setvbuf(stdout,nullptr,_IOLBF,0);
    setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);setenv("BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1",1);
    setenv("BEM_FMM_MULTILEVEL","1",1);setenv("BEM_FMM_PAIR_CURRENTS","0",1);
    // Match the geometry-derived tree depths used by the transpose controls.
    const int seg=1<<ref;int nz=std::max(seg,int(std::ceil(std::sqrt(3.)*aspect*seg)));nz+=nz%2;
    const int points=7*12*seg*(seg+nz),depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
    setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
    setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    const auto begin=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;struct Cleanup{MullerFmmOperator& op;~Cleanup(){op.cleanup();}} cleanup{op};
    op.init(regular_prism(6,aspect,ref,1.,0,false,true),Complex(ka),Complex(1.3116,imag),options,
            7,duffy,5,leaf(depth),false,2,0,.96,(cache+".near").c_str(),3);
    op.enable_prism_face_fft(ref,aspect);op.set_fmm_near_fp32(true);
    const double operator_setup=seconds(begin);const int n=op.system_dofs;
    const auto rows=load<int>(argv[7],1,n);const auto prior=load<Complex>(argv[6],n,64);
    for(auto z:prior)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite prior");
    size_t actions=0,strict_actions=0,adjoint_actions=0;
    const bem_streaming::Action action=[&](const Complex* x,Complex* y){++actions;op.matvec(x,y);};
    const bem_streaming::Action strict=[&](const Complex* x,Complex* y){++strict_actions;op.matvec_strict(x,y);};
    const bem_streaming::Action adjoint=[&](const Complex* x,Complex* y){++adjoint_actions;op.matvec_adjoint_strict(x,y);};
    SelectedOperatorRows constraints(n,rows,strict,adjoint);
    const auto precondition_start=Clock::now();D6hOrbit group;group.build(op.mesh);
    PrismGroupMaps maps;maps.build(group);PrismSymmetrySectors sectors;sectors.build(maps);
    PrismOrbitInverse local;local.build(op,maps,50);
    PrismCirculantInverse circ;circ.build(op.mesh,ref,action,[&](const Complex* x,Complex* y){local.apply(x,y);},0.,0.,3);
    PrismSymmetricCirculant symmetric;symmetric.build(group,circ,local);
    if(!symmetric.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("full FFT/circulant symmetry preparation failed");
    const bem_streaming::Action original=[&](const Complex* x,Complex* y){symmetric.apply_full_fft_symmetry(x,y,.5);};
    const double precondition_s=seconds(precondition_start);
    bool symmetrize=false;if(const char* p=std::getenv("BEM_P12_SYMMETRIC_IMAGES"))symmetrize=std::string(p)!="0";
    PrismD6hModelAction model(op.mesh,maps,strict);CoarseImageExperiment correction(op.mesh,sectors,original);
    const auto images_start=Clock::now();const size_t image_actions_before=actions+strict_actions;
    Vector column(n),image(n);
    for(size_t j=0;j<prior.size()/n;++j){
        std::copy(prior.begin()+j*n,prior.begin()+(j+1)*n,column.begin());
        if(symmetrize)model.apply(column.data(),image.data());else action(column.data(),image.data());
        correction.add_d6h(column,image);
    }
    const double image_s=seconds(images_start);const size_t image_actions=actions+strict_actions-image_actions_before;
    const bem_streaming::Action inverse=[&](const Complex* x,Complex* y){correction.apply(x,y);};
    std::vector<int> vertical,sign,symmetry;D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
    for(int g=0;g<3;++g){std::vector<int> indices,signs;if(g<2)group.image_map(g==1,false,indices,signs);else{indices=vertical;signs=sign;}
        symmetry.insert(symmetry.end(),indices.begin(),indices.end());symmetry.insert(symmetry.end(),signs.begin(),signs.end());}
    save(directory+"/geometry.symmetry.bin",symmetry);save(directory+"/selected_rows.bin",rows);
    std::vector<double> metric(n);const auto quad=tri_quadrature(13);
    for(size_t t=0;t<op.mesh.elements.size();++t)for(int q=0;q<quad.npts;++q){
        const auto frame=evaluate_muller_frame(op.mesh,t,quad.pts[q][0],quad.pts[q][1]);const auto basis=evaluate_muller_basis(op.mesh,t,frame);
        for(int j=0;j<basis.count;++j){const double value=.5*quad.wts[q]*frame.jacobian*basis.values[j].dot(basis.values[j]);
            metric[basis.dofs[j]]+=value;metric[op.current_dofs+basis.dofs[j]]+=value;}}
    for(double x:metric)if(!std::isfinite(x)||x<=0)throw std::runtime_error("nonpositive current metric");
    save(directory+"/geometry.metric.bin",metric);
    Vector probe(n),value(n),rotated(n),actual(n),expected(n);
    for(int j=0;j<n;++j)probe[j]=Complex(std::sin(.031*j),std::cos(.043*j));
    inverse(probe.data(),value.data());double defect=0;
    for(int g=0;g<3;++g){
        if(g<2){group.transform(probe,rotated,g==1,true);group.transform(value,expected,g==1,false);}
        else for(int j=0;j<n;++j){rotated[vertical[j]]=-double(sign[j])*probe[j];expected[vertical[j]]=double(sign[j])*value[j];}
        inverse(rotated.data(),actual.data());defect=std::max(defect,relative(actual,expected));
    }
    if(!std::isfinite(defect)||defect>1e-8)throw std::runtime_error("P2 full D6h preconditioner check failed");
    std::ofstream setup(directory+"/setup.json");setup<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka
        <<",\"aspect\":"<<aspect<<",\"ri_imag\":"<<imag<<",\"duffy\":"<<duffy<<",\"dofs\":"<<n<<",\"constraint_rows\":"<<rows.size()
        <<",\"prior_columns\":"<<prior.size()/n<<",\"operator_setup_s\":"<<operator_setup<<",\"local_FFT_preconditioner_s\":"<<precondition_s
        <<",\"P2_images_s\":"<<image_s<<",\"P2_image_actions\":"<<image_actions<<",\"P2_rank\":"<<correction.rank()
        <<",\"P2_bytes\":"<<correction.bytes()<<",\"P2_symmetric_images\":"<<(symmetrize?"true":"false")<<",\"P2_D6h_defect\":"<<defect
        <<",\"total_setup_s\":"<<seconds(begin)<<",\"constraint_backend\":\"original_strict_FFT_FMM\",\"dense_original_A_assembled\":false}\n";
    setup.close();if(!setup)throw std::runtime_error("setup report");
    fprintf(reply.get(),"READY\n");fflush(reply.get());
    std::map<std::string,std::unique_ptr<bem_streaming::SectorRecycleSpace>> recycles;
    const std::vector<std::string> variants={"P2_zero","P2_zero_repeat","P2_guess","P2_mbs025","P2_mbs1","P2_exact_far"};
    for(const auto& name:variants)recycles[name].reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));
    std::string line;
    while(std::getline(std::cin,line)){
        if(line.empty())continue;
        std::istringstream command(line);std::string kind,extra;command>>kind;
        if(kind=="QUIT"){if(command>>extra)throw std::runtime_error("extra QUIT arguments");break;}
        if(kind=="SOLVE"){
            std::string name,rhs_path,seed_path,prefix;int id;double tol;
            if(!(command>>name>>id>>rhs_path>>seed_path>>prefix>>tol)||command>>extra||!recycles.count(name)||!std::isfinite(tol)||tol<=0||tol>1e-3)
                throw std::runtime_error("invalid solve request");
            const auto rhs=load<Complex>(rhs_path,n);auto x=load<Complex>(seed_path,n);Vector ax(n);
            for(auto z:x)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite seed");
            const bool guessed=bem_streaming::norm(x.data(),n)>0;actions=strict_actions=0;
            const auto before=Clock::now();
            const auto solved=solve_prism_sectors(action,strict,inverse,sectors,rhs.data(),x.data(),64,tol,1000,guessed,false,recycles[name].get());
            const double solve_s=seconds(before);const auto verify=Clock::now();strict(x.data(),ax.data());
            const double residual=relative(ax,rhs),check_s=seconds(verify);
            if(!std::isfinite(residual)||residual>tol)throw std::runtime_error("original strict residual failed");
            struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
            save(prefix+".currents.bin",x);std::ofstream meta(prefix+".json");meta<<std::setprecision(17)
                <<"{\"variant\":\""<<name<<"\",\"column\":"<<id<<",\"iterations\":"<<solved.iterations<<",\"actions\":"<<actions
                <<",\"strict_actions_including_final\":"<<strict_actions<<",\"strict_residual\":"<<residual<<",\"solve_s\":"<<solve_s
                <<",\"final_check_s\":"<<check_s<<",\"recycle_rank\":"<<recycles[name]->rank()<<",\"initial_guess\":"<<(guessed?"true":"false")
                <<",\"peak_rss_bytes\":"<<size_t(usage.ru_maxrss)*1024<<"}\n";
            meta.close();if(!meta)throw std::runtime_error("solve report");fprintf(reply.get(),"SOLVED %s %d\n",name.c_str(),id);fflush(reply.get());continue;
        }
        int id,batch;std::string input,output;
        if(!(command>>id>>batch>>input>>output)||command>>extra||batch<1||batch>64||
           (kind!="A"&&kind!="AT"&&kind!="AH"&&kind!="C"&&kind!="CH"&&kind!="GUESS"))throw std::runtime_error("invalid action request");
        const size_t in_rows=kind=="CH"?rows.size():size_t(n),out_rows=kind=="C"?rows.size():size_t(n);
        const auto x=load<Complex>(input,in_rows*batch);Vector y(out_rows*batch);
        for(auto z:x)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite action input");
        actions=strict_actions=adjoint_actions=0;const auto before=Clock::now();
        if(kind=="C"||kind=="CH")constraints.apply(x.data(),batch,y.data(),kind=="CH");
        else for(int b=0;b<batch;++b){const auto* src=x.data()+size_t(b)*n;auto* dst=y.data()+size_t(b)*n;
            if(kind=="A")strict(src,dst);else if(kind=="AT")op.matvec_transpose_strict(src,dst);
            else if(kind=="AH")adjoint(src,dst);else correction.guess(src,dst);}
        const double elapsed=seconds(before);save(output,y);
        fprintf(reply.get(),"DONE %d %.17g %zu %zu %zu\n",id,elapsed,actions,strict_actions,adjoint_actions);fflush(reply.get());
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
