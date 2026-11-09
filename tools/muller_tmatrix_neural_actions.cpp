// Experimental multi-RHS training/validation bridge. No dense surface matrix.
// Neural inference is outside the Krylov loop; every accepted solve is checked
// by the original FP64 operator. The production T worker consumes the same prior.
#include "muller_fmm.h"
#include "coarse_image_experiment.h"
#include "physical_ritz_prior.h"
#include "prism_d6h_model_action.h"
#include "prism_orbit_inverse.h"
#include "prism_symmetric_circulant.h"
#include "ritz_sector_solver.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unistd.h>
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
static Vector load(const std::string& path,size_t n) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(n*sizeof(Complex)))throw std::runtime_error("input size: "+path);
    Vector x(n);f.seekg(0);f.read(reinterpret_cast<char*>(x.data()),n*sizeof(Complex));
    for(auto z:x)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite input");
    return x;
}
static void save(const std::string& path,const Vector& x) {
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(Complex));
    f.close();if(!f)throw std::runtime_error("output: "+path);
}
static double relative(const Vector& x,const Vector& y) {
    long double a=0,b=0;for(size_t i=0;i<x.size();++i){a+=std::norm(x[i]-y[i]);b+=std::norm(y[i]);}
    return std::sqrt(double(a/std::max(b,1e-300L)));
}
int main(int argc,char** argv)try {
    if(argc!=8)throw std::invalid_argument("REF KA RI_REAL RI_IMAG ASPECT DUFFY DIRECTORY");
    const int ref=std::stoi(argv[1]),duffy=std::stoi(argv[6]);
    const double ka=std::stod(argv[2]),re=std::stod(argv[3]),im=std::stod(argv[4]),aspect=std::stod(argv[5]);
    const std::string dir=argv[7];
    bem_ritz::Parameters parameters;parameters.dofs=1;parameters.columns=1;parameters.ref=ref;
    parameters.ka=ka;parameters.ri_real=re;parameters.ri_imag=im;parameters.aspect=aspect;parameters.duffy=duffy;
    bem_ritz::validate(parameters);
    const int descriptor=dup(STDOUT_FILENO);if(descriptor<0)throw std::runtime_error("protocol dup");
    FILE* reply=fdopen(descriptor,"w");if(!reply)throw std::runtime_error("protocol fdopen");
    fflush(stdout);if(dup2(STDERR_FILENO,STDOUT_FILENO)<0)throw std::runtime_error("redirect");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int seg=1<<ref;int nz=std::max(seg,int(std::ceil(std::sqrt(3.)*aspect*seg)));nz+=nz%2;
    const int points=7*12*seg*(seg+nz),depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
    setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
    setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    for(const char* key:{"BEM_MULLER_GPU_ASSEMBLY","BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","BEM_FMM_MULTILEVEL"})setenv(key,"1",1);
    setenv("BEM_FMM_PAIR_CURRENTS","0",1);
    const auto begin=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;struct Cleanup{MullerFmmOperator& op;~Cleanup(){op.cleanup();}} cleanup{op};
    op.init(regular_prism(6,aspect,ref,1.,0,false,true),Complex(ka),Complex(re,im),options,
        7,duffy,5,leaf(depth),false,2,0,.96,(dir+"/operator.near").c_str(),3);
    op.enable_prism_face_fft(ref,aspect);op.set_fmm_near_fp32(true);
    const int n=op.system_dofs;parameters.dofs=n;parameters.columns=0;
    size_t mixed=0,strict_count=0;
    const bem_streaming::Action action=[&](const Complex* x,Complex* y){++mixed;op.matvec(x,y);};
    const bem_streaming::Action strict=[&](const Complex* x,Complex* y){++strict_count;op.matvec_strict(x,y);};
    D6hOrbit group;group.build(op.mesh);PrismGroupMaps maps;maps.build(group);
    PrismSymmetrySectors sectors;sectors.build(maps);PrismOrbitInverse local;local.build(op,maps,50);
    PrismCirculantInverse circ;circ.build(op.mesh,ref,action,[&](const Complex* x,Complex* y){local.apply(x,y);},0.,0.,3);
    PrismSymmetricCirculant symmetric;symmetric.build(group,circ,local);
    if(!symmetric.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("D6h FFT unavailable");
    const bem_streaming::Action base=[&](const Complex* x,Complex* y){symmetric.apply_full_fft_symmetry(x,y,.5);};
    const auto defect=[&](const bem_streaming::Action& inverse){
        Vector x(n),y(n),rotated(n),expected(n),actual(n);for(int i=0;i<n;++i)x[i]=Complex(std::sin(.13*i),std::cos(.17*i));
        inverse(x.data(),y.data());double worst=0;std::vector<int> vertical,sign;
        D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
        for(int g=0;g<3;++g){
            if(g<2){group.transform(x,rotated,g==1,true);group.transform(y,expected,g==1,false);}
            else for(int i=0;i<n;++i){rotated[vertical[i]]=-double(sign[i])*x[i];expected[vertical[i]]=double(sign[i])*y[i];}
            inverse(rotated.data(),actual.data());worst=std::max(worst,relative(actual,expected));
        }
        std::cerr<<"[neural D6h audit] "<<std::setprecision(17)<<worst<<'\n';
        if(!std::isfinite(worst)||worst>1e-8)throw std::runtime_error("rotation/mirror inverse audit failed");return worst;
    };
    const double base_defect=defect(base);
    // Export the actual mesh representations, never infer DOF permutations in
    // the trainer. Reflections have the opposite sign for equation vectors.
    {
        std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
        std::ofstream file(dir+"/symmetry_maps.bin",std::ios::binary);
        for(int g=0;g<3;++g)for(int part=0;part<2;++part)for(int i=0;i<n;++i){
            const int32_t value=g==2?(part?sign[i]:vertical[i]):
                (part?int(maps.sign[g?6:1][i]):maps.image[g?6:1][i]);
            file.write(reinterpret_cast<const char*>(&value),sizeof(value));
        }
        file.close();if(!file)throw std::runtime_error("symmetry map export");
    }
    std::ofstream setup(dir+"/setup.json");setup<<std::setprecision(17)<<"{\"dofs\":"<<n<<",\"constraint_rows\":1,\"ref\":"<<ref
        <<",\"ka\":"<<ka<<",\"ri_real\":"<<re<<",\"ri_imag\":"<<im<<",\"aspect\":"<<aspect<<",\"quad\":7,\"duffy\":"<<duffy
        <<",\"geometry\":\"regular_prism_D6h\",\"basis\":\"HDivBdm1\",\"total_setup_s\":"<<seconds(begin)
        <<",\"base_D6h_defect\":"<<base_defect<<",\"prior_image_model\":\"d6h_strict\",\"dense_surface_matrix\":false}\n";setup.close();
    if(!setup)throw std::runtime_error("setup write");
    fprintf(reply,"READY\n");fflush(reply);
    std::map<std::string,std::unique_ptr<CoarseImageExperiment>> corrections;
    std::map<std::string,std::unique_ptr<bem_streaming::SectorRecycleSpace>> recycles;
    std::string line;
    while(std::getline(std::cin,line)){
        std::istringstream command(line);std::string kind,extra;command>>kind;
        if(kind=="QUIT"){if(command>>extra)throw std::runtime_error("extra QUIT arguments");break;}
        if(kind=="PILOT"){
            int id,steps,rank;std::string input,prefix;
            if(!(command>>id>>steps>>rank>>input>>prefix)||command>>extra||id<0||steps<4||steps>32||rank<1||rank>8)
                throw std::runtime_error("invalid PILOT");
            const auto rhs=load(input,n);Vector x(n),ax(n),residual(n),coordinates(n),physical_rhs(n),current(n),columns;
            // A local, cold recycle space: training must not warm any of the
            // comparison solvers or use validation/test right-hand sides.
            bem_streaming::SectorRecycleSpace pilot(sectors.offsets,rank);
            mixed=strict_count=0;const auto before=Clock::now();
            const auto result=solve_prism_sectors(action,strict,base,sectors,rhs.data(),x.data(),steps,1e-10,steps,false,false,&pilot);
            strict(x.data(),ax.data());const double residual_norm=relative(ax,rhs);
            for(int i=0;i<n;++i)residual[i]=rhs[i]-ax[i];
            for(int s=0;s<12;++s)for(int j=0;j<pilot.sector(s).rank();++j){
                std::fill(coordinates.begin(),coordinates.end(),Complex(0));
                const auto& u=pilot.sector(s).solution_column(j);
                std::copy(u.begin(),u.end(),coordinates.begin()+sectors.offsets[s]);
                sectors.decode(coordinates.data(),physical_rhs.data(),true);base(physical_rhs.data(),current.data());
                const double length=bem_streaming::norm(current.data(),n);
                if(!std::isfinite(length)||length<=0)throw std::runtime_error("invalid harmonic Ritz pilot");
                for(auto& z:current)z/=length;
                columns.insert(columns.end(),current.begin(),current.end());
            }
            if(columns.empty())throw std::runtime_error("empty harmonic Ritz pilot");
            save(prefix+".currents.bin",columns);save(prefix+".residual.bin",residual);
            save(prefix+".solution.bin",x);
            // This is a truncated training probe, NOT an accepted solution.
            fprintf(reply,"PILOTED %d %zu %d %.17g %.17g %zu %zu\n",id,columns.size()/n,result.iterations,
                residual_norm,seconds(before),mixed,strict_count);fflush(reply);continue;
        }
        if(kind=="RESET"){
            std::string name;if(!(command>>name)||command>>extra||(!corrections.count(name)&&name!="base"))throw std::runtime_error("invalid RESET");
            recycles.erase(name);fprintf(reply,"RESET %s\n",name.c_str());fflush(reply);continue;
        }
        if(kind=="INVERSE"){
            std::string name,input,output;int id,batch;
            if(!(command>>name>>id>>batch>>input>>output)||command>>extra||!corrections.count(name)||batch<1||batch>64)throw std::runtime_error("invalid INVERSE");
            const auto x=load(input,size_t(n)*batch);Vector y(x.size());const auto before=Clock::now();
            for(int j=0;j<batch;++j)corrections[name]->apply(x.data()+size_t(n)*j,y.data()+size_t(n)*j);
            save(output,y);fprintf(reply,"INVERSE %s %d %.17g\n",name.c_str(),id,seconds(before));fflush(reply);continue;
        }
        if(kind=="LOAD"){
            std::string name,path;if(!(command>>name>>path)||command>>extra||name=="base"||corrections.count(name))throw std::runtime_error("invalid LOAD");
            const auto before=Clock::now();bem_ritz::Reader reader(path,parameters);
            std::unique_ptr<CoarseImageExperiment> correction(new CoarseImageExperiment(op.mesh,sectors,base,1e-7));
            Vector u(n),z(n);strict_count=0;
            PrismD6hModelAction model(op.mesh,maps,strict);
            for(size_t j=0;j<reader.parameters().columns;++j){reader.read(j,u);model.apply(u.data(),z.data());correction->add_d6h(u,z);}
            if(!correction->rank())throw std::runtime_error("empty coarse space");
            const double error=defect([&](const Complex* x,Complex* y){correction->apply(x,y);});
            fprintf(reply,"LOADED %s %d %.17g %.17g %zu\n",name.c_str(),correction->rank(),seconds(before),error,strict_count);fflush(reply);
            corrections[name]=std::move(correction);continue;
        }
        if(kind=="SOLVE"){
            std::string name,rhs_path,output;int id;double tolerance;
            if(!(command>>name>>id>>rhs_path>>output>>tolerance)||command>>extra||
                (!corrections.count(name)&&name!="base")||!std::isfinite(tolerance)||tolerance<=0||tolerance>1e-3)throw std::runtime_error("invalid SOLVE");
            const auto rhs=load(rhs_path,n);Vector x(n),ax(n);mixed=strict_count=0;
            if(!recycles.count(name))recycles[name].reset(new bem_streaming::SectorRecycleSpace(sectors.offsets,16));
            bem_streaming::Action inverse=base,encoded;
            if(name!="base"){
                inverse=[&](const Complex* a,Complex* b){corrections[name]->apply(a,b);};
                encoded=[&](const Complex* a,Complex* b){corrections[name]->apply_encoded(a,b);};
            }
            const auto before=Clock::now();const auto result=solve_prism_sectors_with_ritz(action,strict,inverse,encoded,sectors,
                rhs.data(),x.data(),64,tolerance,1000,false,false,recycles[name].get());
            strict(x.data(),ax.data());const double residual=relative(ax,rhs);
            if(!std::isfinite(residual)||residual>tolerance)throw std::runtime_error("original FP64 solve gate failed");
            save(output,x);fprintf(reply,"SOLVED %s %d %d %.17g %.17g %zu %zu\n",name.c_str(),id,result.iterations,seconds(before),residual,mixed,strict_count);fflush(reply);continue;
        }
        int id,batch;std::string input,output;
        if(!(command>>id>>batch>>input>>output)||command>>extra||(kind!="A"&&kind!="GUESS")||batch<1||batch>64)throw std::runtime_error("invalid action");
        const auto x=load(input,size_t(n)*batch);Vector y(x.size());const auto before=Clock::now();mixed=strict_count=0;
        for(int j=0;j<batch;++j)(kind=="A"?strict:base)(x.data()+size_t(n)*j,y.data()+size_t(n)*j);
        save(output,y);fprintf(reply,"DONE %d %.17g %zu %zu 0\n",id,seconds(before),mixed,strict_count);fflush(reply);
    }
    fclose(reply);
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
