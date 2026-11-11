// Isolated reduced-quadrature inner solver. The accepted equation is always A7.
#include "muller_fmm.h"
#include "muller_mbj.h"
#include "streaming_host_gmres.h"
#include "streaming_polynomial_inverse.h"
#include "tmatrix_rhs_assembly.h"
#include "quadrature.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <cuda_runtime_api.h>
using namespace bem_streaming;
using Vector=std::vector<Complex>;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
template<class T>void write(const std::string& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);if(!f.write((char*)v.data(),v.size()*sizeof(T)))throw std::runtime_error(p);}
static double rel(const Vector& a,const Vector& b){Vector d=a;for(size_t i=0;i<a.size();++i)d[i]-=b[i];return norm(d.data(),d.size())/std::max(norm(b.data(),b.size()),1e-300);}
struct Point{int element;MullerFrameSample sample;double weight;};
int main(int argc,char** argv)try{
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=7)throw std::runtime_error("probe geometry|solve SHAPE_OR_OBJ REF KA Q OUT");
    const std::string stage=argv[1],shape=argv[2],out=argv[6];
    const int ref=std::stoi(argv[3]),q=std::stoi(argv[5]);const double ka=std::stod(argv[4]);
    if(ref<0||ref>5||ka<=0||(q!=1&&q!=3&&q!=7))throw std::runtime_error("invalid settings");
    if(tri_quadrature(q).npts!=q)throw std::runtime_error("quadrature request silently fell back");
    Mesh linear;
    if(shape=="prism")linear=regular_prism(6,1./(.7*std::cos(M_PI/6)),ref,1.,0,false,true);
    else{
        linear=load_obj(shape.c_str());Vec3 center(0,0,0);
        for(const auto& v:linear.verts)center=center+v;center=center*(1./linear.verts.size());
        for(auto& v:linear.verts)v=v-center;normalize_mesh(linear);
        for(int i=0;i<ref;++i)linear=subdivide_flat(linear);
    }
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    auto mesh=build_muller_p2_mesh(linear,options);const int n=mesh.system_dofs();
    std::vector<Point> points;std::vector<double> positions;const auto rule=tri_quadrature(13);
    for(int t=0;t<int(mesh.elements.size());++t)for(int j=0;j<rule.npts;++j){
        auto f=evaluate_muller_frame(mesh,t,rule.pts[j][0],rule.pts[j][1]);
        points.push_back({t,f,.5*rule.wts[j]*f.jacobian});
        positions.insert(positions.end(),{f.position.x,f.position.y,f.position.z});
    }
    if(stage=="geometry"){
        write(out+"/points.bin",positions);std::ofstream f(out+"/geometry.json");
        f<<std::setprecision(17)<<"{\"ka\":"<<ka<<",\"ref\":"<<ref<<",\"dofs\":"<<n<<",\"triangles\":"<<mesh.elements.size()<<",\"rhs_points\":"<<points.size()<<"}\n";return 0;
    }
    if(stage!="solve")throw std::runtime_error("unknown stage");
    std::ifstream tasks(out+"/tasks.txt");std::vector<std::string> names;std::string name;
    while(std::getline(tasks,name))if(!name.empty())names.push_back(name);
    if(names.empty())throw std::runtime_error("no RHS");
    std::vector<Vector> rhs;
    bem_tmatrix::StreamingRhsAssembler assembler(13);
    for(const auto& name:names){
        std::ifstream f(out+"/"+name+".fields.bin",std::ios::binary|std::ios::ate);
        if(!f||f.tellg()!=std::streamoff(points.size()*6*sizeof(Complex)))throw std::runtime_error("field size");
        f.seekg(0);Vector fields(points.size()*6),b(n);f.read((char*)fields.data(),fields.size()*sizeof(Complex));
        for(size_t begin=0;begin<points.size();begin+=assembler.chunk_points())
            assembler.add(mesh,points,begin,std::min(assembler.chunk_points(),points.size()-begin),fields.data()+6*begin,b);
        rhs.push_back(std::move(b));
    }
    const int leaf=std::getenv("RQ_LEAF")?std::stoi(std::getenv("RQ_LEAF")):64;
    const int maxiter=std::getenv("RQ_MAXITER")?std::stoi(std::getenv("RQ_MAXITER")):300;
    const auto before=Clock::now();MullerFmmOperator exact,cheap;
    struct Cleanup{MullerFmmOperator& a;MullerFmmOperator& b;~Cleanup(){a.cleanup();b.cleanup();}} cleanup{exact,cheap};
    exact.init(linear,ka,Complex(1.3116),options,7,6,5,leaf,false,2,0,.96,(out+"/q7.near").c_str(),3);
    exact.set_fmm_near_fp32(true);const double exact_setup=elapsed(before);
    const auto cheaper=Clock::now();
    // Keep the FMM spatial levels comparable as the point population changes.
    // Unscaled occupancy thresholds can silently change (or invalidate) bands.
    std::vector<std::pair<std::string,std::string>> restore;
    for(const char* key:{"BEM_FMM_BANDED_COARSE_MAX_LEAF","BEM_FMM_BANDED_MIDDLE_MAX_LEAF"})if(const char* value=std::getenv(key)){
        restore.emplace_back(key,value);const std::string scaled=std::to_string(std::max(1,int(std::ceil(std::stod(value)*q/7.))));
        setenv(key,scaled.c_str(),1);
    }
    cheap.init(linear,ka,Complex(1.3116),options,q,6,5,std::max(1,int(std::ceil(leaf*q/7.))),false,2,0,.96,(out+"/q"+std::to_string(q)+".near").c_str(),3);
    for(const auto& setting:restore)setenv(setting.first.c_str(),setting.second.c_str(),1);
    cheap.set_fmm_near_fp32(true);const double cheap_setup=elapsed(cheaper);
    if(cheap.system_dofs!=n||exact.system_dofs!=n||cheap.quadrature.size()!=mesh.elements.size()*q)throw std::runtime_error("spaces differ");
    const auto mbj_start=Clock::now();MullerMbjPreconditioner mbj;mbj.build_cached(exact,50,0,out+"/mbj50");const double mbj_s=elapsed(mbj_start);
    int na=0,nc=0,ns=0;double ta=0,tc=0,ts=0;
    const Action full=[&](const Complex* a,Complex* b){const auto t=Clock::now();++na;exact.matvec(a,b);ta+=elapsed(t);};
    const Action low=[&](const Complex* a,Complex* b){const auto t=Clock::now();++nc;cheap.matvec(a,b);tc+=elapsed(t);};
    const Action strict=[&](const Complex* a,Complex* b){const auto t=Clock::now();++ns;exact.matvec_strict(a,b);ts+=elapsed(t);};
    const Action local=[&](const Complex* a,Complex* b){mbj.apply(a,b);};
    std::ofstream report(out+"/q"+std::to_string(q)+".jsonl");report<<std::setprecision(17);
    report<<"{\"type\":\"setup\",\"dofs\":"<<n<<",\"q\":"<<q<<",\"ka\":"<<ka<<",\"ref\":"<<ref
      <<",\"exact_setup_s\":"<<exact_setup<<",\"cheap_setup_s\":"<<cheap_setup<<",\"mbj_setup_s\":"<<mbj_s
      <<",\"exact_points\":"<<exact.quadrature.size()<<",\"cheap_points\":"<<cheap.quadrature.size()<<"}\n";report.flush();
    Vector a(n),b(n),x(n),image(n),reference;
    full(rhs[0].data(),a.data());low(rhs[0].data(),b.data());strict(rhs[0].data(),a.data());
    // Interleaved warmed timings; both actions on identical input vectors.
    for(int rep=0;rep<4;++rep){const auto t=Clock::now();if(rep%2){low(rhs[0].data(),b.data());full(rhs[0].data(),a.data());}else{full(rhs[0].data(),a.data());low(rhs[0].data(),b.data());}
        report<<"{\"type\":\"action\",\"repeat\":"<<rep<<",\"full_mean_s\":"<<ta/na<<",\"cheap_mean_s\":"<<tc/nc<<",\"relative_action_difference_on_RHS\":"<<rel(b,a)<<",\"pair_s\":"<<elapsed(t)<<"}\n";
    }
    std::vector<Vec3> directions;
    for(int theta=0;theta<=180;++theta)for(double phi:{0.,.37,1.1}){const double t=theta*M_PI/180;directions.emplace_back(std::sin(t)*std::cos(phi),std::sin(t)*std::sin(phi),std::cos(t));}
    std::vector<std::string> variants={"baseline","refine","nested4","nested8","baseline_repeat"};
    if(const char* selected=std::getenv("RQ_VARIANTS")){variants.clear();std::istringstream s(selected);while(std::getline(s,name,','))variants.push_back(name);}
    std::vector<std::unique_ptr<RecycleSpace>> recycle;
    for(const auto& v:variants)recycle.emplace_back(v.find("nested")==0?nullptr:new RecycleSpace(n,8));
    for(size_t r=0;r<rhs.size();++r){Vector ref_far;
        for(size_t v=0;v<variants.size();++v){
            const std::string variant=variants[v];const bool nested=variant.find("nested")==0;
            na=nc=ns=0;ta=tc=ts=0;
            const bool baseline=variant.find("baseline")==0;
            std::unique_ptr<ArnoldiInverse> inner;
            if(nested)inner.reset(new ArnoldiInverse(n,std::stoi(variant.substr(6)),low,local));
            const Action inverse=[&](const Complex* a,Complex* b){if(inner)inner->solve(a,b,.1);else local(a,b);};
            // A low quadrature action cannot bypass the A7 acceptance gate.
            const auto t=Clock::now();
            auto result=gmres((baseline||nested)?full:low,strict,inverse,rhs[r].data(),x.data(),n,64,2e-6,maxiter,false,false,recycle[v].get(),true,nested);
            const double solve_s=elapsed(t);strict(x.data(),image.data());const double residual=rel(image,rhs[r]);
            Vector far;muller_nodal_farfield(mesh,x.data(),x.data()+mesh.current_dofs(),ka,directions,far,13);
            if(variant=="baseline"){reference=x;ref_far=far;}
            if(reference.empty())throw std::runtime_error("baseline must be first");
            write(out+"/"+names[r]+".q"+std::to_string(q)+"."+variant+".far.bin",far);
            report<<"{\"type\":\"solve\",\"rhs\":\""<<names[r]<<"\",\"variant\":\""<<variant<<"\",\"q\":"<<q
              <<",\"solve_s\":"<<solve_s<<",\"iterations\":"<<result.iterations<<",\"strict_residual\":"<<residual
              <<",\"accepted\":"<<(residual<=2e-6?"true":"false")<<",\"current_relative_error\":"<<rel(x,reference)
              <<",\"far_relative_error\":"<<rel(far,ref_far)<<",\"full_actions\":"<<na<<",\"cheap_actions\":"<<nc<<",\"strict_actions\":"<<ns
              <<",\"full_action_s\":"<<ta<<",\"cheap_action_s\":"<<tc<<",\"strict_action_s\":"<<ts
              <<",\"preconditioner_s\":"<<result.inverse_seconds<<",\"orthogonalization_s\":"<<result.orthogonalization_seconds<<"}\n";report.flush();
            std::cout<<names[r]<<" "<<variant<<" q="<<q<<" seconds="<<solve_s<<" residual="<<residual<<" far="<<rel(far,ref_far)<<std::endl;
            if(baseline&&residual>2e-6)throw std::runtime_error("reference did not converge");
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
