// Isolated experiment; no production checkpoint or operator changes.
#include "muller_fmm.h"
#include "muller_dense.h"
#include "prism_circulant_inverse.h"
#include "prism_symmetric_circulant.h"
#include "prism_sector_solver.h"
#include "prism_edge_correction.h"
#include "prism_vertical_inverse.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
using C=std::complex<double>;
using V=std::vector<C>;
using Action=bem_streaming::Action;
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
double relative(V a,const V& b){for(size_t i=0;i<a.size();++i)a[i]-=b[i];return bem_streaming::norm(a.data(),a.size())/std::max(bem_streaming::norm(b.data(),b.size()),1e-300);}
int main(int argc,char** argv)try {
    if(argc!=6)throw std::runtime_error("REF KA PACKED_RHS REPORT RINGS_COMMA_SEPARATED");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),tol=2e-6,ri=1.3116,aspect=10./7;
    const std::string selection=std::getenv("BEM_EDGE_METHODS")?std::getenv("BEM_EDGE_METHODS"):"";
    const auto wanted=[&](const std::string& name){return selection.empty()||(","+selection+",").find(","+name+",")!=std::string::npos;};
    const int recycle_rank=std::getenv("BEM_EDGE_RECYCLE_RANK")?std::stoi(std::getenv("BEM_EDGE_RECYCLE_RANK")):0;
    if(recycle_rank<0||recycle_rank>64)throw std::runtime_error("edge recycle rank");
    if(ref<1||ref>5||ka<=0)throw std::runtime_error("probe size range");
    std::ofstream log(argv[4]);if(!log)throw std::runtime_error("report path");log<<std::setprecision(17);
    const auto start=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;const auto mesh=regular_prism(6,aspect,ref,1.,0,false,true);
    op.init(mesh,ka,C(ri,0),options,7,6,5,64,false,2,0,.96,nullptr,3);op.set_fmm_near_fp32(true);
    if(std::getenv("BEM_TMATRIX_FACE_FFT")&&std::stoi(std::getenv("BEM_TMATRIX_FACE_FFT")))op.enable_prism_face_fft(ref,aspect);
    const int n=op.system_dofs;int mixed_calls=0,strict_calls=0;
    const Action action=[&](const C* x,C* y){++mixed_calls;op.matvec(x,y);};
    const Action strict=[&](const C* x,C* y){++strict_calls;op.matvec_strict(x,y);};
    bem_tmatrix::D6hOrbit orbit;orbit.build(op.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(orbit);
    bem_tmatrix::PrismSymmetrySectors sectors;sectors.build(maps);
    PrismOrbitInverse local;local.build(op,maps,50);
    PrismCirculantInverse circ;
    const double opposite=std::getenv("BEM_CIRCULANT_OPPOSITE_WEIGHT")?std::stod(std::getenv("BEM_CIRCULANT_OPPOSITE_WEIGHT")):0;
    circ.build(op.mesh,ref,action,[&](const C* x,C* y){local.apply(x,y);},0,opposite);
    PrismSymmetricCirculant sym;sym.build(orbit,circ,local);
    if(!sym.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("FFT D6h symmetry failed");
    const Action base=[&](const C* x,C* y){sym.apply_full_fft_symmetry(x,y,.5);};
    log<<"{\"stage\":\"setup\",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"ri\":"<<ri<<",\"aspect\":"<<aspect
       <<",\"dofs\":"<<n<<",\"seconds\":"<<elapsed(start)<<",\"tolerance\":"<<tol<<",\"opposite_weight\":"<<opposite<<"}\n"<<std::flush;
    std::ifstream input(argv[3],std::ios::binary);if(!input)throw std::runtime_error("RHS missing");
    std::vector<V> rhs;
    while(true){V b(n);input.read(reinterpret_cast<char*>(b.data()),sizeof(C)*n);if(!input.gcount())break;
        if(input.gcount()!=std::streamsize(sizeof(C)*n))throw std::runtime_error("RHS dimension");
        rhs.push_back(std::move(b));}
    if(rhs.empty()||rhs.size()>16)throw std::runtime_error("invalid RHS count");
    const int limit=std::getenv("BEM_EDGE_RHS_LIMIT")?std::stoi(std::getenv("BEM_EDGE_RHS_LIMIT")):rhs.size();
    if(limit<1||limit>int(rhs.size()))throw std::runtime_error("RHS limit");
    rhs.resize(limit);
    V scratch(n);action(rhs[0].data(),scratch.data());strict(rhs[0].data(),scratch.data());
    std::vector<V> reference(rhs.size()),reference_far(rhs.size());
    std::vector<Vec3> directions;for(int t=0;t<=180;t+=3){const double a=t*std::acos(-1.)/180;directions.emplace_back(std::sin(a),0,std::cos(a));}
    const auto run=[&](const std::string& name,const Action& inverse,double setup_s){
        if(name!="baseline"&&name!="baseline_repeat"&&!wanted(name))return;
        bem_streaming::SectorRecycleSpace recycle(sectors.offsets,recycle_rank);
        for(size_t j=0;j<rhs.size();++j){V x(n),ax(n),far;const int before_m=mixed_calls,before_s=strict_calls;const auto t=Clock::now();
            auto result=bem_tmatrix::solve_prism_sectors(action,strict,inverse,sectors,rhs[j].data(),x.data(),64,tol,200,false,false,recycle_rank?&recycle:nullptr);
            strict(x.data(),ax.data());const double residual=relative(ax,rhs[j]),solve_s=elapsed(t);
            muller_nodal_farfield(op.mesh,x.data(),x.data()+n/2,ka,directions,far,13);
            if(reference[j].empty()){reference[j]=x;reference_far[j]=far;}
            const double difference=relative(x,reference[j]),far_error=relative(far,reference_far[j]);
            const bool accepted=std::isfinite(residual)&&std::isfinite(difference)&&std::isfinite(far_error)&&residual<=tol&&difference<=1e-3&&far_error<=1e-3;
            log<<"{\"method\":\""<<name<<"\",\"rhs\":"<<j<<",\"iterations\":"<<result.iterations<<",\"solve_s\":"<<solve_s
               <<",\"extra_setup_s\":"<<setup_s<<",\"strict_residual\":"<<residual<<",\"current_relative_error\":"<<difference
               <<",\"far_field_relative_error\":"<<far_error<<",\"mixed_calls\":"<<mixed_calls-before_m<<",\"strict_calls\":"<<strict_calls-before_s
               <<",\"accepted\":"<<(accepted?"true":"false")<<"}\n"<<std::flush;
            std::cout<<name<<" rhs="<<j<<" iterations="<<result.iterations<<" seconds="<<solve_s<<" residual="<<residual<<std::endl;
            if(!accepted){if(name=="baseline")throw std::runtime_error("baseline solve failed");break;}
        }
    };
    run("baseline",base,0);
    std::vector<std::vector<int>> support(op.mesh.topology_edge_count);
    for(size_t t=0;t<op.mesh.elements.size();++t)for(int e:op.mesh.elements[t].topology_edges)support[e].push_back(t);
    std::stringstream text(argv[5]);std::string part;
    while(std::getline(text,part,',')){const int rings=std::stoi(part);PrismEdgeCorrection edge;
        edge.build(op.mesh,maps,rings,[&](const std::vector<int>& selected){std::vector<int> elements;
            for(int e:selected)elements.insert(elements.end(),support[e].begin(),support[e].end());
            std::sort(elements.begin(),elements.end());elements.erase(std::unique(elements.begin(),elements.end()),elements.end());
            return assemble_muller_nodal_block(op,selected,&elements);});
        log<<"{\"stage\":\"edge_setup\",\"rings\":"<<rings<<",\"seconds\":"<<edge.setup_s<<",\"bytes\":"<<edge.bytes()
           <<",\"patches\":"<<edge.blocks()<<",\"max_block_dofs\":"<<edge.max_block_dofs<<",\"covered_dofs\":"<<edge.covered()<<",\"sharp_edges\":"<<edge.sharp_edges<<"}\n"<<std::flush;
        V correction(n),initial(n),residual(n),image(n);
        for(double blend:{.25,.5,1.}) {
            const Action replace=[&](const C* b,C* x){base(b,x);edge.apply(b,correction.data());
                for(int i=0;i<n;++i)if(edge.covers(i))x[i]=(1-blend)*x[i]+blend*correction[i];};
            run("edge_r"+part+"_blend"+std::to_string(blend),replace,edge.setup_s);
        }
        // Multiplicative defect correction pays one extra original A action.
        // Its cost is counted, so fewer iterations alone cannot look faster.
        const Action multiplicative=[&](const C* b,C* x){base(b,initial.data());action(initial.data(),image.data());
            for(int i=0;i<n;++i)residual[i]=b[i]-image[i];
            edge.apply(residual.data(),correction.data());
            for(int i=0;i<n;++i)x[i]=initial[i]+correction[i];};
        run("edge_r"+part+"_multiplicative",multiplicative,edge.setup_s);
    }
    run("baseline_repeat",base,0);op.cleanup();return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
