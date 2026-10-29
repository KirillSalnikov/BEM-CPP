// Same physical operator / RHS / strict tolerance for FFT-compatible inverses.
// Diagnostic only: does not export or certify a T matrix.
#include "muller_fmm.h"
#include "muller_mbj.h"
#include "streaming_host_gmres.h"
#include "streaming_polynomial_inverse.h"
#include "streaming_idr.h"
#include "prism_circulant_inverse.h"
#include "global_coarse_inverse.h"
#include "prism_wave_basis.h"
#include "streaming_tfqmr.h"
#include "streaming_qmr.h"
#include "prism_orbit_inverse.h"
#include "streaming_sector_gmres.h"
#include "prism_symmetric_circulant.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
using namespace bem_streaming;
using Vector=std::vector<Complex>;
static int integer(const char* key,int fallback,int lo,int hi) {
    const int x=std::getenv(key)?std::stoi(std::getenv(key)):fallback;
    if(x<lo || x>hi)throw std::invalid_argument(key);
    return x;
}
static double real(const char* key,double fallback) {
    const double x=std::getenv(key)?std::stod(std::getenv(key)):fallback;
    if(!std::isfinite(x) || x<=0)throw std::invalid_argument(key);
    return x;
}
static double seconds(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
}
int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=5)throw std::runtime_error("Usage: probe REF KA CACHE_PREFIX OUTPUT_JSONL");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    if(ref<0 || ref>7 || ka<=0)throw std::invalid_argument("geometry");
    const std::string cache=argv[3];
    const double ri=real("BEM_PROBE_RI",1.3116),tol=real("BEM_PROBE_TOL",2e-6);
    const double aspect=real("BEM_PROBE_PRISM_ASPECT",1.);
    const int leaf=integer("BEM_PROBE_MAX_LEAF",29,1,8192);
    const int nodes=integer("BEM_PROBE_MBJ_NODES",12,1,256);
    const int orbit_nodes=integer("BEM_PROBE_ORBIT_NODES",nodes,1,512);
    const int overlap=integer("BEM_PROBE_MBJ_OVERLAP",0,0,nodes-1);
    const int restart=integer("BEM_SOLVER_RESTART",32,2,256);
    const int limit=integer("BEM_SOLVER_LIMIT",400,1,4000);
    const int rhs_count=integer("BEM_SOLVER_RHS_COUNT",2,1,2);
    const int order=integer("BEM_PROBE_PFFT_ORDER",2,2,5);
    const double radius=std::getenv("BEM_PROBE_PFFT_RADIUS")?std::stod(std::getenv("BEM_PROBE_PFFT_RADIUS")):1;
    const double safety=real("BEM_PROBE_PFFT_SAFETY",.96);
    if(!std::isfinite(radius) || radius<0)throw std::invalid_argument("BEM_PROBE_PFFT_RADIUS");
    std::string methods=std::getenv("BEM_SOLVER_METHODS")?std::getenv("BEM_SOLVER_METHODS"):
        "baseline,nested4,nested8,nested12,poly4,poly8,poly12";
    const bool use_fft=methods.find("nested")!=std::string::npos || methods.find("poly")!=std::string::npos || methods=="fft_timing";
    const auto start=std::chrono::steady_clock::now();
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    MullerFmmOperator op;
    const double absorption=std::getenv("BEM_PROBE_IMAG_INDEX")?std::stod(std::getenv("BEM_PROBE_IMAG_INDEX")):0;
    op.init(regular_prism(6,aspect,ref,1.,0,false,true),ka,Complex(ri,absorption),options,7,6,5,leaf,
        use_fft,order,radius,safety,(cache+".near").c_str(),3);
    if(use_fft)op.switch_pfft_to_fmm(5,leaf,true);
    op.set_fmm_near_fp32(true);
    MullerMbjPreconditioner mbj;
    mbj.build_cached(op,nodes,overlap,cache+".mbj"+std::to_string(nodes)+"_"+std::to_string(overlap));
    const double setup=seconds(start);
    const int n=op.system_dofs;
    int actions=0,strict_actions=0,fft_actions=0,local_actions=0;
    const Action mixed=[&](const Complex* a,Complex* b){++actions;op.select_fmm_backend();op.matvec(a,b);};
    const Action strict=[&](const Complex* a,Complex* b){++strict_actions;op.select_fmm_backend();op.matvec_strict(a,b);};
    const Action cheap=[&](const Complex* a,Complex* b){++fft_actions;op.select_pfft_backend();op.matvec(a,b);};
    const Action local=[&](const Complex* a,Complex* b){++local_actions;mbj.apply(a,b);};
    PrismCirculantInverse circulant;
    bem_tmatrix::D6hOrbit symmetry;
    bem_tmatrix::PrismSymmetrySectors sectors;
    PrismOrbitInverse orbit_inverse;
    const bool symmetry_methods=methods.find("symmetry")!=std::string::npos || methods.find("orbit")!=std::string::npos;
    if(symmetry_methods) {
        symmetry.build(op.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(symmetry);
        sectors.build(maps);orbit_inverse.build(op,maps,orbit_nodes);
        std::printf("[orbit MBJ] seed_blocks=%zu setup_s=%.3f storage_MiB=%.3f transform_MiB=%.3f\n",
            orbit_inverse.block_count(),orbit_inverse.setup_seconds,orbit_inverse.storage_bytes()/1048576.,sectors.storage_bytes()/1048576.);
    }
    double circulant_setup=0;
    const bool wave_methods=methods.find("wave")!=std::string::npos;
    if(methods.find("circ")!=std::string::npos || wave_methods) {
        const auto begin=std::chrono::steady_clock::now();
        const double shift=std::getenv("BEM_CIRCULANT_SHIFT")?std::stod(std::getenv("BEM_CIRCULANT_SHIFT")):0;
        circulant.build(op.mesh,ref,mixed,local,shift);circulant_setup=seconds(begin);
    }
    PrismSymmetricCirculant symmetric_circulant;
    if(symmetry_methods && circulant_setup>0)symmetric_circulant.build(symmetry,circulant,orbit_inverse);
    GlobalCoarseInverse wave;
    double wave_setup=0;
    const double wave_blend=real("BEM_PROBE_WAVE_LOCAL_BLEND",.25);
    if(wave_methods) {
        const int rank=integer("BEM_PROBE_WAVE_RANK",128,4,512);
        PrismWaveBasis basis(op.mesh,ka*real("BEM_PROBE_WAVE_K_FACTOR",ri));
        const auto begin=std::chrono::steady_clock::now();
        wave.build(n,rank,[&](int j,Complex* out){basis(j,out);},mixed,
            [&](const Complex* a,Complex* b){circulant.apply(a,b,wave_blend);});
        wave_setup=seconds(begin);
    }
    std::vector<Vector> rhs,reference(rhs_count);
    for(int j=0;j<rhs_count;++j) {
        const double theta=.63+.22*j,phi=.27+.36*j;
        const Vec3 d(std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi),std::cos(theta));
        const Vec3 p(std::cos(theta)*std::cos(phi),std::cos(theta)*std::sin(phi),-std::sin(theta));
        rhs.push_back(muller_nodal_planewave_rhs(op.mesh,ka,p,d,13));
    }
    const char* packet_file=std::getenv("BEM_PROBE_RHS_FILE");
    if(packet_file) {
        if(rhs_count!=2)throw std::invalid_argument("packet needs two phase codes");
        std::ifstream file(packet_file,std::ios::binary);
        for(auto& b:rhs){file.read(reinterpret_cast<char*>(b.data()),n*sizeof(Complex));if(!file)throw std::runtime_error("packet RHS size");}
        if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing packet RHS data");
        if(!symmetry_methods)symmetry.build(op.mesh);
    }
    Vector x(n),work(n),training(n);
    std::mt19937 generator(49231);std::normal_distribution<double> normal;
    for(auto& z:training)z=Complex(normal(generator),normal(generator));
    mixed(rhs[0].data(),work.data());strict(rhs[0].data(),work.data());
    if(use_fft)cheap(rhs[0].data(),work.data());
    std::ofstream report(argv[4]);if(!report)throw std::runtime_error("output");
    report<<std::setprecision(17)<<"{\"diagnostic_only\":true,\"ka\":"<<ka<<",\"ref\":"<<ref<<",\"ri\":"<<ri
        <<",\"prism_aspect\":"<<aspect<<",\"circulant_setup_s\":"<<circulant_setup
        <<",\"rhs_kind\":\""<<(packet_file?"C6_mirror_packet":"plane_wave")<<"\""
        <<",\"circulant_storage_mib\":"<<circulant.storage_bytes()/1048576.
        <<",\"wave_setup_s\":"<<wave_setup<<",\"wave_available_rank\":"<<wave.available_rank()
        <<",\"wave_storage_mib\":"<<wave.storage_bytes()/1048576.
        <<",\"system_dofs\":"<<n<<",\"setup_s\":"<<setup<<",\"tolerance\":"<<tol<<",\"restart\":"<<restart
        <<",\"limit\":"<<limit<<",\"mbj_nodes\":"<<nodes<<",\"mbj_overlap\":"<<overlap
        <<",\"mbj_storage_mib\":"<<mbj.storage_megabytes()<<",\"pfft_order\":"<<order
        <<",\"orbit_setup_s\":"<<orbit_inverse.setup_seconds<<",\"orbit_storage_mib\":"<<orbit_inverse.storage_bytes()/1048576.
        <<",\"orbit_nodes\":"<<orbit_nodes
        <<",\"fast_fft_patch_orbits\":"<<(symmetric_circulant.fast_patch_orbits()?"true":"false")
        <<",\"pfft_radius\":"<<radius<<",\"pfft_safety\":"<<safety<<"}\n";report.flush();
    std::istringstream input(methods);std::string method;
    const std::string original_fast=std::getenv("BEM_FMM_P2P_SHARED_FAST_TRIG")?
        std::getenv("BEM_FMM_P2P_SHARED_FAST_TRIG"):"0";
    while(std::getline(input,method,',')) {
        if(method=="symmetry" || method=="symmetry_circ" || method=="inverse_symmetry_circ") {
            Vector a(n),b(n),c(n),d(n);
            if(method!="symmetry") {
                circulant.apply(training.data(),a.data(),.25);local(training.data(),b.data());
                circulant.apply_fft_part(training.data(),c.data());
                for(const auto& patch:circulant.layout.patches)for(int dof:patch.dof)if(dof>=0)b[dof]*=.75;
                for(int i=0;i<n;++i)b[i]+=.25*c[i]-a[i];
                const double error=norm(b.data(),n)/norm(a.data(),n);
                report<<"{\"method\":\"fft_part_decomposition_check\",\"error\":"<<error<<"}\n";report.flush();
                if(error>1e-12)throw std::runtime_error("FFT part decomposition failed");
                symmetric_circulant.apply(training.data(),a.data());symmetric_circulant.apply(training.data(),b.data(),.25,true);
                for(int i=0;i<n;++i)b[i]-=a[i];
                const double full_error=norm(b.data(),n)/norm(a.data(),n);
                report<<"{\"method\":\"symmetric_fft_full_group_check\",\"error\":"<<full_error<<"}\n";report.flush();
                if(full_error>1e-12)throw std::runtime_error("FFT orbit optimization failed");
            }
            const auto defect=[&](const Action& f,bool inverse,bool reflection) {
                f(training.data(),a.data());symmetry.transform(a,b,reflection,!inverse);
                symmetry.transform(training,c,reflection,inverse);f(c.data(),d.data());
                for(int i=0;i<n;++i)d[i]-=b[i];
                return norm(d.data(),n)/norm(a.data(),n);
            };
            for(bool reflection:{false,true}) {
                const double strict_error=method=="inverse_symmetry_circ"?-1:defect(strict,false,reflection);
                const double mixed_error=method=="inverse_symmetry_circ"?-1:defect(mixed,false,reflection);
                const double mbj_error=defect(local,true,reflection);
                const Action orbit=[&](const Complex* x,Complex* y){orbit_inverse.apply(x,y);};
                const double orbit_error=defect(orbit,true,reflection);
                double circ_error=-1;
                double symmetric_circ_error=-1;
                double side_error=-1,cap_error=-1;
                if(method!="symmetry")circ_error=defect([&](const Complex* x,Complex* y){circulant.apply(x,y,.25);},true,reflection);
                if(method!="symmetry")symmetric_circ_error=defect([&](const Complex* x,Complex* y){symmetric_circulant.apply(x,y);},true,reflection);
                if(method!="symmetry") {
                    side_error=defect([&](const Complex* x,Complex* y){circulant.apply_fft_part(x,y,0);},true,reflection);
                    cap_error=defect([&](const Complex* x,Complex* y){circulant.apply_fft_part(x,y,1);},true,reflection);
                }
                report<<"{\"method\":\"symmetry_check\",\"reflection\":"<<int(reflection)<<",\"strict_operator_defect\":"<<strict_error
                      <<",\"mixed_operator_defect\":"<<mixed_error<<",\"mbj_inverse_defect\":"<<mbj_error
                      <<",\"circulant_inverse_defect\":"<<circ_error<<",\"orbit_inverse_defect\":"<<orbit_error
                      <<",\"symmetric_circulant_defect\":"<<symmetric_circ_error
                      <<",\"fft_side_defect\":"<<side_error<<",\"fft_cap_defect\":"<<cap_error<<"}\n";report.flush();
                if(orbit_error>1e-11)throw std::runtime_error("orbit inverse symmetry failed");
                if(symmetric_circ_error>1e-11)throw std::runtime_error("symmetric circulant inverse symmetry failed");
            }
            continue;
        }
        if(method=="adjoint" || method=="adjoint_circ") {
            Vector y(n),ax(n),aty(n),ahy(n),error(n);
            for(int j=0;j<n;++j)y[j]=Complex(normal(generator),normal(generator));
            const auto bilinear=[&](const Vector& a,const Vector& b){Complex z=0;for(int j=0;j<n;++j)z+=a[j]*b[j];return z;};
            for(bool fp32:{false,true}) {
                op.set_fmm_near_fp32(fp32);
                auto begin=std::chrono::steady_clock::now();op.matvec(training.data(),ax.data());const double forward_s=seconds(begin);
                begin=std::chrono::steady_clock::now();op.matvec_transpose(y.data(),aty.data());const double transpose_s=seconds(begin);
                begin=std::chrono::steady_clock::now();op.matvec_adjoint(y.data(),ahy.data());const double adjoint_s=seconds(begin);
                const double scale=norm(ax.data(),n)*norm(y.data(),n);
                const double et=std::abs(bilinear(y,ax)-bilinear(aty,training))/scale;
                const double eh=std::abs(dot(y.data(),ax.data(),n)-dot(ahy.data(),training.data(),n))/scale;
                double dense_error=-1;
                if(!fp32 && n<=1024) {
                    Vector unit(n),column(n),expected(n);
                    for(int j=0;j<n;++j){unit[j]=1.;op.matvec(unit.data(),column.data());unit[j]=0;expected[j]=bilinear(y,column);}
                    for(int j=0;j<n;++j)error[j]=aty[j]-expected[j];
                    dense_error=norm(error.data(),n)/norm(expected.data(),n);
                }
                report<<"{\"method\":\"adjoint_check\",\"fp32\":"<<int(fp32)<<",\"transpose_error\":"<<et<<",\"adjoint_error\":"<<eh
                      <<",\"dense_transpose_error\":"<<dense_error<<",\"forward_s\":"<<forward_s<<",\"transpose_s\":"<<transpose_s
                      <<",\"adjoint_s\":"<<adjoint_s<<"}\n";report.flush();
                if(et>(fp32?5e-5:5e-11) || eh>(fp32?5e-5:5e-11) || dense_error>5e-11)throw std::runtime_error("operator transpose identity failed");
            }
            op.set_fmm_near_fp32(true);
            const Action ph=[&](const Complex* a,Complex* b){mbj.apply_adjoint(a,b);};
            for(bool circ_test:{false,true}) {
                if(circ_test && method!="adjoint_circ")continue;
                if(circ_test){circulant.apply(training.data(),ax.data(),.25);circulant.apply_adjoint(y.data(),ahy.data(),ph,.25);}
                else{mbj.apply(training.data(),ax.data());mbj.apply_adjoint(y.data(),ahy.data());}
                const double e=std::abs(dot(y.data(),ax.data(),n)-dot(ahy.data(),training.data(),n))/(norm(y.data(),n)*norm(ax.data(),n));
                report<<"{\"method\":\"preconditioner_adjoint_check\",\"circulant\":"<<int(circ_test)<<",\"error\":"<<e<<"}\n";report.flush();
                if(e>5e-12)throw std::runtime_error("preconditioner adjoint identity failed");
            }
            continue;
        }
        if(method=="p2p_tune") {
            const std::vector<std::string> flags={"BEM_FMM_P2P_THREADS","BEM_FMM_P2P_LEAF_SPLIT",
                "BEM_FMM_PROFILE_FAR_DETAIL","BEM_FMM_PROFILE_BATCH3","BEM_MULLER_PROFILE_HOST"};
            std::vector<std::string> previous;std::vector<bool> present;
            for(const auto& key:flags) {const char* value=std::getenv(key.c_str());
                present.push_back(value!=nullptr);previous.push_back(value?value:"");}
            for(size_t i=2;i<flags.size();++i)setenv(flags[i].c_str(),"0",1);
            mixed(rhs[0].data(),work.data());const Vector reference=work;
            strict(rhs[0].data(),work.data());const Vector strict_reference=work;
            const double denominator=norm(strict_reference.data(),n);
            Vector delta(n);
            // Reverse the second sweep to expose thermal/order effects.
            std::vector<std::pair<int,int>> configurations;
            for(int threads:{64,128,256})for(int split:{8,16,32,64})configurations.emplace_back(threads,split);
            for(int pass=0;pass<2;++pass) {
                if(pass)std::reverse(configurations.begin(),configurations.end());
                for(const auto& configuration:configurations) {
                    setenv(flags[0].c_str(),std::to_string(configuration.first).c_str(),1);
                    setenv(flags[1].c_str(),std::to_string(configuration.second).c_str(),1);
                    const auto begin=std::chrono::steady_clock::now();mixed(rhs[0].data(),work.data());
                    const double elapsed=seconds(begin);
                    for(int i=0;i<n;++i)delta[i]=work[i]-reference[i];
                    const double relative=norm(delta.data(),n)/denominator;
                    for(int i=0;i<n;++i)delta[i]=work[i]-strict_reference[i];
                    report<<"{\"method\":\"p2p_tune\",\"pass\":"<<pass<<",\"threads\":"<<configuration.first
                          <<",\"split\":"<<configuration.second<<",\"seconds\":"<<elapsed
                          <<",\"relative_to_initial\":"<<relative<<",\"relative_to_strict\":"<<norm(delta.data(),n)/denominator<<"}\n";
                    report.flush();
                }
            }
            for(size_t i=0;i<flags.size();++i) {
                if(present[i])setenv(flags[i].c_str(),previous[i].c_str(),1);else unsetenv(flags[i].c_str());
            }
            continue;
        }
        if(method=="profile") {
            for(int pass=0;pass<4;++pass) {
                const bool full=pass>=2;const auto begin=std::chrono::steady_clock::now();
                (full?strict:mixed)(rhs[0].data(),work.data());
                report<<"{\"method\":\"profile_"<<(full?"strict":"mixed")<<"\",\"pass\":"<<pass
                      <<",\"seconds\":"<<seconds(begin)<<"}\n";report.flush();
            }
            continue;
        }
        if(method=="fft_timing") {
            const auto begin=std::chrono::steady_clock::now();
            cheap(rhs[0].data(),work.data());
            report<<"{\"method\":\"fft_timing\",\"seconds\":"<<seconds(begin)
                  <<",\"action_norm\":"<<norm(work.data(),n)<<"}\n";report.flush();
            if(const char* prefix=std::getenv("BEM_PROBE_FARFIELD_PREFIX")) {
                std::ofstream file(std::string(prefix)+".fft_action.bin",std::ios::binary);
                file.write(reinterpret_cast<const char*>(work.data()),work.size()*sizeof(Complex));
                if(!file)throw std::runtime_error("cannot save diagnostic FFT action");
            }
            continue;
        }
        const bool nested=method.find("nested")==0,poly=method.find("poly")==0,idrs=method.find("idr")==0;
        const bool circ=method.find("circ")==0;
        const bool global=method.find("wave")==0;
        const bool tf=method=="tfqmr" || method=="tfqmr_circ";
        const bool qm=method=="qmr" || method=="qmr_circ" || method=="qmr_strict_circ";
        const bool orbit=method=="orbit" || method=="sector_orbit" || method=="orbit_circ" || method=="sector_orbit_circ";
        const bool orbit_fft=method=="orbit_circ" || method=="sector_orbit_circ";
        if(global)wave.set_rank(std::stoi(method.substr(4)));
        if(!nested && !poly && !idrs && !circ && !global && !tf && !qm && !orbit && method!="baseline" && method!="baseline_fast")
            throw std::invalid_argument("unknown method "+method);
        setenv("BEM_FMM_P2P_SHARED_FAST_TRIG",method=="baseline_fast"?"1":original_fast.c_str(),1);
        const int degree=nested?std::stoi(method.substr(6)):poly?std::stoi(method.substr(4)):1;
        ArnoldiInverse inverse(n,degree,cheap,local);
        const auto train_start=std::chrono::steady_clock::now();
        fft_actions=0;local_actions=0;
        if(poly)inverse.train(training.data());
        const double training_s=poly?seconds(train_start):0;
        const int training_actions=fft_actions;
        const Action precondition=[&](const Complex* a,Complex* b) {
            if(nested)inverse.solve(a,b,real("BEM_SOLVER_INNER_TOL",.1));
            else if(orbit_fft)symmetric_circulant.apply(a,b);
            else if(orbit)orbit_inverse.apply(a,b);
            else if(poly)inverse.apply(a,b);
            else if(circ)circulant.apply(a,b,method=="circ"?1.:std::stod(method.substr(4)));
            else if(global)wave.apply(a,b);
            else if(method=="tfqmr_circ" || method=="qmr_circ" || method=="qmr_strict_circ")circulant.apply(a,b,.25);
            else local(a,b);
        };
        Vector first_solution;
        for(int j=0;j<rhs_count;++j) {
            actions=0;strict_actions=0;fft_actions=0;local_actions=0;
            std::printf("[solver probe] method=%s rhs=%d\n",method.c_str(),j);
            QmrResult qmr_result;HostResult result;
            bool warm_accepted=false;
            double warm_attempt_seconds=0,warm_strict_seconds=0;int warm_strict_checks=0;
            if(packet_file && j==1 && !first_solution.empty()) {
                const auto before=std::chrono::steady_clock::now();
                Vector guess;const auto fit=bem_tmatrix::symmetry_guess(symmetry,rhs[0],first_solution,rhs[1],guess);
                if(fit.fitted) {
                    const auto check=std::chrono::steady_clock::now();strict(guess.data(),work.data());
                    for(int i=0;i<n;++i)work[i]-=rhs[1][i];
                    result.residual=norm(work.data(),n)/norm(rhs[1].data(),n);result.strict_checks=1;result.strict_seconds=seconds(check);
                    warm_strict_seconds=result.strict_seconds;warm_strict_checks=1;
                    if(result.residual<=tol){x=guess;warm_accepted=true;result.seconds=seconds(before);}
                }
                warm_attempt_seconds=seconds(before);
            }
            if(warm_accepted) {
                std::printf("[solver probe] accepted symmetry warm start for second packet code\n");
            } else if(method=="sector_orbit" || method=="sector_orbit_circ") {
                Vector encoded(n),coordinates(n),physical_rhs(n),physical_x(n),physical_y(n);
                sectors.encode(rhs[j].data(),encoded.data(),true);
                const auto composite=[&](const Complex* a,Complex* b,bool full) {
                    sectors.decode(a,physical_rhs.data(),true);precondition(physical_rhs.data(),physical_x.data());
                    (full?strict:mixed)(physical_x.data(),physical_y.data());sectors.encode(physical_y.data(),b,true);
                };
                const auto sr=sector_gmres([&](const Complex* a,Complex* b){composite(a,b,false);},
                    [&](const Complex* a,Complex* b){composite(a,b,true);},encoded.data(),coordinates.data(),sectors.offsets,
                    restart,tol,limit,true,integer("BEM_HOST_GMRES_STRICT_INTERVAL",4,1,16));
                result=sr;sectors.decode(coordinates.data(),physical_rhs.data(),true);precondition(physical_rhs.data(),x.data());
                report<<"{\"method\":\"sector_action_counts\",\"rhs\":"<<j<<",\"actions\":[";
                for(int s=0;s<12;++s)report<<(s?",":"")<<sr.sector_iterations[s];
                report<<"]}\n";
            } else if(qm) {
                const Action ah=[&](const Complex* a,Complex* b){if(method=="qmr_strict_circ")op.matvec_adjoint_strict(a,b);else op.matvec_adjoint(a,b);};
                const Action local_h=[&](const Complex* a,Complex* b){mbj.apply_adjoint(a,b);};
                const Action ph=[&](const Complex* a,Complex* b){if(method=="qmr")local_h(a,b);else circulant.apply_adjoint(a,b,local_h,.25);};
                qmr_result=qmr(method=="qmr_strict_circ"?strict:mixed,ah,strict,precondition,ph,rhs[j].data(),x.data(),n,tol,limit,true,
                    integer("BEM_SOLVER_VERIFY_INTERVAL",128,2,1000));result=qmr_result;
            } else result=tf?
                tfqmr(mixed,strict,precondition,rhs[j].data(),x.data(),n,tol,limit,true,
                    integer("BEM_SOLVER_VERIFY_INTERVAL",128,1,1000)):
                idrs?
                idr(mixed,strict,local,rhs[j].data(),x.data(),n,std::stoi(method.substr(3)),tol,limit,true):
                gmres(mixed,strict,precondition,rhs[j].data(),x.data(),n,restart,tol,limit,true,
                      false,nullptr,true,nested);
            if(!warm_accepted){result.seconds+=warm_attempt_seconds;result.strict_seconds+=warm_strict_seconds;result.strict_checks+=warm_strict_checks;}
            if(packet_file && j==0 && result.residual<=tol)first_solution=x;
            if((method=="baseline" || (method=="circ0.25" && reference[j].empty())) && result.residual<=tol)reference[j]=x;
            double difference=-1;
            if(!reference[j].empty()) {
                for(int i=0;i<n;++i)work[i]=x[i]-reference[j][i];
                difference=norm(work.data(),n)/norm(reference[j].data(),n);
            }
            report<<"{\"method\":\""<<method<<"\",\"rhs\":"<<j<<",\"iterations\":"<<result.iterations
                <<",\"seconds\":"<<result.seconds<<",\"strict_residual\":"<<result.residual
                <<",\"converged\":"<<(result.residual<=tol?"true":"false")<<",\"solution_relative_difference\":"<<difference
                <<",\"symmetry_warm_start_accepted\":"<<(warm_accepted?"true":"false")
                <<",\"action_s\":"<<result.action_seconds<<",\"inverse_s\":"<<result.inverse_seconds
                <<",\"orthogonalization_s\":"<<result.orthogonalization_seconds<<",\"strict_s\":"<<result.strict_seconds
                <<",\"fmm_actions\":"<<actions<<",\"strict_actions\":"<<strict_actions<<",\"fft_actions\":"<<fft_actions
                <<",\"local_actions\":"<<local_actions<<",\"training_s\":"<<training_s<<",\"training_fft_actions\":"<<training_actions<<"}\n";
            if(qm)report<<"{\"method\":\"qmr_action_counts\",\"solver\":\""<<method<<"\",\"rhs\":"<<j
                <<",\"forward_actions\":"<<qmr_result.forward_actions<<",\"adjoint_actions\":"<<qmr_result.adjoint_actions
                <<",\"adjoint_s\":"<<qmr_result.adjoint_seconds<<"}\n";
            report.flush();
            if(const char* prefix=std::getenv("BEM_PROBE_FARFIELD_PREFIX")) {
                if(result.residual<=tol) {
                    const double theta=.63+.22*j,phi=.27+.36*j;
                    const Vec3 d(std::sin(theta)*std::cos(phi),std::sin(theta)*std::sin(phi),std::cos(theta));
                    const Vec3 p(std::cos(theta)*std::cos(phi),std::cos(theta)*std::sin(phi),-std::sin(theta));
                    std::vector<Vec3> directions;
                    for(int i=0;i<=720;++i){const double a=std::acos(-1.)*i/720;directions.push_back(d*std::cos(a)+p*std::sin(a));}
                    Vector far;op.farfield(x.data(),directions,far);
                    std::ofstream file(std::string(prefix)+"."+method+".rhs"+std::to_string(j)+".bin",std::ios::binary);
                    file.write(reinterpret_cast<const char*>(far.data()),far.size()*sizeof(Complex));
                    if(!file)throw std::runtime_error("cannot save diagnostic far field");
                }
            }
        }
    }
    op.cleanup();return 0;
} catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
