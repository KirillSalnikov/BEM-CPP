#include "muller_fmm_cpu.h"
#include "host_thread_affinity.h"
#include "prism_symmetry_sectors.h"
#include <future>
#include <fstream>
#include <iomanip>
#include <random>

using C=std::complex<double>;using V=std::vector<C>;using Clock=std::chrono::steady_clock;
static double error(const V& a,const V& b){double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::norm(a[i]-b[i]);d+=std::norm(b[i]);}
    double e=std::sqrt(n/std::max(1.,d));if(!std::isfinite(e))throw std::runtime_error("nonfinite action");return e;}
int main(int argc,char** argv)try {
    if(argc!=4)throw std::invalid_argument("usage: REF KA REPORT");
    setvbuf(stdout,nullptr,_IOLBF,0);
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),aspect=10./7;
    if(ref<1||ref>5||ka<=0)throw std::invalid_argument("ref/ka");
    const int threads=std::getenv("BEM_CPU_THREADS")?std::atoi(std::getenv("BEM_CPU_THREADS")):8;
    const size_t budget=size_t(std::getenv("BEM_CPU_PHASE_MB")?std::atoi(std::getenv("BEM_CPU_PHASE_MB")):64)*1024*1024;
    const bool batch_faces=std::getenv("BEM_CPU_FACE_FFT_BATCH") && std::atoi(std::getenv("BEM_CPU_FACE_FFT_BATCH"))!=0;
    const int face_tile=std::getenv("BEM_CPU_FACE_FFT_TILE")?std::atoi(std::getenv("BEM_CPU_FACE_FFT_TILE")):32;
    bem_cpu::configure_threads(4,"BEM_CPU_GPU_AFFINITY");
    setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);setenv("BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1",1);
    setenv("BEM_FACE_FFT_STRICT_ONLY","1",1);setenv("BEM_FACE_FFT_TARGET_FACES","1",1);setenv("BEM_FACE_FFT_REUSE_CROSS","1",1);
    const Mesh mesh=regular_prism(6,aspect,ref,1.,0,false,true);const int points=7*int(mesh.tris.size()/3);
    const int depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);
    setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);
    setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    MullerFmmOperator op;struct Cleanup{MullerFmmOperator& o;~Cleanup(){o.cleanup();}} cleanup{op};
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(mesh,C(ka),C(1.3116),options,7,12,5,leaf(depth));op.set_fmm_near_fp32(true);
    V x(op.system_dofs),reference(x.size()),gpu_face(x.size()),actual(x.size()),other(x.size()),transformed(x.size()),expected(x.size());
    std::mt19937 rng(723);std::normal_distribution<double> normal;for(auto& q:x)q=C(normal(rng),normal(rng));
    op.matvec_strict(x.data(),reference.data());
    bem_cpu::configure_threads(threads,"BEM_CPU_AFFINITY");auto begin=Clock::now();
    std::unique_ptr<bem_cpu::MullerAction> cpu(new bem_cpu::MullerAction(op,budget));
    const double classic_setup=bem_cpu::elapsed(begin);const size_t classic_bytes=cpu->workspace_bytes();
    cpu->apply(x.data(),actual.data());const double classic_error=error(actual,reference);
    if(classic_error>2e-9)throw std::runtime_error("original CPU FMM regression");
    std::vector<double> classic_seconds,face_seconds,serial_seconds,overlap_seconds;
    for(int i=0;i<3;++i){begin=Clock::now();cpu->apply(x.data(),actual.data());classic_seconds.push_back(bem_cpu::elapsed(begin));}
    cpu.reset();bem_cpu::configure_threads(4,"BEM_CPU_GPU_AFFINITY");op.enable_prism_face_fft(ref,aspect);
    op.matvec_strict(x.data(),gpu_face.data());bem_cpu::configure_threads(threads,"BEM_CPU_AFFINITY");begin=Clock::now();
    std::vector<double> serial_face_seconds;std::array<double,6> serial_fmm_times{};size_t serial_face_bytes=0;double serial_face_setup=0;
    if(batch_faces){
        cpu.reset(new bem_cpu::MullerAction(op,budget,true,true));serial_face_setup=bem_cpu::elapsed(begin);
        serial_face_bytes=cpu->workspace_bytes();cpu->apply(x.data(),actual.data());
        if(error(actual,gpu_face)>2e-9)throw std::runtime_error("serial face regression before batch");
        for(int j=0;j<3;++j){begin=Clock::now();cpu->apply(x.data(),actual.data());serial_face_seconds.push_back(bem_cpu::elapsed(begin));}
        serial_fmm_times=cpu->fmm_times;cpu.reset();begin=Clock::now();
    }
    cpu.reset(new bem_cpu::MullerAction(op,budget,true,true,batch_faces,face_tile));const double face_setup=bem_cpu::elapsed(begin);
    cpu->apply(x.data(),actual.data());double face_error=error(actual,gpu_face),mlfma_error=error(actual,reference);
    if(face_error>2e-9||mlfma_error>2e-5)throw std::runtime_error("full CPU face FFT action mismatch");
    double self_s=cpu->face_fft_s,cross_s=cpu->face_cross_s,pack_s=cpu->face_packing_s;
    const double phase_s=cpu->face_phase_s;const size_t phase_evaluations=cpu->face_phase_evaluations;const auto batch_fmm_times=cpu->fmm_times;
    bem_tmatrix::D6hOrbit group;group.build(op.mesh);double symmetry_error=0;
    for(bool mirror:{false,true}){group.transform(x,transformed,mirror,false);cpu->apply(transformed.data(),other.data());
        group.transform(actual,expected,mirror,true);symmetry_error=std::max(symmetry_error,error(other,expected));}
    std::vector<int> target,sign;bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,target,sign);
    for(size_t i=0;i<x.size();++i){transformed[target[i]]=double(sign[i])*x[i];expected[target[i]]=-double(sign[i])*actual[i];}
    cpu->apply(transformed.data(),other.data());symmetry_error=std::max(symmetry_error,error(other,expected));
    if(symmetry_error>2e-5)throw std::runtime_error("CPU face FFT D6h failure");
    double concurrent_error=0;
    for(int repeat=0;repeat<3;++repeat){
        bem_cpu::configure_threads(threads,"BEM_CPU_AFFINITY");begin=Clock::now();cpu->apply(x.data(),actual.data());face_seconds.push_back(bem_cpu::elapsed(begin));
        bem_cpu::configure_threads(4,"BEM_CPU_GPU_AFFINITY");begin=Clock::now();op.matvec_strict(x.data(),gpu_face.data());
        for(int i=0;i<4;++i)op.matvec(x.data(),other.data());serial_seconds.push_back(bem_cpu::elapsed(begin));
        begin=Clock::now();auto task=std::async(std::launch::async,[&]{bem_cpu::configure_threads(threads,"BEM_CPU_AFFINITY");cpu->apply(x.data(),expected.data());});
        for(int i=0;i<4;++i)op.matvec(x.data(),other.data());task.get();overlap_seconds.push_back(bem_cpu::elapsed(begin));
        concurrent_error=std::max(concurrent_error,error(expected,actual));
    }
    bem_cpu::configure_threads(threads,"BEM_CPU_AFFINITY");other=x;cpu->apply(other.data(),other.data());const double inplace_error=error(other,actual);
    for(auto& q:x)q=C(normal(rng),normal(rng));bem_cpu::configure_threads(4,"BEM_CPU_GPU_AFFINITY");op.matvec_strict(x.data(),gpu_face.data());
    bem_cpu::configure_threads(threads,"BEM_CPU_AFFINITY");cpu->apply(x.data(),actual.data());const double second_error=error(actual,gpu_face);
    if(second_error>2e-9||concurrent_error>2e-12||inplace_error>2e-12)throw std::runtime_error("CPU face FFT buffer/overlap failure");
    std::ofstream out(argv[3]);out<<std::setprecision(17)<<"{\"passed\":true,\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"dofs\":"<<op.system_dofs
        <<",\"threads\":"<<threads<<",\"classic_CPU_setup_s\":"<<classic_setup<<",\"face_CPU_setup_s\":"<<face_setup
        <<",\"classic_CPU_workspace_bytes\":"<<classic_bytes<<",\"face_CPU_workspace_bytes\":"<<cpu->workspace_bytes()
        <<",\"classic_relative_error\":"<<classic_error<<",\"face_relative_error\":"<<face_error<<",\"MLFMA_relative_error\":"<<mlfma_error
        <<",\"D6h_defect\":"<<symmetry_error<<",\"second_vector_error\":"<<second_error<<",\"concurrent_error\":"<<concurrent_error
        <<",\"inplace_error\":"<<inplace_error<<",\"self_FFT_s\":"<<self_s<<",\"cross_FMM_s\":"<<cross_s<<",\"packing_s\":"<<pack_s;
    out<<",\"batch_faces\":"<<(batch_faces?"true":"false")<<",\"face_tile\":"<<face_tile<<",\"phase_s\":"<<phase_s
       <<",\"phase_evaluations\":"<<phase_evaluations<<",\"serial_face_CPU_setup_s\":"<<serial_face_setup<<",\"serial_face_CPU_workspace_bytes\":"<<serial_face_bytes;
    auto array=[&](const char* key,const std::vector<double>& v){out<<",\""<<key<<"\":[";for(size_t i=0;i<v.size();++i)out<<(i?",":"")<<v[i];out<<"]";};
    array("classic_CPU_s",classic_seconds);array("face_CPU_s",face_seconds);array("serial_GPU_s",serial_seconds);array("overlap_s",overlap_seconds);
    array("serial_face_CPU_s",serial_face_seconds);
    array("serial_FMM_stage_s",std::vector<double>(serial_fmm_times.begin(),serial_fmm_times.end()));
    array("batch_FMM_stage_s",std::vector<double>(batch_fmm_times.begin(),batch_fmm_times.end()));out<<"}\n";
    std::printf("PASS full CPU FFT, strict defect %.3e, D6h %.3e\n",face_error,symmetry_error);
}catch(const std::exception& e){std::fprintf(stderr,"CPU face FFT: %s\n",e.what());return 1;}
