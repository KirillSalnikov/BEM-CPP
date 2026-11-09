// Persistent matrix-free BEM worker. A private Unix socket carries two
// incident E/H fields at a time; stdout remains available for solver logs.
#include "muller_fmm.h"
#include "muller_mbj.h"
#include "muller_paired_gmres.h"
#include "quadrature.h"
#include "streaming_host_gmres.h"
#include "tmatrix_symmetry_warm_start.h"
#include "tmatrix_rhs_assembly.h"
#include "prism_circulant_inverse.h"
#include "global_coarse_inverse.h"
#include "prism_wave_basis.h"
#include "streaming_polynomial_inverse.h"
#include "prism_sector_solver.h"
#include "sector_recycle_rebudget.h"
#include "prism_vertical_inverse.h"
#include "prism_symmetric_circulant.h"
#include "muller_fmm_cpu.h"
#include "host_cpu_checker_config.h"
#include "planar_preconditioner_config.h"
#include "planar_prism_kernels.h"
#include "host_hybrid_scheduler.h"
#include "host_thread_affinity.h"
#include "coarse_image_experiment.h"
#include "physical_ritz_prior.h"
#include "physical_ritz_images.h"
#include "ritz_sector_solver.h"
#include "resident_prism_inverse.h"
#include "resident_sector_operator.h"
#include "external_fft_action.h"
#include <cuda_profiler_api.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/resource.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Complex=std::complex<double>;
using Clock=std::chrono::steady_clock;

void transfer(int fd, void* data, size_t bytes, bool sending) {
    char* p=static_cast<char*>(data);
    while (bytes) {
        const ssize_t count=sending ? send(fd,p,bytes,MSG_NOSIGNAL) : recv(fd,p,bytes,0);
        if (count<=0) throw std::runtime_error("T-matrix worker connection closed");
        p+=count; bytes-=static_cast<size_t>(count);
    }
}
void message(int fd, const std::string& value) {
    uint64_t count=value.size();
    transfer(fd,&count,sizeof(count),true);
    transfer(fd,const_cast<char*>(value.data()),value.size(),true);
}
double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now()-begin).count();
}
double relative_residual(MullerFmmOperator& op,const std::vector<Complex>& x,
                         const std::vector<Complex>& b) {
    std::vector<Complex> product(x.size()); op.matvec_strict(x.data(),product.data());
    double numerator=0,denominator=0;
    for(size_t i=0;i<x.size();++i) {
        numerator+=std::norm(product[i]-b[i]); denominator+=std::norm(b[i]);
    }
    return std::sqrt(numerator/std::max(denominator,1e-300));
}

int main(int argc,char** argv) try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc!=12) throw std::runtime_error(
        "Usage: worker socket sphere|prism|tetra|OBJ ref ka ri_re ri_im directions.bin cache_prefix tol restart maxiter");
    const std::string socket_path=argv[1],shape=argv[2],cache=argv[8];
    const int ref=std::stoi(argv[3]),restart=std::stoi(argv[10]),maxiter=std::stoi(argv[11]);
    const double ka=std::stod(argv[4]),tol=std::stod(argv[9]);
    const Complex ri(std::stod(argv[5]),std::stod(argv[6]));
    if(ka<=0||tol<=0||ref<0||restart<2||maxiter<1) throw std::runtime_error("invalid worker parameters");
    const auto cpu_backend=bem_cpu::CpuCheckerConfig::from_environment(shape,ref);
    const auto planar_config=bem_osrc::PreconditionerConfig::from_environment(shape,ref,ka,ri);
    const auto ritz_image_config=bem_ritz::ImageConfig::from_environment();
    const char* gpu_batch_setting=std::getenv("BEM_TMATRIX_GPU_BATCH2");
    if(gpu_batch_setting && std::strcmp(gpu_batch_setting,"0") && std::strcmp(gpu_batch_setting,"1"))
        throw std::runtime_error("BEM_TMATRIX_GPU_BATCH2 must be 0 or 1");
    const bool gpu_batch2=gpu_batch_setting && std::strcmp(gpu_batch_setting,"0")!=0;
    const char* resident_setting=std::getenv("BEM_TMATRIX_RESIDENT_COMPOSITE");
    if(resident_setting && std::strcmp(resident_setting,"0") && std::strcmp(resident_setting,"1"))
        throw std::invalid_argument("BEM_TMATRIX_RESIDENT_COMPOSITE must be0 or1");
    const bool resident_requested=resident_setting && std::strcmp(resident_setting,"0")!=0;
    if(resident_requested){
        const char* prior=std::getenv("BEM_TMATRIX_RITZ_PRIOR");
        if(gpu_batch2 || (prior && *prior))throw std::invalid_argument("resident composite is not combined with GPU batch2 or Ritz yet");
        for(const auto& flag:std::vector<std::pair<const char*,const char*>>{
                {"BEM_MULLER_GPU_ASSEMBLY","1"},{"BEM_MULLER_GPU_HOST_CORRECTION","0"},
                {"BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1"},{"BEM_CIRCULANT_CPU_FFT","0"}}){
            const char* value=std::getenv(flag.first);
            if(!value || std::strcmp(value,flag.second))throw std::invalid_argument(std::string("resident composite requires explicit ")+flag.first+"="+flag.second);
        }
    }
    if(gpu_batch2)setenv("BEM_MULLER_BATCH2_HOST","1",1);
    int recycle_capacity=0;
    if(const char* setting=std::getenv("BEM_TMATRIX_RECYCLE_RANK")) {
        char* end=nullptr;const long value=std::strtol(setting,&end,10);
        if(end==setting || *end || value<0 || value>64)
            throw std::runtime_error("BEM_TMATRIX_RECYCLE_RANK must be an integer in [0,64]");
        recycle_capacity=int(value);
    }
    sockaddr_un address={}; address.sun_family=AF_UNIX;
    if(socket_path.size()>=sizeof(address.sun_path)) throw std::runtime_error("Unix socket path is too long");
    std::strcpy(address.sun_path,socket_path.c_str());
    const int listener=socket(AF_UNIX,SOCK_STREAM,0);
    if(listener<0||bind(listener,reinterpret_cast<sockaddr*>(&address),sizeof(address))||listen(listener,1))
        throw std::runtime_error("cannot create private worker socket");
    const int client=accept(listener,nullptr,nullptr);
    if(client<0) throw std::runtime_error("cannot accept worker connection");
    close(listener);
    Mesh mesh;
    const char* d6h_environment=std::getenv("BEM_TMATRIX_D6H_MESH");
    const bool d6h_mesh=d6h_environment && std::strcmp(d6h_environment,"0")!=0;
    const char* generic_d6h_setting=std::getenv("BEM_TMATRIX_GEOMETRY_D6H");
    const bool generic_d6h=generic_d6h_setting && std::strcmp(generic_d6h_setting,"0")!=0;
    const bool d6h_geometry=(shape=="prism" && d6h_mesh)||generic_d6h;
    double prism_aspect=1.;
    if(const char* value=std::getenv("BEM_TMATRIX_PRISM_ASPECT")) {
        size_t used=0;prism_aspect=std::stod(value,&used);
        if(used!=std::strlen(value) || !std::isfinite(prism_aspect) || prism_aspect<=0)
            throw std::runtime_error("BEM_TMATRIX_PRISM_ASPECT must be finite and positive");
    }
    if(shape=="sphere") mesh=icosphere(1.,ref);
    else if(shape=="prism") mesh=regular_prism(6,prism_aspect,ref,1.,0,false,d6h_mesh);
    else {
        if(shape=="tetra") {
            mesh.verts={Vec3(1.2,0,0),Vec3(-.8,.3,.1),Vec3(.1,1,.2),Vec3(.2,.1,1.5)};
            mesh.tris={0,1,2,0,3,1,0,2,3,1,3,2};
        } else mesh=load_obj(shape.c_str());
        Vec3 center(0,0,0);
        for(const Vec3& v:mesh.verts) center=center+v;
        center=center*(1./mesh.verts.size());
        for(Vec3& v:mesh.verts) v=v-center;
        normalize_mesh(mesh);
        for(int i=0;i<ref;++i) mesh=subdivide_flat(mesh);
    }
    std::ifstream direction_input(argv[7],std::ios::binary|std::ios::ate);
    if(!direction_input||direction_input.tellg()<=0||static_cast<size_t>(direction_input.tellg())%(3*sizeof(double)))
        throw std::runtime_error("invalid far-field directions");
    std::vector<double> direction_values(static_cast<size_t>(direction_input.tellg())/sizeof(double));
    direction_input.seekg(0); direction_input.read(reinterpret_cast<char*>(direction_values.data()),direction_values.size()*sizeof(double));
    std::vector<Vec3> directions(direction_values.size()/3);
    for(size_t i=0;i<directions.size();++i)
        directions[i]=Vec3(direction_values[3*i],direction_values[3*i+1],direction_values[3*i+2]);
    const auto setup_start=Clock::now();
    MullerP2BuildOptions options;
    options.project_edge_nodes_to_sphere=shape=="sphere";
    options.edge_mode=shape=="sphere" ? MullerEdgeMode::Smooth : MullerEdgeMode::HDivBdm1;
    const char* leaf_environment=std::getenv("BEM_TMATRIX_MAX_LEAF");
    const int requested_leaf=leaf_environment?std::max(1,std::atoi(leaf_environment)):64;
    // Keep the established depth-five layout by default. A validated explicit
    // multilevel layout may opt into finer near-field cells; retain at least
    // depth-five angular orders just as the native driver's depth-six mode.
    const int depth5_safe_leaf=(2LL*mesh.nt()*7+32767)/32768;
    const char* allow_depth6=std::getenv("BEM_FMM_ALLOW_DEPTH6");
    const char* multilevel=std::getenv("BEM_FMM_MULTILEVEL");
    const bool finer_multilevel=allow_depth6 && std::strcmp(allow_depth6,"0")!=0 &&
        multilevel && std::strcmp(multilevel,"0")!=0;
    const int max_leaf=finer_multilevel?requested_leaf:std::max(requested_leaf,depth5_safe_leaf);
    if(finer_multilevel && max_leaf<depth5_safe_leaf)
        setenv("BEM_FMM_ORDER_REFERENCE_DEPTH","5",0);
    MullerFmmOperator op;
    int duffy_order=4;
    if(const char* value=std::getenv("BEM_TMATRIX_DUFFY_ORDER")) {
        char* end=nullptr;
        const long parsed=std::strtol(value,&end,10);
        if(end==value || *end!='\0' || parsed<2 || parsed>12)
            throw std::runtime_error("BEM_TMATRIX_DUFFY_ORDER must be an integer in [2,12]");
        duffy_order=static_cast<int>(parsed);
    }
    const int fft_inner=std::getenv("BEM_TMATRIX_FFT_INNER")?std::stoi(std::getenv("BEM_TMATRIX_FFT_INNER")):0;
    if(fft_inner<0 || fft_inner>32)throw std::runtime_error("FFT inner steps must be 0..32");
    const double fft_radius=std::getenv("BEM_TMATRIX_FFT_RADIUS")?std::stod(std::getenv("BEM_TMATRIX_FFT_RADIUS")):1.;
    if(!std::isfinite(fft_radius) || fft_radius<0)throw std::runtime_error("invalid FFT correction radius");
    op.init(mesh,Complex(ka,0),ri,options,7,duffy_order,5,max_leaf,fft_inner>0,2,fft_inner?fft_radius:0,.96,(cache+".near").c_str(),3);
    if(fft_inner)op.switch_pfft_to_fmm(5,max_leaf,true);
    if(const char* face_fft=std::getenv("BEM_TMATRIX_FACE_FFT")) {
        if(std::strcmp(face_fft,"0")!=0) {
            if(shape!="prism"||!d6h_mesh)throw std::runtime_error("face FFT requires the regular D6h prism");
            op.enable_prism_face_fft(ref,prism_aspect);
        }
    }
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    op.set_fmm_near_fp32(true);
#endif
    const double setup_seconds=seconds(setup_start);
    MullerMbjPreconditioner mbj;
    const auto mbj_start=Clock::now();
    const char* mbj_environment=std::getenv("BEM_TMATRIX_MBJ_NODES");
    const int mbj_nodes=mbj_environment?std::max(1,std::atoi(mbj_environment)):(op.system_dofs>=2000000?12:50);
    const int mbj_overlap=std::getenv("BEM_TMATRIX_MBJ_OVERLAP")?std::stoi(std::getenv("BEM_TMATRIX_MBJ_OVERLAP")):0;
    if(mbj_overlap<0 || mbj_overlap>=mbj_nodes)throw std::runtime_error("invalid MBJ overlap");
    const char* sector_setting=std::getenv("BEM_TMATRIX_SECTOR_GMRES");
    const bool sector_solver=sector_setting && std::strcmp(sector_setting,"0")!=0;
    const bool hybrid_requested=std::getenv("BEM_TMATRIX_HYBRID") && std::strcmp(std::getenv("BEM_TMATRIX_HYBRID"),"0")!=0;
    const bool cpu_requested=hybrid_requested && (!std::getenv("BEM_TMATRIX_CPU_STRICT") || std::strcmp(std::getenv("BEM_TMATRIX_CPU_STRICT"),"0")!=0);
    if(hybrid_requested&&!sector_solver)throw std::runtime_error("hybrid T worker requires sector GMRES");
    // Sector Arnoldi lives in host RAM, independently of where the operator
    // projects currents and assembles its result. matvec() handles transfers.
    const bool device_solver=op.device_matvec_available() && !sector_solver;
    const auto sector_integer=[](const char* name,int fallback,int maximum) {
        const char* setting=std::getenv(name);if(!setting)return fallback;
        char* end=nullptr;const long value=std::strtol(setting,&end,10);
        if(end==setting || *end || value<0 || value>maximum)throw std::runtime_error(std::string("invalid ")+name);
        return int(value);
    };
    const int sector_recycle_capacity=sector_integer("BEM_TMATRIX_SECTOR_RECYCLE_RANK",0,64);
    const int gpu_pair_wait=sector_integer("BEM_TMATRIX_BATCH2_WAIT_US",2000,1000000);
    // Zero retains the previous unlimited offload policy. Exhausting the
    // budget routes the same strict action to GPU; no residual gate is skipped.
    const int cpu_check_budget=sector_integer("BEM_TMATRIX_CPU_CHECK_BUDGET",0,65536);
    const bool profile_first_batch=sector_integer("BEM_TMATRIX_PROFILE_FIRST_BATCH",0,1)!=0;
    if(gpu_batch2 && !hybrid_requested)throw std::runtime_error("GPU batching requires hybrid solver lanes");
    const bool merge_lane_recycle=sector_integer("BEM_TMATRIX_MERGE_RECYCLE",0,1)!=0;
    if(merge_lane_recycle && (!hybrid_requested || !sector_recycle_capacity))
        throw std::runtime_error("lane recycle merging requires hybrid packets and sector recycling");
    const unsigned sector_recycle_mask=sector_integer("BEM_TMATRIX_SECTOR_RECYCLE_MASK",4095,4095);
    const int sector_budget_packets=sector_integer("BEM_TMATRIX_SECTOR_BUDGET_PACKETS",0,1024);
    if(sector_budget_packets && (!sector_solver || !sector_recycle_capacity))
        throw std::runtime_error("sector budget learning requires sector recycling");
    const bool vertical_preconditioner=sector_integer("BEM_TMATRIX_VERTICAL_PRECONDITIONER",0,1)!=0;
    double sector_first_pass_factor=0;
    if(const char* value=std::getenv("BEM_SECTOR_FIRST_PASS_FACTOR")) {
        char* end=nullptr;sector_first_pass_factor=std::strtod(value,&end);
        if(end==value || *end || !std::isfinite(sector_first_pass_factor) || sector_first_pass_factor<0 ||
           sector_first_pass_factor>64 || (sector_first_pass_factor && !sector_solver))
            throw std::invalid_argument("sector first pass factor");
    }
    if(sector_recycle_capacity && !sector_solver)throw std::runtime_error("sector recycling requires sector GMRES");
    if(vertical_preconditioner && !sector_solver)throw std::runtime_error("vertical inverse requires sector GMRES");
    const int orbit_nodes=std::getenv("BEM_TMATRIX_ORBIT_NODES")?std::stoi(std::getenv("BEM_TMATRIX_ORBIT_NODES")):50;
    if(sector_solver && (device_solver || !d6h_geometry || fft_inner || recycle_capacity || mbj_overlap || orbit_nodes<1 || orbit_nodes>512))
        throw std::runtime_error("sector GMRES requires a verified D6h mesh, orbit blocks 1..512, and no overlap/inner FFT/recycling");
    bem_tmatrix::D6hOrbit symmetry_orbit;
    bem_tmatrix::PrismSymmetrySectors symmetry_sectors;
    PrismOrbitInverse orbit_inverse;
    if(sector_solver) {
        symmetry_orbit.build(op.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(symmetry_orbit);
        symmetry_sectors.build(maps);orbit_inverse.build(op,maps,orbit_nodes);
        std::printf("[sector solver] orbit_nodes=%d local_MiB=%.3f transform_MiB=%.3f\n",orbit_nodes,
            orbit_inverse.storage_bytes()/1048576.,symmetry_sectors.storage_bytes()/1048576.);
    }else {
        if(generic_d6h)symmetry_orbit.build(op.mesh);
        mbj.build_cached(op,mbj_nodes,mbj_overlap,cache+".mbj");
    }
    if(generic_d6h){
        std::vector<int> image,sign;
        bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,image,sign);
    }
    // This optional accelerator stays in RAM between production packets.
    // Column checkpoints are independent; a restarted worker relearns it.
    bem_streaming::SectorRecycleSpace sector_recycle(symmetry_sectors.offsets,sector_recycle_capacity,sector_recycle_mask);
    std::unique_ptr<bem_streaming::SectorBudgetLearner> sector_budget;
    if(sector_budget_packets)sector_budget.reset(new bem_streaming::SectorBudgetLearner(
        symmetry_sectors.offsets,sector_recycle_capacity,sector_recycle_mask,sector_budget_packets));
    const char* circ_setting=std::getenv("BEM_TMATRIX_CIRCULANT_BLEND");
    const double circ_blend=circ_setting?std::stod(circ_setting):0.;
    if(!std::isfinite(circ_blend) || circ_blend<0 || circ_blend>1)
        throw std::runtime_error("circulant blend must be in [0,1]");
    PrismCirculantInverse circulant;
    double circulant_seconds=0;
    double planar_condition=0,planar_defect=0,planar_D6h_defect=0,planar_audit_s=0;
    size_t planar_kernel_bytes=0;
    if(circ_blend>0) {
        if(device_solver || shape!="prism" || !d6h_mesh)
            throw std::runtime_error("circulant requires host GMRES and a regular D6h prism");
        const auto before=Clock::now();
        const double shift=std::getenv("BEM_CIRCULANT_SHIFT")?std::stod(std::getenv("BEM_CIRCULANT_SHIFT")):0;
        const double opposite=std::getenv("BEM_CIRCULANT_OPPOSITE_WEIGHT")?std::stod(std::getenv("BEM_CIRCULANT_OPPOSITE_WEIGHT")):0;
        const int opposite_mask=sector_integer("BEM_CIRCULANT_OPPOSITE_MASK",3,3);
        const PrismCirculantInverse::Action local_action=[&](const Complex* a,Complex* b){
            if(sector_solver)orbit_inverse.apply(a,b);else mbj.apply(a,b);};
        if(planar_config.analytic) {
            const auto kernels=bem_osrc::make_prism_kernels(op.mesh,ref,ka,ri,planar_config.damping);
            planar_condition=kernels.maximum_condition();planar_defect=kernels.maximum_defect();
            planar_kernel_bytes=kernels.kernel_bytes();
            circulant.build_from_inverse_kernels(op.mesh,ref,local_action,kernels.side.values,kernels.cap.values);
        }else circulant.build(op.mesh,ref,[&](const Complex* a,Complex* b){op.matvec(a,b);},
            local_action,shift,opposite,opposite_mask);
        circulant_seconds=seconds(before);
    }
    PrismSymmetricCirculant symmetric_circulant;
    if(sector_solver && circ_blend>0)symmetric_circulant.build(symmetry_orbit,circulant,orbit_inverse);
    if(vertical_preconditioner && circ_blend>0) {
        const auto before=Clock::now();
        const bool prepared=symmetric_circulant.prepare_full_fft_symmetry(op.mesh);
        circulant_seconds+=seconds(before);
        std::printf("[circulant] native_full_D6h=%d setup_s=%.3f GPU_MiB=%.3f\n",int(prepared),seconds(before),circulant.storage_bytes()/1048576.);
    }
    if(planar_config.analytic) {
        if(!symmetric_circulant.full_fft_symmetry())throw std::runtime_error("planar inverse requires full native D6h FFT");
        const auto before=Clock::now();const int n=op.system_dofs;
        std::vector<Complex> probe(n),image(n),transformed(n),actual(n),expected(n);
        for(int i=0;i<n;++i)probe[i]=Complex(std::sin(.021*i),std::cos(.031*i));
        const auto compare=[&]{double numerator=0,denominator=0;
            for(int i=0;i<n;++i){numerator+=std::norm(actual[i]-expected[i]);denominator+=std::norm(expected[i]);}
            const double error=std::sqrt(numerator/std::max(denominator,1e-300));
            if(!std::isfinite(error))throw std::runtime_error("nonfinite planar inverse symmetry audit");
            planar_D6h_defect=std::max(planar_D6h_defect,error);};
        symmetric_circulant.apply_full_fft_symmetry(probe.data(),image.data(),circ_blend);
        for(bool mirror:{false,true}) {
            symmetry_orbit.transform(probe,transformed,mirror,true);
            symmetric_circulant.apply_full_fft_symmetry(transformed.data(),actual.data(),circ_blend);
            symmetry_orbit.transform(image,expected,mirror,false);compare();
        }
        std::vector<int> vertical,sign;bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
        for(int i=0;i<n;++i){transformed[vertical[i]]=-double(sign[i])*probe[i];expected[vertical[i]]=double(sign[i])*image[i];}
        symmetric_circulant.apply_full_fft_symmetry(transformed.data(),actual.data(),circ_blend);compare();
        planar_audit_s=seconds(before);
        if(planar_D6h_defect>1e-8)throw std::runtime_error("planar inverse D6h audit failed");
        std::printf("[planar BDM inverse] damping=%.6g condition_F=%.6g inverse_defect=%.3e D6h=%.3e audit_s=%.6f\n",
            planar_config.damping,planar_condition,planar_defect,planar_D6h_defect,planar_audit_s);
    }
    PrismVerticalInverse vertical_inverse;
    if(vertical_preconditioner)vertical_inverse.build(op.mesh,[&](const Complex* x,Complex* y){
        if(circ_blend>0)symmetric_circulant.apply(x,y,circ_blend);else orbit_inverse.apply(x,y);
    });
    const int wave_rank=std::getenv("BEM_TMATRIX_WAVE_RANK")?std::stoi(std::getenv("BEM_TMATRIX_WAVE_RANK")):0;
    if(wave_rank<0 || wave_rank>512 || wave_rank%4)throw std::runtime_error("wave rank must be 0..512, a multiple of four");
    if(sector_solver && wave_rank)throw std::runtime_error("sector GMRES does not support the wave coarse inverse");
    GlobalCoarseInverse wave;
    double wave_seconds=0;
    if(wave_rank) {
        if(device_solver || shape!="prism" || !d6h_mesh || ri.imag()!=0)
            throw std::runtime_error("wave coarse inverse requires host GMRES and a lossless D6h prism");
        const auto begin=Clock::now();PrismWaveBasis basis(op.mesh,ka*ri.real());
        wave.build(op.system_dofs,wave_rank,[&](int j,Complex* out){basis(j,out);},
            [&](const Complex* a,Complex* b){op.matvec(a,b);},
            [&](const Complex* a,Complex* b){if(circ_blend>0)circulant.apply(a,b,circ_blend);else mbj.apply(a,b);});
        wave_seconds=seconds(begin);
    }
    if(fft_inner && (device_solver || wave_rank || circ_blend>0 || recycle_capacity))
        throw std::runtime_error("nested FFT requires host flexible GMRES with MBJ and no recycling");
    // Keep the complete FMM operator for the outer equation and FP64 gate.
    // Only the variable inner inverse uses the inexpensive pFFT action.
    std::unique_ptr<bem_streaming::ArnoldiInverse> fft_inverse;
    if(fft_inner)fft_inverse.reset(new bem_streaming::ArnoldiInverse(op.system_dofs,fft_inner,
        [&](const Complex* a,Complex* b){op.select_pfft_backend();op.matvec(a,b);op.select_fmm_backend();},
        [&](const Complex* a,Complex* b){mbj.apply(a,b);}));
    if(device_solver && recycle_capacity)
        throw std::runtime_error("T recycling requires the host GMRES backend");
    bem_streaming::RecycleSpace recycle(op.system_dofs,recycle_capacity);
    const char* recycle_path_env=std::getenv("BEM_TMATRIX_RECYCLE_PATH");
    const char* recycle_signature_env=std::getenv("BEM_TMATRIX_RECYCLE_SIGNATURE");
    const std::string recycle_path=recycle_path_env?recycle_path_env:"";
    const std::string recycle_signature=recycle_signature_env?recycle_signature_env:"";
    const bool recycle_loaded=recycle.load(recycle_path,recycle_signature);
    const auto checkpoint_recycle=[&]() {
        const auto start=Clock::now();
        try {recycle.save(recycle_path,recycle_signature);}
        catch(const std::exception& e) {std::fprintf(stderr,"Optional recycle checkpoint: %s\n",e.what());}
        return seconds(start);
    };
    if(device_solver) mbj.upload_device();
    const char* warm_setting=std::getenv("BEM_TMATRIX_SYMMETRY_WARM_START");
    const bool symmetry_warm_start=!device_solver && d6h_geometry &&
        warm_setting && std::strcmp(warm_setting,"0")!=0;
    if(symmetry_warm_start && !sector_solver) symmetry_orbit.build(op.mesh);
    std::unique_ptr<ResidentPrismInverse> resident_inverse;
    std::unique_ptr<ResidentSectorOperator> resident_composite;
    double resident_setup_s=0,resident_audit_s=0,resident_inverse_error=0,resident_action_error=0,resident_D6h_defect=0;
    size_t resident_bytes=0,resident_sector_bytes=0;
    if(resident_requested){
        if(!sector_solver || !symmetric_circulant.full_fft_symmetry() || !op.device_matvec_available())
            throw std::invalid_argument("resident composite needs full D6h sector inverse and complete GPU operator");
        auto before=Clock::now();const size_t budget=size_t(sector_integer("BEM_TMATRIX_RESIDENT_MEMORY_MB",2048,65536))*1024*1024;
        circulant.prepare_device_mapping(true);circulant.enable_device_mapping(false);
        {
            const auto transform=bem_sector_device::Layout::capture(symmetry_sectors);
            const size_t chain_budget=transform.storage_bytes()+2*size_t(op.system_dofs)*sizeof(Complex);
            const size_t mapping_bytes=circulant.device_mapping_storage_bytes();
            if(budget<=chain_budget+mapping_bytes)throw std::runtime_error("resident composite memory budget exhausted by transforms");
            resident_inverse.reset(new ResidentPrismInverse);
            resident_inverse->init(bem_orbit_device::Layout::capture(orbit_inverse),
                bem_orbit_device::SymmetricLayout::capture(symmetric_circulant),circulant,budget-chain_budget-mapping_bytes);
            resident_composite.reset(new ResidentSectorOperator);
            resident_composite->init(op,*resident_inverse,transform,chain_budget);
            resident_sector_bytes=resident_composite->storage_bytes();resident_bytes=resident_inverse->storage_bytes()+resident_sector_bytes+mapping_bytes;
        }
        resident_setup_s=seconds(before);before=Clock::now();const int count=op.system_dofs;
        std::vector<Complex> probe(count),reference(count),actual(count),rotated(count),expected(count),encoded(count),physical(count);
        const auto difference=[&](const std::vector<Complex>& a,const std::vector<Complex>& b){
            long double num=0,den=0;for(int i=0;i<count;++i){num+=std::norm(a[i]-b[i]);den+=std::norm(b[i]);}
            const double error=std::sqrt(double(num/std::max(den,1e-300L)));
            if(!std::isfinite(error))throw std::runtime_error("nonfinite resident startup audit");
            return error;
        };
        for(int i=0;i<count;++i)probe[i]=Complex(std::sin(.031*i),std::cos(.043*i));
        symmetric_circulant.apply_full_fft_symmetry(probe.data(),reference.data(),circ_blend);
        resident_inverse->apply(probe.data(),actual.data(),circ_blend);resident_inverse_error=difference(actual,reference);
        std::vector<int> vertical,sign;bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
        for(int g=0;g<3;++g){
            if(g<2){symmetry_orbit.transform(probe,rotated,g==1,true);symmetry_orbit.transform(actual,expected,g==1,false);}
            else for(int i=0;i<count;++i){rotated[vertical[i]]=-double(sign[i])*probe[i];expected[vertical[i]]=double(sign[i])*actual[i];}
            resident_inverse->apply(rotated.data(),physical.data(),circ_blend);
            resident_D6h_defect=std::max(resident_D6h_defect,difference(physical,expected));
        }
        // Compare the same mixed mathematical action before any solve. The
        // unchanged original physical FP64 gate remains the final authority.
        op.matvec(reference.data(),physical.data());symmetry_sectors.encode(physical.data(),expected.data(),true);
        symmetry_sectors.encode(probe.data(),encoded.data(),true);
        resident_composite->apply(encoded.data(),actual.data(),circ_blend);resident_action_error=difference(actual,expected);
        resident_audit_s=seconds(before);
        std::printf("[resident composite] explicit_MiB=%.3f sector_MiB=%.3f setup=%.6f audit=%.6f inverse=%.3e action=%.3e D6h=%.3e\n",
            resident_bytes/1048576.,resident_sector_bytes/1048576.,resident_setup_s,resident_audit_s,resident_inverse_error,resident_action_error,resident_D6h_defect);
        if(resident_inverse_error>2e-12 || resident_D6h_defect>1e-8 || resident_action_error>2e-6)
            throw std::runtime_error("resident startup inverse/D6h/operator audit failed");
    }
    const bem_streaming::Action perform_local_inverse=[&](const Complex* x,Complex* y){
                if(resident_inverse)resident_inverse->apply(x,y,circ_blend);
                else if(sector_solver){if(symmetric_circulant.full_fft_symmetry())symmetric_circulant.apply_full_fft_symmetry(x,y,circ_blend);else if(vertical_preconditioner)vertical_inverse.apply(x,y);else if(circ_blend>0)symmetric_circulant.apply(x,y,circ_blend);else orbit_inverse.apply(x,y);}
                else if(fft_inner)fft_inverse->solve(x,y,.1);else if(wave_rank)wave.apply(x,y);else if(circ_blend>0)circulant.apply(x,y,circ_blend);else mbj.apply(x,y);};
    std::unique_ptr<CoarseImageExperiment> ritz_correction;
    double ritz_setup_s=0,ritz_D6h_defect=0;
    double ritz_sector_floor=0;
    if(const char* value=std::getenv("BEM_TMATRIX_RITZ_SECTOR_FLOOR")){
        char* end=nullptr;ritz_sector_floor=std::strtod(value,&end);
        if(end==value||*end||!std::isfinite(ritz_sector_floor)||ritz_sector_floor<0||ritz_sector_floor>.01)
            throw std::invalid_argument("BEM_TMATRIX_RITZ_SECTOR_FLOOR must be in 0..0.01");
    }
    size_t ritz_columns=0,ritz_estimated_peak_bytes=0;
    size_t ritz_image_mixed_actions=0,ritz_image_strict_actions=0,ritz_image_model_bytes=0;
    const char* ritz_path=std::getenv("BEM_TMATRIX_RITZ_PRIOR");
    if(ritz_path && *ritz_path){
        if(!sector_solver || !symmetric_circulant.full_fft_symmetry() || gpu_batch2)
            throw std::runtime_error("Ritz correction needs full D6h FFT sector inverse; GPU batch2 is not supported yet");
        const auto before=Clock::now();const size_t n=op.system_dofs;
        bem_ritz::Parameters wanted;wanted.dofs=n;wanted.ref=ref;wanted.ka=ka;wanted.ri_real=ri.real();
        wanted.ri_imag=ri.imag();wanted.aspect=prism_aspect;wanted.duffy=duffy_order;
        bem_ritz::Reader prior(ritz_path,wanted);ritz_columns=prior.parameters().columns;
        // Two vertical images, current/image vectors, permanent scratch and
        // insertion/audit temporaries. Excludes the pre-existing operator.
        ritz_estimated_peak_bytes=(4*ritz_columns+14)*n*sizeof(Complex)+4*n*sizeof(int)+1024*1024;
        ritz_estimated_peak_bytes=bem_cpu::memory::add(ritz_estimated_peak_bytes,ritz_image_config.estimated_additional_bytes(n));
        const size_t budget=size_t(sector_integer("BEM_TMATRIX_RITZ_MEMORY_MB",1024,65536))*1024*1024;
        if(ritz_estimated_peak_bytes>budget)throw std::runtime_error("Ritz correction exceeds its host-memory budget");
        ritz_correction.reset(new CoarseImageExperiment(op.mesh,symmetry_sectors,perform_local_inverse,ritz_sector_floor));
        bem_ritz::Vector column(n),image(n);
        {
            bem_ritz::PhysicalImages images(op.mesh,symmetry_orbit,ritz_image_config,
                [&](const Complex* a,Complex* b){op.matvec(a,b);},
                [&](const Complex* a,Complex* b){op.matvec_strict(a,b);});
            ritz_image_model_bytes=images.storage_bytes();
            if(ritz_image_model_bytes>ritz_image_config.estimated_additional_bytes(n))
                throw std::runtime_error("Ritz image workspace exceeds its estimate");
            for(size_t j=0;j<ritz_columns;++j){prior.read(j,column);images.apply(column.data(),image.data());ritz_correction->add_d6h(column,image);}
            ritz_image_mixed_actions=images.mixed_actions();ritz_image_strict_actions=images.strict_actions();
        }
        if(ritz_correction->rank()==0)throw std::runtime_error("Ritz correction retained no current directions");
        bem_ritz::Vector probe(n),value(n),rotated(n),actual(n),expected(n);
        for(size_t i=0;i<n;++i)probe[i]=Complex(std::sin(.031*i),std::cos(.043*i));
        ritz_correction->apply(probe.data(),value.data());
        std::vector<int> vertical,sign;bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
        for(int g=0;g<3;++g){
            if(g<2){symmetry_orbit.transform(probe,rotated,g==1,true);symmetry_orbit.transform(value,expected,g==1,false);}
            else for(size_t i=0;i<n;++i){rotated[vertical[i]]=-double(sign[i])*probe[i];expected[vertical[i]]=double(sign[i])*value[i];}
            ritz_correction->apply(rotated.data(),actual.data());
            for(size_t i=0;i<n;++i)actual[i]-=expected[i];
            const double defect=bem_streaming::norm(actual.data(),n)/std::max(bem_streaming::norm(expected.data(),n),1e-300);
            if(!std::isfinite(defect)||defect>1e-8)throw std::runtime_error("Ritz correction full D6h gate failed");
            ritz_D6h_defect=std::max(ritz_D6h_defect,defect);
        }
        ritz_setup_s=seconds(before);
        std::printf("[Ritz inverse] columns=%zu rank=%d setup_s=%.6f basis_MiB=%.3f D6h=%.3e\n",
            ritz_columns,ritz_correction->rank(),ritz_setup_s,ritz_correction->bytes()/1048576.,ritz_D6h_defect);
    }
    const bem_streaming::Action perform_inverse=[&](const Complex* x,Complex* y){
        if(ritz_correction)ritz_correction->apply(x,y);else perform_local_inverse(x,y);
    };
    const int cpu_threads=sector_integer("BEM_CPU_THREADS",16,256);
    const int gpu_threads=sector_integer("BEM_CPU_GPU_THREADS",4,256);
    const int solver_threads=sector_integer("BEM_CPU_SOLVER_THREADS",4,256);
    if(hybrid_requested && (!cpu_threads||!gpu_threads||!solver_threads))throw std::runtime_error("hybrid thread counts must be positive");
    if(hybrid_requested)bem_cpu::configure_threads(gpu_threads,"BEM_CPU_GPU_AFFINITY");
    bool cpu_fft_enabled=false,cpu_fft_available=false;
    double cpu_fft_s=0,gpu_fft_s=0,cpu_fft_error=0;
    const char* fft_choice=std::getenv("BEM_CIRCULANT_CPU_FFT");
    const std::string cpu_fft_mode=fft_choice?fft_choice:(hybrid_requested?"auto":"0");
    if(cpu_fft_mode!="0"&&cpu_fft_mode!="1"&&cpu_fft_mode!="auto")throw std::runtime_error("BEM_CIRCULANT_CPU_FFT must be 0, 1 or auto");
    if(gpu_batch2 && cpu_fft_mode!="0")throw std::runtime_error("GPU batching requires explicit BEM_CIRCULANT_CPU_FFT=0");
    if(cpu_fft_mode!="0" && sector_solver && circ_blend>0) {
        cpu_fft_available=circulant.prepare_cpu();
        if(!cpu_fft_available&&cpu_fft_mode=="1")throw std::runtime_error("CPU FFTW runtime unavailable; set BEM_CPU_FFTW_LIBRARY");
        if(cpu_fft_available) {
            std::vector<Complex> input(op.system_dofs),reference(input.size()),host(input.size());
            for(size_t i=0;i<input.size();++i)input[i]=Complex(std::sin(.117*i),std::cos(.193*i));
            // Time the COMPLETE preconditioner, including symmetry and local
            // blocks, rather than timing an isolated FFT plan.
            perform_inverse(input.data(),reference.data());
            for(int j=0;j<3;++j){const auto t=Clock::now();perform_inverse(input.data(),reference.data());gpu_fft_s+=seconds(t)/3;}
            circulant.enable_cpu(true);perform_inverse(input.data(),host.data());
            for(int j=0;j<3;++j){const auto t=Clock::now();perform_inverse(input.data(),host.data());cpu_fft_s+=seconds(t)/3;}
            long double num=0,den=0;for(size_t i=0;i<input.size();++i){num+=std::norm(host[i]-reference[i]);den+=std::norm(reference[i]);}
            cpu_fft_error=std::sqrt(double(num/std::max(den,1e-300L)));
            if(!std::isfinite(cpu_fft_error)||cpu_fft_error>1e-11)throw std::runtime_error("CPU FFT preconditioner disagrees with GPU symbol");
            cpu_fft_enabled=cpu_fft_mode=="1" || cpu_fft_s<.95*gpu_fft_s;
            circulant.enable_cpu(cpu_fft_enabled);
            std::printf("[spectral CPU] full preconditioner CPU %.6fs GPU %.6fs relative error %.3e selected=%d\n",cpu_fft_s,gpu_fft_s,cpu_fft_error,int(cpu_fft_enabled));
        }
    }
    std::unique_ptr<bem_cpu::MullerAction> cpu_checker;
    if(gpu_batch2){
        const char* host_mixed=std::getenv("BEM_MULLER_GPU_STRICT_ONLY");
        if(!op.host_batch2_available() || (host_mixed && std::strcmp(host_mixed,"0")!=0) ||
           !symmetric_circulant.full_fft_symmetry())
            throw std::runtime_error("GPU batching requires mixed GPU assembly, multilevel FMM and native full D6h FFT inverse");
        symmetric_circulant.prepare_batch2();
    }
    double cpu_setup_seconds=0,cpu_validation_error=0,cpu_D6h_error=0;
    size_t cpu_estimated_peak_bytes=0,cpu_memory_budget=0;
    size_t cpu_phase_budget=size_t(sector_integer("BEM_CPU_PHASE_MB",8192,65536))*1024*1024;
    if(cpu_requested) {
        const auto before=Clock::now();
        const size_t budget=size_t(sector_integer("BEM_CPU_MEMORY_MB",24576,262144))*1024*1024;
        cpu_memory_budget=budget;
        const auto estimate=[&]{
            size_t value=bem_cpu::MullerAction::estimated_workspace_bytes(op,cpu_phase_budget,true,cpu_backend.face,cpu_backend.batch,cpu_backend.tile);
            if(cpu_backend.face){
                // Startup symmetry probes, map vectors and an allowance for
                // FFTW plan/internal allocations and per-thread kernel tiles.
                // This is an estimate, not an OS-level memory reservation.
                value=bem_cpu::memory::add(value,bem_cpu::memory::mul(size_t(op.system_dofs),6*sizeof(Complex)+2*sizeof(int)));
                value=bem_cpu::memory::add(value,64ULL*1024*1024+size_t(cpu_threads)*64*1024);
            }
            return value;
        };
        while(cpu_phase_budget && estimate()>budget)cpu_phase_budget/=2;
        cpu_estimated_peak_bytes=estimate();
        if(cpu_estimated_peak_bytes>budget)
            throw std::runtime_error("CPU checker exceeds BEM_CPU_MEMORY_MB before allocating its workspace");
        setenv("BEM_CPU_AVX512","1",0);
        bem_cpu::configure_threads(cpu_threads,"BEM_CPU_AFFINITY");
        cpu_checker.reset(new bem_cpu::MullerAction(op,cpu_phase_budget,true,cpu_backend.face,cpu_backend.batch,cpu_backend.tile,cpu_backend.shared_near));
        if(cpu_backend.shared_near && cpu_checker->shared_avx512_near_bands!=4)
            throw std::runtime_error("CPU shared checker requires AVX512 in both media and face kinds");
        if(cpu_checker->workspace_bytes()>cpu_memory_budget)
            throw std::runtime_error("CPU checker explicit arrays exceed memory budget");
        std::vector<Complex> probe(op.system_dofs),host(probe.size()),device(probe.size());
        for(size_t j=0;j<probe.size();++j)probe[j]=Complex(std::sin(.173*j),std::cos(.137*j));
        cpu_checker->apply(probe.data(),host.data());
        bem_cpu::configure_threads(gpu_threads,"BEM_CPU_GPU_AFFINITY");
        op.matvec_strict(probe.data(),device.data());
        long double num=0,den=0;
        for(size_t j=0;j<probe.size();++j){num+=std::norm(host[j]-device[j]);den+=std::norm(device[j]);}
        cpu_validation_error=std::sqrt(double(num/std::max(den,1e-300L)));
        if(!std::isfinite(cpu_validation_error)||cpu_validation_error>(cpu_backend.face?2e-9:2e-7))
            throw std::runtime_error("CPU checker failed startup GPU FP64 operator comparison");
        if(cpu_backend.face){
            std::vector<Complex> rotated(probe.size()),actual(probe.size()),expected(probe.size());
            std::vector<int> vertical,sign;bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
            bem_cpu::configure_threads(cpu_threads,"BEM_CPU_AFFINITY");
            for(int g=0;g<3;++g){
                if(g<2){symmetry_orbit.transform(probe,rotated,g==1,false);symmetry_orbit.transform(host,expected,g==1,true);}
                else for(size_t i=0;i<probe.size();++i){rotated[vertical[i]]=double(sign[i])*probe[i];expected[vertical[i]]=-double(sign[i])*host[i];}
                cpu_checker->apply(rotated.data(),actual.data());num=0;den=0;
                for(size_t i=0;i<actual.size();++i){num+=std::norm(actual[i]-expected[i]);den+=std::norm(expected[i]);}
                const double defect=std::sqrt(double(num/std::max(den,1e-300L)));
                if(!std::isfinite(defect)||defect>2e-5)throw std::runtime_error("CPU face checker failed original D6h gate");
                cpu_D6h_error=std::max(cpu_D6h_error,defect);
            }
            bem_cpu::configure_threads(gpu_threads,"BEM_CPU_GPU_AFFINITY");
        }
        cpu_setup_seconds=seconds(before);
        std::printf("[hybrid] independent CPU FP64 checker %s: %.3f GiB, estimated_peak %.3f GiB, error %.3e, D6h %.3e, setup %.3fs\n",
            cpu_backend.backend.c_str(),cpu_checker->workspace_bytes()/1073741824.,cpu_estimated_peak_bytes/1073741824.,cpu_validation_error,cpu_D6h_error,cpu_setup_seconds);
    }
    const double mbj_seconds=seconds(mbj_start);
    std::unique_ptr<ExternalFFTAction> external_fft;
    double external_fft_error=0,external_fft_setup_s=0;
    double checker_baseline_s=0,checker_candidate_s=0,checker_difference=0;
    double external_fft_wire_error=0;
    if(const char* path=std::getenv("BEM_TMATRIX_COMPRESSED_FFT_SOCKET")) {
        if(shape!="prism" || !d6h_mesh || !sector_solver || hybrid_requested ||
           device_solver || resident_requested || gpu_batch2 || fft_inner)
            throw std::invalid_argument("compressed FFT requires the host D6h sector solver without hybrid/device/nested modes");
        const auto begin=Clock::now();const int count=op.system_dofs;
        std::vector<Complex> probe(count),reference(count),actual(count);
        for(int i=0;i<count;++i)probe[i]=Complex(std::sin(.021*i),std::cos(.031*i));
        // Materialize accurate FMM scratch before the large FFT upload.
        const bool compare_checker=std::getenv("BEM_TMATRIX_CHECKER_COMPARE") &&
            std::strcmp(std::getenv("BEM_TMATRIX_CHECKER_COMPARE"),"0")!=0;
        if(compare_checker) {
            const char* keys[]={"BEM_FMM_P2P_DS_THREADS","BEM_FMM_P2P_DS_LEAF_SPLIT","BEM_FMM_P2P_DS_FACTORED"};
            std::array<std::string,3> saved;std::array<bool,3> present;
            for(int i=0;i<3;++i){const char* value=std::getenv(keys[i]);present[i]=value;saved[i]=value?value:"";}
            const auto restore=[&](){for(int i=0;i<3;++i){if(present[i])setenv(keys[i],saved[i].c_str(),1);else unsetenv(keys[i]);}};
            // ABBA on this exact operator; the first call materializes scratch.
            for(int trial=0;trial<4;++trial) {
                const bool candidate=trial==1 || trial==2;
                restore();if(!candidate){setenv(keys[0],"128",1);setenv(keys[1],"32",1);setenv(keys[2],"0",1);}
                const auto before=Clock::now();op.matvec_strict(probe.data(),actual.data());const double elapsed=seconds(before);
                if(trial==0)reference=actual;
                else {
                    long double num=0,den=0;for(int i=0;i<count;++i){num+=std::norm(actual[i]-reference[i]);den+=std::norm(reference[i]);}
                    const double difference=std::sqrt(double(num/std::max(den,1e-300L)));
                    if(!std::isfinite(difference))throw std::runtime_error("nonfinite checker comparison");
                    checker_difference=std::max(checker_difference,difference);
                }
                if(candidate)checker_candidate_s+=elapsed*.5;else if(trial==3)checker_baseline_s=elapsed;
                std::printf("[checker comparison] trial=%d candidate=%d seconds=%.6f error=%.3e\n",trial,candidate,elapsed,checker_difference);std::fflush(stdout);
            }
            restore();
            if(!std::isfinite(checker_difference) || checker_difference>1e-10)
                throw std::runtime_error("checker optimization changed accurate action");
        }else op.matvec_strict(probe.data(),reference.data());
        external_fft.reset(new ExternalFFTAction(path,count,ref,ka,ri,prism_aspect,7,duffy_order));
        external_fft->apply(probe.data(),actual.data());
        long double numerator=0,denominator=0;
        for(int i=0;i<count;++i){numerator+=std::norm(actual[i]-reference[i]);denominator+=std::norm(reference[i]);}
        external_fft_error=std::sqrt(double(numerator/std::max(denominator,1e-300L)));
        if(!std::isfinite(external_fft_error) || external_fft_error>5e-4)
            throw std::runtime_error("compressed FFT startup action audit failed");
        const char* compact_setting=std::getenv("BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32");
        if(compact_setting && std::strcmp(compact_setting,"0")!=0) {
            const std::string saved=compact_setting;
            setenv("BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32","0",1);
            external_fft->apply(probe.data(),reference.data());
            setenv("BEM_TMATRIX_COMPRESSED_FFT_WIRE_FP32",saved.c_str(),1);
            numerator=denominator=0;
            for(int i=0;i<count;++i){numerator+=std::norm(actual[i]-reference[i]);denominator+=std::norm(reference[i]);}
            external_fft_wire_error=std::sqrt(double(numerator/std::max(denominator,1e-300L)));
            if(!std::isfinite(external_fft_wire_error) || external_fft_wire_error>1e-7)
                throw std::runtime_error("compact FFT transport action audit failed");
        }
        external_fft_setup_s=seconds(begin);
        std::printf("[compressed FFT] original-checker error=%.3e setup/audit=%.3fs\n",external_fft_error,external_fft_setup_s);
    }
    const TriQuad quad=tri_quadrature(13);
    bem_tmatrix::StreamingRhsAssembler rhs_assembler(quad.npts);
    const size_t sample_count=op.mass_quadrature.size();
    if(sample_count!=op.mesh.elements.size()*size_t(quad.npts))
        throw std::runtime_error("T worker requires the cached 13-point mass quadrature");
    std::vector<double> positions(sample_count*3);
#pragma omp parallel for schedule(static)
    for(size_t q=0;q<sample_count;++q) {
        const Vec3& p=op.mass_quadrature[q].sample.position;
        positions[3*q]=p.x;positions[3*q+1]=p.y;positions[3*q+2]=p.z;
    }
    double radius=0;
    for(const Vec3& v:mesh.verts) radius=std::max(radius,v.norm());
    std::ostringstream hello;
    hello<<std::setprecision(17)<<"{\"system_dofs\":"<<op.system_dofs<<",\"rhs_points\":"<<sample_count
         <<",\"prism_dihedral_mesh\":"<<(shape=="prism" && d6h_mesh ? "true":"false")
         <<",\"geometry_d6h_verified\":"<<(generic_d6h?"true":"false")
         <<",\"prism_height_to_width_across_flats\":"<<prism_aspect
         <<",\"duffy_order\":"<<duffy_order
         <<",\"rhs_assembly\":\"cached_element_chunks\""
         <<",\"rhs_chunk_elements\":"<<rhs_assembler.chunk_points()/quad.npts
         <<",\"rhs_threads_max\":"<<rhs_assembler.thread_limit()
         <<",\"directions\":"<<directions.size()<<",\"radius\":"<<radius
         <<",\"operator_setup_s\":"<<setup_seconds<<",\"mbj_setup_s\":"<<mbj_seconds
         <<",\"mbj_storage_mb\":"<<mbj.storage_megabytes()
         <<",\"sector_gmres\":"<<(sector_solver?"true":"false")<<",\"orbit_nodes\":"<<(sector_solver?orbit_nodes:0)
         <<",\"hybrid_batch_capacity\":"<<(hybrid_requested?2:1)
         <<",\"hybrid_merge_recycle\":"<<(merge_lane_recycle?"true":"false")
         <<",\"hybrid_gpu_batch2\":"<<(gpu_batch2?"true":"false")
         <<",\"hybrid_gpu_batch_wait_us\":"<<(gpu_batch2?gpu_pair_wait:0)
         <<",\"hybrid_cpu_check_budget\":"<<cpu_check_budget
         <<",\"profile_first_batch\":"<<(profile_first_batch?"true":"false")
         <<",\"per_packet_tolerance\":true"
         <<",\"hybrid_cpu_strict\":"<<(cpu_checker?"true":"false")
         <<",\"hybrid_cpu_threads\":"<<(cpu_requested?cpu_threads:0)
         <<",\"hybrid_gpu_host_threads\":"<<(hybrid_requested?gpu_threads:0)
         <<",\"hybrid_solver_threads_per_lane\":"<<(hybrid_requested?solver_threads:0)
         <<",\"hybrid_cpu_avx512_near_bands\":"<<(cpu_checker?cpu_checker->avx512_near_bands:0)
         <<",\"hybrid_cpu_checker_backend\":\""<<cpu_backend.backend<<"\""
         <<",\"hybrid_cpu_shared_near_bands\":"<<(cpu_checker?cpu_checker->shared_avx512_near_bands:0)
         <<",\"hybrid_cpu_face_tile\":"<<(cpu_backend.face?cpu_backend.tile:0)
         <<",\"hybrid_cpu_D6h_error\":"<<cpu_D6h_error
         <<",\"hybrid_cpu_estimated_peak_bytes\":"<<cpu_estimated_peak_bytes
         <<",\"hybrid_cpu_memory_budget_bytes\":"<<cpu_memory_budget
         <<",\"hybrid_cpu_setup_s\":"<<cpu_setup_seconds<<",\"hybrid_cpu_validation_error\":"<<cpu_validation_error
         <<",\"hybrid_cpu_workspace_bytes\":"<<(cpu_checker?cpu_checker->workspace_bytes():0)
         <<",\"hybrid_cpu_phase_budget_bytes_per_band\":"<<cpu_phase_budget
         <<",\"cpu_fft_available\":"<<(cpu_fft_available?"true":"false")<<",\"cpu_fft_enabled\":"<<(cpu_fft_enabled?"true":"false")
         <<",\"cpu_fft_preconditioner_s\":"<<cpu_fft_s<<",\"gpu_fft_preconditioner_s\":"<<gpu_fft_s
         <<",\"cpu_fft_relative_error\":"<<cpu_fft_error<<",\"cpu_fft_storage_bytes\":"<<circulant.cpu_storage_bytes()
         <<",\"gpu_operator_assembly\":"<<(op.gpu_operator_assembly?"true":"false")
         <<",\"gpu_host_correction\":"<<(op.gpu_host_correction?"true":"false")
         <<",\"sector_recycle_capacity\":"<<sector_recycle_capacity<<",\"sector_recycle_mask\":"<<sector_recycle_mask
         <<",\"sector_budget_training_packets\":"<<sector_budget_packets
         <<",\"sector_recycle_disk_checkpoint\":false"
         <<",\"physical_ritz_columns\":"<<ritz_columns
         <<",\"physical_ritz_sector_floor\":"<<ritz_sector_floor
         <<",\"physical_ritz_encoded_cold_start\":"<<(ritz_correction?"true":"false")
         <<",\"physical_ritz_rank\":"<<(ritz_correction?ritz_correction->rank():0)
         <<",\"physical_ritz_basis_bytes\":"<<(ritz_correction?ritz_correction->bytes():0)
         <<",\"physical_ritz_estimated_peak_bytes\":"<<ritz_estimated_peak_bytes
         <<",\"physical_ritz_setup_s\":"<<ritz_setup_s<<",\"physical_ritz_D6h_defect\":"<<ritz_D6h_defect
         <<",\"physical_ritz_image_model\":\""<<ritz_image_config.model<<"\""
         <<",\"physical_ritz_image_mixed_actions\":"<<ritz_image_mixed_actions
         <<",\"physical_ritz_image_strict_actions\":"<<ritz_image_strict_actions
         <<",\"physical_ritz_image_model_bytes\":"<<ritz_image_model_bytes
         <<",\"resident_composite\":"<<(resident_composite?"true":"false")
         <<",\"compressed_fft\":"<<(external_fft?"true":"false")
         <<",\"compressed_fft_setup_s\":"<<external_fft_setup_s
         <<",\"compressed_fft_action_error\":"<<external_fft_error
         <<",\"compressed_fft_wire_error\":"<<external_fft_wire_error
         <<",\"checker_baseline_s\":"<<checker_baseline_s
         <<",\"checker_candidate_s\":"<<checker_candidate_s
         <<",\"checker_relative_difference\":"<<checker_difference
         <<",\"double_single_near_check\":"<<(std::getenv("BEM_FMM_P2P_DOUBLE_SINGLE") && std::strcmp(std::getenv("BEM_FMM_P2P_DOUBLE_SINGLE"),"0")?"true":"false")
         <<",\"resident_explicit_bytes\":"<<resident_bytes<<",\"resident_sector_bytes\":"<<resident_sector_bytes
         <<",\"resident_setup_s\":"<<resident_setup_s<<",\"resident_audit_s\":"<<resident_audit_s
         <<",\"resident_inverse_error\":"<<resident_inverse_error<<",\"resident_action_error\":"<<resident_action_error
         <<",\"resident_D6h_defect\":"<<resident_D6h_defect
         <<",\"sector_first_pass_factor\":"<<sector_first_pass_factor
         <<",\"vertical_preconditioner\":"<<(vertical_preconditioner?"true":"false")
         <<",\"orbit_storage_mib\":"<<orbit_inverse.storage_bytes()/1048576.
         <<",\"fast_fft_patch_orbits\":"<<(symmetric_circulant.fast_patch_orbits()?"true":"false")
         <<",\"native_full_fft_symmetry\":"<<(symmetric_circulant.full_fft_symmetry()?"true":"false")
         <<",\"circulant_blend\":"<<circ_blend<<",\"circulant_setup_s\":"<<circulant_seconds
         <<",\"circulant_storage_mib\":"<<circulant.storage_bytes()/1048576.
         <<",\"circulant_symbol\":\""<<planar_config.symbol<<"\",\"planar_damping\":"<<planar_config.damping
         <<",\"planar_condition_F\":"<<planar_condition<<",\"planar_inverse_defect\":"<<planar_defect
         <<",\"planar_D6h_defect\":"<<planar_D6h_defect<<",\"planar_audit_s\":"<<planar_audit_s
         <<",\"planar_kernel_bytes\":"<<planar_kernel_bytes<<",\"planar_aliases\":"<<(planar_config.analytic?1:0)
         <<",\"planar_gauss_order\":"<<(planar_config.analytic?16:0)
         <<",\"opposite_face_weight\":"<<circulant.opposite_weight()<<",\"opposite_face_mask\":"<<circulant.opposite_mask()
         <<",\"wave_coarse_rank\":"<<wave.rank()<<",\"wave_coarse_setup_s\":"<<wave_seconds
         <<",\"wave_coarse_storage_mib\":"<<wave.storage_bytes()/1048576.
         <<",\"mbj_overlap\":"<<mbj_overlap
         <<",\"fft_inner_steps\":"<<fft_inner<<",\"flexible_gmres\":"<<(fft_inner?"true":"false")
         <<",\"fft_correction_radius_h\":"<<fft_radius
         <<",\"symmetry_warm_start\":"<<(symmetry_warm_start?"true":"false")
         <<",\"krylov_recycle_capacity\":"<<recycle_capacity
         <<",\"krylov_recycle_loaded\":"<<(recycle_loaded?"true":"false")
         <<",\"solver_backend\":\""<<(device_solver?"paired_device_gmres":"host_gmres_gpu_banded_fmm")<<"\""
         <<",\"operator_backend\":\""<<op.backend_name()<<"\""
         <<",\"fmm_max_leaf\":"<<max_leaf<<",\"mbj_nodes\":"<<mbj_nodes
         <<",\"multilevel_fmm\":"<<(multilevel && std::strcmp(multilevel,"0")!=0?"true":"false")
         <<",\"fmm_depth\":"<<op.fmm_exterior.tree.max_level
         <<",\"fmm_order_exterior\":"<<op.fmm_exterior.p
         <<",\"fmm_order_interior\":"<<op.fmm_interior.p
         <<",\"near_field_fp32\":"<<(op.fmm_near_fp32 ? "true" : "false")
         <<",\"dense_matrix_formed\":false,\"resident_solution_columns\":"<<(hybrid_requested?4:2)<<"}";
    message(client,hello.str()); transfer(client,positions.data(),positions.size()*sizeof(double),true);
    std::vector<double>().swap(positions);
    const int n=op.system_dofs;
    struct Packet {
        std::vector<Complex> rhs[2],solution[2];
        MullerPairedGmresResult result;
        bem_tmatrix::WarmStartResult warm;
        int iterations_x=0,iterations_y=0,strict_checks_x=0,strict_checks_y=0,deferred_checks_x=0,deferred_checks_y=0;
        double recycle_update_seconds=0,host_action_seconds=0,host_strict_seconds=0,host_orthogonalization_seconds=0;
        int32_t operation=0;double rhs_seconds=0,batch_seconds=0,tolerance=0;bool hybrid=false;
        double merge_seconds=0,rebudget_seconds=0;int merge_candidates=0;
        double accounted_owner_wait=0,accounted_cpu_wait=0,accounted_cpu_tail=0;
        size_t ritz_physical_calls=0,ritz_encoded_calls=0;
        size_t resident_action_calls=0;
        uint64_t external_fft_calls=0;
        std::array<int,12> sector_iterations_x{},sector_iterations_y{},sector_capacities_used{};
        Clock::time_point started;
        bem_streaming::HybridScheduler::Statistics hybrid_stats;
        explicit Packet(int n):started(Clock::now()){for(int j=0;j<2;++j){rhs[j].resize(n);solution[j].resize(n);}}
    };
    const size_t field_chunk=rhs_assembler.chunk_points();
    std::vector<Complex> fields(field_chunk*6);
    int completed_pairs=0;
    std::unique_ptr<bem_streaming::SectorRecycleSpace> lane_recycle[2];
    if(hybrid_requested)for(auto& lane:lane_recycle)lane.reset(new bem_streaming::SectorRecycleSpace(
        symmetry_sectors.offsets,sector_recycle_capacity,sector_recycle_mask));
    const auto receive_packet=[&](Packet& packet,int32_t operation) {
        packet.tolerance=tol;
        if(operation==5) {
            transfer(client,&operation,sizeof(operation),false);
            transfer(client,&packet.tolerance,sizeof(packet.tolerance),false);
            if(!std::isfinite(packet.tolerance) || packet.tolerance<=0 || packet.tolerance>tol)
                throw std::runtime_error("packet tolerance must be positive and no looser than the worker ceiling");
        }
        packet.operation=operation;packet.started=Clock::now();auto& rhs=packet.rhs;
        if(operation==1 || operation==3) {
            std::fill(rhs[0].begin(),rhs[0].end(),Complex(0));
            std::fill(rhs[1].begin(),rhs[1].end(),Complex(0));
            for(int pol=0;pol<2;++pol)
                for(size_t begin=0;begin<sample_count;begin+=field_chunk) {
                    const size_t count=std::min(field_chunk,sample_count-begin);
                    transfer(client,fields.data(),count*6*sizeof(Complex),false);
                    rhs_assembler.add(op.mesh,op.mass_quadrature,begin,count,fields.data(),rhs[pol]);
                }
        } else if(operation==2) {
            double plane[9]; transfer(client,plane,sizeof(plane),false);
            const Vec3 khat(plane[6],plane[7],plane[8]);
            rhs[0]=muller_nodal_planewave_rhs(op.mesh,ka,Vec3(plane[0],plane[1],plane[2]),khat,13);
            rhs[1]=muller_nodal_planewave_rhs(op.mesh,ka,Vec3(plane[3],plane[4],plane[5]),khat,13);
        } else throw std::runtime_error("invalid worker operation");
        packet.rhs_seconds=seconds(packet.started);
    };
    const auto solve_packet=[&](Packet& packet,int lane,bem_streaming::HybridScheduler* scheduler,
                               bem_streaming::SectorRecycleSpace* active_recycle) {
        const uint64_t external_before=external_fft?external_fft->calls():0;
        const double tol=packet.tolerance;
        auto& rhs=packet.rhs;auto& solution=packet.solution;auto& result=packet.result;auto& warm=packet.warm;
        const int32_t operation=packet.operation;
        if(operation==3 && active_recycle)for(int s=0;s<12;++s)
            packet.sector_capacities_used[s]=active_recycle->sector(s).capacity();
        auto& iterations_x=packet.iterations_x;
        auto& iterations_y=packet.iterations_y;
        auto& strict_checks_x=packet.strict_checks_x;
        auto& strict_checks_y=packet.strict_checks_y;
        auto& deferred_checks_x=packet.deferred_checks_x;
        auto& deferred_checks_y=packet.deferred_checks_y;
        auto& recycle_update_seconds=packet.recycle_update_seconds;
        auto& host_action_seconds=packet.host_action_seconds;
        auto& host_strict_seconds=packet.host_strict_seconds;
        auto& host_orthogonalization_seconds=packet.host_orthogonalization_seconds;
        if(scheduler)bem_cpu::configure_threads(solver_threads,"BEM_CPU_GPU_AFFINITY");
        for(auto& vector:solution) std::fill(vector.begin(),vector.end(),Complex(0));
        if(device_solver) {
            result=solve_muller_paired_gmres_device(op,mbj,rhs[0].data(),rhs[1].data(),
                solution[0].data(),solution[1].data(),restart,tol,maxiter,true);
        } else {
            const bem_streaming::Action action=[&](const Complex* x,Complex* y){
                if(external_fft){external_fft->apply(x,y);return;}
                if(scheduler && gpu_batch2)scheduler->gpu_pairable(lane,1,x,y,[&]{op.matvec(x,y);},
                    [&](const void* a,void* b,const void* c,void* d){
                        op.matvec_batch2(static_cast<const Complex*>(a),static_cast<const Complex*>(c),
                                        static_cast<Complex*>(b),static_cast<Complex*>(d));});
                else if(scheduler)scheduler->gpu(lane,[&]{op.matvec(x,y);});else op.matvec(x,y);
            };
            const bem_streaming::Action strict=[&](const Complex* x,Complex* y){
                if(!scheduler){op.matvec_strict(x,y);return;}
                // The CPU task owns the exact candidate being checked. The
                // owning solver cannot advance until its original gate returns.
                const std::vector<Complex> snapshot(x,x+n);
                bem_streaming::HybridScheduler::Work host;
                if(cpu_checker)host=[&]{bem_cpu::configure_threads(cpu_threads,"BEM_CPU_AFFINITY");cpu_checker->apply(snapshot.data(),y);};
                scheduler->strict(lane,[&]{op.matvec_strict(snapshot.data(),y);},host);
            };
            const bem_streaming::Action inverse=[&](const Complex* x,Complex* y){
                if(ritz_correction)++packet.ritz_physical_calls;
                if(scheduler && gpu_batch2)scheduler->gpu_pairable(lane,2,x,y,[&]{perform_inverse(x,y);},
                    [&](const void* a,void* b,const void* c,void* d){
                        symmetric_circulant.apply_full_fft_symmetry_batch2(static_cast<const Complex*>(a),static_cast<const Complex*>(c),
                            static_cast<Complex*>(b),static_cast<Complex*>(d),circ_blend);});
                else if(scheduler)scheduler->gpu(lane,[&]{perform_inverse(x,y);});else perform_inverse(x,y);
            };
            bem_streaming::Action encoded_inverse;
            if(ritz_correction)encoded_inverse=[&](const Complex* a,Complex* b){
                ++packet.ritz_encoded_calls;
                if(scheduler)scheduler->gpu(lane,[&]{ritz_correction->apply_encoded(a,b);});
                else ritz_correction->apply_encoded(a,b);
            };
            bem_streaming::Action encoded_action;
            if(resident_composite)encoded_action=[&](const Complex* a,Complex* b){
                ++packet.resident_action_calls;
                if(scheduler)scheduler->gpu(lane,[&]{resident_composite->apply(a,b,circ_blend);});
                else resident_composite->apply(a,b,circ_blend);
            };
            const auto x=sector_solver?bem_tmatrix::solve_prism_sectors_with_ritz(action,strict,inverse,encoded_inverse,symmetry_sectors,
                rhs[0].data(),solution[0].data(),restart,tol,maxiter,false,true,
                operation==3 && sector_recycle_capacity?active_recycle:nullptr,&packet.sector_iterations_x,encoded_action):
                bem_streaming::gmres(action,strict,inverse,rhs[0].data(),solution[0].data(),n,restart,tol,maxiter,
                true,false,fft_inner?nullptr:&recycle,operation==3,fft_inner>0);
            // Operation 3 is reserved for production packets. Independent
            // reference columns and phase-code audits always use operation 1.
            if(symmetry_warm_start && operation==3 && x.residual<=tol) {
                warm=bem_tmatrix::symmetry_guess(symmetry_orbit,rhs[0],solution[0],rhs[1],solution[1]);
                std::printf("  [T symmetry warm start] rank=%d rhs_fit=%.6e amplification=%.3f fitted=%d seconds=%.3f\n",
                    warm.rank,warm.relative_rhs_error,warm.amplification,warm.fitted,warm.seconds);
                std::fflush(stdout);
            }
            const auto y=sector_solver?bem_tmatrix::solve_prism_sectors_with_ritz(action,strict,inverse,encoded_inverse,symmetry_sectors,
                rhs[1].data(),solution[1].data(),restart,tol,maxiter,warm.fitted,true,
                operation==3 && sector_recycle_capacity?active_recycle:nullptr,&packet.sector_iterations_y,encoded_action):
                bem_streaming::gmres(action,strict,inverse,rhs[1].data(),solution[1].data(),n,restart,tol,maxiter,
                true,warm.fitted,fft_inner?nullptr:&recycle,operation==3,fft_inner>0);
            recycle_update_seconds=x.recycle_seconds+y.recycle_seconds;
            host_action_seconds=x.action_seconds+y.action_seconds;
            host_strict_seconds=x.strict_seconds+y.strict_seconds;
            host_orthogonalization_seconds=x.orthogonalization_seconds+y.orthogonalization_seconds;
            iterations_x=x.iterations;iterations_y=y.iterations;
            strict_checks_x=x.strict_checks;strict_checks_y=y.strict_checks;
            deferred_checks_x=x.deferred_strict_checks;deferred_checks_y=y.deferred_strict_checks;
            result.iterations=std::max(x.iterations,y.iterations);
            result.seconds=x.seconds+y.seconds+warm.seconds;
            result.final_residual_x=x.residual;result.final_residual_y=y.residual;
            result.converged_x=x.residual<=tol;result.converged_y=y.residual<=tol;
        }
        packet.external_fft_calls=external_fft?external_fft->calls()-external_before:0;
    };
    const auto record_budget=[&](const Packet& packet){
        if(sector_budget && packet.operation==3)
            sector_budget->record_packet(packet.sector_iterations_x,packet.sector_iterations_y);
    };
    const auto freeze_budget=[&](){
        const auto start=Clock::now();
        if(!sector_budget || !sector_budget->freeze_after_batch())return 0.;
        const auto& plan=sector_budget->plan();
        if(plan.trained)bem_streaming::rebudget_sector_recycle(sector_recycle,symmetry_sectors.offsets,plan);
        std::printf("[sector rank budget] frozen after %u packets, trained=%d, entries=%zu/%zu\n",
            sector_budget->packets(),plan.trained,plan.entries,plan.entry_budget);
        std::fflush(stdout);return seconds(start);
    };
    const auto json_array=[](const std::array<int,12>& values){
        std::ostringstream out;out<<"[";for(int s=0;s<12;++s){if(s)out<<",";out<<values[s];}out<<"]";return out.str();
    };
    const auto finish_packet=[&](Packet& packet) {
        const double tol=packet.tolerance;
        auto& rhs=packet.rhs;auto& solution=packet.solution;auto& result=packet.result;auto& warm=packet.warm;
        const int32_t operation=packet.operation;
        auto& iterations_x=packet.iterations_x;
        auto& iterations_y=packet.iterations_y;
        auto& strict_checks_x=packet.strict_checks_x;
        auto& strict_checks_y=packet.strict_checks_y;
        auto& deferred_checks_x=packet.deferred_checks_x;
        auto& deferred_checks_y=packet.deferred_checks_y;
        auto& recycle_update_seconds=packet.recycle_update_seconds;
        auto& host_action_seconds=packet.host_action_seconds;
        auto& host_strict_seconds=packet.host_strict_seconds;
        auto& host_orthogonalization_seconds=packet.host_orthogonalization_seconds;
        const auto request_start=packet.started;const double rhs_seconds=packet.rhs_seconds;
        const auto check_start=Clock::now();
        const char* refinement=std::getenv("BEM_MIXED_ITERATIVE_REFINEMENT");
        const char* recheck=std::getenv("BEM_TMATRIX_VERIFY_RESIDUAL");
        const bool solver_strict=!device_solver || (refinement && std::strcmp(refinement,"0")!=0);
        const bool independent_check=!solver_strict || (recheck && std::strcmp(recheck,"0")!=0) ||
            (packet.hybrid_stats.cpu_checks && std::max(result.final_residual_x,result.final_residual_y)>.9*tol);
        // In strict refinement mode the final solver residual already comes
        // from a complete FP64 operator action, not the Arnoldi estimate.
        const double rx=independent_check ? relative_residual(op,solution[0],rhs[0]) : result.final_residual_x;
        const double ry=independent_check ? relative_residual(op,solution[1],rhs[1]) : result.final_residual_y;
        const double check_seconds=seconds(check_start);
        const auto farfield_start=Clock::now();
        std::vector<Complex> far_x,far_y;
        op.farfield_pair(solution[0].data(),solution[1].data(),directions,far_x,far_y);
        const double farfield_seconds=seconds(farfield_start);
        ++completed_pairs;
        // At ka=112 rank 16 is 1.54 GiB: avoid rewriting it for every RHS.
        const double recycle_checkpoint_seconds=completed_pairs==1 || completed_pairs%16==0?checkpoint_recycle():0.;
        rusage usage={}; getrusage(RUSAGE_SELF,&usage);
        std::ostringstream status;
        status<<std::setprecision(17)<<"{\"iterations\":"<<result.iterations<<",\"solve_s\":"<<result.seconds
              <<",\"requested_tolerance\":"<<tol
              <<",\"hybrid_accounted_worker_s\":"<<(packet.hybrid?.5*packet.batch_seconds+rhs_seconds+check_seconds+farfield_seconds+recycle_checkpoint_seconds:seconds(request_start))
              <<",\"hybrid_packet\":"<<(packet.hybrid?"true":"false")
              <<",\"hybrid_merge_recycle_s\":"<<packet.merge_seconds
              <<",\"hybrid_merge_recycle_candidates\":"<<packet.merge_candidates
              <<",\"sector_rebudget_s\":"<<packet.rebudget_seconds
              <<",\"sector_iterations_x\":"<<json_array(packet.sector_iterations_x)
              <<",\"sector_iterations_y\":"<<json_array(packet.sector_iterations_y)
              <<",\"sector_capacities_used\":"<<json_array(packet.sector_capacities_used)
              <<",\"sector_budget_frozen\":"<<(sector_budget && sector_budget->frozen()?"true":"false")
              <<",\"sector_budget_trained\":"<<(sector_budget && sector_budget->plan().trained?"true":"false")
              <<",\"sector_budget_observed_packets\":"<<(sector_budget?sector_budget->packets():0)
              <<",\"sector_budget_entry_limit\":"<<(sector_budget?sector_budget->plan().entry_budget:0)
              <<",\"sector_budget_entries\":"<<(sector_budget?sector_budget->plan().entries:0)
              <<",\"hybrid_batch_s\":"<<packet.batch_seconds
              <<",\"hybrid_cpu_checks\":"<<packet.hybrid_stats.cpu_checks
              <<",\"hybrid_gpu_checks\":"<<packet.hybrid_stats.gpu_checks
              <<",\"hybrid_cpu_action_s\":"<<packet.hybrid_stats.cpu_seconds
              <<",\"hybrid_gpu_paired_actions\":"<<packet.hybrid_stats.gpu_paired_actions
              <<",\"hybrid_gpu_paired_matvecs\":"<<packet.hybrid_stats.paired_kinds[1]
              <<",\"hybrid_gpu_paired_inverses\":"<<packet.hybrid_stats.paired_kinds[2]
              <<",\"hybrid_gpu_actions\":"<<packet.hybrid_stats.gpu_actions
              <<",\"hybrid_gpu_pair_wait_s\":"<<packet.hybrid_stats.pair_wait_seconds
              <<",\"hybrid_accounted_owner_queue_wait_s\":"<<packet.accounted_owner_wait
              <<",\"hybrid_accounted_cpu_check_wait_s\":"<<packet.accounted_cpu_wait
              <<",\"hybrid_accounted_cpu_tail_wait_s\":"<<packet.accounted_cpu_tail
              <<",\"sector_gmres\":"<<(sector_solver?"true":"false")
              <<",\"host_timings_available\":"<<(!device_solver?"true":"false")
              <<",\"host_krylov_action_s\":"<<host_action_seconds
              <<",\"host_strict_s\":"<<host_strict_seconds
              <<",\"physical_ritz_inverse_calls\":"<<packet.ritz_physical_calls
              <<",\"physical_ritz_encoded_calls\":"<<packet.ritz_encoded_calls
              <<",\"resident_action_calls\":"<<packet.resident_action_calls
              <<",\"compressed_fft_action_calls\":"<<packet.external_fft_calls
              <<",\"host_orthogonalization_s\":"<<host_orthogonalization_seconds
              <<",\"host_iterations_x\":"<<iterations_x<<",\"host_iterations_y\":"<<iterations_y
              <<",\"host_strict_checks_x\":"<<strict_checks_x<<",\"host_strict_checks_y\":"<<strict_checks_y
              <<",\"host_deferred_strict_checks_x\":"<<deferred_checks_x<<",\"host_deferred_strict_checks_y\":"<<deferred_checks_y
              <<",\"krylov_recycle_used\":"<<(operation==3 && (recycle_capacity>0 || sector_recycle_capacity>0)?"true":"false")
              <<",\"krylov_recycle_rank\":"<<(sector_solver?sector_recycle.rank():recycle.rank())
              <<",\"krylov_recycle_gib\":"<<(sector_solver?sector_recycle.storage_gib():recycle.storage_gib())
              <<",\"krylov_recycle_update_s\":"<<recycle_update_seconds
              <<",\"krylov_recycle_checkpoint_s\":"<<recycle_checkpoint_seconds
              <<",\"symmetry_warm_start_fitted\":"<<(warm.fitted?"true":"false")
              <<",\"symmetry_warm_start_requested\":"<<(symmetry_warm_start && operation==3?"true":"false")
              <<",\"symmetry_warm_start_rhs_error\":"<<warm.relative_rhs_error
              <<",\"symmetry_warm_start_s\":"<<warm.seconds
              <<",\"rhs_s\":"<<rhs_seconds<<",\"strict_residual_x\":"<<rx<<",\"strict_residual_y\":"<<ry
              <<",\"check_s\":"<<check_seconds<<",\"farfield_s\":"<<farfield_seconds
              <<",\"independent_residual_recheck\":"<<(independent_check ? "true" : "false")
              <<",\"worker_peak_rss_mb\":"<<usage.ru_maxrss/1024.
              <<",\"total_worker_s\":"<<seconds(request_start)<<"}";
        message(client,status.str());
        transfer(client,far_x.data(),far_x.size()*sizeof(Complex),true);
        transfer(client,far_y.data(),far_y.size()*sizeof(Complex),true);
    };
    bool profile_batch_finished=false;
    while(true) {
        int32_t operation=0;transfer(client,&operation,sizeof(operation),false);
        if(operation==0)break;
        if(operation==4) {
            int32_t count=0;transfer(client,&count,sizeof(count),false);
            if(!hybrid_requested||count!=2)throw std::runtime_error("unsupported hybrid batch");
            Packet first(n),second(n);
            for(Packet* packet:{&first,&second}) {
                int32_t kind=0;transfer(client,&kind,sizeof(kind),false);
                if(kind!=1 && kind!=2 && kind!=3 && kind!=5)throw std::runtime_error("invalid hybrid operation");
                receive_packet(*packet,kind);packet->hybrid=true;
            }
            bem_cpu::configure_threads(gpu_threads,"BEM_CPU_GPU_AFFINITY");
            *lane_recycle[0]=sector_recycle;
            const bool capture=profile_first_batch && !profile_batch_finished;
            if(capture && cudaProfilerStart()!=cudaSuccess)throw std::runtime_error("cannot start CUDA batch capture");
            bem_streaming::HybridScheduler scheduler(gpu_batch2?gpu_pair_wait:0,cpu_check_budget);const auto before=Clock::now();
            scheduler.run([&]{solve_packet(first,0,&scheduler,lane_recycle[0].get());},
                          [&]{solve_packet(second,1,&scheduler,lane_recycle[1].get());},true,
                          [&]{*lane_recycle[1]=*lane_recycle[0];});
            sector_recycle=*lane_recycle[1];
            if(merge_lane_recycle){
                const auto merge_start=Clock::now();
                const int added=sector_recycle.merge_from(*lane_recycle[0]);
                first.merge_seconds=second.merge_seconds=.5*seconds(merge_start);
                first.merge_candidates=second.merge_candidates=added;
            }
            record_budget(first);record_budget(second);
            first.rebudget_seconds=second.rebudget_seconds=.5*freeze_budget();
            first.batch_seconds=second.batch_seconds=seconds(before);
            // Split batch-wide times equally: summing packet records counts
            // each actual wait once, unlike per-lane paired-call participation.
            first.accounted_owner_wait=second.accounted_owner_wait=.5*scheduler.idle_statistics().empty_queue_seconds;
            first.accounted_cpu_wait=second.accounted_cpu_wait=.5*scheduler.idle_statistics().cpu_check_seconds;
            first.accounted_cpu_tail=second.accounted_cpu_tail=.5*scheduler.idle_statistics().cpu_tail_seconds;
            first.hybrid_stats=scheduler.statistics(0);second.hybrid_stats=scheduler.statistics(1);
            finish_packet(first);finish_packet(second);
            if(capture) {
                if(cudaProfilerStop()!=cudaSuccess)throw std::runtime_error("cannot stop CUDA batch capture");
                profile_batch_finished=true;
            }
        }else {
            Packet packet(n);receive_packet(packet,operation);
            solve_packet(packet,0,nullptr,&sector_recycle);
            record_budget(packet);packet.rebudget_seconds=freeze_budget();finish_packet(packet);
        }
    }
    checkpoint_recycle();
    mbj.cleanup_device(); op.cleanup(); close(client); unlink(socket_path.c_str());
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"T-matrix FMM worker: %s\n",error.what()); return 1;
}
