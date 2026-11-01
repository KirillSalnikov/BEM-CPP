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
#include "sector_solver_experiment.h"
#include "prism_vertical_inverse.h"
#include "prism_symmetric_circulant.h"
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
    const unsigned sector_recycle_mask=sector_integer("BEM_TMATRIX_SECTOR_RECYCLE_MASK",4095,4095);
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
    if(sector_solver && (device_solver || shape!="prism" || !d6h_mesh || fft_inner || recycle_capacity || mbj_overlap || orbit_nodes<1 || orbit_nodes>512))
        throw std::runtime_error("sector GMRES requires the host D6h prism, orbit blocks 1..512, and no overlap/inner FFT/recycling");
    bem_tmatrix::D6hOrbit symmetry_orbit;
    bem_tmatrix::PrismSymmetrySectors symmetry_sectors;
    PrismOrbitInverse orbit_inverse;
    if(sector_solver) {
        symmetry_orbit.build(op.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(symmetry_orbit);
        symmetry_sectors.build(maps);orbit_inverse.build(op,maps,orbit_nodes);
        std::printf("[sector solver] orbit_nodes=%d local_MiB=%.3f transform_MiB=%.3f\n",orbit_nodes,
            orbit_inverse.storage_bytes()/1048576.,symmetry_sectors.storage_bytes()/1048576.);
    }else mbj.build_cached(op,mbj_nodes,mbj_overlap,cache+".mbj");
    // This optional accelerator stays in RAM between production packets.
    // Column checkpoints are independent; a restarted worker relearns it.
    bem_streaming::ExperimentalSectorRecycleSpace sector_recycle(symmetry_sectors.offsets,sector_recycle_capacity,sector_recycle_mask);
    const char* circ_setting=std::getenv("BEM_TMATRIX_CIRCULANT_BLEND");
    const double circ_blend=circ_setting?std::stod(circ_setting):0.;
    if(!std::isfinite(circ_blend) || circ_blend<0 || circ_blend>1)
        throw std::runtime_error("circulant blend must be in [0,1]");
    PrismCirculantInverse circulant;
    double circulant_seconds=0;
    if(circ_blend>0) {
        if(device_solver || shape!="prism" || !d6h_mesh)
            throw std::runtime_error("circulant requires host GMRES and a regular D6h prism");
        const auto before=Clock::now();
        const double shift=std::getenv("BEM_CIRCULANT_SHIFT")?std::stod(std::getenv("BEM_CIRCULANT_SHIFT")):0;
        const double opposite=std::getenv("BEM_CIRCULANT_OPPOSITE_WEIGHT")?std::stod(std::getenv("BEM_CIRCULANT_OPPOSITE_WEIGHT")):0;
        const int opposite_mask=sector_integer("BEM_CIRCULANT_OPPOSITE_MASK",3,3);
        circulant.build(op.mesh,ref,[&](const Complex* a,Complex* b){op.matvec(a,b);},
            [&](const Complex* a,Complex* b){if(sector_solver)orbit_inverse.apply(a,b);else mbj.apply(a,b);},shift,opposite,opposite_mask);
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
    const bool symmetry_warm_start=!device_solver && shape=="prism" && d6h_mesh &&
        warm_setting && std::strcmp(warm_setting,"0")!=0;
    if(symmetry_warm_start && !sector_solver) symmetry_orbit.build(op.mesh);
    const double mbj_seconds=seconds(mbj_start);
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
         <<",\"prism_height_to_width_across_flats\":"<<prism_aspect
         <<",\"duffy_order\":"<<duffy_order
         <<",\"rhs_assembly\":\"cached_element_chunks\""
         <<",\"rhs_chunk_elements\":"<<rhs_assembler.chunk_points()/quad.npts
         <<",\"rhs_threads_max\":"<<rhs_assembler.thread_limit()
         <<",\"directions\":"<<directions.size()<<",\"radius\":"<<radius
         <<",\"operator_setup_s\":"<<setup_seconds<<",\"mbj_setup_s\":"<<mbj_seconds
         <<",\"mbj_storage_mb\":"<<mbj.storage_megabytes()
         <<",\"sector_gmres\":"<<(sector_solver?"true":"false")<<",\"orbit_nodes\":"<<(sector_solver?orbit_nodes:0)
         <<",\"gpu_operator_assembly\":"<<(op.gpu_operator_assembly?"true":"false")
         <<",\"gpu_host_correction\":"<<(op.gpu_host_correction?"true":"false")
         <<",\"sector_recycle_capacity\":"<<sector_recycle_capacity<<",\"sector_recycle_mask\":"<<sector_recycle_mask
         <<",\"sector_recycle_disk_checkpoint\":false"
         <<",\"sector_first_pass_factor\":"<<sector_first_pass_factor
         <<",\"vertical_preconditioner\":"<<(vertical_preconditioner?"true":"false")
         <<",\"orbit_storage_mib\":"<<orbit_inverse.storage_bytes()/1048576.
         <<",\"fast_fft_patch_orbits\":"<<(symmetric_circulant.fast_patch_orbits()?"true":"false")
         <<",\"native_full_fft_symmetry\":"<<(symmetric_circulant.full_fft_symmetry()?"true":"false")
         <<",\"circulant_blend\":"<<circ_blend<<",\"circulant_setup_s\":"<<circulant_seconds
         <<",\"circulant_storage_mib\":"<<circulant.storage_bytes()/1048576.
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
         <<",\"dense_matrix_formed\":false,\"resident_solution_columns\":2}";
    message(client,hello.str()); transfer(client,positions.data(),positions.size()*sizeof(double),true);
    std::vector<double>().swap(positions);
    const int n=op.system_dofs;
    std::vector<Complex> rhs[2]={std::vector<Complex>(n),std::vector<Complex>(n)};
    std::vector<Complex> solution[2]={std::vector<Complex>(n),std::vector<Complex>(n)};
    // Consume incident samples in bounded chunks; do not duplicate the full
    // surface frame cache or a two-polarization E/H slab on the worker.
    const size_t field_chunk=rhs_assembler.chunk_points();
    std::vector<Complex> fields(field_chunk*6);
    int completed_pairs=0;
    while(true) {
        int32_t operation=0; transfer(client,&operation,sizeof(operation),false);
        if(operation==0) break;
        const auto request_start=Clock::now();
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
        const double rhs_seconds=seconds(request_start);
        for(auto& vector:solution) std::fill(vector.begin(),vector.end(),Complex(0));
        MullerPairedGmresResult result;
        bem_tmatrix::WarmStartResult warm;
        int iterations_x=0,iterations_y=0;
        int strict_checks_x=0,strict_checks_y=0,deferred_checks_x=0,deferred_checks_y=0;
        double recycle_update_seconds=0;
        double host_action_seconds=0,host_strict_seconds=0,host_orthogonalization_seconds=0;
        if(device_solver) {
            result=solve_muller_paired_gmres_device(op,mbj,rhs[0].data(),rhs[1].data(),
                solution[0].data(),solution[1].data(),restart,tol,maxiter,true);
        } else {
            const bem_streaming::Action action=[&](const Complex* x,Complex* y){op.matvec(x,y);};
            const bem_streaming::Action strict=[&](const Complex* x,Complex* y){op.matvec_strict(x,y);};
            const bem_streaming::Action inverse=[&](const Complex* x,Complex* y){
                if(sector_solver){if(symmetric_circulant.full_fft_symmetry())symmetric_circulant.apply_full_fft_symmetry(x,y,circ_blend);else if(vertical_preconditioner)vertical_inverse.apply(x,y);else if(circ_blend>0)symmetric_circulant.apply(x,y,circ_blend);else orbit_inverse.apply(x,y);}
                else if(fft_inner)fft_inverse->solve(x,y,.1);else if(wave_rank)wave.apply(x,y);else if(circ_blend>0)circulant.apply(x,y,circ_blend);else mbj.apply(x,y);};
            const auto x=sector_solver?bem_tmatrix::solve_prism_sectors_experiment(action,strict,inverse,symmetry_sectors,
                rhs[0].data(),solution[0].data(),restart,tol,maxiter,false,true,
                operation==3 && sector_recycle_capacity?&sector_recycle:nullptr):
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
            const auto y=sector_solver?bem_tmatrix::solve_prism_sectors_experiment(action,strict,inverse,symmetry_sectors,
                rhs[1].data(),solution[1].data(),restart,tol,maxiter,warm.fitted,true,
                operation==3 && sector_recycle_capacity?&sector_recycle:nullptr):
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
        const auto check_start=Clock::now();
        const char* refinement=std::getenv("BEM_MIXED_ITERATIVE_REFINEMENT");
        const char* recheck=std::getenv("BEM_TMATRIX_VERIFY_RESIDUAL");
        const bool solver_strict=!device_solver || (refinement && std::strcmp(refinement,"0")!=0);
        const bool independent_check=!solver_strict || (recheck && std::strcmp(recheck,"0")!=0);
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
              <<",\"sector_gmres\":"<<(sector_solver?"true":"false")
              <<",\"host_timings_available\":"<<(!device_solver?"true":"false")
              <<",\"host_krylov_action_s\":"<<host_action_seconds
              <<",\"host_strict_s\":"<<host_strict_seconds
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
    }
    checkpoint_recycle();
    mbj.cleanup_device(); op.cleanup(); close(client); unlink(socket_path.c_str());
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"T-matrix FMM worker: %s\n",error.what()); return 1;
}
