// Persistent matrix-free BEM worker. A private Unix socket carries two
// incident E/H fields at a time; stdout remains available for solver logs.
#include "muller_fmm.h"
#include "muller_mbj.h"
#include "muller_paired_gmres.h"
#include "quadrature.h"
#include "streaming_host_gmres.h"
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
    if(shape=="sphere") mesh=icosphere(1.,ref);
    else if(shape=="prism") mesh=regular_prism(6,1.,ref,1.,0,false,d6h_mesh);
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
    op.init(mesh,Complex(ka,0),ri,options,7,duffy_order,5,max_leaf,false,2,0,.96,(cache+".near").c_str(),3);
#ifdef BEM_DEFAULT_FMM_NEAR_FP32
    op.set_fmm_near_fp32(true);
#endif
    const double setup_seconds=seconds(setup_start);
    MullerMbjPreconditioner mbj;
    const auto mbj_start=Clock::now();
    const char* mbj_environment=std::getenv("BEM_TMATRIX_MBJ_NODES");
    const int mbj_nodes=mbj_environment?std::max(1,std::atoi(mbj_environment)):(op.system_dofs>=2000000?12:50);
    mbj.build_cached(op,mbj_nodes,0,cache+".mbj");
    const bool device_solver=op.device_matvec_available();
    if(device_solver) mbj.upload_device();
    const double mbj_seconds=seconds(mbj_start);
    const TriQuad quad=tri_quadrature(13);
    std::vector<double> positions;
    for(int e=0;e<static_cast<int>(op.mesh.elements.size());++e)
        for(int q=0;q<quad.npts;++q) {
            MullerFmmQuadraturePoint point;
            point.element=e; point.sample=evaluate_muller_frame(op.mesh,e,quad.pts[q][0],quad.pts[q][1]);
            point.weight=.5*quad.wts[q]*point.sample.jacobian;
            positions.push_back(point.sample.position.x); positions.push_back(point.sample.position.y); positions.push_back(point.sample.position.z);
        }
    double radius=0;
    for(const Vec3& v:mesh.verts) radius=std::max(radius,v.norm());
    std::ostringstream hello;
    const size_t sample_count=positions.size()/3;
    hello<<std::setprecision(17)<<"{\"system_dofs\":"<<op.system_dofs<<",\"rhs_points\":"<<sample_count
         <<",\"prism_dihedral_mesh\":"<<(shape=="prism" && d6h_mesh ? "true":"false")
         <<",\"duffy_order\":"<<duffy_order
         <<",\"directions\":"<<directions.size()<<",\"radius\":"<<radius
         <<",\"operator_setup_s\":"<<setup_seconds<<",\"mbj_setup_s\":"<<mbj_seconds
         <<",\"mbj_storage_mb\":"<<mbj.storage_megabytes()
         <<",\"solver_backend\":\""<<(device_solver?"paired_device_gmres":"host_gmres_gpu_banded_fmm")<<"\""
         <<",\"fmm_max_leaf\":"<<max_leaf<<",\"mbj_nodes\":"<<mbj_nodes
         <<",\"multilevel_fmm\":"<<(multilevel && std::strcmp(multilevel,"0")!=0?"true":"false")
         <<",\"fmm_depth\":"<<op.fmm_exterior.tree.max_level
         <<",\"fmm_order_exterior\":"<<op.fmm_exterior.p
         <<",\"fmm_order_interior\":"<<op.fmm_interior.p
         <<",\"near_field_fp32\":"<<(op.fmm_near_fp32 ? "true" : "false")
         <<",\"dense_matrix_formed\":false,\"resident_solution_columns\":2}";
    message(client,hello.str()); transfer(client,positions.data(),positions.size()*sizeof(double),true);
    const int n=op.system_dofs;
    std::vector<Complex> rhs[2]={std::vector<Complex>(n),std::vector<Complex>(n)};
    std::vector<Complex> solution[2]={std::vector<Complex>(n),std::vector<Complex>(n)};
    // Consume incident samples in bounded chunks; do not duplicate the full
    // surface frame cache or a two-polarization E/H slab on the worker.
    const size_t field_chunk=4096;
    std::vector<Complex> fields(field_chunk*6);
    while(true) {
        int32_t operation=0; transfer(client,&operation,sizeof(operation),false);
        if(operation==0) break;
        const auto request_start=Clock::now();
        if(operation==1) {
            std::fill(rhs[0].begin(),rhs[0].end(),Complex(0));
            std::fill(rhs[1].begin(),rhs[1].end(),Complex(0));
            for(int pol=0;pol<2;++pol)
                for(size_t begin=0;begin<sample_count;begin+=field_chunk) {
                    const size_t count=std::min(field_chunk,sample_count-begin);
                    transfer(client,fields.data(),count*6*sizeof(Complex),false);
                    for(size_t local=0;local<count;++local) {
                        const size_t index=begin+local;
                        const int element=index/quad.npts,q=index%quad.npts;
                        const auto sample=evaluate_muller_frame(op.mesh,element,quad.pts[q][0],quad.pts[q][1]);
                        const double weight=.5*quad.wts[q]*sample.jacobian;
                        const auto basis=evaluate_muller_basis(op.mesh,element,sample);
                        const Complex* f=fields.data()+local*6;
                        for(int a=0;a<basis.count;++a) {
                        const Vec3 t=basis.values[a].cross(sample.normal)*weight;
                        rhs[pol][basis.dofs[a]]-=t.x*f[0]+t.y*f[1]+t.z*f[2];
                        rhs[pol][op.current_dofs+basis.dofs[a]]+=t.x*f[3]+t.y*f[4]+t.z*f[5];
                        }
                    }
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
        if(device_solver) {
            result=solve_muller_paired_gmres_device(op,mbj,rhs[0].data(),rhs[1].data(),
                solution[0].data(),solution[1].data(),restart,tol,maxiter,true);
        } else {
            const bem_streaming::Action action=[&](const Complex* x,Complex* y){op.matvec(x,y);};
            const bem_streaming::Action strict=[&](const Complex* x,Complex* y){op.matvec_strict(x,y);};
            const bem_streaming::Action inverse=[&](const Complex* x,Complex* y){mbj.apply(x,y);};
            const auto x=bem_streaming::gmres(action,strict,inverse,rhs[0].data(),solution[0].data(),n,restart,tol,maxiter);
            const auto y=bem_streaming::gmres(action,strict,inverse,rhs[1].data(),solution[1].data(),n,restart,tol,maxiter);
            result.iterations=std::max(x.iterations,y.iterations);
            result.seconds=x.seconds+y.seconds;
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
        rusage usage={}; getrusage(RUSAGE_SELF,&usage);
        std::ostringstream status;
        status<<std::setprecision(17)<<"{\"iterations\":"<<result.iterations<<",\"solve_s\":"<<result.seconds
              <<",\"rhs_s\":"<<rhs_seconds<<",\"strict_residual_x\":"<<rx<<",\"strict_residual_y\":"<<ry
              <<",\"check_s\":"<<check_seconds<<",\"farfield_s\":"<<farfield_seconds
              <<",\"independent_residual_recheck\":"<<(independent_check ? "true" : "false")
              <<",\"worker_peak_rss_mb\":"<<usage.ru_maxrss/1024.
              <<",\"total_worker_s\":"<<seconds(request_start)<<"}";
        message(client,status.str());
        transfer(client,far_x.data(),far_x.size()*sizeof(Complex),true);
        transfer(client,far_y.data(),far_y.size()*sizeof(Complex),true);
    }
    mbj.cleanup_device(); op.cleanup(); close(client); unlink(socket_path.c_str());
    return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"T-matrix FMM worker: %s\n",error.what()); return 1;
}
