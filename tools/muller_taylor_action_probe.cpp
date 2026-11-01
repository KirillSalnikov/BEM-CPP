#include "muller_fmm_taylor.h"
#include "muller_fmm_cpu.h"
#include "matrix_free_taylor_moments.h"
#include "streaming_host_gmres.h"
#include "tmatrix_symmetry_warm_start.h"
#include <fstream>
#include <iomanip>
#include <random>

using C=std::complex<double>;using V=std::vector<C>;using Clock=std::chrono::steady_clock;
constexpr int Degree=4;
#ifdef BEM_TAYLOR_HOST_FIXTURE
// Standalone CPU oracle uses exact all-pair P2P metadata. CUDA is never called.
void HelmholtzFMM::cleanup(){if(d_multi_re||d_local_re||d_charges_re)std::abort();initialized=false;}
struct PrismFaceFFT::Impl{};
PrismFaceFFT::~PrismFaceFFT(){if(p)std::abort();}
PrismFaceFFT::HostView PrismFaceFFT::host_view()const{throw std::runtime_error("unexpected GPU face view in pure CPU probe");}
void initialize(MullerFmmOperator& op,int ref,double ka,int /*duffy*/){
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
    op.current_dofs=op.mesh.current_dofs();op.system_dofs=op.mesh.system_dofs();op.k_exterior=ka;op.k_interior=ka*C(1.3116);op.epsilon_interior=C(1.3116)*C(1.3116);
    for(int order:{7,13}){const auto rule=tri_quadrature(order);auto& out=order==7?op.quadrature:op.mass_quadrature;
        for(size_t t=0;t<op.mesh.elements.size();++t)for(int q=0;q<rule.npts;++q){MullerFmmQuadraturePoint p;p.element=t;
            p.sample=evaluate_muller_frame(op.mesh,t,rule.pts[q][0],rule.pts[q][1]);p.weight=.5*rule.wts[q]*p.sample.jacobian;out.push_back(p);}}
    for(auto* f:{&op.fmm_exterior,&op.fmm_interior}){f->initialized=true;f->k=f==&op.fmm_exterior?op.k_exterior:op.k_interior;
        f->Ns=f->Nt=op.quadrature.size();f->n_nodes=1;f->L=1;f->p=0;f->n_unique_transfers=0;f->p2p_enabled=true;f->minimum_m2l_level=1;f->maximum_m2l_level=0;
        f->tree.max_level=0;f->tree.nodes.resize(1);f->tree.nodes[0]={};f->tree.level_nodes={{0}};f->h_leaf_indices={0};
        f->h_src_id_offsets=f->h_tgt_id_offsets={0,f->Ns};f->h_src_ids_flat.resize(f->Ns);f->h_tgt_ids_flat.resize(f->Nt);
        std::iota(f->h_src_ids_flat.begin(),f->h_src_ids_flat.end(),0);std::iota(f->h_tgt_ids_flat.begin(),f->h_tgt_ids_flat.end(),0);
        f->squad.dirs={1,0,0};f->squad.weights={1};}
}
#else
void initialize(MullerFmmOperator& op,int ref,double ka,int duffy){
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    // A small diagnostic mesh still needs a sufficiently deep tree to test
    // M2L; the production large-mesh band split cannot be copied here.
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,duffy,5,10);
}
#endif
double relative(const V& a,const V& b){double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::norm(a[i]-b[i]);d+=std::norm(b[i]);}return std::sqrt(n/std::max(d,1e-280));}
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
int main(int argc,char** argv)try{
    if(argc<5||argc>8)throw std::invalid_argument("REF KA DENSE_A_FILE REPORT [DIRECT_TEMPLATE_CHECK] [MOMENT_SOLVE] [DUFFY]");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);if(ref<0||ref>2||ka<=0)throw std::invalid_argument("small action probe ref/ka");
    const int duffy=argc>=8?std::stoi(argv[7]):4;
    setvbuf(stdout,nullptr,_IOLBF,0);MullerFmmOperator op;initialize(op,ref,ka,duffy);const int n=op.system_dofs;
    size_t far_pairs=0;
    for(const auto* f:{&op.fmm_exterior,&op.fmm_interior})for(const auto& level:f->m2l_batches)far_pairs+=level.n_pairs;
#ifndef BEM_TAYLOR_HOST_FIXTURE
    if(!far_pairs)throw std::runtime_error("native Taylor test must exercise M2L interactions");
#endif
    auto start=Clock::now();bem_cpu::MullerTaylorAction<Degree> action(op,duffy);const double setup=elapsed(start);
    std::vector<V> input(Degree+1,V(n)),actual,expected(Degree+1,V(n));std::mt19937 rng(481);std::normal_distribution<double> normal;
    for(auto& v:input)for(auto& q:v)q=C(normal(rng),normal(rng));
    start=Clock::now();action.apply(input,actual);const double action_s=elapsed(start);
    std::ifstream file(argv[3],std::ios::binary|std::ios::ate);const size_t matrix_bytes=size_t(n)*n*sizeof(C);
    if(!file||size_t(file.tellg())%matrix_bytes||size_t(file.tellg())<size_t(Degree+1)*matrix_bytes)throw std::runtime_error("dense matrix dimensions/order");
    file.seekg(0);V matrix(size_t(n)*n);
    for(int q=0;q<=Degree;++q){file.read(reinterpret_cast<char*>(matrix.data()),matrix_bytes);if(!file)throw std::runtime_error("dense matrix read");
#pragma omp parallel for schedule(static)
        for(int row=0;row<n;++row)for(int j=q;j<=Degree;++j){C sum=0;for(int col=0;col<n;++col)sum+=matrix[size_t(row)*n+col]*input[j-q][col];expected[j][row]+=sum;}
    }
    std::vector<double> errors;for(int j=0;j<=Degree;++j){const double e=relative(actual[j],expected[j]);errors.push_back(e);
        std::printf("moment %d dense relative error %.6e\n",j,e);if(!std::isfinite(e)||e>2e-7)throw std::runtime_error("complete Muller Taylor action mismatch");}
    double strict_gpu_error=-1;
#ifndef BEM_TAYLOR_HOST_FIXTURE
    {V strict(n);op.matvec_strict(input[0].data(),strict.data());strict_gpu_error=relative(actual[0],strict);
        if(!std::isfinite(strict_gpu_error)||strict_gpu_error>2e-9)throw std::runtime_error("Taylor central action versus original strict GPU operator");}
#endif
    bem_tmatrix::D6hOrbit orbit;orbit.build(op.mesh);std::vector<int> vertical,sign;
    bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,vertical,sign);double symmetry_error=0,dense_symmetry_error=0,symmetry_oracle_error=0;
    for(int generator=0;generator<3;++generator){std::vector<V> transformed(Degree+1,V(n)),image;
        for(int j=0;j<=Degree;++j){if(generator<2)orbit.transform(input[j],transformed[j],generator==1,false);
            else for(int i=0;i<n;++i)transformed[j][vertical[i]]=double(sign[i])*input[j][i];}
        action.apply(transformed,image);
        std::vector<V> dense_image(Degree+1,V(n));file.clear();file.seekg(0);
        for(int q=0;q<=Degree;++q){file.read(reinterpret_cast<char*>(matrix.data()),matrix_bytes);if(!file)throw std::runtime_error("dense symmetry matrix read");
#pragma omp parallel for schedule(static)
            for(int row=0;row<n;++row)for(int j=q;j<=Degree;++j){C sum=0;for(int col=0;col<n;++col)sum+=matrix[size_t(row)*n+col]*transformed[j-q][col];dense_image[j][row]+=sum;}
        }
        for(int j=0;j<=Degree;++j){V desired(n);if(generator<2)orbit.transform(actual[j],desired,generator==1,true);
            else for(int i=0;i<n;++i)desired[vertical[i]]=-double(sign[i])*actual[j][i];
            symmetry_error=std::max(symmetry_error,relative(image[j],desired));
            if(generator<2)orbit.transform(expected[j],desired,generator==1,true);
            else for(int i=0;i<n;++i)desired[vertical[i]]=-double(sign[i])*expected[j][i];
            dense_symmetry_error=std::max(dense_symmetry_error,relative(dense_image[j],desired));
            symmetry_oracle_error=std::max(symmetry_oracle_error,relative(image[j],dense_image[j]));}
    }
    std::printf("full D6h moment action defect %.6e; dense defect %.6e; transformed action oracle error %.6e\n",symmetry_error,dense_symmetry_error,symmetry_oracle_error);
    if(!std::isfinite(symmetry_error)||symmetry_error>2e-5)throw std::runtime_error("Muller Taylor D6h action defect");
    double template_error=-1,direct_setup=-1;
    if(argc>=6&&std::stoi(argv[5])){start=Clock::now();bem_cpu::MullerTaylorAction<Degree> direct(op,duffy,false);direct_setup=elapsed(start);
        std::vector<V> answer;direct.apply(input,answer);template_error=0;for(int j=0;j<=Degree;++j)template_error=std::max(template_error,relative(actual[j],answer[j]));
        if(!std::isfinite(template_error)||template_error>2e-9)throw std::runtime_error("derivative correction template orientation mismatch");}
    double moment_error=-1,moment_seconds=-1,moment_residual=-1;int iterations=0,recycle_rank=0;
    if(argc>=7&&std::stoi(argv[6])){
        // Reuse one block-Jacobi preconditioner and one recycle space. In this
        // small oracle only, local blocks are extracted from the saved dense
        // A0; no dense derivative is used by the candidate recurrence.
        file.clear();file.seekg(0);file.read(reinterpret_cast<char*>(matrix.data()),matrix_bytes);
        struct Block{int size;V factor;std::vector<int> indices,pivots;};std::vector<Block> blocks;
        for(int first=0;first<op.current_dofs;first+=16){Block b;const int count=std::min(16,op.current_dofs-first);b.size=2*count;b.factor.resize(b.size*b.size);
            for(int c=0;c<2;++c)for(int j=0;j<count;++j)b.indices.push_back(c*op.current_dofs+first+j);
            for(int i=0;i<b.size;++i)for(int j=0;j<b.size;++j)b.factor[i*b.size+j]=matrix[size_t(b.indices[i])*n+b.indices[j]];
            if(!bem_krylov::dense_lu_factor(b.factor,b.size,b.pivots))throw std::runtime_error("moment local LU");
            blocks.push_back(std::move(b));}
        matrix.clear();matrix.shrink_to_fit();
        // Supply the ordinary A0 action with the independently verified order
        // zero sparse correction; its physical normalization is unscaled K1.
        op.correction.row_offsets=action.correction.offsets;op.correction.entries.clear();
        for(auto e:action.correction.entries){MullerNearCorrectionEntry v;v.column=e.column;v.k1=C(0,-ka)*e.value.diagonal[0];
            v.k2_epsilon=e.value.epsilon[0];v.k2_mu=e.value.mu[0];op.correction.entries.push_back(v);}
        bem_cpu::MullerAction central(op,0);bem_streaming::RecycleSpace recycle(n,8);
        const bem_streaming::Action a0=[&](const C* x,C* y){central.apply(x,y);};
        const bem_streaming::Action inverse=[&](const C* x,C* y){for(const auto& b:blocks){V local(b.size);for(int j=0;j<b.size;++j)local[j]=x[b.indices[j]];
            bem_krylov::dense_lu_solve(b.factor,b.pivots,b.size,local.data());for(int j=0;j<b.size;++j)y[b.indices[j]]=local[j];}};
        start=Clock::now();const auto solved=bem_taylor::solve_moments(expected,[&](const std::vector<V>& x,std::vector<V>& y){action.apply(x,y);},
            [&](const V& rhs,V& x){const auto result=bem_streaming::gmres(a0,a0,inverse,rhs.data(),x.data(),n,40,1e-11,400,false,false,&recycle);
                iterations+=result.iterations;if(result.residual>1e-11)throw std::runtime_error("central moment GMRES did not converge");},a0,2e-11);
        moment_seconds=elapsed(start);moment_error=moment_residual=0;recycle_rank=recycle.rank();
        for(int j=0;j<=Degree;++j){moment_error=std::max(moment_error,relative(solved.current[j],input[j]));moment_residual=std::max(moment_residual,solved.original_rhs_residuals[j]);}
        if(moment_error>2e-8)throw std::runtime_error("matrix-free moments differ from dense manufactured solution");
        std::printf("PASS recycled central solves: %d iterations, rank %d, moment error %.3e, %.3f s\n",iterations,recycle_rank,moment_error,moment_seconds);
    }
    auto alias=input;action.apply(alias,alias);double alias_error=0;for(int j=0;j<=Degree;++j)alias_error=std::max(alias_error,relative(alias[j],actual[j]));
    if(alias_error>1e-12)throw std::runtime_error("complete Muller Taylor aliased buffer");
    for(auto& v:input)std::fill(v.begin(),v.end(),C(0));
    action.apply(input,alias);
    for(const auto& v:alias)for(C q:v)if(q!=C(0))throw std::runtime_error("complete Muller Taylor zero buffer");
    std::ofstream out(argv[4]);out<<std::setprecision(17)<<"{\"passed\":true,\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"degree\":"<<Degree<<",\"duffy\":"<<duffy<<",\"dofs\":"<<n
        <<",\"setup_s\":"<<setup<<",\"action_s\":"<<action_s<<",\"workspace_bytes\":"<<action.workspace_bytes()<<",\"sparse_correction_bytes\":"<<action.correction.storage_bytes()
        <<",\"adjacent_pairs\":"<<action.correction.pairs<<",\"unique_templates\":"<<action.correction.templates<<",\"template_error\":"<<template_error
        <<",\"direct_setup_s\":"<<direct_setup<<",\"alias_error\":"<<alias_error<<",\"central_correction_error\":"<<action.central_correction_error
        <<",\"moment_error\":"<<moment_error<<",\"moment_seconds\":"<<moment_seconds<<",\"moment_rhs_residual\":"<<moment_residual
        <<",\"moment_iterations\":"<<iterations<<",\"recycle_rank\":"<<recycle_rank<<",\"D6h_defect\":"<<symmetry_error
        <<",\"dense_D6h_defect\":"<<dense_symmetry_error<<",\"symmetry_oracle_error\":"<<symmetry_oracle_error
        <<",\"strict_gpu_error\":"<<strict_gpu_error<<",\"far_pairs\":"<<far_pairs<<",\"dense_relative_errors\":[";
    for(size_t j=0;j<errors.size();++j)out<<(j?",":"")<<errors[j];
    out<<"],\"production_enabled\":false,\"host_fixture\":";
#ifdef BEM_TAYLOR_HOST_FIXTURE
    out<<"true";
#else
    out<<"false";
    op.cleanup();
#endif
    out<<"}\n";std::puts("PASS complete matrix-free Muller moment action");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
