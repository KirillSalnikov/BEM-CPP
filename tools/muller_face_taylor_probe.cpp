// Combined same-face FFT / cross-face FMM Taylor control, isolated from the old native probe.
#include "muller_fmm_taylor.h"
#include "muller_fmm_cpu.h"
#include "host_taylor_central_config.h"
#include "matrix_free_taylor_moments.h"
#include "streaming_host_gmres.h"
#include "tmatrix_symmetry_warm_start.h"
#include <fstream>
#include <iomanip>
#include <random>

using C=std::complex<double>;using V=std::vector<C>;using Clock=std::chrono::steady_clock;
constexpr int Degree=4;
#ifdef BEM_TAYLOR_HOST_FIXTURE
#include "../tests/physical_face_cpu_fixture.h"
void initialize(MullerFmmOperator& op,int ref,double ka,int /*duffy*/){
    initialize_physical_face_fixture(op,ref,ka);
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
    if(argc<5||argc>8)throw std::invalid_argument("REF KA DENSE_A_FILE REPORT [DIRECT_TEMPLATE_CHECK] [MOMENT_SOLVE:0|1=history|2=incremental] [DUFFY]");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);if(ref<1||ref>2||ka<=0)throw std::invalid_argument("small action probe ref/ka");
    const int duffy=argc>=8?std::stoi(argv[7]):4;
    const int moment_mode=argc>=7?std::stoi(argv[6]):0;
    if(moment_mode<0||moment_mode>2)throw std::invalid_argument("Taylor moment solve mode");
    const auto central_config=bem_cpu::TaylorCentralConfig::from_environment();
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(!bem_cpu::FftwRuntime::get().available())throw std::runtime_error("Taylor face probe requires the validated FFTW library");
    MullerFmmOperator op;initialize(op,ref,ka,duffy);
#ifndef BEM_TAYLOR_HOST_FIXTURE
    op.enable_prism_face_fft(ref,10./7);
#endif
    const int n=op.system_dofs;
    size_t far_pairs=0;
    for(const auto* f:{&op.fmm_exterior,&op.fmm_interior})for(const auto& level:f->m2l_batches)far_pairs+=level.n_pairs;
    size_t cross_far_pairs=0;
    for(const auto* face_operator:{op.prism_fft_exterior.get(),op.prism_fft_interior.get()}){
        const auto view=face_operator->host_view();
        for(int face=0;face<8;++face)if(view.faces[face].engine==face)
            for(const auto* f:{view.faces[face].fine,view.faces[face].middle,view.faces[face].coarse})
                for(const auto& level:f->m2l_batches)cross_far_pairs+=level.n_pairs;
    }
#ifndef BEM_TAYLOR_HOST_FIXTURE
    if(!cross_far_pairs)throw std::runtime_error("combined Taylor test must exercise cross-face M2L");
    if(!far_pairs)throw std::runtime_error("native Taylor test must exercise M2L interactions");
#endif
    auto start=Clock::now();bem_cpu::MullerTaylorAction<Degree> action(op,duffy,true,true);const double setup=elapsed(start);
    std::vector<V> input(Degree+1,V(n)),actual,expected(Degree+1,V(n)),single_expected(Degree+1,V(n));std::mt19937 rng(481);std::normal_distribution<double> normal;
    for(auto& v:input)for(auto& q:v)q=C(normal(rng),normal(rng));
    start=Clock::now();action.apply(input,actual);const double action_s=elapsed(start);
    const double self_s=action.face_fft_seconds,cross_s=action.face_cross_seconds,packing_s=action.face_packing_seconds;
    std::ifstream file(argv[3],std::ios::binary|std::ios::ate);const size_t matrix_bytes=size_t(n)*n*sizeof(C);
    if(!file||size_t(file.tellg())%matrix_bytes||size_t(file.tellg())<size_t(Degree+1)*matrix_bytes)throw std::runtime_error("dense matrix dimensions/order");
    file.seekg(0);V matrix(size_t(n)*n);
    for(int q=0;q<=Degree;++q){file.read(reinterpret_cast<char*>(matrix.data()),matrix_bytes);if(!file)throw std::runtime_error("dense matrix read");
#pragma omp parallel for schedule(static)
        for(int row=0;row<n;++row)for(int j=q;j<=Degree;++j){C sum=0;for(int col=0;col<n;++col)sum+=matrix[size_t(row)*n+col]*input[j-q][col];expected[j][row]+=sum;if(j==q)single_expected[q][row]=sum;}
    }
    std::vector<double> errors;for(int j=0;j<=Degree;++j){const double e=relative(actual[j],expected[j]);errors.push_back(e);
        std::printf("moment %d dense relative error %.6e\n",j,e);if(!std::isfinite(e)||e>2e-7)throw std::runtime_error("complete Muller Taylor action mismatch");}
    double prefix_error=0,prefix_alias_error=0;
    for(size_t maximum=0;maximum<=Degree;++maximum){std::vector<V> prefix;action.apply_current(input[0],maximum,prefix);
        if(prefix.size()!=maximum+1)throw std::runtime_error("physical single-current output shape");
        for(size_t q=0;q<=maximum;++q)prefix_error=std::max(prefix_error,relative(prefix[q],single_expected[q]));}
    auto prefix_alias=std::vector<V>{input[0]};action.apply_current(prefix_alias[0],Degree,prefix_alias);
    for(size_t q=0;q<=Degree;++q)prefix_alias_error=std::max(prefix_alias_error,relative(prefix_alias[q],single_expected[q]));
    if(!std::isfinite(prefix_error+prefix_alias_error)||std::max(prefix_error,prefix_alias_error)>2e-7)
        throw std::runtime_error("physical single-current derivative prefix mismatch");
    std::printf("single-current prefixes0..4 dense %.6e, aliased %.6e\n",prefix_error,prefix_alias_error);
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
    if(!std::isfinite(symmetry_oracle_error)||symmetry_oracle_error>2e-7)throw std::runtime_error("transformed Taylor action differs from dense oracle");
    double template_error=-1,direct_setup=-1;
    if(argc>=6&&std::stoi(argv[5])){start=Clock::now();bem_cpu::MullerTaylorAction<Degree> direct(op,duffy,false,true);direct_setup=elapsed(start);
        std::vector<V> answer;direct.apply(input,answer);template_error=0;for(int j=0;j<=Degree;++j)template_error=std::max(template_error,relative(actual[j],answer[j]));
        if(!std::isfinite(template_error)||template_error>2e-9)throw std::runtime_error("derivative correction template orientation mismatch");}
    double moment_error=-1,moment_seconds=-1,moment_residual=-1,dense_moment_residual=-1;int iterations=0,recycle_rank=0;
    double moment_central_s=0,moment_central_fft_s=0,moment_central_cross_s=0,moment_central_packing_s=0;
    double moment_central_project_s=0,moment_central_test_s=0,moment_inverse_s=0;
    double moment_derivative_s=0,moment_derivative_fft_s=0,moment_derivative_cross_s=0,moment_derivative_packing_s=0;
    size_t moment_central_calls=0,moment_derivative_calls=0,moment_derivative_coefficients=0;
    size_t central_workspace=0,central_estimated_workspace=0;int central_shared_bands=0;
    if(moment_mode){
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
        central_estimated_workspace=bem_cpu::MullerAction::estimated_workspace_bytes(op,0,true,true,central_config.batch,central_config.tile);
        bem_cpu::MullerAction central(op,0,true,true,central_config.batch,central_config.tile,central_config.shared);
        central_workspace=central.workspace_bytes();central_shared_bands=central.shared_avx512_near_bands;
        if(central_workspace>central_estimated_workspace)throw std::runtime_error("Taylor central numerical workspace underestimated");
        if(central_config.shared&&central_shared_bands==0)throw std::runtime_error("Taylor shared central kernel not exercised");
        bem_streaming::RecycleSpace recycle(n,8);
        const bem_streaming::Action a0=[&](const C* x,C* y){auto begin=Clock::now();central.apply(x,y);
            moment_central_s+=elapsed(begin);++moment_central_calls;
            moment_central_fft_s+=central.face_fft_s;moment_central_cross_s+=central.face_cross_s;
            moment_central_packing_s+=central.face_packing_s;moment_central_project_s+=central.project_s;moment_central_test_s+=central.test_s;};
        const bem_streaming::Action inverse=[&](const C* x,C* y){auto begin=Clock::now();for(const auto& b:blocks){V local(b.size);for(int j=0;j<b.size;++j)local[j]=x[b.indices[j]];
            bem_krylov::dense_lu_solve(b.factor,b.pivots,b.size,local.data());for(int j=0;j<b.size;++j)y[b.indices[j]]=local[j];}moment_inverse_s+=elapsed(begin);};
        const bem_taylor::CentralMomentSolve solve=[&](const V& rhs,V& x){
            const auto result=bem_streaming::gmres(a0,a0,inverse,rhs.data(),x.data(),n,40,1e-11,400,false,false,&recycle);
            iterations+=result.iterations;if(result.residual>1e-11)throw std::runtime_error("central moment GMRES did not converge");};
        const auto account_derivative=[&](Clock::time_point begin,size_t coefficients){
            moment_derivative_s+=elapsed(begin);++moment_derivative_calls;moment_derivative_coefficients+=coefficients;
            moment_derivative_fft_s+=action.face_fft_seconds;moment_derivative_cross_s+=action.face_cross_seconds;moment_derivative_packing_s+=action.face_packing_seconds;};
        start=Clock::now();const auto solved=moment_mode==2
            ?bem_taylor::solve_moments_incremental(expected,[&](const V& x,size_t maximum,std::vector<V>& y){auto begin=Clock::now();action.apply_current(x,maximum,y);account_derivative(begin,maximum+1);},solve,a0,2e-11)
            :bem_taylor::solve_moments(expected,[&](const std::vector<V>& x,std::vector<V>& y){auto begin=Clock::now();action.apply(x,y);account_derivative(begin,Degree+1);},solve,a0,2e-11);
        moment_seconds=elapsed(start);moment_error=moment_residual=0;recycle_rank=recycle.rank();
        for(int j=0;j<=Degree;++j){moment_error=std::max(moment_error,relative(solved.current[j],input[j]));moment_residual=std::max(moment_residual,solved.original_rhs_residuals[j]);}
        if(moment_error>2e-8)throw std::runtime_error("matrix-free moments differ from dense manufactured solution");
        // Recheck the complete original block system with the separate dense
        // coefficients, not just A0 against the recursively updated RHS.
        std::vector<V> dense_action(Degree+1,V(n));matrix.resize(size_t(n)*n);file.clear();file.seekg(0);
        for(int q=0;q<=Degree;++q){file.read(reinterpret_cast<char*>(matrix.data()),matrix_bytes);
            if(!file)throw std::runtime_error("dense full moment residual read");
#pragma omp parallel for schedule(static)
            for(int row=0;row<n;++row)for(int j=q;j<=Degree;++j){C sum=0;
                for(int col=0;col<n;++col)sum+=matrix[size_t(row)*n+col]*solved.current[j-q][col];
                dense_action[j][row]+=sum;}}
        dense_moment_residual=0;for(int j=0;j<=Degree;++j)dense_moment_residual=std::max(dense_moment_residual,relative(dense_action[j],expected[j]));
        if(!std::isfinite(dense_moment_residual)||dense_moment_residual>2e-9)throw std::runtime_error("full original dense moment RHS residual failed");
        std::printf("PASS recycled central solves: %d iterations, rank %d, moment error %.3e, %.3f s\n",iterations,recycle_rank,moment_error,moment_seconds);
    }
    auto alias=input;action.apply(alias,alias);double alias_error=0;for(int j=0;j<=Degree;++j)alias_error=std::max(alias_error,relative(alias[j],actual[j]));
    if(alias_error>1e-12)throw std::runtime_error("complete Muller Taylor aliased buffer");
    for(auto& v:input)std::fill(v.begin(),v.end(),C(0));
    action.apply(input,alias);
    for(const auto& v:alias)for(C q:v)if(q!=C(0))throw std::runtime_error("complete Muller Taylor zero buffer");
    std::ofstream out(argv[4]);out<<std::setprecision(17)<<"{\"passed\":true,\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"degree\":"<<Degree<<",\"duffy\":"<<duffy<<",\"dofs\":"<<n
        <<",\"moment_mode\":"<<moment_mode<<",\"single_current_prefix_error\":"<<prefix_error<<",\"single_current_alias_error\":"<<prefix_alias_error
        <<",\"face_fft\":true,\"face_self_s\":"<<self_s<<",\"face_cross_s\":"<<cross_s<<",\"face_packing_s\":"<<packing_s
        <<",\"setup_s\":"<<setup<<",\"action_s\":"<<action_s<<",\"workspace_bytes\":"<<action.workspace_bytes()<<",\"sparse_correction_bytes\":"<<action.correction.storage_bytes()
        <<",\"adjacent_pairs\":"<<action.correction.pairs<<",\"unique_templates\":"<<action.correction.templates<<",\"template_error\":"<<template_error
        <<",\"direct_setup_s\":"<<direct_setup<<",\"alias_error\":"<<alias_error<<",\"central_correction_error\":"<<action.central_correction_error
        <<",\"moment_error\":"<<moment_error<<",\"moment_seconds\":"<<moment_seconds<<",\"moment_rhs_residual\":"<<moment_residual
        <<",\"full_dense_moment_rhs_residual\":"<<dense_moment_residual
        <<",\"central_backend\":\""<<central_config.backend<<"\",\"central_tile\":"<<central_config.tile
        <<",\"central_workspace_bytes\":"<<central_workspace<<",\"central_estimated_workspace_bytes\":"<<central_estimated_workspace
        <<",\"central_shared_AVX512_bands\":"<<central_shared_bands
        <<",\"moment_central_calls\":"<<moment_central_calls<<",\"moment_central_s\":"<<moment_central_s
        <<",\"moment_central_fft_s\":"<<moment_central_fft_s<<",\"moment_central_cross_s\":"<<moment_central_cross_s
        <<",\"moment_central_packing_s\":"<<moment_central_packing_s<<",\"moment_central_project_s\":"<<moment_central_project_s
        <<",\"moment_central_test_s\":"<<moment_central_test_s<<",\"moment_inverse_s\":"<<moment_inverse_s
        <<",\"moment_derivative_calls\":"<<moment_derivative_calls<<",\"moment_derivative_coefficients\":"<<moment_derivative_coefficients
        <<",\"moment_derivative_s\":"<<moment_derivative_s<<",\"moment_derivative_fft_s\":"<<moment_derivative_fft_s
        <<",\"moment_derivative_cross_s\":"<<moment_derivative_cross_s<<",\"moment_derivative_packing_s\":"<<moment_derivative_packing_s
        <<",\"moment_iterations\":"<<iterations<<",\"recycle_rank\":"<<recycle_rank<<",\"D6h_defect\":"<<symmetry_error
        <<",\"dense_D6h_defect\":"<<dense_symmetry_error<<",\"symmetry_oracle_error\":"<<symmetry_oracle_error
        <<",\"strict_gpu_error\":"<<strict_gpu_error<<",\"far_pairs\":"<<far_pairs<<",\"cross_far_pairs\":"<<cross_far_pairs<<",\"dense_relative_errors\":[";
    for(size_t j=0;j<errors.size();++j)out<<(j?",":"")<<errors[j];
    out<<"],\"production_enabled\":false,\"host_fixture\":";
#ifdef BEM_TAYLOR_HOST_FIXTURE
    out<<"true,\"GPU_used\":false,\"native_far_tree_verified\":false";
#else
    out<<"false";
    op.cleanup();
#endif
    out<<"}\n";std::puts("PASS complete matrix-free Muller moment action");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
