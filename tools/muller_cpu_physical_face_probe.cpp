#include "../tests/physical_face_cpu_fixture.h"
#include "muller_fmm_cpu.h"
#include "muller_taylor_correction.h"
#include "tmatrix_symmetry_warm_start.h"
#include <fstream>
#include <iomanip>
#include <random>
#include <sys/resource.h>

using C=std::complex<double>;using Vector=std::vector<C>;using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
double relative(const Vector& a,const Vector& b){
    long double delta=0,norm=0;for(size_t i=0;i<a.size();++i){delta+=std::norm(a[i]-b[i]);norm+=std::norm(b[i]);}
    return std::sqrt(delta/std::max(norm,1e-280L));
}
Vector multiply(const Vector& matrix,const Vector& x){
    const size_t n=x.size();Vector y(n);
#pragma omp parallel for schedule(static)
    for(size_t row=0;row<n;++row){C sum=0;for(size_t col=0;col<n;++col)sum+=matrix[row*n+col]*x[col];y[row]=sum;}
    return y;
}
void require_error(double value,double limit,const char* what){
    if(!std::isfinite(value)||value>limit)throw std::runtime_error(std::string(what)+": "+std::to_string(value));
}
struct Record {
    std::string name;std::vector<double> dense_errors;
    double setup=0,action=0,fft=0,cross=0,alias=0,symmetry=0;
    size_t workspace=0,estimated=0;int shared_bands=0;
};
int main(int argc,char** argv)try{
#ifndef BEM_CPU_HAS_AVX512_P2P
    throw std::runtime_error("This physical shared-AVX512 probe must be built with native AVX512 support");
#endif
    if(argc!=6)throw std::invalid_argument("REF KA DUFFY DENSE_A_FILE REPORT");
    const int ref=std::stoi(argv[1]),duffy=std::stoi(argv[3]);const double ka=std::stod(argv[2]);
    if(ref<1||ref>2||!std::isfinite(ka)||ka<=0||duffy<2||duffy>16)throw std::invalid_argument("small physical face probe inputs");
    if(!bem_cpu::FftwRuntime::get().available())throw std::runtime_error("Set BEM_CPU_FFTW_LIBRARY to the validated local FFTW library");
    setvbuf(stdout,nullptr,_IOLBF,0);const auto begin=Clock::now();MullerFmmOperator op;
    initialize_physical_face_fixture(op,ref,ka);const int n=op.system_dofs;
    std::ifstream input(argv[4],std::ios::binary|std::ios::ate);const size_t bytes=size_t(n)*n*sizeof(C);
    if(!input||input.tellg()!=std::streamoff(bytes))throw std::runtime_error("dense reference must contain exactly one row-major physical matrix");
    input.seekg(0);Vector matrix(size_t(n)*n);input.read(reinterpret_cast<char*>(matrix.data()),bytes);
    if(!input)throw std::runtime_error("dense reference read failed");
    auto start=Clock::now();size_t correction_pairs=0,correction_templates=0;
    {bem_taylor::SparseCorrection<0> correction(op.mesh,ka,C(1.3116),7,duffy);
        op.correction.row_offsets=correction.offsets;correction_pairs=correction.pairs;correction_templates=correction.templates;
        for(const auto& e:correction.entries){MullerNearCorrectionEntry value;value.column=e.column;
            value.k1=C(0,-ka)*e.value.diagonal[0];value.k2_epsilon=e.value.epsilon[0];value.k2_mu=e.value.mu[0];
            op.correction.entries.push_back(value);}}
    const double correction_s=elapsed(start);
    std::printf("physical ref%d ka%.6g dofs%d points%zu Duffy%d: correction %.3fs, %zu templates\n",ref,ka,n,op.quadrature.size(),duffy,correction_s,correction_templates);
    std::mt19937 random(10973);std::normal_distribution<double> normal;
    std::vector<Vector> currents(6,Vector(n)),expected(6);std::vector<std::string> labels={"mixed","electric_only","magnetic_only","rotation","horizontal","vertical"};
    for(auto& z:currents[0])z=C(normal(random),normal(random));
    for(int i=0;i<n;++i)currents[i<op.current_dofs?1:2][i]=currents[0][i];
    bem_tmatrix::D6hOrbit group;group.build(op.mesh);std::vector<int> vertical,sign;
    bem_tmatrix::D6hOrbit::vertical_image_map(op.mesh,vertical,sign);
    const auto transform=[&](const Vector& from,int generator,bool rhs){Vector to(n);
        if(generator<2)group.transform(from,to,generator==1,rhs);
        else for(int i=0;i<n;++i)to[vertical[i]]=double((rhs?-1:1)*sign[i])*from[i];
        return to;};
    for(int j=0;j<3;++j)currents[3+j]=transform(currents[0],j,false);
    for(size_t j=0;j<currents.size();++j)expected[j]=multiply(matrix,currents[j]);
    double dense_symmetry=0;
    for(int j=0;j<3;++j)dense_symmetry=std::max(dense_symmetry,relative(expected[3+j],transform(expected[0],j,true)));
    require_error(dense_symmetry,2e-8,"reference D6h covariance");
    const bool supports_avx=__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512vl");
    std::vector<Record> records;
    for(int variant=0;variant<5;++variant){
        const bool face=variant>0,batch=variant>=2,shared=variant>=3,portable=variant==4;
        setenv("BEM_CPU_AVX512",portable?"0":"1",1);
        Record row;row.name=variant==0?"all_pair":variant==1?"face_serial":variant==2?"face_batch":variant==3?"face_shared":"face_shared_portable";
        row.estimated=bem_cpu::MullerAction::estimated_workspace_bytes(op,0,true,face,batch,32);
        start=Clock::now();bem_cpu::MullerAction action(op,0,true,face,batch,32,shared);row.setup=elapsed(start);
        row.workspace=action.workspace_bytes();row.shared_bands=action.shared_avx512_near_bands;
        if(row.workspace>row.estimated)throw std::runtime_error("underestimated physical CPU numerical arrays");
        if(row.shared_bands!=(shared&&!portable&&supports_avx?4:0))throw std::runtime_error("shared AVX512 physical dispatch count");
        std::vector<Vector> actual(portable?3:6,Vector(n));
        for(size_t j=0;j<actual.size();++j){start=Clock::now();action.apply(currents[j].data(),actual[j].data());row.action+=elapsed(start);
            row.fft+=action.face_fft_s;row.cross+=action.face_cross_s;const double error=relative(actual[j],expected[j]);
            row.dense_errors.push_back(error);require_error(error,2e-9,"physical CPU face versus dense BEM");
            std::printf("%s %s dense error %.6e\n",row.name.c_str(),labels[j].c_str(),error);}
        if(face&&(row.fft<=0||row.cross<=0))throw std::runtime_error("physical FFT or cross stage not exercised");
        if(!portable)for(int j=0;j<3;++j)row.symmetry=std::max(row.symmetry,relative(actual[3+j],transform(actual[0],j,true)));
        require_error(row.symmetry,2e-8,"physical CPU D6h covariance");
        Vector alias=currents[0];action.apply(alias.data(),alias.data());row.alias=relative(alias,actual[0]);require_error(row.alias,2e-12,"physical CPU in-place action");
        Vector zero(n),zero_image(n,C(1,2));action.apply(zero.data(),zero_image.data());
        for(C z:zero_image)if(z!=C(0))throw std::runtime_error("physical CPU stale zero output");
        std::printf("%s: shared bands %d, numerical workspace %zu, estimate %zu, D6h %.3e\n",row.name.c_str(),row.shared_bands,row.workspace,row.estimated,row.symmetry);
        records.push_back(row);
    }
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::ofstream report(argv[5]);if(!report)throw std::runtime_error("cannot write physical face report");report<<std::setprecision(17);
    report<<"{\n\"passed\":true,\"GPU_used\":false,\"native_far_tree_verified\":false,\"full_T_speedup_proven\":false,\n"
          <<"\"scope\":\"Physical quadrature7 BDM prism, exact rectangular cross-face P2P metadata, real production CPU FFT/projection/testing and sparse Duffy corrections\",\n"
          <<"\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"ri\":1.3116,\"aspect\":"<<10./7<<",\"duffy\":"<<duffy<<",\"dofs\":"<<n
          <<",\"points\":"<<op.quadrature.size()<<",\"avx512_supported\":"<<(supports_avx?"true":"false")<<",\"correction_pairs\":"<<correction_pairs
          <<",\"correction_templates\":"<<correction_templates<<",\"correction_s\":"<<correction_s<<",\"dense_D6h_error\":"<<dense_symmetry
          <<",\"elapsed_s\":"<<elapsed(begin)<<",\"peak_RSS_KiB\":"<<usage.ru_maxrss<<",\n\"variants\":[\n";
    for(size_t j=0;j<records.size();++j){const auto& r=records[j];if(j)report<<",\n";
        report<<"{\"name\":\""<<r.name<<"\",\"dense_errors\":[";
        for(size_t q=0;q<r.dense_errors.size();++q){if(q)report<<",";report<<r.dense_errors[q];}
        report<<"],\"setup_s\":"<<r.setup<<",\"diagnostic_action_s\":"<<r.action<<",\"diagnostic_fft_s\":"<<r.fft<<",\"diagnostic_cross_s\":"<<r.cross
              <<",\"workspace_bytes\":"<<r.workspace<<",\"estimated_bytes\":"<<r.estimated<<",\"shared_AVX512_bands\":"<<r.shared_bands
              <<",\"D6h_generators_checked\":"<<(j==4?0:3)<<",\"D6h_error\":";
        if(j==4)report<<"null";else report<<r.symmetry;
        report<<",\"alias_error\":"<<r.alias<<",\"zero_passed\":true}";
    }
    report<<"\n]}\n";report.close();if(!report)throw std::runtime_error("physical face report write failed");
    std::puts("PASS physical CPU face action against independent dense BEM; no native far-tree or complete-solve timing claim");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
