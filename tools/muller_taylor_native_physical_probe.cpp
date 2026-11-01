// Native-tree physical moment worker: same-face FFT, cross-face FMM, and
// independent original FP64 GPU residuals. No global dense A/B/D input.
#include "muller_fmm_taylor.h"
#include "muller_fmm_cpu.h"
#include "muller_taylor_fields.h"
#include "streaming_host_gmres.h"
#include "tmatrix_rhs_assembly.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

using C=std::complex<double>;using V=std::vector<C>;using H=std::vector<V>;
using Clock=std::chrono::steady_clock;
constexpr int Degree=4;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
double relative(const V& a,const V& b){double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}return std::sqrt(e/std::max(n,1e-280));}
void save(const std::string& name,const H& values){std::ofstream out(name,std::ios::binary);if(!out)throw std::runtime_error("open "+name);
    for(const auto& row:values)out.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(C));
    if(!out)throw std::runtime_error("write "+name);}
int main(int argc,char** argv)try{
    if(argc!=5)throw std::invalid_argument("REF KA GRID SETUP_REPORT; stdin: column fields_file output_prefix");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    if(ref<1||ref>2||!std::isfinite(ka)||ka<=0||ka>10)throw std::invalid_argument("native physical control bounds");
    // Native preparation/actions write diagnostics to stdout. Keep the wire
    // protocol on its own descriptor and redirect all diagnostics to stderr.
    FILE* protocol=fdopen(dup(STDOUT_FILENO),"w");
    if(!protocol||dup2(STDERR_FILENO,STDOUT_FILENO)<0)throw std::runtime_error("physical protocol descriptors");
    setvbuf(protocol,nullptr,_IOLBF,0);setvbuf(stdout,nullptr,_IOLBF,0);
    auto t=Clock::now();MullerFmmOperator op;
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,C(1.3116),options,7,8,5,10);
    op.enable_prism_face_fft(ref,10./7);
    size_t far_pairs=0;
    for(const auto* face_operator:{op.prism_fft_exterior.get(),op.prism_fft_interior.get()}){
        const auto view=face_operator->host_view();
        for(int face=0;face<8;++face)if(view.faces[face].engine==face)
            for(const auto* f:{view.faces[face].fine,view.faces[face].middle,view.faces[face].coarse})
                for(const auto& level:f->m2l_batches)far_pairs+=level.n_pairs;
    }
    if(!far_pairs)throw std::runtime_error("physical worker must exercise native cross-face M2L");
    const int n=op.system_dofs,c=op.current_dofs;
    bem_cpu::MullerTaylorAction<Degree> action(op,8,true,true);
    // Preserve the originally assembled Duffy operator for both central
    // actions and strict GPU checks. Its derivative counterpart is audited
    // by MullerTaylorAction; do not replace it with the candidate correction.
    bem_cpu::MullerAction central(op,0,true,true);
    const double operator_s=seconds(t);t=Clock::now();
    struct Block{int size;V factor;std::vector<int> indices,pivots;};std::vector<Block> blocks;
    size_t inverse_bytes=0;
    for(int first=0;first<c/2;first+=8){std::vector<int> edges;for(int i=first;i<std::min(first+8,c/2);++i)edges.push_back(i);
        Block b;b.size=4*edges.size();b.factor=assemble_muller_nodal_block(op,edges);
        for(int edge:edges)for(int current=0;current<2;++current)for(int moment=0;moment<2;++moment)b.indices.push_back(current*c+2*edge+moment);
        if(!bem_krylov::dense_lu_factor(b.factor,b.size,b.pivots))throw std::runtime_error("native physical local inverse");
        inverse_bytes+=b.factor.size()*sizeof(C)+(b.indices.size()+b.pivots.size())*sizeof(int);blocks.push_back(std::move(b));}
    const double inverse_setup_s=seconds(t);
    std::ifstream grid_file(argv[3],std::ios::binary|std::ios::ate);if(!grid_file||size_t(grid_file.tellg())%(9*sizeof(double)))throw std::runtime_error("physical grid shape");
    std::vector<double> grid(size_t(grid_file.tellg())/sizeof(double));grid_file.seekg(0);grid_file.read(reinterpret_cast<char*>(grid.data()),grid.size()*sizeof(double));
    if(grid.empty()||!grid_file)throw std::runtime_error("empty/unreadable grid");
    std::vector<Vec3> directions;for(size_t i=0;i<grid.size();i+=9)directions.emplace_back(grid[i],grid[i+1],grid[i+2]);
    size_t cpu_calls=0,strict_calls=0;double cpu_s=0,strict_s=0;
    const bem_streaming::Action a0=[&](const C* x,C* y){auto begin=Clock::now();central.apply(x,y);cpu_s+=seconds(begin);++cpu_calls;};
    const bem_streaming::Action strict=[&](const C* x,C* y){auto begin=Clock::now();op.matvec_strict(x,y);strict_s+=seconds(begin);++strict_calls;};
    const bem_streaming::Action inverse=[&](const C* x,C* y){for(const auto& b:blocks){V local(b.size);for(int j=0;j<b.size;++j)local[j]=x[b.indices[j]];
        bem_krylov::dense_lu_solve(b.factor,b.pivots,b.size,local.data());for(int j=0;j<b.size;++j)y[b.indices[j]]=local[j];}};
    bem_streaming::RecycleSpace recycle(n,8);bem_tmatrix::StreamingRhsAssembler assembler(tri_quadrature(13).npts,256,2);
    std::ofstream setup(argv[4]);setup<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"degree\":"<<Degree<<",\"duffy\":8,\"quad\":7,\"dofs\":"<<n
        <<",\"points\":"<<op.mass_quadrature.size()<<",\"observation_directions\":"<<directions.size()<<",\"operator_setup_s\":"<<operator_s
        <<",\"inverse_setup_s\":"<<inverse_setup_s<<",\"inverse_bytes\":"<<inverse_bytes<<",\"derivative_workspace_bytes\":"<<action.workspace_bytes()
        <<",\"central_workspace_bytes\":"<<central.workspace_bytes()<<",\"global_dense_A_stored\":false,\"host_fixture\":false,\"face_fft\":true"
        <<",\"cross_far_pairs\":"<<far_pairs<<",\"central_correction_error\":"<<action.central_correction_error<<"}\n";setup.close();
    if(!setup)throw std::runtime_error("physical setup report");
    std::fprintf(protocol,"READY %d %zu %d\n",n,op.mass_quadrature.size(),Degree);
    std::string line;
    while(std::getline(std::cin,line)){
        if(line=="QUIT")break;
        std::istringstream request(line);int column;std::string input,prefix,trailing;
        if(!(request>>column>>input>>prefix)||(request>>trailing))throw std::runtime_error("physical request format");
        std::ifstream field_file(input,std::ios::binary);if(!field_file)throw std::runtime_error("physical fields missing");
        t=Clock::now();H rhs(Degree+1,V(n));V chunk(6*assembler.chunk_points());
        for(int j=0;j<=Degree;++j)for(size_t begin=0;begin<op.mass_quadrature.size();begin+=assembler.chunk_points()){
            const size_t count=std::min(assembler.chunk_points(),op.mass_quadrature.size()-begin);
            field_file.read(reinterpret_cast<char*>(chunk.data()),count*6*sizeof(C));if(!field_file)throw std::runtime_error("short physical fields");
            for(size_t i=0;i<count*6;++i)if(!std::isfinite(chunk[i].real())||!std::isfinite(chunk[i].imag()))throw std::runtime_error("nonfinite physical field");
            assembler.add(op.mesh,op.mass_quadrature,begin,count,chunk.data(),rhs[j]);}
        if(field_file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing physical fields");
        const double rhs_s=seconds(t);t=Clock::now();int iterations=0;cpu_s=strict_s=0;cpu_calls=strict_calls=0;
        double fft_s=0,cross_s=0,packing_s=0;
        auto solved=bem_taylor::solve_moments(rhs,[&](const H& x,H& y){action.apply(x,y);
            fft_s+=action.face_fft_seconds;cross_s+=action.face_cross_seconds;packing_s+=action.face_packing_seconds;},
            [&](const V& b,V& x){const auto r=bem_streaming::gmres(a0,a0,inverse,b.data(),x.data(),n,40,2e-10,800,false,false,&recycle);
                iterations+=r.iterations;if(r.residual>2e-10)throw std::runtime_error("native physical GMRES failed");},strict,5e-8);
        const double solve_s=seconds(t);t=Clock::now();const H far=bem_taylor::farfield_moments<Degree>(op.mesh,ka,solved.current,directions);
        const double far_s=seconds(t);save(prefix+".rhs.bin",rhs);save(prefix+".currents.bin",solved.current);save(prefix+".far.bin",far);
        std::ofstream report(prefix+".json");report<<std::setprecision(17)<<"{\"column\":"<<column<<",\"iterations\":"<<iterations
            <<",\"rhs_s\":"<<rhs_s<<",\"solve_s\":"<<solve_s<<",\"far_s\":"<<far_s<<",\"recycle_rank\":"<<recycle.rank()
            <<",\"cpu_actions\":"<<cpu_calls<<",\"cpu_action_s\":"<<cpu_s<<",\"original_gpu_checks\":"<<strict_calls<<",\"original_gpu_s\":"<<strict_s
            <<",\"moment_fft_s\":"<<fft_s<<",\"moment_cross_s\":"<<cross_s<<",\"moment_packing_s\":"<<packing_s<<",\"original_rhs_residuals\":[";
        for(size_t j=0;j<solved.original_rhs_residuals.size();++j)report<<(j?",":"")<<solved.original_rhs_residuals[j];
        report<<"],\"passed\":true}\n";report.close();if(!report)throw std::runtime_error("physical column report");
        std::fprintf(protocol,"DONE %d\n",column);
    }
    std::fclose(protocol);
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
