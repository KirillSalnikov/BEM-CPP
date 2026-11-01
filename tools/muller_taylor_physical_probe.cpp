// CPU-only streaming current-moment integration control. No global dense A,
// B, or radiation matrices are read or constructed by this candidate.
#include "muller_fmm_taylor.h"
#include "muller_fmm_cpu.h"
#include "muller_taylor_fields.h"
#include "streaming_host_gmres.h"
#include "tmatrix_rhs_assembly.h"
#include "../tests/taylor_cpu_fixture.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using C=std::complex<double>;using V=std::vector<C>;using H=std::vector<V>;
using Clock=std::chrono::steady_clock;
#ifndef BEM_TAYLOR_DEGREE
#define BEM_TAYLOR_DEGREE 4
#endif
constexpr int Degree=BEM_TAYLOR_DEGREE;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void save(const std::string& name,const H& values){std::ofstream out(name,std::ios::binary);if(!out)throw std::runtime_error("open "+name);
    for(const auto& row:values)out.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(C));
    if(!out)throw std::runtime_error("write "+name);}
int main(int argc,char** argv)try{
    if(argc!=5)throw std::invalid_argument("REF KA GRID SETUP_REPORT; stdin: column fields_file output_prefix");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    if(ref<0||ref>1||ka<=0||ka>10)throw std::invalid_argument("small CPU control bounds");
    setvbuf(stdout,nullptr,_IOLBF,0);auto t=Clock::now();MullerFmmOperator op;initialize_taylor_cpu_fixture(op,ref,ka);
    const int n=op.system_dofs,c=op.current_dofs;bem_cpu::MullerTaylorAction<Degree> action(op,8);
    op.correction.row_offsets=action.correction.offsets;
    for(auto e:action.correction.entries){MullerNearCorrectionEntry v;v.column=e.column;v.k1=C(0,-ka)*e.value.diagonal[0];
        v.k2_epsilon=e.value.epsilon[0];v.k2_mu=e.value.mu[0];op.correction.entries.push_back(v);}
    bem_cpu::MullerAction central(op,0);
    const double operator_s=seconds(t);t=Clock::now();
    // Assemble only bounded local blocks from the existing physical Galerkin
    // assembler. Its Duffy=4 approximation is a preconditioner; the equation
    // and every true residual use Duffy=8, as required by the symmetry gate.
    struct Block{int size;V factor;std::vector<int> indices,pivots;};std::vector<Block> blocks;
    size_t inverse_bytes=0;
    for(int first=0;first<c/2;first+=8){std::vector<int> edges;for(int i=first;i<std::min(first+8,c/2);++i)edges.push_back(i);
        Block b;b.size=4*edges.size();b.factor=assemble_muller_nodal_block(op,edges);
        for(int edge:edges)for(int current=0;current<2;++current)for(int moment=0;moment<2;++moment)b.indices.push_back(current*c+2*edge+moment);
        if(!bem_krylov::dense_lu_factor(b.factor,b.size,b.pivots))throw std::runtime_error("physical moment local inverse");
        inverse_bytes+=b.factor.size()*sizeof(C)+(b.indices.size()+b.pivots.size())*sizeof(int);blocks.push_back(std::move(b));}
    const double inverse_setup_s=seconds(t);
    std::ifstream grid_file(argv[3],std::ios::binary|std::ios::ate);if(!grid_file||size_t(grid_file.tellg())%(9*sizeof(double)))throw std::runtime_error("physical grid shape");
    std::vector<double> grid(size_t(grid_file.tellg())/sizeof(double));grid_file.seekg(0);grid_file.read(reinterpret_cast<char*>(grid.data()),grid.size()*sizeof(double));
    if(grid.empty()||!grid_file)throw std::runtime_error("empty/unreadable grid");
    std::vector<Vec3> directions;
    for(size_t i=0;i<grid.size();i+=9)directions.emplace_back(grid[i],grid[i+1],grid[i+2]);
    const bem_streaming::Action a0=[&](const C* x,C* y){central.apply(x,y);};
    const bem_streaming::Action inverse=[&](const C* x,C* y){for(const auto& b:blocks){V local(b.size);for(int j=0;j<b.size;++j)local[j]=x[b.indices[j]];
        bem_krylov::dense_lu_solve(b.factor,b.pivots,b.size,local.data());for(int j=0;j<b.size;++j)y[b.indices[j]]=local[j];}};
    bem_streaming::RecycleSpace recycle(n,8);bem_tmatrix::StreamingRhsAssembler assembler(tri_quadrature(13).npts,256,2);
    std::ofstream setup(argv[4]);setup<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"degree\":"<<Degree<<",\"duffy\":8,\"quad\":7,\"dofs\":"<<n
        <<",\"points\":"<<op.mass_quadrature.size()<<",\"observation_directions\":"<<directions.size()<<",\"operator_setup_s\":"<<operator_s
        <<",\"inverse_setup_s\":"<<inverse_setup_s<<",\"inverse_bytes\":"<<inverse_bytes<<",\"derivative_workspace_bytes\":"<<action.workspace_bytes()
        <<",\"central_workspace_bytes\":"<<central.workspace_bytes()<<",\"global_dense_A_stored\":false,\"host_fixture\":true}\n";setup.close();
    std::cout<<"READY "<<n<<' '<<op.mass_quadrature.size()<<' '<<Degree<<std::endl;
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
        const double rhs_s=seconds(t);t=Clock::now();int iterations=0;
        auto solved=bem_taylor::solve_moments(rhs,[&](const H& x,H& y){action.apply(x,y);},[&](const V& b,V& x){
            const auto r=bem_streaming::gmres(a0,a0,inverse,b.data(),x.data(),n,40,1e-10,600,false,false,&recycle);
            iterations+=r.iterations;if(r.residual>1e-10)throw std::runtime_error("physical moment GMRES failed");},a0,2e-10);
        const double solve_s=seconds(t);t=Clock::now();const H far=bem_taylor::farfield_moments<Degree>(op.mesh,ka,solved.current,directions);
        const double far_s=seconds(t);save(prefix+".rhs.bin",rhs);save(prefix+".currents.bin",solved.current);save(prefix+".far.bin",far);
        std::ofstream report(prefix+".json");report<<std::setprecision(17)<<"{\"column\":"<<column<<",\"iterations\":"<<iterations
            <<",\"rhs_s\":"<<rhs_s<<",\"solve_s\":"<<solve_s<<",\"far_s\":"<<far_s<<",\"recycle_rank\":"<<recycle.rank()<<",\"original_rhs_residuals\":[";
        for(size_t j=0;j<solved.original_rhs_residuals.size();++j)report<<(j?",":"")<<solved.original_rhs_residuals[j];
        report<<"],\"passed\":true}\n";report.close();
        if(!report)throw std::runtime_error("physical column report");
        std::cout<<"DONE "<<column<<std::endl;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
