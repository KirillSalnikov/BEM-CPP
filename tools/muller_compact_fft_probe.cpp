// Isolated operator/preconditioner bridge for a compact inverse experiment.
// Existing production worker and checkpoints are never replaced by this tool.
#include "muller_fmm.h"
#include "prism_face_fft.h"
#include "prism_symmetric_circulant.h"
#include "quadrature.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
using C=std::complex<double>;using V=std::vector<C>;using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
template<class T>void save(const std::string&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));if(!f)throw std::runtime_error("write "+p);}
V read(const std::string&p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(count*sizeof(C)))throw std::runtime_error("input size "+p);V v(count);f.seekg(0);f.read((char*)v.data(),count*sizeof(C));return v;}
double rel(const V&a,const V&b){double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}return std::sqrt(e/std::max(n,1e-300));}
template<class F,class H>double adjoint_check(int n,F f,H h){
    std::mt19937 rng(915);std::normal_distribution<double>d;V x(n),y(n),fx(n),hy(n);
    for(int i=0;i<n;++i){x[i]=C(d(rng),d(rng));y[i]=C(d(rng),d(rng));}f(x.data(),fx.data());h(y.data(),hy.data());
    C left=0,right=0;double norm=0,yn=0;for(int i=0;i<n;++i){left+=std::conj(y[i])*fx[i];right+=std::conj(hy[i])*x[i];norm+=std::norm(fx[i]);yn+=std::norm(y[i]);}
    return std::abs(left-right)/std::sqrt(norm*yn);
}
int main(int argc,char**argv)try{
    if(argc!=5)throw std::runtime_error("REF KA DUFFY OUTPUT_DIRECTORY");
    const int ref=std::stoi(argv[1]),duffy=std::stoi(argv[3]);const double ka=std::stod(argv[2]),aspect=10./7;const std::string dir=argv[4];
    if(ref<2||ref>5||ka<=0||duffy<6)throw std::runtime_error("invalid parameters");
    FILE*reply=fdopen(dup(STDOUT_FILENO),"w");if(!reply)throw std::runtime_error("protocol");fflush(stdout);dup2(STDERR_FILENO,STDOUT_FILENO);setvbuf(stdout,nullptr,_IOLBF,0);
    setenv("BEM_MULLER_GPU_ASSEMBLY","1",1);setenv("BEM_FMM_MULTILEVEL_GPU_ASSEMBLY","1",1);setenv("BEM_FMM_MULTILEVEL","1",1);setenv("BEM_FMM_PAIR_CURRENTS","0",1);
    const int seg=1<<ref;int nz=std::max(seg,int(std::ceil(std::sqrt(3.)*aspect*seg)));nz+=nz%2;
    const int points=7*12*seg*(seg+nz),depth=std::min(6,std::max(3,int(std::ceil(std::log(2.*points/64)/std::log(8.)))));
    auto leaf=[&](int d){return int(std::ceil(2.*points/std::pow(8.,d)));};
    setenv("BEM_FMM_BANDED_SPLIT_DEPTH",std::to_string(depth-2).c_str(),1);setenv("BEM_FMM_BANDED_COARSE_MAX_LEAF",std::to_string(leaf(depth-2)).c_str(),1);setenv("BEM_FMM_BANDED_MIDDLE_MAX_LEAF",std::to_string(leaf(depth-1)).c_str(),1);
    auto start=Clock::now();MullerP2BuildOptions opt;opt.edge_mode=MullerEdgeMode::HDivBdm1;MullerFmmOperator op;
    struct Cleanup{MullerFmmOperator&op;~Cleanup(){op.cleanup();}}cleanup{op};
    op.init(regular_prism(6,aspect,ref,1.,0,false,true),C(ka),C(1.3116),opt,7,duffy,5,leaf(depth),false,2,0,.96,(dir+"/native.near").c_str(),3);
    op.enable_prism_face_fft(ref,aspect);op.set_fmm_near_fp32(true);const double op_s=seconds(start);const int n=op.system_dofs;
    const auto A=[&](const C*x,C*y){op.matvec_strict(x,y);};const auto AH=[&](const C*x,C*y){op.matvec_adjoint_strict(x,y);};
    bem_tmatrix::D6hOrbit orbit;orbit.build(op.mesh);bem_tmatrix::PrismGroupMaps maps;maps.build(orbit);
    start=Clock::now();PrismOrbitInverse local;local.build(op,maps,50);PrismCirculantInverse circ;
    circ.build(op.mesh,ref,[&](const C*x,C*y){op.matvec(x,y);},[&](const C*x,C*y){local.apply(x,y);},0,0,3);PrismSymmetricCirculant sym;sym.build(orbit,circ,local);
    if(!sym.prepare_full_fft_symmetry(op.mesh))throw std::runtime_error("full FFT symmetry unavailable");
    const auto P=[&](const C*x,C*y){sym.apply_full_fft_symmetry(x,y,.5);};const auto PH=[&](const C*x,C*y){sym.apply_full_fft_symmetry_adjoint(x,y,.5);};
    double gpu_adjoint=adjoint_check(n,P,PH);
    if(!circ.prepare_cpu())throw std::runtime_error("FFTW unavailable");circ.enable_cpu(true);
    const double pre_s=seconds(start),p_adjoint=adjoint_check(n,P,PH),a_adjoint=adjoint_check(n,A,AH),local_adjoint=adjoint_check(n,[&](const C*x,C*y){local.apply(x,y);},[&](const C*x,C*y){local.apply_adjoint(x,y);});
    if(std::max({gpu_adjoint,p_adjoint,a_adjoint,local_adjoint})>2e-11)throw std::runtime_error("adjoint identity failure");
    std::vector<int> perms;for(int g=0;g<3;++g){std::vector<int>idx,sgn;if(g<2)orbit.image_map(g==1,false,idx,sgn);else orbit.vertical_image_map(op.mesh,idx,sgn);perms.insert(perms.end(),idx.begin(),idx.end());perms.insert(perms.end(),sgn.begin(),sgn.end());}save(dir+"/symmetry.bin",perms);
    std::vector<double>metric(n);const auto quad=tri_quadrature(13);
    for(size_t t=0;t<op.mesh.elements.size();++t)for(int q=0;q<quad.npts;++q){const auto f=evaluate_muller_frame(op.mesh,t,quad.pts[q][0],quad.pts[q][1]);const auto b=evaluate_muller_basis(op.mesh,t,f);for(int j=0;j<b.count;++j){const double v=.5*quad.wts[q]*f.jacobian*b.values[j].dot(b.values[j]);metric[b.dofs[j]]+=v;metric[op.current_dofs+b.dofs[j]]+=v;}}save(dir+"/metric.bin",metric);
    V x(n),px(n),rx(n),prx(n),rpx(n);for(int i=0;i<n;++i)x[i]=C(std::sin(.17*i),std::cos(.23*i));P(x.data(),px.data());double covariance=0;
    for(int g=0;g<3;++g){const int*idx=perms.data()+2*g*n,*sgn=idx+n;for(int i=0;i<n;++i){rx[idx[i]]=double(g? -sgn[i]:sgn[i])*x[i];rpx[idx[i]]=double(sgn[i])*px[i];}P(rx.data(),prx.data());covariance=std::max(covariance,rel(prx,rpx));}
    if(covariance>1e-8)throw std::runtime_error("FFT preconditioner D6h failure");
    V zero(n),z(n,C(1));PH(zero.data(),z.data());for(C v:z)if(v!=C(0))throw std::runtime_error("stale adjoint output");
    struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::ofstream meta(dir+"/setup.json");meta<<std::setprecision(17)<<"{\"dofs\":"<<n<<",\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"duffy\":"<<duffy<<",\"operator_s\":"<<op_s<<",\"preconditioner_s\":"<<pre_s<<",\"P_adjoint_CPU_error\":"<<p_adjoint<<",\"P_adjoint_GPU_error\":"<<gpu_adjoint<<",\"A_adjoint_error\":"<<a_adjoint<<",\"local_adjoint_error\":"<<local_adjoint<<",\"P_D6h_covariance_error\":"<<covariance<<",\"local_storage_bytes\":"<<local.storage_bytes()<<",\"FFT_CPU_storage_bytes\":"<<circ.cpu_storage_bytes()<<",\"FFT_GPU_storage_bytes\":"<<circ.storage_bytes()<<",\"rss_bytes\":"<<size_t(usage.ru_maxrss)*1024<<",\"preconditioner_backend\":\"production full-symmetry FFT/circulant CPU path, blend 0.5\",\"operator_backend\":\"native strict FP64 FFT/FMM\"}\n";meta.close();
    fprintf(reply,"READY %d\n",n);fflush(reply);
    std::string line;while(std::getline(std::cin,line)){if(line=="QUIT")break;std::istringstream in(line);std::string kind,src,dst;int id,count;if(!(in>>kind>>id>>count>>src>>dst)||count<1||count>64)throw std::runtime_error("request");V v=read(src,size_t(n)*count),out(v.size());start=Clock::now();
        for(int j=0;j<count;++j){const C*x=v.data()+size_t(j)*n;C*y=out.data()+size_t(j)*n;if(kind=="A")A(x,y);else if(kind=="AH")AH(x,y);else if(kind=="AM")op.matvec(x,y);else if(kind=="AHM")op.matvec_adjoint(x,y);else if(kind=="P")P(x,y);else if(kind=="PH")PH(x,y);else throw std::runtime_error("kind");}
        const double t=seconds(start);save(dst,out);fprintf(reply,"DONE %d %.17g\n",id,t);fflush(reply);
    }fclose(reply);
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
