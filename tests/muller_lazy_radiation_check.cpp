// GPU integration check: lazy allocation, single/pair transitions, standalone
// samples and cleanup. Compare the actual field to independent CPU quadrature.
#include "muller_fmm_gpu.h"
#include "muller_dense.h"
#include "quadrature.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

using C = std::complex<double>;
using V = std::vector<C>;
size_t free_memory() {
    size_t free=0, total=0;
    if(cudaDeviceSynchronize()!=cudaSuccess || cudaMemGetInfo(&free,&total)!=cudaSuccess)
        throw std::runtime_error("CUDA memory query failed");
    return free;
}
double check(const V& actual,const V& expected) {
    if(actual.size()!=expected.size())throw std::runtime_error("field size mismatch");
    double error=0,norm=0;
    for(size_t i=0;i<actual.size();++i){error+=std::norm(actual[i]-expected[i]);norm+=std::norm(expected[i]);}
    error=std::sqrt(error/std::max(norm,1e-300));
    if(!std::isfinite(error)||error>1e-11)throw std::runtime_error("GPU vs CPU field mismatch");
    return error;
}
int main(int argc,char**argv)try{
    if(argc!=2)throw std::runtime_error("muller_lazy_radiation_check REPORT");
    setenv("BEM_FARFIELD_FAST_PHASE","0",1);
    setenv("BEM_FARFIELD_FP32_ACCUM","0",1);
    MullerP2BuildOptions opts;opts.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,1.6495721976846451,3,1.,0,false,true),opts);
    const int n=mesh.system_dofs(),m=n/2;
    V x(n),y(n);for(int i=0;i<n;++i){x[i]=C(std::sin(.17*i),std::cos(.31*i));y[i]=C(std::cos(.23*i),std::sin(.57*i));}
    std::vector<Vec3> directions;
    for(int i=0;i<=20;++i){const double t=std::acos(-1.)*i/20;directions.emplace_back(std::sin(t),0,std::cos(t));}
    const auto q=tri_quadrature(13);const size_t points=mesh.elements.size()*q.npts;
    std::vector<int> counts(points),dofs(points*12,-1);
    std::vector<double> basis(points*36),positions(points*3),weights(points);
    V sx(points*6),sy(points*6);
    for(size_t e=0;e<mesh.elements.size();++e)for(int j=0;j<q.npts;++j){const size_t p=e*q.npts+j;
        const auto frame=evaluate_muller_frame(mesh,e,q.pts[j][0],q.pts[j][1]);
        const auto b=evaluate_muller_basis(mesh,e,frame);
        counts[p]=b.count;weights[p]=.5*q.wts[j]*frame.jacobian;
        positions[3*p]=frame.position.x;positions[3*p+1]=frame.position.y;positions[3*p+2]=frame.position.z;
        for(int a=0;a<b.count;++a){dofs[12*p+a]=b.dofs[a];
            const double xyz[3]={b.values[a].x,b.values[a].y,b.values[a].z};
            for(int c=0;c<3;++c){basis[36*p+3*a+c]=xyz[c];
                for(int current=0;current<2;++current){
                    sx[6*p+3*current+c]+=weights[p]*xyz[c]*x[current*m+b.dofs[a]];
                    sy[6*p+3*current+c]+=weights[p]*xyz[c]*y[current*m+b.dofs[a]];
                }
            }
        }
    }
    MullerGpuAssembly gpu;
    auto init=[&](){gpu.init(m,{0},std::vector<int>(12,-1),std::vector<double>(36),{0,0,1},{0},
        counts,dofs,basis,positions,weights,std::vector<int>(m+1),{});};
    const size_t before=free_memory();init();const size_t initialized=free_memory();
    V a,b,rx,ry;
    gpu.farfield(x.data(),10.,{},a);gpu.farfield_pair(x.data(),y.data(),10.,{},a,b);
    if(!a.empty()||!b.empty()||free_memory()!=initialized)throw std::runtime_error("empty radiation allocated memory");
    double worst=0;size_t after_single=0,after_pair=0;
    for(const C k : {C(10.,0.),C(10.,.15)}){
        muller_nodal_farfield_pair(mesh,x.data(),x.data()+m,y.data(),y.data()+m,k,directions,rx,ry,13);
        gpu.farfield(x.data(),k,directions,a);worst=std::max(worst,check(a,rx));
        if(after_single==0)after_single=free_memory();
        gpu.farfield_pair(x.data(),y.data(),k,directions,a,b);
        worst=std::max(worst,std::max(check(a,rx),check(b,ry)));
        if(after_pair==0)after_pair=free_memory();
        gpu.farfield(y.data(),k,directions,a);worst=std::max(worst,check(a,ry));
        gpu.farfield_pair(y.data(),x.data(),k,directions,a,b);
        worst=std::max(worst,std::max(check(a,ry),check(b,rx)));
    }
    gpu.cleanup();gpu.cleanup();init();
    gpu.farfield_pair(x.data(),y.data(),C(10.,.15),directions,a,b);
    worst=std::max(worst,std::max(check(a,rx),check(b,ry)));
    // This also replaces an initialized full operator by standalone storage.
    gpu.upload_farfield_samples(positions,sx,sy);
    gpu.farfield_pair(nullptr,nullptr,C(10.,.15),directions,a,b);
    worst=std::max(worst,std::max(check(a,rx),check(b,ry)));
    for(auto& v:sx)v*=2.;for(auto& v:sy)v=-v;
    for(auto& v:rx)v*=2.;for(auto& v:ry)v=-v;
    gpu.upload_farfield_samples(positions,sx,sy);
    gpu.farfield_pair(nullptr,nullptr,C(10.,.15),directions,a,b);
    worst=std::max(worst,std::max(check(a,rx),check(b,ry)));
    gpu.cleanup();gpu.cleanup();const size_t after=free_memory();
    std::ofstream out(argv[1]);out<<std::setprecision(17)
        <<"{\"status\":\"pass\",\"points\":"<<points<<",\"worst_field_relative_error\":"<<worst
        <<",\"init_bytes\":"<<static_cast<long long>(before)-initialized
        <<",\"first_single_extra_bytes\":"<<static_cast<long long>(initialized)-after_single
        <<",\"second_polarization_extra_bytes\":"<<static_cast<long long>(after_single)-after_pair
        <<",\"expected_current_bytes_per_polarization\":"<<points*12*sizeof(double)
        <<",\"remaining_bytes_after_cleanup\":"<<static_cast<long long>(before)-after<<"}\n";
    std::cout<<"PASS field error="<<worst<<std::endl;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
