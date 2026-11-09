#include "host_sector_transform_layout.h"
#include "prism_symmetry_sectors.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#ifdef BEM_TEST_SECTOR_GPU
#include "sector_transform_gpu.h"
#include <cuda_runtime.h>
#endif
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point begin){return std::chrono::duration<double>(Clock::now()-begin).count();}
static double error(const Vector& a,const Vector& b){long double u=0,v=0;
    for(size_t i=0;i<a.size();++i){u+=std::norm(a[i]-b[i]);v+=std::norm(b[i]);}
    const double result=std::sqrt(double(u/std::max(v,1e-300L)));if(!std::isfinite(result))throw std::runtime_error("nonfinite transform error");return result;
}
#ifdef BEM_TEST_SECTOR_GPU
static void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
#endif
int main(int argc,char** argv)try {
    bool capacity=argc==3 && std::string(argv[1])=="--capacity";
    if(!capacity && argc!=1 && (argc!=2 || std::string(argv[1])!="--check"))throw std::invalid_argument("probe [--check|--capacity REF]");
    int large=capacity?std::stoi(argv[2]):0;if(capacity && (large<1 || large>7))throw std::invalid_argument("capacity ref");
#ifdef BEM_TEST_SECTOR_GPU
    if(capacity)throw std::invalid_argument("capacity is a CPU-only geometry/storage study");
#endif
    std::cout<<std::setprecision(17);int cases=0,rejected=0;double worst=0;
    const std::vector<int> refs=capacity?std::vector<int>{large}:std::vector<int>{0,1,2,3};
    const std::vector<double> aspects=capacity?std::vector<double>{10./7}:std::vector<double>{.2,10./7,5.};
    for(int ref:refs)for(double aspect:aspects){
        auto start=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,aspect,ref,1.,0,false,true),options);
        D6hOrbit orbit;orbit.build(mesh);PrismGroupMaps maps;maps.build(orbit);PrismSymmetrySectors original;original.build(maps);
        const double geometry_s=seconds(start);start=Clock::now();auto layout=bem_sector_device::Layout::capture(original);
        const double export_s=seconds(start);const int n=layout.dofs;
        Vector x(n),expected(n),actual(n),back(n),rotated(n),rotated_coordinates(n);
        for(int i=0;i<n;++i)x[i]=Complex(std::sin(.131*i),std::cos(.217*i));
        double defect=0,export_defect=0;
        for(bool rhs:{false,true}){
            original.encode(x.data(),expected.data(),rhs);layout.apply_host(x.data(),actual.data(),true,rhs);defect=std::max(defect,error(actual,expected));
            export_defect=std::max(export_defect,error(actual,expected));
            original.decode(x.data(),expected.data(),rhs);layout.apply_host(x.data(),actual.data(),false,rhs);defect=std::max(defect,error(actual,expected));
            export_defect=std::max(export_defect,error(actual,expected));
            layout.apply_host(x.data(),actual.data(),true,rhs);layout.apply_host(actual.data(),back.data(),false,rhs);defect=std::max(defect,error(back,x));
        }
        if(!capacity){
            const auto bad=[&](bem_sector_device::Layout q){try{q.validate();}catch(const std::invalid_argument&){++rejected;return;}throw std::runtime_error("corrupt sector layout accepted");};
            auto q=layout;q.dofs=0;bad(q);q=layout;q.sectors[1]=-1;bad(q);q=layout;q.physical[0]=q.physical[1];bad(q);
            q=layout;q.solution[0]=q.solution[1];bad(q);q=layout;q.rhs[0]=q.rhs[1];bad(q);
            q=layout;q.orbit_matrix[0]=q.matrix_dimensions.size();bad(q);q=layout;--q.orbit_offsets.back();bad(q);
            q=layout;q.matrix_dimensions[0]=0;bad(q);q=layout;--q.matrix_offsets.back();bad(q);
            q=layout;q.palette[0]=Complex(std::numeric_limits<double>::infinity(),0);bad(q);
            q=layout;q.weights[0]=q.palette.size();bad(q);q=layout;std::fill(q.palette.begin(),q.palette.end(),Complex(0));bad(q);
            q=layout;
            const auto sector=[&](int i){return int(std::upper_bound(q.sectors.begin(),q.sectors.end(),i)-q.sectors.begin())-1;};
            for(int i=1;i<n;++i)if(sector(q.rhs[i])!=sector(q.rhs[0])){std::swap(q.rhs[i],q.rhs[0]);break;}
            bad(q);
            for(int delta:{0,1}){bool caught=false;try{layout.apply_host(x.data(),x.data()+delta,true,true);}catch(const std::invalid_argument&){caught=true;}
                if(!caught)throw std::runtime_error("sector overlap accepted");
                ++rejected;}
        }
        std::cout<<"{\"ref\":"<<ref<<",\"aspect\":"<<aspect<<",\"dofs\":"<<n<<",\"orbits\":"<<layout.orbit_matrix.size()
                 <<",\"unique_matrices\":"<<layout.matrix_dimensions.size()<<",\"palette_values\":"<<layout.palette.size()
                 <<",\"original_host_storage_bytes\":"<<original.storage_bytes()<<",\"packed_array_bytes\":"<<layout.storage_bytes()
                 <<",\"geometry_and_basis_s\":"<<geometry_s<<",\"export_and_validation_s\":"<<export_s<<",\"host_export_error\":"<<export_defect<<",\"host_error\":"<<defect;
#ifdef BEM_TEST_SECTOR_GPU
        SectorTransformGPU gpu;gpu.init(layout,size_t(512)*1024*1024);
        struct Device {Complex *input=nullptr,*output=nullptr,*back=nullptr;~Device(){cudaFree(input);cudaFree(output);cudaFree(back);}} device;
        check(cudaMalloc(&device.input,n*sizeof(Complex)));check(cudaMalloc(&device.output,n*sizeof(Complex)));check(cudaMalloc(&device.back,n*sizeof(Complex)));
        check(cudaMemcpyAsync(device.input,x.data(),n*sizeof(Complex),cudaMemcpyHostToDevice,cudaStreamPerThread));
        for(bool rhs:{false,true}){
            for(bool encode:{false,true}){
                if(encode)original.encode(x.data(),expected.data(),rhs);else original.decode(x.data(),expected.data(),rhs);
                gpu.enqueue(device.input,device.output,encode,rhs);
                check(cudaMemcpyAsync(actual.data(),device.output,n*sizeof(Complex),cudaMemcpyDeviceToHost,cudaStreamPerThread));
                check(cudaStreamSynchronize(cudaStreamPerThread));defect=std::max(defect,error(actual,expected));
            }
            gpu.enqueue(device.input,device.output,true,rhs);gpu.enqueue(device.output,device.back,false,rhs);
            check(cudaMemcpyAsync(back.data(),device.back,n*sizeof(Complex),cudaMemcpyDeviceToHost,cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));defect=std::max(defect,error(back,x));
            for(bool mirror:{false,true}){
                orbit.transform(x,rotated,mirror,rhs);
                check(cudaMemcpyAsync(device.back,rotated.data(),n*sizeof(Complex),cudaMemcpyHostToDevice,cudaStreamPerThread));
                gpu.enqueue(device.back,device.output,true,rhs);
                check(cudaMemcpyAsync(rotated_coordinates.data(),device.output,n*sizeof(Complex),cudaMemcpyDeviceToHost,cudaStreamPerThread));
                check(cudaStreamSynchronize(cudaStreamPerThread));original.encode(x.data(),expected.data(),rhs);
                for(int s=0;s<12;++s){const Complex eigen=mirror?Complex(s<6?1.:-1.):std::polar(1.,2*std::acos(-1.)*(s%6)/6.);
                    for(int i=original.offsets[s];i<original.offsets[s+1];++i)expected[i]*=eigen;}
                defect=std::max(defect,error(rotated_coordinates,expected));
            }
        }
        for(int delta:{0,1}){bool caught=false;try{gpu.enqueue(device.input,device.input+delta,true,true);}catch(const std::invalid_argument&){caught=true;}
            if(!caught)throw std::runtime_error("GPU sector overlap accepted");}
        check(cudaMemsetAsync(device.input,0,n*sizeof(Complex),cudaStreamPerThread));
        gpu.enqueue(device.input,device.output,true,true);gpu.enqueue(device.output,device.back,false,true);
        check(cudaMemcpyAsync(back.data(),device.back,n*sizeof(Complex),cudaMemcpyDeviceToHost,cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));for(auto v:back)if(v!=Complex(0))throw std::runtime_error("GPU sector zero leakage");
        std::cout<<",\"device_array_bytes\":"<<gpu.storage_bytes()<<",\"GPU_error\":"<<defect;
#endif
        std::cout<<"}\n"<<std::flush;worst=std::max(worst,defect);++cases;
    }
    if(worst>2e-13)throw std::runtime_error("sector transform control failed");
    std::cout<<"PASS "<<cases<<" sector exports; rejected="<<rejected<<" maximum_relative="<<worst<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
