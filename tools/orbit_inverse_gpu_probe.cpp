#include "host_orbit_inverse_layout.h"
#include "prism_orbit_inverse.h"
#include <iomanip>
#include <iostream>
#ifdef BEM_TEST_ORBIT_GPU
#include "orbit_inverse_gpu.h"
#include <cuda_runtime.h>
#endif
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
#ifdef BEM_TEST_ORBIT_GPU
static double seconds(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
#endif
static double relative(Vector a,const Vector& b){
    for(size_t i=0;i<a.size();++i)a[i]-=b[i];
    return bem_streaming::norm(a.data(),a.size())/std::max(bem_streaming::norm(b.data(),b.size()),1e-300);
}
#ifdef BEM_TEST_ORBIT_GPU
static void cuda_check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
#endif
int main(int argc,char** argv)try {
    if(argc!=1 && (argc!=2 || std::string(argv[1])!="--check"))throw std::invalid_argument("probe [--check]");
    int cases=0,rejected=0;double worst=0;
    for(int ref:{1,2,3})for(int nodes:{3,7,50}){
        MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        D6hOrbit group;group.build(mesh);PrismGroupMaps maps;maps.build(group);PrismOrbitInverse original;
        original.build(mesh,maps,nodes,[](const std::vector<int>& ids){
            const int d=4*ids.size();Vector matrix(size_t(d)*d);
            for(int i=0;i<d;++i)for(int j=0;j<d;++j)matrix[i*d+j]=Complex(.0007*std::sin(i+2*j+.01*ids[0]),.0009*std::cos(2*i-j));
            for(int i=0;i<d;++i)matrix[i*d+4*(i/4)+(i+2)%4]+=Complex(2.+.003*i,.3);
            for(int j=0;j<d;++j)std::swap(matrix[j],matrix[size_t(d-1)*d+j]);
            return matrix;
        });
        auto layout=bem_orbit_device::Layout::capture(original);layout.validate();const int n=layout.dofs;
        Vector x(n),expected(n),actual(n),aliased,rotated(n),transformed(n),twice(n);
        for(int i=0;i<n;++i)x[i]=Complex(std::sin(.131*i),std::cos(.217*i));
        original.apply(x.data(),expected.data());layout.apply_host_reference(x.data(),actual.data());
        double error=relative(actual,expected);aliased=x;layout.apply_host_reference(aliased.data(),aliased.data());
        error=std::max(error,relative(aliased,expected));
        std::vector<Complex> overlap(n+3);std::copy(x.begin(),x.end(),overlap.begin());
        layout.apply_host_reference(overlap.data(),overlap.data()+3);actual.assign(overlap.begin()+3,overlap.end());
        error=std::max(error,relative(actual,expected));
        if(error>2e-12)throw std::runtime_error("CPU exported orbit inverse differs from original");
        for(int fault=0;fault<11;++fault){auto broken=layout;
            switch(fault){case 0:broken.dofs=0;break;case 1:broken.dimensions[0]=0;break;
            case 2:broken.pivot_offsets[0]=1;break;case 3:broken.pivots[0]=-1;break;
            case 4:broken.input_indices[0]=n;break;case 5:broken.input_signs[0]=0;break;
            case 6:broken.row_offsets[1]=-1;break;case 7:broken.scatter_slots[0]=-1;break;
            case 8:broken.scatter_weights[0]=0;break;case 9:broken.lu[0]=Complex(std::numeric_limits<double>::quiet_NaN());break;
            case 10:broken.lu_offsets[0]=1;break;}
            bool failed=false;try{broken.validate();}catch(const std::invalid_argument&){failed=true;}
            if(!failed)throw std::runtime_error("corrupt packed orbit layout accepted");
            ++rejected;
        }
        std::cout<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"nodes\":"<<nodes<<",\"dofs\":"<<n
            <<",\"blocks\":"<<original.block_count()<<",\"CPU_export_relative\":"<<error;
#ifdef BEM_TEST_ORBIT_GPU
        OrbitInverseGPU device;device.init(layout,size_t(512)*1024*1024);const size_t bytes=device.storage_bytes();
        device.apply(x.data(),actual.data());double gpu_error=relative(actual,expected);
        aliased=x;device.apply(aliased.data(),aliased.data());gpu_error=std::max(gpu_error,relative(aliased,expected));
        cuda_check(cudaMemcpy(device.device_input(),x.data(),n*sizeof(Complex),cudaMemcpyHostToDevice));
        device.enqueue_device(device.device_input(),device.device_input());
        cuda_check(cudaMemcpyAsync(actual.data(),device.device_input(),n*sizeof(Complex),cudaMemcpyDeviceToHost,cudaStreamPerThread));
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));gpu_error=std::max(gpu_error,relative(actual,expected));
        // Two applications remain entirely on device between calls.
        original.apply(expected.data(),twice.data());
        cuda_check(cudaMemcpy(device.device_input(),x.data(),n*sizeof(Complex),cudaMemcpyHostToDevice));
        device.enqueue_device(device.device_input(),device.device_output());
        device.enqueue_device(device.device_output(),device.device_input());
        cuda_check(cudaMemcpyAsync(actual.data(),device.device_input(),n*sizeof(Complex),cudaMemcpyDeviceToHost,cudaStreamPerThread));
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));gpu_error=std::max(gpu_error,relative(actual,twice));
        double symmetry=0;
        for(bool mirror:{false,true}){
            group.transform(x,rotated,mirror,true);device.apply(rotated.data(),actual.data());
            group.transform(expected,transformed,mirror,false);symmetry=std::max(symmetry,relative(actual,transformed));
        }
        Vector zero(n);device.apply(zero.data(),actual.data());
        if(bem_streaming::norm(actual.data(),n)!=0)throw std::runtime_error("resident orbit nonzero response to zero input");
        if(gpu_error>2e-12 || symmetry>2e-12 || bytes!=device.storage_bytes())throw std::runtime_error("resident orbit accuracy/symmetry/workspace failed");
        double cpu_s=0,host_s=0;
        for(int repeat=0;repeat<3;++repeat){auto t=Clock::now();original.apply(x.data(),actual.data());cpu_s+=seconds(t)/3;
            t=Clock::now();device.apply(x.data(),actual.data());host_s+=seconds(t)/3;}
        cuda_check(cudaMemcpy(device.device_input(),x.data(),n*sizeof(Complex),cudaMemcpyHostToDevice));
        const auto start=Clock::now();for(int repeat=0;repeat<8;++repeat)device.enqueue_device(device.device_input(),device.device_output());
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));const double resident_s=seconds(start)/8;
        std::cout<<",\"GPU_relative\":"<<gpu_error<<",\"C6_horizontal_defect\":"<<symmetry<<",\"explicit_device_bytes\":"<<bytes
            <<",\"cpu_apply_s\":"<<cpu_s<<",\"GPU_including_transfers_s\":"<<host_s<<",\"GPU_resident_s\":"<<resident_s;
        error=std::max({error,gpu_error,symmetry});
#endif
        std::cout<<"}\n";worst=std::max(worst,error);++cases;
    }
    std::cout<<"PASS "<<cases<<" orbit exports; rejected="<<rejected<<" maximum_relative="<<worst<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
