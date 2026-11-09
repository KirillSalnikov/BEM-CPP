#include "resident_prism_inverse.h"
#include "prism_circulant_inverse.h"
#include <cuda_runtime.h>
#include <string>
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("resident inverse: ")+cudaGetErrorString(status));}
__global__ void reflect(int n,const int* image,const int* sign,const double2* input,double2* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const auto v=input[i];const double s=-sign[i];output[image[i]]=make_double2(s*v.x,s*v.y);
}
__global__ void combine(int n,const double* coverage,const int* image,const int* sign,
                        const double2* first,const double2* reflected,const double2* fft,double blend,double2* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const int j=image[i];
    const double a=.5*(1-blend*coverage[i]),b=.5*sign[j]*(1-blend*coverage[j]);
    const auto x=first[i],y=reflected[j],z=fft[i];
    output[i]=make_double2(a*x.x+b*y.x+blend*z.x,a*x.y+b*y.y+blend*z.y);
}
}
struct ResidentPrismInverse::Impl {
    int n=0;size_t bytes=0;OrbitInverseGPU local;PrismCirculantInverse* fft=nullptr;
    double* coverage=nullptr;int *image=nullptr,*sign=nullptr;double2 *first=nullptr,*reflected=nullptr,*fft_result=nullptr;
    ~Impl(){cudaFree(coverage);cudaFree(image);cudaFree(sign);cudaFree(first);cudaFree(reflected);cudaFree(fft_result);}
};
ResidentPrismInverse::ResidentPrismInverse()=default;
ResidentPrismInverse::~ResidentPrismInverse()=default;
void ResidentPrismInverse::init(const bem_orbit_device::Layout& local,const bem_orbit_device::SymmetricLayout& sym,
                                PrismCirculantInverse& fft,size_t budget){
    local.validate();sym.validate();
    if(sym.coverage.size()!=size_t(local.dofs) || fft.layout.system_dofs!=local.dofs || !fft.device_mapping_matches(true))
        throw std::invalid_argument("resident inverse requires matching symmetric GPU FFT");
    const size_t extra=size_t(local.dofs)*(sizeof(double)+2*sizeof(int)+3*sizeof(double2));
    if(budget<=extra)throw std::runtime_error("resident inverse memory budget");
    std::unique_ptr<Impl> candidate(new Impl);auto& s=*candidate;s.n=local.dofs;s.fft=&fft;
    s.local.init(local,budget-extra);
    check(cudaMalloc(&s.coverage,s.n*sizeof(double)));check(cudaMalloc(&s.image,s.n*sizeof(int)));check(cudaMalloc(&s.sign,s.n*sizeof(int)));
    check(cudaMemcpy(s.coverage,sym.coverage.data(),s.n*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.image,sym.vertical_image.data(),s.n*sizeof(int),cudaMemcpyHostToDevice));
    check(cudaMemcpy(s.sign,sym.vertical_sign.data(),s.n*sizeof(int),cudaMemcpyHostToDevice));
    check(cudaMalloc(&s.first,s.n*sizeof(double2)));check(cudaMalloc(&s.reflected,s.n*sizeof(double2)));check(cudaMalloc(&s.fft_result,s.n*sizeof(double2)));
    s.bytes=s.local.storage_bytes()+extra;impl_.swap(candidate);
}
void ResidentPrismInverse::enqueue_device(const Complex* input,Complex* output,double blend){
    if(!impl_ || !input || !output || !std::isfinite(blend) || blend<0 || blend>1)
        throw std::invalid_argument("resident inverse application");
    auto& s=*impl_;
    if(!s.fft->device_mapping_matches(true))throw std::logic_error("resident FFT mapping was changed");
    s.local.enqueue_device(input,reinterpret_cast<Complex*>(s.first));
    reflect<<<(s.n+255)/256,256,0,cudaStreamPerThread>>>(s.n,s.image,s.sign,reinterpret_cast<const double2*>(input),s.reflected);
    check(cudaPeekAtLastError());
    s.local.enqueue_device(reinterpret_cast<const Complex*>(s.reflected),reinterpret_cast<Complex*>(s.reflected));
    s.fft->enqueue_fft_part_device(input,reinterpret_cast<Complex*>(s.fft_result),true);
    combine<<<(s.n+255)/256,256,0,cudaStreamPerThread>>>(s.n,s.coverage,s.image,s.sign,s.first,s.reflected,s.fft_result,blend,reinterpret_cast<double2*>(output));
    check(cudaPeekAtLastError());
}
void ResidentPrismInverse::apply(const Complex* input,Complex* output,double blend){
    if(!impl_ || !input || !output || !std::isfinite(blend) || blend<0 || blend>1)throw std::invalid_argument("resident host arguments");
    const size_t bytes=size_t(impl_->n)*sizeof(Complex);
    check(cudaMemcpyAsync(device_input(),input,bytes,cudaMemcpyHostToDevice,cudaStreamPerThread));
    enqueue_device(device_input(),device_output(),blend);
    check(cudaMemcpyAsync(output,device_output(),bytes,cudaMemcpyDeviceToHost,cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
}
ResidentPrismInverse::Complex* ResidentPrismInverse::device_input(){if(!impl_)throw std::logic_error("resident inverse uninitialized");return impl_->local.device_input();}
ResidentPrismInverse::Complex* ResidentPrismInverse::device_output(){if(!impl_)throw std::logic_error("resident inverse uninitialized");return impl_->local.device_output();}
size_t ResidentPrismInverse::storage_bytes()const{return impl_?impl_->bytes:0;}
int ResidentPrismInverse::system_dofs()const{return impl_?impl_->n:0;}
