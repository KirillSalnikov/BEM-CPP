#include "resident_sector_operator.h"
#include "resident_prism_inverse.h"
#include "muller_fmm.h"
#include <cuda_runtime.h>
#include <string>
namespace {
void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("resident sector operator: ")+cudaGetErrorString(status));}
}
struct ResidentSectorOperator::Impl {
    int n=0;size_t bytes=0;SectorTransformGPU transform;
    MullerFmmOperator* op=nullptr;ResidentPrismInverse* inverse=nullptr;
    Complex *input=nullptr,*output=nullptr;cudaEvent_t ready=nullptr,done=nullptr;
    ~Impl(){cudaFree(input);cudaFree(output);if(ready)cudaEventDestroy(ready);if(done)cudaEventDestroy(done);}
};
ResidentSectorOperator::ResidentSectorOperator()=default;
ResidentSectorOperator::~ResidentSectorOperator()=default;
void ResidentSectorOperator::init(MullerFmmOperator& op,ResidentPrismInverse& inverse,
                                 const bem_sector_device::Layout& transform,size_t budget){
    transform.validate();
    if(!op.device_matvec_available() || op.system_dofs!=transform.dofs || inverse.system_dofs()!=transform.dofs)
        throw std::invalid_argument("resident sector operator requires complete device assembly and matching layouts");
    const size_t vectors=2*size_t(transform.dofs)*sizeof(Complex);
    if(budget<=vectors)throw std::runtime_error("resident sector operator memory budget");
    std::unique_ptr<Impl> candidate(new Impl);auto& s=*candidate;s.n=transform.dofs;s.op=&op;s.inverse=&inverse;
    s.transform.init(transform,budget-vectors);
    check(cudaMalloc(&s.input,s.n*sizeof(Complex)));check(cudaMalloc(&s.output,s.n*sizeof(Complex)));
    check(cudaEventCreateWithFlags(&s.ready,cudaEventDisableTiming));check(cudaEventCreateWithFlags(&s.done,cudaEventDisableTiming));
    s.bytes=s.transform.storage_bytes()+vectors;impl_.swap(candidate);
}
void ResidentSectorOperator::enqueue(const Complex* input,Complex* output,double blend){
    if(!impl_ || !input || !output || !std::isfinite(blend) || blend<0 || blend>1)throw std::invalid_argument("resident sector action");
    auto& s=*impl_;auto* physical_rhs=s.inverse->device_input();auto* physical_current=s.inverse->device_output();
    s.transform.enqueue(input,physical_rhs,false,true);
    s.inverse->enqueue_device(physical_rhs,physical_current,blend);
    // The production assembly is normally compiled with per-thread streams.
    // Event edges also support legacy-compiled assembly without a CPU wait.
    const auto stream=reinterpret_cast<cudaStream_t>(s.op->device_matvec_stream());
    if(stream!=cudaStreamPerThread){check(cudaEventRecord(s.ready,cudaStreamPerThread));check(cudaStreamWaitEvent(stream,s.ready,0));}
    s.op->matvec_device(physical_current,physical_rhs);
    if(stream!=cudaStreamPerThread){check(cudaEventRecord(s.done,stream));check(cudaStreamWaitEvent(cudaStreamPerThread,s.done,0));}
    s.transform.enqueue(physical_rhs,output,true,true);
}
void ResidentSectorOperator::apply(const Complex* input,Complex* output,double blend){
    if(!impl_ || !input || !output || !std::isfinite(blend) || blend<0 || blend>1)throw std::invalid_argument("resident sector host action");
    const size_t bytes=size_t(impl_->n)*sizeof(Complex);
    check(cudaMemcpyAsync(impl_->input,input,bytes,cudaMemcpyHostToDevice,cudaStreamPerThread));
    enqueue(impl_->input,impl_->output,blend);
    check(cudaMemcpyAsync(output,impl_->output,bytes,cudaMemcpyDeviceToHost,cudaStreamPerThread));
    check(cudaStreamSynchronize(cudaStreamPerThread));
}
ResidentSectorOperator::Complex* ResidentSectorOperator::device_input(){if(!impl_)throw std::logic_error("resident sector not initialized");return impl_->input;}
ResidentSectorOperator::Complex* ResidentSectorOperator::device_output(){if(!impl_)throw std::logic_error("resident sector not initialized");return impl_->output;}
size_t ResidentSectorOperator::storage_bytes()const{return impl_?impl_->bytes:0;}
