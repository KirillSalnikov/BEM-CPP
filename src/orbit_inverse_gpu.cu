#include "orbit_inverse_gpu.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <map>
#include <string>

namespace {
void check(cudaError_t status,const char* operation){if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));}
void blas(cublasStatus_t status,const char* operation){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(operation)+": "+std::to_string(int(status)));}
__global__ void gather(int count,const int* indices,const std::int8_t* signs,const double2* input,double2* values){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto x=input[indices[i]];const double s=signs[i];values[i]=make_double2(s*x.x,s*x.y);
}
__global__ void pivot_columns(const int* dimensions,const int* slot_offsets,const int* pivot_offsets,const int* pivots,double2* values){
    const int block=blockIdx.x,g=threadIdx.x;if(g>=12)return;
    const int d=dimensions[block];auto* column=values+slot_offsets[block]+g*d;
    for(int k=0;k<d;++k){const int other=pivots[pivot_offsets[block]+k];const auto swap=column[k];column[k]=column[other];column[other]=swap;}
}
__global__ void scatter(int n,const int* offsets,const int* slots,const double* weights,const double2* values,double2* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;double real=0,imag=0;
    for(int k=offsets[i];k<offsets[i+1];++k){const auto x=values[slots[k]];real+=weights[k]*x.x;imag+=weights[k]*x.y;}
    output[i]=make_double2(real,imag);
}
}

struct OrbitInverseGPU::Impl {
    struct Group {int dimension=0,count=0;cuDoubleComplex** factors=nullptr;cuDoubleComplex** rhs=nullptr;};
    int n=0,slots=0,blocks=0;size_t bytes=0,budget=0;
    int *indices=nullptr,*dimensions=nullptr,*slot_offsets=nullptr,*pivot_offsets=nullptr,*pivots=nullptr,*row_offsets=nullptr,*scatter_slots=nullptr;
    std::int8_t* signs=nullptr;double* weights=nullptr;
    cuDoubleComplex *factors=nullptr,*values=nullptr,*input=nullptr,*output=nullptr;
    cublasHandle_t handle=nullptr;
    std::vector<void*> allocated;
    std::vector<Group> groups;
    ~Impl(){if(handle)cublasDestroy(handle);for(void* p:allocated)cudaFree(p);}
    template<class T>T* allocate(size_t count){
        if(count>std::numeric_limits<size_t>::max()/sizeof(T) || count*sizeof(T)>budget-bytes)
            throw std::runtime_error("resident orbit inverse exceeds memory budget");
        T* pointer=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&pointer),count*sizeof(T)),"allocate orbit inverse");
        try{allocated.push_back(pointer);}catch(...){cudaFree(pointer);throw;}
        bytes+=count*sizeof(T);return pointer;
    }
    template<class T>T* upload(const std::vector<T>& host){
        T* pointer=allocate<T>(host.size());check(cudaMemcpy(pointer,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice),"upload orbit inverse");return pointer;
    }
};

OrbitInverseGPU::OrbitInverseGPU()=default;
OrbitInverseGPU::~OrbitInverseGPU()=default;
void OrbitInverseGPU::init(const bem_orbit_device::Layout& layout,size_t budget){
    if(layout.dofs<1 || layout.dimensions.empty() || !budget)throw std::invalid_argument("resident orbit dimensions/budget");
    layout.validate();
    std::unique_ptr<Impl> next(new Impl);auto& s=*next;s.budget=budget;s.n=layout.dofs;s.slots=layout.input_indices.size();s.blocks=layout.dimensions.size();
    s.indices=s.upload(layout.input_indices);s.signs=s.upload(layout.input_signs);s.dimensions=s.upload(layout.dimensions);
    s.slot_offsets=s.upload(layout.slot_offsets);s.pivot_offsets=s.upload(layout.pivot_offsets);s.pivots=s.upload(layout.pivots);
    s.row_offsets=s.upload(layout.row_offsets);s.scatter_slots=s.upload(layout.scatter_slots);s.weights=s.upload(layout.scatter_weights);
    s.factors=s.allocate<cuDoubleComplex>(layout.lu.size());
    check(cudaMemcpy(s.factors,layout.lu.data(),layout.lu.size()*sizeof(Complex),cudaMemcpyHostToDevice),"upload orbit LU");
    s.values=s.allocate<cuDoubleComplex>(s.slots);s.input=s.allocate<cuDoubleComplex>(s.n);s.output=s.allocate<cuDoubleComplex>(s.n);
    std::map<int,std::vector<int>> by_dimension;
    for(int b=0;b<s.blocks;++b)by_dimension[layout.dimensions[b]].push_back(b);
    for(const auto& entry:by_dimension){
        Impl::Group group;group.dimension=entry.first;group.count=entry.second.size();
        std::vector<cuDoubleComplex*> a,b;
        for(int block:entry.second){a.push_back(s.factors+layout.lu_offsets[block]);b.push_back(s.values+layout.slot_offsets[block]);}
        group.factors=s.upload(a);group.rhs=s.upload(b);s.groups.push_back(group);
    }
    blas(cublasCreate(&s.handle),"create orbit cuBLAS");blas(cublasSetStream(s.handle,cudaStreamPerThread),"set orbit stream");
    blas(cublasSetPointerMode(s.handle,CUBLAS_POINTER_MODE_HOST),"set orbit scalar mode");
    impl_.swap(next);
}
void OrbitInverseGPU::enqueue_device(const Complex* input,Complex* output){
    if(!impl_ || !input || !output)throw std::invalid_argument("resident orbit application");auto& s=*impl_;
    gather<<<(s.slots+255)/256,256,0,cudaStreamPerThread>>>(s.slots,s.indices,s.signs,reinterpret_cast<const double2*>(input),reinterpret_cast<double2*>(s.values));
    pivot_columns<<<s.blocks,32,0,cudaStreamPerThread>>>(s.dimensions,s.slot_offsets,s.pivot_offsets,s.pivots,reinterpret_cast<double2*>(s.values));
    check(cudaPeekAtLastError(),"orbit gather/pivots");const cuDoubleComplex one=make_cuDoubleComplex(1.,0.);
    for(const auto& group:s.groups){
        // Row-major LU viewed by column-major cuBLAS contains L^T in its
        // upper triangle and U^T in its lower triangle. Ordinary transpose,
        // not conjugate transpose, reconstructs the original complex factors.
        blas(cublasZtrsmBatched(s.handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,CUBLAS_DIAG_UNIT,
            group.dimension,12,&one,group.factors,group.dimension,group.rhs,group.dimension,group.count),"orbit lower solve");
        blas(cublasZtrsmBatched(s.handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,
            group.dimension,12,&one,group.factors,group.dimension,group.rhs,group.dimension,group.count),"orbit upper solve");
    }
    scatter<<<(s.n+255)/256,256,0,cudaStreamPerThread>>>(s.n,s.row_offsets,s.scatter_slots,s.weights,reinterpret_cast<const double2*>(s.values),reinterpret_cast<double2*>(output));
    check(cudaPeekAtLastError(),"orbit scatter");
}
void OrbitInverseGPU::apply(const Complex* input,Complex* output){
    if(!impl_ || !input || !output)throw std::invalid_argument("host orbit application");auto& s=*impl_;
    check(cudaMemcpyAsync(s.input,input,s.n*sizeof(Complex),cudaMemcpyHostToDevice,cudaStreamPerThread),"orbit RHS upload");
    enqueue_device(reinterpret_cast<const Complex*>(s.input),reinterpret_cast<Complex*>(s.output));
    check(cudaMemcpyAsync(output,s.output,s.n*sizeof(Complex),cudaMemcpyDeviceToHost,cudaStreamPerThread),"orbit result download");
    check(cudaStreamSynchronize(cudaStreamPerThread),"orbit host completion");
}
OrbitInverseGPU::Complex* OrbitInverseGPU::device_input(){if(!impl_)throw std::logic_error("orbit not initialized");return reinterpret_cast<Complex*>(impl_->input);}
OrbitInverseGPU::Complex* OrbitInverseGPU::device_output(){if(!impl_)throw std::logic_error("orbit not initialized");return reinterpret_cast<Complex*>(impl_->output);}
size_t OrbitInverseGPU::storage_bytes()const{return impl_?impl_->bytes:0;}
int OrbitInverseGPU::system_dofs()const{return impl_?impl_->n:0;}
