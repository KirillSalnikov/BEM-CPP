#include "sector_transform_gpu.h"
#include <cuda_runtime.h>
#include <string>
namespace {
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(std::string("sector transform GPU: ")+cudaGetErrorString(s));}
__global__ void transform(int blocks,const int* offsets,const int* matrices,const int* matrix_offsets,
                          const int* physical,const int* coordinate,const std::uint16_t* weights,const double2* palette,
                          const double2* input,double2* output,bool encode){
    const int block=blockIdx.x*4+threadIdx.x/32,i=threadIdx.x%32;if(block>=blocks)return;
    const int lo=offsets[block],r=offsets[block+1]-lo;if(i>=r)return;
    const int base=matrix_offsets[matrices[block]];double2 sum=make_double2(0,0);
    for(int j=0;j<r;++j){auto w=palette[weights[base+(encode?i*r+j:j*r+i)]];
        if(encode)w.y=-w.y;const auto x=input[encode?physical[lo+j]:coordinate[lo+j]];
        sum.x+=w.x*x.x-w.y*x.y;sum.y+=w.x*x.y+w.y*x.x;
    }
    output[encode?coordinate[lo+i]:physical[lo+i]]=sum;
}
}
struct SectorTransformGPU::Impl {
    int n=0,blocks=0;size_t bytes=0,budget=0;
    int *offsets=nullptr,*matrices=nullptr,*matrix_offsets=nullptr,*physical=nullptr,*solution=nullptr,*rhs=nullptr;
    std::uint16_t* weights=nullptr;double2* palette=nullptr;std::vector<void*> allocations;
    ~Impl(){for(auto p:allocations)cudaFree(p);}
    template<class T>T* upload(const std::vector<T>& values){
        if(values.size()>std::numeric_limits<size_t>::max()/sizeof(T) || values.size()*sizeof(T)>budget-bytes)
            throw std::runtime_error("sector transform GPU memory budget");
        T* pointer=nullptr;check(cudaMalloc(reinterpret_cast<void**>(&pointer),values.size()*sizeof(T)));
        try{allocations.push_back(pointer);}catch(...){cudaFree(pointer);throw;}
        check(cudaMemcpy(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));bytes+=values.size()*sizeof(T);return pointer;
    }
};
SectorTransformGPU::SectorTransformGPU()=default;
SectorTransformGPU::~SectorTransformGPU()=default;
void SectorTransformGPU::init(const bem_sector_device::Layout& layout,size_t budget){
    layout.validate();if(layout.storage_bytes()>budget)throw std::runtime_error("sector transform GPU memory budget");
    std::unique_ptr<Impl> candidate(new Impl);auto& s=*candidate;s.n=layout.dofs;s.blocks=layout.orbit_matrix.size();s.budget=budget;
    s.offsets=s.upload(layout.orbit_offsets);s.matrices=s.upload(layout.orbit_matrix);s.matrix_offsets=s.upload(layout.matrix_offsets);
    s.physical=s.upload(layout.physical);s.solution=s.upload(layout.solution);s.rhs=s.upload(layout.rhs);
    s.weights=s.upload(layout.weights);s.palette=reinterpret_cast<double2*>(s.upload(layout.palette));impl_.swap(candidate);
}
void SectorTransformGPU::enqueue(const Complex* input,Complex* output,bool encode,bool equation){
    if(!impl_ || !input || !output)throw std::invalid_argument("sector transform GPU application");
    auto& s=*impl_;const auto a=reinterpret_cast<std::uintptr_t>(input),b=reinterpret_cast<std::uintptr_t>(output);
    if((a>b?a-b:b-a)<size_t(s.n)*sizeof(Complex))throw std::invalid_argument("sector transform GPU overlap");
    transform<<<(s.blocks+3)/4,128,0,cudaStreamPerThread>>>(s.blocks,s.offsets,s.matrices,s.matrix_offsets,s.physical,
        equation?s.rhs:s.solution,s.weights,s.palette,reinterpret_cast<const double2*>(input),reinterpret_cast<double2*>(output),encode);
    check(cudaPeekAtLastError());
}
size_t SectorTransformGPU::storage_bytes()const{return impl_?impl_->bytes:0;}
int SectorTransformGPU::system_dofs()const{return impl_?impl_->n:0;}
