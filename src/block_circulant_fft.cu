#include "block_circulant_fft.h"
#include "krylov_deflation.h"
#include "host_spectral_inverse.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace {
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(std::string("circulant CUDA: ")+cudaGetErrorString(e));}
void fftcheck(cufftResult e) {if(e!=CUFFT_SUCCESS)throw std::runtime_error("circulant cuFFT: "+std::to_string(e));}
__global__ void contract(const double2* inverse,const double2* in,double2* out,int grid,int channels,int batches,bool adjoint=false) {
    const int f=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;
    if(f>=grid || b>=batches)return;
    for(int row=0;row<channels;++row) {
        double2 sum=make_double2(0,0);
        for(int col=0;col<channels;++col) {
            double2 a=inverse[(adjoint?(size_t(col)*channels+row):(size_t(row)*channels+col))*grid+f];
            if(adjoint)a.y=-a.y;
            const double2 x=in[(size_t(b)*channels+col)*grid+f];
            sum.x+=a.x*x.x-a.y*x.y;sum.y+=a.x*x.y+a.y*x.x;
        }
        out[(size_t(b)*channels+row)*grid+f]=make_double2(sum.x/grid,sum.y/grid);
    }
}
bool invert(std::vector<std::complex<double>> a,int c,std::vector<std::complex<double>>& out) {
    double scale=0;for(auto z:a)scale=std::max(scale,std::abs(z));
    if(!(scale>0) || !std::isfinite(scale))return false;
    for(auto& z:a)z/=scale;
    std::vector<int> pivots;
    if(!bem_krylov::dense_lu_factor(a,c,pivots))return false;
    out.assign(c*c,0);std::vector<std::complex<double>> v(c);
    for(int col=0;col<c;++col) {
        std::fill(v.begin(),v.end(),0);v[col]=1.;
        bem_krylov::dense_lu_solve(a,pivots,c,v.data());
        for(int row=0;row<c;++row) {
            out[row*c+col]=v[row]/scale;
            if(!std::isfinite(std::abs(out[row*c+col])))return false;
        }
    }
    return true;
}
}
struct BlockCirculantFFT::Impl {
    int nx=0,ny=0,grid=0,c=0,b=0,fallbacks=0;
    size_t bytes=0,work=0;
    cufftHandle plan=0;
    double2 *inverse=nullptr,*in=nullptr,*out=nullptr;
    cufftHandle pair_plan=0;double2 *pair_in=nullptr,*pair_out=nullptr;
    std::unique_ptr<bem_cpu::SpectralInverse> cpu;
    bool use_cpu=false;
    ~Impl() {if(plan)cufftDestroy(plan);if(pair_plan)cufftDestroy(pair_plan);cudaFree(pair_in);cudaFree(pair_out);cudaFree(inverse);cudaFree(in);cudaFree(out);}
};
BlockCirculantFFT::BlockCirculantFFT():impl(new Impl){}
BlockCirculantFFT::~BlockCirculantFFT()=default;
void BlockCirculantFFT::init(int nx,int ny,int c,int batches,const std::vector<Complex>& kernel,double shift) {
    if(nx<1 || ny<1 || c<1 || c>32 || batches<1 || kernel.size()!=size_t(c)*c*nx*ny || shift<0 || !std::isfinite(shift))
        throw std::invalid_argument("circulant dimensions/shift");
    std::unique_ptr<Impl> p(new Impl);p->nx=nx;p->ny=ny;p->grid=nx*ny;p->c=c;p->b=batches;
    const size_t count=kernel.size();
    check(cudaMalloc(&p->inverse,count*sizeof(double2)));
    check(cudaMemcpy(p->inverse,kernel.data(),count*sizeof(double2),cudaMemcpyHostToDevice));
    int dims[2]={nx,ny};cufftHandle kernel_plan=0;
    fftcheck(cufftPlanMany(&kernel_plan,2,dims,nullptr,1,p->grid,nullptr,1,p->grid,CUFFT_Z2Z,c*c));
    const cufftResult result=cufftExecZ2Z(kernel_plan,p->inverse,p->inverse,CUFFT_FORWARD);
    cufftDestroy(kernel_plan);fftcheck(result);
    std::vector<Complex> frequency(count),local(c*c),local_inverse;
    check(cudaMemcpy(frequency.data(),p->inverse,count*sizeof(double2),cudaMemcpyDeviceToHost));
    for(int i=0;i<c*c;++i)local[i]=kernel[size_t(i)*p->grid];
    if(!invert(local,c,local_inverse))throw std::runtime_error("singular zero-offset circulant block");
    int fallbacks=0;
#pragma omp parallel for reduction(+:fallbacks) schedule(static)
    for(int f=0;f<p->grid;++f) {
        std::vector<Complex> a(c*c),inv;
        for(int i=0;i<c*c;++i)a[i]=frequency[size_t(i)*p->grid+f]+shift*local[i];
        if(!invert(a,c,inv)) {inv=local_inverse;++fallbacks;}
        for(int i=0;i<c*c;++i)frequency[size_t(i)*p->grid+f]=inv[i];
    }
    p->fallbacks=fallbacks;
    check(cudaMemcpy(p->inverse,frequency.data(),count*sizeof(double2),cudaMemcpyHostToDevice));
    const size_t vectors=size_t(c)*batches*p->grid;
    check(cudaMalloc(&p->in,vectors*sizeof(double2)));check(cudaMalloc(&p->out,vectors*sizeof(double2)));
    fftcheck(cufftPlanMany(&p->plan,2,dims,nullptr,1,p->grid,nullptr,1,p->grid,CUFFT_Z2Z,c*batches));
    fftcheck(cufftGetSize(p->plan,&p->work));
    p->bytes=(count+2*vectors)*sizeof(double2)+p->work;
    impl.swap(p);
}
std::vector<BlockCirculantFFT::Complex> BlockCirculantFFT::inverse_kernel()const {
    const auto& p=*impl;if(!p.plan)throw std::runtime_error("uninitialized circulant");
    // Temporary RAII owner also handles failures while constructing the plan.
    Impl temporary;const size_t count=size_t(p.c)*p.c*p.grid;
    check(cudaMalloc(&temporary.inverse,count*sizeof(double2)));
    check(cudaMemcpy(temporary.inverse,p.inverse,count*sizeof(double2),cudaMemcpyDeviceToDevice));
    int dims[2]={p.nx,p.ny};
    fftcheck(cufftPlanMany(&temporary.plan,2,dims,nullptr,1,p.grid,nullptr,1,p.grid,CUFFT_Z2Z,p.c*p.c));
    fftcheck(cufftExecZ2Z(temporary.plan,temporary.inverse,temporary.inverse,CUFFT_INVERSE));
    std::vector<Complex> kernel(count);
    check(cudaMemcpy(kernel.data(),temporary.inverse,count*sizeof(double2),cudaMemcpyDeviceToHost));
    for(auto& z:kernel)z/=p.grid;
    return kernel;
}
void BlockCirculantFFT::init_inverse_kernel(int nx,int ny,int c,int batches,const std::vector<Complex>& kernel) {
    if(nx<1 || ny<1 || c<1 || c>32 || batches<1 || kernel.size()!=size_t(c)*c*nx*ny)
        throw std::invalid_argument("inverse circulant dimensions");
    std::unique_ptr<Impl> p(new Impl);p->nx=nx;p->ny=ny;p->grid=nx*ny;p->c=c;p->b=batches;
    const size_t count=kernel.size();check(cudaMalloc(&p->inverse,count*sizeof(double2)));
    check(cudaMemcpy(p->inverse,kernel.data(),count*sizeof(double2),cudaMemcpyHostToDevice));
    int dims[2]={nx,ny};
    fftcheck(cufftPlanMany(&p->plan,2,dims,nullptr,1,p->grid,nullptr,1,p->grid,CUFFT_Z2Z,c*c));
    fftcheck(cufftExecZ2Z(p->plan,p->inverse,p->inverse,CUFFT_FORWARD));
    cufftDestroy(p->plan);p->plan=0;
    const size_t vectors=size_t(c)*batches*p->grid;
    check(cudaMalloc(&p->in,vectors*sizeof(double2)));check(cudaMalloc(&p->out,vectors*sizeof(double2)));
    fftcheck(cufftPlanMany(&p->plan,2,dims,nullptr,1,p->grid,nullptr,1,p->grid,CUFFT_Z2Z,c*batches));
    fftcheck(cufftGetSize(p->plan,&p->work));p->bytes=(count+2*vectors)*sizeof(double2)+p->work;
    impl.swap(p);
}
void BlockCirculantFFT::apply(const Complex* input,Complex* output) {
    auto& p=*impl;if(!p.plan)throw std::runtime_error("uninitialized circulant");
    if(p.use_cpu){p.cpu->apply(input,output);return;}
    const size_t bytes=size_t(p.b)*p.c*p.grid*sizeof(double2);
    check(cudaMemcpy(p.in,input,bytes,cudaMemcpyHostToDevice));
    fftcheck(cufftExecZ2Z(p.plan,p.in,p.in,CUFFT_FORWARD));
    contract<<<dim3((p.grid+127)/128,p.b),128>>>(p.inverse,p.in,p.out,p.grid,p.c,p.b);
    check(cudaGetLastError());
    fftcheck(cufftExecZ2Z(p.plan,p.out,p.out,CUFFT_INVERSE));
    check(cudaMemcpy(output,p.out,bytes,cudaMemcpyDeviceToHost));
}
void BlockCirculantFFT::apply_adjoint(const Complex* input,Complex* output) {
    auto& p=*impl;if(!p.plan)throw std::runtime_error("uninitialized circulant");
    if(p.use_cpu){p.cpu->apply(input,output,true);return;}
    const size_t bytes=size_t(p.b)*p.c*p.grid*sizeof(double2);
    check(cudaMemcpy(p.in,input,bytes,cudaMemcpyHostToDevice));
    fftcheck(cufftExecZ2Z(p.plan,p.in,p.in,CUFFT_FORWARD));
    contract<<<dim3((p.grid+127)/128,p.b),128>>>(p.inverse,p.in,p.out,p.grid,p.c,p.b,true);
    check(cudaGetLastError());
    fftcheck(cufftExecZ2Z(p.plan,p.out,p.out,CUFFT_INVERSE));
    check(cudaMemcpy(output,p.out,bytes,cudaMemcpyDeviceToHost));
}
void BlockCirculantFFT::prepare_batch2() {
    auto& p=*impl;if(!p.plan)throw std::runtime_error("uninitialized circulant pair");
    if(p.use_cpu)throw std::runtime_error("GPU pair requires GPU FFT backend");
    if(p.pair_plan)return;
    const size_t bytes=2*size_t(p.b)*p.c*p.grid*sizeof(double2);
    // Keep allocations owned even if cuFFT planning throws.
    if(!p.pair_in)check(cudaMalloc(&p.pair_in,bytes));
    if(!p.pair_out)check(cudaMalloc(&p.pair_out,bytes));
    int dims[2]={p.nx,p.ny};
    fftcheck(cufftPlanMany(&p.pair_plan,2,dims,nullptr,1,p.grid,nullptr,1,p.grid,CUFFT_Z2Z,2*p.c*p.b));
    size_t work=0;fftcheck(cufftGetSize(p.pair_plan,&work));p.bytes+=2*bytes+work;
}
void BlockCirculantFFT::apply_batch2(const Complex* first,const Complex* second,Complex* first_result,Complex* second_result) {
    if(!first || !second || !first_result || !second_result || first_result==second_result)
        throw std::invalid_argument("circulant pair arguments");
    prepare_batch2();auto& p=*impl;
    const size_t count=size_t(p.b)*p.c*p.grid,bytes=count*sizeof(double2);
    check(cudaMemcpy(p.pair_in,first,bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(p.pair_in+count,second,bytes,cudaMemcpyHostToDevice));
    fftcheck(cufftExecZ2Z(p.pair_plan,p.pair_in,p.pair_in,CUFFT_FORWARD));
    contract<<<dim3((p.grid+127)/128,2*p.b),128>>>(p.inverse,p.pair_in,p.pair_out,p.grid,p.c,2*p.b);
    check(cudaGetLastError());fftcheck(cufftExecZ2Z(p.pair_plan,p.pair_out,p.pair_out,CUFFT_INVERSE));
    check(cudaMemcpy(first_result,p.pair_out,bytes,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(second_result,p.pair_out+count,bytes,cudaMemcpyDeviceToHost));
}
void BlockCirculantFFT::apply_transpose(const Complex* input,Complex* output) {
    const size_t n=size_t(impl->b)*impl->c*impl->grid;std::vector<Complex> conjugated(n);
    for(size_t j=0;j<n;++j)conjugated[j]=std::conj(input[j]);
    apply_adjoint(conjugated.data(),output);for(size_t j=0;j<n;++j)output[j]=std::conj(output[j]);
}
size_t BlockCirculantFFT::storage_bytes()const{return impl->bytes;}
int BlockCirculantFFT::fallback_count()const{return impl->fallbacks;}
bool BlockCirculantFFT::prepare_cpu() {
    auto& p=*impl;if(!p.plan)return true;
    if(!bem_cpu::FftwRuntime::get().available())return false;
    if(!p.cpu) {
        std::vector<Complex> symbol(size_t(p.c)*p.c*p.grid);
        check(cudaMemcpy(symbol.data(),p.inverse,symbol.size()*sizeof(Complex),cudaMemcpyDeviceToHost));
        p.cpu.reset(new bem_cpu::SpectralInverse(p.nx,p.ny,p.c,p.b,symbol));
    }
    return true;
}
void BlockCirculantFFT::enable_cpu(bool enabled) {
    if(enabled && impl->plan && !impl->cpu)throw std::runtime_error("CPU circulant not prepared");
    impl->use_cpu=enabled && bool(impl->cpu);
}
size_t BlockCirculantFFT::cpu_storage_bytes()const{return impl->cpu?impl->cpu->storage_bytes():0;}

size_t BlockCirculantFFT::device_vector_size()const {
    return size_t(impl->b)*impl->c*impl->grid;
}
BlockCirculantFFT::Complex* BlockCirculantFFT::device_input() {
    if(!impl->plan || impl->use_cpu)throw std::logic_error("circulant device input requires GPU bank");
    return reinterpret_cast<Complex*>(impl->in);
}
const BlockCirculantFFT::Complex* BlockCirculantFFT::device_output()const {
    if(!impl->plan || impl->use_cpu)throw std::logic_error("circulant device output requires GPU bank");
    return reinterpret_cast<const Complex*>(impl->out);
}
void BlockCirculantFFT::enqueue_device(bool adjoint) {
    auto& p=*impl;
    if(!p.plan || p.use_cpu)throw std::logic_error("circulant device action requires GPU bank");
    fftcheck(cufftExecZ2Z(p.plan,p.in,p.in,CUFFT_FORWARD));
    contract<<<dim3((p.grid+127)/128,p.b),128>>>(p.inverse,p.in,p.out,p.grid,p.c,p.b,adjoint);
    check(cudaGetLastError());
    fftcheck(cufftExecZ2Z(p.plan,p.out,p.out,CUFFT_INVERSE));
}
void BlockCirculantFFT::enqueue_device_per_thread(bool adjoint) {
    auto& p=*impl;
    if(!p.plan || p.use_cpu)throw std::logic_error("circulant device action requires GPU bank");
    fftcheck(cufftSetStream(p.plan,cudaStreamPerThread));
    struct Restore {cufftHandle plan;~Restore(){cufftSetStream(plan,0);}} restore{p.plan};
    fftcheck(cufftExecZ2Z(p.plan,p.in,p.in,CUFFT_FORWARD));
    contract<<<dim3((p.grid+127)/128,p.b),128,0,cudaStreamPerThread>>>(p.inverse,p.in,p.out,p.grid,p.c,p.b,adjoint);
    check(cudaGetLastError());
    fftcheck(cufftExecZ2Z(p.plan,p.out,p.out,CUFFT_INVERSE));
}

namespace {
__global__ void mapped_gather(const double2* input,double2* bank,const MappedBlockCirculantFFT::Slot* slots,
                             size_t count,int parity,double scale) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto s=slots[i];double2 a=make_double2(0,0),b=make_double2(0,0);
    if(s.first>=0)a=input[s.first];
    if(s.second>=0)b=input[s.second];
    bank[i]=make_double2(scale*(s.rhs_first*a.x+parity*s.rhs_second*b.x),
                        scale*(s.rhs_first*a.y+parity*s.rhs_second*b.y));
}
__global__ void mapped_scatter(const double2* bank,double2* output,const MappedBlockCirculantFFT::Slot* slots,
                              size_t count,int parity,double scale) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto s=slots[i];const auto v=bank[i];
    // Injectivity is verified on the host. Different terms are serialized.
    if(s.first>=0){output[s.first].x+=scale*s.solution_first*v.x;output[s.first].y+=scale*s.solution_first*v.y;}
    if(s.second>=0){output[s.second].x+=scale*parity*s.solution_second*v.x;output[s.second].y+=scale*parity*s.solution_second*v.y;}
}
}
struct MappedBlockCirculantFFT::Impl {
    struct Entry {
        BlockCirculantFFT* bank=nullptr;Slot* slots=nullptr;size_t count=0;int parity=0;
        ~Entry(){cudaFree(slots);}
    };
    int dofs=0;size_t bytes=0;double2 *input=nullptr,*output=nullptr;
    std::vector<std::unique_ptr<Entry>> entries;
    ~Impl(){cudaFree(input);cudaFree(output);}
};
MappedBlockCirculantFFT::MappedBlockCirculantFFT():impl(new Impl){}
MappedBlockCirculantFFT::~MappedBlockCirculantFFT()=default;
void MappedBlockCirculantFFT::validate(int dofs,const std::vector<Term>& terms) {
    if(dofs<1 || terms.empty())throw std::invalid_argument("mapped circulant dimensions");
    for(const auto& term:terms){
        if(!term.bank || term.slots.empty() || term.pair_sign<-1 || term.pair_sign>1)
            throw std::invalid_argument("mapped circulant term");
        std::vector<bool> seen(dofs,false);
        for(const auto& s:term.slots){
            if((term.pair_sign==0 && s.second!=-1) || (term.pair_sign!=0 && (s.first<0)!=(s.second<0)))
                throw std::invalid_argument("mapped circulant pair support");
            for(int d:{s.first,s.second}){
                if(d < -1 || d>=dofs || (d>=0 && seen[d]))throw std::invalid_argument("mapped circulant scatter is not injective");
                if(d>=0)seen[d]=true;
            }
            for(int sign:{s.rhs_first,s.rhs_second,s.solution_first,s.solution_second})
                if(sign!=1 && sign!=-1)throw std::invalid_argument("mapped circulant orientation sign");
        }
    }
}
void MappedBlockCirculantFFT::init(int dofs,const std::vector<Term>& terms) {
    validate(dofs,terms);std::unique_ptr<Impl> p(new Impl);p->dofs=dofs;
    for(const auto& term:terms){
        if(term.slots.size()!=term.bank->device_vector_size())throw std::invalid_argument("mapped circulant bank size");
        term.bank->device_input(); // also rejects CPU banks before allocating
    }
    check(cudaMalloc(&p->input,size_t(dofs)*sizeof(double2)));
    check(cudaMalloc(&p->output,size_t(dofs)*sizeof(double2)));p->bytes=2*size_t(dofs)*sizeof(double2);
    for(const auto& term:terms){
        std::unique_ptr<Impl::Entry> entry(new Impl::Entry);
        entry->bank=term.bank;entry->parity=term.pair_sign;entry->count=term.slots.size();
        const size_t bytes=entry->count*sizeof(Slot);
        check(cudaMalloc(&entry->slots,bytes));check(cudaMemcpy(entry->slots,term.slots.data(),bytes,cudaMemcpyHostToDevice));
        p->bytes+=bytes;p->entries.push_back(std::move(entry));
    }
    impl.swap(p);
}
void MappedBlockCirculantFFT::apply(const Complex* input,Complex* output) {
    auto& p=*impl;if(!p.dofs || !input || !output)throw std::invalid_argument("mapped circulant apply");
    const size_t bytes=size_t(p.dofs)*sizeof(double2);
    check(cudaMemcpy(p.input,input,bytes,cudaMemcpyHostToDevice));
    check(cudaMemsetAsync(p.output,0,bytes));
    for(const auto& entry:p.entries){
        const auto& e=*entry;const int blocks=int((e.count+255)/256);const double scale=e.parity?1/std::sqrt(2.):1.;
        mapped_gather<<<blocks,256>>>(p.input,reinterpret_cast<double2*>(e.bank->device_input()),e.slots,e.count,e.parity,scale);
        check(cudaGetLastError());e.bank->enqueue_device();
        mapped_scatter<<<blocks,256>>>(reinterpret_cast<const double2*>(e.bank->device_output()),p.output,e.slots,e.count,e.parity,scale);
        check(cudaGetLastError());
    }
    check(cudaMemcpy(output,p.output,bytes,cudaMemcpyDeviceToHost));
}
size_t MappedBlockCirculantFFT::storage_bytes()const{return impl->bytes;}
void MappedBlockCirculantFFT::enqueue_device(const Complex* input,Complex* output) {
    auto& p=*impl;
    if(!p.dofs || !input || !output || input==output)throw std::invalid_argument("mapped device arguments/alias");
    const auto a=reinterpret_cast<std::uintptr_t>(input),b=reinterpret_cast<std::uintptr_t>(output);
    if((a>b?a-b:b-a)<size_t(p.dofs)*sizeof(Complex))throw std::invalid_argument("mapped device overlapping arrays");
    const auto* in=reinterpret_cast<const double2*>(input);auto* out=reinterpret_cast<double2*>(output);
    check(cudaMemsetAsync(out,0,size_t(p.dofs)*sizeof(double2),cudaStreamPerThread));
    for(const auto& entry:p.entries){
        const auto& e=*entry;const int blocks=int((e.count+255)/256);const double scale=e.parity?1/std::sqrt(2.):1.;
        mapped_gather<<<blocks,256,0,cudaStreamPerThread>>>(in,reinterpret_cast<double2*>(e.bank->device_input()),e.slots,e.count,e.parity,scale);
        check(cudaGetLastError());e.bank->enqueue_device_per_thread();
        mapped_scatter<<<blocks,256,0,cudaStreamPerThread>>>(reinterpret_cast<const double2*>(e.bank->device_output()),out,e.slots,e.count,e.parity,scale);
        check(cudaGetLastError());
    }
}
