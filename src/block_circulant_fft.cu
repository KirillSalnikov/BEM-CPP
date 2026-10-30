#include "block_circulant_fft.h"
#include "krylov_deflation.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>

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
    ~Impl() {if(plan)cufftDestroy(plan);cudaFree(inverse);cudaFree(in);cudaFree(out);}
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
    const size_t bytes=size_t(p.b)*p.c*p.grid*sizeof(double2);
    check(cudaMemcpy(p.in,input,bytes,cudaMemcpyHostToDevice));
    fftcheck(cufftExecZ2Z(p.plan,p.in,p.in,CUFFT_FORWARD));
    contract<<<dim3((p.grid+127)/128,p.b),128>>>(p.inverse,p.in,p.out,p.grid,p.c,p.b,true);
    check(cudaGetLastError());
    fftcheck(cufftExecZ2Z(p.plan,p.out,p.out,CUFFT_INVERSE));
    check(cudaMemcpy(output,p.out,bytes,cudaMemcpyDeviceToHost));
}
void BlockCirculantFFT::apply_transpose(const Complex* input,Complex* output) {
    const size_t n=size_t(impl->b)*impl->c*impl->grid;std::vector<Complex> conjugated(n);
    for(size_t j=0;j<n;++j)conjugated[j]=std::conj(input[j]);
    apply_adjoint(conjugated.data(),output);for(size_t j=0;j<n;++j)output[j]=std::conj(output[j]);
}
size_t BlockCirculantFFT::storage_bytes()const{return impl->bytes;}
int BlockCirculantFFT::fallback_count()const{return impl->fallbacks;}
