#include "regular_face_fft.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string("face FFT CUDA: ")+cudaGetErrorString(e));
}
void fftcheck(cufftResult e) {
    if(e!=CUFFT_SUCCESS) throw std::runtime_error("face cuFFT error "+std::to_string(int(e)));
}
int padded(int n) {
    if(n<1 || n>16384) throw std::runtime_error("face FFT dimension must be 1..16384");
    int m=1; while(m<2*n-1) m*=2; return m;
}
__device__ double2 add(double2 a,double2 b) { return make_double2(a.x+b.x,a.y+b.y); }
__device__ double2 sub(double2 a,double2 b) { return make_double2(a.x-b.x,a.y-b.y); }
__device__ double2 mul(double2 a,double2 b) { return make_double2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ double2 scale(double2 a,double b) { return make_double2(a.x*b,a.y*b); }

__global__ void kernels(double2* out,int nx,int ny,int fx,int fy,
                        double3 u,double3 v,double3 offset,double kr,double ki,int components) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int n=fx*fy; if(i>=n) return;
    const int ix=i/fy,iy=i%fy;
    const int dx=ix<nx?ix:ix-fx,dy=iy<ny?iy:iy-fy;
    double2 g[9]; for(int a=0;a<9;++a) g[a]=make_double2(0,0);
    if(abs(dx)<nx && abs(dy)<ny) {
        const double x=dx*u.x+dy*v.x+offset.x;
        const double y=dx*u.y+dy*v.y+offset.y;
        const double z=dx*u.z+dy*v.z+offset.z;
        const double r2=x*x+y*y+z*z;
        if(r2>1e-28) {
            const double r=sqrt(r2),inv=1/r;
            double sn,cs; sincos(kr*r,&sn,&cs);
            const double magnitude=0.07957747154594766788*exp(-ki*r)*inv;
            const double2 green=make_double2(magnitude*cs,magnitude*sn);
            const double2 radial=mul(green,make_double2((-ki-inv)*inv,kr*inv));
            g[0]=scale(radial,x);g[1]=scale(radial,y);g[2]=scale(radial,z);
            const double2 a=mul(green,make_double2(3*inv*inv+3*ki*inv-kr*kr+ki*ki,
                                                   -3*kr*inv-2*kr*ki));
            const double2 b=mul(green,make_double2(inv*inv+ki*inv,-kr*inv));
            g[3]=sub(scale(a,x*x/r2),b);g[4]=scale(a,x*y/r2);g[5]=scale(a,x*z/r2);
            g[6]=sub(scale(a,y*y/r2),b);g[7]=scale(a,y*z/r2);g[8]=sub(scale(a,z*z/r2),b);
        }
    }
    const int planar[6]={0,1,3,4,6,8};
    for(int a=0;a<components;++a) out[size_t(a)*n+i]=g[components==6?planar[a]:a];
}

__global__ void scatter(const double2* input,double2* spectrum,int nx,int ny,int fy,int grid,int channels) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int cells=nx*ny;if(i>=channels*3*cells) return;
    const int lane=i/cells,cell=i%cells;
    spectrum[size_t(lane)*grid+(cell/ny)*fy+cell%ny]=input[i];
}

__global__ void contract(const double2* kernels,const double2* charges,double2* output,int grid,int channels,
                         int fx,int fy,int components,bool parity) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int t=blockIdx.y;if(i>=grid) return;
    double2 result[6];for(int a=0;a<6;++a) result[a]=make_double2(0,0);
    for(int s=0;s<channels;++s) {
        double2 g[9];for(int a=0;a<9;++a)g[a]=make_double2(0,0);
        const bool reversed=parity && s>t;
        const int hi=t>s?t:s,lo=t<s?t:s;
        const int frequency=reversed?((fx-i/fy)%fx)*fy+(fy-i%fy)%fy:i;
        const size_t pair=parity?size_t(hi)*(hi+1)/2+lo:size_t(t)*channels+s;
        const size_t base=pair*components*grid+frequency;
        const int planar[6]={0,1,3,4,6,8};
        for(int a=0;a<components;++a) {
            const int component=components==6?planar[a]:a;
            g[component]=kernels[base+size_t(a)*grid];
            if(reversed && component<3)g[component]=scale(g[component],-1.);
        }
        const double2 x=charges[size_t(3*s)*grid+i];
        const double2 y=charges[size_t(3*s+1)*grid+i];
        const double2 z=charges[size_t(3*s+2)*grid+i];
        result[0]=add(result[0],sub(mul(g[1],x),mul(g[0],y)));
        result[1]=add(result[1],sub(mul(g[2],x),mul(g[0],z)));
        result[2]=add(result[2],sub(mul(g[2],y),mul(g[1],z)));
        result[3]=add(result[3],sub(add(mul(g[4],y),mul(g[5],z)),mul(add(g[6],g[8]),x)));
        result[4]=add(result[4],sub(add(mul(g[4],x),mul(g[7],z)),mul(add(g[3],g[8]),y)));
        result[5]=add(result[5],sub(add(mul(g[5],x),mul(g[7],y)),mul(add(g[3],g[6]),z)));
    }
    for(int a=0;a<6;++a) output[size_t(t*6+a)*grid+i]=result[a];
}

__global__ void gather(const double2* spectrum,double2* output,int nx,int ny,int fy,int grid,int channels) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const int cells=nx*ny;if(i>=channels*6*cells) return;
    const int lane=i/cells,cell=i%cells;
    output[i]=scale(spectrum[size_t(lane)*grid+(cell/ny)*fy+cell%ny],1./grid);
}
}

struct RegularFaceFFT::Impl {
    int nx=0,ny=0,fx=0,fy=0,grid=0,channels=0;
    int components=9;bool parity=false,ready=false;
    double2 *kernel=nullptr,*qhat=nullptr,*ohat=nullptr,*input=nullptr,*output=nullptr;
    void* workspace=nullptr;
    cufftHandle kplan=0,qplan=0,oplan=0;
    size_t bytes=0;
    ~Impl() {
        if(kplan)cufftDestroy(kplan);if(qplan)cufftDestroy(qplan);if(oplan)cufftDestroy(oplan);
        cudaFree(kernel);cudaFree(qhat);cudaFree(ohat);cudaFree(input);cudaFree(output);cudaFree(workspace);
    }
};
RegularFaceFFT::RegularFaceFFT():p(new Impl){}
RegularFaceFFT::~RegularFaceFFT()=default;
void RegularFaceFFT::cleanup(){p.reset(new Impl);}
std::size_t RegularFaceFFT::storage_bytes() const{return p->bytes;}
std::size_t RegularFaceFFT::estimated_array_bytes(int nx,int ny,int c) {
    if(c<1||c>128) throw std::runtime_error("face FFT channels must be 1..128");
    const size_t grid=size_t(padded(nx))*padded(ny);
    if(grid>INT_MAX || size_t(c)*6*nx*ny>INT_MAX) throw std::runtime_error("face FFT indexing limit");
    return sizeof(double2)*(grid*(size_t(9)*c*c+9*c)+size_t(nx)*ny*9*c);
}
void RegularFaceFFT::init(int nx,int ny,const double u[3],const double v[3],
                         const std::vector<double>& source,const std::vector<double>& target,Complex k) {
    cleanup();
    if(source.empty() || source.size()%3 || source.size()!=target.size() || !std::isfinite(k.real()) || !std::isfinite(k.imag()))
        throw std::runtime_error("invalid face FFT offsets or wave number");
    for(double value:source)if(!std::isfinite(value))throw std::runtime_error("nonfinite source offset");
    for(double value:target)if(!std::isfinite(value))throw std::runtime_error("nonfinite target offset");
    for(int a=0;a<3;++a)if(!std::isfinite(u[a])||!std::isfinite(v[a]))throw std::runtime_error("nonfinite lattice step");
    const double cross[3]={u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]};
    if(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]<=0)throw std::runtime_error("degenerate lattice");
    p->nx=nx;p->ny=ny;p->fx=padded(nx);p->fy=padded(ny);p->grid=p->fx*p->fy;p->channels=int(source.size()/3);
    estimated_array_bytes(nx,ny,p->channels); // Validate indexing before allocation.
    const char* flag=std::getenv("BEM_FACE_FFT_COMPACT");
    const bool compact=!flag || std::string(flag)!="0";
    p->parity=compact && source==target;
    bool xy=u[2]==0 && v[2]==0;
    for(size_t i=2;i<source.size();i+=3)xy=xy && source[i]==source[2] && target[i]==source[2];
    p->components=compact && xy?6:9;
    const size_t pairs=p->parity?size_t(p->channels)*(p->channels+1)/2:size_t(p->channels)*p->channels;
    p->bytes=sizeof(double2)*(size_t(p->grid)*(pairs*p->components+9*p->channels)+size_t(nx)*ny*9*p->channels);
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
    if(p->bytes>free || free-p->bytes<2ULL*1024*1024*1024)
        throw std::runtime_error("face FFT arrays would violate 2 GiB GPU reserve");
    const size_t grid=p->grid,c=p->channels,cells=size_t(nx)*ny;
    check(cudaMalloc(&p->kernel,grid*pairs*p->components*sizeof(double2)));
    check(cudaMalloc(&p->qhat,grid*c*3*sizeof(double2)));
    check(cudaMalloc(&p->ohat,grid*c*6*sizeof(double2)));
    check(cudaMalloc(&p->input,cells*c*3*sizeof(double2)));
    check(cudaMalloc(&p->output,cells*c*6*sizeof(double2)));
    const int dims[2]={p->fx,p->fy};int n[2]={dims[0],dims[1]};
    size_t work=0;
    auto plan=[&](cufftHandle& handle,int batch) {
        fftcheck(cufftCreate(&handle));fftcheck(cufftSetAutoAllocation(handle,0));
        size_t bytes=0;fftcheck(cufftMakePlanMany(handle,2,n,nullptr,1,p->grid,nullptr,1,p->grid,CUFFT_Z2Z,batch,&bytes));
        work=std::max(work,bytes);
    };
    plan(p->kplan,p->components);plan(p->qplan,int(3*c));plan(p->oplan,int(6*c));
    check(cudaMemGetInfo(&free,&total));
    if(work>free || free-work<2ULL*1024*1024*1024)throw std::runtime_error("face FFT workspace would violate GPU reserve");
    if(work)check(cudaMalloc(&p->workspace,work));
    for(cufftHandle handle:{p->kplan,p->qplan,p->oplan})fftcheck(cufftSetWorkArea(handle,p->workspace));
    p->bytes+=work;
    for(int t=0;t<int(c);++t)for(int s=0;s<(p->parity?t+1:int(c));++s) {
        const size_t pair=p->parity?size_t(t)*(t+1)/2+s:size_t(t)*c+s;
        double2* kernel=p->kernel+pair*p->components*grid;
        kernels<<<(grid+255)/256,256>>>(kernel,nx,ny,p->fx,p->fy,
            make_double3(u[0],u[1],u[2]),make_double3(v[0],v[1],v[2]),
            make_double3(target[3*t]-source[3*s],target[3*t+1]-source[3*s+1],target[3*t+2]-source[3*s+2]),k.real(),k.imag(),p->components);
        check(cudaGetLastError());fftcheck(cufftExecZ2Z(p->kplan,kernel,kernel,CUFFT_FORWARD));
    }
    check(cudaDeviceSynchronize());
    p->ready=true;
}
void RegularFaceFFT::evaluate(const Complex* charges,Complex* result) {
    if(!p->ready)throw std::runtime_error("face FFT is not initialized");
    const size_t count=size_t(p->nx)*p->ny*p->channels;
    check(cudaMemcpy(p->input,charges,3*count*sizeof(double2),cudaMemcpyHostToDevice));
    evaluate_device();
    check(cudaMemcpy(result,p->output,6*count*sizeof(double2),cudaMemcpyDeviceToHost));
}
RegularFaceFFT::Complex* RegularFaceFFT::device_input() {
    if(!p->ready)throw std::runtime_error("face FFT is not initialized");
    return reinterpret_cast<Complex*>(p->input);
}
const RegularFaceFFT::Complex* RegularFaceFFT::device_output() const {
    if(!p->ready)throw std::runtime_error("face FFT is not initialized");
    return reinterpret_cast<const Complex*>(p->output);
}
void RegularFaceFFT::evaluate_device() {
    if(!p->ready)throw std::runtime_error("face FFT is not initialized");
    const size_t count=size_t(p->nx)*p->ny*p->channels;
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    // Each medium owns its plan; execute it on its current worker's stream.
    fftcheck(cufftSetStream(p->qplan,cudaStreamPerThread));
    fftcheck(cufftSetStream(p->oplan,cudaStreamPerThread));
#endif
    check(cudaMemset(p->qhat,0,size_t(p->grid)*p->channels*3*sizeof(double2)));
    scatter<<<(3*count+255)/256,256>>>(p->input,p->qhat,p->nx,p->ny,p->fy,p->grid,p->channels);
    check(cudaGetLastError());fftcheck(cufftExecZ2Z(p->qplan,p->qhat,p->qhat,CUFFT_FORWARD));
    contract<<<dim3((p->grid+127)/128,p->channels),128>>>(p->kernel,p->qhat,p->ohat,p->grid,p->channels,
                                                                      p->fx,p->fy,p->components,p->parity);
    check(cudaGetLastError());fftcheck(cufftExecZ2Z(p->oplan,p->ohat,p->ohat,CUFFT_INVERSE));
    gather<<<(6*count+255)/256,256>>>(p->ohat,p->output,p->nx,p->ny,p->fy,p->grid,p->channels);
    check(cudaGetLastError());
}
