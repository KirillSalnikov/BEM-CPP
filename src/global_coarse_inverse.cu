#include "global_coarse_inverse.h"
#include "krylov_deflation.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("coarse CUDA: ")+cudaGetErrorString(e));}
void blas(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("coarse BLAS: "+std::to_string(e));}
const cuDoubleComplex one=make_cuDoubleComplex(1,0),zero=make_cuDoubleComplex(0,0),minus=make_cuDoubleComplex(-1,0);
}
struct GlobalCoarseInverse::Impl {
    int n=0,capacity=0,available=0,active=0;
    double scale=1;
    cublasHandle_t handle=nullptr;
    cuDoubleComplex *v=nullptr,*update=nullptr,*work=nullptr,*coeff=nullptr;
    std::vector<Complex> projected,lu,beta;
    std::vector<int> pivots;
    Action local;
    ~Impl(){if(handle)cublasDestroy(handle);cudaFree(v);cudaFree(update);cudaFree(work);cudaFree(coeff);}
};
GlobalCoarseInverse::GlobalCoarseInverse():impl(new Impl){}
GlobalCoarseInverse::~GlobalCoarseInverse()=default;
void GlobalCoarseInverse::build(int n,int candidates,const Generator& generate,const Action& action,const Action& local) {
    if(n<1 || candidates<1 || candidates>1024)throw std::invalid_argument("coarse dimensions");
    std::unique_ptr<Impl> p(new Impl);p->n=n;p->capacity=candidates;p->local=local;
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
    const size_t bytes=(size_t(2)*n*candidates+n+candidates)*sizeof(Complex);
    if(bytes+512ull*1024*1024>free)throw std::runtime_error("insufficient GPU memory for global coarse space");
    check(cudaMalloc(&p->v,size_t(n)*candidates*sizeof(Complex)));
    check(cudaMalloc(&p->update,size_t(n)*candidates*sizeof(Complex)));
    check(cudaMalloc(&p->work,size_t(n)*sizeof(Complex)));
    check(cudaMalloc(&p->coeff,size_t(candidates)*sizeof(Complex)));
    blas(cublasCreate(&p->handle));
    std::vector<Complex> x(n),y(n),by(n);
    for(int j=0;j<candidates;++j) {
        generate(j,x.data());check(cudaMemcpy(p->work,x.data(),size_t(n)*sizeof(Complex),cudaMemcpyHostToDevice));
        double initial=0,length=0;blas(cublasDznrm2(p->handle,n,p->work,1,&initial));
        if(!std::isfinite(initial))throw std::runtime_error("nonfinite coarse candidate");
        for(int pass=0;pass<2 && p->available;++pass) {
            blas(cublasZgemv(p->handle,CUBLAS_OP_C,n,p->available,&one,p->v,n,p->work,1,&zero,p->coeff,1));
            blas(cublasZgemv(p->handle,CUBLAS_OP_N,n,p->available,&minus,p->v,n,p->coeff,1,&one,p->work,1));
        }
        blas(cublasDznrm2(p->handle,n,p->work,1,&length));
        if(!std::isfinite(length))throw std::runtime_error("nonfinite coarse orthogonalization");
        if(length<=1e-9*std::max(initial,1e-300))continue;
        const cuDoubleComplex scale=make_cuDoubleComplex(1/length,0);
        blas(cublasZscal(p->handle,n,&scale,p->work,1));
        check(cudaMemcpy(p->v+size_t(p->available)*n,p->work,size_t(n)*sizeof(Complex),cudaMemcpyDeviceToDevice));
        ++p->available;
    }
    if(!p->available)throw std::runtime_error("empty coarse basis");
    const int r=p->available;p->projected.resize(size_t(r)*r);p->beta.resize(r);
    for(int j=0;j<r;++j) {
        check(cudaMemcpy(x.data(),p->v+size_t(j)*n,size_t(n)*sizeof(Complex),cudaMemcpyDeviceToHost));
        action(x.data(),y.data());
        check(cudaMemcpy(p->work,y.data(),size_t(n)*sizeof(Complex),cudaMemcpyHostToDevice));
        blas(cublasZgemv(p->handle,CUBLAS_OP_C,n,r,&one,p->v,n,p->work,1,&zero,p->coeff,1));
        check(cudaMemcpy(p->beta.data(),p->coeff,size_t(r)*sizeof(Complex),cudaMemcpyDeviceToHost));
        for(int i=0;i<r;++i)p->projected[size_t(i)*r+j]=p->beta[i];
        local(y.data(),by.data());
        for(int i=0;i<n;++i)x[i]-=by[i];
        check(cudaMemcpy(p->update+size_t(j)*n,x.data(),size_t(n)*sizeof(Complex),cudaMemcpyHostToDevice));
        if((j+1)%16==0 || j+1==r)std::printf("[wave coarse] full-operator sample %d/%d\n",j+1,r);
    }
    impl.swap(p);set_rank(r);
    std::printf("[wave coarse] rank=%d/%d GPU_MiB=%.3f\n",r,candidates,storage_bytes()/1048576.);
}
void GlobalCoarseInverse::set_rank(int r) {
    auto& p=*impl;if(r<0 || r>p.available)throw std::invalid_argument("coarse rank");
    p.active=r;if(!r)return;
    p.lu.resize(size_t(r)*r);p.scale=0;
    for(int i=0;i<r;++i)for(int j=0;j<r;++j) {
        const Complex z=p.projected[size_t(i)*p.available+j];p.lu[size_t(i)*r+j]=z;p.scale=std::max(p.scale,std::abs(z));
    }
    if(!(p.scale>0) || !std::isfinite(p.scale))throw std::runtime_error("invalid coarse Galerkin matrix");
    for(auto& z:p.lu)z/=p.scale;
    if(!bem_krylov::dense_lu_factor(p.lu,r,p.pivots))throw std::runtime_error("singular coarse Galerkin matrix");
}
void GlobalCoarseInverse::apply(const Complex* rhs,Complex* out,double weight) {
    auto& p=*impl;if(!p.local)throw std::runtime_error("uninitialized coarse inverse");
    if(!std::isfinite(weight))throw std::invalid_argument("coarse weight");
    std::vector<Complex> alias;
    if(rhs==out){alias.assign(rhs,rhs+p.n);rhs=alias.data();}
    if(!p.active || weight==0){p.local(rhs,out);return;}
    check(cudaMemcpy(p.work,rhs,size_t(p.n)*sizeof(Complex),cudaMemcpyHostToDevice));
    blas(cublasZgemv(p.handle,CUBLAS_OP_C,p.n,p.active,&one,p.v,p.n,p.work,1,&zero,p.coeff,1));
    check(cudaMemcpy(p.beta.data(),p.coeff,size_t(p.active)*sizeof(Complex),cudaMemcpyDeviceToHost));
    for(int j=0;j<p.active;++j)p.beta[j]/=p.scale;
    bem_krylov::dense_lu_solve(p.lu,p.pivots,p.active,p.beta.data());
    // Finish the projection before invoking local: it can use cuFFT or its
    // own CUDA stream and scratch. The local action is fixed throughout build.
    p.local(rhs,out);
    check(cudaMemcpy(p.work,out,size_t(p.n)*sizeof(Complex),cudaMemcpyHostToDevice));
    check(cudaMemcpy(p.coeff,p.beta.data(),size_t(p.active)*sizeof(Complex),cudaMemcpyHostToDevice));
    const cuDoubleComplex alpha=make_cuDoubleComplex(weight,0);
    blas(cublasZgemv(p.handle,CUBLAS_OP_N,p.n,p.active,&alpha,p.update,p.n,p.coeff,1,&one,p.work,1));
    check(cudaMemcpy(out,p.work,size_t(p.n)*sizeof(Complex),cudaMemcpyDeviceToHost));
}
int GlobalCoarseInverse::rank()const{return impl->active;}
int GlobalCoarseInverse::available_rank()const{return impl->available;}
size_t GlobalCoarseInverse::storage_bytes()const{return (size_t(2)*impl->n*impl->capacity+impl->n+impl->capacity)*sizeof(Complex);}

struct DisjointCoarseInverse::Impl {
    int n=0,r=0;
    double scale=0;
    std::vector<int> offsets,rows,pivots;
    std::vector<double> weights;
    std::vector<Complex> lu,beta;
    Action local;
    cublasHandle_t handle=nullptr;
    cuDoubleComplex *update=nullptr,*work=nullptr,*coeff=nullptr;
    ~Impl(){if(handle)cublasDestroy(handle);cudaFree(update);cudaFree(work);cudaFree(coeff);}
    void project(const Complex* x,std::vector<Complex>& out)const {
        out.resize(r);
#pragma omp parallel for schedule(static)
        for(int j=0;j<r;++j) {
            Complex sum=0;
            for(int k=offsets[j];k<offsets[j+1];++k)sum+=weights[k]*x[rows[k]];
            out[j]=sum;
        }
    }
};
DisjointCoarseInverse::DisjointCoarseInverse():impl(new Impl){}
DisjointCoarseInverse::~DisjointCoarseInverse()=default;
void DisjointCoarseInverse::build(int columns,const std::vector<int>& column_of,const std::vector<double>& weight,
        const Sample& sample,const Action& local) {
    const int n=column_of.size();
    if(n<1 || weight.size()!=column_of.size() || columns<1 || columns>1024 || !sample || !local)
        throw std::invalid_argument("disjoint coarse dimensions/actions");
    std::unique_ptr<Impl> p(new Impl);p->n=n;p->r=columns;p->local=local;p->offsets.assign(columns+1,0);
    std::vector<double> norms(columns,0.);
    for(int i=0;i<n;++i) {
        const int j=column_of[i];
        if(j< -1 || j>=columns || !std::isfinite(weight[i]) || (j<0 && weight[i]!=0))
            throw std::invalid_argument("disjoint coarse row");
        if(j>=0){++p->offsets[j+1];norms[j]+=weight[i]*weight[i];}
    }
    for(int j=0;j<columns;++j) {
        if(!p->offsets[j+1] || std::abs(norms[j]-1)>1e-10)throw std::invalid_argument("disjoint coarse columns must be orthonormal");
        p->offsets[j+1]+=p->offsets[j];
    }
    p->rows.resize(p->offsets.back());p->weights.resize(p->rows.size());auto cursor=p->offsets;
    for(int i=0;i<n;++i)if(column_of[i]>=0){const int k=cursor[column_of[i]]++;p->rows[k]=i;p->weights[k]=weight[i];}
    const size_t bytes=(size_t(n)*columns+n+columns)*sizeof(Complex);
    size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
    if(bytes+512ull*1024*1024>free)throw std::runtime_error("insufficient GPU memory for disjoint coarse updates");
    check(cudaMalloc(&p->update,size_t(n)*columns*sizeof(Complex)));
    check(cudaMalloc(&p->work,size_t(n)*sizeof(Complex)));
    check(cudaMalloc(&p->coeff,size_t(columns)*sizeof(Complex)));blas(cublasCreate(&p->handle));
    p->lu.resize(size_t(columns)*columns);p->beta.resize(columns);
    std::vector<Complex> x(n),y(n),by(n);
    for(int j=0;j<columns;++j) {
        std::fill(x.begin(),x.end(),Complex(0));
        for(int k=p->offsets[j];k<p->offsets[j+1];++k)x[p->rows[k]]=p->weights[k];
        sample(j,x.data(),y.data());p->project(y.data(),p->beta);
        for(int i=0;i<columns;++i)p->lu[size_t(i)*columns+j]=p->beta[i];
        local(y.data(),by.data());
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)x[i]-=by[i];
        check(cudaMemcpy(p->update+size_t(j)*n,x.data(),size_t(n)*sizeof(Complex),cudaMemcpyHostToDevice));
    }
    for(const auto& z:p->lu) {
        if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))throw std::runtime_error("nonfinite disjoint coarse matrix");
        p->scale=std::max(p->scale,std::abs(z));
    }
    if(!(p->scale>0))throw std::runtime_error("zero disjoint coarse matrix");
    for(auto& z:p->lu)z/=p->scale;
    if(!bem_krylov::dense_lu_factor(p->lu,columns,p->pivots))throw std::runtime_error("singular disjoint coarse matrix");
    impl.swap(p);
    std::printf("[face coarse] rank=%d GPU_MiB=%.3f implicit_basis=1\n",columns,storage_bytes()/1048576.);
}
void DisjointCoarseInverse::apply(const Complex* rhs,Complex* out,double weight) {
    auto& p=*impl;
    if(!p.local || !std::isfinite(weight))throw std::invalid_argument("disjoint coarse application");
    std::vector<Complex> alias;
    if(rhs==out){alias.assign(rhs,rhs+p.n);rhs=alias.data();}
    if(weight==0){p.local(rhs,out);return;}
    p.project(rhs,p.beta);
    for(auto& z:p.beta)z/=p.scale;
    bem_krylov::dense_lu_solve(p.lu,p.pivots,p.r,p.beta.data());
    p.local(rhs,out);
    check(cudaMemcpy(p.work,out,size_t(p.n)*sizeof(Complex),cudaMemcpyHostToDevice));
    check(cudaMemcpy(p.coeff,p.beta.data(),size_t(p.r)*sizeof(Complex),cudaMemcpyHostToDevice));
    const cuDoubleComplex alpha=make_cuDoubleComplex(weight,0);
    blas(cublasZgemv(p.handle,CUBLAS_OP_N,p.n,p.r,&alpha,p.update,p.n,p.coeff,1,&one,p.work,1));
    check(cudaMemcpy(out,p.work,size_t(p.n)*sizeof(Complex),cudaMemcpyDeviceToHost));
}
int DisjointCoarseInverse::rank()const{return impl->r;}
size_t DisjointCoarseInverse::storage_bytes()const{return (size_t(impl->n)*impl->r+impl->n+impl->r)*sizeof(Complex);}
