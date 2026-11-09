#ifndef BEM_P2P_TEST_BUFFERS_H
#define BEM_P2P_TEST_BUFFERS_H
#include <algorithm>
#include <cmath>
#include <vector>
template<class T> struct Buffer {
    T* p=nullptr; size_t n;
    explicit Buffer(const std::vector<T>&v):n(v.size()) {
        CUDA_CHECK(cudaMalloc(&p,n*sizeof(T)));
        CUDA_CHECK(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    }
    ~Buffer(){cudaFree(p);}
    Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
    void zero(){CUDA_CHECK(cudaMemset(p,0,n*sizeof(T)));}
    std::vector<T> read(){std::vector<T>v(n);CUDA_CHECK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
double difference(const std::vector<double>&a,const std::vector<double>&b){
    // Check every target and each vector action separately so that a large
    // near-singular target cannot mask errors at the other targets.
    const size_t nt=a.size()/24;double worst=0;
    for(size_t block=0;block<8;block+=2)for(size_t t=0;t<nt;++t){
        long double num=0,den=0;
        for(size_t ri=0;ri<2;++ri)for(size_t d=0;d<3;++d){
            const size_t i=(block+ri)*3*nt+3*t+d;
            num+=(long double)(a[i]-b[i])*(a[i]-b[i]);den+=(long double)b[i]*b[i];
        }
        worst=std::max(worst,std::sqrt(double(num/std::max(den,1e-300L))));
    }
    return worst;
}
#endif
