#include "block_circulant_fft.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

using C=std::complex<double>;
using V=std::vector<C>;
using Map=MappedBlockCirculantFFT;
static void cuda_check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
static int host_checks() {
    // validate() only examines topology, never dereferences the bank pointer.
    int token=0;Map::Term t;t.bank=reinterpret_cast<BlockCirculantFFT*>(&token);t.slots.resize(2);
    t.slots[0].first=0;t.slots[1].first=1;Map::validate(4,{t});int rejected=0;
    auto bad=[&](const Map::Term& q){try{Map::validate(4,{q});}catch(const std::invalid_argument&){++rejected;return;}
        throw std::runtime_error("invalid map accepted");};
    auto q=t;q.slots[1].first=0;bad(q);
    q=t;q.slots[1].first=4;bad(q);
    q=t;q.slots[1].first=-2;bad(q);
    q=t;q.slots[0].rhs_first=0;bad(q);
    q=t;q.slots[0].solution_first=2;bad(q);
    q=t;q.slots[0].second=2;bad(q);
    q=t;q.pair_sign=1;bad(q);
    q=t;q.pair_sign=2;bad(q);
    q=t;q.bank=nullptr;bad(q);
    q=t;q.slots.clear();bad(q);
    q=t;q.pair_sign=-1;q.slots[0].second=2;q.slots[1].second=3;Map::validate(4,{q});
    q.slots[1].second=0;bad(q);
    std::printf("{\"host_validation_passed\":true,\"invalid_maps_rejected\":%d}\n",rejected);return rejected;
}
static double error(const V& a,const V& b) {
    double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}
    const double r=std::sqrt(e/std::max(n,1e-300));if(!std::isfinite(r))throw std::runtime_error("nonfinite error");return r;
}
int main(int argc,char** argv)try {
    host_checks();if(argc==2 && std::string(argv[1])=="--host-only")return 0;
    std::mt19937 rng(71629);std::normal_distribution<double> normal;double maximum=0;
    for(int channels:{2,3})for(int nx:{3,4})for(bool paired:{false,true}){
        const int ny=5,grid=nx*ny,batches=2,count=channels*grid*batches,n=2*count+17;
        BlockCirculantFFT positive,negative;V kernels[2];
        for(int branch=0;branch<2;++branch){
            auto& kernel=kernels[branch];kernel.resize(channels*channels*grid);
            for(auto& z:kernel)z=C(normal(rng),normal(rng))*.03;
            for(int c=0;c<channels;++c)kernel[(c*channels+c)*grid]+=1.+.2*branch;
            (branch?negative:positive).init_inverse_kernel(nx,ny,channels,batches,kernel);
        }
        std::vector<int> order(n);std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
        std::vector<Map::Term> terms;
        for(int branch=0;branch<(paired?2:1);++branch){
            Map::Term term;term.bank=branch?&negative:&positive;term.pair_sign=paired?(branch?-1:1):0;
            term.slots.resize(count);
            for(int i=0;i<count;++i){auto& s=term.slots[i];if(i%7==0)continue;
                s.first=order[i];s.second=paired?order[count+i]:-1;
                s.rhs_first=i%2?-1:1;s.solution_first=i%3?-1:1;
                s.rhs_second=i%5?-1:1;s.solution_second=i%4?-1:1;
            }
            terms.push_back(term);
        }
        Map inverse;inverse.init(n,terms);V x(n),expected(n),actual(n);
        for(auto& z:x)z=C(normal(rng),normal(rng));
        // Independent spatial convolution, including zero slots and signed
        // J/M parity maps. This does not call the host FFT implementation.
        for(size_t branch=0;branch<terms.size();++branch){
            const auto& term=terms[branch];const double scale=paired?1/std::sqrt(2.):1.;V in(count),out(count);
            for(int i=0;i<count;++i){const auto s=term.slots[i];
                if(s.first>=0)in[i]=scale*double(s.rhs_first)*x[s.first];
                if(s.second>=0)in[i]+=scale*double(term.pair_sign*s.rhs_second)*x[s.second];
            }
            for(int b=0;b<batches;++b)for(int r=0;r<channels;++r)for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)
                for(int c=0;c<channels;++c)for(int u=0;u<nx;++u)for(int v=0;v<ny;++v)
                    out[(b*channels+r)*grid+i*ny+j]+=kernels[branch][(r*channels+c)*grid+(i-u+nx)%nx*ny+(j-v+ny)%ny]*in[(b*channels+c)*grid+u*ny+v];
            for(int i=0;i<count;++i){const auto s=term.slots[i];
                if(s.first>=0)expected[s.first]+=scale*double(s.solution_first)*out[i];
                if(s.second>=0)expected[s.second]+=scale*double(term.pair_sign*s.solution_second)*out[i];
            }
        }
        inverse.apply(x.data(),actual.data());maximum=std::max(maximum,error(actual,expected));
        struct Device {C *in=nullptr,*out=nullptr;~Device(){cudaFree(in);cudaFree(out);}} device;
        cuda_check(cudaMalloc(&device.in,n*sizeof(C)));cuda_check(cudaMalloc(&device.out,n*sizeof(C)));
        cuda_check(cudaMemcpyAsync(device.in,x.data(),n*sizeof(C),cudaMemcpyHostToDevice,cudaStreamPerThread));
        inverse.enqueue_device(device.in,device.out);
        cuda_check(cudaMemcpyAsync(actual.data(),device.out,n*sizeof(C),cudaMemcpyDeviceToHost,cudaStreamPerThread));
        cuda_check(cudaStreamSynchronize(cudaStreamPerThread));maximum=std::max(maximum,error(actual,expected));
        bool alias_rejected=false;try{inverse.enqueue_device(device.in,device.in);}catch(const std::invalid_argument&){alias_rejected=true;}
        if(!alias_rejected)throw std::runtime_error("resident FFT alias accepted");
        alias_rejected=false;try{inverse.enqueue_device(device.in,device.in+1);}catch(const std::invalid_argument&){alias_rejected=true;}
        if(!alias_rejected)throw std::runtime_error("resident FFT overlapping ranges accepted");
        // Alternating device/host entry points exercises cuFFT stream restore.
        actual=x;inverse.apply(actual.data(),actual.data());maximum=std::max(maximum,error(actual,expected));
        std::fill(x.begin(),x.end(),C(0));inverse.apply(x.data(),actual.data());
        for(auto z:actual)if(z!=C(0))throw std::runtime_error("zero RHS leakage");
    }
    if(maximum>2e-12)throw std::runtime_error("mapped FFT disagrees with spatial convolution");
    std::printf("{\"passed\":true,\"direct_convolution_error\":%.17g}\n",maximum);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"mapped FFT test: %s\n",e.what());return 1;}
