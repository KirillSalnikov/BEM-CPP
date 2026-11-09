#include "host_fmm_band.h"
#include "host_shared_near_avx512.h"
#include <cstdio>
#include <numeric>
#include <random>
using C=bem_cpu::C;using Six=bem_cpu::Six;using Twelve=bem_cpu::Twelve;

// Independent radial Green derivatives, without the vector kernel's a/b
// factorization. Accumulate Hessian minus its trace for both currents.
static void oracle(const std::vector<Vec3>& points,const std::vector<int>& ids,const Vec3& target,C k,
                   const std::vector<Six>& charges,Twelve& out){
    for(int s:ids){const Vec3 v=target-points[s];const double r2=v.norm2();if(r2<1e-28)continue;
        const double r=std::sqrt(r2),x[3]={v.x,v.y,v.z};const C ik=C(0,1)*k,g=std::exp(ik*r)/(4*M_PI*r);
        const C first=g*(ik-1/r),second=g*(-k*k-2.*ik/r+2/r2);C grad[3],h[3][3],trace=0;
        for(int i=0;i<3;++i){grad[i]=first*x[i]/r;for(int j=0;j<3;++j)h[i][j]=(second-first/r)*x[i]*x[j]/r2+(i==j?first/r:C(0));trace+=h[i][i];}
        for(int current=0;current<2;++current){const auto* q=charges[s].data()+3*current;auto* y=out.data()+6*current;
            y[0]+=grad[1]*q[0]-grad[0]*q[1];y[1]+=grad[2]*q[0]-grad[0]*q[2];y[2]+=grad[2]*q[1]-grad[1]*q[2];
            for(int i=0;i<3;++i){for(int j=0;j<3;++j)y[i+3]+=h[i][j]*q[j];y[i+3]-=trace*q[i];}
        }
    }
}
static double error(const Twelve& a,const Twelve& b){double e=0,n=0;for(int i=0;i<12;++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}
    const double r=std::sqrt(e/std::max(1e-300,n));if(!std::isfinite(r))throw std::runtime_error("nonfinite near result");return r;}
int main()try{
#ifdef BEM_CPU_HAS_AVX512_P2P
    if(!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512vl"))throw std::runtime_error("AVX512 runtime unavailable");
    std::mt19937 rng(42071);std::uniform_real_distribution<double> normal(-1.,1.);
    double worst=0,serial_error=0;int cases=0;
    for(int count:{0,1,7,8,9,127,128,129,255,256,257,519})for(int batches:{1,2,6,8})
    for(C k:{C(3),C(40),C(3,.2),C(1.3116,10),C(3,-.1)})for(bool coincident:{false,true}){
        const int points_count=std::max(1,count);std::vector<Vec3> points(points_count);
        for(auto& p:points)p=Vec3(normal(rng),normal(rng),normal(rng));
        Vec3 target(.3,-.2,.1);if(coincident)points[0]=target;
        if(count>1 && coincident)points[1]=target+Vec3(1e-15,0,0);
        std::vector<int> ids(count);std::iota(ids.begin(),ids.end(),0);std::shuffle(ids.begin(),ids.end(),rng);
        std::array<std::vector<double>,15> soa;for(auto& a:soa)a.resize(points_count);
        for(int p=0;p<points_count;++p){soa[0][p]=points[p].x;soa[1][p]=points[p].y;soa[2][p]=points[p].z;}
        std::vector<std::vector<Six>> charges(batches,std::vector<Six>(points_count));
        std::vector<Twelve> actual(batches),expected(batches),serial(batches);
        const Six* in[8];Twelve* out[8];
        for(int b=0;b<batches;++b){in[b]=charges[b].data();out[b]=&actual[b];
            for(auto& p:charges[b])for(auto& v:p)v=b==batches-1&&batches>1?C(0):C(normal(rng),normal(rng));
            for(int c=0;c<12;++c)actual[b][c]=expected[b][c]=serial[b][c]=C(.1*(c+1),-.07*(b+1));
            oracle(points,ids,target,k,charges[b],expected[b]);
            for(int p=0;p<points_count;++p)for(int c=0;c<6;++c){soa[3+2*c][p]=charges[b][p][c].real();soa[4+2*c][p]=charges[b][p][c].imag();}
            bem_cpu::avx512::near_field(soa,ids,target,k,serial[b]);
        }
        bem_cpu::avx512::near_field_batch(soa,ids,target,k,in,out,batches);
        for(int b=0;b<batches;++b){worst=std::max(worst,error(actual[b],expected[b]));serial_error=std::max(serial_error,error(actual[b],serial[b]));}
        const auto saved=actual;for(auto& q:charges)for(auto& p:q)p=Six{};
        bem_cpu::avx512::near_field_batch(soa,ids,target,k,in,out,batches);
        for(int b=0;b<batches;++b)if(error(actual[b],saved[b])!=0)throw std::runtime_error("nonzero zero-current output");
        ++cases;
    }
    if(worst>2e-12 || serial_error>2e-12)throw std::runtime_error("shared near exceeds original accuracy gate");
    std::printf("{\"passed\":true,\"cases\":%d,\"independent_derivative_error\":%.17g,\"original_AVX512_error\":%.17g,\"GPU_used\":false}\n",cases,worst,serial_error);
#else
    throw std::runtime_error("Compile shared-near controls with AVX512F/VL and glibc");
#endif
}catch(const std::exception& e){std::fprintf(stderr,"shared near: %s\n",e.what());return 1;}
