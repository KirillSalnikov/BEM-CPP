#include "host_regular_face_fft.h"
#include <cstdio>
#include <random>

int main()try {
    using C=std::complex<double>;
    std::mt19937 rng(719);std::normal_distribution<double> normal;
    double worst=0;
    for(int nx:{2,3,5})for(C k:{C(0),C(10),C(10,3)}){
        const int ny=4,channels=3,cells=nx*ny;
        const std::array<double,2> u{{.3,0}},v{{.11,.23}};
        std::vector<std::array<double,2>> offsets{{{0,0}},{{.07,.05}},{{.13,.11}}};
        bem_cpu::RegularFaceAction fft(nx,ny,u,v,offsets,k);
        if(fft.workspace_bytes()!=bem_cpu::RegularFaceAction::estimated_workspace_bytes(nx,ny,channels))
            throw std::runtime_error("FFT preallocation estimate differs from explicit arrays");
        std::vector<C> input(size_t(channels)*6*cells),result(size_t(channels)*12*cells),expected(result.size());
        for(auto& q:input)q=C(normal(rng),normal(rng));
        // Keep one of the two independent currents zero in the first channel.
        for(int c=3;c<6;++c)for(int j=0;j<cells;++j)input[c*cells+j]=0;
        fft.evaluate(input.data(),result.data());
        for(int t=0;t<channels;++t)for(int ti=0;ti<nx;++ti)for(int tj=0;tj<ny;++tj)
            for(int s=0;s<channels;++s)for(int si=0;si<nx;++si)for(int sj=0;sj<ny;++sj){
                const double rvec[3]={(ti-si)*u[0]+(tj-sj)*v[0]+offsets[t][0]-offsets[s][0],
                                     (ti-si)*u[1]+(tj-sj)*v[1]+offsets[t][1]-offsets[s][1],0};
                const double radius=std::hypot(rvec[0],rvec[1]);if(radius<1e-14)continue;
                const C ik=C(0,1)*k,G=std::exp(ik*radius)/(4*std::acos(-1.)*radius);
                const C first=G*(ik-1./radius),second=G*(-k*k-2.*ik/radius+2./(radius*radius));
                C grad[3],hessian[3][3],trace=0;
                for(int i=0;i<3;++i){
                    grad[i]=first*rvec[i]/radius;
                    for(int j=0;j<3;++j)hessian[i][j]=(second-first/radius)*rvec[i]*rvec[j]/(radius*radius)+(i==j?first/radius:C(0));
                    trace+=hessian[i][i];
                }
                for(int current=0;current<2;++current){
                    C q[3];for(int c=0;c<3;++c)q[c]=input[(size_t(s)*6+3*current+c)*cells+si*ny+sj];
                    C answer[6]={grad[1]*q[0]-grad[0]*q[1],grad[2]*q[0]-grad[0]*q[2],grad[2]*q[1]-grad[1]*q[2]};
                    for(int i=0;i<3;++i){for(int j=0;j<3;++j)answer[3+i]+=hessian[i][j]*q[j];answer[3+i]-=trace*q[i];}
                    for(int c=0;c<6;++c)expected[(size_t(t)*12+6*current+c)*cells+ti*ny+tj]+=answer[c];
                }
            }
        double numerator=0,denominator=0;
        for(size_t i=0;i<result.size();++i){numerator+=std::norm(result[i]-expected[i]);denominator+=std::norm(expected[i]);}
        const double error=std::sqrt(numerator/denominator);worst=std::max(worst,error);
        if(!std::isfinite(error)||error>3e-12)throw std::runtime_error("CPU face FFT versus direct Green derivatives");
        std::fill(input.begin(),input.end(),C(0));fft.evaluate(input.data(),result.data());
        for(auto q:result)if(std::abs(q)>1e-14)throw std::runtime_error("CPU face FFT stale buffer");
    }
    std::printf("PASS 9 planar lattice cases, two currents, complex k, exclusion and zero buffers; worst %.3e\n",worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
