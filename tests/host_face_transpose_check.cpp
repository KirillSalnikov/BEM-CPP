#include "host_regular_face_fft.h"
#include <cstdio>
#include <random>

using C=std::complex<double>;
double relative(const std::vector<C>& a,const std::vector<C>& b){
    double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::norm(a[i]-b[i]);d+=std::norm(b[i]);}
    return std::sqrt(n/std::max(d,1e-300));
}
int main()try {
    if(!bem_cpu::FftwRuntime::get().available()){std::puts("SKIP FFTW runtime unavailable");return 0;}
    std::mt19937 rng(214733);std::normal_distribution<double> normal;
    double worst=0,identity_worst=0;int cases=0,identities=0;
    for(int nx:{1,3,5})for(int channels:{1,3})for(C k:{C(0),C(10),C(10,3)}) {
        const int ny=nx==1?1:4,cells=nx*ny;
        const std::array<double,2> u{{.3,.03}},v{{.11,.23}};
        std::vector<std::array<double,2>> offsets;
        for(int c=0;c<channels;++c)offsets.push_back({{.07*c,.043*c}});
        bem_cpu::RegularFaceAction fft(nx,ny,u,v,offsets,k,5,C(.7,.2));
        const size_t bytes=fft.workspace_bytes();
        std::vector<C> q(size_t(channels)*6*cells),seed(size_t(channels)*12*cells),forward(seed.size());
        std::vector<C> expected(q.size()),expected_h(q.size()),transposed(q.size()),adjoint(q.size());
        for(auto& x:q)x=C(normal(rng),normal(rng));
        for(auto& x:seed)x=C(normal(rng),normal(rng));
        // Independent direct Green-derivative matrix; every target/source
        // cell participates, including cells at both edges of the crop.
        for(int t=0;t<channels;++t)for(int ti=0;ti<nx;++ti)for(int tj=0;tj<ny;++tj)
        for(int s=0;s<channels;++s)for(int si=0;si<nx;++si)for(int sj=0;sj<ny;++sj) {
            double r[3]={(ti-si)*u[0]+(tj-sj)*v[0]+offsets[t][0]-offsets[s][0],
                         (ti-si)*u[1]+(tj-sj)*v[1]+offsets[t][1]-offsets[s][1],0};
            const double radius=std::hypot(r[0],r[1]);if(radius<1e-14)continue;
            const C ik=C(0,1)*k,g=std::exp(ik*radius)/(4*std::acos(-1.)*radius);
            const C first=g*(ik-1./radius),second=g*(-k*k-2.*ik/radius+2./(radius*radius));
            C grad[3],h[3][3],trace=0,a[6][3]{};
            for(int i=0;i<3;++i){
                grad[i]=first*r[i]/radius;
                for(int j=0;j<3;++j)h[i][j]=(second-first/radius)*r[i]*r[j]/(radius*radius)+(i==j?first/radius:C(0));
                trace+=h[i][i];
            }
            a[0][0]=grad[1];a[0][1]=-grad[0];a[1][0]=grad[2];a[1][2]=-grad[0];a[2][1]=grad[2];a[2][2]=-grad[1];
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)a[i+3][j]=h[i][j]-(i==j?trace:C(0));
            for(int current=0;current<2;++current)for(int i=0;i<6;++i)for(int j=0;j<3;++j){
                const size_t src=size_t(s*6+3*current+j)*cells+si*ny+sj;
                const C v=seed[size_t(t*12+6*current+i)*cells+ti*ny+tj];
                expected[src]+=a[i][j]*v;expected_h[src]+=std::conj(a[i][j])*v;
            }
        }
        fft.evaluate_transpose(seed.data(),transposed.data());fft.evaluate_adjoint(seed.data(),adjoint.data());
        worst=std::max(worst,std::max(relative(transposed,expected),relative(adjoint,expected_h)));
        if(!std::isfinite(worst)||worst>3e-12)throw std::runtime_error("direct transpose/adjoint mismatch");
        for(int order:{0,1,2,5}) {
            fft.evaluate_coefficient(order,q.data(),forward.data());
            fft.evaluate_coefficient_transpose(order,seed.data(),transposed.data());
            fft.evaluate_coefficient_adjoint(order,seed.data(),adjoint.data());
            C lhs=0,rhs=0,lh=0,rh=0;double fn=0,sn=0;
            for(size_t i=0;i<seed.size();++i){lhs+=seed[i]*forward[i];lh+=std::conj(seed[i])*forward[i];fn+=std::norm(forward[i]);sn+=std::norm(seed[i]);}
            for(size_t i=0;i<q.size();++i){rhs+=transposed[i]*q[i];rh+=std::conj(adjoint[i])*q[i];}
            const double error=std::max(std::abs(lhs-rhs),std::abs(lh-rh))/std::sqrt(std::max(fn*sn,1e-300));
            identity_worst=std::max(identity_worst,error);++identities;
            if(!std::isfinite(error)||error>3e-12)throw std::runtime_error("coefficient bilinear identity failed");
            auto alias=seed;fft.evaluate_coefficient_adjoint(order,alias.data(),alias.data());alias.resize(q.size());
            if(relative(alias,adjoint)>1e-14)throw std::runtime_error("reverse alias mismatch");
        }
        std::fill(seed.begin(),seed.end(),C(0));fft.evaluate_transpose(seed.data(),transposed.data());
        for(C x:transposed)if(std::abs(x)>1e-14)throw std::runtime_error("stale transpose buffer");
        if(bytes!=fft.workspace_bytes())throw std::runtime_error("reverse allocated spectral workspace");
        bool rejected=false;try{fft.evaluate_coefficient_transpose(6,seed.data(),transposed.data());}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid reverse order accepted");
        ++cases;
    }
    std::printf("PASS %d direct transpose/adjoint cases, %d coefficient identities, alias/zero/order/memory; direct %.17g identity %.17g\n",cases,identities,worst,identity_worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
