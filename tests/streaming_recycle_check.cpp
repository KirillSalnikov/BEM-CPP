#include "streaming_host_gmres.h"
#include <cstdio>
#include <limits>

using namespace bem_streaming;
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
int main() try {
    const int n=80;
    std::vector<Complex> matrix(n*n,0.),expected(n),b(n),x(n),work(n);
    for(int i=0;i<n;++i) {
        matrix[i*n+i]=i<6?Complex(.003*(i+1),.0005*i):Complex(.5+1.5*(i-6)/(n-6),.03);
        if(i+1<n)matrix[i*n+i+1]=Complex(.007,.003);
    }
    const Action exact=[&](const Complex* v,Complex* w) {
        for(int i=0;i<n;++i) {w[i]=0.;for(int j=0;j<n;++j)w[i]+=matrix[i*n+j]*v[j];}
    };
    const Action inverse=[&](const Complex* v,Complex* w){std::copy(v,v+n,w);};
    const Action mixed=[&](const Complex* v,Complex* w) {
        exact(v,w);for(int i=0;i<n;++i)w[i]+=1e-6*Complex(.2,.1)*v[i];
    };
    setenv("BEM_MIXED_ADAPTIVE_INNER","1",1);
    RecycleSpace space(n,8);
    int plain_iterations=0,recycled_iterations=0;
    for(int rhs=0;rhs<6;++rhs) {
        for(int i=0;i<n;++i)expected[i]=Complex(std::sin((.12+.03*rhs)*i),std::cos((.17+.07*rhs)*i));
        exact(expected.data(),b.data());
        const auto plain=gmres(mixed,exact,inverse,b.data(),x.data(),n,20,1e-9,2000,false);
        const auto recycled=gmres(mixed,exact,inverse,b.data(),x.data(),n,20,1e-9,2000,false,false,&space,rhs!=0);
        exact(x.data(),work.data());
        for(int i=0;i<n;++i)work[i]=b[i]-work[i];
        const double strict=norm(work.data(),n)/norm(b.data(),n);
        for(int i=0;i<n;++i)work[i]=expected[i]-x[i];
        const double error=norm(work.data(),n)/norm(expected.data(),n);
        std::printf("rhs=%d plain=%d recycle=%d rank=%d strict=%.3e error=%.3e\n",
            rhs,plain.iterations,recycled.iterations,space.rank(),strict,error);
        require(plain.residual<1e-9 && strict<1e-9 && error<1e-6,"recycling failed strict accuracy");
        require(std::abs(strict-recycled.residual)<1e-14,"reported residual is not strict");
        require(space.validate(),"recycle image orthogonality lost");
        if(rhs==0)require(plain.iterations==recycled.iterations,"cold audit used recycled vectors");
        else {plain_iterations+=plain.iterations;recycled_iterations+=recycled.iterations;}
    }
    require(recycled_iterations<plain_iterations*.7,"slow modes did not accelerate subsequent RHS");
    const std::string path="/tmp/bem-recycle-test-"+std::to_string(getpid())+".bin",signature(64,'a');
    space.save(path,signature);
    RecycleSpace restored(n,8),wrong_dimension(n+1,8);
    require(!wrong_dimension.load(path,signature),"wrong dimension cache accepted");
    require(!restored.load(path,std::string(64,'b')),"wrong signature cache accepted");
    require(restored.load(path,signature) && restored.rank()==space.rank(),"checkpoint roundtrip failed");
    const auto original=gmres(mixed,exact,inverse,b.data(),x.data(),n,20,1e-9,2000,false,false,&space);
    const auto resumed=gmres(mixed,exact,inverse,b.data(),work.data(),n,20,1e-9,2000,false,false,&restored);
    require(original.iterations==resumed.iterations,"resume changed iteration count");
    for(int i=0;i<n;++i)work[i]-=x[i];
    require(norm(work.data(),n)<1e-12,"resume changed solution");
    {std::fstream file(path,std::ios::in|std::ios::out|std::ios::binary);file.seekp(120);file.put('!');}
    require(!restored.load(path,signature),"corrupt checkpoint accepted");
    require(restored.validate(),"cache miss damaged existing space");
    {std::ofstream file(path,std::ios::binary|std::ios::trunc);file<<"truncated";}
    require(!restored.load(path,signature),"truncated checkpoint accepted");
    std::remove(path.c_str());

    // An exact coarse solve for A_inner can be wrong for A_strict. It must
    // trigger strict verification and fall back, never return false success.
    RecycleSpace inaccurate(2,2);
    std::vector<Complex> basis={1.,0.,0.,1.,0.,0.},h={1.,0.,0.,1.,0.,0.},empty;
    const Action identity=[](const Complex* v,Complex* w){w[0]=v[0];w[1]=v[1];};
    const Action doubled=[](const Complex* v,Complex* w){w[0]=2.*v[0];w[1]=2.*v[1];};
    require(inaccurate.learn(basis,h,2,2,identity,empty,0)>0,"coarse fallback setup failed");
    const Complex small_b[2]={Complex(1,.3),Complex(-.4,.2)};
    Complex small_x[2]={};
    const auto fallback=gmres(doubled,doubled,identity,small_b,small_x,2,2,1e-10,10,false,false,&inaccurate);
    require(fallback.residual<1e-10 && fallback.recycle_fallbacks==1 && fallback.strict_checks>=2,"coarse residual bypassed strict check");
    require(std::abs(small_x[0]-.5*small_b[0])<1e-10,"incorrect fallback solution");
    std::printf("subsequent RHS: %d -> %d iterations; checkpoint and strict fallback passed\n",plain_iterations,recycled_iterations);
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
