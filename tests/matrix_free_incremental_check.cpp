#include "matrix_free_taylor_moments.h"
#include <array>
#include <cstdio>
#include <limits>
using C=std::complex<double>;using V=bem_taylor::MomentVector;using H=bem_taylor::MomentHistory;
using Matrix=std::array<C,4>;
V apply(const Matrix& a,const V& x){return {a[0]*x[0]+a[1]*x[1],a[2]*x[0]+a[3]*x[1]};}
double relative(const V& a,const V& b){double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}return std::sqrt(e/std::max(n,1e-280));}
int main()try{
    double worst=0;int cases=0,rejected=0;
    for(size_t degree:{size_t(0),size_t(1),size_t(4),size_t(16)})for(bool zero:{false,true}){
        std::vector<Matrix> a(degree+1);a[0]={{C(2,.1),C(.3,-.2),C(-.17,.05),C(1.7,-.03)}};
        for(size_t q=1;q<=degree;++q)for(int i=0;i<4;++i)a[q][i]=C(std::sin(q+.3*i),std::cos(.2*q-i))*(.1/(q+1));
        H x(degree+1,V(2)),b(degree+1,V(2));
        if(!zero)for(size_t q=0;q<=degree;++q)for(int i=0;i<2;++i)x[q][i]=C(std::sin(.3*q+i),std::cos(q-.4*i));
        for(size_t j=0;j<=degree;++j)for(size_t q=0;q<=j;++q){const auto v=apply(a[q],x[j-q]);for(int i=0;i<2;++i)b[j][i]+=v[i];}
        const C determinant=a[0][0]*a[0][3]-a[0][1]*a[0][2];
        const bem_taylor::CentralMomentSolve solve=[&](const V& rhs,V& y){
            if(y!=V(2))throw std::runtime_error("incremental solve must receive zero initial current");
            y={(a[0][3]*rhs[0]-a[0][1]*rhs[1])/determinant,(-a[0][2]*rhs[0]+a[0][0]*rhs[1])/determinant};};
        size_t strict_calls=0,column_calls=0;
        const bem_taylor::CentralMomentAction strict=[&](const C* input,C* output){const auto y=apply(a[0],V(input,input+2));std::copy(y.begin(),y.end(),output);++strict_calls;};
        const bem_taylor::MomentConvolution history=[&](const H& input,H& output){output.assign(degree+1,V(2));
            for(size_t j=0;j<=degree;++j)for(size_t q=0;q<=j;++q){const auto v=apply(a[q],input[j-q]);for(int i=0;i<2;++i)output[j][i]+=v[i];}};
        const auto old=bem_taylor::solve_moments(b,history,solve,strict,1e-12);strict_calls=0;
        const bem_taylor::MomentColumnAction column=[&](const V& input,size_t maximum,H& output){
            if(maximum!=degree-column_calls)throw std::runtime_error("incremental requested derivative order");
            ++column_calls;output.resize(maximum+1);for(size_t q=0;q<=maximum;++q)output[q]=apply(a[q],input);};
        const auto result=bem_taylor::solve_moments_incremental(b,column,solve,strict,1e-12);
        if(strict_calls!=degree+1||column_calls!=degree)throw std::runtime_error("incremental callback accounting");
        for(size_t q=0;q<=degree;++q){worst=std::max({worst,relative(result.current[q],x[q]),relative(result.current[q],old.current[q])});}
        ++cases;
    }
    const H b(2,V(2,C(1)));
    const bem_taylor::CentralMomentSolve solve=[](const V& b,V& x){x=b;};
    const bem_taylor::CentralMomentAction strict=[](const C* x,C* y){std::copy(x,x+2,y);};
    const bem_taylor::MomentColumnAction column=[](const V& x,size_t d,H& y){y.assign(d+1,V(2));y[0]=x;};
    for(int kind=0;kind<8;++kind){auto rhs=b;auto col=column;auto sol=solve;auto exact=strict;double tolerance=1e-12;
        if(kind==0)rhs.clear();
        if(kind==1)rhs[1].pop_back();
        if(kind==2)tolerance=0;
        if(kind==3)col=[](const V&,size_t,H& y){y.clear();};
        if(kind==4)col=[](const V&,size_t d,H& y){y.assign(d+1,V(1));};
        if(kind==5)col=[](const V&,size_t d,H& y){y.assign(d+1,V(2,C(std::numeric_limits<double>::quiet_NaN())));};
        if(kind==6)sol=[](const V&,V& x){x.assign(1,C(0));};
        if(kind==7)exact=[](const C* x,C* y){for(int i=0;i<2;++i)y[i]=x[i]+C(.01);};
        bool caught=false;try{bem_taylor::solve_moments_incremental(rhs,col,sol,exact,tolerance);}catch(const std::exception&){caught=true;}
        if(!caught)throw std::runtime_error("incremental invalid callback/input accepted");
        ++rejected;
    }
    if(worst>2e-13)throw std::runtime_error("incremental manufactured polynomial mismatch");
    std::printf("PASS incremental moment recurrence: %d polynomial/zero cases, %d rejected, max %.3e; shrinking orders and original strict calls\n",cases,rejected,worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
