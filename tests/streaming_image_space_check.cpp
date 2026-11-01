#include "streaming_image_space.h"
#include <iostream>
int main()try {
    using namespace bem_streaming;using V=std::vector<Complex>;
    const int n=12;ImageSpace space({0,4,4,8,12});V rhs(n),a(n),u(n),c(n);
    for(int i=0;i<n;++i){rhs[i]=Complex(std::sin(.7*i),std::cos(.2*i));a[i]=Complex(1+.1*i,.03*i);}
    for(int k=0;k<4;++k){std::fill(u.begin(),u.end(),0.);std::fill(c.begin(),c.end(),0.);
        for(int block=0;block<3;++block){int i=4*block+k;u[i]=Complex(1.,.1*k);c[i]=a[i]*u[i];}
        if(!space.add(u,c))throw std::runtime_error("independent column rejected");
        if(space.add(u,c))throw std::runtime_error("dependent column accepted");}
    if(space.rank()!=12)throw std::runtime_error("wrong sector rank");
    V x,removed;space.apply(rhs,x,removed);double err=0;
    for(int i=0;i<n;++i)err=std::max(err,std::max(std::abs(a[i]*x[i]-rhs[i]),std::abs(removed[i]-rhs[i])));
    if(err>1e-12)throw std::runtime_error("image-space inverse relation failed");
    // An incomplete space must minimize the residual and preserve A*guess=removed.
    ImageSpace partial({0,n});for(int i=0;i<n;++i){u[i]=Complex(std::cos(.4*i),std::sin(.3*i));c[i]=a[i]*u[i];}
    partial.add(u,c);partial.apply(rhs,x,removed);V residual=rhs;
    for(int i=0;i<n;++i){residual[i]-=removed[i];if(std::abs(a[i]*x[i]-removed[i])>1e-12)throw std::runtime_error("coarse image mismatch");}
    if(std::abs(dot(c.data(),residual.data(),n))>1e-11||norm(residual.data(),n)>norm(rhs.data(),n))
        throw std::runtime_error("coarse residual not least squares");
    std::cout<<"image-space diagonal, dependent-column, empty-sector and least-squares checks passed\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
