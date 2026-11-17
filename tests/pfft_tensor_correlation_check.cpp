#include "pfft_tensor_correlation.h"
#include <random>
#include <vector>
#include <iostream>
using C=std::complex<double>;
int main(){
    std::mt19937 random(89);std::uniform_real_distribution<double> real(-1,1);
    double worst=0;
    for(int nodes:{3,4,6,8})for(int sample=0;sample<5;++sample){
        const int range=nodes+2,span=2*range+1;
        std::array<std::vector<C>,10> values;const C* kernel[10];
        for(int q=0;q<10;++q){values[q].resize(span*span*span);for(auto& z:values[q])z=C(real(random),real(random));kernel[q]=values[q].data();}
        double t[24],s[24];int delta[3]={sample-2,2-sample,sample%2};
        for(int i=0;i<3*nodes;++i){t[i]=real(random);s[i]=real(random);}
        std::array<C,10> direct{};
        for(int a=0;a<nodes;++a)for(int b=0;b<nodes;++b)for(int c=0;c<nodes;++c)
        for(int d=0;d<nodes;++d)for(int e=0;e<nodes;++e)for(int f=0;f<nodes;++f){
            const int x=delta[0]+a-d,y=delta[1]+b-e,z=delta[2]+c-f;
            if(std::abs(x)>range||std::abs(y)>range||std::abs(z)>range)continue;
            const double w=t[a]*t[nodes+b]*t[2*nodes+c]*s[d]*s[nodes+e]*s[2*nodes+f];
            const int index=((x+range)*span+y+range)*span+z+range;
            for(int q=0;q<10;++q)direct[q]+=w*kernel[q][index];
        }
        for(int count:{4,10}){
            auto fast=bem_pfft::tensor_correlation(nodes,t,s,delta,range,kernel,count);
            double error=0,norm=0;for(int q=0;q<count;++q){error+=std::norm(fast[q]-direct[q]);norm+=std::norm(direct[q]);}
            worst=std::max(worst,std::sqrt(error/norm));
            if(worst>3e-12)return 1;
        }
    }
    std::cout<<"tensor correlation vs six-loop sum: "<<worst<<"\n";
}
