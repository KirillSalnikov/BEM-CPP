#include "selected_operator_rows.h"
#include <cmath>
#include <cstdio>
#include <random>
using C=std::complex<double>;
int main()try {
    std::mt19937 random(42177);std::normal_distribution<double> normal;
    const int n=13;std::vector<C> a(n*n);for(auto& z:a)z=C(normal(random),normal(random));
    auto forward=[&](const C* x,C* y){for(int i=0;i<n;++i){y[i]=0;for(int j=0;j<n;++j)y[i]+=a[i*n+j]*x[j];}};
    auto adjoint=[&](const C* x,C* y){for(int j=0;j<n;++j){y[j]=0;for(int i=0;i<n;++i)y[j]+=std::conj(a[i*n+j])*x[i];}};
    double worst=0;int cases=0;
    for(auto rows:{std::vector<int>{12,0,7,3},std::vector<int>{8},std::vector<int>{}})
    for(int batch:{1,2,7})for(bool adj:{false,true}) {
        SelectedOperatorRows action(n,rows,forward,adjoint);
        const size_t in=adj?rows.size():n,out=adj?n:rows.size();
        std::vector<C> input(std::max(size_t(1),in*batch)),result(std::max(size_t(1),out*batch)),expected(result.size());
        for(auto& z:input)z=C(normal(random),normal(random));
        action.apply(input.data(),batch,result.data(),adj);
        for(int b=0;b<batch;++b)for(size_t r=0;r<out;++r)for(size_t c=0;c<in;++c)
            expected[b*out+r]+=(adj?std::conj(a[rows[c]*n+r]):a[rows[r]*n+c])*input[b*in+c];
        double num=0,den=0;for(size_t j=0;j<result.size();++j){num+=std::norm(result[j]-expected[j]);den+=std::norm(expected[j]);}
        const double error=std::sqrt(num/std::max(den,1e-300));worst=std::max(worst,error);
        if(!std::isfinite(error)||error>2e-15)throw std::runtime_error("selected complex rows/direct adjoint mismatch");
        std::vector<C> alias(std::max(input.size(),result.size())+2);std::copy(input.begin(),input.end(),alias.begin());
        action.apply(alias.data(),batch,alias.data()+1,adj);
        for(size_t j=0;j<out*batch;++j)if(alias[1+j]!=result[j])throw std::runtime_error("overlapping selected-row block");
        ++cases;
    }
    int rejected=0;
    for(auto rows:{std::vector<int>{0,0},std::vector<int>{-1},std::vector<int>{n}})
        try{SelectedOperatorRows action(n,rows,forward,adjoint);}catch(const std::invalid_argument&){++rejected;}
    if(rejected!=3)throw std::runtime_error("bad row indices accepted");
    std::printf("PASS %d rectangular complex row/adjoint blocks, overlap and three invalid selections; worst %.17g\n",cases,worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
