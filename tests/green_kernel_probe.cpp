#include "muller_kernel_bundle.h"
#include <iomanip>
#include <iostream>

int main() try {
    using C=std::complex<double>;
    double er,ei,ir,ii,x,y,z;
    std::cout << std::setprecision(17);
    while(std::cin>>er>>ei>>ir>>ii>>x>>y>>z) {
        const auto a=bem_muller::evaluate_kernel_values(C(er,ei),C(ir,ii),1.,0.,0.,1.,Vec3(x,y,z));
        auto emit=[](C value){std::cout<<value.real()<<' '<<value.imag()<<' ';};
        emit(a.k1_scalar);
        for(C value:a.hessian)emit(value);
        for(C value:a.gradient_epsilon)emit(value);
        for(C value:a.gradient_mu)emit(value);
        std::cout<<'\n';
    }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
