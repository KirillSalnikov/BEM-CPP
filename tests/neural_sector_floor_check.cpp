#include "streaming_image_space.h"
#include <iostream>
#include <limits>
int main()try {
    using V=std::vector<bem_streaming::Complex>;
    bem_streaming::ImageSpace legacy({0,2,4}),guarded({0,2,4},1e-7);
    // A tiny image in an analytically empty sector must not amplify its
    // unrelated current direction by 1e14. The meaningful sector stays exact.
    const V u{{1,0},{0,0},{1,0},{0,0}},c{{1,0},{0,0},{1e-14,0},{0,0}};
    legacy.add(u,c);guarded.add(u,c);
    if(legacy.rank()!=2||guarded.rank()!=1)throw std::runtime_error("sector rank guard");
    V guess,removed;guarded.apply(V{{0,0},{0,0},{1,0},{0,0}},guess,removed);
    if(bem_streaming::norm(guess.data(),4)!=0)throw std::runtime_error("null sector amplification");
    guarded.apply(V{{2,1},{0,0},{0,0},{0,0}},guess,removed);
    if(std::abs(guess[0]-bem_streaming::Complex(2,1))>1e-14)throw std::runtime_error("kept-sector inverse");
    for(double invalid:{-1.,.02,std::numeric_limits<double>::quiet_NaN()}){
        bool rejected=false;try{bem_streaming::ImageSpace bad({0,2},invalid);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("invalid floor accepted");
    }
    std::cout<<"PASS null-sector normalization, retained inverse, legacy default and invalid floors\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
