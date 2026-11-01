#include "coarse_ritz_experiment.h"
#include <cstdio>
int main(int argc,char** argv)try {
    if(argc!=2)throw std::invalid_argument("output checkpoint path required");
    using namespace bem_streaming;
    using Vector=coarse_ritz_experiment::Vector;
    RecycleSpace space(5,3);std::vector<Complex> basis(25),h(20),couplings;
    for(int i=0;i<5;++i)basis[5*i+i]=1;
    for(int i=0;i<4;++i)h[4*i+i]=.1+.2*i;h[4*4+3]=.03;
    const Action identity=[](const Complex* a,Complex* b){std::copy(a,a+5,b);};
    space.learn(basis,h,4,4,identity,couplings,0);
    if(!space.validate())throw std::runtime_error("synthetic recycle space failed");
    const std::string signature(64,'0');space.save(argv[1],signature);
    const auto columns=coarse_ritz_experiment::read_columns(argv[1],5,signature);
    if(int(columns.size())!=space.rank())throw std::runtime_error("Ritz checkpoint rank");
    RecycleSpace loaded(5,3);if(!loaded.load(argv[1],signature))throw std::runtime_error("public checkpoint roundtrip");
    Vector rhs(5),a(5),b(5),ra(5),rb(5);
    for(int i=0;i<5;++i)rhs[i]=Complex(std::sin(.31*i),std::cos(.17*i));
    ra=rb=rhs;space.correct(a.data(),ra.data());loaded.correct(b.data(),rb.data());
    for(int i=0;i<5;++i)if(std::abs(a[i]-b[i])>1e-14)throw std::runtime_error("checkpoint solution mismatch");
    std::fstream f(argv[1],std::ios::binary|std::ios::in|std::ios::out);f.seekg(120);char byte;f.read(&byte,1);byte^=1;f.seekp(120);f.write(&byte,1);f.close();
    bool rejected=false;try{coarse_ritz_experiment::read_columns(argv[1],5,signature);}catch(const std::runtime_error&){rejected=true;}
    std::remove(argv[1]);if(!rejected)throw std::runtime_error("corrupt Ritz checkpoint accepted");
    std::puts("Ritz export format, public roundtrip and checksum rejection passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
