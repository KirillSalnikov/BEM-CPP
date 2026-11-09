#include "physical_ritz_prior.h"
#include <cstdio>
#include <iostream>
#include <unistd.h>
using namespace bem_ritz;
int main()try {
    char temporary[]="/tmp/bem-ritz-prior-XXXXXX";
    if(!mkdtemp(temporary))throw std::runtime_error("temporary directory");
    const std::string dir=temporary,path=dir+"/prior.bin";
    Parameters p;p.dofs=8;p.columns=2;p.ref=2;p.ka=3;p.ri_real=1.3116;p.aspect=10./7;p.duffy=12;
    const auto source=[](size_t j,Vector& v){for(size_t i=0;i<v.size();++i)v[i]=Complex(i+.25*j,-2.*i-.5*j);};
    write(path,p,source);Reader reader(path,p);Vector a,b(p.dofs);int checks=0;
    for(size_t j=0;j<p.columns;++j){reader.read(j,a);source(j,b);if(a!=b)throw std::runtime_error("column layout");++checks;}
    const auto rejected=[&](const std::string& f,const Parameters& expected){
        bool failed=false;try{Reader r(f,expected);}catch(const std::exception&){failed=true;}
        if(!failed)throw std::runtime_error("invalid physical prior accepted");
        ++checks;
    };
    for(int field=0;field<8;++field){auto wrong=p;
        switch(field){case 0:++wrong.dofs;break;case 1:++wrong.columns;break;case 2:++wrong.ref;break;
        case 3:wrong.ka+=.1;break;case 4:wrong.ri_real+=.01;break;case 5:wrong.ri_imag=.1;break;
        case 6:wrong.aspect+=.1;break;case 7:wrong.duffy=6;break;}
        rejected(path,wrong);
    }
    std::ifstream input(path,std::ios::binary);std::vector<char> data((std::istreambuf_iterator<char>(input)),{});
    for(int corrupt=0;corrupt<4;++corrupt){auto changed=data;
        if(corrupt==0)changed[0]^=1;
        if(corrupt==1)changed.back()^=1;
        if(corrupt==2)changed.resize(changed.size()-1);
        if(corrupt==3){const double nan=std::numeric_limits<double>::quiet_NaN();std::memcpy(changed.data()+header_bytes,&nan,sizeof(nan));}
        const auto name=dir+"/bad"+std::to_string(corrupt);std::ofstream out(name,std::ios::binary);
        out.write(changed.data(),changed.size());out.close();rejected(name,p);std::remove(name.c_str());
    }
    bool overwrite=false;try{write(path,p,source);}catch(const std::exception&){overwrite=true;}
    if(!overwrite)throw std::runtime_error("existing prior overwritten");
    ++checks;
    std::remove(path.c_str());rmdir(dir.c_str());std::cout<<"PASS "<<checks<<" physical Ritz file controls\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
