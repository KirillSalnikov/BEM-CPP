// Persistent CPU forward/adjoint radiation bridge for P12. Raw complex128
// arrays are column-major; grid rows are [direction, e_theta, e_phi].
#include "muller_radiation_actions.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
template<class T> std::vector<T> read(const std::string& path,size_t count){
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(count*sizeof(T)))throw std::runtime_error("input size "+path);
    std::vector<T> x(count);f.seekg(0);if(!f.read(reinterpret_cast<char*>(x.data()),count*sizeof(T)))throw std::runtime_error("read "+path);return x;}
int main(int argc,char** argv)try {
    if(argc!=6)throw std::invalid_argument("REF KA ASPECT GRID_JSON_FREE_BINARY SETUP_JSON");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),aspect=std::stod(argv[3]);
    if(ref<0||ref>6||!std::isfinite(ka)||ka<=0||!std::isfinite(aspect)||aspect<=0)throw std::invalid_argument("invalid geometry");
    const auto start=std::chrono::steady_clock::now();std::ifstream gridfile(argv[4],std::ios::binary|std::ios::ate);
    if(!gridfile||gridfile.tellg()<=0||gridfile.tellg()%std::streamoff(9*sizeof(double)))throw std::runtime_error("invalid grid");
    const size_t count=size_t(gridfile.tellg())/(9*sizeof(double));gridfile.close();auto grid=read<double>(argv[4],9*count);
    std::vector<Vec3> directions,first,second;
    for(size_t i=0;i<count;++i){const auto* a=grid.data()+9*i;directions.emplace_back(a[0],a[1],a[2]);first.emplace_back(a[3],a[4],a[5]);second.emplace_back(a[6],a[7],a[8]);}
    MullerP2BuildOptions opts;opts.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,aspect,ref,1,0,false,true),opts);
    const MullerRadiationActions op(mesh,ka,directions,first,second);
    const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::ofstream meta(argv[5]);meta<<std::setprecision(17)<<"{\"dofs\":"<<op.dofs()<<",\"channels\":"<<op.channels()
        <<",\"quadrature_points\":"<<op.quadrature_points()<<",\"storage_bytes\":"<<op.storage_bytes()
        <<",\"setup_s\":"<<seconds<<",\"dense_radiation_assembled\":false,\"quadrature\":13}\n";meta.close();
    if(!meta)throw std::runtime_error("setup write");
    std::cout<<"READY\n"<<std::flush;
    std::string operation;int id,batch;std::string input,output;
    while(std::cin>>operation){if(operation=="QUIT")break;
        if(!(std::cin>>id>>batch>>input>>output)||batch<1||batch>64||(operation!="R"&&operation!="RH"))throw std::runtime_error("invalid request");
        auto x=read<MullerRadiationActions::Complex>(input,size_t(batch)*(operation=="R"?op.dofs():op.channels()));
        const auto before=std::chrono::steady_clock::now();MullerRadiationActions::Vector y;
        if(operation=="R")op.apply(x,batch,y);else op.adjoint(x,batch,y);
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-before).count();
        std::ofstream f(output,std::ios::binary);f.write(reinterpret_cast<const char*>(y.data()),y.size()*sizeof(y[0]));f.close();
        if(!f)throw std::runtime_error("action result write");
        std::cout<<std::setprecision(17)<<"DONE "<<id<<' '<<elapsed<<'\n'<<std::flush;
    }
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
