// CPU-only preparation of exactly the worker's incident quadrature and RHS.
#include "tmatrix_rhs_assembly.h"
#include "quadrature.h"
#include <fstream>
#include <iostream>
struct Point {int element;MullerFrameSample sample;double weight;};
int main(int argc,char** argv)try {
 if(argc!=5)throw std::runtime_error("usage: prism_packet_rhs REF ASPECT PREFIX positions|assemble");
 const int ref=std::stoi(argv[1]);const double aspect=std::stod(argv[2]);const std::string prefix=argv[3],mode=argv[4];
 if(ref<0 || ref>7 || aspect<=0)throw std::invalid_argument("geometry");
 MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
 const auto mesh=build_muller_p2_mesh(regular_prism(6,aspect,ref,1.,0,false,true),options);const auto rule=tri_quadrature(13);
 if(mode=="positions") {
    std::ofstream file(prefix+".positions.bin",std::ios::binary);
    for(int e=0;e<int(mesh.elements.size());++e)for(int q=0;q<rule.npts;++q) {
        const auto p=evaluate_muller_frame(mesh,e,rule.pts[q][0],rule.pts[q][1]).position;
        const double v[3]={p.x,p.y,p.z};file.write(reinterpret_cast<const char*>(v),sizeof(v));
    }
    if(!file)throw std::runtime_error("write positions");
 }else if(mode=="assemble") {
    std::vector<Point> points;for(int e=0;e<int(mesh.elements.size());++e)for(int q=0;q<rule.npts;++q){const auto f=evaluate_muller_frame(mesh,e,rule.pts[q][0],rule.pts[q][1]);points.push_back({e,f,.5*rule.wts[q]*f.jacobian});}
    bem_tmatrix::StreamingRhsAssembler assembler(rule.npts,2048,4);using Complex=std::complex<double>;
    std::ifstream input(prefix+".fields.bin",std::ios::binary);std::ofstream output(prefix+".rhs.bin",std::ios::binary);
    std::vector<Complex> fields(assembler.chunk_points()*6),rhs(mesh.system_dofs());
    for(int code=0;code<2;++code) {
        std::fill(rhs.begin(),rhs.end(),Complex(0));
        for(size_t begin=0;begin<points.size();begin+=assembler.chunk_points()) {
            const size_t count=std::min(assembler.chunk_points(),points.size()-begin);
            input.read(reinterpret_cast<char*>(fields.data()),count*6*sizeof(Complex));if(!input)throw std::runtime_error("read incident fields");
            assembler.add(mesh,points,begin,count,fields.data(),rhs);
        }
        output.write(reinterpret_cast<const char*>(rhs.data()),rhs.size()*sizeof(Complex));
    }
    if(!output || input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("RHS output or trailing fields");
 }else throw std::invalid_argument("mode");
 std::cout<<"dofs="<<mesh.system_dofs()<<" points="<<mesh.elements.size()*rule.npts<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
