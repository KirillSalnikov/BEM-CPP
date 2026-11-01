// Small dense CPU reference for absorbing-prism plane-wave comparisons.
// No GPU operator, production checkpoint, or production profile is modified.
#include "muller_dense.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using C = std::complex<double>;
template<class T> std::vector<T> read(const std::string& name) {
    std::ifstream f(name, std::ios::binary|std::ios::ate);
    if (!f || f.tellg()<=0 || size_t(f.tellg())%sizeof(T)) throw std::runtime_error("invalid "+name);
    std::vector<T> x(size_t(f.tellg())/sizeof(T)); f.seekg(0);
    if (!f.read(reinterpret_cast<char*>(x.data()), x.size()*sizeof(T))) throw std::runtime_error("read "+name);
    return x;
}
template<class T> void write(const std::string& name, const std::vector<T>& x) {
    std::ofstream f(name, std::ios::binary);
    if (!f.write(reinterpret_cast<const char*>(x.data()), x.size()*sizeof(T))) throw std::runtime_error("write "+name);
}
int main(int argc, char** argv) try {
    if (argc!=11) throw std::runtime_error("assemble|far REF KA RI_RE RI_IM ASPECT DUFFY INPUT PREFIX");
    const std::string mode=argv[1], input=argv[9], prefix=argv[10];
    const int ref=std::stoi(argv[2]), duffy=std::stoi(argv[8]);
    const double ka=std::stod(argv[3]), aspect=std::stod(argv[6]);
    // argv[7] is reserved for the regular surface quadrature order.
    const int quad=std::stoi(argv[7]);
    const C index(std::stod(argv[4]), std::stod(argv[5]));
    if (ref<0 || ref>2 || ka<=0 || aspect<=0) throw std::runtime_error("CPU reference requires ref<=2");
    const auto linear=regular_prism(6,aspect,ref,1.,0,false,true);
    MullerP2BuildOptions options; options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto start=std::chrono::steady_clock::now();
    if (mode=="assemble") {
        const auto incoming=read<double>(input);
        if (incoming.size()%9) throw std::runtime_error("incoming rows are [direction, p, s]");
        auto system=assemble_muller_nodal_dense(linear,ka,index,options,quad,duffy);
        const int n=system.system_dofs, count=2*(incoming.size()/9);
        std::vector<C> rhs(size_t(n)*count);
        for (int c=0;c<count;++c) {
            const auto a=incoming.data()+9*(c/2);
            const auto p=a+3+3*(c%2);
            auto b=muller_nodal_planewave_rhs(system.mesh,ka,Vec3(p[0],p[1],p[2]),Vec3(a[0],a[1],a[2]),13);
            for (int j=0;j<n;++j) rhs[size_t(j)*count+c]=b[j];
        }
        write(prefix+".A.bin",system.matrix);write(prefix+".B.bin",rhs);
        std::ofstream f(prefix+".json");
        f<<std::setprecision(17)<<"{\"dofs\":"<<n<<",\"rhs_count\":"<<count
          <<",\"ka\":"<<ka<<",\"ri\":"<<index.real()<<",\"ri_imag\":"<<index.imag()
          <<",\"ref\":"<<ref<<",\"prism_aspect\":"<<aspect<<",\"quad\":"<<quad
          <<",\"duffy\":"<<duffy<<",\"assembly_s\":"
          <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<"}\n";
    } else if (mode=="far") {
        const auto mesh=build_muller_p2_mesh(linear,options);
        const int n=mesh.system_dofs(), currents=mesh.current_dofs();
        const auto solution=read<C>(prefix+".X.bin");
        if (solution.size()%n || (solution.size()/n)%2) throw std::runtime_error("solution dimensions");
        const int columns=solution.size()/n, orientations=columns/2;
        const auto flat=read<double>(input);
        if (flat.size()%(3*orientations)) throw std::runtime_error("outgoing dimensions");
        const int count=flat.size()/(3*orientations);
        std::vector<C> all;
        for (int i=0;i<orientations;++i) {
            std::vector<Vec3> directions;
            for (int j=0;j<count;++j) {
                const auto d=flat.data()+3*(i*count+j);directions.emplace_back(d[0],d[1],d[2]);
            }
            std::vector<C> x(n),y(n),fx,fy;
            for(int j=0;j<n;++j){x[j]=solution[size_t(j)*columns+2*i];y[j]=solution[size_t(j)*columns+2*i+1];}
            muller_nodal_farfield_pair(mesh,x.data(),x.data()+currents,y.data(),y.data()+currents,ka,directions,fx,fy,13);
            all.insert(all.end(),fx.begin(),fx.end());all.insert(all.end(),fy.begin(),fy.end());
        }
        write(prefix+".far.bin",all);
    } else throw std::runtime_error("unknown mode");
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
