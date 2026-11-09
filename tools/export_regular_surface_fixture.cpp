// CPU-only independent geometry/Green oracle for small regular OBJ surfaces.
// The quadratic matrices are validation data, never a large-run backend.
#include "muller_dense.h"
#include "quadrature.h"
#include <fstream>
#include <iomanip>
#include <iostream>
using C=std::complex<double>;
template<class T>void save(const std::string& path,const std::vector<T>& data){
    std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(T));
    if(!out)throw std::runtime_error("Cannot write fixture: "+path);
}
void append(std::vector<double>& out,Vec3 x){out.insert(out.end(),{x.x,x.y,x.z});}
int main(int argc,char** argv)try{
    if(argc!=6)throw std::invalid_argument("MESH.obj OUTPUT KA RI_REAL RI_IMAG");
    const double ka=std::stod(argv[3]);const C ri(std::stod(argv[4]),std::stod(argv[5]));
    if(!std::isfinite(ka)||ka<=0||!std::isfinite(ri.real())||!std::isfinite(ri.imag()))throw std::invalid_argument("Invalid material");
    const std::string dir=argv[2];auto linear=load_obj(argv[1]);
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    auto mesh=build_muller_p2_mesh(linear,options);const int n=mesh.current_dofs(),N=2*n;
    if(N>1000)throw std::invalid_argument("Validation fixture is limited to 1000 DOFs");
    const auto q=tri_quadrature(7);std::vector<double>vertices,points,basis,normals;
    std::vector<int>dofs;std::vector<Vec3>positions,weighted,normal;
    for(size_t t=0;t<mesh.elements.size();++t){
        for(int j=0;j<3;++j)append(vertices,mesh.nodes[mesh.elements[t].nodes[j]]);
        for(int j=0;j<q.npts;++j){
            auto f=evaluate_muller_frame(mesh,t,q.pts[j][0],q.pts[j][1]);auto b=evaluate_muller_basis(mesh,t,f);
            append(points,f.position);positions.push_back(f.position);
            if(j==0){append(normals,f.normal);normal.push_back(f.normal);for(int a=0;a<6;++a)dofs.push_back(b.dofs[a]);}
            for(int a=0;a<6;++a){auto v=b.values[a]*(.5*q.wts[j]*f.jacobian);append(basis,v);weighted.push_back(v);}
        }
    }
    save(dir+"/vertices.bin",vertices);save(dir+"/points.bin",points);save(dir+"/basis.bin",basis);
    save(dir+"/normals.bin",normals);save(dir+"/dofs.bin",dofs);
    std::vector<C>regular(size_t(N)*N);
    for(size_t t=0;t<normal.size();++t)for(size_t s=0;s<normal.size();++s)
        for(int tq=0;tq<7;++tq)for(int sq=0;sq<7;++sq){
            auto r=positions[7*t+tq]-positions[7*s+sq];if(r.norm2()<1e-24)continue;
            for(int a=0;a<6;++a)for(int b=0;b<6;++b){
                auto u=weighted[(7*t+tq)*6+a],v=weighted[(7*s+sq)*6+b];
                const int i=dofs[6*t+a],j=dofs[6*s+b];
                const C h=C(0,1)/ka*muller_k1_kernel(ka,ka*ri,r,u.cross(normal[t]),v);
                regular[size_t(i)*N+j]+=h;regular[size_t(i+n)*N+j+n]-=h;
                regular[size_t(i)*N+j+n]+=muller_k2_kernel(C(1),ri*ri,ka,ka*ri,r,u,normal[t],v);
                regular[size_t(i+n)*N+j]+=muller_k2_kernel(C(1),C(1),ka,ka*ri,r,u,normal[t],v);
            }
        }
    save(dir+"/regular.bin",regular);
    // Small tetrahedra also supply the full independently assembled operator.
    if(N<=100){auto dense=assemble_muller_nodal_dense(linear,C(ka),ri,options,7,6);save(dir+"/dense.bin",dense.matrix);}
    std::ofstream meta(dir+"/geometry.json");meta<<std::setprecision(17)
        <<"{\"current_dofs\":"<<n<<",\"system_dofs\":"<<N<<",\"ka\":"<<ka
        <<",\"ri\":["<<ri.real()<<','<<ri.imag()<<"],\"quadrature\":7,\"duffy\":6}\n";
    if(!meta)throw std::runtime_error("Cannot write metadata");return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
