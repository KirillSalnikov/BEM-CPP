// Bounded-memory export of the exact regular-prism BDM geometry and mass.
// This emits no FMM, radiation grid or solution vectors.
#include "muller_nodal.h"
#include "quadrature.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>

int main(int argc,char**argv)try{
    if(argc!=5)throw std::runtime_error("REF KA ASPECT OUTPUT_DIRECTORY");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]),aspect=std::stod(argv[3]);
    if(ref<0||ref>8||!(ka>0)||!(aspect>0))throw std::runtime_error("invalid parameters");
    const std::string directory=argv[4];
    auto linear=regular_prism(6,aspect,ref,1.,0,false,true);
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    auto mesh=build_muller_p2_mesh(linear,options);
    std::map<std::string,std::unique_ptr<std::ofstream>> files;
    for(const char* name:{"vertices","points","basis","normals","dofs","mass_rows","mass_cols","mass_values"}){
        files[name].reset(new std::ofstream(directory+"/"+name+".bin",std::ios::binary));
        if(!*files[name])throw std::runtime_error("cannot open geometry output");
    }
    const auto real=[&](const char*name,double x){files[name]->write((char*)&x,sizeof(x));};
    const auto integer=[&](const char*name,int x){files[name]->write((char*)&x,sizeof(x));};
    const auto vector=[&](const char*name,const Vec3&v){real(name,v.x);real(name,v.y);real(name,v.z);};
    const auto quadrature=tri_quadrature(7),mass_rule=tri_quadrature(13);
    for(size_t t=0;t<mesh.elements.size();++t){
        for(int j=0;j<3;++j)vector("vertices",mesh.nodes[mesh.elements[t].nodes[j]]);
        int dofs[6];double mass[6][6]={{0}};
        for(int j=0;j<quadrature.npts;++j){
            auto frame=evaluate_muller_frame(mesh,t,quadrature.pts[j][0],quadrature.pts[j][1]);
            auto basis=evaluate_muller_basis(mesh,t,frame);vector("points",frame.position);
            if(j==0){vector("normals",frame.normal);for(int a=0;a<6;++a){dofs[a]=basis.dofs[a];integer("dofs",dofs[a]);}}
            for(int a=0;a<6;++a)vector("basis",basis.values[a]*(.5*quadrature.wts[j]*frame.jacobian));
        }
        for(int j=0;j<mass_rule.npts;++j){
            auto frame=evaluate_muller_frame(mesh,t,mass_rule.pts[j][0],mass_rule.pts[j][1]);
            auto basis=evaluate_muller_basis(mesh,t,frame);
            for(int a=0;a<6;++a)for(int b=0;b<6;++b)
                mass[a][b]+=.5*mass_rule.wts[j]*frame.jacobian*basis.values[a].dot(basis.values[b]);
        }
        for(int a=0;a<6;++a)for(int b=0;b<6;++b){integer("mass_rows",dofs[a]);integer("mass_cols",dofs[b]);real("mass_values",mass[a][b]);}
    }
    for(auto& item:files){item.second->close();if(!*item.second)throw std::runtime_error("geometry write failed");}
    std::ofstream metadata(directory+"/geometry.json");
    metadata<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"ri\":[1.3116,0],\"aspect\":"<<aspect
        <<",\"triangles\":"<<mesh.elements.size()<<",\"current_dofs\":"<<mesh.current_dofs()<<",\"system_dofs\":"<<2*mesh.current_dofs()
        <<",\"quadrature\":7,\"duffy\":6}\n";
    metadata.close();if(!metadata)throw std::runtime_error("metadata write failed");
    std::cout<<"Exported "<<mesh.elements.size()<<" triangles, "<<2*mesh.current_dofs()<<" unknowns\n";
}catch(const std::exception&error){std::cerr<<error.what()<<'\n';return 1;}
