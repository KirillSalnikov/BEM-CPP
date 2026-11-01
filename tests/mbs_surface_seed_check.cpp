// Independent finite Fabry-Perot series on the prism caps under axial light.
// Checks absolute phase, both polarizations, E/H scale, and reflected signs.
#include "mbs_surface_seed.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv)try {
    if(argc!=2)throw std::runtime_error("Usage: mbs_surface_seed_check AXIAL_V2_RAY_FILE");
    MbsSurfaceSeed ray(argv[1]);using C=std::complex<double>;
    if(!ray.internal_traces||(ray.direction-Vec3(0,0,1)).norm()>1e-12)throw std::runtime_error("requires axial internal trace export");
    int entrance=-1,exit=-1;
    for(size_t f=0;f<ray.faces.size();++f){if(ray.faces[f].normal.z<-.99)entrance=f;if(ray.faces[f].normal.z>.99)exit=f;}
    if(entrance<0||exit<0)throw std::runtime_error("missing caps");
    const double a=ray.faces[entrance].point.z,b=ray.faces[exit].point.z,height=b-a;
    const double n=ray.ri,r=(n-1)/(n+1),t01=2/(n+1),t10=2*n/(n+1);
    double worst=0;
    for(double k:{.2,5.,20.,40.}) {
        const C z=std::exp(C(0,2*n*k*height)),q=r*r*z;
        C reflected=-r,transmitted=0,term=1;
        for(int m=1;m<=ray.depth;++m){
            if(m%2)transmitted+=t01*t10*std::exp(C(0,n*k*height))*term;
            else {reflected+=t01*t10*r*z*term;term*=q;}
        }
        const C phase=std::exp(C(0,k*a));
        for(int pol=0;pol<2;++pol)for(int f:{entrance,exit}){
            const Vec3 p=ray.polarization(pol),h=ray.direction.cross(p);
            std::array<C,3> e,g;ray.fields(f,Vec3(.071,.043,ray.faces[f].point.z),k,pol,e,g);
            const C eamp=phase*(f==entrance?1.+reflected:transmitted),hamp=phase*(f==entrance?1.-reflected:transmitted);
            const double pv[]={p.x,p.y,p.z},hv[]={h.x,h.y,h.z};
            for(int j=0;j<3;++j)worst=std::max(worst,std::max(std::abs(e[j]-eamp*pv[j]),std::abs(g[j]-hamp*hv[j])));
        }
    }
    std::cout<<std::setprecision(17)<<"{\"slab_series_max_absolute_field_error\":"<<worst<<"}\n";
    if(worst>1e-10)throw std::runtime_error("native boundary currents disagree with independent Fresnel series");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
