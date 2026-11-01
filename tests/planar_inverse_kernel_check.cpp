#include "planar_inverse_kernel.h"
#include <cstdio>
#include <omp.h>
int main()try{
    omp_set_num_threads(2);MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,2,1.,0,false,true),options);
    PrismCirculantLayout layout;layout.build(mesh,2);double error=0;
    for(int kind:{0,1}){
        bem_osrc::PlanarBdmSymbol stencil(mesh,layout.patches[kind?24:0]);const int nx=4,ny=5,grid=nx*ny;
        const auto inverse=bem_osrc::make_inverse_kernel(stencil,nx,ny,10.,bem_osrc::C(1.3116),.05);
        for(int fx=0;fx<nx;++fx)for(int fy=0;fy<ny;++fy){const double a=2*std::acos(-1.)*fx/nx,b=2*std::acos(-1.)*fy/ny;
            const auto original=stencil.symbol(10.,bem_osrc::C(1.3116),.05,a,b);
            std::array<bem_osrc::C,144> recovered{};
            for(int i=0;i<12;++i)for(int j=0;j<12;++j)for(int x=0;x<nx;++x)for(int y=0;y<ny;++y)
                recovered[i*12+j]+=inverse.values[size_t(i*12+j)*grid+x*ny+y]*std::exp(bem_osrc::C(0,-a*x-b*y));
            for(int i=0;i<12;++i)for(int j=0;j<12;++j){bem_osrc::C value=0;for(int k=0;k<12;++k)value+=original[i*12+k]*recovered[k*12+j];
                error=std::max(error,std::abs(value-bem_osrc::C(i==j)));}
        }
        std::printf("kind=%d grid=4x5 inverse FFT/direct DFT defect %.3e condition_F %.3e\n",kind,error,inverse.maximum_frobenius_condition);
    }
    if(error>1e-11)throw std::runtime_error("inverse FFT kernel normalization/sign");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
