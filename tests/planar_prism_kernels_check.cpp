#include "planar_prism_kernels.h"
#include <cstdio>
#include <omp.h>
int main()try {
    omp_set_num_threads(2);MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,2,1.,0,false,true),options);
    PrismCirculantLayout layout;layout.build(mesh,2);double worst=0;int checks=0;
    for(auto material:{bem_osrc::C(1.3116),bem_osrc::C(1.3116,.5)}) {
        const double ka=material.imag()?20.:3.;const auto kernels=bem_osrc::make_prism_kernels(mesh,2,ka,material,.01);
        if(kernels.kernel_bytes()!=(kernels.side.values.size()+kernels.cap.values.size())*sizeof(bem_osrc::C))
            throw std::runtime_error("kernel byte report wrong");
        for(int kind:{0,1}) {
            const auto& patch=layout.patches.at(kind?24:0);const auto& kernel=kind?kernels.cap:kernels.side;
            const int nx=kind?kernels.cap_nx:kernels.side_nx,ny=kind?kernels.cap_ny:kernels.side_ny,grid=nx*ny;
            if(nx!=patch.nx || ny!=patch.ny || kernel.values.size()!=size_t(144)*grid)
                throw std::runtime_error("worker inverse bank grid mismatch");
            bem_osrc::PlanarBdmSymbol stencil(mesh,patch,16);
            // Independent direct DFT of the returned spatial kernel, then
            // multiply by the original Bloch matrix. No FFTW inverse reused.
            for(auto frequency:{std::array<int,2>{{0,0}},std::array<int,2>{{nx/2,ny/2}},
                                 std::array<int,2>{{1,ny-1}},std::array<int,2>{{nx-1,1}}}) {
                const double a=2*std::acos(-1.)*frequency[0]/nx,b=2*std::acos(-1.)*frequency[1]/ny;
                const auto symbol=stencil.symbol(ka,material,.01,a,b,1);std::array<bem_osrc::C,144> inverse{};
                for(int i=0;i<144;++i)for(int x=0;x<nx;++x)for(int y=0;y<ny;++y)
                    inverse[i]+=kernel.values[size_t(i)*grid+x*ny+y]*std::exp(bem_osrc::C(0,-a*x-b*y));
                for(int i=0;i<12;++i)for(int j=0;j<12;++j){bem_osrc::C value=0;
                    for(int k=0;k<12;++k)value+=symbol[i*12+k]*inverse[k*12+j];
                    worst=std::max(worst,std::abs(value-bem_osrc::C(i==j)));}
                ++checks;
            }
        }
    }
    if(!std::isfinite(worst)||worst>1e-10)throw std::runtime_error("worker planar prism inverse normalization/sign failed");
    std::printf("PASS planar prism banks: %d independent DFT matrix checks, worst %.3e\n",checks,worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
