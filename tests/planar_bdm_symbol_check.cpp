#include "planar_bdm_symbol.h"
#include "krylov_deflation.h"
#include <cstdio>
using C=std::complex<double>;using V=std::vector<C>;
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
template<class A,class B>static double error(const A& a,const B& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}return std::sqrt(x/std::max(y,1e-300));}
int main()try{
    double mass_error=0,quadrature_error=0,inverse_error=0;
    for(int ref:{2,3}){
        MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        PrismCirculantLayout layout;layout.build(mesh,ref);
        for(int kind:{0,1}){
            const auto& patch=layout.patches[kind?24:0];bem_osrc::PlanarBdmSymbol stencil(mesh,patch,16),fine(mesh,patch,24);
            const auto& sources=stencil.source_dofs();const auto& signs=stencil.source_signs();const auto rule=tri_quadrature(13);
            for(const auto& theta:std::vector<std::array<double,2>>{{{0,0}},{{.4,-.8}},{{2.3,1.9}},{{std::acos(-1.),0}}}){
                const auto mass=stencil.mass(theta[0],theta[1]);bem_osrc::PlanarBdmSymbol::Mass reference{};
                // Independently assemble convolution columns over each compact
                // source basis support, not a Bloch integral over one cell.
                for(int t=0;t<int(mesh.elements.size());++t){const auto frame=evaluate_muller_frame(mesh,t,1./3,1./3);const auto center=evaluate_muller_basis(mesh,t,frame);bool support=false;
                    for(int j=0;j<center.count;++j)for(int c=0;c<6;++c)support=support||center.dofs[j]==sources[c];
                    if(!support)continue;
                    for(int q=0;q<rule.npts;++q){const auto point=evaluate_muller_frame(mesh,t,rule.pts[q][0],rule.pts[q][1]);const auto values=evaluate_muller_basis(mesh,t,point);
                        for(int j=0;j<values.count;++j)for(int c=0;c<6;++c)if(values.dofs[j]==sources[c])for(int i=0;i<values.count;++i){const auto target=stencil.channel(values.dofs[i]);
                            reference[target.channel*6+c]+=.5*rule.wts[q]*point.jacobian*double(signs[c]*target.sign)*values.values[i].dot(values.values[j])*
                                std::exp(C(0,-theta[0]*target.dx-theta[1]*target.dy));}
                    }
                }
                mass_error=std::max(mass_error,error(mass,reference));
                for(int i=0;i<6;++i)for(int j=0;j<6;++j)require(std::abs(mass[i*6+j]-std::conj(mass[j*6+i]))<1e-12,"mass Hermiticity");
                for(int trial=0;trial<4;++trial){C form=0;for(int i=0;i<6;++i)for(int j=0;j<6;++j)form+=std::conj(C(std::sin(i+trial),std::cos(2*i-trial)))*mass[i*6+j]*C(std::sin(j+trial),std::cos(2*j-trial));
                    require(form.real()>0 && std::abs(form.imag())<1e-12,"mass positivity");}
                const auto symbol=stencil.symbol(20.,C(1.3116),.05,theta[0],theta[1],1);
                const auto finer=fine.symbol(20.,C(1.3116),.05,theta[0],theta[1],1);
                quadrature_error=std::max(quadrature_error,error(symbol,finer));
                if(theta[0]==0&&theta[1]==0){
                    const auto f=stencil.fourier(0,0);V mass_lu(mass.begin(),mass.end());std::vector<int> mass_pivots;
                    require(bem_krylov::dense_lu_factor(mass_lu,6,mass_pivots),"constant-current mass solve");
                    const auto b=bem_osrc::diagonal_block(20.,C(20,1),C(20,1)*1.3116,0,0);
                    for(int polarization=0;polarization<2;++polarization){V current(6);
                        for(int i=0;i<6;++i)current[i]=std::conj(f[2*i+polarization]);
                        bem_krylov::dense_lu_solve(mass_lu,mass_pivots,6,current.data());
                        for(int i=0;i<6;++i){C actual=0,expected=0;
                            for(int j=0;j<6;++j)actual+=symbol[i*12+j]*current[j];
                            for(int component=0;component<2;++component)expected+=std::conj(f[2*i+component])*b[2*component+polarization];
                            require(std::abs(actual-expected)<2e-12,"physical plane-wave/BDM reciprocal-cell normalization");}
                    }
                }
                const auto periodic=stencil.symbol(20.,C(1.3116),.05,theta[0]+2*std::acos(-1.),theta[1],1);
                require(error(periodic,symbol)<2e-12,"symbol Bloch periodicity/Nyquist");
                V lu(symbol.begin(),symbol.end());std::vector<int> pivots;require(bem_krylov::dense_lu_factor(lu,12,pivots),"BDM symbol singular");
                for(int j=0;j<12;++j){V column(12);column[j]=1.;bem_krylov::dense_lu_solve(lu,pivots,12,column.data());
                    for(int i=0;i<12;++i){C v=0;for(int k=0;k<12;++k)v+=symbol[i*12+k]*column[k];inverse_error=std::max(inverse_error,std::abs(v-C(i==j)));}}
                const auto equal=stencil.symbol(20.,C(1.),.05,theta[0],theta[1],1);
                for(int i=0;i<6;++i)for(int j=0;j<6;++j){require(equal[i*12+j]==C(0)&&equal[(i+6)*12+j+6]==C(0),"equal-media contrast cancellation");
                    require(std::abs(equal[i*12+j+6]-mass[i*6+j])<1e-12&&std::abs(equal[(i+6)*12+j]-mass[i*6+j])<1e-12,"BDM mass normalization");}
            }
            std::printf("ref=%d kind=%d exact mass %.3e, Fourier quadrature %.3e, inverse %.3e\n",ref,kind,mass_error,quadrature_error,inverse_error);
        }
    }
    require(mass_error<2e-12&&quadrature_error<2e-9&&inverse_error<2e-11,"planar BDM symbol thresholds");
    std::puts("PASS exact BDM1 mass, reciprocal normalization, passive two-medium symbol, Nyquist periodicity and invertibility");
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
