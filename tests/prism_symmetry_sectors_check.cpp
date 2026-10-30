#include "prism_orbit_inverse.h"
#include "prism_sector_solver.h"
#include "prism_vertical_inverse.h"
#include <cstdio>
using namespace bem_tmatrix;
using bem_streaming::norm;
static double error(Vector a,const Vector& b){for(size_t j=0;j<a.size();++j)a[j]-=b[j];return norm(a.data(),a.size())/std::max(norm(b.data(),b.size()),1e-300);}
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main()try {
 for(int ref:{0,1,2}) {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
    D6hOrbit group;group.build(mesh);PrismGroupMaps maps;maps.build(group);
    PrismSymmetrySectors sectors;sectors.build(maps);const int n=group.size();
    Vector x(n),a(n),b(n),y(n),c(n),d(n);
    for(int j=0;j<n;++j)x[j]=Complex(std::sin(.217*j),std::cos(.391*j));
    for(bool rhs:{false,true}) {
        sectors.encode(x.data(),a.data(),rhs);sectors.decode(a.data(),b.data(),rhs);
        require(error(x,b)<2e-14,"sector transform inverse");require(std::abs(norm(a.data(),n)/norm(x.data(),n)-1)<2e-14,"sector transform norm");
        for(bool reflection:{false,true}) {
            group.transform(x,y,reflection,rhs);sectors.encode(y.data(),b.data(),rhs);
            for(int s=0;s<12;++s) {
                const Complex eigenvalue=reflection?Complex(s<6?1.:-1.):std::polar(1.,2*std::acos(-1.)*(s%6)/6.);
                for(int j=sectors.offsets[s];j<sectors.offsets[s+1];++j)c[j]=eigenvalue*a[j];
            }
            require(error(b,c)<2e-14,"sector character/parity sign");
        }
    }
    // Deliberately nonsymmetric, complex local matrices with pivoting. Group
    // invariance must follow from the construction, not local reciprocity.
    PrismOrbitInverse inverse;inverse.build(mesh,maps,7,[&](const std::vector<int>& ids) {
        const int d=4*ids.size();Vector matrix(d*d);
        for(int i=0;i<d;++i)for(int j=0;j<d;++j)matrix[i*d+j]=Complex(.003*std::sin(i+2*j+ids[0]),.002*std::cos(2*i-j));
        for(int i=0;i<d;++i)matrix[i*d+i]+=Complex(2.+.01*i,.3);
        if(d>1)for(int j=0;j<d;++j)std::swap(matrix[j],matrix[d+j]);
        return matrix;
    });
    inverse.apply(x.data(),a.data());b=x;inverse.apply(b.data(),b.data());require(error(a,b)<1e-14,"orbit inverse alias");
    for(bool reflection:{false,true}) {
        group.transform(x,b,reflection,true);inverse.apply(b.data(),c.data());group.transform(a,d,reflection,false);
        const double defect=error(c,d);std::printf("ref=%d mirror=%d orbit_inverse_defect=%.3e\n",ref,reflection,defect);
        require(defect<2e-14,"orbit inverse intertwining");
    }
    PrismVerticalInverse full;
    full.build(mesh,[&](const Complex* v,Complex* out){inverse.apply(v,out);});
    full.transform(x.data(),a.data(),false);full.transform(a.data(),b.data(),false);
    require(error(x,b)<1e-14,"vertical involution");
    // V R V = R^-1, and V H = H V: averaging the extra coset must
    // preserve the two already-established symmetries.
    group.transform(a,c,false,false);full.transform(c.data(),b.data(),false);
    c=x;for(int j=0;j<5;++j){group.transform(c,d,false,false);c.swap(d);}
    require(error(b,c)<2e-14,"dihedral generator relation");
    full.apply(x.data(),a.data());
    full.transform(x.data(),b.data(),true);full.apply(b.data(),c.data());full.transform(a.data(),d.data(),false);
    require(error(c,d)<2e-14,"vertical inverse intertwining");
    for(bool reflection:{false,true}) {
        group.transform(x,b,reflection,true);full.apply(b.data(),c.data());group.transform(a,d,reflection,false);
        require(error(c,d)<2e-14,"full inverse lost commuting subgroup symmetry");
    }
    // Independently invert an equivariant, per-edge electric/magnetic
    // exchange operator. Every local factorization requires row pivoting.
    const Complex coefficients[4]={{2.,.3},{1.7,-.2},{-.8,.4},{1.2,.7}};
    PrismOrbitInverse exchange;exchange.build(mesh,maps,7,[&](const std::vector<int>& ids) {
        const int d=4*ids.size();Vector matrix(d*d);
        for(size_t j=0;j<ids.size();++j)for(int c=0;c<4;++c)
            matrix[(4*j+c)*d+4*j+(c+2)%4]=coefficients[c];
        return matrix;
    });
    exchange.apply(x.data(),a.data());const int edges=mesh.topology_edge_count;
    for(int e=0;e<edges;++e)for(int c=0;c<4;++c) {
        const int d=(c/2)*2*edges+2*e+c%2,other=((c+2)%4/2)*2*edges+2*e+c%2;
        b[d]=x[other]/coefficients[(c+2)%4];
    }
    require(error(a,b)<2e-14,"orbit inverse disagrees with analytic local inverse");
    const bem_streaming::Action action=[&](const Complex* input,Complex* output) {
        for(int e=0;e<edges;++e)for(int c=0;c<4;++c) {
            const int d=(c/2)*2*edges+2*e+c%2,other=((c+2)%4/2)*2*edges+2*e+c%2;
            output[d]=coefficients[c]*input[other];
        }
    };
    const bem_streaming::Action precondition=[&](const Complex* input,Complex* output){exchange.apply(input,output);};
    action(x.data(),b.data());
    auto solved=solve_prism_sectors(action,action,precondition,sectors,b.data(),a.data(),8,1e-11,20,false,false);
    require(solved.iterations==1 && error(a,x)<2e-14,"physical sector wrapper inverse/coordinate ordering");
    solved=solve_prism_sectors(action,action,precondition,sectors,b.data(),a.data(),8,1e-11,20,true,false);
    require(solved.iterations==0 && solved.strict_checks==1,"physical sector wrapper warm start gate");
    a[0]+=Complex(.2,.1);
    solved=solve_prism_sectors(action,action,precondition,sectors,b.data(),a.data(),8,1e-11,20,true,false);
    require(solved.iterations>0 && error(a,x)<2e-14,"physical sector wrapper warm start correction");
 }
 std::puts("Sector unitarity, rotation/mirror characters, pivoting, fixed-edge orbits and inverse intertwining passed");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
