#include "tmatrix_symmetry_warm_start.h"
#include "muller_dense.h"
#include <cstdio>
#include <stdexcept>

using namespace bem_tmatrix;
double error(Vector a,const Vector& b) {
    for(size_t i=0;i<a.size();++i)a[i]-=b[i];
    return bem_streaming::norm(a.data(),a.size())/std::max(bem_streaming::norm(b.data(),b.size()),1e-300);
}
void require(bool condition,const char* text) { if(!condition)throw std::runtime_error(text); }
int main(int argc,char** argv) try {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    options.project_edge_nodes_to_sphere=false;
    if(argc==2) {
        const int ref=std::stoi(argv[1]);
        const auto mesh=build_muller_p2_mesh(regular_prism(6,1.,ref,1.,0,false,true),options);
        D6hOrbit group;group.build(mesh);
        Vector x(group.size()),a,b,c;
        for(int i=0;i<group.size();++i)x[i]=Complex(std::sin(.17*i),std::cos(.23*i));
        a=x;
        for(int j=0;j<6;++j) {group.transform(a,b,false,false);a.swap(b);}
        require(error(a,x)==0.,"full mesh R^6 != I");
        group.transform(x,a,true,false);group.transform(a,b,true,false);
        require(error(b,x)==0.,"full mesh H^2 != I");
        group.transform(a,c,false,false);
        group.transform(x,a,false,false);group.transform(a,b,true,false);
        require(error(b,c)==0.,"full mesh RH != HR");
        std::printf("D6h permutation relations exact: ref=%d dofs=%d\n",ref,group.size());
        return 0;
    }
    const auto dense=assemble_muller_nodal_dense(regular_prism(6,1.,0,1.,0,false,true),3.,Complex(1.3,.03),options,7,6);
    D6hOrbit group;group.build(dense.mesh);
    const int n=group.size();
    Vector x(n),b(n),temp,transformed,action;
    for(int i=0;i<n;++i)x[i]=Complex(std::sin(.17*i),std::cos(.23*i));
    const auto apply=[&](const Vector& v,Vector& w) {
        w.assign(n,0.);for(int i=0;i<n;++i)for(int j=0;j<n;++j)w[i]+=dense.matrix[size_t(i)*n+j]*v[j];
    };
    apply(x,b);
    temp=x;
    for(int j=0;j<6;++j) {group.transform(temp,transformed,false,false);temp.swap(transformed);}
    require(error(temp,x)<1e-14,"rotation does not have order six");
    group.transform(x,temp,true,false);group.transform(temp,transformed,true,false);
    require(error(transformed,x)<1e-14,"reflection does not square to identity");
    for(bool reflection: {false,true}) {
        group.transform(x,temp,reflection,false);apply(temp,action);group.transform(b,transformed,reflection,true);
        const double e=error(action,transformed);
        std::printf("dense equivariance reflection=%d error=%.6e\n",reflection,e);
        require(e<2e-4,"incorrect current/equation symmetry transform");
    }
    // A generic combination of all twelve group actions represents arbitrary
    // phase changes in the twelve distinct C6 x reflection sectors.
    Vector second(n,0.),expected(n,0.),orbit_b=b,orbit_x=x,scratch;
    for(int g=0;g<12;++g) {
        if(g==6) {group.transform(b,orbit_b,true,true);group.transform(x,orbit_x,true,false);}
        else if(g%6) {
            group.transform(orbit_b,scratch,false,true);orbit_b.swap(scratch);
            group.transform(orbit_x,scratch,false,false);orbit_x.swap(scratch);
        }
        const Complex coefficient(.13*std::cos(g),.19*std::sin(.7*g));
        for(int i=0;i<n;++i) {second[i]+=coefficient*orbit_b[i];expected[i]+=coefficient*orbit_x[i];}
    }
    Vector guess;
    const auto fit=symmetry_guess(group,b,x,second,guess);
    std::printf("orbit QR rank=%d rhs_fit=%.6e solution_error=%.6e\n",fit.rank,fit.relative_rhs_error,error(guess,expected));
    require(fit.fitted && error(guess,expected)<1e-10,"warm start lost phase-code combination");
    for(int i=0;i<n;++i)second[i]=Complex(std::cos(.43*i),std::sin(.47*i));
    require(!symmetry_guess(group,b,x,second,guess).fitted,"unrelated RHS accepted");
    // Rank-deficient orbit: an axial plane wave spans only two sectors.
    b=muller_nodal_planewave_rhs(dense.mesh,3.,Vec3(1,0,0),Vec3(0,0,1),13);
    second=muller_nodal_planewave_rhs(dense.mesh,3.,Vec3(0,1,0),Vec3(0,0,1),13);
    const auto axial=symmetry_guess(group,b,x,second,guess);
    std::printf("axial RHS rank=%d fit=%.6e\n",axial.rank,axial.relative_rhs_error);
    require(axial.fitted && axial.rank<12,"rank-deficient orbit rejected");
    return 0;
} catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
