// Dense CPU experiment: analytic Taylor coefficients of A(k), B(k), D(k).
// Coefficient j is derivative/j!, for k=k0+delta; material and mesh fixed.
// No production FMM code or existing checkpoint is changed by this probe.
#include "muller_dense.h"
#include "muller_duffy.h"
#include "quadrature.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
using C=std::complex<double>;
#ifndef MULLER_TAYLOR_MAX_DEGREE
#define MULLER_TAYLOR_MAX_DEGREE 8
#endif
static_assert(MULLER_TAYLOR_MAX_DEGREE >= 0 && MULLER_TAYLOR_MAX_DEGREE <= 16,
              "dense derivative experiment degree must be between 0 and 16");
using Series=std::array<C,MULLER_TAYLOR_MAX_DEGREE+1>;
using Clock=std::chrono::steady_clock;
const C I(0,1);

template<class T> void write(const std::string& path,const std::vector<T>& x) {
    std::ofstream f(path,std::ios::binary);
    if(!f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(T)))throw std::runtime_error("write "+path);
}
double binomial(int n,int j) {
    if(j>n)return 0;
    double b=1;for(int a=1;a<=j;++a)b*=double(n-a+1)/a;return b;
}
struct Radial {
    Series diagonal{},along{},epsilon{},mu{};
};
Radial radial(double k,C m,double r,int order) {
    if(!(r>0))throw std::runtime_error("zero quadrature separation");
    Series phi{},hdiag{},huu{},ge{},gm{};
    for(int medium=0;medium<2;++medium) {
        const C s=medium?m:C(1),sign=medium?C(-1):C(1);
        const C chi=medium?m*m:C(1),z=I*s*r;
        Series e{};e[0]=std::exp(z*k);
        for(int j=1;j<=order;++j)e[j]=e[j-1]*z/double(j);
        for(int j=0;j<=order;++j) {
            const C em1=j?e[j-1]:C(0),em2=j>1?e[j-2]:C(0);
            phi[j]+=sign*s*s*INV4PI/r*(k*k*e[j]+2*k*em1+em2);
            const C f=INV4PI/(r*r*r)*((3.-3.*z*k-s*s*k*k*r*r)*e[j]+(-3.*z-2.*s*s*k*r*r)*em1-s*s*r*r*em2);
            const C g=INV4PI/(r*r*r)*((1.-z*k)*e[j]-z*em1);
            hdiag[j]-=sign*g;huu[j]+=sign*f;
            const C grad=INV4PI/r*((I*s*k-1./r)*e[j]+I*s*em1);
            ge[j]+=sign*chi*grad;gm[j]+=sign*grad;
        }
    }
    // Match the library's stable small-distance polynomial, including its
    // existing truncation, so this tests continuation of the same operator.
    if(std::max(1.,std::abs(m))*k*r<1e-2) {
        hdiag.fill(0);huu.fill(0);ge.fill(0);gm.fill(0);
        double factorial=1;
        for(int n=0;n<=6;++n) {
            if(n)factorial*=n;
            for(int j=0;j<=std::min(n,order);++j) {
                const C c=std::pow(I,n)*binomial(n,j)*std::pow(k,n-j)/factorial*INV4PI;
                if(n>=2) {
                    const C h=c*(1.-std::pow(m,n))*std::pow(r,n-3);
                    hdiag[j]+=h*double(n-1);huu[j]+=h*double(n-3);
                }
                if(n!=1) {
                    const C g=c*double(n-1)*std::pow(r,n-2);
                    ge[j]+=g*(1.-std::pow(m,n+2));gm[j]+=g*(1.-std::pow(m,n));
                }
            }
        }
    }
    Radial a;
    for(int j=0;j<=order;++j) {
        a.diagonal[j]=(I*(phi[j]+hdiag[j])-(j?a.diagonal[j-1]:C(0)))/k;
        a.along[j]=(I*huu[j]-(j?a.along[j-1]:C(0)))/k;
        a.epsilon[j]=ge[j];a.mu[j]=gm[j];
    }
    return a;
}
struct Adjacency { int count=0;int t[3]{},s[3]{}; };
Adjacency adjacent(const MullerP2Element& a,const MullerP2Element& b) {
    Adjacency x;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)if(a.topology_vertices[i]==b.topology_vertices[j]) {
        x.t[x.count]=i;x.s[x.count++]=j;
    }
    return x;
}
void remap(MullerDuffyPoint& p,const Adjacency& a) {
    if(a.count==1) {
        muller_duffy_remap_shared_vertex(p.test_xi,p.test_eta,a.t[0]);
        muller_duffy_remap_shared_vertex(p.trial_xi,p.trial_eta,a.s[0]);
    } else if(a.count==2) {
        muller_duffy_remap_shared_edge(p.test_xi,p.test_eta,a.t[0],a.t[1]);
        muller_duffy_remap_shared_edge(p.trial_xi,p.trial_eta,a.s[0],a.s[1]);
    }
}
int main(int argc,char** argv) try {
    if(argc!=12)throw std::runtime_error("REF KA RI_RE RI_IM ASPECT GRID PREFIX QUAD DUFFY DEGREE VERIFY");
    const int ref=std::stoi(argv[1]),quad_order=std::stoi(argv[8]),duffy=std::stoi(argv[9]),degree=std::stoi(argv[10]);
    const double ka=std::stod(argv[2]),aspect=std::stod(argv[5]);
    const C m(std::stod(argv[3]),std::stod(argv[4]));
    const std::string prefix=argv[7];
    if(degree<0||degree>MULLER_TAYLOR_MAX_DEGREE||ref<0||ref>2||ka<=0||aspect<=0)throw std::runtime_error("invalid parameters");
    std::ifstream f(argv[6],std::ios::binary|std::ios::ate);
    if(!f||f.tellg()<=0||size_t(f.tellg())%(9*sizeof(double)))throw std::runtime_error("invalid angular grid");
    std::vector<double> grid(size_t(f.tellg())/sizeof(double));f.seekg(0);
    f.read(reinterpret_cast<char*>(grid.data()),grid.size()*sizeof(double));
    const auto start=Clock::now();
    const auto linear=regular_prism(6,aspect,ref,1.,0,false,true);
    MullerP2BuildOptions opts;opts.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(linear,opts);
    const int n=mesh.system_dofs(),c=mesh.current_dofs(),nt=mesh.elements.size(),points=grid.size()/9,channels=2*points;
    if(n>5000)throw std::runtime_error("dense experiment limited to 5000 DOFs");
    const size_t nn=size_t(n)*n;
    std::vector<C> A(size_t(degree+1)*nn),B(size_t(degree+1)*n*channels),D(size_t(degree+1)*channels*n);
    // Elements in one color share no BDM row DOFs; scatter is race-free.
    std::vector<std::vector<int>> colors;
    std::vector<std::set<int>> occupied;
    for(int t=0;t<nt;++t) {
        const auto basis=evaluate_muller_basis(mesh,t,evaluate_muller_frame(mesh,t,1./3,1./3));
        int color=0;
        for(;color<int(colors.size());++color) {
            bool conflict=false;for(int j=0;j<basis.count;++j)conflict|=occupied[color].count(basis.dofs[j])>0;
            if(!conflict)break;
        }
        if(color==int(colors.size())){colors.emplace_back();occupied.emplace_back();}
        colors[color].push_back(t);for(int j=0;j<basis.count;++j)occupied[color].insert(basis.dofs[j]);
    }
    const TriQuad reg=tri_quadrature(quad_order),q13=tri_quadrature(13);
    std::vector<MullerDuffyPoint> rules[3]={muller_duffy_rule(duffy,MullerDuffyAdjacency::Coincident),
        muller_duffy_rule(duffy,MullerDuffyAdjacency::EdgeAdjacent),muller_duffy_rule(duffy,MullerDuffyAdjacency::VertexAdjacent)};
    for(const auto& color:colors) {
#pragma omp parallel for schedule(dynamic,1)
        for(size_t ti=0;ti<color.size();++ti) {
            const int t=color[ti];
            for(int q=0;q<q13.npts;++q) {
                const auto sample=evaluate_muller_frame(mesh,t,q13.pts[q][0],q13.pts[q][1]);
                const auto basis=evaluate_muller_basis(mesh,t,sample);
                const double weight=.5*q13.wts[q]*sample.jacobian;
                for(int a=0;a<basis.count;++a)for(int b=0;b<basis.count;++b) {
                    const int row=basis.dofs[a],col=basis.dofs[b];
                    const double mass=weight*basis.values[a].dot(basis.values[b]);
                    A[size_t(row)*n+c+col]+=.5*(1.+m*m)*mass;
                    A[size_t(c+row)*n+col]+=mass;
                }
            }
            for(int s=0;s<nt;++s) {
                const auto adjacency=adjacent(mesh.elements[t],mesh.elements[s]);
                auto sample=[&](double tx,double ty,double sx,double sy,double weight) {
                    const auto test=evaluate_muller_frame(mesh,t,tx,ty),source=evaluate_muller_frame(mesh,s,sx,sy);
                    const auto bt=evaluate_muller_basis(mesh,t,test),bs=evaluate_muller_basis(mesh,s,source);
                    const Vec3 separation=test.position-source.position;
                    const double r=separation.norm();const Vec3 unit=separation*(1./r);
                    const auto radial_parts=radial(ka,m,r,degree);
                    weight*=test.jacobian*source.jacobian;
                    for(int a=0;a<bt.count;++a)for(int b=0;b<bs.count;++b) {
                        const Vec3 v=bt.values[a],w=bs.values[b],rot=v.cross(test.normal);
                        const double dot=rot.dot(w),along=rot.dot(unit)*w.dot(unit),
                            grad=v.dot(w)*test.normal.dot(unit)-test.normal.dot(w)*v.dot(unit);
                        const int row=bt.dofs[a],col=bs.dofs[b];
                        for(int j=0;j<=degree;++j) {
                            C* matrix=A.data()+size_t(j)*nn;
                            const C diagonal=weight*(radial_parts.diagonal[j]*dot+radial_parts.along[j]*along);
                            matrix[size_t(row)*n+col]+=diagonal;
                            matrix[size_t(c+row)*n+c+col]-=diagonal;
                            matrix[size_t(row)*n+c+col]+=weight*grad*radial_parts.epsilon[j];
                            matrix[size_t(c+row)*n+col]+=weight*grad*radial_parts.mu[j];
                        }
                    }
                };
                if(adjacency.count) {
                    const auto& rule=rules[t==s?0:adjacency.count==2?1:2];
                    for(auto q:rule){remap(q,adjacency);sample(q.test_xi,q.test_eta,q.trial_xi,q.trial_eta,q.weight);}
                } else {
                    for(int x=0;x<reg.npts;++x)for(int y=0;y<reg.npts;++y)
                        sample(reg.pts[x][0],reg.pts[x][1],reg.pts[y][0],reg.pts[y][1],.25*reg.wts[x]*reg.wts[y]);
                }
            }
        }
    }
    const auto assembled=Clock::now();
#pragma omp parallel for schedule(static)
    for(int p=0;p<points;++p) {
        const double* g=grid.data()+9*p;
        const Vec3 direction(g[0],g[1],g[2]),tangent[2]={Vec3(g[3],g[4],g[5]),Vec3(g[6],g[7],g[8])};
        for(int element=0;element<nt;++element)for(int q=0;q<q13.npts;++q) {
            const auto sample=evaluate_muller_frame(mesh,element,q13.pts[q][0],q13.pts[q][1]);
            const auto basis=evaluate_muller_basis(mesh,element,sample);
            const C phase=I*direction.dot(sample.position);
            const double weight=.5*q13.wts[q]*sample.jacobian;
            Series plus{},minus{};plus[0]=std::exp(ka*phase);minus[0]=std::exp(-ka*phase);
            for(int j=1;j<=degree;++j){plus[j]=plus[j-1]*phase/double(j);minus[j]=-minus[j-1]*phase/double(j);}
            for(int a=0;a<basis.count;++a)for(int pol=0;pol<2;++pol) {
                const Vec3 v=basis.values[a],rot=v.cross(sample.normal),e=tangent[pol],h=direction.cross(e);
                const Vec3 jfar=v-direction*direction.dot(v),mfar=direction.cross(v)*(-1.);
                const int index=basis.dofs[a],channel=2*p+pol;
                for(int j=0;j<=degree;++j) {
                    B[size_t(j)*n*channels+size_t(index)*channels+channel]-=weight*rot.dot(e)*plus[j];
                    B[size_t(j)*n*channels+size_t(c+index)*channels+channel]+=weight*rot.dot(h)*plus[j];
                    const C scale=-I*INV4PI*weight*(ka*minus[j]+(j?minus[j-1]:C(0)));
                    D[size_t(j)*channels*n+size_t(channel)*n+index]+=scale*e.dot(jfar);
                    D[size_t(j)*channels*n+size_t(channel)*n+c+index]+=scale*e.dot(mfar);
                }
            }
        }
    }
    const auto projected=Clock::now();
    double matrix_error=-1,rhs_error=-1,radiation_error=-1;
    if(std::stoi(argv[11])) {
        const auto reference=assemble_muller_nodal_dense(linear,ka,m,opts,quad_order,duffy);
        double delta=0,norm=0;for(size_t i=0;i<nn;++i){delta+=std::norm(A[i]-reference.matrix[i]);norm+=std::norm(reference.matrix[i]);}
        matrix_error=std::sqrt(delta/norm);
        const Vec3 d(grid[0],grid[1],grid[2]),e(grid[3],grid[4],grid[5]);
        const auto b=muller_nodal_planewave_rhs(mesh,ka,e,d,13);delta=norm=0;
        for(int j=0;j<n;++j){delta+=std::norm(B[size_t(j)*channels]-b[j]);norm+=std::norm(b[j]);}
        rhs_error=std::sqrt(delta/norm);
        std::vector<C> x(n),far;for(int j=0;j<n;++j)x[j]=C(std::sin(.7*j),std::cos(.3*j));
        muller_nodal_farfield(mesh,x.data(),x.data()+c,ka,{d},far,13);
        C predicted=0;for(int j=0;j<n;++j)predicted+=D[j]*x[j];
        const C actual=e.x*far[0]+e.y*far[1]+e.z*far[2];radiation_error=std::abs(predicted-actual)/std::abs(actual);
        if(std::max({matrix_error,rhs_error,radiation_error})>1e-10)throw std::runtime_error("independent library comparison failed");
    }
    const auto verified=Clock::now();
    write(prefix+".A.bin",A);write(prefix+".B.bin",B);write(prefix+".D.bin",D);
    const auto seconds=[](Clock::time_point a,Clock::time_point b){return std::chrono::duration<double>(b-a).count();};
    std::ofstream meta(prefix+".json");
    meta<<std::setprecision(17)<<"{\"dofs\":"<<n<<",\"elements\":"<<nt<<",\"points\":"<<points
        <<",\"ka\":"<<ka<<",\"ri\":"<<m.real()<<",\"ri_imag\":"<<m.imag()<<",\"prism_aspect\":"<<aspect
        <<",\"ref\":"<<ref<<",\"degree\":"<<degree<<",\"quad\":"<<quad_order<<",\"duffy\":"<<duffy
        <<",\"assembly_s\":"<<seconds(start,assembled)<<",\"incident_radiation_s\":"<<seconds(assembled,projected)
        <<",\"verification_s\":"<<seconds(projected,verified)<<",\"matrix_library_error\":"<<matrix_error
        <<",\"rhs_library_error\":"<<rhs_error<<",\"radiation_library_error\":"<<radiation_error<<"}\n";
    std::cout<<"Taylor bridge n="<<n<<" degree="<<degree<<" assembly="<<seconds(start,assembled)<<" s\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
