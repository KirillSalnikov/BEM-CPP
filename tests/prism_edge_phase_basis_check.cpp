#include "prism_edge_phase_basis.h"
#include "tmatrix_symmetry_warm_start.h"
#include <cstdio>
using namespace bem_edge_phase;
using V=std::vector<C>;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
double relative(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}return std::sqrt(x/std::max(y,1e-300));}
Vec3 axis(int i){return i==0?Vec3(1,0,0):i==1?Vec3(0,1,0):Vec3(0,0,1);}
Vec3 map(const Vec3& p,int symmetry){return symmetry==0?Vec3(.5*p.x-std::sqrt(3.)*.5*p.y,std::sqrt(3.)*.5*p.x+.5*p.y,p.z):symmetry==1?Vec3(p.x,p.y,-p.z):Vec3(p.x,-p.y,p.z);}
double normal_jump(const MullerP2Mesh& mesh,const V& current) {
    V first(size_t(mesh.topology_edge_count)*4);std::vector<int> seen(mesh.topology_edge_count);double difference=0,norm=0;
    for(size_t t=0;t<mesh.elements.size();++t){const auto& e=mesh.elements[t];for(int edge=0;edge<3;++edge){
        int a=e.topology_vertices[edge],b=e.topology_vertices[(edge+1)%3];const bool forward=a<b;if(!forward)std::swap(a,b);
        const Vec3 delta=mesh.nodes[b]-mesh.nodes[a];const int id=e.topology_edges[edge];
        for(int sample=0;sample<2;++sample){const double global=sample?.71:.23,u=forward?global:1-global;
            const auto frame=evaluate_muller_frame(mesh,t,edge==0?u:edge==1?1-u:0,edge==0?0:edge==1?u:1-u);
            const auto basis=evaluate_muller_basis(mesh,t,frame);const Vec3 conormal=delta.cross(frame.normal);
            for(int kind=0;kind<2;++kind){C flux=0;for(int j=0;j<basis.count;++j)
                    flux+=current[kind*mesh.current_dofs()+basis.dofs[j]]*basis.values[j].dot(conormal);
                const int at=4*id+2*sample+kind;
                if(!seen[id])first[at]=flux;else {difference+=std::norm(flux-first[at]);norm+=std::norm(first[at]);}}}
        ++seen[id];}}
    for(int visits:seen)require(visits==2,"missing incident face in normal-flux audit");
    return std::sqrt(difference/std::max(norm,1e-300));
}
int main()try {
    Specification s;s.center=Vec3(.75,.43,.05);s.tangent=Vec3(0,0,1);s.phase_vector=Vec3(.6,.3,.85);
    s.index=1.3116;s.core=.06;s.width=.31;s.half_length=.53;s.path=.21;
    double jet_error=0,quadrature_error=0,symmetry_error=0,plane_error=0,flux_jump=0;
    const double offsets[]={-2,-1,1,2},coefficients[]={1,-8,8,-1};
    for(auto family:{Family::EdgeCone,Family::ShadowPlane})for(double k:{.3,3.,40.})for(double parallel:{.85,1.7}) {
        s.family=family;s.k=k;s.phase_vector.z=parallel;HertzTrace wave(s);
        require(wave.radial_wave_number().imag()>=0,"growing evanescent phase");
        for(const Vec3 p:{s.center,Vec3(.42,.19,.17),Vec3(.89,.48,-.15)}) {
            const auto j=wave.scalar(p);const double h=2e-4/(1+k);
            for(int a=0;a<3;++a){C derivative=0;
                for(int q=0;q<4;++q)derivative+=coefficients[q]*wave.scalar(p+axis(a)*(h*offsets[q])).value/(12*h);
                jet_error=std::max(jet_error,std::abs(derivative-j.gradient[a])/(1+std::abs(j.gradient[a])));
                for(int b=0;b<3;++b){C second=0;
                    // Independent fourth-order tensor-product finite difference
                    // of the scalar value, not a differentiation of our gradient.
                    for(int q=0;q<4;++q)for(int r=0;r<4;++r)
                        second+=coefficients[q]*coefficients[r]*wave.scalar(p+axis(a)*(h*offsets[q])+axis(b)*(h*offsets[r])).value/(144*h*h);
                    jet_error=std::max(jet_error,std::abs(second-j.hessian[a][b])/(1+std::abs(j.hessian[a][b])));}
            }
            for(int pol=0;pol<2;++pol){const auto f=wave.fields(p,pol);
                for(auto z:f.electric)require(std::isfinite(z.real())&&std::isfinite(z.imag()),"nonfinite E");
                for(auto z:f.magnetic)require(std::isfinite(z.real())&&std::isfinite(z.imag()),"nonfinite H");}
        }
    }
    require(jet_error<5e-5,"scalar phase derivatives disagree with independent finite differences");
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,1,1.,0,false,true),options);
    TraceProjector projector(mesh);bem_tmatrix::D6hOrbit group;group.build(mesh);
    for(auto family:{Family::EdgeCone,Family::ShadowPlane})for(double k:{3.,40.})for(int pol=0;pol<2;++pol) {
        s.family=family;s.k=k;s.phase_vector=Vec3(.6,.3,.85);HertzTrace wave(s);
        const auto v=projector.project(wave,pol),fine=projector.project(wave,pol,2);
        quadrature_error=std::max(quadrature_error,relative(v,fine));
        flux_jump=std::max(flux_jump,normal_jump(mesh,v));
        for(int g=0;g<3;++g){auto other=s;other.center=map(s.center,g);other.tangent=map(s.tangent,g);other.phase_vector=map(s.phase_vector,g);
            const auto actual=projector.project(HertzTrace(other),pol);V expected(v.size());
            if(g<2)group.transform(v,expected,g==1,false);
            else {std::vector<int> image,sign;bem_tmatrix::D6hOrbit::vertical_image_map(mesh,image,sign);
                for(size_t i=0;i<v.size();++i)expected[image[i]]=double(sign[i])*v[i];}
            // Magnetic-Hertz parameters transform as axial vectors; using a
            // polar tangent instead supplies the same reflected basis up to -1.
            if(pol && g)for(auto& z:expected)z=-z;
            symmetry_error=std::max(symmetry_error,relative(actual,expected));
        }
    }
    require(quadrature_error<2e-10,"edge integration has not converged");
    require(symmetry_error<2e-10,"edge phase lost rotation or mirror covariance");
    require(flux_jump<5e-12,"projected electric/magnetic normal flux is discontinuous");
    // Wide-envelope limit: independently integrated constant plane-wave traces.
    s.family=Family::ShadowPlane;s.k=7.3;s.width=s.half_length=1e8;
    const Vec3 direction=Vec3(.3,.4,std::sqrt(.75));s.phase_vector=direction*s.index;
    const double xi=s.phase_vector.dot(s.tangent);const Vec3 e0=s.tangent*(s.index*s.index)-s.phase_vector*xi;
    const Vec3 h0=s.phase_vector.cross(s.tangent)*(s.index*s.index);
    for(int pol=0;pol<2;++pol){const auto value=projector.project(HertzTrace(s),pol);V exact(value.size());
        const Vec3 e=pol?h0*(1/s.index):e0,h=pol?e0*(-s.index):h0;
        for(const auto& element:mesh.elements)for(int edge=0;edge<3;++edge){
            int a=element.topology_vertices[edge],b=element.topology_vertices[(edge+1)%3];if(a>b)std::swap(a,b);
            const Vec3 delta=mesh.nodes[b]-mesh.nodes[a],center=(mesh.nodes[a]+mesh.nodes[b])*.5;
            const double u=.5*s.k*s.phase_vector.dot(delta),u2=u*u;
            const double sinc=std::abs(u)<1e-3?1-u2/6+u2*u2/120:std::sin(u)/u;
            const double j1=std::abs(u)<1e-3?u*(1./3-u2/30+u2*u2/840):(std::sin(u)-u*std::cos(u))/u2;
            const C phase=std::exp(C(0,s.k*(s.path+s.phase_vector.dot(center-s.center))));const int id=element.topology_edges[edge];
            for(int m=0;m<2;++m){const C integral=phase*(m?C(0,j1):C(sinc));exact[2*id+m]=-h.dot(delta)*integral;exact[mesh.current_dofs()+2*id+m]=e.dot(delta)*integral;}}
        plane_error=std::max(plane_error,relative(value,exact));
    }
    require(plane_error<2e-12,"Hertz-to-BDM current signs or edge moments wrong");
    auto bad=s;bad.core=0;bool rejected=false;try{HertzTrace invalid(bad);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"invalid core accepted");
    std::printf("{\"scalar_jet_error\":%.17g,\"quadrature_error\":%.17g,\"D6h_defect\":%.17g,\"plane_trace_error\":%.17g,\"normal_flux_jump\":%.17g,\"passed\":true}\n",jet_error,quadrature_error,symmetry_error,plane_error,flux_jump);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
