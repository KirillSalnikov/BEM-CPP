#pragma once
#include "muller_nodal.h"
#include <algorithm>
#include <array>
#include <complex>
#include <stdexcept>
#include <vector>

// Research correction space inspired by HNA phase enrichment. These smooth
// 3-D Hertz traces are NOT an exact penetrable-wedge diffraction solution.
// Their unknown amplitudes are found through the original Muller equation.
// Both electric and magnetic Hertz types are supplied. The scalar potential
// is globally defined, including across creases, so its BDM edge flux is unique.
namespace bem_edge_phase {
using C=std::complex<double>;
using CV=std::array<C,3>;
inline double component(const Vec3& v,int i){return i==0?v.x:i==1?v.y:v.z;}
inline bool finite(const Vec3& p){return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);}
inline CV cross(const CV& a,const Vec3& b){return {{a[1]*b.z-a[2]*b.y,a[2]*b.x-a[0]*b.z,a[0]*b.y-a[1]*b.x}};}
struct ScalarJet {C value;CV gradient;std::array<CV,3> hessian;};
struct Fields {CV electric,magnetic;};
enum class Family { EdgeCone, ShadowPlane };
struct Specification {
    Vec3 center,tangent,phase_vector;
    double k=0,index=1,core=0,width=0,half_length=0,path=0;
    Family family=Family::EdgeCone;
};

class HertzTrace {
    Specification s_;
    double parallel_;
    C radial_;
public:
    explicit HertzTrace(Specification s):s_(s) {
        if(!finite(s.center)||!finite(s.tangent)||!finite(s.phase_vector)||
           !std::isfinite(s.k)||s.k<=0||!std::isfinite(s.index)||s.index<=0||
           !std::isfinite(s.core)||s.core<=0||!std::isfinite(s.width)||s.width<=0||
           !std::isfinite(s.half_length)||s.half_length<=0||!std::isfinite(s.path)||
           std::abs(s.tangent.norm()-1)>2e-12)
            throw std::invalid_argument("invalid edge-phase specification");
        parallel_=s.tangent.dot(s.phase_vector);
        // Principal root: outgoing propagating cone or exponentially decaying
        // evanescent cone when an internal ray's tangential phase is too large.
        radial_=std::sqrt(C(s.index*s.index-parallel_*parallel_,0));
        if(radial_.imag()<0)radial_=-radial_;
    }
    const Specification& specification()const{return s_;}
    C radial_wave_number()const{return radial_;}
    ScalarJet scalar(const Vec3& point)const {
        const auto& s=s_;const C I(0,1);const Vec3 x=point-s.center;
        const double along=s.tangent.dot(x),a=1/(s.half_length*s.half_length),b=1/(s.width*s.width);
        const Vec3 transverse=x-s.tangent*along;
        const double rho2=transverse.dot(transverse),radius=std::sqrt(rho2+s.core*s.core);
        CV g;std::array<CV,3> h;
        C logarithm=-.5*(a*along*along+b*rho2)+I*s.k*s.path;
        if(s.family==Family::ShadowPlane) {
            logarithm+=I*s.k*s.phase_vector.dot(x);
            for(int i=0;i<3;++i){g[i]=-a*along*component(s.tangent,i)-b*component(transverse,i)+I*s.k*component(s.phase_vector,i);
                for(int j=0;j<3;++j){const double tt=component(s.tangent,i)*component(s.tangent,j);
                    h[i][j]=-a*tt-b*((i==j?1.:0.)-tt);}}
        }else {
            logarithm+=I*s.k*(parallel_*along+radial_*(radius-s.core))-.5*std::log1p(s.k*radius);
            const C u=-b*radius-.5*s.k/(1+s.k*radius)+I*s.k*radial_;
            const double derivative=-b+.5*s.k*s.k/std::pow(1+s.k*radius,2);
            for(int i=0;i<3;++i){const double ti=component(s.tangent,i),ri=component(transverse,i);
                g[i]=ti*(-a*along+I*s.k*parallel_)+ri*u/radius;
                for(int j=0;j<3;++j){const double tj=component(s.tangent,j),rj=component(transverse,j);
                    h[i][j]=-a*ti*tj+u/radius*((i==j?1.:0.)-ti*tj)
                        +(derivative/(radius*radius)-u/(radius*radius*radius))*ri*rj;}}
        }
        ScalarJet jet;jet.value=std::exp(logarithm);
        for(int i=0;i<3;++i){jet.gradient[i]=jet.value*g[i];
            for(int j=0;j<3;++j)jet.hessian[i][j]=jet.value*(h[i][j]+g[i]*g[j]);}
        return jet;
    }
    Fields fields(const Vec3& point,int polarization)const {
        if(polarization!=0 && polarization!=1)throw std::invalid_argument("Hertz polarization");
        const auto jet=scalar(point);const auto curl=cross(jet.gradient,s_.tangent);
        Fields f;const double n2=s_.index*s_.index;
        // Electric Hertz potential Pi=t*phi: E=n² Pi+grad(div Pi)/k²,
        // H=n² curl(Pi)/(i*k). This curl relation is exact for our smooth
        // potential; the localized envelope need not satisfy Helmholtz.
        for(int i=0;i<3;++i){f.electric[i]=n2*component(s_.tangent,i)*jet.value;
            for(int j=0;j<3;++j)f.electric[i]+=jet.hessian[i][j]*component(s_.tangent,j)/(s_.k*s_.k);
            f.magnetic[i]=C(0,-n2/s_.k)*curl[i];}
        if(polarization)for(int i=0;i<3;++i){const C e=f.electric[i];f.electric[i]=f.magnetic[i]/s_.index;f.magnetic[i]=-s_.index*e;}
        return f;
    }
};

class TraceProjector {
    struct Edge {Vec3 start,delta;};
    std::vector<Edge> edges_;
public:
    explicit TraceProjector(const MullerP2Mesh& mesh) {
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1)throw std::invalid_argument("phase trace needs BDM1");
        edges_.resize(mesh.topology_edge_count);std::vector<int> seen(edges_.size());
        for(const auto& e:mesh.elements)for(int j=0;j<3;++j){
            int a=e.topology_vertices[j],b=e.topology_vertices[(j+1)%3];if(a>b)std::swap(a,b);
            const int id=e.topology_edges[j];++seen[id];edges_[id]={mesh.nodes[a],mesh.nodes[b]-mesh.nodes[a]};}
        for(int count:seen)if(count!=2)throw std::invalid_argument("phase trace requires a closed manifold");
    }
    size_t bytes()const{return edges_.size()*sizeof(Edge);}
    std::vector<C> project(const HertzTrace& field,int polarization,int refinement=1)const {
        if(polarization<0||polarization>1||refinement<1||refinement>8)throw std::invalid_argument("phase projection controls");
        const auto& s=field.specification();const size_t currents=2*edges_.size();
        std::vector<C> out(2*currents);std::vector<int> parts(edges_.size());
        const double inverse_scale=std::max(s.k*(s.phase_vector.norm()+s.index)/8,
            .5/std::min({s.core,s.width,s.half_length}));
        for(size_t e=0;e<edges_.size();++e){const double wanted=std::ceil(edges_[e].delta.norm()*inverse_scale);
            if(!std::isfinite(wanted)||wanted>4096/refinement)throw std::invalid_argument("phase quadrature limit");
            parts[e]=refinement*std::max(1,int(wanted));}
        const double nodes[]={-.9894009349916499,-.9445750230732326,-.8656312023878318,-.7554044083550030,-.6178762444026438,-.4580167776572274,-.2816035507792589,-.0950125098376374,.0950125098376374,.2816035507792589,.4580167776572274,.6178762444026438,.7554044083550030,.8656312023878318,.9445750230732326,.9894009349916499};
        const double weights[]={.0271524594117541,.0622535239386479,.0951585116824928,.1246289712555339,.1495959888165767,.1691565193950025,.1826034150449236,.1894506104550685,.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,.1246289712555339,.0951585116824928,.0622535239386479,.0271524594117541};
#pragma omp parallel for schedule(dynamic,16)
        for(size_t e=0;e<edges_.size();++e)for(int part=0;part<parts[e];++part)for(int q=0;q<16;++q){
            const auto& edge=edges_[e];const double u=(part+.5*(nodes[q]+1))/parts[e],weight=.5*weights[q]/parts[e];
            const auto f=field.fields(edge.start+edge.delta*u,polarization);
            C jflux=0,mflux=0;for(int axis=0;axis<3;++axis){jflux-=component(edge.delta,axis)*f.magnetic[axis];mflux+=component(edge.delta,axis)*f.electric[axis];}
            for(int moment=0;moment<2;++moment){const double w=weight*(moment?2*u-1:1);
                out[2*e+moment]+=w*jflux;out[currents+2*e+moment]+=w*mflux;}}
        return out;
    }
};
}
