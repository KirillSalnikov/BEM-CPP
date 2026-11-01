#pragma once
#include "muller_nodal.h"
#include <array>
#include <complex>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Experimental geometric-optics boundary trace. An initial guess only: the
// final solution must satisfy the original BEM operator's strict residual.
// Fields use exp(+ik d.r), J=n x H, M=-n x E, exterior impedance = 1.
class MbsSurfaceSeed {
public:
    using C=std::complex<double>;
    struct Face {Vec3 normal,point;};
    struct Ray {
        int face,depth;double path;Vec3 d,s;std::array<C,4> j;std::vector<Vec3> vertices,inward;
        std::array<std::array<C,3>,2> electric,magnetic;
    };
    double ri=0,aspect=0,trace_seconds=0;int depth=0;
    bool internal_traces=false;
    Vec3 direction,polar_basis;
    std::vector<Face> faces;std::vector<Ray> rays;
    std::array<std::vector<size_t>,8> face_rays;
    explicit MbsSurfaceSeed(const std::string& path) {
        std::ifstream in(path);std::string magic;in>>magic;
        if(magic!="MBS_SURFACE_RAYS_V1" && magic!="MBS_SURFACE_RAYS_V2")throw std::runtime_error("invalid MBS ray export");
        internal_traces=magic=="MBS_SURFACE_RAYS_V2";
        in>>ri>>aspect>>depth>>trace_seconds;
        auto vec=[&](Vec3& p){in>>p.x>>p.y>>p.z;};vec(direction);vec(polar_basis);
        size_t nf=0,nr=0;in>>nf>>nr;
        if(!in||nf!=8||nr>1000000||!std::isfinite(ri)||ri<=0||!std::isfinite(aspect)||aspect<=0)
            throw std::runtime_error("invalid ray dimensions or material");
        faces.resize(nf);for(auto& f:faces){vec(f.normal);vec(f.point);}
        rays.resize(nr);for(auto& r:rays){size_t nv=0;in>>r.face>>r.depth>>nv>>r.path;
            if(!in||r.face<0||r.face>=int(nf)||nv<3||nv>256)throw std::runtime_error("invalid ray polygon");
            vec(r.d);vec(r.s);for(auto& z:r.j){double a,b;in>>a>>b;z=C(a,b);}
            r.vertices.resize(nv);for(auto& p:r.vertices)vec(p);}
        if(!in)throw std::runtime_error("truncated ray export");
        // Cache geometric half-spaces and vector polarization amplitudes once;
        // the projection evaluates these same polygons at many BDM edge nodes.
        const double index=internal_traces?ri:1.;
        for(size_t id=0;id<rays.size();++id){auto& r=rays[id];face_rays[r.face].push_back(id);
            Vec3 center;for(const auto& p:r.vertices)center=center+p;center=center*(1./r.vertices.size());
            for(size_t j=0;j<r.vertices.size();++j){const Vec3 a=r.vertices[j],edge=r.vertices[(j+1)%r.vertices.size()]-a;
                Vec3 normal=faces[r.face].normal.cross(edge);if(normal.dot(center-a)<0)normal=normal*(-1.);
                r.inward.push_back(normal);}
            const Vec3 p=r.s.cross(r.d);
            for(int pol=0;pol<2;++pol){auto& e=r.electric[pol];auto& h=r.magnetic[pol];
                const C a=r.j[pol],b=r.j[2+pol];e={{a*p.x+b*r.s.x,a*p.y+b*r.s.y,a*p.z+b*r.s.z}};
                h={{index*(r.d.y*e[2]-r.d.z*e[1]),index*(r.d.z*e[0]-r.d.x*e[2]),index*(r.d.x*e[1]-r.d.y*e[0])}};}
        }
    }
    Vec3 polarization(int pol)const {return pol?polar_basis:polar_basis.cross(direction);}
    int face(const Vec3& normal,const Vec3& point)const {
        for(size_t f=0;f<faces.size();++f)
            if((faces[f].normal-normal).norm()<1e-7 && std::abs(normal.dot(point-faces[f].point))<1e-7)return int(f);
        throw std::runtime_error("MBS/BEM face geometry mismatch");
    }
    bool inside(const Ray& r,const Vec3& point)const {
        for(size_t v=0;v<r.vertices.size();++v)
            if(r.inward[v].dot(point-r.vertices[v]) < -1e-14)return false;
        return true;
    }
    void fields(int f,const Vec3& point,double k,int pol,std::array<C,3>& e,std::array<C,3>& h)const {
        e.fill(0);h.fill(0);
        const auto add=[&](const Vec3& d,const Vec3& u,const Vec3& v,C a,C b,C phase,double index){
            const C ex=phase*(a*u.x+b*v.x),ey=phase*(a*u.y+b*v.y),ez=phase*(a*u.z+b*v.z);
            e[0]+=ex;e[1]+=ey;e[2]+=ez;h[0]+=index*(d.y*ez-d.z*ey);h[1]+=index*(d.z*ex-d.x*ez);h[2]+=index*(d.x*ey-d.y*ex);};
        // Incident ray exists on illuminated faces. At exactly grazing faces
        // ray optics does not supply a reliable boundary trace; include the
        // incident limiting field and leave its correction to BEM.
        if(!internal_traces && faces[f].normal.dot(direction)<=1e-12)
            add(direction,polarization(pol),Vec3(),1.,0.,std::exp(C(0,k*direction.dot(point))),1.);
        const double index=internal_traces?ri:1.;
        for(size_t id:face_rays.at(f)){const auto& r=rays[id];if(inside(r,point)){
            const C phase=std::exp(C(0,k*(r.path+index*r.d.dot(point))));
            for(int j=0;j<3;++j){e[j]+=phase*r.electric[pol][j];h[j]+=phase*r.magnetic[pol][j];}}}
    }
    std::vector<C> project(const MullerP2Mesh& mesh,double k,int pol)const {
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1)throw std::runtime_error("MBS seed requires BDM1");
        const double nodes[]={-.9894009349916499,-.9445750230732326,-.8656312023878318,-.7554044083550030,-.6178762444026438,-.4580167776572274,-.2816035507792589,-.0950125098376374,.0950125098376374,.2816035507792589,.4580167776572274,.6178762444026438,.7554044083550030,.8656312023878318,.9445750230732326,.9894009349916499};
        const double weights[]={.0271524594117541,.0622535239386479,.0951585116824928,.1246289712555339,.1495959888165767,.1691565193950025,.1826034150449236,.1894506104550685,.1894506104550685,.1826034150449236,.1691565193950025,.1495959888165767,.1246289712555339,.0951585116824928,.0622535239386479,.0271524594117541};
        std::vector<C> out(mesh.system_dofs());std::vector<int> count(mesh.topology_edge_count);
        const int current=mesh.current_dofs();
        for(size_t t=0;t<mesh.elements.size();++t){const auto& el=mesh.elements[t];
            const auto frame=evaluate_muller_frame(mesh,t,1./3,1./3);const int f=face(frame.normal,frame.position);
            for(int edge=0;edge<3;++edge){int a=el.topology_vertices[edge],b=el.topology_vertices[(edge+1)%3];if(a>b)std::swap(a,b);
                const Vec3 start=mesh.nodes[a],delta=mesh.nodes[b]-start;const int id=el.topology_edges[edge];++count[id];
                for(int q=0;q<16;++q){const double s=(1+nodes[q])*.5;const Vec3 on_edge=start+delta*s;
                    // Take the incident-face limit before averaging the two
                    // traces at feature edges and ray-polygon discontinuities.
                    const Vec3 point=on_edge+(frame.position-on_edge)*1e-8;
                    std::array<C,3> e,h;fields(f,point,k,pol,e,h);
                    const C jflux=-(delta.x*h[0]+delta.y*h[1]+delta.z*h[2]);
                    const C mflux=delta.x*e[0]+delta.y*e[1]+delta.z*e[2];
                    for(int m=0;m<2;++m){const double w=.5*weights[q]*(m?nodes[q]:1.);
                        out[2*id+m]+=w*jflux;out[current+2*id+m]+=w*mflux;}
                }
            }
        }
        for(size_t id=0;id<count.size();++id){if(count[id]!=2)throw std::runtime_error("seed needs closed manifold edges");
            for(int m=0;m<2;++m){out[2*id+m]/=count[id];out[current+2*id+m]/=count[id];}}
        return out;
    }
};
