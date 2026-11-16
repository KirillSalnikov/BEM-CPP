#pragma once
// Optional CPU radiation backend. Requires the FINUFFT public headers/library.
// Original BDM1/Dunavant sum on affine planar triangles; no current or angular
// interpolation. Store the affine basis once per triangle, not per quad point.
#include "muller_nodal.h"
#include "quadrature.h"
#include <finufft.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class MullerFaceRadiation {
public:
    using Complex=std::complex<double>;
    using Vector=std::vector<Complex>;
private:
    struct Element {
        std::array<int,6> dofs;
        double coefficients[6][6]; // (tangent*3 + vertex) x local BDM dof
        double area;
    };
    struct Face {
        Vec3 origin,normal,u,v;
        std::vector<Element> elements;
        std::vector<double> x,y,s,t;
        Vector phase,strengths,transformed;
        finufft_plan plan=nullptr;
        ~Face(){if(plan)finufft_destroy(plan);}
    };
    int dofs_,threads_,points_per_element_;
    double wave_,eps_;
    bool parallel_faces_;
    TriQuad quad_;
    std::vector<Vec3> directions_;
    std::vector<std::unique_ptr<Face>> faces_;
    static void check(int code,const char* operation) {
        if(code)throw std::runtime_error(std::string(operation)+": FINUFFT status "+std::to_string(code));
    }
    static bool finite(const Vec3& x){return std::isfinite(x.x)&&std::isfinite(x.y)&&std::isfinite(x.z);}
    void project(Face& f,const Complex* const inputs[2]) {
        const size_t count=f.x.size();
#pragma omp parallel for schedule(static) if(!parallel_faces_) num_threads(threads_)
        for(size_t e=0;e<f.elements.size();++e){const auto& element=f.elements[e];
            for(int rhs=0;rhs<2;++rhs)for(int current=0;current<2;++current){
                Complex nodal[6]={};for(int j=0;j<6;++j){const Complex a=inputs[rhs][current*dofs_+element.dofs[j]];
                    for(int c=0;c<6;++c)nodal[c]+=element.coefficients[c][j]*a;}
                for(int q=0;q<quad_.npts;++q){const double a=quad_.pts[q][0],b=quad_.pts[q][1],w=element.area*quad_.wts[q];
                    for(int tangent=0;tangent<2;++tangent)
                        f.strengths[size_t(rhs*4+current*2+tangent)*count+e*quad_.npts+q]=
                            w*((1-a-b)*nodal[3*tangent]+a*nodal[3*tangent+1]+b*nodal[3*tangent+2]);}
            }
        }
    }
public:
    MullerFaceRadiation(const MullerP2Mesh& mesh,Complex wave,
                       const std::vector<Vec3>& directions,double eps=1e-10,
                       int threads=4,int quadrature=13,int max_faces=128,bool parallel_faces=false)
        :dofs_(mesh.current_dofs()),threads_(threads),wave_(wave.real()),eps_(eps),
         parallel_faces_(parallel_faces),quad_(tri_quadrature(quadrature)),directions_(directions) {
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1 || dofs_<=0 || mesh.elements.empty() ||
           directions.empty() || !std::isfinite(wave.real()) || wave.imag()!=0 || wave.real()<=0 ||
           !std::isfinite(eps) || eps<1e-14 || eps>1e-6 || threads<1 || threads>64 ||
           max_faces<1 || max_faces>1024 || (quadrature!=7 && quadrature!=13))
            throw std::invalid_argument("face radiation requires planar BDM1, positive real k, eps1e-14..1e-6 and threads1..64");
        for(const auto& d:directions)if(!finite(d)||std::abs(d.norm()-1)>1e-10)
            throw std::invalid_argument("face radiation needs finite unit observation directions");
        Vec3 lo=mesh.nodes.front(),hi=lo;
        for(const auto& p:mesh.nodes){if(!finite(p))throw std::invalid_argument("nonfinite face radiation mesh");
            lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
            hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);}
        const double scale=(hi-lo).norm();if(!(scale>0))throw std::invalid_argument("zero size radiation mesh");
        points_per_element_=quad_.npts;
        const double xi[3]={0,1,0},eta[3]={0,0,1};
        for(size_t e=0;e<mesh.elements.size();++e) {
            const auto center=evaluate_muller_frame(mesh,e,1./3,1./3);
            if(!finite(center.normal)||!std::isfinite(center.jacobian)||center.jacobian<=0)
                throw std::invalid_argument("degenerate radiation triangle");
            Face* face=nullptr;
            for(auto& candidate:faces_)
                if((center.normal-candidate->normal).norm()<2e-11 &&
                   std::abs((center.position-candidate->origin).dot(candidate->normal))<2e-11*scale){face=candidate.get();break;}
            if(!face){
                if(int(faces_.size())>=max_faces)throw std::invalid_argument("too many planar radiation faces");
                std::unique_ptr<Face> next(new Face);next->origin=center.position;next->normal=center.normal;
                Vec3 axis(1,0,0);if(std::abs(center.normal.y)<std::abs(center.normal.x))axis=Vec3(0,1,0);
                if(std::abs(center.normal.z)<std::min(std::abs(center.normal.x),std::abs(center.normal.y)))axis=Vec3(0,0,1);
                next->u=center.normal.cross(axis);next->u=next->u*(1/next->u.norm());next->v=center.normal.cross(next->u);
                face=next.get();faces_.push_back(std::move(next));
            }
            Element element;Vec3 vertices[3];
            for(int vertex=0;vertex<3;++vertex){
                const auto sample=evaluate_muller_frame(mesh,e,xi[vertex],eta[vertex]);vertices[vertex]=sample.position;
                const auto basis=evaluate_muller_basis(mesh,e,sample);
                if(basis.count!=6 || (sample.normal-face->normal).norm()>2e-11)
                    throw std::invalid_argument("non-affine or curved BDM radiation triangle");
                for(int j=0;j<6;++j){if(vertex==0)element.dofs[j]=basis.dofs[j];
                    else if(element.dofs[j]!=basis.dofs[j])throw std::invalid_argument("BDM basis map changes within triangle");
                    if(!finite(basis.values[j]) || std::abs(basis.values[j].dot(face->normal))>2e-11*basis.values[j].norm())
                        throw std::invalid_argument("radiation basis is not tangential");
                    element.coefficients[vertex][j]=basis.values[j].dot(face->u);
                    element.coefficients[3+vertex][j]=basis.values[j].dot(face->v);}
            }
            element.area=.5*center.jacobian;
            for(int q=0;q<quad_.npts;++q){const double a=quad_.pts[q][0],b=quad_.pts[q][1];
                const auto sample=evaluate_muller_frame(mesh,e,a,b);
                const Vec3 interpolated=vertices[0]*(1-a-b)+vertices[1]*a+vertices[2]*b;
                if((sample.position-interpolated).norm()>2e-11*scale ||
                   (sample.normal-face->normal).norm()>2e-11 ||
                   std::abs(sample.jacobian-center.jacobian)>2e-11*center.jacobian)
                    throw std::invalid_argument("face radiation requires affine planar triangles");
                const Vec3 local=sample.position-face->origin;
                face->x.push_back(local.dot(face->u));face->y.push_back(local.dot(face->v));}
            face->elements.push_back(element);
        }
        for(auto& pointer:faces_){auto& face=*pointer;const size_t count=directions_.size();
            face.s.resize(count);face.t.resize(count);face.phase.resize(count);
            for(size_t d=0;d<count;++d){const auto& r=directions_[d];
                face.s[d]=wave_*r.dot(face.u);face.t[d]=wave_*r.dot(face.v);
                face.phase[d]=Complex(0,-wave_*INV4PI)*std::exp(Complex(0,-wave_*r.dot(face.origin)));}
            face.strengths.resize(8*face.x.size());face.transformed.resize(8*count);
            finufft_opts opts;finufft_default_opts(&opts);opts.nthreads=parallel_faces_?1:threads_;opts.showwarn=0;
            // All 8 fields in one batch; each source has only two tangential
            // components for each of J/M and two independent incident modes.
            opts.maxbatchsize=8;
            check(finufft_makeplan(3,2,nullptr,-1,8,eps_,&face.plan,&opts),"make face plan");
            check(finufft_setpts(face.plan,face.x.size(),face.x.data(),face.y.data(),nullptr,
                                count,face.s.data(),face.t.data(),nullptr),"set face points");
        }
    }
    MullerFaceRadiation(const MullerFaceRadiation&)=delete;
    MullerFaceRadiation& operator=(const MullerFaceRadiation&)=delete;
    size_t faces()const{return faces_.size();}
    size_t points()const{size_t n=0;for(const auto& f:faces_)n+=f->x.size();return n;}
    // Owned arrays only: FINUFFT internal plans/workspace are measured by RSS.
    size_t owned_bytes()const{size_t n=directions_.capacity()*sizeof(Vec3);
        for(const auto& f:faces_)n+=f->elements.capacity()*sizeof(Element)+
            (f->x.capacity()+f->y.capacity()+f->s.capacity()+f->t.capacity())*sizeof(double)+
            (f->phase.capacity()+f->strengths.capacity()+f->transformed.capacity())*sizeof(Complex);
        return n;}
    void apply_pair(const Complex* first,const Complex* second,Vector& field_first,Vector& field_second) {
        if(!first||!second||&field_first==&field_second)throw std::invalid_argument("radiation pair input/output alias");
        for(int j=0;j<2*dofs_;++j)if(!std::isfinite(first[j].real())||!std::isfinite(first[j].imag())||
            !std::isfinite(second[j].real())||!std::isfinite(second[j].imag()))throw std::invalid_argument("nonfinite current");
        const Complex* inputs[2]={first,second};
        // Independent face plans can execute concurrently. Every plan is used
        // by only one thread; the final reduction keeps a fixed face order.
        // Finish gathering all inputs before touching outputs (in-place safe).
        std::vector<int> status(faces_.size());
#pragma omp parallel for schedule(dynamic,1) if(parallel_faces_) num_threads(threads_)
        for(size_t i=0;i<faces_.size();++i){auto& f=*faces_[i];project(f,inputs);
            status[i]=finufft_execute(f.plan,f.strengths.data(),f.transformed.data());}
        for(int code:status)check(code,"execute face plan");
        const size_t nd=directions_.size();field_first.assign(3*nd,0.);field_second.assign(3*nd,0.);
        Vector* fields[2]={&field_first,&field_second};
#pragma omp parallel for schedule(static) num_threads(threads_)
        for(size_t d=0;d<nd;++d){const auto& r=directions_[d];
            for(auto& pointer:faces_){const auto& f=*pointer;
                const Vec3 ju=f.u-r*r.dot(f.u),jv=f.v-r*r.dot(f.v),mu=r.cross(f.u)*(-1.),mv=r.cross(f.v)*(-1.);
                for(int rhs=0;rhs<2;++rhs){const Complex u=f.transformed[size_t(rhs*4)*nd+d],v=f.transformed[size_t(rhs*4+1)*nd+d],
                    a=f.transformed[size_t(rhs*4+2)*nd+d],b=f.transformed[size_t(rhs*4+3)*nd+d],p=f.phase[d];
                    auto* out=fields[rhs]->data()+3*d;
                    out[0]+=p*(ju.x*u+jv.x*v+mu.x*a+mv.x*b);
                    out[1]+=p*(ju.y*u+jv.y*v+mu.y*a+mv.y*b);
                    out[2]+=p*(ju.z*u+jv.z*v+mu.z*a+mv.z*b);
                }
            }
        }
    }
};
