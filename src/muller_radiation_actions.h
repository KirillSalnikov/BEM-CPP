#pragma once
#include "muller_nodal.h"
#include "quadrature.h"
#include <complex>
#include <stdexcept>
#include <vector>

// Direct FP64 transverse radiation and its Euclidean complex adjoint.
// BDM geometry/quadrature and incidence maps are cached; no angle-by-DOF
// radiation matrix or all-angle phase cache is stored. A block reuses each
// phase across independent currents. This is a CPU action, not an FFT.
class MullerRadiationActions {
public:
    using Complex=std::complex<double>;
    using Vector=std::vector<Complex>;
private:
    struct Point {Vec3 position;double weight;MullerBasisSample basis;};
    struct Incident {int point,local;};
    int current_=0;
    Complex wave_,prefactor_;
    std::vector<Vec3> directions_,electric_,magnetic_;
    std::vector<Point> points_;
    std::vector<int> offsets_;
    std::vector<Incident> incidence_;
    static bool finite(Complex z){return std::isfinite(z.real())&&std::isfinite(z.imag());}
    void validate(const Vector& input,int batch,int rows)const {
        if(batch<1||batch>64||input.size()!=size_t(batch)*rows)throw std::invalid_argument("radiation batch dimensions");
        for(auto z:input)if(!finite(z))throw std::invalid_argument("nonfinite radiation input");
    }
public:
    MullerRadiationActions(const MullerP2Mesh& mesh,Complex wave,
        const std::vector<Vec3>& directions,const std::vector<Vec3>& first,
        const std::vector<Vec3>& second,int quadrature_order=13)
        :current_(mesh.current_dofs()),wave_(wave),prefactor_(Complex(0,-INV4PI)*wave),directions_(directions) {
        if(current_<1||directions.empty()||first.size()!=directions.size()||second.size()!=directions.size()||!finite(wave))
            throw std::invalid_argument("radiation geometry dimensions");
        for(size_t d=0;d<directions.size();++d){const auto& r=directions[d];
            for(const auto& t:{first[d],second[d]}){
                if(!std::isfinite(r.norm())||!std::isfinite(t.norm())||std::abs(r.norm()-1)>1e-10||
                   std::abs(t.norm()-1)>1e-10||std::abs(r.dot(t))>1e-10)
                    throw std::invalid_argument("radiation frames must be unit and transverse");
                electric_.push_back(t-r*r.dot(t));magnetic_.push_back(r.cross(t));}
            if(std::abs(first[d].dot(second[d]))>1e-10)throw std::invalid_argument("radiation frame is not orthogonal");}
        const auto quadrature=tri_quadrature(quadrature_order);
        offsets_.assign(current_+1,0);
        for(size_t e=0;e<mesh.elements.size();++e)for(int q=0;q<quadrature.npts;++q){
            const auto frame=evaluate_muller_frame(mesh,e,quadrature.pts[q][0],quadrature.pts[q][1]);
            Point p{frame.position,.5*quadrature.wts[q]*frame.jacobian,evaluate_muller_basis(mesh,e,frame)};
            for(int j=0;j<p.basis.count;++j)++offsets_[p.basis.dofs[j]+1];
            points_.push_back(p);}
        for(int j=0;j<current_;++j)offsets_[j+1]+=offsets_[j];
        incidence_.resize(offsets_.back());auto cursor=offsets_;
        for(size_t q=0;q<points_.size();++q)for(int j=0;j<points_[q].basis.count;++j)
            incidence_[cursor[points_[q].basis.dofs[j]]++]={int(q),j};
    }
    int dofs()const{return 2*current_;}
    int channels()const{return 2*int(directions_.size());}
    size_t quadrature_points()const{return points_.size();}
    size_t storage_bytes()const{return points_.capacity()*sizeof(Point)+incidence_.capacity()*sizeof(Incident)+
        offsets_.capacity()*sizeof(int)+(directions_.capacity()+electric_.capacity()+magnetic_.capacity())*sizeof(Vec3);}
    // Inputs and outputs use contiguous column-major blocks: current/field
    // index is fastest, then RHS. In-place calls are safe through temporaries.
    void apply(const Vector& input,int batch,Vector& output)const {
        validate(input,batch,dofs());const size_t nq=points_.size();
        Vector samples(nq*batch*6),result(size_t(channels())*batch);
#pragma omp parallel for schedule(static)
        for(size_t q=0;q<nq;++q){const auto& b=points_[q].basis;
            for(int j=0;j<b.count;++j){const auto& v=b.values[j];
                for(int h=0;h<batch;++h){auto* a=samples.data()+(q*batch+h)*6;
                    const auto ej=input[size_t(h)*dofs()+b.dofs[j]],em=input[size_t(h)*dofs()+current_+b.dofs[j]];
                    a[0]+=v.x*ej;a[1]+=v.y*ej;a[2]+=v.z*ej;
                    a[3]+=v.x*em;a[4]+=v.y*em;a[5]+=v.z*em;}}}
#pragma omp parallel for schedule(static)
        for(int d=0;d<int(directions_.size());++d){
            for(size_t q=0;q<nq;++q){const auto& p=points_[q];
                const Complex factor=prefactor_*p.weight*std::exp(Complex(0,-1)*wave_*directions_[d].dot(p.position));
                for(int pol=0;pol<2;++pol){const int row=2*d+pol;const auto& e=electric_[row];const auto& m=magnetic_[row];
                    for(int h=0;h<batch;++h){const auto* a=samples.data()+(q*batch+h)*6;
                        result[size_t(h)*channels()+row]+=factor*(e.x*a[0]+e.y*a[1]+e.z*a[2]+m.x*a[3]+m.y*a[4]+m.z*a[5]);}}}}
        output.swap(result);
    }
    void adjoint(const Vector& input,int batch,Vector& output)const {
        validate(input,batch,channels());const size_t nq=points_.size();
        Vector samples(nq*batch*6),result(size_t(dofs())*batch);
#pragma omp parallel for schedule(static)
        for(size_t q=0;q<nq;++q){const auto& p=points_[q];
            for(size_t d=0;d<directions_.size();++d){
                const Complex factor=std::conj(prefactor_*p.weight*std::exp(Complex(0,-1)*wave_*directions_[d].dot(p.position)));
                for(int pol=0;pol<2;++pol){const int row=2*d+pol;const auto& e=electric_[row];const auto& m=magnetic_[row];
                    for(int h=0;h<batch;++h){auto* a=samples.data()+(q*batch+h)*6;const auto f=factor*input[size_t(h)*channels()+row];
                        a[0]+=e.x*f;a[1]+=e.y*f;a[2]+=e.z*f;a[3]+=m.x*f;a[4]+=m.y*f;a[5]+=m.z*f;}}}}
#pragma omp parallel for schedule(static)
        for(int j=0;j<current_;++j)for(int h=0;h<batch;++h){Complex ej=0,em=0;
            for(int k=offsets_[j];k<offsets_[j+1];++k){const auto& at=incidence_[k];
                const auto& v=points_[at.point].basis.values[at.local];const auto* a=samples.data()+(size_t(at.point)*batch+h)*6;
                ej+=v.x*a[0]+v.y*a[1]+v.z*a[2];em+=v.x*a[3]+v.y*a[4]+v.z*a[5];}
            result[size_t(h)*dofs()+j]=ej;result[size_t(h)*dofs()+current_+j]=em;}
        output.swap(result);
    }
};
