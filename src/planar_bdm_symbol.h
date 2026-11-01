#pragma once
#include "planar_muller_symbol.h"
#include "prism_circulant_layout.h"
#include "quadrature.h"
#include <algorithm>
#include <limits>
#include <map>
#include <vector>

namespace bem_osrc {
// A genuine BDM1 Bloch-cell mass matrix plus the planar Muller diagonal
// symbol in the same six electric/six magnetic channels. Retained reciprocal
// aliases approximate only the compact contrast term. The mass is exact for
// the affine basis and is never replaced by a rank-two pointwise projection.
class PlanarBdmSymbol {
public:
    struct Channel {int channel,dx,dy,sign;};
    using Mass=std::array<C,36>;
    using Symbol=std::array<C,144>;
    using Transform=std::array<C,12>; // six tangential vectors, component-major within a channel
private:
    struct FourierPoint {double x,y;std::array<double,12> weighted;};
    struct CellTerm {int channel,dx,dy;double x,y;};
    struct CellPoint {double weight;std::vector<CellTerm> terms;};
    const MullerP2Mesh& mesh_;
    const PrismCirculantPatch& patch_;
    Vec3 origin_,first_,second_,reciprocal1_,reciprocal2_;
    int ci_,cj_;
    std::vector<std::array<int,2>> endpoints_;
    std::vector<FourierPoint> fourier_;
    std::vector<CellPoint> cell_;
    std::array<int,6> source_;
    std::array<int,6> source_sign_;
    double area_;
    std::array<double,2> coordinates(const Vec3& p)const{
        const Vec3 d=p-patch_.origin;return {{d.dot(reciprocal1_),d.dot(reciprocal2_)}};
    }
    static std::vector<std::array<double,2>> gauss(int n){
        std::vector<std::array<double,2>> result(n);const double pi=std::acos(-1.);
        for(int i=0;i<(n+1)/2;++i){double z=std::cos(pi*(i+.75)/(n+.5)),derivative=0;
            for(int iteration=0;iteration<100;++iteration){double a=1,b=0;
                for(int j=1;j<=n;++j){const double next=((2*j-1)*z*a-(j-1)*b)/j;b=a;a=next;}
                derivative=n*(z*a-b)/(z*z-1);const double next=z-a/derivative;
                if(std::abs(next-z)<2e-15){z=next;break;}z=next;
            }
            const double weight=1/((1-z*z)*derivative*derivative);
            result[i]={{(1-z)*.5,weight}};result[n-1-i]={{(1+z)*.5,weight}};
        }return result;
    }
public:
    double area()const{return area_;}
    const std::array<int,6>& source_dofs()const{return source_;}
    const std::array<int,6>& source_signs()const{return source_sign_;}
    const Vec3& first_axis()const{return first_;}
    const Vec3& second_axis()const{return second_;}
    const Vec3& cell_origin()const{return origin_;}
    std::array<double,2> wavevector(double theta1,double theta2)const{
        const Vec3 xi=reciprocal1_*theta1+reciprocal2_*theta2;return {{xi.dot(first_),xi.dot(second_)}};
    }
    Channel channel(int dof)const{
        const auto& edge=endpoints_.at(dof/2);
        const auto p=coordinates(mesh_.nodes[edge[0]]),q=coordinates(mesh_.nodes[edge[1]]);
        const int a[2]={int(std::llround(p[0])),int(std::llround(p[1]))},b[2]={int(std::llround(q[0])),int(std::llround(q[1]))};
        for(int j=0;j<2;++j)if(std::abs(p[j]-a[j])>1e-7||std::abs(q[j]-b[j])>1e-7)throw std::runtime_error("BDM symbol off-lattice edge");
        const int dx=b[0]-a[0],dy=b[1]-a[1];int kind=-1;
        if(std::abs(dx)==1&&dy==0)kind=0;
        else if(dx==0&&std::abs(dy)==1)kind=1;
        else if(std::abs(dx)==1&&dy==(patch_.kind?-dx:dx))kind=2;
        if(kind<0)throw std::runtime_error("BDM symbol edge kind");
        const bool increasing=a[0]<b[0]||(a[0]==b[0]&&a[1]<b[1]);
        return {2*kind+dof%2,std::min(a[0],b[0])-ci_,std::min(a[1],b[1])-cj_,dof%2?1:increasing?1:-1};
    }
    PlanarBdmSymbol(const MullerP2Mesh& mesh,const PrismCirculantPatch& patch,int gauss_order=16)
        :mesh_(mesh),patch_(patch),endpoints_(mesh.topology_edge_count){
        if(mesh.basis_kind!=MullerBasisKind::HDivBdm1||patch.parity!=1||gauss_order<4||gauss_order>40)
            throw std::invalid_argument("canonical planar BDM stencil");
        const double a=patch.u.norm2(),b=patch.u.dot(patch.v),c=patch.v.norm2(),det=a*c-b*b;
        if(det<=0||patch.u.cross(patch.v).dot(patch.normal)<=0)throw std::invalid_argument("planar cell orientation");
        area_=std::sqrt(det);reciprocal1_=(patch.u*c-patch.v*b)*(1/det);reciprocal2_=(patch.v*a-patch.u*b)*(1/det);
        first_=patch.u*(1/std::sqrt(a));second_=patch.normal.cross(first_);
        double best=std::numeric_limits<double>::infinity();ci_=cj_=-1;
        for(int i=0;i<patch.nx;++i)for(int j=0;j<patch.ny;++j){bool valid=true;
            for(int ch=0;ch<6;++ch)valid=valid&&patch.dof[patch.slot(ch,i,j)]>=0;
            const double distance=std::pow(i-.5*(patch.nx-1),2)+std::pow(j-.5*(patch.ny-1),2);
            if(valid&&distance<best){best=distance;ci_=i;cj_=j;}}
        if(ci_<0)throw std::runtime_error("no complete BDM symbol source cell");
        origin_=patch.origin+patch.u*ci_+patch.v*cj_;
        for(int c0=0;c0<6;++c0){source_[c0]=patch.dof[patch.slot(c0,ci_,cj_)];source_sign_[c0]=patch.solution_sign[patch.slot(c0,ci_,cj_)];}
        for(const auto& e:mesh.elements)for(int j=0;j<3;++j){int a0=e.topology_vertices[j],b0=e.topology_vertices[(j+1)%3];
            if(a0>b0)std::swap(a0,b0);
            endpoints_[e.topology_edges[j]]={{a0,b0}};}
        const auto line=gauss(gauss_order);const auto mass_rule=tri_quadrature(13);int cells=0;
        for(int t=0;t<int(mesh.elements.size());++t){
            const auto center=evaluate_muller_frame(mesh,t,1./3,1./3);
            if((center.normal-patch.normal).norm()>1e-8||std::abs((center.position-origin_).dot(patch.normal))>1e-8)continue;
            const auto basis=evaluate_muller_basis(mesh,t,center);bool support=false;
            for(int c0=0;c0<6;++c0)for(int j=0;j<basis.count;++j)support=support||basis.dofs[j]==source_[c0];
            if(support)for(const auto& u:line)for(const auto& v:line){
                const auto frame=evaluate_muller_frame(mesh,t,u[0],(1-u[0])*v[0]);const auto values=evaluate_muller_basis(mesh,t,frame);
                const Vec3 p=frame.position-origin_;FourierPoint point{};point.x=p.dot(first_);point.y=p.dot(second_);
                const double weight=u[1]*v[1]*(1-u[0])*frame.jacobian;
                for(int c0=0;c0<6;++c0)for(int j=0;j<values.count;++j)if(values.dofs[j]==source_[c0]){
                    point.weighted[2*c0]=weight*source_sign_[c0]*values.values[j].dot(first_);
                    point.weighted[2*c0+1]=weight*source_sign_[c0]*values.values[j].dot(second_);}
                fourier_.push_back(point);
            }
            const auto coordinate=coordinates(center.position);
            if(int(std::floor(coordinate[0]))!=ci_||int(std::floor(coordinate[1]))!=cj_)continue;
            ++cells;
            for(int q=0;q<mass_rule.npts;++q){const auto frame=evaluate_muller_frame(mesh,t,mass_rule.pts[q][0],mass_rule.pts[q][1]);
                const auto values=evaluate_muller_basis(mesh,t,frame);CellPoint point;point.weight=.5*mass_rule.wts[q]*frame.jacobian;
                for(int j=0;j<values.count;++j){const auto c0=channel(values.dofs[j]);point.terms.push_back({c0.channel,c0.dx,c0.dy,
                    c0.sign*values.values[j].dot(first_),c0.sign*values.values[j].dot(second_)});}
                cell_.push_back(std::move(point));}
        }
        if(cells!=2||fourier_.empty())throw std::runtime_error("incomplete BDM Bloch cell");
    }
    Transform fourier(double x,double y)const{
        Transform f{};for(const auto& p:fourier_){const C phase=std::exp(C(0,-x*p.x-y*p.y));
            for(int c=0;c<12;++c)f[c]+=phase*p.weighted[c];}return f;
    }
    Mass mass(double theta1,double theta2)const{
        Mass result{};for(const auto& p:cell_){Transform value{};
            for(const auto& t:p.terms){const C phase=std::exp(C(0,theta1*t.dx+theta2*t.dy));value[2*t.channel]+=phase*t.x;value[2*t.channel+1]+=phase*t.y;}
            for(int i=0;i<6;++i)for(int j=0;j<6;++j)result[i*6+j]+=p.weight*(std::conj(value[2*i])*value[2*j]+std::conj(value[2*i+1])*value[2*j+1]);
        }return result;
    }
    Symbol symbol(double k0,C index,double damping,double theta1,double theta2,int aliases=1)const{
        if(!finite(index)||index.real()<=0||index.imag()<0||damping<=0||!std::isfinite(damping)||aliases<0||aliases>4)
            throw std::invalid_argument("planar BDM material/regularization/alias count");
        const double pi=std::acos(-1.);theta1=std::remainder(theta1,2*pi);theta2=std::remainder(theta2,2*pi);
        const C ke(k0,damping*k0),physical_interior=k0*index,ki=physical_interior+C(0,damping*std::abs(physical_interior));
        const C epsilon=.5*(1.+index*index);const auto m=mass(theta1,theta2);Symbol result{};
        for(int i=0;i<6;++i)for(int j=0;j<6;++j){result[i*12+j+6]=epsilon*m[i*6+j];result[(i+6)*12+j]=m[i*6+j];}
        // Cut off physical reciprocal coordinates symmetrically. At a Nyquist
        // boundary both end points are retained; arbitrary +pi/-pi choices
        // must not break a mirror or the periodicity of the discrete symbol.
        for(int a=-aliases-1;a<=aliases+1;++a)for(int b=-aliases-1;b<=aliases+1;++b){
            const double ta=theta1+2*pi*a,tb=theta2+2*pi*b,limit=(2*aliases+1)*pi+1e-12;
            if(std::abs(ta)>limit||std::abs(tb)>limit)continue;
            const auto xi=wavevector(ta,tb);
            const auto f=fourier(xi[0],xi[1]);const auto diagonal=diagonal_block(k0,ke,ki,xi[0],xi[1]);
            for(int i=0;i<6;++i)for(int j=0;j<6;++j){C value=0;
                for(int u=0;u<2;++u)for(int v=0;v<2;++v)value+=std::conj(f[2*i+u])*diagonal[2*u+v]*f[2*j+v]/area_;
                result[i*12+j]+=value;result[(i+6)*12+j+6]-=value;}
        }return result;
    }
};
}
