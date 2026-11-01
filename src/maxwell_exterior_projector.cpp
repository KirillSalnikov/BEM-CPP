#include "maxwell_exterior_projector.h"
#include "quadrature.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace {
using C=std::complex<double>;using V=std::vector<C>;using K=MaxwellExteriorProjector::Kernel;
const double inv4pi=1/(4*std::acos(-1.));
double length(const V& x){double r=0;for(auto z:x)r+=std::norm(z);return std::sqrt(r);}
C dot(const V& a,const V& b){C r=0;for(size_t i=0;i<a.size();++i)r+=std::conj(a[i])*b[i];return r;}
std::vector<std::pair<double,double>> gauss(int order){
    std::vector<std::pair<double,double>> result(order);
    for(int i=0;i<(order+1)/2;++i){double x=std::cos(std::acos(-1.)*(i+.75)/(order+.5)),d=0;
        for(int it=0;it<40;++it){double a=1,b=x;for(int j=2;j<=order;++j){double c=((2*j-1)*x*b-(j-1)*a)/j;a=b;b=c;}
            d=order*(x*b-a)/(x*x-1);double change=b/d;x-=change;if(std::abs(change)<1e-15)break;}
        const double w=1/((1-x*x)*d*d);result[i]={(1-x)*.5,w};result[order-1-i]={(1+x)*.5,w};}
    return result;
}
std::array<C,4> kernel(const Vec3& r,double k){const double d=r.norm();if(d<1e-14)return {};
    const C g=inv4pi*std::exp(C(0,k*d))/d,h=g*(C(0,k*d)-1.)/(d*d);
    return {{g,h*r.x,h*r.y,h*r.z}};
}
}

MaxwellExteriorProjector::Kernel MaxwellExteriorProjector::integrated_kernel(
        const Triangle& t,const Vec3& x,double k,int order){
    const Vec3 diff=x-t.vertex[0];const double xi=t.gxi.dot(diff),eta=t.geta.dot(diff),height=t.normal.dot(diff);
    const double at_x[]={1.,xi,eta};
    Vec3 apex=x-t.normal*height;
    if(xi<0||eta<0||xi+eta>1){double best=1e300;
        for(int e=0;e<3;++e){Vec3 a=t.vertex[e],d=t.vertex[(e+1)%3]-a;
            Vec3 candidate=a+d*std::max(0.,std::min(1.,d.dot(x-a)/d.dot(d)));
            double distance=(candidate-x).dot(candidate-x);if(distance<best){best=distance;apex=candidate;}}}
    Vec3 static_gradient;
    for(int e=0;e<3;++e){Vec3 a=t.vertex[e],b=t.vertex[(e+1)%3],edge=b-a;double l=edge.norm();
        double sum=(x-a).norm()+(x-b).norm();
        // Targets are interior triangle quadrature nodes, never source edges.
        if(sum-l<=1e-15*l)throw std::runtime_error("exterior projection target lies on a source edge");
        static_gradient=static_gradient-edge.cross(t.normal)*(std::log1p(2*l/(sum-l))/l);}
    if(std::abs(height)>1e-12*t.diameter){const Vec3 a=t.vertex[0]-x,b=t.vertex[1]-x,c=t.vertex[2]-x;
        const double omega=2*std::atan2(a.dot(b.cross(c)),a.norm()*b.norm()*c.norm()+a.dot(b)*c.norm()+b.dot(c)*a.norm()+c.dot(a)*b.norm());
        static_gradient=static_gradient+t.normal*omega;}
    static_gradient=static_gradient*inv4pi;
    Kernel out;
    const double sg[]={static_gradient.x,static_gradient.y,static_gradient.z};
    for(int l=0;l<3;++l)for(int a=0;a<3;++a)out.value[4*l+1+a]=at_x[l]*sg[a];
    static thread_local std::map<int,std::vector<std::pair<double,double>>> rules;
    auto found=rules.find(order);if(found==rules.end())found=rules.emplace(order,gauss(order)).first;
    const auto& rule=found->second;
    for(int edge=0;edge<3;++edge){const Vec3 a=t.vertex[edge]-apex,b=t.vertex[(edge+1)%3]-apex;
        const double jac=a.cross(b).norm();if(jac<1e-20*t.jacobian)continue;
        for(const auto& u:rule)for(const auto& v:rule){Vec3 y=apex+(a*(1-v.first)+b*v.first)*u.first,r=x-y;
            const double d=r.norm(),w=jac*u.first*u.second*v.second;
            const double phi[]={1.,t.gxi.dot(y-t.vertex[0]),t.geta.dot(y-t.vertex[0])};
            const C z(0,k*d),expz=std::exp(z),potential=inv4pi*expz/d;
            C remainder;
            if(std::abs(z)<1e-3){C term=z*z*.5;remainder=term;for(int j=3;j<14;++j){term*=z/double(j);remainder+=double(j-1)*term;}}
            else remainder=expz*(z-1.)+1.;
            const C smooth=inv4pi*remainder/(d*d*d);const double singular=-inv4pi/(d*d*d);
            const double rv[]={r.x,r.y,r.z};
            for(int l=0;l<3;++l){out.value[4*l]+=w*potential*phi[l];
                for(int a=0;a<3;++a)out.value[4*l+1+a]+=w*rv[a]*(smooth*phi[l]+singular*(phi[l]-at_x[l]));}
        }
    }
    return out;
}

MaxwellExteriorProjector::MaxwellExteriorProjector(const MullerP2Mesh& mesh,double k,int quadrature,int order,double radius)
    :mesh_(mesh),k_(k),n_(mesh.current_dofs()){
    if(mesh.basis_kind!=MullerBasisKind::HDivBdm1||k<=0||order<2||radius<1)throw std::invalid_argument("exterior projector arguments");
    const auto rule=tri_quadrature(quadrature);q_=rule.npts;double max_h=0;
    for(size_t e=0;e<mesh.elements.size();++e){const auto& el=mesh.elements[e];Triangle t;
        for(int j=0;j<3;++j)t.vertex[j]=mesh.nodes[el.topology_vertices[j]];
        Vec3 a=t.vertex[1]-t.vertex[0],b=t.vertex[2]-t.vertex[0];t.jacobian=a.cross(b).norm();t.normal=a.cross(b)*(1/t.jacobian);
        t.gxi=b.cross(t.normal)*(1/t.jacobian);t.geta=t.normal.cross(a)*(1/t.jacobian);t.center=(t.vertex[0]+t.vertex[1]+t.vertex[2])*(1./3);
        t.diameter=std::max({a.norm(),b.norm(),(a-b).norm()});max_h=std::max(max_h,t.diameter);
        const double xy[3][2]={{0,0},{1,0},{0,1}};MullerBasisSample basis[3];
        for(int j=0;j<3;++j){auto f=evaluate_muller_frame(mesh,e,xy[j][0],xy[j][1]);
            if((f.normal-t.normal).norm()>1e-10)throw std::runtime_error("exterior projector requires flat triangles");
            basis[j]=evaluate_muller_basis(mesh,e,f);}
        for(int j=0;j<6;++j){t.dof[j]=basis[0].dofs[j];t.basis[j]={{basis[0].values[j],basis[1].values[j]-basis[0].values[j],basis[2].values[j]-basis[0].values[j]}};
            t.divergence[j]=t.basis[j][1].dot(t.gxi)+t.basis[j][2].dot(t.geta);}
        triangles_.push_back(t);
        for(int q=0;q<q_;++q){const auto f=evaluate_muller_frame(mesh,e,rule.pts[q][0],rule.pts[q][1]);
            points_.push_back({f.position,f.normal,.5*rule.wts[q]*f.jacobian,int(e),rule.pts[q][0],rule.pts[q][1]});}
    }
    // Sparse SPD L2 mass matrix; solves do not involve a dense surface matrix.
    std::vector<std::vector<std::pair<int,double>>> rows(n_);
    for(size_t e=0;e<triangles_.size();++e)for(int q=0;q<q_;++q){const auto& p=points_[e*q_+q];const auto& t=triangles_[e];
        for(int i=0;i<6;++i)for(int j=0;j<6;++j){const Vec3 a=t.basis[i][0]+t.basis[i][1]*p.xi+t.basis[i][2]*p.eta;
            const Vec3 b=t.basis[j][0]+t.basis[j][1]*p.xi+t.basis[j][2]*p.eta;
            rows[t.dof[i]].push_back({t.dof[j],p.weight*a.dot(b)});}}
    mass_offsets_.push_back(0);inverse_diagonal_.resize(n_);
    for(int i=0;i<n_;++i){auto& row=rows[i];std::sort(row.begin(),row.end());
        for(size_t j=0;j<row.size();){const int col=row[j].first;double value=0;while(j<row.size()&&row[j].first==col)value+=row[j++].second;
            mass_columns_.push_back(col);mass_values_.push_back(value);if(col==i)inverse_diagonal_[i]=1/value;}
        mass_offsets_.push_back(mass_columns_.size());}
    using Cell=std::array<int,3>;std::map<Cell,std::vector<int>> cells;const double cell=(radius+1)*max_h;
    auto key=[&](const Vec3& p){return Cell{{int(std::floor(p.x/cell)),int(std::floor(p.y/cell)),int(std::floor(p.z/cell))}};};
    for(size_t e=0;e<triangles_.size();++e)cells[key(triangles_[e].center)].push_back(e);
    std::vector<std::vector<int>> candidates(triangles_.size());
    for(size_t e=0;e<triangles_.size();++e){const auto at=key(triangles_[e].center);
        for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)for(int c=-1;c<=1;++c){auto it=cells.find(Cell{{at[0]+a,at[1]+b,at[2]+c}});if(it==cells.end())continue;
            for(int s:it->second)if((triangles_[s].center-triangles_[e].center).norm()<radius*triangles_[s].diameter+triangles_[e].diameter)candidates[e].push_back(s);}}
    std::vector<std::vector<Correction>> blocks(points_.size());
#pragma omp parallel for schedule(dynamic,16)
    for(size_t p=0;p<points_.size();++p){const auto& target=points_[p];
        for(int e:candidates[target.element]){const auto& t=triangles_[e];if((target.position-t.center).norm()>radius*t.diameter)continue;
            Correction correction;correction.element=e;correction.kernel=integrated_kernel(t,target.position,k,order);
            for(int q=0;q<q_;++q){const auto& s=points_[e*q_+q];const auto sample=kernel(target.position-s.position,k);const double phi[]={1.,s.xi,s.eta};
                for(int l=0;l<3;++l)for(int a=0;a<4;++a)correction.kernel.value[4*l+a]-=s.weight*phi[l]*sample[a];}
            blocks[p].push_back(correction);}}
    offsets_.push_back(0);for(auto& block:blocks){corrections_.insert(corrections_.end(),block.begin(),block.end());offsets_.push_back(corrections_.size());}
}

void MaxwellExteriorProjector::mass(const V& x,V& y)const{y.assign(n_,0);
#pragma omp parallel for schedule(static)
    for(int i=0;i<n_;++i)for(int j=mass_offsets_[i];j<mass_offsets_[i+1];++j)y[i]+=mass_values_[j]*x[mass_columns_[j]];
}
V MaxwellExteriorProjector::solve_mass(const V& rhs)const{
    V x(n_),r=rhs,z(n_),p(n_),ap;const double bn=length(rhs);if(bn==0)return x;
    for(int i=0;i<n_;++i)p[i]=z[i]=r[i]*inverse_diagonal_[i];
    C rz=dot(r,z);
    for(int it=0;it<200;++it){mass(p,ap);const C alpha=rz/dot(p,ap);
        for(int i=0;i<n_;++i){x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
        if(length(r)<=bn*1e-11){mass(x,ap);for(int i=0;i<n_;++i)ap[i]-=rhs[i];
            const double error=length(ap)/bn;last_mass_residual_=std::max(last_mass_residual_,error);
            if(error>2e-11)throw std::runtime_error("exterior trace mass solve failed");
            return x;}
        for(int i=0;i<n_;++i)z[i]=r[i]*inverse_diagonal_[i];
        const C next=dot(r,z),beta=next/rz;rz=next;
        for(int i=0;i<n_;++i)p[i]=z[i]+beta*p[i];}
    throw std::runtime_error("exterior trace mass solve did not converge");
}
V MaxwellExteriorProjector::incident(const V& rhs)const{
    if(int(rhs.size())!=2*n_)throw std::invalid_argument("incident RHS dimensions");
    V e(rhs.begin(),rhs.begin()+n_),h(rhs.begin()+n_,rhs.end());e=solve_mass(e);h=solve_mass(h);h.insert(h.end(),e.begin(),e.end());return h;
}
void MaxwellExteriorProjector::direct_pair(const C* a,const C* b,C* pa,C* ga,C* pb,C* gb)const{
#pragma omp parallel for schedule(static)
    for(size_t t=0;t<points_.size();++t){pa[t]=pb[t]=0;for(int j=0;j<3;++j)ga[3*t+j]=gb[3*t+j]=0;
        for(size_t s=0;s<points_.size();++s){const auto g=kernel(points_[t].position-points_[s].position,k_);pa[t]+=g[0]*a[s];pb[t]+=g[0]*b[s];
            for(int j=0;j<3;++j){ga[3*t+j]+=g[1+j]*a[s];gb[3*t+j]+=g[1+j]*b[s];}}}
}
void MaxwellExteriorProjector::apply(const C* x,C* out,const ScalarPair& action){
    const size_t np=points_.size();std::array<V,8> charge,potential,gradient;
    for(int a=0;a<8;++a){charge[a].resize(np);potential[a].resize(np);gradient[a].resize(3*np);}
    using Affine=std::array<std::array<C,3>,3>;std::vector<std::array<Affine,2>> values(triangles_.size());
    std::vector<std::array<C,2>> divergence(triangles_.size());
    for(size_t e=0;e<triangles_.size();++e){const auto& t=triangles_[e];
        for(int pol=0;pol<2;++pol)for(int i=0;i<6;++i){const C c=x[pol*n_+t.dof[i]];divergence[e][pol]+=c*t.divergence[i];
            for(int l=0;l<3;++l){const auto& b=t.basis[i][l];values[e][pol][l][0]+=c*b.x;values[e][pol][l][1]+=c*b.y;values[e][pol][l][2]+=c*b.z;}}}
#pragma omp parallel for schedule(static)
    for(size_t p=0;p<np;++p){const auto& pt=points_[p];for(int pol=0;pol<2;++pol){const auto& v=values[pt.element][pol];
        for(int a=0;a<3;++a)charge[4*pol+a][p]=pt.weight*(v[0][a]+pt.xi*v[1][a]+pt.eta*v[2][a]);
        charge[4*pol+3][p]=pt.weight*divergence[pt.element][pol];}}
    for(int a=0;a<8;a+=2)action(charge[a].data(),charge[a+1].data(),potential[a].data(),gradient[a].data(),potential[a+1].data(),gradient[a+1].data());
#pragma omp parallel for schedule(static)
    for(size_t p=0;p<np;++p)for(size_t j=offsets_[p];j<offsets_[p+1];++j){const auto& correction=corrections_[j];const auto& g=correction.kernel.value;const int e=correction.element;
        for(int pol=0;pol<2;++pol){for(int a=0;a<3;++a)for(int l=0;l<3;++l){const C c=values[e][pol][l][a];potential[4*pol+a][p]+=g[4*l]*c;
                for(int d=0;d<3;++d)gradient[4*pol+a][3*p+d]+=g[4*l+1+d]*c;}
            for(int d=0;d<3;++d)gradient[4*pol+3][3*p+d]+=g[1+d]*divergence[e][pol];}}
    V weak_j(n_),weak_m(n_);const C ik(0,k_),overk(0,1/k_);
    // Element-local accumulation avoids atomic complex additions.
    std::vector<std::array<C,12>> weak(triangles_.size());
#pragma omp parallel for schedule(static)
    for(size_t e=0;e<triangles_.size();++e){const auto& t=triangles_[e];for(int q=0;q<q_;++q){size_t p=e*q_+q;const auto& pt=points_[p];
        std::array<C,3> curl[2];for(int pol=0;pol<2;++pol){int a=4*pol;curl[pol]={{gradient[a+2][3*p+1]-gradient[a+1][3*p+2],gradient[a][3*p+2]-gradient[a+2][3*p],gradient[a+1][3*p]-gradient[a][3*p+1]}};}
        C E[3],H[3];for(int a=0;a<3;++a){E[a]=ik*potential[a][p]+overk*gradient[3][3*p+a]-curl[1][a];H[a]=curl[0][a]+ik*potential[4+a][p]+overk*gradient[7][3*p+a];}
        for(int i=0;i<6;++i){const Vec3 b=t.basis[i][0]+t.basis[i][1]*pt.xi+t.basis[i][2]*pt.eta,r=b.cross(t.normal);
            weak[e][i]+=pt.weight*(r.x*H[0]+r.y*H[1]+r.z*H[2]);weak[e][6+i]-=pt.weight*(r.x*E[0]+r.y*E[1]+r.z*E[2]);}}}
    for(size_t e=0;e<triangles_.size();++e)for(int i=0;i<6;++i){int d=triangles_[e].dof[i];weak_j[d]+=weak[e][i];weak_m[d]+=weak[e][6+i];}
    last_mass_residual_=0;weak_j=solve_mass(weak_j);weak_m=solve_mass(weak_m);
    for(int i=0;i<n_;++i){out[i]=.5*x[i]+weak_j[i];out[n_+i]=.5*x[n_+i]+weak_m[i];}
}
