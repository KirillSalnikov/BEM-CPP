#include "dirac_transmission.h"
#include "quadrature.h"
#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <limits>

namespace bem_dirac {
namespace {
using CV = std::array<C,3>;
// Give geometrically equal threshold pairs the same quadrature under symmetry.
constexpr double NEAR_TIE=128.*std::numeric_limits<double>::epsilon();
bool within_near(double distance,double radius,double diameter){return distance<=radius*diameter*(1.+NEAR_TIE);}
std::array<double,10> polynomial(double xi,double eta){
    const double s=xi+eta,d=xi-eta;
    return {{1.,std::sqrt(6.)*d,std::sqrt(2.)*(2.-3.*s),
        std::sqrt(3.)*(10.*s*s-12.*s+3.),3.*d*(5.*s-4.),std::sqrt(15.)*(xi*xi-4.*xi*eta+eta*eta),
        2.*(35.*s*s*s-60.*s*s+30.*s-4.),2.*std::sqrt(3.)*d*(21.*s*s-30.*s+10.),
        std::sqrt(5.)*(3.*d*d-s*s)*(7.*s-6.),std::sqrt(7.)*d*(5.*d*d-3.*s*s)}};
}
CV cross(const CV&a,const CV&b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
C dot(const CV&a,const CV&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
C dot(const CV&a,const Vec3&b){return a[0]*b.x+a[1]*b.y+a[2]*b.z;}
std::vector<std::pair<double,double>> gauss(int order){
    std::vector<std::pair<double,double>> r(order);
    for(int i=0;i<(order+1)/2;++i){double x=std::cos(M_PI*(i+.75)/(order+.5)),d=0;
        for(int it=0;it<50;++it){double a=1,b=x;for(int j=2;j<=order;++j){double c=((2*j-1)*x*b-(j-1)*a)/j;a=b;b=c;}
            d=order*(x*b-a)/(x*x-1);double dx=b/d;x-=dx;if(std::abs(dx)<1e-15)break;}
        double w=1/((1-x*x)*d*d);r[i]={(1-x)*.5,w};r[order-1-i]={(1+x)*.5,w};}
    return r;
}
std::vector<std::array<double,3>> target_points(int order,int depth){
    auto rule=tri_quadrature(order);
    if(order==1){rule.npts=1;rule.pts[0][0]=rule.pts[0][1]=1./3;rule.wts[0]=1.;}
    using Triangle=std::array<Vec3,3>;
    std::vector<Triangle> triangles{Triangle{{Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)}}};
    for(int level=0;level<depth;++level){std::vector<Triangle> next;
        for(const auto&t:triangles){Vec3 ab=(t[0]+t[1])*.5,bc=(t[1]+t[2])*.5,ca=(t[2]+t[0])*.5;
            next.push_back(Triangle{{t[0],ab,ca}});next.push_back(Triangle{{ab,t[1],bc}});next.push_back(Triangle{{ca,bc,t[2]}});next.push_back(Triangle{{ab,bc,ca}});}
        triangles.swap(next);}
    std::vector<std::array<double,3>> result;
    for(const auto&t:triangles)for(int q=0;q<rule.npts;++q){Vec3 v=t[0]+(t[1]-t[0])*rule.pts[q][0]+(t[2]-t[0])*rule.pts[q][1];result.push_back({{v.x,v.y,rule.wts[q]/triangles.size()}});}
    return result;
}
Eight charges(const Panel&p,const C*h){
    Eight q{};const double n[]={p.normal.x,p.normal.y,p.normal.z},t[]={p.tangent.x,p.tangent.y,p.tangent.z},u[]={p.second.x,p.second.y,p.second.z};
    for(int j=0;j<3;++j){q[j]=u[j]*h[2]-t[j]*h[3]-n[j]*h[0];q[4+j]=u[j]*h[6]-t[j]*h[7]+n[j]*h[4];}
    q[3]=h[1];q[7]=h[5];return q;
}
Eight action(const Kernel&k,const C*q){
    CV h{{q[0],q[1],q[2]}},e{{q[4],q[5],q[6]}};
    auto bh=cross(k.d,h),be=cross(k.d,e);Eight v{};
    v[0]=dot(k.d,h)+k.s*q[7];v[4]=-dot(k.d,e)+k.s*q[3];
    for(int j=0;j<3;++j){v[1+j]=bh[j]-k.d[j]*q[3]+k.s*e[j];v[5+j]=be[j]-k.d[j]*q[7]-k.s*h[j];}
    return v;
}
void localize(const Panel&p,const Eight&a,C*out){
    CV h{{a[1],a[2],a[3]}},e{{a[5],a[6],a[7]}};
    out[0]=a[0];out[1]=dot(h,p.normal);out[2]=dot(h,p.tangent);out[3]=dot(h,p.second);
    out[4]=a[4];out[5]=dot(e,p.normal);out[6]=dot(e,p.tangent);out[7]=dot(e,p.second);
}
}
Scaling scaling(C m){
    if(!std::isfinite(m.real())||!std::isfinite(m.imag())||m.real()<=0||m.imag()<0)
        throw std::invalid_argument("Dirac screen requires passive index with positive real part");
    const C c=1./m,eps=m*m,r=std::sqrt(c),s=std::sqrt(c+std::abs(c));
    const C beta=1.-C(0,.2/M_PI)*std::arg(c);
    Scaling a;
    a.p={{beta/(c+beta),1./s,1./(2.*r),1./(2.*r),std::abs(c)/(c+std::abs(c)),eps/(eps+1.),1.,1.}};
    a.n={{c/(c+beta),c/s,r/2.,r/2.,c/(c+std::abs(c)),1./(eps+1.),1.,1.}};
    a.pp={{1.,1./s,1./r,1./r,1.,1.,1./(c+1.),1./(c+1.)}};
    a.np={{1.,std::abs(c)/s,r,r,1.,1.,c/(c+1.),c/(c+1.)}};
    for(int j=0;j<8;++j)if(std::abs(a.p[j]*a.np[j]+a.n[j]*a.pp[j]-1.)>1e-12)
        throw std::runtime_error("Dirac scaling identity");
    return a;
}
std::array<C,8> kernel_action(const Kernel&k,const Panel&p,const C*h){auto q=charges(p,h);return action(k,q.data());}

std::array<Kernel,10> Transmission::integrate(const Panel&t,const Vec3&x,C k,bool near)const{
    std::array<Kernel,10> out{};std::array<C,10> potential{};std::array<CV,10> gx{};
    if(!near){const auto rule=tri_quadrature(far_order_);
        for(int j=0;j<rule.npts;++j){const Vec3 y=t.vertex[0]+(t.vertex[1]-t.vertex[0])*rule.pts[j][0]+(t.vertex[2]-t.vertex[0])*rule.pts[j][1];
            const Vec3 r=x-y;const double d=r.norm();const C g=t.area*rule.wts[j]*INV4PI*std::exp(C(0,1)*k*d)/d;
            const auto phi=polynomial(rule.pts[j][0],rule.pts[j][1]);const C z=g*(C(0,1)*k*d-1.)/(d*d);
            for(int b=0;b<basis_count_;++b){potential[b]+=g*phi[b];gx[b][0]+=r.x*z*phi[b];gx[b][1]+=r.y*z*phi[b];gx[b][2]+=r.z*z*phi[b];}}
    }else{
        const Vec3 a=t.vertex[1]-t.vertex[0],b=t.vertex[2]-t.vertex[0];
        const Vec3 gxi=b.cross(t.normal)*(1./(2*t.area)),geta=t.normal.cross(a)*(1./(2*t.area));
        const double xi=gxi.dot(x-t.vertex[0]),eta=geta.dot(x-t.vertex[0]),height=t.normal.dot(x-t.vertex[0]);
        Vec3 apex=x-t.normal*height;
        if(xi<0||eta<0||xi+eta>1){double best=1e300;
            for(int j=0;j<3;++j){const Vec3 v=t.vertex[j],e=t.vertex[(j+1)%3]-v;
                const Vec3 c=v+e*std::max(0.,std::min(1.,e.dot(x-v)/e.norm2()));
                if((c-x).norm2()<best){best=(c-x).norm2();apex=c;}}}
        Vec3 gradient;
        for(int j=0;j<3;++j){const Vec3 v=t.vertex[j],w=t.vertex[(j+1)%3],edge=w-v;const double l=edge.norm(),sum=(x-v).norm()+(x-w).norm();
            if(sum-l<=1e-15*l)throw std::runtime_error("Dirac integration target on source edge");
            gradient=gradient-edge.cross(t.normal)*(std::log1p(2*l/(sum-l))/l);}
        if(std::abs(height)>1e-12*t.diameter){const Vec3 a=t.vertex[0]-x,b=t.vertex[1]-x,c=t.vertex[2]-x;
            const double omega=2*std::atan2(a.dot(b.cross(c)),a.norm()*b.norm()*c.norm()+a.dot(b)*c.norm()+b.dot(c)*a.norm()+c.dot(a)*b.norm());
            gradient=gradient+t.normal*omega;}
        const auto at_x=polynomial(xi,eta);
        for(int b=0;b<basis_count_;++b)gx[b]={{gradient.x*INV4PI*at_x[b],gradient.y*INV4PI*at_x[b],gradient.z*INV4PI*at_x[b]}};
        const auto& rule=near_gauss_;
        for(int edge=0;edge<3;++edge){const Vec3 a=t.vertex[edge]-apex,b=t.vertex[(edge+1)%3]-apex;const double jac=a.cross(b).norm();
            if(jac<1e-20*t.area)continue;
            for(const auto&u:rule)for(const auto&v:rule){const Vec3 y=apex+(a*(1-v.first)+b*v.first)*u.first,r=x-y;const double d=r.norm();
                const double w=INV4PI*jac*u.first*u.second*v.second;const C z=C(0,1)*k*d,ez=std::exp(z);
                const auto phi=polynomial(gxi.dot(y-t.vertex[0]),geta.dot(y-t.vertex[0]));C rem;
                if(std::abs(z)<1e-3){C term=z*z*.5;rem=term;for(int j=3;j<14;++j){term*=z/double(j);rem+=double(j-1)*term;}}
                else rem=ez*(z-1.)+1.;
                for(int b=0;b<basis_count_;++b){potential[b]+=w*ez/d*phi[b];
                    const C g=w*(rem*phi[b]-(phi[b]-at_x[b]))/(d*d*d);
                    gx[b][0]+=r.x*g;gx[b][1]+=r.y*g;gx[b][2]+=r.z*g;}}
        }
    }
    for(int b=0;b<basis_count_;++b){out[b].s=2.*C(0,1)*k*potential[b];for(int j=0;j<3;++j)out[b].d[j]=-2.*gx[b][j];}return out;
}

Transmission::Transmission(const Mesh&mesh,double k,C m,int order,int far,double radius,size_t max_bytes,int target_order,int degree,bool correction_only,int near_subdiv,bool defer_dense)
    :wave_{C(k),C(k)*m},index_(m),scale_(scaling(m)),near_order_(order),far_order_(far),target_order_(target_order),basis_count_(degree==0?1:degree==1?3:degree==2?6:10),near_radius_(radius),correction_only_(correction_only),near_subdiv_(near_subdiv){
    if(defer_dense&&correction_only)throw std::invalid_argument("deferred cache is dense only");
    if(!(k>0)||!std::isfinite(k)||order<4||order>48||(far!=7&&far!=13)||radius<1.)throw std::invalid_argument("Dirac arguments");
    near_gauss_=gauss(order);
    if(target_order!=1&&target_order!=7&&target_order!=13)throw std::invalid_argument("Dirac target rule");
    if((degree<0||degree>3)||(degree&&target_order==1))throw std::invalid_argument("Dirac P1/P2/P3 requires Galerkin quadrature");
    if(mesh.tris.size()%3)throw std::invalid_argument("Dirac triangle index count");
    for(const auto&v:mesh.verts)if(!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z))throw std::invalid_argument("Dirac nonfinite vertex");
    for(int v:mesh.tris)if(v<0||size_t(v)>=mesh.verts.size())throw std::invalid_argument("Dirac triangle vertex index");
    const size_t count=mesh.nt();
    const size_t coefficient_count=count*basis_count_;
    if(!count||(!correction_only_&&!defer_dense&&coefficient_count>std::sqrt(double(max_bytes)/(2*sizeof(Kernel)))))throw std::runtime_error("Dirac pair-cache memory budget exceeded");
    if(correction_only_&&(target_order_!=13||far_order_!=13))throw std::invalid_argument("Dirac FMM correction requires matching 13-point rules");
    if(near_subdiv<0||near_subdiv>2||(near_subdiv&&target_order!=13))throw std::invalid_argument("Dirac target refinement requires target13 and depth 0..2");
    std::map<std::pair<int,int>,std::pair<int,int>> edges;double volume=0;
    for(int i=0;i<mesh.nt();++i){Panel p;mesh.tri_verts(i,p.vertex[0],p.vertex[1],p.vertex[2]);
        const Vec3 a=p.vertex[1]-p.vertex[0],b=p.vertex[2]-p.vertex[0],n=a.cross(b);p.area=n.norm()*.5;
        if(!(p.area>1e-18))throw std::invalid_argument("degenerate Dirac panel");
        p.normal=n*(1./(2*p.area));p.tangent=a.normalized();p.second=p.normal.cross(p.tangent);
        p.center=(p.vertex[0]+p.vertex[1]+p.vertex[2])*(1./3);p.diameter=std::max({a.norm(),b.norm(),(a-b).norm()});panels_.push_back(p);
        volume+=p.vertex[0].dot(p.vertex[1].cross(p.vertex[2]))/6.;
        for(int j=0;j<3;++j){int u=mesh.tris[3*i+j],v=mesh.tris[3*i+(j+1)%3];auto& e=edges[{std::min(u,v),std::max(u,v)}];++e.first;e.second+=u<v?1:-1;}}
    for(auto e:edges)if(e.second.first!=2||e.second.second!=0)throw std::invalid_argument("Dirac requires a closed consistently wound surface");
    if(!(volume>0))throw std::invalid_argument("Dirac requires outward normal winding");
    if(defer_dense)return;
    if(correction_only_){
        // A symmetric geometric superset of every pair for which either
        // target quadrature may enter the exact source-panel near rule.
        // Spatial bins avoid constructing an Npanels squared distance table.
        double cell=0.;for(const auto&p:panels_)cell=std::max(cell,(near_radius_+1.)*p.diameter*(1.+NEAR_TIE));
        using Key=std::tuple<int,int,int>;std::map<Key,std::vector<int>> bins;
        auto key=[&](const Vec3&v){return Key(int(std::floor(v.x/cell)),int(std::floor(v.y/cell)),int(std::floor(v.z/cell)));};
        for(size_t j=0;j<count;++j)bins[key(panels_[j].center)].push_back(int(j));
        near_rows_.push_back(0);
        for(size_t i=0;i<count;++i){const auto index=key(panels_[i].center);std::vector<int> neighbors;
            for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z){
                auto it=bins.find(Key(std::get<0>(index)+x,std::get<1>(index)+y,std::get<2>(index)+z));if(it==bins.end())continue;
                for(int j:it->second)if(within_near((panels_[i].center-panels_[j].center).norm(),near_radius_+1.,std::max(panels_[i].diameter,panels_[j].diameter)))neighbors.push_back(j);}
            std::sort(neighbors.begin(),neighbors.end());near_columns_.insert(near_columns_.end(),neighbors.begin(),neighbors.end());near_rows_.push_back(near_columns_.size());}
        const size_t entries=near_columns_.size()*basis_count_*basis_count_;
        if(entries>max_bytes/(2*sizeof(Kernel)))throw std::runtime_error("Dirac near-correction memory budget exceeded");
        for(auto&v:kernels_)v.resize(entries);
        const auto rule=tri_quadrature(13);
        // Enrich only the exact near integral. The subtracted point rule
        // remains 13x13, exactly matching the FMM charge/target discretization.
        const auto targets=target_points(13,near_subdiv);
#pragma omp parallel for schedule(dynamic,1)
        for(size_t i=0;i<count;++i)for(size_t pair=near_rows_[i];pair<near_rows_[i+1];++pair){
            const size_t j=near_columns_[pair];const auto&t=panels_[i];const auto&s=panels_[j];
            for(int medium=0;medium<2;++medium){std::array<Kernel,100> total{};
                for(const auto&q:targets){const Vec3 x=t.vertex[0]+(t.vertex[1]-t.vertex[0])*q[0]+(t.vertex[2]-t.vertex[0])*q[1];
                    auto exact=integrate(s,x,wave_[medium],within_near((x-s.center).norm(),near_radius_,s.diameter));
                    const auto phi=polynomial(q[0],q[1]);
                    for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){auto&v=total[10*a+b];const double w=q[2]*phi[a];v.s+=w*exact[b].s;for(int c=0;c<3;++c)v.d[c]+=w*exact[b].d[c];}}
                for(int q=0;q<13;++q){const Vec3 x=t.vertex[0]+(t.vertex[1]-t.vertex[0])*rule.pts[q][0]+(t.vertex[2]-t.vertex[0])*rule.pts[q][1];
                    std::array<Kernel,10> exact{};
                    // Remove the point-quadrature interaction which FMM adds.
                    for(int u=0;u<13;++u){const Vec3 y=s.vertex[0]+(s.vertex[1]-s.vertex[0])*rule.pts[u][0]+(s.vertex[2]-s.vertex[0])*rule.pts[u][1],r=x-y;
                        const double d2=r.norm2();if(d2<1e-28)continue;const double d=std::sqrt(d2);
                        const C g=s.area*rule.wts[u]*INV4PI*std::exp(C(0,1)*wave_[medium]*d)/d;
                        const C pot=2.*C(0,1)*wave_[medium]*g,grad=-2.*g*(C(0,1)*wave_[medium]*d-1.)/d2;
                        auto phi=polynomial(rule.pts[u][0],rule.pts[u][1]);
                        for(int b=0;b<basis_count_;++b){exact[b].s-=pot*phi[b];exact[b].d[0]-=grad*r.x*phi[b];exact[b].d[1]-=grad*r.y*phi[b];exact[b].d[2]-=grad*r.z*phi[b];}}
                    const auto phi=polynomial(rule.pts[q][0],rule.pts[q][1]);
                    for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){auto&v=total[10*a+b];const double w=rule.wts[q]*phi[a];v.s+=w*exact[b].s;for(int c=0;c<3;++c)v.d[c]+=w*exact[b].d[c];}}
                for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b)kernels_[medium][pair*basis_count_*basis_count_+a*basis_count_+b]=total[10*a+b];}}
        // Enforce precisely the same weighted reciprocity as the dense oracle.
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<count;++i)for(size_t pair=near_rows_[i];pair<near_rows_[i+1];++pair){const size_t j=near_columns_[pair];if(j>i)continue;
            auto reverse=std::lower_bound(near_columns_.begin()+near_rows_[j],near_columns_.begin()+near_rows_[j+1],int(i));const size_t back=reverse-near_columns_.begin();
            const double ratio=panels_[j].area/panels_[i].area;
            for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){if(i==j&&b>a)continue;
                for(int m=0;m<2;++m){auto&u=kernels_[m][pair*basis_count_*basis_count_+a*basis_count_+b];auto&v=kernels_[m][back*basis_count_*basis_count_+b*basis_count_+a];
                    C s=.5*(u.s+ratio*v.s);CV d;for(int c=0;c<3;++c)d[c]=.5*(u.d[c]-ratio*v.d[c]);u.s=s;v.s=s/ratio;
                    for(int c=0;c<3;++c){u.d[c]=d[c];v.d[c]=-d[c]/ratio;}}}}
        return;
    }
    for(auto&v:kernels_)v.resize(coefficient_count*coefficient_count);
    const auto ordinary_targets=target_points(target_order_,0),refined_targets=target_points(target_order_,near_subdiv);
#pragma omp parallel for schedule(dynamic,1)
    for(size_t i=0;i<count;++i)for(size_t j=0;j<count;++j){const auto&t=panels_[i];
        const bool refine=near_subdiv&&within_near((t.center-panels_[j].center).norm(),near_radius_+1.,std::max(t.diameter,panels_[j].diameter));
        const auto& targets=refine?refined_targets:ordinary_targets;
        for(int medium=0;medium<2;++medium){std::array<Kernel,100> total{};
            for(const auto&q:targets){const Vec3 x=t.vertex[0]+(t.vertex[1]-t.vertex[0])*q[0]+(t.vertex[2]-t.vertex[0])*q[1];
                auto v=integrate(panels_[j],x,wave_[medium],within_near((x-panels_[j].center).norm(),near_radius_,panels_[j].diameter));
                const auto phi=polynomial(q[0],q[1]);
                for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){auto&c=total[10*a+b];double w=q[2]*phi[a];
                    c.s+=w*v[b].s;for(int j=0;j<3;++j)c.d[j]+=w*v[b].d[j];}}
            for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b)
                kernels_[medium][(i*basis_count_+a)*coefficient_count+j*basis_count_+b]=total[10*a+b];}}
    if(target_order_!=1){
        // Galerkin scalar kernels obey weighted reciprocity. Average the two
        // quadrature directions. A diagonal basis-pair gradient is exactly
        // zero; off-diagonal polynomial pairs on the same panel are retained.
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<coefficient_count;++i)for(size_t j=0;j<=i;++j)for(int m=0;m<2;++m){
            auto&a=kernels_[m][i*coefficient_count+j];auto&b=kernels_[m][j*coefficient_count+i];
            const double ratio=panels_[j/basis_count_].area/panels_[i/basis_count_].area;
            C s=.5*(a.s+ratio*b.s);CV d;for(int c=0;c<3;++c)d[c]=.5*(a.d[c]-ratio*b.d[c]);
            a.s=s;b.s=s/ratio;for(int c=0;c<3;++c){a.d[c]=d[c];b.d[c]=-d[c]/ratio;}}
    }
}
// Direct row construction keeps the uncompressed constructor as an independent
// reference. Both directed quadratures enter the same reciprocal Galerkin rule.
std::array<Kernel,100> Transmission::pair_integrals(int i,int j,int medium,
    const std::vector<std::array<double,3>>&ordinary,const std::vector<std::array<double,3>>&refined)const{
    const auto&t=panels_[i];const auto&s=panels_[j];
    const bool refine=near_subdiv_&&within_near((t.center-s.center).norm(),near_radius_+1.,std::max(t.diameter,s.diameter));
    const auto&targets=refine?refined:ordinary;std::array<Kernel,100> total{};
    for(const auto&q:targets){const Vec3 x=t.vertex[0]+(t.vertex[1]-t.vertex[0])*q[0]+(t.vertex[2]-t.vertex[0])*q[1];
        const auto v=integrate(s,x,wave_[medium],within_near((x-s.center).norm(),near_radius_,s.diameter));
        const auto phi=polynomial(q[0],q[1]);
        for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){auto&c=total[10*a+b];const double w=q[2]*phi[a];
            c.s+=w*v[b].s;for(int k=0;k<3;++k)c.d[k]+=w*v[b].d[k];}}
    return total;
}
std::array<Kernel,100> Transmission::reciprocal_pair(int i,int j,int medium,
    const std::vector<std::array<double,3>>&ordinary,const std::vector<std::array<double,3>>&refined)const{
    const auto f=pair_integrals(i,j,medium,ordinary,refined);
    if(target_order_==1)return f;
    const auto r=i==j?f:pair_integrals(j,i,medium,ordinary,refined);auto result=f;
    const double ratio=panels_[j].area/panels_[i].area;
    for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){auto&v=result[10*a+b];const auto&u=r[10*b+a];
        v.s=.5*(v.s+ratio*u.s);for(int k=0;k<3;++k)v.d[k]=.5*(v.d[k]-ratio*u.d[k]);}
    return result;
}
void Transmission::build_rows(int count,const int*ids,size_t budget){
    if(correction_only_||!kernels_[0].empty()||!dense_row_map_.empty())throw std::logic_error("row preparation requires deferred empty dense cache");
    if(count<1||size_t(count)>panels_.size())throw std::invalid_argument("selected panel count");
    for(int j=0;j<count;++j)if(ids[j]<0||size_t(ids[j])>=panels_.size()||(j&&ids[j]<=ids[j-1]))throw std::invalid_argument("selected panels must be sorted and unique");
    const size_t n=size()/8,nr=size_t(count)*basis_count_;
    if(nr>budget/(2*sizeof(Kernel)*n))throw std::runtime_error("representative row cache exceeds budget");
    std::array<std::vector<Kernel>,2> next;for(auto&v:next)v.resize(nr*n);
    const auto ordinary=target_points(target_order_,0),refined=target_points(target_order_,near_subdiv_);
#pragma omp parallel for schedule(dynamic,1)
    for(int slot=0;slot<count;++slot)for(size_t j=0;j<panels_.size();++j)for(int m=0;m<2;++m){
        const auto block=reciprocal_pair(ids[slot],j,m,ordinary,refined);
        for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b)
            next[m][(size_t(slot)*basis_count_+a)*n+j*basis_count_+b]=block[10*a+b];}
    std::vector<int> map(n,-1);for(int slot=0;slot<count;++slot)for(int a=0;a<basis_count_;++a)map[ids[slot]*basis_count_+a]=slot*basis_count_+a;
    kernels_.swap(next);dense_row_map_.swap(map);
}
void Transmission::apply_rows_reference(const C*h,int nr,const int*rows,C*out)const{
    if(correction_only_||kernels_[0].empty())throw std::logic_error("reference rows require prepared dense cache");
    if(nr<1||nr>size()/8)throw std::invalid_argument("reference row count");
    for(int a=0;a<nr;++a)if(rows[a]<0||rows[a]>=size()/8||(!dense_row_map_.empty()&&dense_row_map_[rows[a]]<0))throw std::invalid_argument("reference row not stored");
    const auto qe=to_charges(h,&scale_.pp),qi=to_charges(h,&scale_.np);const size_t n=size()/8;
#pragma omp parallel for schedule(static)
    for(int t=0;t<nr;++t){const int i=rows[t],stored=dense_row_map_.empty()?i:dense_row_map_[i];Eight ex{},in{},e{},v{};
        for(size_t j=0;j<n;++j){const auto a=action(kernels_[0][size_t(stored)*n+j],qe.data()+8*j),b=action(kernels_[1][size_t(stored)*n+j],qi.data()+8*j);
            for(int c=0;c<8;++c){ex[c]+=a[c];in[c]+=b[c];}}
        localize(panels_[i/basis_count_],ex,e.data());localize(panels_[i/basis_count_],in,v.data());
        for(int c=0;c<8;++c)out[8*t+c]=h[8*i+c]+scale_.p[c]*v[c]-scale_.n[c]*e[c];}
}
// Independent contraction table for the Cartesian Clifford action above.
// SIMD lanes are RHS, never source nodes: each lane retains increasing j and
// one accumulated contribution per component/source. Do not reuse dense_batch
// (the iteration kernel) for a supposedly independent residual check.
template<int W> static void reference_packet(const std::vector<Panel>&panels,int nb,
        const std::array<std::vector<Kernel>,2>&kernels,const std::vector<int>&map,
        const Scaling&scale,int active,const C*h,int nr,const int*rows,C*out){
    const size_t n=panels.size()*nb,width=8*n,reduced=8*size_t(nr);
    // [source][component][real/imaginary][RHS]. Process media separately so
    // the charges of up to eight RHS fit in the CPU's last-level cache.
    std::vector<double> q(n*16*W,0.);
    static const int ki[8][4]={{1,2,3,0},{2,3,1,0},{3,1,2,0},{1,2,3,0},
        {1,2,3,0},{2,3,1,0},{3,1,2,0},{1,2,3,0}};
    static const int qi[8][4]={{0,1,2,7},{2,1,3,4},{0,2,3,5},{1,0,3,6},
        {4,5,6,3},{6,5,7,0},{4,6,7,1},{5,4,7,2}};
    static const double sign[8][4]={{1,1,1,1},{1,-1,-1,1},{1,-1,-1,1},{1,-1,-1,1},
        {-1,-1,-1,1},{1,-1,-1,-1},{1,-1,-1,-1},{1,-1,-1,-1}};
    for(int medium=0;medium<2;++medium){
        const auto&s=medium?scale.np:scale.pp;
#pragma omp parallel for schedule(static)
        for(size_t j=0;j<n;++j)for(int r=0;r<active;++r){
            Eight v{};for(int c=0;c<8;++c)v[c]=h[size_t(r)*width+8*j+c]*s[c];
            const auto z=charges(panels[j/nb],v.data());
            for(int c=0;c<8;++c){q[(j*16+2*c)*W+r]=z[c].real();q[(j*16+2*c+1)*W+r]=z[c].imag();}
        }
#pragma omp parallel for schedule(static)
        for(int t=0;t<nr;++t){const int i=rows[t],stored=map.empty()?i:map[i];
            const Kernel*row=kernels[medium].data()+size_t(stored)*n;
            double sum[16][W]={};
            for(size_t j=0;j<n;++j){const auto&k=row[j];
                const double kr[4]={k.s.real(),k.d[0].real(),k.d[1].real(),k.d[2].real()};
                const double km[4]={k.s.imag(),k.d[0].imag(),k.d[1].imag(),k.d[2].imag()};
                const double*charge=q.data()+j*16*W;
#pragma GCC unroll 8
                for(int c=0;c<8;++c){
                    const int a=qi[c][0],b=qi[c][1],d=qi[c][2],e=qi[c][3];
                    const double ar=sign[c][0]*kr[ki[c][0]],ai=sign[c][0]*km[ki[c][0]],
                        br=sign[c][1]*kr[ki[c][1]],bi=sign[c][1]*km[ki[c][1]],
                        dr=sign[c][2]*kr[ki[c][2]],di=sign[c][2]*km[ki[c][2]],
                        er=sign[c][3]*kr[ki[c][3]],ei=sign[c][3]*km[ki[c][3]];
#pragma omp simd
                    for(int r=0;r<W;++r){
                        const double re=((ar*charge[2*a*W+r]-ai*charge[(2*a+1)*W+r])
                            +(br*charge[2*b*W+r]-bi*charge[(2*b+1)*W+r]))
                            +(dr*charge[2*d*W+r]-di*charge[(2*d+1)*W+r])
                            +(er*charge[2*e*W+r]-ei*charge[(2*e+1)*W+r]);
                        const double im=((ar*charge[(2*a+1)*W+r]+ai*charge[2*a*W+r])
                            +(br*charge[(2*b+1)*W+r]+bi*charge[2*b*W+r]))
                            +(dr*charge[(2*d+1)*W+r]+di*charge[2*d*W+r])
                            +(er*charge[(2*e+1)*W+r]+ei*charge[2*e*W+r]);
                        sum[2*c][r]+=re;sum[2*c+1][r]+=im;
                    }
                }
            }
            for(int r=0;r<active;++r){Eight v{},local{};
                for(int c=0;c<8;++c)v[c]=C(sum[2*c][r],sum[2*c+1][r]);
                localize(panels[i/nb],v,local.data());
                for(int c=0;c<8;++c){C&y=out[size_t(r)*reduced+8*t+c];
                    if(medium==0)y=local[c];
                    else y=h[size_t(r)*width+8*i+c]+scale.p[c]*local[c]-scale.n[c]*y;}
            }
        }
    }
}
void Transmission::apply_rows_reference_batch(int count,const C*h,int nr,const int*rows,C*out)const{
    if(correction_only_||kernels_[0].empty())throw std::logic_error("reference batch requires prepared dense cache");
    if(count<1||count>16||nr<1||nr>size()/8)throw std::invalid_argument("reference batch dimensions");
    for(int a=0;a<nr;++a)if(rows[a]<0||rows[a]>=size()/8||(!dense_row_map_.empty()&&dense_row_map_[rows[a]]<0))throw std::invalid_argument("reference batch row not stored");
    if(count==1){apply_rows_reference(h,nr,rows,out);return;}
    for(int begin=0;begin<count;begin+=8){const int active=std::min(8,count-begin);
        const C*x=h+size_t(begin)*size();C*y=out+size_t(begin)*nr*8;
        if(active<=2)reference_packet<2>(panels_,basis_count_,kernels_,dense_row_map_,scale_,active,x,nr,rows,y);
        else if(active<=4)reference_packet<4>(panels_,basis_count_,kernels_,dense_row_map_,scale_,active,x,nr,rows,y);
        else reference_packet<8>(panels_,basis_count_,kernels_,dense_row_map_,scale_,active,x,nr,rows,y);
    }
}
void Transmission::apply_direct_panels(const C*h,int count,const int*ids,C*out)const{
    if(count<1||size_t(count)>panels_.size())throw std::invalid_argument("direct panel count");
    for(int t=0;t<count;++t)if(ids[t]<0||size_t(ids[t])>=panels_.size())throw std::invalid_argument("direct panel index");
    const auto qe=to_charges(h,&scale_.pp),qi=to_charges(h,&scale_.np);
    const auto ordinary=target_points(target_order_,0),refined=target_points(target_order_,near_subdiv_);
#pragma omp parallel for schedule(dynamic,1)
    for(int t=0;t<count;++t){const int i=ids[t];std::array<Eight,10> ex{},in{};
        for(size_t j=0;j<panels_.size();++j){const auto ke=reciprocal_pair(i,j,0,ordinary,refined),ki=reciprocal_pair(i,j,1,ordinary,refined);
            for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){
                const auto e=action(ke[10*a+b],qe.data()+8*(j*basis_count_+b)),v=action(ki[10*a+b],qi.data()+8*(j*basis_count_+b));
                for(int c=0;c<8;++c){ex[a][c]+=e[c];in[a][c]+=v[c];}}}
        for(int a=0;a<basis_count_;++a){Eight e{},v{};localize(panels_[i],ex[a],e.data());localize(panels_[i],in[a],v.data());
            for(int c=0;c<8;++c)out[(t*basis_count_+a)*8+c]=h[(i*basis_count_+a)*8+c]+scale_.p[c]*v[c]-scale_.n[c]*e[c];}}
}
void Transmission::build_near_rows(int count,const int*ids,size_t budget){
    if(correction_only_||!kernels_[0].empty()||!near_rows_.empty())throw std::logic_error("near rows require empty deferred geometry");
    if(target_order_!=13||far_order_!=13||count<1||size_t(count)>panels_.size())throw std::invalid_argument("near rows require target13");
    std::vector<int> selected(panels_.size(),0);
    for(int j=0;j<count;++j){if(ids[j]<0||size_t(ids[j])>=panels_.size()||(j&&ids[j]<=ids[j-1]))throw std::invalid_argument("near panels must be sorted and unique");selected[ids[j]]=1;}
    double cell=0.;for(const auto&p:panels_)cell=std::max(cell,(near_radius_+1.)*p.diameter*(1.+NEAR_TIE));
    using Key=std::tuple<int,int,int>;std::map<Key,std::vector<int>> bins;
    auto key=[&](const Vec3&v){return Key(int(std::floor(v.x/cell)),int(std::floor(v.y/cell)),int(std::floor(v.z/cell)));};
    for(size_t j=0;j<panels_.size();++j)bins[key(panels_[j].center)].push_back(int(j));
    std::vector<size_t> offsets(1,0);std::vector<int> columns;
    for(size_t i=0;i<panels_.size();++i){
        if(selected[i]){auto index=key(panels_[i].center);std::vector<int> neighbors;
            for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z){
                auto it=bins.find(Key(std::get<0>(index)+x,std::get<1>(index)+y,std::get<2>(index)+z));if(it==bins.end())continue;
                for(int j:it->second)if(within_near((panels_[i].center-panels_[j].center).norm(),near_radius_+1.,std::max(panels_[i].diameter,panels_[j].diameter)))neighbors.push_back(j);}
            std::sort(neighbors.begin(),neighbors.end());columns.insert(columns.end(),neighbors.begin(),neighbors.end());
            if(columns.size()>budget/(2*sizeof(Kernel)*basis_count_*basis_count_))throw std::runtime_error("selected near cache exceeds budget");}
        offsets.push_back(columns.size());
    }
    const size_t entries=columns.size()*basis_count_*basis_count_;
    if(entries*2*sizeof(Kernel)+columns.size()*sizeof(int)+offsets.size()*sizeof(size_t)>budget)throw std::runtime_error("selected near cache and indices exceed budget");
    std::array<std::vector<Kernel>,2> next;for(auto&v:next)v.resize(entries);
    const auto rule=tri_quadrature(13);const auto ordinary=target_points(13,0),refined=target_points(13,near_subdiv_);
#pragma omp parallel for schedule(dynamic,1)
    for(int slot=0;slot<count;++slot){const int i=ids[slot];const auto&t=panels_[i];
        for(size_t pair=offsets[i];pair<offsets[i+1];++pair){const int j=columns[pair];const auto&s=panels_[j];
            for(int medium=0;medium<2;++medium){auto total=reciprocal_pair(i,j,medium,ordinary,refined);
                // FMM's complete 13x13 point rule is subtracted, including
                // its self-source omission. No far pairs are stored here.
                for(int q=0;q<13;++q){const Vec3 x=t.vertex[0]+(t.vertex[1]-t.vertex[0])*rule.pts[q][0]+(t.vertex[2]-t.vertex[0])*rule.pts[q][1];
                    const auto phi=polynomial(rule.pts[q][0],rule.pts[q][1]);
                    for(int u=0;u<13;++u){const Vec3 y=s.vertex[0]+(s.vertex[1]-s.vertex[0])*rule.pts[u][0]+(s.vertex[2]-s.vertex[0])*rule.pts[u][1],r=x-y;
                        const double d2=r.norm2();if(d2<1e-28)continue;const double distance=std::sqrt(d2);
                        const C g=s.area*rule.wts[u]*INV4PI*std::exp(C(0,1)*wave_[medium]*distance)/distance;
                        const C pot=2.*C(0,1)*wave_[medium]*g,grad=-2.*g*(C(0,1)*wave_[medium]*distance-1.)/d2;
                        const auto source=polynomial(rule.pts[u][0],rule.pts[u][1]);
                        for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b){auto&v=total[10*a+b];const double w=rule.wts[q]*phi[a]*source[b];
                            v.s-=w*pot;v.d[0]-=w*grad*r.x;v.d[1]-=w*grad*r.y;v.d[2]-=w*grad*r.z;}}
                }
                for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b)next[medium][pair*basis_count_*basis_count_+a*basis_count_+b]=total[10*a+b];
            }
        }
    }
    kernels_.swap(next);near_rows_.swap(offsets);near_columns_.swap(columns);correction_only_=true;
}
void Transmission::near_correction_rows(int medium,const C*h,int count,const int*ids,C*out)const{
    if(!correction_only_||medium<0||medium>1||count<1||size_t(count)>panels_.size())throw std::invalid_argument("near row action controls");
    for(int j=0;j<count;++j)if(ids[j]<0||size_t(ids[j])>=panels_.size()||near_rows_[ids[j]]==near_rows_[ids[j]+1])throw std::invalid_argument("near row not stored");
    const auto q=to_charges(h,nullptr);
#pragma omp parallel for schedule(static)
    for(int slot=0;slot<count;++slot){const int i=ids[slot];
        for(int a=0;a<basis_count_;++a){Eight sum{};
            for(size_t pair=near_rows_[i];pair<near_rows_[i+1];++pair)for(int b=0;b<basis_count_;++b){
                auto v=action(kernels_[medium][pair*basis_count_*basis_count_+a*basis_count_+b],q.data()+8*(near_columns_[pair]*basis_count_+b));
                for(int c=0;c<8;++c)sum[c]+=v[c];}
            localize(panels_[i],sum,out+8*(slot*basis_count_+a));
        }
    }
}
void Transmission::self_blocks_direct(int count,const int*ids,C*out)const{
    if(count<1||size_t(count)>panels_.size())throw std::invalid_argument("self block count");
    for(int j=0;j<count;++j)if(ids[j]<0||size_t(ids[j])>=panels_.size())throw std::invalid_argument("self block panel");
    const auto ordinary=target_points(target_order_,0),refined=target_points(target_order_,near_subdiv_);const int width=basis_count_*8;
#pragma omp parallel for schedule(dynamic,1)
    for(int slot=0;slot<count;++slot){const int i=ids[slot];const auto e=reciprocal_pair(i,i,0,ordinary,refined),v=reciprocal_pair(i,i,1,ordinary,refined);
        for(int a=0;a<basis_count_;++a)for(int b=0;b<basis_count_;++b)for(int c=0;c<8;++c){
            Eight he{},hi{},ex{},in{};he[c]=scale_.pp[c];hi[c]=scale_.np[c];
            localize(panels_[i],action(e[10*a+b],charges(panels_[i],he.data()).data()),ex.data());
            localize(panels_[i],action(v[10*a+b],charges(panels_[i],hi.data()).data()),in.data());
            for(int r=0;r<8;++r)out[(size_t(slot)*width+8*a+r)*width+8*b+c]=C(a==b&&r==c)+scale_.p[r]*in[r]-scale_.n[r]*ex[r];}
    }
}
void Transmission::direct_transformed_columns(int nt,const int*targets,int ns,const int*sources,const double*basis,const double*components,C*out)const{
    direct_transformed_columns_basis(nt,targets,ns,sources,basis_count_,basis,components,out);
}
void Transmission::direct_transformed_columns_basis(int nt,const int*targets,int ns,const int*sources,int cb,const double*basis,const double*components,C*out)const{
    if(nt<1||ns<1||cb<1||cb>basis_count_)throw std::invalid_argument("direct coarse tile dimensions");
    for(int j=0;j<nt;++j)if(targets[j]<0||size_t(targets[j])>=panels_.size())throw std::invalid_argument("direct target panel");
    for(int j=0;j<ns;++j)if(sources[j]<0||size_t(sources[j])>=panels_.size())throw std::invalid_argument("direct source panel");
    const int nb=basis_count_,width=8*nb;const auto ordinary=target_points(target_order_,0),refined=target_points(target_order_,near_subdiv_);
    std::vector<std::array<Eight,8>> qe(ns),qi(ns);
    for(int j=0;j<ns;++j)for(int c=0;c<8;++c){Eight h{};
        for(int r=0;r<8;++r)h[r]=scale_.pp[r]*components[64*j+8*r+c];
        qe[j][c]=charges(panels_[sources[j]],h.data());
        for(int r=0;r<8;++r)h[r]=scale_.np[r]*components[64*j+8*r+c];
        qi[j][c]=charges(panels_[sources[j]],h.data());}
#pragma omp parallel for schedule(dynamic,1)
    for(int t=0;t<nt;++t){const int row=targets[t];
        for(int j=0;j<ns;++j){const auto e=reciprocal_pair(row,sources[j],0,ordinary,refined),v=reciprocal_pair(row,sources[j],1,ordinary,refined);
            for(int b=0;b<nb;++b)for(int a=0;a<cb;++a){Kernel ke{},ki{};
                for(int c=0;c<cb;++c){const double s=basis[(j*cb+c)*cb+a];ke.s+=s*e[10*b+c].s;ki.s+=s*v[10*b+c].s;
                    for(int r=0;r<3;++r){ke.d[r]+=s*e[10*b+c].d[r];ki.d[r]+=s*v[10*b+c].d[r];}}
                const double identity=row==sources[j]&&b<cb?basis[(j*cb+b)*cb+a]:0.;
                for(int c=0;c<8;++c){Eight ex{},in{};localize(panels_[row],action(ke,qe[j][c].data()),ex.data());localize(panels_[row],action(ki,qi[j][c].data()),in.data());
                    for(int r=0;r<8;++r)out[(size_t(t)*width+8*b+r)*(size_t(ns)*cb*8)+j*cb*8+8*a+c]=identity*components[64*j+8*r+c]+scale_.p[r]*in[r]-scale_.n[r]*ex[r];}
            }
        }
    }
}
void Transmission::direct_panel_pairs(int count,const int*targets,const int*sources,C*out)const{
    if(count<1)throw std::invalid_argument("direct pair count");
    for(int j=0;j<count;++j)if(targets[j]<0||size_t(targets[j])>=panels_.size()||sources[j]<0||size_t(sources[j])>=panels_.size())throw std::invalid_argument("direct pair panel index");
    const int nb=basis_count_,width=8*nb;const auto ordinary=target_points(target_order_,0),refined=target_points(target_order_,near_subdiv_);
#pragma omp parallel for schedule(dynamic,1)
    for(int pair=0;pair<count;++pair){const int row=targets[pair],col=sources[pair];
        const auto e=reciprocal_pair(row,col,0,ordinary,refined),v=reciprocal_pair(row,col,1,ordinary,refined);
        std::array<Eight,8> qe,qi;
        for(int c=0;c<8;++c){Eight h{};h[c]=scale_.pp[c];qe[c]=charges(panels_[col],h.data());h[c]=scale_.np[c];qi[c]=charges(panels_[col],h.data());}
        for(int a=0;a<nb;++a)for(int b=0;b<nb;++b)for(int c=0;c<8;++c){Eight ex{},in{};
            localize(panels_[row],action(e[10*a+b],qe[c].data()),ex.data());localize(panels_[row],action(v[10*a+b],qi[c].data()),in.data());
            for(int r=0;r<8;++r)out[(size_t(pair)*width+8*a+r)*width+8*b+c]=C(row==col&&a==b&&r==c)+scale_.p[r]*in[r]-scale_.n[r]*ex[r];}
    }
}
void Transmission::self_blocks_selected(int count,const int*ids,C*out)const{
    if(correction_only_||kernels_[0].empty()||count<1||size_t(count)>panels_.size())throw std::invalid_argument("selected self blocks");
    for(int p=0;p<count;++p){if(ids[p]<0||size_t(ids[p])>=panels_.size())throw std::invalid_argument("self panel index");
        for(int a=0;a<basis_count_;++a)if(!dense_row_map_.empty()&&dense_row_map_[ids[p]*basis_count_+a]<0)throw std::invalid_argument("self row not stored");}
    const int width=8*basis_count_;const size_t n=size()/8;
#pragma omp parallel for schedule(static)
    for(int slot=0;slot<count;++slot){const int p=ids[slot];
        for(int col=0;col<width;++col){const int b=col/8,c=col%8;Eight h{};h[c]=scale_.pp[c];const auto qe=charges(panels_[p],h.data());h[c]=scale_.np[c];const auto qi=charges(panels_[p],h.data());
            for(int a=0;a<basis_count_;++a){const int row=p*basis_count_+a,stored=dense_row_map_.empty()?row:dense_row_map_[row];
                const size_t index=size_t(stored)*n+p*basis_count_+b;Eight e{},v{};
                localize(panels_[p],action(kernels_[0][index],qe.data()),e.data());localize(panels_[p],action(kernels_[1][index],qi.data()),v.data());
                for(int r=0;r<8;++r)out[(size_t(slot)*width+8*a+r)*width+col]=double(8*a+r==col)+scale_.p[r]*v[r]-scale_.n[r]*e[r];}}}
}

void Transmission::submatrix(int nr,const int*rows,int nc,const int*cols,C*out)const{
    if(correction_only_||kernels_[0].empty())throw std::logic_error("submatrix requires prepared dense cache");
    const int n=size()/8;
    if(nr<1||nr>n||nc<1||nc>n)throw std::invalid_argument("submatrix dimensions");
    for(int i=0;i<nr;++i)if(rows[i]<0||rows[i]>=n||(!dense_row_map_.empty()&&dense_row_map_[rows[i]]<0))throw std::invalid_argument("submatrix row not stored");
    for(int j=0;j<nc;++j)if(cols[j]<0||cols[j]>=n)throw std::invalid_argument("submatrix column");
    std::vector<std::array<Eight,8>> qe(nc),qi(nc);
    for(int j=0;j<nc;++j)for(int c=0;c<8;++c){Eight unit{};unit[c]=scale_.pp[c];qe[j][c]=charges(panels_[cols[j]/basis_count_],unit.data());
        unit[c]=scale_.np[c];qi[j][c]=charges(panels_[cols[j]/basis_count_],unit.data());}
#pragma omp parallel for schedule(static)
    for(int i=0;i<nr;++i){const int row=rows[i],stored=dense_row_map_.empty()?row:dense_row_map_[row];
        for(int j=0;j<nc;++j){const size_t index=size_t(stored)*n+cols[j];
            for(int c=0;c<8;++c){Eight e{},v{};localize(panels_[row/basis_count_],action(kernels_[0][index],qe[j][c].data()),e.data());
                localize(panels_[row/basis_count_],action(kernels_[1][index],qi[j][c].data()),v.data());
                for(int r=0;r<8;++r)out[size_t(8*i+r)*(8*nc)+8*j+c]=double(row==cols[j]&&r==c)+scale_.p[r]*v[r]-scale_.n[r]*e[r];}}}
}

void Transmission::transformed_columns(int nr,const int*rows,int ns,const int*ids,int cb,
    const double*basis,const double*components,C*out)const{
    if(correction_only_||kernels_[0].empty())throw std::logic_error("transformed columns require prepared dense cache");
    const int n=size()/8;
    if(nr<1||nr>n||ns<1||size_t(ns)>panels_.size()||cb>basis_count_||(cb!=1&&cb!=3&&cb!=6&&cb!=10))throw std::invalid_argument("transformed column dimensions");
    for(int i=0;i<nr;++i)if(rows[i]<0||rows[i]>=n||(!dense_row_map_.empty()&&dense_row_map_[rows[i]]<0))throw std::invalid_argument("transformed row not stored");
    for(int j=0;j<ns;++j)if(ids[j]<0||size_t(ids[j])>=panels_.size())throw std::invalid_argument("transformed source panel");
    std::vector<std::array<Eight,8>> qe(ns),qi(ns);
    for(int j=0;j<ns;++j)for(int c=0;c<8;++c){Eight unit{};
        for(int r=0;r<8;++r){unit[r]=scale_.pp[r]*components[64*j+8*r+c];}qe[j][c]=charges(panels_[ids[j]],unit.data());
        for(int r=0;r<8;++r){unit[r]=scale_.np[r]*components[64*j+8*r+c];}qi[j][c]=charges(panels_[ids[j]],unit.data());}
#pragma omp parallel for schedule(static)
    for(int i=0;i<nr;++i){const int row=rows[i],stored=dense_row_map_.empty()?row:dense_row_map_[row];
        for(int j=0;j<ns;++j)for(int a=0;a<cb;++a){Kernel ke{},ki{};
            for(int b=0;b<cb;++b){const double s=basis[(j*cb+b)*cb+a];const size_t at=size_t(stored)*n+ids[j]*basis_count_+b;
                ke.s+=s*kernels_[0][at].s;ki.s+=s*kernels_[1][at].s;
                for(int r=0;r<3;++r){ke.d[r]+=s*kernels_[0][at].d[r];ki.d[r]+=s*kernels_[1][at].d[r];}}
            const double identity=(row/basis_count_==ids[j]&&row%basis_count_<cb)?basis[(j*cb+row%basis_count_)*cb+a]:0.;
            for(int c=0;c<8;++c){Eight e{},v{};localize(panels_[row/basis_count_],action(ke,qe[j][c].data()),e.data());
                localize(panels_[row/basis_count_],action(ki,qi[j][c].data()),v.data());
                for(int r=0;r<8;++r)out[size_t(8*i+r)*(8*cb*ns)+8*(j*cb+a)+c]=identity*components[64*j+8*r+c]+scale_.p[r]*v[r]-scale_.n[r]*e[r];}}}
}

V Transmission::to_charges(const C*h,const Eight*scale)const{
    V result(size());
#pragma omp parallel for
    for(int j=0;j<size()/8;++j){Eight v;for(int a=0;a<8;++a)v[a]=h[8*j+a]*(scale?(*scale)[a]:C(1));
        const auto q=charges(panels_[j/basis_count_],v.data());for(int a=0;a<8;++a)result[8*j+a]=q[a];}return result;
}
void Transmission::cauchy_charges(int medium,const C*q,C*out)const{
    const size_t n=size()/8;
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i){Eight sum{};const auto*row=kernels_[medium].data()+i*n;
        for(size_t j=0;j<n;++j){auto v=action(row[j],q+8*j);for(int a=0;a<8;++a)sum[a]+=v[a];}
        localize(panels_[i/basis_count_],sum,out+8*i);}
}
void Transmission::cauchy(int medium,const C*h,C*out)const{
    if(correction_only_||!dense_row_map_.empty()||kernels_[0].empty())throw std::logic_error("Dirac full Cauchy action requires full dense cache");
    if(medium!=0&&medium!=1)throw std::invalid_argument("medium");
    auto q=to_charges(h,nullptr);cauchy_charges(medium,q.data(),out);
}
void Transmission::apply(const C*h,C*out)const{
    if(correction_only_||!dense_row_map_.empty()||kernels_[0].empty())throw std::logic_error("Dirac full native action requires full dense cache");
    auto qe=to_charges(h,&scale_.pp),qi=to_charges(h,&scale_.np);V e(size()),i(size());
    cauchy_charges(0,qe.data(),e.data());cauchy_charges(1,qi.data(),i.data());
#pragma omp parallel for
    for(int j=0;j<size();++j)out[j]=h[j]+scale_.p[j%8]*i[j]-scale_.n[j%8]*e[j];
}
template<int Component> static inline double kernel_component(const Kernel*row,size_t j,size_t){
    const C& value=Component<2?row[j].s:row[j].d[(Component-2)/2];return Component%2?value.imag():value.real();}
template<int Component> static inline double kernel_component(const float*row,size_t j,size_t n){return row[Component*n+j];}
template<class K> void dense_simd(const K*matrix,size_t row_stride,const std::vector<Panel>&panels,int basis_count,size_t n,const C*q,C*out){
    // Structure-of-arrays charges fit in cache. Keep the existing kernel
    // storage, including the exact near entries and reciprocal projection.
    std::vector<double> charge(16*n);
    for(size_t j=0;j<n;++j)for(int a=0;a<8;++a){charge[(2*a)*n+j]=q[8*j+a].real();charge[(2*a+1)*n+j]=q[8*j+a].imag();}
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i){
        const auto* row=matrix+i*row_stride;
        double r0=0,i0=0,r1=0,i1=0,r2=0,i2=0,r3=0,i3=0,r4=0,i4=0,r5=0,i5=0,r6=0,i6=0,r7=0,i7=0;
#pragma omp simd reduction(+:r0,i0,r1,i1,r2,i2,r3,i3,r4,i4,r5,i5,r6,i6,r7,i7)
        for(size_t j=0;j<n;++j){
            const double sr=kernel_component<0>(row,j,n),si=kernel_component<1>(row,j,n);
            const double xr=kernel_component<2>(row,j,n),xi=kernel_component<3>(row,j,n),yr=kernel_component<4>(row,j,n),yi=kernel_component<5>(row,j,n),zr=kernel_component<6>(row,j,n),zi=kernel_component<7>(row,j,n);
#define ADD_COMPONENT(A,K,Q,SIGN) r##A+=SIGN*(K##r*charge[(2*(Q))*n+j]-K##i*charge[(2*(Q)+1)*n+j]); i##A+=SIGN*(K##r*charge[(2*(Q)+1)*n+j]+K##i*charge[(2*(Q))*n+j]);
            ADD_COMPONENT(0,x,0,1) ADD_COMPONENT(0,y,1,1) ADD_COMPONENT(0,z,2,1) ADD_COMPONENT(0,s,7,1)
            ADD_COMPONENT(1,y,2,1) ADD_COMPONENT(1,z,1,-1) ADD_COMPONENT(1,x,3,-1) ADD_COMPONENT(1,s,4,1)
            ADD_COMPONENT(2,z,0,1) ADD_COMPONENT(2,x,2,-1) ADD_COMPONENT(2,y,3,-1) ADD_COMPONENT(2,s,5,1)
            ADD_COMPONENT(3,x,1,1) ADD_COMPONENT(3,y,0,-1) ADD_COMPONENT(3,z,3,-1) ADD_COMPONENT(3,s,6,1)
            ADD_COMPONENT(4,x,4,-1) ADD_COMPONENT(4,y,5,-1) ADD_COMPONENT(4,z,6,-1) ADD_COMPONENT(4,s,3,1)
            ADD_COMPONENT(5,y,6,1) ADD_COMPONENT(5,z,5,-1) ADD_COMPONENT(5,x,7,-1) ADD_COMPONENT(5,s,0,-1)
            ADD_COMPONENT(6,z,4,1) ADD_COMPONENT(6,x,6,-1) ADD_COMPONENT(6,y,7,-1) ADD_COMPONENT(6,s,1,-1)
            ADD_COMPONENT(7,x,5,1) ADD_COMPONENT(7,y,4,-1) ADD_COMPONENT(7,z,7,-1) ADD_COMPONENT(7,s,2,-1)
#undef ADD_COMPONENT
        }
        const Eight sum{{C(r0,i0),C(r1,i1),C(r2,i2),C(r3,i3),C(r4,i4),C(r5,i5),C(r6,i6),C(r7,i7)}};
        localize(panels[i/basis_count],sum,out+8*i);
    }
}
// Retain source tiles in private cache while independently accumulating RHS.
// No matrix approximation and no coupling between Krylov spaces.
static void dense_batch(const Kernel*matrix,const std::vector<Panel>&panels,int basis_count,
                        size_t n,int count,const C*q,C*out,int tile,const int*selected=nullptr,int row_count=-1,const int*storage_rows=nullptr){
    const size_t targets=row_count<0?n:size_t(row_count);
    std::vector<double> charges_soa(size_t(count)*16*n);
    for(int r=0;r<count;++r)for(size_t j=0;j<n;++j)for(int a=0;a<8;++a){
        const C z=q[size_t(r)*8*n+8*j+a];
        charges_soa[(size_t(r)*16+2*a)*n+j]=z.real();
        charges_soa[(size_t(r)*16+2*a+1)*n+j]=z.imag();}
#pragma omp parallel for schedule(static)
    for(size_t ti=0;ti<targets;++ti){const size_t i=selected?selected[ti]:ti;
        Eight sums[16]{};const Kernel*row=matrix+size_t(storage_rows?storage_rows[i]:i)*n;
        for(size_t begin=0;begin<n;begin+=tile){const size_t end=std::min(n,begin+tile);
            for(int r=0;r<count;++r){const double*charge=charges_soa.data()+size_t(r)*16*n;
        double r0=0,i0=0,r1=0,i1=0,r2=0,i2=0,r3=0,i3=0,r4=0,i4=0,r5=0,i5=0,r6=0,i6=0,r7=0,i7=0;
#pragma omp simd reduction(+:r0,i0,r1,i1,r2,i2,r3,i3,r4,i4,r5,i5,r6,i6,r7,i7)
        for(size_t j=begin;j<end;++j){
            const double sr=kernel_component<0>(row,j,n),si=kernel_component<1>(row,j,n);
            const double xr=kernel_component<2>(row,j,n),xi=kernel_component<3>(row,j,n),yr=kernel_component<4>(row,j,n),yi=kernel_component<5>(row,j,n),zr=kernel_component<6>(row,j,n),zi=kernel_component<7>(row,j,n);
#define ADD_COMPONENT(A,K,Q,SIGN) r##A+=SIGN*(K##r*charge[(2*(Q))*n+j]-K##i*charge[(2*(Q)+1)*n+j]); i##A+=SIGN*(K##r*charge[(2*(Q)+1)*n+j]+K##i*charge[(2*(Q))*n+j]);
            ADD_COMPONENT(0,x,0,1) ADD_COMPONENT(0,y,1,1) ADD_COMPONENT(0,z,2,1) ADD_COMPONENT(0,s,7,1)
            ADD_COMPONENT(1,y,2,1) ADD_COMPONENT(1,z,1,-1) ADD_COMPONENT(1,x,3,-1) ADD_COMPONENT(1,s,4,1)
            ADD_COMPONENT(2,z,0,1) ADD_COMPONENT(2,x,2,-1) ADD_COMPONENT(2,y,3,-1) ADD_COMPONENT(2,s,5,1)
            ADD_COMPONENT(3,x,1,1) ADD_COMPONENT(3,y,0,-1) ADD_COMPONENT(3,z,3,-1) ADD_COMPONENT(3,s,6,1)
            ADD_COMPONENT(4,x,4,-1) ADD_COMPONENT(4,y,5,-1) ADD_COMPONENT(4,z,6,-1) ADD_COMPONENT(4,s,3,1)
            ADD_COMPONENT(5,y,6,1) ADD_COMPONENT(5,z,5,-1) ADD_COMPONENT(5,x,7,-1) ADD_COMPONENT(5,s,0,-1)
            ADD_COMPONENT(6,z,4,1) ADD_COMPONENT(6,x,6,-1) ADD_COMPONENT(6,y,7,-1) ADD_COMPONENT(6,s,1,-1)
            ADD_COMPONENT(7,x,5,1) ADD_COMPONENT(7,y,4,-1) ADD_COMPONENT(7,z,7,-1) ADD_COMPONENT(7,s,2,-1)
#undef ADD_COMPONENT
        }

                const Eight partial{{C(r0,i0),C(r1,i1),C(r2,i2),C(r3,i3),C(r4,i4),C(r5,i5),C(r6,i6),C(r7,i7)}};
                for(int c=0;c<8;++c)sums[r][c]+=partial[c];
            }
        }
        for(int r=0;r<count;++r)localize(panels[i/basis_count],sums[r],out+size_t(r)*8*targets+8*ti);
    }
}
void Transmission::apply_batch(int count,const C*h,C*out,int tile)const{
    if(correction_only_||!dense_row_map_.empty()||kernels_[0].empty())throw std::logic_error("Dirac native batch requires full dense cache");
    if(count<1||count>16||tile<16||tile>16384)throw std::invalid_argument("Dirac batch count or tile");
    if(count==1){apply_simd(h,out);return;}
    const size_t width=size();V qe(size_t(count)*width),qi(qe.size()),ex(qe.size()),in(qe.size());
    for(int r=0;r<count;++r){const auto e=to_charges(h+r*width,&scale_.pp),i=to_charges(h+r*width,&scale_.np);
        std::copy(e.begin(),e.end(),qe.begin()+r*width);std::copy(i.begin(),i.end(),qi.begin()+r*width);}
    dense_batch(kernels_[0].data(),panels_,basis_count_,width/8,count,qe.data(),ex.data(),tile);
    dense_batch(kernels_[1].data(),panels_,basis_count_,width/8,count,qi.data(),in.data(),tile);
#pragma omp parallel for schedule(static)
    for(size_t j=0;j<size_t(count)*width;++j)out[j]=h[j]+scale_.p[j%8]*in[j]-scale_.n[j%8]*ex[j];
}

void Transmission::apply_rows(int count,const C*h,int nr,const int*rows,C*out,int tile)const{
    if(correction_only_||kernels_[0].empty())throw std::logic_error("Dirac selected rows require prepared dense cache");
    if(count<1||count>16||nr<1||nr>size()/8||tile<16||tile>16384)throw std::invalid_argument("Dirac row dimensions");
    for(int j=0;j<nr;++j)if(rows[j]<0||rows[j]>=size()/8||(!dense_row_map_.empty()&&dense_row_map_[rows[j]]<0))throw std::invalid_argument("Dirac row not stored");
    const size_t width=size(),reduced=size_t(nr)*8;V qe(size_t(count)*width),qi(qe.size()),ex(size_t(count)*reduced),in(ex.size());
    for(int r=0;r<count;++r){const auto e=to_charges(h+r*width,&scale_.pp),i=to_charges(h+r*width,&scale_.np);
        std::copy(e.begin(),e.end(),qe.begin()+r*width);std::copy(i.begin(),i.end(),qi.begin()+r*width);}
    dense_batch(kernels_[0].data(),panels_,basis_count_,width/8,count,qe.data(),ex.data(),tile,rows,nr,dense_row_map_.empty()?nullptr:dense_row_map_.data());
    dense_batch(kernels_[1].data(),panels_,basis_count_,width/8,count,qi.data(),in.data(),tile,rows,nr,dense_row_map_.empty()?nullptr:dense_row_map_.data());
#pragma omp parallel for schedule(static)
    for(int r=0;r<count;++r)for(int j=0;j<nr;++j)for(int a=0;a<8;++a){const size_t k=size_t(r)*reduced+8*j+a;
        out[k]=h[size_t(r)*width+8*rows[j]+a]+scale_.p[a]*in[k]-scale_.n[a]*ex[k];}
}

void Transmission::cauchy_charges_simd(int medium,const C*q,C*out,bool mixed)const{
    if(mixed)dense_simd(kernels32_[medium].data(),size(),panels_,basis_count_,size()/8,q,out);
    else dense_simd(kernels_[medium].data(),size()/8,panels_,basis_count_,size()/8,q,out);
}
void Transmission::prepare_mixed_cache(size_t budget){
    if(correction_only_||!dense_row_map_.empty()||kernels_[0].empty())throw std::logic_error("Dirac mixed cache requires full dense cache");
    const size_t count=kernels_[0].size();
    if(count>budget/(16*sizeof(float)))throw std::runtime_error("Dirac additional mixed cache exceeds budget");
    if(kernels32_[0].size()==8*count&&kernels32_[1].size()==8*count)return;
    const size_t n=size()/8;
    for(int m=0;m<2;++m){kernels32_[m].resize(8*count);
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j){const auto&k=kernels_[m][i*n+j];
            kernels32_[m][8*i*n+j]=float(k.s.real());kernels32_[m][(8*i+1)*n+j]=float(k.s.imag());
            for(int c=0;c<3;++c){kernels32_[m][(8*i+2*c+2)*n+j]=float(k.d[c].real());kernels32_[m][(8*i+2*c+3)*n+j]=float(k.d[c].imag());}}
        int bad=0;
#pragma omp parallel for reduction(|:bad)
        for(size_t j=0;j<8*count;++j)bad|=!std::isfinite(kernels32_[m][j]);
        if(bad){for(auto&v:kernels32_)v.clear();throw std::runtime_error("Dirac mixed cache overflow");}
    }
}
void Transmission::apply_simd(const C*h,C*out,bool mixed)const{
    if(correction_only_||!dense_row_map_.empty()||kernels_[0].empty())throw std::logic_error("Dirac full native action requires full dense cache");
    if(mixed&&(kernels32_[0].size()!=8*kernels_[0].size()||kernels32_[1].size()!=8*kernels_[1].size()))throw std::logic_error("Dirac mixed cache not prepared");
    auto qe=to_charges(h,&scale_.pp),qi=to_charges(h,&scale_.np);V e(size()),i(size());
    cauchy_charges_simd(0,qe.data(),e.data(),mixed);cauchy_charges_simd(1,qi.data(),i.data(),mixed);
#pragma omp parallel for
    for(int j=0;j<size();++j)out[j]=h[j]+scale_.p[j%8]*i[j]-scale_.n[j%8]*e[j];
}
void Transmission::self_blocks(C*out)const{diagonal_blocks(1,out);}
void Transmission::diagonal_blocks(int group,C*out)const{
    if(correction_only_||!dense_row_map_.empty()||kernels_[0].empty())throw std::logic_error("Dirac full self blocks require full dense cache");
    if(group<1||group>8)throw std::invalid_argument("Dirac block group must contain 1..8 panels");
    const size_t n=size()/8;const int panel_width=8*basis_count_,width=panel_width*group;
    const size_t blocks=(panels_.size()+group-1)/group;
#pragma omp parallel for schedule(static)
    for(size_t block=0;block<blocks;++block)for(int col=0;col<width;++col){
        const size_t p=block*group+col/panel_width;const int b=(col%panel_width)/8,c=col%8;
        Eight qe{},qi{};if(p<panels_.size()){Eight column{};column[c]=scale_.pp[c];qe=charges(panels_[p],column.data());
            column[c]=scale_.np[c];qi=charges(panels_[p],column.data());}
        for(int row_panel=0;row_panel<group;++row_panel)for(int a=0;a<basis_count_;++a){
            const size_t q=block*group+row_panel;Eight e{},i{};
            if(p<panels_.size()&&q<panels_.size()){const size_t offset=(q*basis_count_+a)*n+p*basis_count_+b;
                localize(panels_[q],action(kernels_[0][offset],qe.data()),e.data());
                localize(panels_[q],action(kernels_[1][offset],qi.data()),i.data());}
            for(int r=0;r<8;++r){const int row=row_panel*panel_width+8*a+r;
                out[(block*width+row)*width+col]=double(row==col)+scale_.p[r]*i[r]-scale_.n[r]*e[r];}
        }
    }
}
void Transmission::rhs(const C*f,C*b)const{for(int j=0;j<size();++j)b[j]=2.*scale_.n[j%8]*f[j];}
void Transmission::incident(const Vec3&d,const Vec3&e,C*out)const{
    if(std::abs(d.norm()-1)>1e-12||std::abs(d.dot(e))>1e-12)throw std::invalid_argument("Dirac incident geometry");
    const auto rule=tri_quadrature(13);const Vec3 h=d.cross(e);
    for(size_t j=0;j<panels_.size();++j){const auto&p=panels_[j];std::array<C,10> phase{};
        if(target_order_==1)phase[0]=std::exp(C(0,1)*wave_[0]*d.dot(p.center));
        else for(int q=0;q<rule.npts;++q){const Vec3 x=p.vertex[0]+(p.vertex[1]-p.vertex[0])*rule.pts[q][0]+(p.vertex[2]-p.vertex[0])*rule.pts[q][1];
            auto phi=polynomial(rule.pts[q][0],rule.pts[q][1]);C z=rule.wts[q]*std::exp(C(0,1)*wave_[0]*d.dot(x));for(int a=0;a<basis_count_;++a)phase[a]+=z*phi[a];}
        for(int a=0;a<basis_count_;++a){C z=phase[a];Eight f{{0.,z*h.x,z*h.y,z*h.z,0.,z*e.x,z*e.y,z*e.z}};localize(p,f,out+8*(j*basis_count_+a));}}
}
void Transmission::project_fields(const C*field,C*out)const{
    const auto rule=tri_quadrature(13);
    for(size_t j=0;j<panels_.size();++j)for(int a=0;a<basis_count_;++a){Eight sum{};
        for(int q=0;q<rule.npts;++q){auto phi=polynomial(rule.pts[q][0],rule.pts[q][1]);const double w=rule.wts[q]*phi[a];const C*f=field+6*(13*j+q);
            for(int c=0;c<3;++c){sum[1+c]+=w*f[3+c];sum[5+c]+=w*f[c];}}
        localize(panels_[j],sum,out+8*(j*basis_count_+a));}
}
void Transmission::near_correction(int medium,const C*h,C*out)const{
    if(!correction_only_||(medium!=0&&medium!=1))throw std::invalid_argument("Dirac sparse correction mode");
    const auto q=to_charges(h,nullptr);
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<panels_.size();++i)for(int a=0;a<basis_count_;++a){Eight sum{};
        for(size_t p=near_rows_[i];p<near_rows_[i+1];++p)for(int b=0;b<basis_count_;++b){
            const auto v=action(kernels_[medium][p*basis_count_*basis_count_+a*basis_count_+b],q.data()+8*(near_columns_[p]*basis_count_+b));
            for(int c=0;c<8;++c)sum[c]+=v[c];}
        localize(panels_[i],sum,out+8*(i*basis_count_+a));}
}
void Transmission::quadrature_charges(const C*h,C*out)const{
    const auto q=to_charges(h,nullptr);const auto rule=tri_quadrature(13);
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<panels_.size();++i)for(int t=0;t<13;++t){auto phi=polynomial(rule.pts[t][0],rule.pts[t][1]);Eight sum{};
        for(int b=0;b<basis_count_;++b)for(int c=0;c<8;++c)sum[c]+=phi[b]*q[8*(i*basis_count_+b)+c];
        for(int c=0;c<8;++c)out[8*(13*i+t)+c]=panels_[i].area*rule.wts[t]*sum[c];}
}
void Transmission::project_cauchy(const C*field,C*out)const{
    const auto rule=tri_quadrature(13);
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<panels_.size();++i)for(int a=0;a<basis_count_;++a){Eight sum{};
        for(int t=0;t<13;++t){auto phi=polynomial(rule.pts[t][0],rule.pts[t][1]);for(int c=0;c<8;++c)sum[c]+=rule.wts[t]*phi[a]*field[8*(13*i+t)+c];}
        localize(panels_[i],sum,out+8*(i*basis_count_+a));}
}
void Transmission::field(int medium,const C*h,const std::vector<Vec3>&x,C*out)const{
    if(medium!=0&&medium!=1)throw std::invalid_argument("medium");
    auto q=to_charges(h,medium?&scale_.np:&scale_.pp);
#pragma omp parallel for schedule(dynamic,1)
    for(size_t i=0;i<x.size();++i){Eight sum{};
        for(size_t j=0;j<panels_.size();++j){auto k=integrate(panels_[j],x[i],wave_[medium],within_near((x[i]-panels_[j].center).norm(),near_radius_,panels_[j].diameter));
            for(int b=0;b<basis_count_;++b){auto v=action(k[b],q.data()+8*(j*basis_count_+b));for(int a=0;a<8;++a)sum[a]+=.5*v[a];}}
        if(medium)for(int a=1;a<=3;++a)sum[a]*=index_;
        for(int a=0;a<8;++a)out[8*i+a]=sum[a];}
}
void Transmission::farfield(const C*h,const std::vector<Vec3>&directions,C*out)const{
    for(const auto&d:directions)if(!std::isfinite(d.norm2())||std::abs(d.norm2()-1.)>1e-10)throw std::invalid_argument("Dirac farfield direction must be unit length");
    auto q=to_charges(h,&scale_.pp);const auto rule=tri_quadrature(13);
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<directions.size();++i){const Vec3 d=directions[i];Eight sum{};
        for(size_t j=0;j<panels_.size();++j){const auto&p=panels_[j];std::array<C,10> g{};
            for(int t=0;t<rule.npts;++t){const Vec3 y=p.vertex[0]+(p.vertex[1]-p.vertex[0])*rule.pts[t][0]+(p.vertex[2]-p.vertex[0])*rule.pts[t][1];
                auto phi=polynomial(rule.pts[t][0],rule.pts[t][1]);C z=p.area*rule.wts[t]*INV4PI*std::exp(-C(0,1)*wave_[0]*d.dot(y));
                for(int b=0;b<basis_count_;++b)g[b]+=z*phi[b];}
            for(int b=0;b<basis_count_;++b){Kernel k;k.s=2.*C(0,1)*wave_[0]*g[b];k.d={{-d.x*k.s,-d.y*k.s,-d.z*k.s}};
                auto v=action(k,q.data()+8*(j*basis_count_+b));for(int a=0;a<8;++a)sum[a]+=.5*v[a];}}
        for(int a=0;a<8;++a)out[8*i+a]=sum[a];}
}
}
