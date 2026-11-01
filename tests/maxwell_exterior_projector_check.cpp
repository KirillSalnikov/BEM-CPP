#include "maxwell_exterior_projector.h"
#include <iomanip>
#include <iostream>
using C=std::complex<double>;using V=std::vector<C>;
double norm(const V& v){double s=0;for(auto z:v)s+=std::norm(z);return std::sqrt(s);}
V trace(const MullerP2Mesh& mesh,const MaxwellExteriorProjector& op,double k,bool dipole){
    const int n=mesh.current_dofs();V rhs(2*n);const Vec3 d(.36,.48,.8),polar(.8,-.6,0),h=d.cross(polar),p(.7,.2,.4);
    for(const auto& sample:op.points()){
        C E[3],H[3];const Vec3 x=sample.position;
        if(!dipole){const C phase=std::exp(C(0,k*d.dot(x)));const double e[]={polar.x,polar.y,polar.z},g[]={h.x,h.y,h.z};
            for(int a=0;a<3;++a){E[a]=phase*e[a];H[a]=phase*g[a];}}
        else {const double r=x.norm();const Vec3 u=x*(1/r),cross=u.cross(p);const double uv[]={u.x,u.y,u.z},pv[]={p.x,p.y,p.z},cv[]={cross.x,cross.y,cross.z};
            const C G=std::exp(C(0,k*r))/(4*std::acos(-1.)*r),a=-k*k-C(0,3*k/r)+3/(r*r),b=C(0,k/r)-1/(r*r);
            for(int j=0;j<3;++j){E[j]=C(0,k)*G*pv[j]+C(0,1/k)*G*(a*uv[j]*u.dot(p)+b*pv[j]);H[j]=G*(C(0,k)-1/r)*cv[j];}}
        const auto frame=evaluate_muller_frame(mesh,sample.element,sample.xi,sample.eta);const auto basis=evaluate_muller_basis(mesh,sample.element,frame);
        for(int i=0;i<basis.count;++i){const Vec3 t=basis.values[i].cross(sample.normal);int row=basis.dofs[i];
            rhs[row]-=sample.weight*(t.x*E[0]+t.y*E[1]+t.z*E[2]);rhs[n+row]+=sample.weight*(t.x*H[0]+t.y*H[1]+t.z*H[2]);}}
    return op.incident(rhs);
}
int main(int argc,char**argv)try{
    using T=MaxwellExteriorProjector::Triangle;T t;t.vertex={{Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)}};t.normal=Vec3(0,0,1);t.gxi=Vec3(1,0,0);t.geta=Vec3(0,1,0);t.jacobian=1;t.diameter=std::sqrt(2.);
    const Vec3 x(.2,.3,.4);double error=0;
    for(double k:{0.,1.7}){const auto a=MaxwellExteriorProjector::integrated_kernel(t,x,k,28);
        for(int axis=0;axis<3;++axis){Vec3 delta(axis==0?1e-5:0,axis==1?1e-5:0,axis==2?1e-5:0);
            const auto plus=MaxwellExteriorProjector::integrated_kernel(t,x+delta,k,28),minus=MaxwellExteriorProjector::integrated_kernel(t,x-delta,k,28);
            for(int p=0;p<3;++p)error=std::max(error,std::abs((plus.value[4*p]-minus.value[4*p])/(2e-5)-a.value[4*p+1+axis]));}}
    std::cout<<std::setprecision(17)<<"{\"kernel_gradient_absolute_error\":"<<error<<"}\n";
    if(error>1e-8)throw std::runtime_error("analytic singular gradient disagrees with independent potential derivative");
    int first=argc>1?std::stoi(argv[1]):0,last=argc>2?std::stoi(argv[2]):2;
    double previous[2]={1.,1.};
    for(int ref=first;ref<=last;++ref){MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        MaxwellExteriorProjector op(mesh,1.,7,10,2.);
        for(bool dipole:{false,true}){V input=trace(mesh,op,1.,dipole),result(input.size());
            op.apply(input.data(),result.data(),[&](const C*a,const C*b,C*p,C*g,C*q,C*h){op.direct_pair(a,b,p,g,q,h);});
            if(dipole)for(size_t i=0;i<result.size();++i)result[i]-=input[i];
            const double relative=norm(result)/norm(input);
            std::cout<<"{\"ref\":"<<ref<<",\"dofs\":"<<mesh.system_dofs()<<",\"field\":\""<<(dipole?"outgoing_dipole":"entire_plane_wave")
                     <<"\",\"relative_trace_error\":"<<relative<<",\"mass_residual\":"<<op.last_mass_residual()<<"}\n"<<std::flush;
            if(!std::isfinite(relative)||relative>.2)throw std::runtime_error("exterior trace physics check failed");
            if(ref==2&&relative>(dipole?.04:.006))throw std::runtime_error("refined exterior trace error too large");
            if(ref==2&&first<2&&relative>.55*previous[dipole])throw std::runtime_error("exterior trace does not converge with refinement");
            previous[dipole]=relative;
        }
    }
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
