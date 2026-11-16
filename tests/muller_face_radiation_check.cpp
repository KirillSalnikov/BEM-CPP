#include "muller_face_radiation.h"
#include "muller_dense.h"
#include <iostream>
#include <iomanip>
using C=std::complex<double>;using V=std::vector<C>;
double rel(const V&a,const V&b){double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}return std::sqrt(e/std::max(n,1e-300));}
int main()try {
    std::vector<Vec3> directions;for(int i=0;i<=60;++i){double z=-1+2.*i/60,phi=.713*i,r=std::sqrt(std::max(0.,1-z*z));directions.emplace_back(r*std::cos(phi),r*std::sin(phi),z);}
    int cases=0,rejected=0;double worst=0;
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    for(int shape=0;shape<4;++shape){auto linear=regular_prism(shape==3?4:6,shape==1?.4:1.43,0,1.,0,false,true);
        if(shape==2){const Vec3 axis=Vec3(.3,.5,.8)*(1/std::sqrt(.98));const double a=.49;
            for(auto& p:linear.verts){p=p*std::cos(a)+axis.cross(p)*std::sin(a)+axis*(axis.dot(p)*(1-std::cos(a)));p.x=-p.x;p=p+Vec3(1.1,-.7,.3);}
            for(int t=0;t<linear.nt();++t)std::swap(linear.tris[3*t+1],linear.tris[3*t+2]);}
        if(shape==3){ // Dented polyhedral surface: no convexity or C6 assumed.
            for(auto& p:linear.verts)if(p.x>0&&p.z>0)p.x*=.2;
        }
        const auto mesh=build_muller_p2_mesh(linear,options);const int n=mesh.system_dofs();V x(n),y(n);
        for(int j=0;j<n;++j){x[j]=C(std::sin(.73*j),std::cos(.19*j));y[j]=C(std::cos(.53*j),std::sin(.37*j));}
        for(int quadrature:{7,13})for(double k:{3.,20.,100.})for(bool parallel:{false,true}){
            MullerFaceRadiation plan(mesh,k,directions,1e-10,parallel?4:1,quadrature,128,parallel);
            V a,b,aa,bb;plan.apply_pair(x.data(),y.data(),a,b);
            muller_nodal_farfield_pair(mesh,x.data(),x.data()+n/2,y.data(),y.data()+n/2,k,directions,aa,bb,quadrature);
            worst=std::max(worst,std::max(rel(a,aa),rel(b,bb)));if(worst>2e-8)throw std::runtime_error("full independent field error");
            V xx=x,yy=y;plan.apply_pair(xx.data(),yy.data(),xx,yy);
            if(rel(xx,a)>1e-14||rel(yy,b)>1e-14)throw std::runtime_error("in-place field error");
            V zero(n);plan.apply_pair(zero.data(),zero.data(),xx,yy);for(auto c:xx)if(c!=C(0))throw std::runtime_error("zero not exact");
            plan.apply_pair(y.data(),x.data(),xx,yy);if(rel(xx,b)>1e-14||rel(yy,a)>1e-14)throw std::runtime_error("stale reused plan");
            ++cases;
        }
    }
    auto mesh=build_muller_p2_mesh(regular_prism(6,1.43,0,1.,0,false,true),options);
    for(int failure=0;failure<6;++failure){auto bad=mesh;auto d=directions;C k=20;int faces=128;
        if(failure==0)k=C(20,.1);
        if(failure==1)bad.nodes[bad.elements[0].nodes[3]].x+=.03;
        if(failure==2)bad.basis_kind=MullerBasisKind::NodalP2;
        if(failure==3)d[0]=d[0]*2;
        if(failure==4)faces=2;
        if(failure==5)k=0.;
        try{MullerFaceRadiation plan(bad,k,d,1e-10,1,13,faces);}catch(const std::invalid_argument&){++rejected;continue;}
        throw std::runtime_error("unsupported geometry/media accepted");
    }
    std::cout<<std::setprecision(17)<<"{\"passed\":true,\"cases\":"<<cases<<",\"rejected\":"<<rejected<<",\"max_field_error\":"<<worst<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
