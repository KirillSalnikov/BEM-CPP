#include "muller_radiation_actions.h"
#include "muller_dense.h"
#include <iostream>
#include <iomanip>
using C=std::complex<double>;using V=std::vector<C>;
double relative(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}return std::sqrt(x/std::max(y,1e-300));}
int main()try {
    double maximum=0;int cases=0;
    for(bool sphere:{false,true}){
        MullerP2BuildOptions opts;opts.edge_mode=sphere?MullerEdgeMode::Smooth:MullerEdgeMode::HDivBdm1;opts.project_edge_nodes_to_sphere=sphere;
        const auto mesh=build_muller_p2_mesh(sphere?icosphere(1,0):regular_prism(6,10./7,0,1,0,false,true),opts);
        std::vector<Vec3> d,e,f;
        for(int k=0;k<17;++k){const double z=-1+2.*k/16,phi=.731*k,rr=std::sqrt(std::max(0.,1-z*z));
            d.emplace_back(rr*std::cos(phi),rr*std::sin(phi),z);e.emplace_back(z*std::cos(phi),z*std::sin(phi),-rr);f.emplace_back(-std::sin(phi),std::cos(phi),0);}
        for(C wave:{C(.3,0),C(3,0),C(2,.2)}){
            MullerRadiationActions op(mesh,wave,d,e,f);const int n=op.dofs(),m=op.channels();
            for(int batch:{1,2,7}){V x(size_t(n)*batch),y(size_t(m)*batch),rx,rhy,oracle(size_t(m)*batch);
                for(size_t i=0;i<x.size();++i)x[i]=C(std::sin(.73*i),std::cos(.31*i));
                for(size_t i=0;i<y.size();++i)y[i]=C(std::cos(.13*i),std::sin(.21*i));
                op.apply(x,batch,rx);op.adjoint(y,batch,rhy);
                for(int h=0;h<batch;++h){V field;muller_nodal_farfield(mesh,x.data()+size_t(h)*n,x.data()+size_t(h)*n+n/2,wave,d,field,13);
                    for(int j=0;j<m;++j){const auto& t=j%2?f[j/2]:e[j/2];oracle[size_t(h)*m+j]=t.x*field[3*(j/2)]+t.y*field[3*(j/2)+1]+t.z*field[3*(j/2)+2];}}
                const double error=relative(rx,oracle);maximum=std::max(maximum,error);
                if(error>2e-12)throw std::runtime_error("independent farfield mismatch");
                C lhs=0,rhs=0;double normx=0,normy=0;
                for(size_t i=0;i<rx.size();++i){lhs+=std::conj(y[i])*rx[i];normy+=std::norm(y[i]);}
                for(size_t i=0;i<x.size();++i){rhs+=std::conj(rhy[i])*x[i];normx+=std::norm(x[i]);}
                const double dual=std::abs(lhs-rhs)/std::sqrt(normx*normy);maximum=std::max(maximum,dual);
                if(dual>2e-12)throw std::runtime_error("complex adjoint mismatch");
                auto inplace=x;op.apply(inplace,batch,inplace);if(relative(inplace,rx)!=0)throw std::runtime_error("forward alias");
                inplace=y;op.adjoint(inplace,batch,inplace);if(relative(inplace,rhy)!=0)throw std::runtime_error("adjoint alias");
                ++cases;}
        }
    }
    std::cout<<std::setprecision(17)<<"{\"passed\":true,\"cases\":"<<cases<<",\"maximum_error\":"<<maximum<<"}\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
