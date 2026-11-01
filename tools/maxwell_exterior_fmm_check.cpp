#include "maxwell_exterior_fmm.h"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv)try{
    using C=std::complex<double>;using V=std::vector<C>;
    const int ref=argc>1?std::stoi(argv[1]):0;
    const double ka=argc>2?std::stod(argv[2]):3.;
    if(ref<0||ref>4||ka<=0)throw std::invalid_argument("unsupported check size");
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    // Radius one forces nonempty far interactions even on this small mesh.
    MullerFmmOperator op;op.init(regular_prism(6,10./7,ref,1.,0,false,true),ka,1.3116,options,7,6,5,64,false,2,0,.96,nullptr,ref?3:1);
    MaxwellExteriorProjector projection(op.mesh,ka,7,10,2.);MaxwellExteriorFmm fmm(op,projection);
    // On the larger mesh compare potentials AND gradients at distributed
    // targets against all source points, without an O(N^2) full reference.
    if(ref>0){const auto& points=projection.points();const size_t np=points.size();
        V a(np),b(np),p(np),q(np),g(3*np),h(3*np);
        for(size_t s=0;s<np;++s){a[s]=points[s].weight*C(std::cos(.71*s),std::sin(.37*s));
            b[s]=points[s].weight*std::exp(C(0,ka*(.36*points[s].position.x+.48*points[s].position.y+.8*points[s].position.z)));}
        fmm(a.data(),b.data(),p.data(),g.data(),q.data(),h.data());
        double error[2]={0,0},norm[2]={0,0};
        for(int sample=0;sample<32;++sample){const size_t t=size_t(sample)*(np-1)/31;C direct[2][4]{};
            for(size_t s=0;s<np;++s){const Vec3 r=points[t].position-points[s].position;const double d=r.norm();if(d<1e-14)continue;
                const C potential=std::exp(C(0,ka*d))/(4*std::acos(-1.)*d),gradient=potential*(C(0,ka*d)-1.)/(d*d);
                for(int channel=0;channel<2;++channel){const C charge=channel?b[s]:a[s];direct[channel][0]+=potential*charge;
                    direct[channel][1]+=gradient*r.x*charge;direct[channel][2]+=gradient*r.y*charge;direct[channel][3]+=gradient*r.z*charge;}}
            for(int channel=0;channel<2;++channel)for(int j=0;j<4;++j){const C observed=j?(channel?h[3*t+j-1]:g[3*t+j-1]):(channel?q[t]:p[t]);
                const int kind=j?1:0;error[kind]+=std::norm(observed-direct[channel][j]);norm[kind]+=std::norm(direct[channel][j]);}}
        for(int j=0;j<2;++j)error[j]=std::sqrt(error[j]/norm[j]);
        std::cout<<std::setprecision(17)<<"{\"ref\":"<<ref<<",\"ka\":"<<ka<<",\"banded\":"<<(op.banded_fmm?"true":"false")
                 <<",\"sampled_targets\":32,\"source_points\":"<<np<<",\"potential_relative_error\":"<<error[0]<<",\"gradient_relative_error\":"<<error[1]<<"}\n";
        op.cleanup();if(!std::isfinite(error[0])||!std::isfinite(error[1])||std::max(error[0],error[1])>1e-5)
            throw std::runtime_error("sampled exterior FMM/direct mismatch");
        return 0;
    }
    const int n=op.system_dofs;V x(n),cpu(n),gpu(n);for(int i=0;i<n;++i)x[i]=C(std::cos(i*.71),std::sin(i*.37));
    projection.apply(x.data(),cpu.data(),[&](const C*a,const C*b,C*p,C*g,C*q,C*h){projection.direct_pair(a,b,p,g,q,h);});
    projection.apply(x.data(),gpu.data(),[&](const C*a,const C*b,C*p,C*g,C*q,C*h){fmm(a,b,p,g,q,h);});
    double error=0,norm=0;for(int i=0;i<n;++i){error+=std::norm(cpu[i]-gpu[i]);norm+=std::norm(cpu[i]);}
    error=std::sqrt(error/norm);std::cout<<std::setprecision(17)<<"{\"banded\":"<<(op.banded_fmm?"true":"false")<<",\"projector_fmm_vs_direct_relative_error\":"<<error<<"}\n";
    op.cleanup();if(!std::isfinite(error)||error>1e-5)throw std::runtime_error("exterior projector FMM/direct mismatch");return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
