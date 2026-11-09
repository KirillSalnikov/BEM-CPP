#include "regular_face_fft.h"
#include "quadrature.h"
#ifdef BEM_FACE_FMM_COMPARE
#include "fmm.h"
#endif
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

using Complex=std::complex<double>;
using Clock=std::chrono::steady_clock;
struct Case {
    int nx,ny,c;
    std::array<double,3> u,v;
    std::vector<double> source,target;
    std::vector<Complex> q;
    std::vector<int> active;
    Complex k;
};

std::array<double,3> position(const Case& a,int point,bool target) {
    const int cells=a.nx*a.ny,ch=point/cells,cell=point%cells;
    std::array<double,3> p;
    for(int d=0;d<3;++d)p[d]=(target?a.target:a.source)[3*ch+d]+(cell/a.ny)*a.u[d]+(cell%a.ny)*a.v[d];
    return p;
}
std::array<Complex,6> direct(const Case& a,int target) {
    const auto tp=position(a,target,true);
    std::array<Complex,6> out{};
    const Complex imaginary(0,1);
    const int cells=a.nx*a.ny;
    for(int source:a.active) {
        const auto sp=position(a,source,false);
        const double rvec[3]={tp[0]-sp[0],tp[1]-sp[1],tp[2]-sp[2]};
        const double r2=rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2];
        if(r2<=1e-28)continue;
        const double r=std::sqrt(r2),ir=1/r;
        const Complex g=std::exp(imaginary*a.k*r)*(ir/(4*std::acos(-1.)));
        const Complex radial=g*(imaginary*a.k-ir)*ir;
        Complex grad[3],h[3][3],q[3];
        for(int i=0;i<3;++i) {
            grad[i]=radial*rvec[i];
            q[i]=a.q[size_t((source/cells)*3+i)*cells+source%cells];
            for(int j=0;j<3;++j) {
                h[i][j]=g*(3*ir*ir-3.*imaginary*a.k*ir-a.k*a.k)*rvec[i]*rvec[j]/r2;
                if(i==j)h[i][j]-=g*(ir*ir-imaginary*a.k*ir);
            }
        }
        out[0]+=grad[1]*q[0]-grad[0]*q[1];
        out[1]+=grad[2]*q[0]-grad[0]*q[2];
        out[2]+=grad[2]*q[1]-grad[1]*q[2];
        const Complex trace=h[0][0]+h[1][1]+h[2][2];
        for(int i=0;i<3;++i) {for(int j=0;j<3;++j)out[3+i]+=h[i][j]*q[j];out[3+i]-=trace*q[i];}
    }
    return out;
}

double validate(const Case& a,const std::vector<Complex>& fft,int maximum_targets) {
    const int cells=a.nx*a.ny;
    double difference=0,norm=0;
    const int n=std::min(int(a.active.size()),maximum_targets);
    #pragma omp parallel for reduction(+:difference,norm) schedule(static)
    for(int index=0;index<n;++index) {
        const int t=a.active[size_t(index)*a.active.size()/n];
        const auto expected=direct(a,t);
        for(int d=0;d<6;++d) {
            const Complex actual=fft[size_t((t/cells)*6+d)*cells+t%cells];
            difference+=std::norm(actual-expected[d]);norm+=std::norm(expected[d]);
        }
    }
    return std::sqrt(difference/std::max(norm,1e-300));
}

Case prism_face(int ref,Complex k,double aspect=1.,bool cap=false) {
    if(ref<1||ref>7)throw std::runtime_error("ref must be 1..7");
    if(!std::isfinite(aspect)||aspect<=0)throw std::runtime_error("invalid prism aspect");
    Case a;a.nx=1<<ref;a.ny=std::max(a.nx,int(std::ceil(std::sqrt(3.)*aspect*a.nx)));if(a.ny%2)++a.ny;
    a.c=cap?14:28;a.k=k;
    // regular_prism(6,aspect,ref,1,...,d6h=true): volume-equivalent radius 1.
    const double scale=std::cbrt((4*std::acos(-1.)/3)/(aspect*std::sqrt(3.)/2));
    const double side=scale/std::sqrt(3.),height=aspect*scale;
    a.u={{side/a.nx,0,0}};a.v={{0,height/a.ny,0}};
    if(cap){a.nx=a.ny=2*(1<<ref);a.v={{.5*a.u[0],std::sqrt(3.)*.5*a.u[0],0}};}
    const TriQuad rule=tri_quadrature(7);
    const double vertices[4][3][2]={{{0,0},{1,0},{1,1}},{{0,0},{1,1},{0,1}},
                                   {{0,0},{1,0},{0,1}},{{1,0},{1,1},{0,1}}};
    std::vector<std::array<double,2>> fractional;
    for(int tri=cap?2:0;tri<4;++tri)for(int j=0;j<7;++j) {
        const double x=rule.pts[j][0],y=rule.pts[j][1];
        const double u=(1-x-y)*vertices[tri][0][0]+x*vertices[tri][1][0]+y*vertices[tri][2][0];
        const double v=(1-x-y)*vertices[tri][0][1]+x*vertices[tri][1][1]+y*vertices[tri][2][1];
        fractional.push_back({{u,v}});
        for(int d=0;d<3;++d)a.source.push_back(u*a.u[d]+v*a.v[d]);
    }
    a.target=a.source;
    const int cells=a.nx*a.ny;a.q.resize(size_t(cells)*a.c*3);
    std::mt19937 random(20261028);std::uniform_real_distribution<double> uniform(-1,1);
    for(int ch=0;ch<a.c;++ch)for(int x=0;x<a.nx;++x)for(int y=0;y<a.ny;++y) {
        const bool forward=(2*x<a.nx)==(2*y<a.ny);
        if(cap) {
            const double diagonal=x+y+fractional[ch][0]+fractional[ch][1];
            if(diagonal<(1<<ref) || diagonal>3*(1<<ref))continue;
        }else if((ch<14)!=forward)continue;
        const int cell=x*a.ny+y;a.active.push_back(ch*cells+cell);
        for(int d=0;d<3;++d)a.q[size_t(ch*3+d)*cells+cell]=Complex(uniform(random),uniform(random))*.5*a.u[0]*a.v[1]*rule.wts[ch%7];
    }
    return a;
}

double validate_transpose(const Case& a,RegularFaceFFT& fft) {
    const int cells=a.nx*a.ny,n=cells*a.c;
    std::vector<Complex> seed(size_t(n)*6),expected(size_t(n)*3),actual(expected.size());
    for(size_t i=0;i<seed.size();++i)seed[i]=Complex(std::sin(.173*(i+1)),std::cos(.127*(i+1)));
    #pragma omp parallel for schedule(static)
    for(int s=0;s<n;++s)for(int t=0;t<n;++t) {
        const auto sp=position(a,s,false),tp=position(a,t,true);
        double r[3];double r2=0;for(int d=0;d<3;++d){r[d]=tp[d]-sp[d];r2+=r[d]*r[d];}
        if(r2<=1e-28)continue;
        const double radius=std::sqrt(r2),ir=1./radius;
        const Complex ik=Complex(0,1)*a.k,g=std::exp(ik*radius)*(ir/(4*std::acos(-1.)));
        const Complex first=g*(ik-ir),second=g*(-a.k*a.k-2.*ik*ir+2.*ir*ir);
        Complex grad[3],h[3][3],trace=0,m[6][3]{};
        for(int d=0;d<3;++d){
            grad[d]=first*r[d]*ir;
            for(int j=0;j<3;++j)h[d][j]=(second-first*ir)*r[d]*r[j]/r2+(d==j?first*ir:Complex(0));
            trace+=h[d][d];
        }
        m[0][0]=grad[1];m[0][1]=-grad[0];m[1][0]=grad[2];m[1][2]=-grad[0];m[2][1]=grad[2];m[2][2]=-grad[1];
        for(int d=0;d<3;++d)for(int j=0;j<3;++j)m[3+d][j]=h[d][j]-(d==j?trace:Complex(0));
        for(int d=0;d<3;++d)for(int j=0;j<6;++j)
            expected[size_t(s/cells*3+d)*cells+s%cells]+=m[j][d]*seed[size_t(t/cells*6+j)*cells+t%cells];
    }
    const auto bytes=fft.storage_bytes();fft.evaluate_transpose(seed.data(),actual.data());
    double nrm=0,diff=0;for(size_t i=0;i<actual.size();++i){nrm+=std::norm(expected[i]);diff+=std::norm(actual[i]-expected[i]);}
    const double error=std::sqrt(diff/std::max(nrm,1e-300));
    if(!std::isfinite(error)||error>2e-11)throw std::runtime_error("regular face transpose versus direct Green derivatives");
    auto alias=seed;fft.evaluate_transpose(alias.data(),alias.data());
    diff=0;nrm=0;for(size_t i=0;i<actual.size();++i){diff+=std::norm(alias[i]-actual[i]);nrm+=std::norm(actual[i]);}
    if(std::sqrt(diff/std::max(nrm,1e-300))>1e-14)throw std::runtime_error("regular face transpose alias mismatch");
    std::fill(seed.begin(),seed.end(),Complex(0));fft.evaluate_transpose(seed.data(),actual.data());
    for(auto x:actual)if(std::abs(x)>1e-14)throw std::runtime_error("regular face transpose stale buffer");
    if(bytes!=fft.storage_bytes())throw std::runtime_error("transpose grew GPU workspace");
    return error;
}

void unit_checks() {
    for(int test=0;test<6;++test) {
        Case a=prism_face(1,Complex(2.3,.08),1.,test==5);
        if(test==1) {for(size_t i=2;i<a.target.size();i+=3)a.target[i]+=.63;}
        if(test==2) {a.k=Complex(0);a.u[2]=.13;a.v[0]=.07;}
        if(test==3) {a.k=Complex(0,.4);std::fill(a.q.begin(),a.q.end(),Complex(0));a.q[0]=1.;a.active.clear();for(int i=0;i<a.nx*a.ny*a.c;++i)a.active.push_back(i);}
        if(test==4) {
            // Almost equal shifts next to lattice coincidences must not be
            // merged: their O(r^-3) interactions are sensitive to roundoff.
            a.nx=3;a.ny=5;a.c=3;a.u={{1,0,0}};a.v={{0,1,0}};
            a.source={0,0,0,1+1e-10,0,0,2+2e-10+1e-15,0,0};a.target=a.source;
            a.active.clear();a.q.resize(size_t(a.nx)*a.ny*a.c*3);
            for(size_t i=0;i<a.q.size();++i)a.q[i]=Complex(std::sin(.13*i),std::cos(.17*i));
            for(int i=0;i<a.nx*a.ny*a.c;++i)a.active.push_back(i);
        }
        RegularFaceFFT fft;fft.init(a.nx,a.ny,a.u.data(),a.v.data(),a.source,a.target,a.k);
        std::vector<Complex> result(size_t(a.nx)*a.ny*a.c*6);fft.evaluate(a.q.data(),result.data());
        const double error=validate(a,result,int(a.active.size()));
        const auto dimensions=fft.padded_dimensions();
        std::printf("case=%d relative=%.8e kernels=%d grid=%dx%d bytes=%zu\n",test,error,
                    fft.stored_kernel_count(),dimensions[0],dimensions[1],fft.storage_bytes());
        if(!std::isfinite(error)||error>2e-11)throw std::runtime_error("regular face FFT disagrees with direct FP64");
        const double transpose_error=validate_transpose(a,fft);
        fft.evaluate(a.q.data(),result.data());
        if(validate(a,result,int(a.active.size()))>2e-11)throw std::runtime_error("forward FFT after transpose changed");
        std::printf("case=%d transpose_direct_relative=%.17g alias_zero_workspace=PASS\n",test,transpose_error);
    }
    bool rejected=false;try {RegularFaceFFT::estimated_array_bytes(0,1,1);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid dimensions accepted");
}

#ifdef BEM_FACE_FMM_COMPARE
void compare_fmm(const Case& a,int ref,const std::vector<Complex>& reference,std::ostream& report) {
    const int n=int(a.active.size()),cells=a.nx*a.ny;
    std::vector<double> xyz(size_t(n)*3);
    std::vector<Complex> q(size_t(n)*3),curl(q.size()),hessian(q.size());
    for(int i=0;i<n;++i) {
        const auto x=position(a,a.active[i],false);
        for(int d=0;d<3;++d) {
            xyz[3*i+d]=x[d];
            q[size_t(d)*n+i]=a.q[size_t((a.active[i]/cells)*3+d)*cells+a.active[i]%cells];
        }
    }
    const int depth=std::min(6,std::max(3,ref-1));
    auto leaf=[&](int d){const long long boxes=1LL<<(3*d);return std::max(1,int((2LL*n+boxes-1)/boxes));};
    const auto start=Clock::now();
    HelmholtzFMM fine,middle,coarse;
    fine.configure_interaction_band(depth,depth,true);fine.configure_order_reference_depth(depth);
    middle.configure_interaction_band(depth-1,depth-1,false);middle.configure_order_reference_depth(depth);
    coarse.configure_interaction_band(1,depth-2,false);coarse.configure_order_reference_depth(std::max(1,depth-3));
    fine.init(xyz.data(),n,xyz.data(),n,a.k,5,leaf(depth),3,false,false,true);
    middle.init(xyz.data(),n,xyz.data(),n,a.k,5,leaf(depth-1),3,false,false,true);
    coarse.init(xyz.data(),n,xyz.data(),n,a.k,5,leaf(depth-2),3,false,false,true);
    const double setup=std::chrono::duration<double>(Clock::now()-start).count();
    auto evaluate=[&] {
        const auto begin=Clock::now();
        fine.evaluate_vector_actions_batch3(q.data(),q.data()+n,q.data()+2LL*n,curl.data(),hessian.data(),&middle,&coarse);
        return std::chrono::duration<double>(Clock::now()-begin).count();
    };
    auto error=[&] {
        double delta=0,norm=0;
        for(int i=0;i<n;++i)for(int d=0;d<6;++d) {
            const int slot=a.active[i];
            const Complex expected=reference[size_t((slot/cells)*6+d)*cells+slot%cells];
            const Complex actual=d<3?curl[3*i+d]:hessian[3*i+d-3];
            delta+=std::norm(actual-expected);norm+=std::norm(expected);
        }
        return std::sqrt(delta/norm);
    };
    report<<",\"FMM_comparison\":{\"setup_s\":"<<setup<<",\"fine_depth\":"<<depth;
    for(int strict=0;strict<2;++strict) {
        fine.near_field_fp32=middle.near_field_fp32=coarse.near_field_fp32=!strict;
        evaluate();report<<(strict?",\"strict\":{":",\"mixed\":{")<<"\"action_s\":[";
        for(int repeat=0;repeat<3;++repeat){const double seconds=evaluate();report<<(repeat?",":"")<<seconds;std::printf("FMM strict=%d seconds=%.6f\n",strict,seconds);}
        const double relative=error();report<<"],\"relative_error_vs_FFT\":"<<relative<<"}";
        if(!std::isfinite(relative)||relative>1e-3)throw std::runtime_error("FMM comparison accuracy is insufficient");
    }
    report<<"}";
}
#endif

int main(int argc,char** argv)try {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc==2 && std::string(argv[1])=="--check") {unit_checks();return 0;}
    if(argc!=4)throw std::runtime_error("Usage: regular_face_fft_probe REF KA REPORT.json, or --check");
    const int ref=std::stoi(argv[1]);const double ka=std::stod(argv[2]);
    const char* aspect_setting=std::getenv("BEM_PROBE_PRISM_ASPECT");
    const double aspect=aspect_setting?std::stod(aspect_setting):1.;
    const char* face_setting=std::getenv("BEM_PROBE_PRISM_FACE");
    const std::string face=face_setting?face_setting:"side";
    if(face!="side" && face!="cap")throw std::runtime_error("face must be side or cap");
    Case a=prism_face(ref,Complex(ka),aspect,face=="cap");
    std::printf("nx=%d ny=%d channels=%d active_points=%zu dense_array_upper_bound_GiB=%.4f\n",a.nx,a.ny,a.c,a.active.size(),
                RegularFaceFFT::estimated_array_bytes(a.nx,a.ny,a.c)/std::pow(2.,30));
    const auto start=Clock::now();RegularFaceFFT fft;fft.init(a.nx,a.ny,a.u.data(),a.v.data(),a.source,a.target,a.k);
    const double setup=std::chrono::duration<double>(Clock::now()-start).count();
    std::vector<Complex> result(size_t(a.nx)*a.ny*a.c*6);
    fft.evaluate(a.q.data(),result.data());std::vector<double> seconds;
    for(int repeat=0;repeat<4;++repeat) {const auto begin=Clock::now();fft.evaluate(a.q.data(),result.data());seconds.push_back(std::chrono::duration<double>(Clock::now()-begin).count());}
    const auto begin=Clock::now();const int checked=std::min(64,int(a.active.size()));
    const double error=validate(a,result,checked);
    const double check_s=std::chrono::duration<double>(Clock::now()-begin).count();
    const size_t storage=fft.storage_bytes();
    const int stored_kernels=fft.stored_kernel_count();const auto dimensions=fft.padded_dimensions();
    fft.cleanup(); // FMM benchmark must not coexist with the large FFT kernel bank.
    std::ofstream report(argv[3]);if(!report)throw std::runtime_error("cannot open report");
    report<<std::setprecision(17)<<"{\"scope\":\"one prism face, point-quadrature vector action, no Galerkin correction or BEM solve\",\"face\":\""<<face<<"\",\"ref\":"<<ref
          <<",\"ka\":"<<ka<<",\"nx\":"<<a.nx<<",\"ny\":"<<a.ny<<",\"channels\":"<<a.c<<",\"active_points\":"<<a.active.size()
          <<",\"prism_aspect\":"<<aspect<<",\"stored_kernels\":"<<stored_kernels<<",\"FFT_dimensions\":["<<dimensions[0]<<","<<dimensions[1]<<"]"
          <<",\"mirror_diagonal_mask\":true,\"precision\":\"fp64\",\"storage_bytes\":"<<storage<<",\"setup_s\":"<<setup<<",\"action_s\":[";
    for(size_t i=0;i<seconds.size();++i)report<<(i?",":"")<<seconds[i];
    report<<"],\"checked_targets\":"<<checked<<",\"relative_error_vs_direct\":"<<error<<",\"direct_check_s\":"<<check_s;
    std::printf("setup_s=%.6f actions=%.6f,%.6f,%.6f,%.6f error=%.8e check_s=%.6f storage_GiB=%.4f\n",setup,seconds[0],seconds[1],seconds[2],seconds[3],error,check_s,storage/std::pow(2.,30));
#ifdef BEM_FACE_FMM_COMPARE
    compare_fmm(a,ref,result,report);
#endif
    report<<"}\n";
    return std::isfinite(error)&&error<1e-9?0:2;
}catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
