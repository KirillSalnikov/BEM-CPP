#include "prism_d6h_model_action.h"
#include <cstdio>
#include <fstream>
using namespace bem_tmatrix;
static double relative(const Vector& x,const Vector& y){
    double a=0,b=0;for(size_t i=0;i<x.size();++i){a+=std::norm(x[i]-y[i]);b+=std::norm(y[i]);}
    return std::sqrt(a/std::max(b,1e-300));
}
static void dense(const Vector& a,const Complex* x,Complex* y,int n){
#pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){Complex sum=0;for(int j=0;j<n;++j)sum+=a[size_t(i)*n+j]*x[j];y[i]=sum;}
}
int main(int argc,char** argv)try {
    if(argc!=1 && argc!=3)throw std::invalid_argument("[REF PHYSICAL_A_FILE]");
    for(int ref:argc==1?std::vector<int>{0,1}:std::vector<int>{std::stoi(argv[1])}){
        MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        const int n=mesh.system_dofs();if(n>2000)throw std::invalid_argument("small dense oracle only");
        D6hOrbit orbit;orbit.build(mesh);PrismGroupMaps group;group.build(orbit);
        Vector a(size_t(n)*n),oracle(a.size()),x(n),actual(n),expected(n),rotated(n),other(n);
        if(argc==3){std::ifstream f(argv[2],std::ios::binary|std::ios::ate);
            if(!f || f.tellg()!=std::streamoff(a.size()*sizeof(Complex)))throw std::runtime_error("physical matrix size");
            f.seekg(0);if(!f.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(Complex)))throw std::runtime_error("physical matrix read");
        }else for(int i=0;i<n;++i)for(int j=0;j<n;++j)
            a[size_t(i)*n+j]=Complex(std::sin(.013*(i+1)*(j+2)),.7*std::cos(.023*(i+3)*(j+1)))/double(n);
        // Independent explicit matrix oracle, constructed from generators.
        std::vector<int> r,rs,h,hs,v,vs;
        orbit.image_map(false,false,r,rs);orbit.image_map(true,false,h,hs);
        D6hOrbit::vertical_image_map(mesh,v,vs);
        std::array<std::vector<int>,24> at,sgn;
        for(int vertical=0;vertical<2;++vertical)for(int horizontal=0;horizontal<2;++horizontal)
            for(int power=0;power<6;++power){const int g=12*vertical+6*horizontal+power;at[g].resize(n);sgn[g].resize(n);
                for(int i=0;i<n;++i){int j=i,s=1;
                    if(horizontal){s*=hs[j];j=h[j];}
                    for(int p=0;p<power;++p){s*=rs[j];j=r[j];}
                    if(vertical){s*=vs[j];j=v[j];}at[g][i]=j;sgn[g][i]=s;
                }
            }
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i)for(int j=0;j<n;++j){Complex sum=0;
            for(int g=0;g<24;++g){const int det=((g/12)?-1:1)*((g%12>=6)?-1:1);
                sum+=double(det*sgn[g][i]*sgn[g][j])*a[size_t(at[g][i])*n+at[g][j]];}
            oracle[size_t(i)*n+j]=sum/24.;
        }
        const Vector original=a;size_t actions=0;
        PrismD6hModelAction model(mesh,group,[&](const Complex* in,Complex* out){++actions;dense(a,in,out,n);});
        for(int i=0;i<n;++i)x[i]=Complex(std::sin(.031*i),std::cos(.043*i));
        model.apply(x.data(),actual.data());dense(oracle,x.data(),expected.data(),n);
        const double oracle_error=relative(actual,expected);double covariance=0;
        for(int g:{1,6,12}){
            for(int i=0;i<n;++i)rotated[at[g][i]]=double(sgn[g][i])*x[i];
            model.apply(rotated.data(),other.data());const int det=g==1?1:-1;
            for(int i=0;i<n;++i)expected[at[g][i]]=double(det*sgn[g][i])*actual[i];
            covariance=std::max(covariance,relative(other,expected));
        }
        other=x;model.apply(other.data(),other.data());const double alias=relative(other,actual);
        PrismD6hModelAction projected(mesh,group,[&](const Complex* in,Complex* out){dense(oracle,in,out,n);});
        projected.apply(x.data(),other.data());const double idempotence=relative(other,actual);
        if(oracle_error>2e-12 || covariance>2e-12 || alias>2e-12 || idempotence>2e-12)
            throw std::runtime_error("D6h matrix-free model accuracy/covariance/idempotence failed");
        if(actions!=120 || model.original_actions()!=120 || model.model_calls()!=5 || a!=original)
            throw std::runtime_error("D6h action accounting or original operator preservation failed");
        if(model.storage_bytes()!=size_t(n)*(3*sizeof(Complex)+sizeof(int)+sizeof(int8_t)))
            throw std::runtime_error("D6h model linear storage accounting failed");
        std::printf("ref=%d n=%d physical=%d oracle=%.5e D6h=%.5e alias=%.5e idempotence=%.5e bytes=%zu actions=%zu original_unchanged=true\n",
            ref,n,argc==3,oracle_error,covariance,alias,idempotence,model.storage_bytes(),actions);
    }
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
