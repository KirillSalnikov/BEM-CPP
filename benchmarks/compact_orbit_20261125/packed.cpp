// CPU reproduction of the warm-start gate on archived physical Muller RHS.
#include "prism_sector_solver.h"
#include "prism_orbit_inverse.h"
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace bem_tmatrix;

static Vector read_vector(const std::string& path, size_t count) {
    Vector values(count);
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(values.data()), count*sizeof(Complex)))
        throw std::runtime_error("cannot read " + path);
    return values;
}

int main(int argc, char** argv) try {
    if (argc!=7) throw std::invalid_argument("warm_trial CASE EXACT_CURRENT_DIR");
    const std::string dir=argv[1], exact=argv[2];
    MullerP2BuildOptions options;
    options.edge_mode=MullerEdgeMode::HDivBdm1;
    auto mesh=build_muller_p2_mesh(regular_prism(6,1.6495721976846451,std::stoi(argv[5]),1.,0,false,true),options);
    D6hOrbit group; group.build(mesh);
    PrismGroupMaps maps; maps.build(group);
    PrismSymmetrySectors sectors; sectors.build(maps);
    const int n=sectors.size();
    const Vector a=read_vector(dir+"/A.bin",size_t(n)*n);
    PrismOrbitInverse block;
    block.build(mesh,maps,std::stoi(argv[3]),[&](const std::vector<int>& edges) {
        std::vector<int> idx;
        for(int e:edges) for(int c=0;c<4;++c) idx.push_back((c/2)*(n/2)+2*e+c%2);
        Vector sub(idx.size()*idx.size());
        for(size_t i=0;i<idx.size();++i)
            for(size_t j=0;j<idx.size();++j)
                sub[i*idx.size()+j]=a[size_t(idx[i])*n+idx[j]];
        return sub;
    },std::stoi(argv[4]),std::stoi(argv[6]));
    auto apply=[&](const Complex* x, Complex* y) {
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i) {
            Complex s=0;
            for(int j=0;j<n;++j) s+=a[size_t(i)*n+j]*x[j];
            y[i]=s;
        }
    };
    auto inverse=[&](const Complex* x, Complex* y){block.apply(x,y);};
    Vector v(n),w(n),u(n),z(n);
    for(int j=0;j<n;++j){v[j]=Complex(std::cos(.23*j),std::sin(.41*j));w[j]=Complex(std::sin(.19*j),std::cos(.31*j));}
    block.apply(v.data(),u.data());block.apply_adjoint(w.data(),z.data());
    Complex left=0,right=0;
    for(int i=0;i<n;++i){left+=std::conj(w[i])*u[i];right+=std::conj(z[i])*v[i];}
    if(std::abs(left-right)>1e-11*bem_streaming::norm(w.data(),n)*bem_streaming::norm(u.data(),n))
        throw std::runtime_error("adjoint error");
    const char* audit_path=std::getenv("BEM_BENCH_AUDIT_CURRENT");
    const double tol=std::getenv("BEM_BENCH_TOL")?std::stod(std::getenv("BEM_BENCH_TOL")):2e-6;
    for(int degree:{10,15,20,25}){
        if(audit_path && degree!=20)continue;
        Vector b=read_vector(exact+"/p"+std::to_string(degree)+".bin",n),x(n),image(n);
        auto r=solve_prism_sectors(apply,apply,inverse,sectors,b.data(),x.data(),64,tol,400,false,false);
        apply(x.data(),image.data());
        for(int j=0;j<n;++j)image[j]-=b[j];
        double res=bem_streaming::norm(image.data(),n)/bem_streaming::norm(b.data(),n);
        if(!(res<=tol))throw std::runtime_error("independent physical gate");
        if(audit_path){
            std::ofstream f(audit_path,std::ios::binary);
            f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(Complex));
            if(!f)throw std::runtime_error("failed to write audit current");
        }
        std::cout<<argv[3]<<','<<argv[4]<<','<<argv[6]<<','<<degree<<','<<r.iterations<<','<<r.strict_checks<<','
                 <<res<<','<<r.seconds<<','<<block.storage_bytes()<<','<<block.setup_seconds<<std::endl;
    }
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
