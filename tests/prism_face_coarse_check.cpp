#include "prism_face_coarse.h"
#include <cstdio>
using C=std::complex<double>;
using V=std::vector<C>;
double relative(const V& a,const V& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}return std::sqrt(x/std::max(y,1e-300));}
int main()try {
 for(double aspect:{1.,10./7.})for(int ref:{2,3,5}) {
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    auto mesh=build_muller_p2_mesh(regular_prism(6,aspect,ref,1.,0,false,true),options);
    PrismCirculantLayout layout;layout.build(mesh,ref);
    bem_tmatrix::D6hOrbit orbit;orbit.build(mesh);
    PrismFaceCoarseBasis basis(layout,orbit,mesh);
    const int n=basis.size(),r=basis.rank();V x(n),tx(n),expected(n),y(n);
    std::vector<double> norms(r,0.);
    for(int i=0;i<n;++i)if(basis.columns()[i]>=0)norms[basis.columns()[i]]+=basis.weights()[i]*basis.weights()[i];
    for(double z:norms)if(std::abs(z-1)>1e-11)throw std::runtime_error("face coarse normalization");
    // All three generators, on all aggregate columns; signed RHS and
    // solution representations differ by the reflection determinant.
    double maximum=0;
    for(int g:{1,6,12})for(int j=0;j<r;++j) {
        basis.generate(j,x.data());basis.transform(g,x.data(),tx.data(),false);
        basis.generate(basis.column_image(g,j),expected.data());
        for(auto& z:expected)z*=double(basis.column_sign(g,j));
        maximum=std::max(maximum,relative(tx,expected));
    }
    if(maximum>1e-13)throw std::runtime_error("face coarse span is not D6h invariant");
    if(ref<=3) {
        // Swapping electric/magnetic current spaces intertwines their
        // opposite mirror parities. Use it as an independent exact action.
        const auto exact=[&](const C* a,C* out){for(int i=0;i<n/2;++i){out[i]=C(2,.3)*a[i+n/2];out[i+n/2]=C(-.4,.7)*a[i];}};
        for(int j=0;j<r;++j){basis.generate(j,x.data());basis.sample(j,x.data(),y.data(),exact);exact(x.data(),expected.data());maximum=std::max(maximum,relative(y,expected));}
        if(maximum>1e-13 || basis.samples()!=basis.independent_columns())throw std::runtime_error("symmetry sample reconstruction");
        if(ref==2) {
            // Deliberately break A's symmetry. Stabilizer projection plus
            // orbit completion must still produce an equivariant coarse
            // approximation, even on rows outside the aggregate supports.
            const auto approximate=[&](const C* a,C* out){exact(a,out);for(int i=0;i<n;++i)out[i]+=C(.013,.007)*a[(17*i+3)%n];};
            V images(size_t(n)*r);
            for(int j=0;j<r;++j){basis.generate(j,x.data());basis.sample(j,x.data(),images.data()+size_t(j)*n,approximate);}
            for(int g:{1,6,12})for(int j=0;j<r;++j) {
                basis.transform(g,images.data()+size_t(j)*n,tx.data(),true);
                const int target=basis.column_image(g,j),sign=basis.column_sign(g,j);
                for(int i=0;i<n;++i)expected[i]=double(sign)*images[size_t(target)*n+i];
                maximum=std::max(maximum,relative(tx,expected));
            }
            if(maximum>1e-13)throw std::runtime_error("non-equivariant sample was not symmetrized");
        }
    }
    std::printf("face coarse aspect=%.8g ref=%d rank=%d samples=%d error=%.3e\n",aspect,ref,r,basis.independent_columns(),maximum);
 }
 return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
