#include "prism_sparse_inverse.h"
#include "prism_vertical_inverse.h"
#include <cstdio>
#include <omp.h>
using namespace bem_tmatrix;
using bem_spai::least_squares;
static void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
static double error(const Vector& a,const Vector& b){double x=0,y=0;for(size_t i=0;i<a.size();++i){x+=std::norm(a[i]-b[i]);y+=std::norm(b[i]);}return std::sqrt(x/std::max(y,1e-300));}
int main()try{
    omp_set_num_threads(2);
    // Pivoting, complex transpose convention, exact rectangular solves, and
    // independent first-order optimality of an inconsistent least-squares fit.
    const int m=31,n=12,h=4;Vector matrix(m*n),x(n*h),b(m*h);
    for(int j=0;j<n;++j)for(int i=0;i<m;++i)matrix[j*m+i]=Complex(.03*std::sin(.7*i+j),.1*std::cos(.3*i-.9*j))+(i==j?Complex(1.+j,.2):0.);
    for(int j=0;j<n*h;++j)x[j]=Complex(std::sin(.21*j),std::cos(.33*j));
    for(int k=0;k<h;++k)for(int i=0;i<m;++i)for(int j=0;j<n;++j)b[k*m+i]+=matrix[j*m+i]*x[k*n+j];
    int rank;auto solved=least_squares(matrix,m,n,b,h,rank);require(rank==n&&error(solved,x)<2e-13,"rectangular complex QR solve");
    for(int i=0;i<m*h;++i)b[i]+=Complex(.2*std::sin(.51*i),.1*std::cos(.27*i));
    solved=least_squares(matrix,m,n,b,h,rank);double stationarity=0;
    for(int k=0;k<h;++k){Vector residual(m);for(int i=0;i<m;++i){residual[i]=-b[k*m+i];for(int j=0;j<n;++j)residual[i]+=matrix[j*m+i]*solved[k*n+j];}
        for(int j=0;j<n;++j){Complex value=0;for(int i=0;i<m;++i)value+=std::conj(matrix[j*m+i])*residual[i];stationarity=std::max(stationarity,std::abs(value));}}
    require(stationarity<2e-12,"least-squares stationarity");
    for(int i=0;i<m;++i)matrix[(n-1)*m+i]=matrix[i];
    solved=least_squares(matrix,m,n,b,h,rank);require(rank==n-1,"dependent QR column not detected");
    std::printf("QR exact solve, least-squares stationarity %.3e, rank loss passed\n",stationarity);
    for(int ref:{0,1,2}){
        MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
        const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
        D6hOrbit orbit;orbit.build(mesh);PrismGroupMaps maps;maps.build(orbit);
        const int size=mesh.system_dofs(),edges=mesh.topology_edge_count;
        std::vector<Vec3> points;for(int e=0;e<edges;++e)points.push_back(mesh.current_dof_points[2*e]);
        bem_spai::NearestEdges nearest(points);
        for(int seed:{0,edges/3,edges-1})for(int k:{1,8,31}){
            auto actual=nearest.nearest(seed,k);std::vector<std::pair<double,int>> brute;
            for(int e=0;e<edges;++e)if(e!=seed){const auto d=points[e]-points[seed];brute.emplace_back(d.x*d.x+d.y*d.y+d.z*d.z,e);}
            std::sort(brute.begin(),brute.end());std::vector<int> expected{seed};for(int j=0;j<std::min(k,edges)-1;++j)expected.push_back(brute[j].second);
            require(actual==expected,"geometric stencil differs from brute force");}
        bem_spai::PrismSparseInverse inverse;
        inverse.build(mesh,maps,8,16,[&](const std::vector<int>& ids){const int d=4*ids.size();Vector a(d*d);
            for(int i=0;i<d;++i)for(int j=0;j<d;++j)a[i*d+j]=Complex(.003*std::sin(i+2*j+ids[0]),.002*std::cos(2*i-j));
            for(int i=0;i<d;++i)a[i*d+i]+=Complex(2.+.01*i,.3);
            for(int j=0;j<d;++j)std::swap(a[j],a[d+j]);
            return a;});
        Vector input(size),a(size),v(size),actual(size),expected(size);for(int i=0;i<size;++i)input[i]=Complex(std::sin(.217*i),std::cos(.391*i));
        inverse.apply(input.data(),a.data());actual=input;inverse.apply(actual.data(),actual.data());require(error(actual,a)<1e-15,"SPAI aliased input");
        double max_defect=0;
        for(bool mirror:{false,true}){orbit.transform(input,v,mirror,true);inverse.apply(v.data(),actual.data());orbit.transform(a,expected,mirror,false);max_defect=std::max(max_defect,error(actual,expected));}
        PrismVerticalInverse vertical;vertical.build(mesh,[&](const Complex* in,Complex* out){inverse.apply(in,out);});
        vertical.transform(input.data(),v.data(),true);inverse.apply(v.data(),actual.data());vertical.transform(a.data(),expected.data(),false);max_defect=std::max(max_defect,error(actual,expected));
        require(max_defect<2e-13,"SPAI D6h covariance");
        // Exact inverse of electric/magnetic exchange tests channel order,
        // all 24 signed images, and weights at symmetry-fixed edges.
        const Complex coefficient[4]={{2.,.3},{1.7,-.2},{-.8,.4},{1.2,.7}};
        inverse.build(mesh,maps,5,11,[&](const std::vector<int>& ids){const int d=4*ids.size();Vector a(d*d);
            for(int i=0;i<d;++i)a[i*d+(i/4)*4+(i+2)%4]=coefficient[i%4];
            return a;});
        inverse.apply(input.data(),actual.data());
        for(int e=0;e<edges;++e)for(int c=0;c<4;++c){const int i=(c/2)*2*edges+2*e+c%2,other=((c+2)%4/2)*2*edges+2*e+c%2;expected[i]=input[other]/coefficient[(c+2)%4];}
        require(error(actual,expected)<2e-13&&inverse.maximum_fit_defect<2e-13&&inverse.rank_deficient_patches==0,"SPAI exact exchange inverse");
        std::fill(input.begin(),input.end(),0.);inverse.apply(input.data(),actual.data());for(auto z:actual)require(z==Complex(0),"SPAI zero RHS stale workspace");
        std::printf("ref=%d patches=%zu bytes=%zu D6h_defect=%.3e exact exchange/alias/zero/stencils passed\n",ref,inverse.patches(),inverse.storage_bytes(),max_defect);
    }
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
