// Small CPU-only P12 geometry bridge. Dense transfer is restricted to <=5000
// DOFs; it is a validation artifact, never a production-sized allocation.
#include "bdm_mesh_transfer.h"
#include "tmatrix_symmetry_warm_start.h"
#include "quadrature.h"
#include <fstream>
#include <iostream>
#include <iomanip>
template<class T> void save(const std::string& path,const std::vector<T>& x) {
    std::ofstream f(path,std::ios::binary);
    if(!f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(T)))
        throw std::runtime_error("write "+path);
}
void maps(const MullerP2Mesh& mesh,const std::string& prefix) {
    bem_tmatrix::D6hOrbit group;group.build(mesh);
    std::vector<int> data,image,sign;
    for(int s=0;s<3;++s) {
        if(s<2)group.image_map(s==1,false,image,sign);
        else bem_tmatrix::D6hOrbit::vertical_image_map(mesh,image,sign);
        data.insert(data.end(),image.begin(),image.end());
        data.insert(data.end(),sign.begin(),sign.end());
    }
    save(prefix+".symmetry.bin",data);
}
int main(int argc,char** argv)try {
    if(argc!=5)throw std::runtime_error("COARSE_REF FINE_REF ASPECT PREFIX");
    const int cr=std::stoi(argv[1]),fr=std::stoi(argv[2]);const double aspect=std::stod(argv[3]);
    if(cr<0||fr<cr||fr>2||!std::isfinite(aspect)||aspect<=0)throw std::runtime_error("invalid fixture parameters");
    const std::string prefix=argv[4];MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto coarse=build_muller_p2_mesh(regular_prism(6,aspect,cr,1,0,false,true),options);
    const auto fine=build_muller_p2_mesh(regular_prism(6,aspect,fr,1,0,false,true),options);
    const int nc=coarse.system_dofs(),nf=fine.system_dofs();if(nf>5000)throw std::runtime_error("fixture size limit");
    BdmMeshTransfer transfer;transfer.build(coarse,fine);
    std::vector<std::complex<double>> x(nc),y(nf);std::vector<double> dense(size_t(nf)*nc);
    for(int col=0;col<nc;++col){x[col]=1.;transfer.apply(x.data(),y.data());x[col]=0.;
        for(int row=0;row<nf;++row){if(y[row].imag()!=0.)throw std::runtime_error("nonreal transfer");dense[size_t(row)*nc+col]=y[row].real();}}
    save(prefix+".transfer.bin",dense);maps(coarse,prefix+".coarse");maps(fine,prefix+".fine");
    std::vector<double> metric(nf);const auto quad=tri_quadrature(13);
    for(size_t t=0;t<fine.elements.size();++t)for(int q=0;q<quad.npts;++q){
        const auto frame=evaluate_muller_frame(fine,t,quad.pts[q][0],quad.pts[q][1]);
        const auto b=evaluate_muller_basis(fine,t,frame);
        for(int j=0;j<b.count;++j){const double d=.5*quad.wts[q]*frame.jacobian*b.values[j].dot(b.values[j]);
            metric[b.dofs[j]]+=d;metric[fine.current_dofs()+b.dofs[j]]+=d;}}
    for(double value:metric)if(!(value>0))throw std::runtime_error("nonpositive metric");
    save(prefix+".metric.bin",metric);
    std::ofstream meta(prefix+".json");meta<<std::setprecision(17)<<"{\"coarse_dofs\":"<<nc<<",\"fine_dofs\":"<<nf
        <<",\"coarse_ref\":"<<cr<<",\"fine_ref\":"<<fr<<",\"aspect\":"<<aspect<<",\"transfer_bytes\":"<<transfer.bytes()<<"}\n";
    std::cout<<"P12 geometry "<<nc<<" -> "<<nf<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
