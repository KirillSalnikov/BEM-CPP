// P12 physical seeds with the actual sector harmonic-Ritz learner and
// CoarseImageExperiment. Small dense operator only; no production speed claim.
#include "bdm_mesh_transfer.h"
#include "coarse_image_experiment.h"
#include "coarse_ritz_experiment.h"
#include "prism_sector_solver.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace bem_tmatrix;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
Vector read_vector(const std::string& path,size_t n){
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(n*sizeof(Complex)))throw std::runtime_error("input size: "+path);
    f.seekg(0);Vector out(n);if(!f.read(reinterpret_cast<char*>(out.data()),n*sizeof(Complex)))throw std::runtime_error("read "+path);return out;
}
void write_vector(const std::string& path,const Vector& x){std::ofstream f(path,std::ios::binary);
    if(!f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(Complex)))throw std::runtime_error("write "+path);}
double relative(const Vector& a,const Vector& b){double n=0,d=0;for(size_t i=0;i<a.size();++i){n+=std::norm(a[i]-b[i]);d+=std::norm(b[i]);}return std::sqrt(n/std::max(d,1e-300));}
struct DenseAction {
    int n;Vector a;
    explicit DenseAction(int size):n(size),a(size_t(size)*size){}
    DenseAction(int size,const std::string& file):n(size),a(read_vector(file,size_t(size)*size)){}
    void apply(const Complex* x,Complex* y)const {
        const auto* xx=reinterpret_cast<const double*>(x);
#pragma omp parallel for schedule(static)
        for(int row=0;row<n;++row){const auto* aa=reinterpret_cast<const double*>(a.data()+size_t(row)*n);double re=0,im=0;
#pragma omp simd reduction(+:re,im)
            for(int j=0;j<n;++j){re+=aa[2*j]*xx[2*j]-aa[2*j+1]*xx[2*j+1];im+=aa[2*j]*xx[2*j+1]+aa[2*j+1]*xx[2*j];}
            y[row]=Complex(re,im);}
    }
};
// Duffy quadrature has a small geometric symmetry defect. Nearly dependent
// enriched images can amplify it if a mirrored image is inferred rather than
// evaluated. Average ONLY the preconditioner model; strict solves retain A.
DenseAction symmetric_model(const DenseAction& original,const MullerP2Mesh& mesh,const PrismGroupMaps& group) {
    const int n=original.n;DenseAction model(n);std::vector<int> vertical,vs;D6hOrbit::vertical_image_map(mesh,vertical,vs);
    std::array<std::vector<int>,24> image,sign;int determinant[24];
    for(int v=0;v<2;++v)for(int g=0;g<12;++g){const int h=12*v+g;image[h].resize(n);sign[h].resize(n);determinant[h]=(v?-1:1)*(g>=6?-1:1);
        for(int i=0;i<n;++i){int target=group.image[g][i],s=group.sign[g][i];if(v){s*=vs[target];target=vertical[target];}image[h][i]=target;sign[h][i]=s;}}
#pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){Complex sum=0;for(int h=0;h<24;++h)
            sum+=double(determinant[h]*sign[h][i]*sign[h][j])*original.a[size_t(image[h][i])*n+image[h][j]];
        model.a[size_t(i)*n+j]=sum/24.;}
    return model;
}
// Whole J/M edge blocks averaged over exact current/equation D6h actions.
// This is a small CPU control, not the production FFT/circulant local inverse.
struct LocalInverse {
    std::vector<std::array<int,4>> indices;std::vector<std::array<Complex,16>> inverse;
    LocalInverse(const DenseAction& a,const MullerP2Mesh& mesh,const PrismGroupMaps& maps){
        std::vector<int> vertical,vs;D6hOrbit::vertical_image_map(mesh,vertical,vs);
        const int edges=mesh.topology_edge_count;indices.resize(edges);inverse.resize(edges);
        for(int e=0;e<edges;++e){auto& at=indices[e];at={{2*e,2*e+1,2*edges+2*e,2*edges+2*e+1}};
            Complex matrix[4][8]={};
            for(int i=0;i<4;++i){matrix[i][4+i]=1.;for(int j=0;j<4;++j)
                for(int v=0;v<2;++v)for(int g=0;g<12;++g){int r=maps.image[g][at[i]],c=maps.image[g][at[j]];
                    int rs=maps.sign[g][at[i]]*(g>=6?-1:1),cs=maps.sign[g][at[j]];
                    if(v){rs*=-vs[r];cs*=vs[c];r=vertical[r];c=vertical[c];}
                    matrix[i][j]+=double(rs*cs)*a.a[size_t(r)*a.n+c]/24.;}}
            for(int col=0;col<4;++col){int pivot=col;for(int i=col+1;i<4;++i)if(std::abs(matrix[i][col])>std::abs(matrix[pivot][col]))pivot=i;
                if(std::abs(matrix[pivot][col])<1e-14)throw std::runtime_error("singular local edge block");
                if(pivot!=col)for(int j=0;j<8;++j)std::swap(matrix[col][j],matrix[pivot][j]);
                const auto scale=matrix[col][col];for(auto& z:matrix[col])z/=scale;
                for(int i=0;i<4;++i)if(i!=col){const auto scale=matrix[i][col];for(int j=0;j<8;++j)matrix[i][j]-=scale*matrix[col][j];}}
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)inverse[e][4*i+j]=matrix[i][4+j];
        }
    }
    void apply(const Complex* rhs,Complex* out)const {
        for(size_t e=0;e<indices.size();++e){const auto& at=indices[e];Complex b[4];for(int j=0;j<4;++j)b[j]=rhs[at[j]];
            for(int i=0;i<4;++i){Complex v=0;for(int j=0;j<4;++j)v+=inverse[e][4*i+j]*b[j];out[at[i]]=v;}}
    }
};
int main(int argc,char** argv)try {
    if(argc!=8)throw std::invalid_argument("COARSE_REF FINE_REF COARSE_A FINE_A TRAIN_B DIRECTORY SETUP_JSON");
    const int cr=std::stoi(argv[1]),fr=std::stoi(argv[2]);if(cr<0||fr<=cr||fr>2)throw std::invalid_argument("small CPU refinements");
    const std::string directory=argv[6];const auto beginning=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto cm=build_muller_p2_mesh(regular_prism(6,10./7,cr,1.,0,false,true),options),fm=build_muller_p2_mesh(regular_prism(6,10./7,fr,1.,0,false,true),options);
    const int nc=cm.system_dofs(),n=fm.system_dofs();if(n>5000)throw std::runtime_error("dense CPU size limit");
    DenseAction coarse(nc,argv[3]),fine(n,argv[4]);D6hOrbit cg,fg;cg.build(cm);fg.build(fm);
    PrismGroupMaps cgroups,fgroups;cgroups.build(cg);fgroups.build(fg);
    PrismSymmetrySectors cs,fs;cs.build(cgroups);fs.build(fgroups);
    const auto sym_start=Clock::now();const auto model=symmetric_model(fine,fm,fgroups);
    const double sym_seconds=seconds(sym_start),model_difference=relative(model.a,fine.a);
    LocalInverse clocal(coarse,cm,cgroups),flocal(fine,fm,fgroups);
    const bem_streaming::Action ci=[&](const Complex* x,Complex* y){clocal.apply(x,y);},fi=[&](const Complex* x,Complex* y){flocal.apply(x,y);};
    int training_actions=0;
    const bem_streaming::Action ca=[&](const Complex* x,Complex* y){++training_actions;coarse.apply(x,y);};
    const auto local_setup=seconds(beginning);const auto train_start=Clock::now();
    auto train=read_vector(argv[5],nc);Vector solution(nc);bem_streaming::SectorRecycleSpace learned(cs.offsets,16);
    const auto result=solve_prism_sectors(ca,ca,ci,cs,train.data(),solution.data(),40,1e-3,500,false,false,&learned);
    Vector check(nc);coarse.apply(solution.data(),check.data());const double training_residual=relative(check,train);
    if(training_residual>1.001e-3)throw std::runtime_error("coarse physical training did not converge");
    std::array<std::vector<Vector>,12> saved;int packed=8;const std::string signature(64,'0');
    for(int sector=0;sector<12;++sector){const auto name=directory+"/sector_"+std::to_string(sector)+".cache";
        learned.sector(sector).save(name,signature);saved[sector]=coarse_ritz_experiment::read_columns(name,cs.offsets[sector+1]-cs.offsets[sector],signature);
        packed=std::min(packed,int(saved[sector].size()));}
    if(packed<4)throw std::runtime_error("insufficient harmonic Ritz directions in training sectors");
    BdmMeshTransfer transfer;transfer.build(cm,fm);std::vector<Vector> priors,images;
    Vector encoded(nc),decoded(nc),physical(nc),column(n),image(n);
    for(int j=0;j<packed;++j){for(int sector=0;sector<12;++sector)
            std::copy(saved[sector][j].begin(),saved[sector][j].end(),encoded.begin()+cs.offsets[sector]);
        cs.decode(encoded.data(),decoded.data(),true);ci(decoded.data(),physical.data());
        transfer.apply(physical.data(),column.data());const double norm=bem_streaming::norm(column.data(),n);
        if(!std::isfinite(norm)||norm<=1e-280)throw std::runtime_error("invalid transferred Ritz current");
        for(auto& z:column)z/=norm;
        model.apply(column.data(),image.data());priors.push_back(column);images.push_back(image);
    }

    const double ritz_setup=seconds(train_start);const auto before=Clock::now();
    CoarseImageExperiment correction(fm,fs,fi);
    for(int j=0;j<packed;++j)correction.add_d6h(priors[j],images[j]);
    const bem_streaming::Action inverse=[&](const Complex* a,Complex* b){correction.apply(a,b);};
    const double build_s=seconds(before);
    Vector raw;for(const auto& column:priors)raw.insert(raw.end(),column.begin(),column.end());
    write_vector(directory+"/ritz_prior_columns.bin",raw);
    double defect=0;Vector probe(n),value(n),rotated(n),actual(n),expected(n);
    for(int i=0;i<n;++i)probe[i]=Complex(std::sin(.031*i),std::cos(.043*i));
    inverse(probe.data(),value.data());
    for(int g=0;g<3;++g){if(g<2){fg.transform(probe,rotated,g==1,true);fg.transform(value,expected,g==1,false);}
        else {std::vector<int> target,sign;D6hOrbit::vertical_image_map(fm,target,sign);for(int i=0;i<n;++i){rotated[target[i]]=-double(sign[i])*probe[i];expected[target[i]]=double(sign[i])*value[i];}}
        inverse(rotated.data(),actual.data());defect=std::max(defect,relative(actual,expected));}
    if(defect>1e-8)throw std::runtime_error("P2 D6h inverse check failed");
    std::ofstream setup(argv[7]);setup<<std::setprecision(17)<<"{\"coarse_dofs\":"<<nc<<",\"fine_dofs\":"<<n<<",\"packed_ritz_columns\":"<<packed
        <<",\"local_setup_s\":"<<local_setup<<",\"ritz_training_transfer_images_s\":"<<ritz_setup<<",\"training_iterations\":"<<result.iterations
        <<",\"training_actions\":"<<training_actions<<",\"training_physical_residual\":"<<training_residual
        <<",\"image_model_symmetrization_s\":"<<sym_seconds<<",\"image_model_relative_change\":"<<model_difference
        <<",\"strict_operator_symmetrized\":false,\"image_setup_actions\":"<<packed<<",\"rank\":"<<correction.rank()
        <<",\"correction_bytes\":"<<correction.bytes()<<",\"build_s\":"<<build_s<<",\"D6h_defect\":"<<defect<<"}\n";
    setup.close();if(!setup)throw std::runtime_error("P2 setup report");std::cout<<"READY\n"<<std::flush;
    std::map<std::string,std::unique_ptr<bem_streaming::RecycleSpace>> recycles;
    for(const auto* name:{"P2_zero","P2_guess","P2_mbs025","P2_mbs1","P2_exact_far"})
        recycles.emplace(name,std::unique_ptr<bem_streaming::RecycleSpace>(new bem_streaming::RecycleSpace(n,16)));
    std::string name;
    while(std::cin>>name){if(name=="QUIT")break;int id;std::string rhs_file,seed_file,prefix;
        if(!(std::cin>>id>>rhs_file))throw std::runtime_error("invalid P12 request");
        const auto rhs=read_vector(rhs_file,n);
        if(name=="GUESS"){
            if(!(std::cin>>prefix))throw std::runtime_error("invalid P2 guess request");
            Vector guess(n);const auto start=Clock::now();correction.guess(rhs.data(),guess.data());const double elapsed=seconds(start);
            write_vector(prefix+".guess.bin",guess);std::ofstream report(prefix+".guess.json");report<<std::setprecision(17)<<"{\"application_s\":"<<elapsed<<"}\n";
            std::cout<<"GUESSED "<<id<<'\n'<<std::flush;continue;
        }
        if(!(std::cin>>seed_file>>prefix)||!recycles.count(name))throw std::runtime_error("invalid P12 seeded request");
        Vector x=read_vector(seed_file,n),ax(n);for(auto z:x)if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite seed");
        int actions=0,strict_actions=0;
        const bem_streaming::Action action=[&](const Complex* a,Complex* b){++actions;fine.apply(a,b);},strict=[&](const Complex* a,Complex* b){++strict_actions;fine.apply(a,b);};
        const auto begin=Clock::now();const auto solved=bem_streaming::gmres(action,strict,inverse,rhs.data(),x.data(),n,40,1e-8,500,false,true,recycles[name].get());
        const double solve_s=seconds(begin);const auto verify=Clock::now();strict(x.data(),ax.data());const double residual=relative(ax,rhs),check_s=seconds(verify);
        if(!std::isfinite(residual)||residual>1.001e-8)throw std::runtime_error("original fine BEM residual failed");
        write_vector(prefix+".currents.bin",x);std::ofstream report(prefix+".json");report<<std::setprecision(17)
            <<"{\"variant\":\""<<name<<"\",\"column\":"<<id<<",\"iterations\":"<<solved.iterations<<",\"actions\":"<<actions
            <<",\"strict_actions_including_initial_and_final\":"<<strict_actions<<",\"strict_residual\":"<<residual<<",\"solve_s_shared_cpu\":"<<solve_s
            <<",\"final_check_s\":"<<check_s<<",\"recycle_rank\":"<<recycles[name]->rank()<<"}\n";
        report.close();if(!report)throw std::runtime_error("P12 solve report");std::cout<<"DONE "<<name<<' '<<id<<'\n'<<std::flush;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
