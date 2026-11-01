// CPU physical control of matrix-free D6h preconditioner images using the
// CoarseImageExperiment. Small dense operator only; no production speed claim.
#include "bdm_mesh_transfer.h"
#include "coarse_image_experiment.h"
#include "prism_d6h_model_action.h"
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
    if(argc!=9)throw std::invalid_argument("COARSE_REF FINE_REF COARSE_A FINE_A TRAIN_B PHASE_COLUMNS DIRECTORY SETUP_JSON");
    const int cr=std::stoi(argv[1]),fr=std::stoi(argv[2]);if(cr<0||fr<=cr||fr>2)throw std::invalid_argument("small CPU refinements");
    const std::string directory=argv[7];const auto beginning=Clock::now();MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto cm=build_muller_p2_mesh(regular_prism(6,10./7,cr,1.,0,false,true),options),fm=build_muller_p2_mesh(regular_prism(6,10./7,fr,1.,0,false,true),options);
    const int nc=cm.system_dofs(),n=fm.system_dofs();if(n>5000)throw std::runtime_error("dense CPU size limit");
    DenseAction coarse(nc,argv[3]),fine(n,argv[4]);D6hOrbit cg,fg;cg.build(cm);fg.build(fm);
    PrismGroupMaps cgroups,fgroups;cgroups.build(cg);fgroups.build(fg);
    PrismSymmetrySectors cs,fs;cs.build(cgroups);fs.build(fgroups);
    const auto sym_start=Clock::now();
    PrismD6hModelAction model(fm,fgroups,[&](const Complex* x,Complex* y){fine.apply(x,y);});
    Vector test_model(n),test_original(n),test_input(n);
    for(int i=0;i<n;++i)test_input[i]=Complex(std::sin(.031*i),std::cos(.043*i));
    model.apply(test_input.data(),test_model.data());fine.apply(test_input.data(),test_original.data());
    const double sym_seconds=seconds(sym_start),model_difference=relative(test_model,test_original);
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
    const double ritz_setup=seconds(train_start);const auto phase_start=Clock::now();
    const auto phase=read_vector(argv[6],size_t(24)*n);
    // Both polarizations of physical edges and internal reflected beam
    // boundaries are present already in the eight-column control.
    const int selected[]={0,1,10,11,20,21,22,23,4,5,14,15,16,17,18,19};
    std::vector<Vector> additions,added_images;
    for(int index:selected){column.assign(phase.begin()+size_t(index)*n,phase.begin()+size_t(index+1)*n);
        model.apply(column.data(),image.data());additions.push_back(column);added_images.push_back(image);}
    const double phase_images_s=seconds(phase_start);
    struct Variant {std::string name;int extra;std::unique_ptr<CoarseImageExperiment> correction;std::unique_ptr<bem_streaming::RecycleSpace> recycle;bem_streaming::Action apply;double build_s=0,defect=0;};
    std::vector<Variant> variants;
    for(int extra:{-1,0,8,16}){Variant v;v.extra=extra;v.name=extra<0?"local":extra==0?"P2_ritz":"P2_phase"+std::to_string(extra);
        const auto before=Clock::now();
        if(extra>=0){v.correction.reset(new CoarseImageExperiment(fm,fs,fi));
            for(int j=0;j<packed;++j)v.correction->add_d6h(priors[j],images[j]);
            for(int j=0;j<extra;++j)v.correction->add_d6h(additions[j],added_images[j]);
            auto* pointer=v.correction.get();v.apply=[pointer](const Complex* a,Complex* b){pointer->apply(a,b);};
        }else v.apply=fi;
        v.build_s=seconds(before);v.recycle.reset(new bem_streaming::RecycleSpace(n,16));
        Vector rhs(n),value(n),rotated(n),actual(n),expected(n);
        for(int i=0;i<n;++i)rhs[i]=Complex(std::sin(.031*i),std::cos(.043*i));
        v.apply(rhs.data(),value.data());
        for(int g=0;g<3;++g){if(g<2){fg.transform(rhs,rotated,g==1,true);fg.transform(value,expected,g==1,false);}
            else {std::vector<int> target,sign;D6hOrbit::vertical_image_map(fm,target,sign);for(int i=0;i<n;++i){rotated[target[i]]=-double(sign[i])*rhs[i];expected[target[i]]=double(sign[i])*value[i];}}
            v.apply(rotated.data(),actual.data());v.defect=std::max(v.defect,relative(actual,expected));}
        std::cerr<<std::setprecision(17)<<"inverse "<<v.name<<" rank="<<(v.correction?v.correction->rank():0)<<" D6h_defect="<<v.defect<<'\n';
        if(v.defect>1e-8)throw std::runtime_error("full D6h inverse check failed: "+v.name);
        variants.push_back(std::move(v));
    }
    std::ofstream setup(argv[8]);setup<<std::setprecision(17)<<"{\"coarse_dofs\":"<<nc<<",\"fine_dofs\":"<<n<<",\"packed_ritz_columns\":"<<packed
        <<",\"local_setup_s\":"<<local_setup<<",\"ritz_training_transfer_images_s\":"<<ritz_setup<<",\"training_iterations\":"<<result.iterations
        <<",\"training_actions\":"<<training_actions<<",\"training_physical_residual\":"<<training_residual<<",\"phase_images_s\":"<<phase_images_s
        <<",\"image_model_symmetrization_s\":"<<sym_seconds<<",\"image_model_probe_relative_change\":"<<model_difference
        <<",\"strict_operator_symmetrized\":false,\"dense_operator_bytes\":"<<(coarse.a.size()+fine.a.size())*sizeof(Complex)<<",\"variants\":[";
    for(size_t j=0;j<variants.size();++j){const auto& v=variants[j];if(j)setup<<',';
        setup<<"{\"name\":\""<<v.name<<"\",\"phase_columns\":"<<std::max(0,v.extra)<<",\"image_setup_actions\":"<<(v.extra>=0?24*(packed+v.extra)+25:0)
            <<",\"rank\":"<<(v.correction?v.correction->rank():0)<<",\"correction_bytes\":"<<(v.correction?v.correction->bytes():0)
            <<",\"build_s\":"<<v.build_s<<",\"D6h_defect\":"<<v.defect<<'}';}
    setup<<"],\"matrixfree_model_bytes\":"<<model.storage_bytes()<<",\"model_original_actions\":"<<model.original_actions()<<",\"model_calls\":"<<model.model_calls()<<",\"original_model_probe_actions\":1}\n";setup.close();std::cout<<"READY\n"<<std::flush;
    std::string name;
    while(std::cin>>name){if(name=="QUIT")break;int id;std::string rhs_file,prefix;if(!(std::cin>>id>>rhs_file>>prefix))throw std::runtime_error("invalid CPU request");
        auto found=std::find_if(variants.begin(),variants.end(),[&](const Variant& v){return v.name==name;});if(found==variants.end())throw std::runtime_error("unknown variant");
        auto& v=*found;const auto rhs=read_vector(rhs_file,n);Vector x(n),ax(n);int actions=0,strict_actions=0;
        const bem_streaming::Action action=[&](const Complex* a,Complex* b){++actions;fine.apply(a,b);},strict=[&](const Complex* a,Complex* b){++strict_actions;fine.apply(a,b);};
        const auto begin=Clock::now();const auto solved=bem_streaming::gmres(action,strict,v.apply,rhs.data(),x.data(),n,40,1e-8,500,false,false,v.recycle.get());
        const double solve_s=seconds(begin);const auto verify=Clock::now();strict(x.data(),ax.data());const double residual=relative(ax,rhs),check_s=seconds(verify);
        if(residual>1.001e-8)throw std::runtime_error("original fine BEM residual failed");
        write_vector(prefix+".currents.bin",x);std::ofstream report(prefix+".json");report<<std::setprecision(17)
            <<"{\"variant\":\""<<name<<"\",\"column\":"<<id<<",\"iterations\":"<<solved.iterations<<",\"actions\":"<<actions
            <<",\"strict_actions_including_final\":"<<strict_actions<<",\"strict_residual\":"<<residual<<",\"solve_s_shared_cpu\":"<<solve_s
            <<",\"final_check_s\":"<<check_s<<",\"recycle_rank\":"<<v.recycle->rank()<<"}\n";
        report.close();std::cout<<"DONE "<<name<<' '<<id<<'\n'<<std::flush;
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
