#include "host_fmm_taylor.h"
#include <cstdio>

// Pure host fixture, no CUDA context or buffers.
void HelmholtzFMM::cleanup(){if(d_multi_re||d_local_re||d_charges_re)std::abort();initialized=false;}
using C=std::complex<double>;using LC=std::complex<long double>;
LC hankel(int l,LC z){const LC I(0,1);LC sum=1,term=1;
    for(int a=1;a<=l;++a){term*=static_cast<long double>((l+a)*(l-a+1))/static_cast<long double>(a)*I/(2.L*z);sum+=term;}
    return std::pow(-I,l+1)*std::exp(I*z)/z*sum;
}
void retune(HelmholtzFMM& f,C k){
    f.k=k;f.transfer_cache.assign(size_t(2)*f.L,C(0));
    for(int id=0;id<2;++id){const int s=id?2:1,t=id?1:2;
        const double distance=std::abs(f.tree.nodes[t].center[0]-f.tree.nodes[s].center[0]);
        for(int l=0;l<f.L;++l){const long double cosine=(id?-1:1)*f.squad.dirs[3*l];
            LC sum=0,power=1;long double previous=1,current=cosine;
            for(int p=0;p<=f.p;++p){long double legendre=1;
                if(p==1)legendre=current;
                if(p>1){legendre=((2.L*p-1)*cosine*current-(p-1)*previous)/p;previous=current;current=legendre;}
                sum+=(2.L*p+1)*power*hankel(p,LC(k)*static_cast<long double>(distance))*legendre;power*=LC(0,1);}
            f.transfer_cache[size_t(id)*f.L+l]=C(sum);
        }
    }
    for(int sign:{-1,1}){auto& levels=sign<0?f.m2m_data:f.l2l_data;
        for(auto& level:levels){level.shifts.clear();for(auto e:level.pairs)for(int l=0;l<f.L;++l){double dot=0;
            for(int a=0;a<3;++a)dot+=(f.tree.nodes[e.child].center[a]-f.tree.nodes[e.parent].center[a])*f.squad.dirs[3*l+a];
            level.shifts.push_back(std::exp(C(0,sign)*k*dot));}}
    }
}
int main()try{
    constexpr int degree=4,samples=48;using Band=bem_cpu::TaylorFmmBand<degree>;
    const std::vector<Vec3> points{{-1.2,.1,0},{-1.,.1,.1},{1.2,-.1,.1},{1.4,0,-.1}};
    HelmholtzFMM f;f.initialized=true;f.Ns=f.Nt=points.size();f.p=12;f.squad.init(f.p);f.L=f.squad.L;
    f.p2p_enabled=true;f.n_nodes=5;f.n_unique_transfers=2;f.minimum_m2l_level=1;f.maximum_m2l_level=1;f.tree.max_level=2;
    f.tree.nodes.resize(5);for(int i=0;i<5;++i){f.tree.nodes[i]={};f.tree.nodes[i].index=i;f.tree.nodes[i].level=i<3?1:2;}
    f.tree.nodes[1].center[0]=-.9;f.tree.nodes[2].center[0]=1.1;f.tree.nodes[3].center[0]=-1.1;f.tree.nodes[4].center[0]=1.3;
    f.tree.level_nodes={{0},{1,2},{3,4}};f.h_leaf_indices={3,4};
    f.h_src_id_offsets=f.h_tgt_id_offsets={0,2,4};f.h_src_ids_flat={1,0,3,2};f.h_tgt_ids_flat={0,1,2,3};
    f.m2l_batches.resize(3);for(auto& b:f.m2l_batches)b.n_pairs=0;
    auto& batch=f.m2l_batches[1];batch.n_pairs=2;batch.src_idx={1,2};batch.tgt_idx={2,1};batch.transfer_idx={0,1};
    f.m2m_data.resize(3);f.l2l_data.resize(3);f.m2m_data[1].pairs={{1,3},{2,4}};f.l2l_data[2].pairs={{1,3},{2,4}};
    const C k(3,.2),slope(1.3116,.1);retune(f,k);
    Band action(f,points,slope);std::vector<Band::Charge> charge(points.size());std::vector<Band::Field> actual(points.size());
    for(size_t p=0;p<points.size();++p)for(int j=0;j<=degree;++j)for(int c=0;c<6;++c)
        charge[p][j][c]=C(std::sin(.3*p+.7*j+.2*c),std::cos(.7*p-.2*j+.3*c));
    action.add(charge,actual);
    std::vector<std::array<std::array<LC,12>,degree+1>> oracle(points.size());
    const long double pi=std::acos(-1.L),radius=.4;long double norm=0;
    for(int p=0;p<samples;++p){const C z=std::polar(double(radius),double(2*pi*p/samples));retune(f,k+slope*z);
        bem_cpu::FmmBand reference(f,points,0);std::vector<bem_cpu::Six> input(points.size());std::vector<bem_cpu::Twelve> output(points.size());
        for(size_t i=0;i<points.size();++i)for(int c=0;c<6;++c)for(int j=degree;j>=0;--j)input[i][c]=z*input[i][c]+charge[i][j][c];
        reference.add(input,output);
        for(size_t i=0;i<points.size();++i)for(int c=0;c<12;++c){norm+=std::norm(output[i][c])/samples;
            for(int j=0;j<=degree;++j)oracle[i][j][c]+=LC(output[i][c])*std::polar(1.L/samples,-2*pi*p*j/samples);}
    }
    long double error=0;for(int j=0;j<=degree;++j){long double delta=0;
        for(size_t i=0;i<points.size();++i)for(int c=0;c<12;++c)delta+=std::norm(LC(actual[i][j][c])*std::pow(radius,j)-oracle[i][j][c]);
        error=std::max(error,std::sqrt(delta/norm));}
    retune(f,k);bem_cpu::FmmBand central(f,points,0);
    std::vector<bem_cpu::Six> input(points.size());std::vector<bem_cpu::Twelve> expected(points.size());
    for(size_t i=0;i<points.size();++i)input[i]=charge[i][0];
    central.add(input,expected);
    double n=0,d=0;for(size_t i=0;i<points.size();++i)for(int c=0;c<12;++c){n+=std::norm(actual[i][0][c]-expected[i][c]);d+=std::norm(expected[i][c]);}
    const double central_error=std::sqrt(n/d);
    if(!std::isfinite(error+central_error)||error>2e-10||central_error>2e-10)throw std::runtime_error("full FMM Taylor versus independent retuned action");
    std::fill(charge.begin(),charge.end(),Band::Charge{});std::fill(actual.begin(),actual.end(),Band::Field{});action.add(charge,actual);
    for(const auto& row:actual)for(const auto& moment:row)for(C z:moment)if(z!=C(0))throw std::runtime_error("FMM Taylor stale buffer");
    std::printf("PASS matrix-free FMM moments 0..4: central %.3e, independent contour %.3Le; P2M/M2M/M2L/L2L/L2P/P2P, two currents and zero buffers; workspace %zu bytes\n",central_error,error,action.workspace_bytes());
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
