#include "host_prism_face_taylor.h"
#include <cstdio>
#include <numeric>
#include <random>

// CPU-only D6h point set: six rotated side lattices and two hexagonal caps.
// The cross trees are exact rectangular P2P, with permuted source ordinals.
// Native far trees are verified separately, never implied by this fixture.
void HelmholtzFMM::cleanup(){if(d_multi_re||d_local_re||d_charges_re)std::abort();initialized=false;}
using C=std::complex<double>;using LC=std::complex<long double>;
using Matrix=std::array<double,9>;
Vec3 rotate(const Matrix& r,Vec3 p){return {r[0]*p.x+r[1]*p.y+r[2]*p.z,r[3]*p.x+r[4]*p.y+r[5]*p.z,r[6]*p.x+r[7]*p.y+r[8]*p.z};}
struct Fixture {
    std::vector<Vec3> points;
    std::vector<double> offsets{0,0,0};
    std::vector<int> global[8],slots[8],other[8],targets[8];
    HelmholtzFMM tree[8];PrismFaceFFT::HostView view{};
    explicit Fixture(C k){
        view.k=k;view.target_faces=true;
        view.side={2,2,{{.2,0,0}},{{0,.3,0}},&offsets};
        view.cap={3,3,{{.25,0,0}},{{.125,std::sqrt(3.)*.125,0}},&offsets};
        for(int f=0;f<8;++f){auto& face=view.faces[f];const double angle=f*std::acos(-1.)/3,c=std::cos(angle),s=std::sin(angle);
            Matrix rotation=f<6?Matrix{{c,-s,0,s,c,0,0,0,1}}:Matrix{{1,0,0,0,f==6?1.:-1.,0,0,0,f==6?1.:-1.}};
            face.rotation=rotation;face.engine=f<6?0:6;
            const Matrix canonical=f<6?Matrix{{0,1,0,0,0,1,1,0,0}}:Matrix{{1,0,0,0,1,0,0,0,1}};
            for(int i=0;i<3;++i)for(int j=0;j<3;++j){face.frame[3*i+j]=0;
                for(int a=0;a<3;++a)face.frame[3*i+j]+=canonical[3*i+a]*rotation[3*j+a];}
            const int nx=f<6?2:3,ny=nx;
            for(int i=0;i<nx;++i)for(int j=0;j<ny;++j){
                if(f>=6 && ((i==0&&j==0)||(i==2&&j==2)))continue;
                Vec3 p=f<6?Vec3(1,-.1+.2*i,-.15+.3*j):Vec3(.25*(i-1)+.125*(j-1),std::sqrt(3.)*.125*(j-1),.6);
                global[f].push_back(points.size());slots[f].push_back(i*ny+j);points.push_back(rotate(rotation,p));
            }
            face.global=&global[f];face.slot=&slots[f];face.other=&other[f];face.cross_global=&targets[f];
        }
        for(int engine:{0,6}){
            targets[engine]=global[engine];
            for(size_t p=0;p<points.size();++p)if(std::find(global[engine].begin(),global[engine].end(),int(p))==global[engine].end())other[engine].push_back(p);
            auto& f=tree[engine];f.initialized=true;f.k=k;f.Ns=other[engine].size();f.Nt=targets[engine].size();
            f.n_nodes=1;f.L=1;f.p=0;f.n_unique_transfers=0;f.p2p_enabled=true;f.minimum_m2l_level=1;f.maximum_m2l_level=0;
            f.tree.max_level=0;f.tree.nodes.resize(1);f.tree.nodes[0]={};f.tree.level_nodes={{0}};f.h_leaf_indices={0};
            f.h_src_id_offsets={0,f.Ns};f.h_tgt_id_offsets={0,f.Nt};f.h_src_ids_flat.resize(f.Ns);f.h_tgt_ids_flat.resize(f.Nt);
            std::iota(f.h_src_ids_flat.rbegin(),f.h_src_ids_flat.rend(),0);std::iota(f.h_tgt_ids_flat.begin(),f.h_tgt_ids_flat.end(),0);
            f.squad.dirs={1,0,0};f.squad.weights={1};
        }
        for(int f=0;f<8;++f){auto& face=view.faces[f];const int engine=face.engine;
            face.fine=&tree[engine];face.middle=face.coarse=nullptr;
            if(f==engine)continue;
            const auto mapped=[&](int p){const Vec3 desired=rotate(face.rotation,points[p]);int found=-1;
                for(size_t j=0;j<points.size();++j)if((points[j]-desired).norm()<2e-12){if(found!=-1)throw std::runtime_error("duplicate fixture point");found=j;}
                if(found<0)throw std::runtime_error("fixture rotation not closed");
                return found;};
            for(int p:other[engine])other[f].push_back(mapped(p));
            for(int p:targets[engine])targets[f].push_back(mapped(p));
        }
    }
};
std::vector<bem_cpu::Twelve> direct(const std::vector<Vec3>& points,C k,const std::vector<bem_cpu::Six>& charge){
    std::vector<bem_cpu::Twelve> field(points.size());
    for(size_t t=0;t<points.size();++t)for(size_t s=0;s<points.size();++s)if(t!=s){
        const Vec3 d=points[t]-points[s];const double r=d.norm();const C g=std::exp(C(0,1)*k*r)/(4*std::acos(-1.)*r);
        const C a=g*(C(0,1)*k-1./r)/r,b=g*(3./(r*r)-C(0,3)*k/r-k*k)/(r*r);
        const C trace=3.*a+b*r*r;const double v[3]={d.x,d.y,d.z};
        for(int current=0;current<2;++current){const C* q=charge[s].data()+3*current;C* out=field[t].data()+6*current;
            out[0]+=a*(d.y*q[0]-d.x*q[1]);out[1]+=a*(d.z*q[0]-d.x*q[2]);out[2]+=a*(d.z*q[1]-d.y*q[2]);
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[3+i]+=((i==j?a-trace:C(0))+b*v[i]*v[j])*q[j];
        }
    }
    return field;
}
int main()try{
    constexpr int degree=4,samples=48;using Action=bem_cpu::TaylorPrismFaceAction<degree>;
    double worst=0,central=0;size_t workspace=0;
    for(C index:{C(1),C(1.3116,.4)}){
        const C k=3.*index;Fixture f(k);Action action(f.view,f.points,index);workspace=std::max(workspace,action.workspace_bytes());
        std::mt19937 rng(915);std::normal_distribution<double> normal;
        std::vector<Action::Charge> input(f.points.size());std::vector<Action::Field> output(f.points.size());
        for(auto& row:input)for(int j=0;j<=degree;++j)for(auto& q:row[j])q=j==2?C(0):C(normal(rng),normal(rng));
        action.add(input,output);
        std::vector<std::array<std::array<LC,12>,degree+1>> contour(f.points.size());
        const long double radius=.5,pi=std::acos(-1.L);long double scale=0;
        for(int p=0;p<samples;++p){const C delta=std::polar(double(radius),double(2*pi*p/samples));
            std::vector<bem_cpu::Six> q(f.points.size());
            for(size_t t=0;t<q.size();++t)for(int c=0;c<6;++c)for(int j=degree;j>=0;--j)q[t][c]=delta*q[t][c]+input[t][j][c];
            const auto exact=direct(f.points,k+index*delta,q);
            for(size_t t=0;t<q.size();++t)for(int c=0;c<12;++c){scale+=std::norm(exact[t][c])/samples;
                for(int j=0;j<=degree;++j)contour[t][j][c]+=LC(exact[t][c])*std::polar(1.L/samples,-2*pi*p*j/samples);}
        }
        for(int j=0;j<=degree;++j){long double numerator=0;
            for(size_t t=0;t<f.points.size();++t)for(int c=0;c<12;++c)numerator+=std::norm(LC(output[t][j][c])*std::pow(radius,j)-contour[t][j][c]);
            const double e=std::sqrt(numerator/scale);worst=std::max(worst,e);if(j==0)central=std::max(central,e);
            if(!std::isfinite(e)||e>2e-10)throw std::runtime_error("face+cross Taylor versus direct contour");
        }
        auto twice=output;action.add(input,twice);
        for(size_t t=0;t<f.points.size();++t)for(int j=0;j<=degree;++j)for(int c=0;c<12;++c)
            if(std::abs(twice[t][j][c]-2.*output[t][j][c])>1e-11*std::max(1.,std::abs(output[t][j][c])))throw std::runtime_error("additive face history");
        std::fill(input.begin(),input.end(),Action::Charge{});std::fill(output.begin(),output.end(),Action::Field{});action.add(input,output);
        for(const auto& row:output)for(const auto& moment:row)for(C q:moment)if(q!=C(0))throw std::runtime_error("face history stale buffer");
        auto bad=f.view;bad.faces[1].slot=bad.faces[0].slot; // same size: still valid, now corrupt a point explicitly
        auto duplicate=f.global[1];duplicate[0]=f.global[0][0];bad.faces[1].global=&duplicate;bool rejected=false;
        try{Action invalid(bad,f.points,index);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("duplicate face point accepted");
    }
    std::printf("PASS prism FFT+cross Taylor 0..4, two currents/materials, canonical rotations, cap reversal, sparse lattice masks, mapped P2P, additive/zero: central %.3e contour %.3e workspace %zu bytes\n",central,worst,workspace);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
