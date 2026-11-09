#pragma once
#include "muller_fmm.h"
#include "prism_face_fft.h"
#include "quadrature.h"
#include <map>
#include <numeric>

// A small physical BDM prism, without a CUDA runtime. The ordinary production
// CPU FFT/FMM consumers see real quadrature points and lattice maps. Cross-face
// trees deliberately contain exact rectangular P2P only: this fixture does not
// validate production tree construction, far translations, or GPU execution.
namespace physical_face_fixture {
using Matrix=std::array<double,9>;
using Key=std::array<long long,3>;
inline Vec3 rotate(const Matrix& r,Vec3 x){
    return {r[0]*x.x+r[1]*x.y+r[2]*x.z,r[3]*x.x+r[4]*x.y+r[5]*x.z,r[6]*x.x+r[7]*x.y+r[8]*x.z};
}
inline void tree(HelmholtzFMM& f,cdouble k,int sources,int targets,bool near){
    f.initialized=true;f.k=k;f.Ns=sources;f.Nt=targets;f.n_nodes=1;f.L=1;f.p=0;f.n_unique_transfers=0;
    f.p2p_enabled=near;f.minimum_m2l_level=1;f.maximum_m2l_level=0;
    f.tree.max_level=0;f.tree.nodes.resize(1);f.tree.nodes[0]={};f.tree.level_nodes={{0}};f.h_leaf_indices={0};
    f.h_src_id_offsets={0,sources};f.h_tgt_id_offsets={0,targets};
    f.h_src_ids_flat.resize(sources);f.h_tgt_ids_flat.resize(targets);
    // Nontrivial source and target ordinal order exercises the mapped gather.
    std::iota(f.h_src_ids_flat.rbegin(),f.h_src_ids_flat.rend(),0);
    std::iota(f.h_tgt_ids_flat.rbegin(),f.h_tgt_ids_flat.rend(),0);
    f.squad.dirs={1,0,0};f.squad.weights={1};
}
struct Layout {
    int nx=0,ny=0;Vec3 u,v;std::vector<double> offsets;
    std::vector<std::array<double,2>> fractional;
    void triangles(const std::vector<std::array<double,6>>& triangles){
        const auto rule=tri_quadrature(7);
        for(const auto& t:triangles)for(int q=0;q<rule.npts;++q){
            const double a=rule.pts[q][0],b=rule.pts[q][1];
            const double x=(1-a-b)*t[0]+a*t[2]+b*t[4],y=(1-a-b)*t[1]+a*t[3]+b*t[5];
            fractional.push_back({{x,y}});const Vec3 p=u*x+v*y;
            offsets.insert(offsets.end(),{p.x,p.y,p.z});
        }
    }
};
struct Face {
    Vec3 origin,e1,e2,e3;Matrix rotation{{1,0,0,0,1,0,0,0,1}};
    std::vector<int> global,slot,other,targets;
    int engine=0;HelmholtzFMM fine,middle,coarse;
};
}

void HelmholtzFMM::cleanup(){
    if(d_multi_re||d_local_re||d_charges_re)std::abort();
    initialized=false;
}
struct PrismFaceFFT::Impl {
    cdouble k;physical_face_fixture::Layout side,cap;
    std::array<physical_face_fixture::Face,8> faces;
};
PrismFaceFFT::PrismFaceFFT()=default;
PrismFaceFFT::~PrismFaceFFT()=default;
void PrismFaceFFT::init(const std::vector<double>& xyz,int ref,cdouble k,int digits,
                        const PrismFaceFFT* shared_geometry,double aspect){
    using namespace physical_face_fixture;
    if(ref<1||ref>2||xyz.size()%3||!std::isfinite(aspect)||aspect<=0||shared_geometry)
        throw std::invalid_argument("CPU fixture requires an independent small regular prism");
    (void)digits;p.reset(new Impl);p->k=k;
    const int seg=1<<ref;int nz=std::max(seg,int(std::ceil(std::sqrt(3.)*aspect*seg)));if(nz%2)++nz;
    const int count=xyz.size()/3;
    if(count!=7*12*seg*(seg+nz))throw std::runtime_error("CPU fixture quadrature point count");
    const double pi=std::acos(-1.),width=std::cbrt(8*pi/(3*std::sqrt(3.)*aspect));
    const double height=aspect*width,radius=width/std::sqrt(3.),h=radius/seg;
    p->side.nx=seg;p->side.ny=nz;p->side.u={h,0,0};p->side.v={0,height/nz,0};
    p->side.triangles({{{0,0,1,0,1,1}},{{0,0,1,1,0,1}},{{0,0,1,0,0,1}},{{1,0,1,1,0,1}}});
    p->cap.nx=p->cap.ny=2*seg;p->cap.u={h,0,0};p->cap.v={h*.5,h*std::sqrt(3.)*.5,0};
    p->cap.triangles({{{0,0,1,0,0,1}},{{1,0,1,1,0,1}}});
    Vec3 vertices[6];for(int f=0;f<6;++f){const double a=(2*f+1)*pi/6;vertices[f]={radius*std::cos(a),radius*std::sin(a),0};}
    for(int f=0;f<8;++f){auto& face=p->faces[f];face.engine=f<6?0:6;
        if(f<6){face.origin=vertices[f];face.origin.z=-height*.5;face.e1=(vertices[(f+1)%6]-vertices[f])*(1/radius);
            face.e2={0,0,1};face.e3=face.e1.cross(face.e2);
            const double c=std::cos(f*pi/3),s=std::sin(f*pi/3);face.rotation={{c,-s,0,s,c,0,0,0,1}};
        }else{face.e1=vertices[0]*(1/radius);face.e2={-face.e1.y,face.e1.x,0};face.e3={0,0,1};
            face.origin=(vertices[0]+vertices[1])*(-1);face.origin.z=(f==6?.5:-.5)*height;
            if(f==7)face.rotation={{1,0,0,0,-1,0,0,0,-1}};
        }
    }
    std::vector<Vec3> points;std::vector<int> point_face(count,-1);
    std::map<Key,int> index;
    const double quantum=1e-8;
    const auto key=[&](Vec3 x){return Key{{std::llround(x.x/quantum),std::llround(x.y/quantum),std::llround(x.z/quantum)}};};
    for(int i=0;i<count;++i){const Vec3 x(xyz[3*i],xyz[3*i+1],xyz[3*i+2]);points.push_back(x);
        if(!index.emplace(key(x),i).second)throw std::runtime_error("duplicate CPU fixture point");
        int f=-1;if(std::abs(x.z-height*.5)<1e-9)f=6;else if(std::abs(x.z+height*.5)<1e-9)f=7;
        else for(int j=0;j<6;++j)if(std::abs((x-p->faces[j].origin).dot(p->faces[j].e3))<1e-9){f=j;break;}
        if(f<0)throw std::runtime_error("CPU fixture point off prism");
        auto& face=p->faces[f];const auto& l=f<6?p->side:p->cap;const Vec3 d=x-face.origin;
        const double b=d.dot(face.e2)/l.v.y,a=(d.dot(face.e1)-b*l.v.x)/l.u.x;int slot=-1;
        for(size_t ch=0;ch<l.fractional.size();++ch){const double u=a-l.fractional[ch][0],v=b-l.fractional[ch][1];
            const int ix=std::llround(u),iy=std::llround(v);
            if(ix>=0&&ix<l.nx&&iy>=0&&iy<l.ny&&std::abs(u-ix)<1e-8&&std::abs(v-iy)<1e-8){slot=ch*l.nx*l.ny+ix*l.ny+iy;break;}}
        if(slot<0)throw std::runtime_error("CPU fixture quadrature lattice mismatch");
        face.global.push_back(i);face.slot.push_back(slot);point_face[i]=f;
    }
    const auto image=[&](int i,const Matrix& r){const Vec3 x=rotate(r,points[i]);const Key center=key(x);int found=-1;
        for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)for(int c=-1;c<=1;++c){
            const auto it=index.find(Key{{center[0]+a,center[1]+b,center[2]+c}});
            if(it!=index.end()&&(points[it->second]-x).norm()<1e-10){
                if(found!=-1)throw std::runtime_error("ambiguous fixture rotation");
                found=it->second;}}
        if(found<0)throw std::runtime_error("CPU fixture not closed under D6h");
        return found;};
    for(int f=0;f<8;++f){auto& face=p->faces[f];
        auto slots=face.slot;std::sort(slots.begin(),slots.end());
        if(std::adjacent_find(slots.begin(),slots.end())!=slots.end())throw std::runtime_error("duplicate fixture lattice slot");
        if(f==face.engine){face.targets=face.global;for(int i=0;i<count;++i)if(point_face[i]!=f)face.other.push_back(i);
            tree(face.fine,k,face.other.size(),face.targets.size(),true);
            tree(face.middle,k,face.other.size(),face.targets.size(),false);
            tree(face.coarse,k,face.other.size(),face.targets.size(),false);
        }else{const auto& engine=p->faces[face.engine];
            for(int i:engine.other)face.other.push_back(image(i,face.rotation));
            for(int i:engine.targets)face.targets.push_back(image(i,face.rotation));
        }
        auto targets=face.targets,global=face.global;std::sort(targets.begin(),targets.end());std::sort(global.begin(),global.end());
        if(targets!=global)throw std::runtime_error("CPU fixture rotated target bijection");
    }
}
PrismFaceFFT::HostView PrismFaceFFT::host_view()const{
    if(!p)throw std::runtime_error("CPU face fixture not initialized");
    HostView view{};view.k=p->k;view.target_faces=true;
    const auto lattice=[](const physical_face_fixture::Layout& l){return HostView::Lattice{l.nx,l.ny,{{l.u.x,l.u.y,l.u.z}},{{l.v.x,l.v.y,l.v.z}},&l.offsets};};
    view.side=lattice(p->side);view.cap=lattice(p->cap);
    for(int f=0;f<8;++f){const auto& face=p->faces[f];const auto& engine=p->faces[face.engine];auto& to=view.faces[f];
        to.frame={{face.e1.x,face.e1.y,face.e1.z,face.e2.x,face.e2.y,face.e2.z,face.e3.x,face.e3.y,face.e3.z}};
        to.rotation=face.rotation;to.global=&face.global;to.slot=&face.slot;to.other=&face.other;to.cross_global=&face.targets;
        to.engine=face.engine;to.fine=&engine.fine;to.middle=&engine.middle;to.coarse=&engine.coarse;
    }
    return view;
}
inline void initialize_physical_face_fixture(MullerFmmOperator& op,int ref,double ka){
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    op.mesh=build_muller_p2_mesh(regular_prism(6,10./7,ref,1.,0,false,true),options);
    op.current_dofs=op.mesh.current_dofs();op.system_dofs=op.mesh.system_dofs();
    op.k_exterior=ka;op.k_interior=ka*cdouble(1.3116);op.epsilon_interior=cdouble(1.3116)*cdouble(1.3116);
    op.quadrature_order=7;
    std::vector<double> xyz;
    for(int order:{7,13}){const auto rule=tri_quadrature(order);auto& out=order==7?op.quadrature:op.mass_quadrature;
        for(size_t t=0;t<op.mesh.elements.size();++t)for(int q=0;q<rule.npts;++q){MullerFmmQuadraturePoint p;p.element=t;
            p.sample=evaluate_muller_frame(op.mesh,t,rule.pts[q][0],rule.pts[q][1]);p.weight=.5*rule.wts[q]*p.sample.jacobian;out.push_back(p);
            if(order==7)xyz.insert(xyz.end(),{p.sample.position.x,p.sample.position.y,p.sample.position.z});}}
    for(auto* f:{&op.fmm_exterior,&op.fmm_interior})
        physical_face_fixture::tree(*f,f==&op.fmm_exterior?op.k_exterior:op.k_interior,op.quadrature.size(),op.quadrature.size(),true);
    op.prism_fft_exterior.reset(new PrismFaceFFT);op.prism_fft_interior.reset(new PrismFaceFFT);
    op.prism_fft_exterior->init(xyz,ref,op.k_exterior,5,nullptr,10./7);
    op.prism_fft_interior->init(xyz,ref,op.k_interior,5,nullptr,10./7);
}
