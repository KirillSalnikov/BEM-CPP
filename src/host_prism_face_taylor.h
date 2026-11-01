#ifndef BEM_HOST_PRISM_FACE_TAYLOR_H
#define BEM_HOST_PRISM_FACE_TAYLOR_H
#include "host_fmm_taylor.h"
#include "host_regular_face_fft.h"
#include "prism_face_fft.h"

namespace bem_cpu {
// Same disjoint self-face/cross-face partition as PrismFaceAction, for every
// Taylor coefficient. Geometry, angular grids and FMM topology stay frozen.
// Only geometry is borrowed; all FFT spectra and moment workspaces are owned.
template<size_t N> class TaylorPrismFaceAction {
    using Band=TaylorFmmBand<N>;
public:
    using Charge=typename Band::Charge;
    using Field=typename Band::Field;
private:
    PrismFaceFFT::HostView geometry_;
    const std::vector<Vec3>& points_;
    std::unique_ptr<RegularFaceAction> self_[2];
    std::vector<std::unique_ptr<Band>> cross_[8];
    std::array<std::vector<C>,N+1> input_[2],output_[2];
    std::vector<Charge> charges_;
    std::vector<Field> fields_;
public:
    double self_seconds=0,cross_seconds=0,packing_seconds=0;
    TaylorPrismFaceAction(const PrismFaceFFT& prepared,const std::vector<Vec3>& points,C slope)
        :TaylorPrismFaceAction(prepared.host_view(),points,slope){}
    TaylorPrismFaceAction(const PrismFaceFFT::HostView& geometry,const std::vector<Vec3>& points,C slope)
        :geometry_(geometry),points_(points),charges_(points.size()),fields_(points.size()) {
        static_assert(N<=16,"Prism FFT Taylor order");
        if(!geometry_.target_faces)throw std::invalid_argument("Taylor prism requires target-face cross trees");
        for(int kind=0;kind<2;++kind){
            const auto& l=kind?geometry_.cap:geometry_.side;
            if(l.u[2]!=0||l.v[2]!=0||!l.offsets||l.offsets->size()%3)
                throw std::invalid_argument("Taylor face local plane/offsets");
            std::vector<std::array<double,2>> offsets;
            for(size_t i=0;i<l.offsets->size();i+=3){
                if((*l.offsets)[i+2]!=0)throw std::invalid_argument("Taylor face offset plane");
                offsets.push_back({{(*l.offsets)[i],(*l.offsets)[i+1]}});
            }
            self_[kind].reset(new RegularFaceAction(l.nx,l.ny,{{l.u[0],l.u[1]}},{{l.v[0],l.v[1]}},offsets,geometry_.k,N,slope));
            for(size_t j=0;j<=N;++j){input_[kind][j].resize(offsets.size()*6*l.nx*l.ny);output_[kind][j].resize(offsets.size()*12*l.nx*l.ny);}
        }
        std::vector<bool> owned(points.size());
        for(int f=0;f<8;++f){const auto& face=geometry_.faces[f];const auto& l=f<6?geometry_.side:geometry_.cap;
            if(face.engine<0||face.engine>=8||!face.global||!face.slot||!face.other||!face.cross_global||face.global->size()!=face.slot->size())
                throw std::invalid_argument("Taylor face mapping");
            std::vector<bool> used(size_t(l.nx)*l.ny*l.offsets->size()/3);
            for(size_t j=0;j<face.global->size();++j){const int global=(*face.global)[j],slot=(*face.slot)[j];
                if(global<0||size_t(global)>=owned.size()||owned[global]||slot<0||size_t(slot)>=used.size()||used[slot])
                    throw std::invalid_argument("Taylor duplicate/invalid face point");
                owned[global]=true;used[slot]=true;
            }
            const auto& engine=geometry_.faces[face.engine];
            if(engine.engine!=face.engine||!engine.other||!engine.cross_global||face.other->size()!=engine.other->size()||face.cross_global->size()!=engine.cross_global->size())
                throw std::invalid_argument("Taylor canonical cross mapping");
            std::vector<bool> target(points.size()),source(points.size()),self(points.size());
            for(int p:*face.global)self[p]=true;
            for(int p:*face.cross_global){
                if(p<0||size_t(p)>=points.size()||target[p]||!self[p])throw std::invalid_argument("Taylor invalid cross target");
                target[p]=true;
            }
            for(int p:*face.other){
                if(p<0||size_t(p)>=points.size()||source[p]||self[p])throw std::invalid_argument("Taylor invalid cross source");
                source[p]=true;
            }
            for(size_t p=0;p<points.size();++p)if(target[p]!=self[p]||source[p]==self[p])
                throw std::invalid_argument("Taylor cross partition does not cover point pairs");
            if(face.engine!=f)continue;
            for(const auto* band:{face.fine,face.middle,face.coarse})if(band&&band->initialized)
                cross_[f].emplace_back(new Band(*band,points_,slope,face.other,face.cross_global));
            if(cross_[f].empty())throw std::invalid_argument("Taylor face has no cross bands");
        }
        if(std::find(owned.begin(),owned.end(),false)!=owned.end())throw std::invalid_argument("Taylor uncovered face point");
    }
    size_t workspace_bytes()const{
        size_t bytes=charges_.size()*sizeof(Charge)+fields_.size()*sizeof(Field);
        for(int kind=0;kind<2;++kind){bytes+=self_[kind]->workspace_bytes();
            for(size_t j=0;j<=N;++j)bytes+=(input_[kind][j].size()+output_[kind][j].size())*sizeof(C);}
        for(const auto& bands:cross_)for(const auto& band:bands)bytes+=band->workspace_bytes();
        return bytes;
    }
    void add(const std::vector<Charge>& charges,std::vector<Field>& output){
        if(charges.size()!=points_.size()||output.size()!=points_.size())throw std::invalid_argument("Taylor prism field sizes");
        self_seconds=cross_seconds=packing_seconds=0;
        for(int f=0;f<8;++f){
            const auto& face=geometry_.faces[f];const auto& engine=geometry_.faces[face.engine];
            const int kind=f<6?0:1;const auto& l=kind?geometry_.cap:geometry_.side;
            const int cells=l.nx*l.ny;const auto& frame=face.frame;
            auto start=Clock::now();for(auto& v:input_[kind])std::fill(v.begin(),v.end(),C(0));
#pragma omp parallel for schedule(static)
            for(size_t p=0;p<face.global->size();++p){
                const int global=(*face.global)[p],slot=(*face.slot)[p],channel=slot/cells,cell=slot%cells;
                for(size_t j=0;j<=N;++j)for(int current=0;current<2;++current)for(int d=0;d<3;++d){C value=0;
                    for(int k=0;k<3;++k)value+=frame[3*d+k]*charges[global][j][3*current+k];
                    input_[kind][j][size_t(channel*6+3*current+d)*cells+cell]=value;
                }
            }
            packing_seconds+=elapsed(start);start=Clock::now();
            const C* inputs[N+1];C* outputs[N+1];
            for(size_t j=0;j<=N;++j){inputs[j]=input_[kind][j].data();outputs[j]=output_[kind][j].data();}
            self_[kind]->evaluate_history(inputs,outputs);self_seconds+=elapsed(start);start=Clock::now();
#pragma omp parallel for schedule(static)
            for(size_t p=0;p<face.global->size();++p){
                const int global=(*face.global)[p],slot=(*face.slot)[p],channel=slot/cells,cell=slot%cells;
                for(size_t j=0;j<=N;++j)for(int current=0;current<2;++current)for(int d=0;d<3;++d)for(int k=0;k<3;++k){
                    const double curl=(d==1?1.:-1.)*(k==1?1.:-1.)*frame[3*(2-k)+(2-d)];
                    output[global][j][6*current+d]+=curl*output_[kind][j][size_t(channel*12+6*current+k)*cells+cell];
                    output[global][j][6*current+3+d]+=frame[3*k+d]*output_[kind][j][size_t(channel*12+6*current+3+k)*cells+cell];
                }
            }
            std::fill(charges_.begin(),charges_.end(),Charge{});std::fill(fields_.begin(),fields_.end(),Field{});
#pragma omp parallel for schedule(static)
            for(size_t p=0;p<face.other->size();++p){const int source=(*face.other)[p],canonical=(*engine.other)[p];
                for(size_t j=0;j<=N;++j)for(int current=0;current<2;++current)for(int d=0;d<3;++d){C value=0;
                    for(int k=0;k<3;++k)value+=face.rotation[3*k+d]*charges[source][j][3*current+k];
                    charges_[canonical][j][3*current+d]=value;
                }
            }
            packing_seconds+=elapsed(start);start=Clock::now();
            for(auto& band:cross_[face.engine])band->add(charges_,fields_);
            cross_seconds+=elapsed(start);start=Clock::now();
#pragma omp parallel for schedule(static)
            for(size_t p=0;p<face.cross_global->size();++p){const int target=(*face.cross_global)[p],canonical=(*engine.cross_global)[p];
                for(size_t j=0;j<=N;++j)for(int current=0;current<2;++current)for(int d=0;d<3;++d)for(int k=0;k<3;++k){
                    const double curl=(d==1?1.:-1.)*(k==1?1.:-1.)*face.rotation[3*(2-d)+(2-k)];
                    output[target][j][6*current+d]+=curl*fields_[canonical][j][6*current+k];
                    output[target][j][6*current+3+d]+=face.rotation[3*d+k]*fields_[canonical][j][6*current+3+k];
                }
            }
            packing_seconds+=elapsed(start);
        }
    }
};
}
#endif
