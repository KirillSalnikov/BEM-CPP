#ifndef BEM_HOST_PRISM_FACE_FFT_H
#define BEM_HOST_PRISM_FACE_FFT_H
#include "host_fmm_band.h"
#include "host_fmm_batch.h"
#include "host_regular_face_fft.h"
#include "prism_face_fft.h"

namespace bem_cpu {
// Same-face FFT plus cross-face host FMM. Only immutable geometry/translation
// metadata is borrowed from the prepared GPU backend; all active arrays and
// FFT symbols below are CPU-owned, with no CUDA calls during evaluation.
class PrismFaceAction {
    PrismFaceFFT::HostView geometry_;
    const std::vector<Vec3>& points_;
    std::unique_ptr<RegularFaceAction> self_[2];
    std::vector<std::unique_ptr<FmmBand>> cross_[8];
    std::vector<std::unique_ptr<FmmBandBatch>> batches_[8];
    std::vector<std::vector<Six>> batch_charges_[2];
    std::vector<std::vector<Twelve>> batch_fields_[2];
    bool batched_=false;
    std::vector<C> input_[2],output_[2];
    std::vector<Six> charges_;
    std::vector<Twelve> fields_;
public:
    double self_seconds=0,cross_seconds=0,packing_seconds=0;
    double phase_seconds=0;
    size_t phase_evaluations=0;
    std::array<double,6> fmm_times{};
    PrismFaceAction(const PrismFaceFFT& prepared,const std::vector<Vec3>& points,size_t phase_budget,
                    bool batch_faces=false,int tile=32)
        :geometry_(prepared.host_view()),points_(points),batched_(batch_faces),
         charges_(batch_faces?0:points.size()),fields_(batch_faces?0:points.size()) {
        if(!geometry_.target_faces)throw std::invalid_argument("CPU face FFT requires target-face cross hierarchies");
        if(batched_)for(int f=0;f<8;++f)if(geometry_.faces[f].engine!=(f<6?0:6))
            throw std::invalid_argument("CPU face batches require canonical side/cap trees");
        for(int kind=0;kind<2;++kind){
            const auto& l=kind?geometry_.cap:geometry_.side;
            if(l.u[2]!=0||l.v[2]!=0)throw std::invalid_argument("CPU face FFT local plane");
            std::vector<std::array<double,2>> offsets;
            for(size_t i=0;i<l.offsets->size();i+=3){
                if((*l.offsets)[i+2]!=0)throw std::invalid_argument("CPU face FFT offset plane");
                offsets.push_back({{(*l.offsets)[i],(*l.offsets)[i+1]}});
            }
            self_[kind].reset(new RegularFaceAction(l.nx,l.ny,{{l.u[0],l.u[1]}},{{l.v[0],l.v[1]}},offsets,geometry_.k));
            input_[kind].resize(offsets.size()*6*l.nx*l.ny);output_[kind].resize(offsets.size()*12*l.nx*l.ny);
        }
        for(int f=0;f<8;++f){
            const auto& face=geometry_.faces[f];if(face.engine!=f)continue;
            for(const auto* band:{face.fine,face.middle,face.coarse}){
                cross_[f].emplace_back(new FmmBand(*band,points_,phase_budget,nullptr,face.other,face.cross_global,batched_));
                if(batched_)batches_[f].emplace_back(new FmmBandBatch(*cross_[f].back(),f<6?6:2,tile));
            }
        }
        if(batched_)for(int kind=0;kind<2;++kind){
            batch_charges_[kind].assign(kind?2:6,std::vector<Six>(points.size()));
            batch_fields_[kind].assign(kind?2:6,std::vector<Twelve>(points.size()));
        }
    }
    size_t workspace_bytes()const {
        size_t bytes=(charges_.size()*sizeof(Six)+fields_.size()*sizeof(Twelve));
        for(int kind=0;kind<2;++kind)bytes+=self_[kind]->workspace_bytes()+(input_[kind].size()+output_[kind].size())*sizeof(C);
        for(const auto& bands:cross_)for(const auto& band:bands)bytes+=band->workspace_bytes();
        for(const auto& bands:batches_)for(const auto& band:bands)bytes+=band->workspace_bytes();
        for(int kind=0;kind<2;++kind){
            for(const auto& v:batch_charges_[kind])bytes+=v.size()*sizeof(Six);
            for(const auto& v:batch_fields_[kind])bytes+=v.size()*sizeof(Twelve);
        }
        return bytes;
    }
    void add(const std::vector<Six>& charges,std::vector<Twelve>& output){
        if(charges.size()!=points_.size()||output.size()!=points_.size())throw std::invalid_argument("CPU prism field sizes");
        self_seconds=cross_seconds=packing_seconds=phase_seconds=0;phase_evaluations=0;fmm_times.fill(0);
        for(int f=0;f<8;++f){
            const auto& face=geometry_.faces[f];const auto& engine=geometry_.faces[face.engine];
            const int kind=f<6?0:1;const auto& l=kind?geometry_.cap:geometry_.side;
            const int cells=l.nx*l.ny;const auto& frame=face.frame;
            auto start=Clock::now();std::fill(input_[kind].begin(),input_[kind].end(),C(0));
#pragma omp parallel for schedule(static)
            for(size_t j=0;j<face.global->size();++j){
                const int global=(*face.global)[j],slot=(*face.slot)[j],channel=slot/cells,cell=slot%cells;
                for(int current=0;current<2;++current)for(int d=0;d<3;++d){
                    C value=0;for(int k=0;k<3;++k)value+=frame[3*d+k]*charges[global][3*current+k];
                    input_[kind][size_t(channel*6+3*current+d)*cells+cell]=value;
                }
            }
            packing_seconds+=elapsed(start);start=Clock::now();
            self_[kind]->evaluate(input_[kind].data(),output_[kind].data());self_seconds+=elapsed(start);start=Clock::now();
#pragma omp parallel for schedule(static)
            for(size_t j=0;j<face.global->size();++j){
                const int global=(*face.global)[j],slot=(*face.slot)[j],channel=slot/cells,cell=slot%cells;
                for(int current=0;current<2;++current)for(int d=0;d<3;++d)for(int k=0;k<3;++k){
                    const double curl=(d==1?1.:-1.)*(k==1?1.:-1.)*frame[3*(2-k)+(2-d)];
                    output[global][6*current+d]+=curl*output_[kind][size_t(channel*12+6*current+k)*cells+cell];
                    output[global][6*current+3+d]+=frame[3*k+d]*output_[kind][size_t(channel*12+6*current+3+k)*cells+cell];
                }
            }
            if(batched_){packing_seconds+=elapsed(start);continue;}
            // Gather each physical face's sources into its canonical FMM tree.
            std::fill(charges_.begin(),charges_.end(),Six{});std::fill(fields_.begin(),fields_.end(),Twelve{});
#pragma omp parallel for schedule(static)
            for(size_t j=0;j<face.other->size();++j){
                const int source=(*face.other)[j],canonical=(*engine.other)[j];
                for(int current=0;current<2;++current)for(int d=0;d<3;++d){
                    C value=0;for(int k=0;k<3;++k)value+=face.rotation[3*k+d]*charges[source][3*current+k];
                    charges_[canonical][3*current+d]=value;
                }
            }
            packing_seconds+=elapsed(start);start=Clock::now();
            for(auto& band:cross_[face.engine]){
                band->add(charges_,fields_);for(int j=0;j<6;++j)fmm_times[j]+=band->times[j];
            }
            cross_seconds+=elapsed(start);start=Clock::now();
#pragma omp parallel for schedule(static)
            for(size_t j=0;j<face.cross_global->size();++j){
                const int target=(*face.cross_global)[j],canonical=(*engine.cross_global)[j];
                for(int current=0;current<2;++current)for(int d=0;d<3;++d)for(int k=0;k<3;++k){
                    const double curl=(d==1?1.:-1.)*(k==1?1.:-1.)*face.rotation[3*(2-d)+(2-k)];
                    output[target][6*current+d]+=curl*fields_[canonical][6*current+k];
                    output[target][6*current+3+d]+=face.rotation[3*d+k]*fields_[canonical][6*current+3+k];
                }
            }
            packing_seconds+=elapsed(start);
        }
        if(batched_)for(int kind=0;kind<2;++kind){
            const int first=kind?6:0,count=kind?2:6;const auto& engine=geometry_.faces[first];
            auto start=Clock::now();auto& inputs=batch_charges_[kind];auto& outputs=batch_fields_[kind];
            for(int batch=0;batch<count;++batch){const auto& face=geometry_.faces[first+batch];
                std::fill(inputs[batch].begin(),inputs[batch].end(),Six{});
                std::fill(outputs[batch].begin(),outputs[batch].end(),Twelve{});
#pragma omp parallel for schedule(static)
                for(size_t j=0;j<face.other->size();++j){
                    const int source=(*face.other)[j],canonical=(*engine.other)[j];
                    for(int current=0;current<2;++current)for(int d=0;d<3;++d){C value=0;
                        for(int k=0;k<3;++k)value+=face.rotation[3*k+d]*charges[source][3*current+k];
                        inputs[batch][canonical][3*current+d]=value;
                    }
                }
            }
            packing_seconds+=elapsed(start);start=Clock::now();
            for(auto& band:batches_[first]){band->add(inputs,outputs);phase_seconds+=band->phase_seconds;phase_evaluations+=band->phase_evaluations;
                for(int j=0;j<6;++j)fmm_times[j]+=band->times[j];}
            cross_seconds+=elapsed(start);start=Clock::now();
            for(int batch=0;batch<count;++batch){const auto& face=geometry_.faces[first+batch];
#pragma omp parallel for schedule(static)
                for(size_t j=0;j<face.cross_global->size();++j){
                    const int target=(*face.cross_global)[j],canonical=(*engine.cross_global)[j];
                    for(int current=0;current<2;++current)for(int d=0;d<3;++d)for(int k=0;k<3;++k){
                        const double curl=(d==1?1.:-1.)*(k==1?1.:-1.)*face.rotation[3*(2-d)+(2-k)];
                        output[target][6*current+d]+=curl*outputs[batch][canonical][6*current+k];
                        output[target][6*current+3+d]+=face.rotation[3*d+k]*outputs[batch][canonical][6*current+3+k];
                    }
                }
            }
            packing_seconds+=elapsed(start);
        }
    }
};
}
#endif
