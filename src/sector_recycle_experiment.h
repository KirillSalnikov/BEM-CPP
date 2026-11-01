#pragma once
#include "streaming_sector_gmres.h"
#include "sector_ritz_experiment.h"
namespace bem_streaming {
class ExperimentalRecycleSpace {
    using Vector=std::vector<Complex>;
    int size_,capacity_;
    std::vector<Vector> solutions_,images_;
    bool dirty_=false;
    std::uint64_t generation_=0;
    static std::uint64_t hash_bytes(std::uint64_t h,const void* pointer,size_t bytes) {
        const auto* data=static_cast<const unsigned char*>(pointer);
        for(size_t i=0;i<bytes;++i) {h^=data[i];h*=1099511628211ULL;}
        return h;
    }
public:
    ExperimentalRecycleSpace(int size,int capacity):size_(size),capacity_(capacity) {
        if(size<1 || capacity<0 || capacity>64)throw std::invalid_argument("invalid recycle dimensions");
    }
    int size() const {return size_;}
    int rank() const {return int(images_.size());}
    int capacity() const {return capacity_;}
    bool dirty() const {return dirty_;}
    std::uint64_t generation() const {return generation_;}
    double storage_gib() const {return 2.*rank()*size_*sizeof(Complex)/(1024.*1024.*1024.);}
    void clear() {solutions_.clear();images_.clear();dirty_=true;++generation_;}

    void correct(Complex* solution,Complex* residual) const {
        for(int pass=0;pass<2;++pass)for(int j=0;j<rank();++j) {
            const Complex a=dot(images_[j].data(),residual,size_);
#pragma omp parallel for schedule(static)
            for(int i=0;i<size_;++i) {
                solution[i]+=a*solutions_[j][i];residual[i]-=a*images_[j][i];
            }
        }
    }
    void project(Complex* image,int column,int stride,std::vector<Complex>& couplings) const {
        for(int pass=0;pass<2;++pass)for(int j=0;j<rank();++j) {
            const Complex a=dot(images_[j].data(),image,size_);
            couplings[size_t(j)*stride+column]+=a;
#pragma omp parallel for schedule(static)
            for(int i=0;i<size_;++i)image[i]-=a*images_[j][i];
        }
    }
    void subtract_coupled_update(Complex* update,const Complex* coefficients,int count,
        const std::vector<Complex>& couplings,int stride,int original_rank) const {
        std::vector<Complex> values(original_rank,0.);
        for(int j=0;j<original_rank;++j)for(int c=0;c<count;++c)
            values[j]+=couplings[size_t(j)*stride+c]*coefficients[c];
#pragma omp parallel for schedule(static)
        for(int i=0;i<size_;++i)for(int j=0;j<original_rank;++j)
            update[i]-=values[j]*solutions_[j][i];
    }

    // Keep slow directions learned from different RHS, rather than freezing
    // the first RHS's symmetry sectors. Each update adds at most eight new
    // directions after the first extraction. Dropping orthonormal image
    // columns preserves the remaining A_inner U ~= C relations.
    int learn(const std::vector<Complex>& basis,const std::vector<Complex>& h,
        int stride,int used,const Action& inverse,const std::vector<Complex>& couplings,
        int projected_rank) {
        return learn_strided(basis.data(),size_,h,stride,used,inverse,couplings,projected_rank);
    }
    // A sector occupies a contiguous segment of every full Arnoldi vector.
    // Read it in place instead of copying another restart-sized basis.
    int learn_strided(const Complex* basis,size_t leading_dimension,const std::vector<Complex>& h,
        int stride,int used,const Action& inverse,const std::vector<Complex>& couplings,
        int projected_rank) {
        if(capacity_==0 || used<2)return 0;
        if(leading_dimension<size_t(size_))throw std::invalid_argument("recycle basis stride");
        const int wanted=rank()==0?capacity_:std::min(8,capacity_);
        std::vector<Complex> coefficients;
        const int candidates=bem_sector_experiment::harmonic_ritz_coefficients(h,stride,used,wanted,coefficients);
        if(candidates==0)return 0;
        Vector vector(size_),solution(size_),image(size_);
        int added=0;
        for(int k=0;k<candidates;++k) {
            const Complex* coefficient=coefficients.data()+size_t(k)*used;
            std::vector<Complex> image_coefficient(used+1,0.);
            for(int row=0;row<=used;++row)for(int col=0;col<used;++col)
                image_coefficient[row]+=h[size_t(row)*stride+col]*coefficient[col];
#pragma omp parallel for schedule(static)
            for(int i=0;i<size_;++i) {
                Complex v=0.,c=0.;
                for(int col=0;col<used;++col)v+=coefficient[col]*basis[size_t(col)*leading_dimension+i];
                for(int row=0;row<=used;++row)c+=image_coefficient[row]*basis[size_t(row)*leading_dimension+i];
                vector[i]=v;image[i]=c;
            }
            inverse(vector.data(),solution.data());
            // If Arnoldi used (I-CC*) A M^-1, its physical update is
            // M^-1 V y - U B y. Original columns stay in place until pruning.
            subtract_coupled_update(solution.data(),coefficient,used,couplings,stride,projected_rank);
            const double before=norm(image.data(),size_);
            for(int pass=0;pass<2;++pass)for(int j=0;j<rank();++j) {
                const Complex a=dot(images_[j].data(),image.data(),size_);
#pragma omp parallel for schedule(static)
                for(int i=0;i<size_;++i) {image[i]-=a*images_[j][i];solution[i]-=a*solutions_[j][i];}
            }
            const double length=norm(image.data(),size_);
            const double unorm=norm(solution.data(),size_);
            if(!std::isfinite(length) || !std::isfinite(unorm) || length<=1e-10*std::max(before,1e-300))continue;
#pragma omp parallel for schedule(static)
            for(int i=0;i<size_;++i) {image[i]/=length;solution[i]/=length;}
            images_.push_back(image);solutions_.push_back(solution);++added;
        }
        if(rank()>capacity_) {
            std::vector<int> order(rank());std::iota(order.begin(),order.end(),0);
            std::vector<double> score(rank());
            for(int j=0;j<rank();++j)score[j]=norm(solutions_[j].data(),size_);
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return score[a]>score[b];});
            std::vector<Vector> kept_u,kept_c;
            for(int j=0;j<capacity_;++j) {
                kept_u.push_back(std::move(solutions_[order[j]]));kept_c.push_back(std::move(images_[order[j]]));
            }
            solutions_.swap(kept_u);images_.swap(kept_c);
        }
        if(added) {dirty_=true;++generation_;}
        return added;
    }

    bool validate() const {
        for(int j=0;j<rank();++j) {
            for(const auto& x:solutions_[j])if(!std::isfinite(x.real()) || !std::isfinite(x.imag()))return false;
            for(const auto& x:images_[j])if(!std::isfinite(x.real()) || !std::isfinite(x.imag()))return false;
            for(int k=0;k<=j;++k)
                if(std::abs(dot(images_[j].data(),images_[k].data(),size_)-(j==k?1.:0.))>1e-8)return false;
        }
        return rank()>0;
    }
    // Exact builder signature binds mesh, material, binary and solver knobs.
    // Checksums and atomic replacement make a corrupt/interrupted cache a miss.
    bool load(const std::string& path,const std::string& signature) {
        if(capacity_==0 || signature.size()!=64)return false;
        std::ifstream stream(path.c_str(),std::ios::binary|std::ios::ate);
        if(!stream)return false;
        const auto file_size=stream.tellg();stream.seekg(0);
        char magic[16]={},stored_signature[64]={};
        std::uint64_t dimensions[3]={};
        stream.read(magic,16);stream.read(stored_signature,64);
        stream.read(reinterpret_cast<char*>(dimensions),sizeof(dimensions));
        const std::uint64_t count=dimensions[1];
        const char expected[16]="BEM_T_RECYCLE_1";
        if(!stream || std::memcmp(magic,expected,16) || std::string(stored_signature,64)!=signature ||
           dimensions[0]!=std::uint64_t(size_) || count==0 || count>std::uint64_t(capacity_) ||
           file_size!=std::streamoff(104+2*count*size_*sizeof(Complex)))return false;
        std::vector<Vector> u(count,Vector(size_)),c(count,Vector(size_));
        std::uint64_t checksum=1469598103934665603ULL;
        for(auto* list: {&u,&c})for(auto& v:*list) {
            stream.read(reinterpret_cast<char*>(v.data()),v.size()*sizeof(Complex));
            checksum=hash_bytes(checksum,v.data(),v.size()*sizeof(Complex));
        }
        if(!stream || checksum!=dimensions[2])return false;
        solutions_.swap(u);images_.swap(c);
        if(!validate()) {solutions_.swap(u);images_.swap(c);return false;}
        dirty_=false;++generation_;return true;
    }
    void save(const std::string& path,const std::string& signature) {
        if(!dirty_ || rank()==0 || path.empty() || signature.size()!=64)return;
        std::uint64_t checksum=1469598103934665603ULL;
        for(const auto* list: {&solutions_,&images_})for(const auto& v:*list)
            checksum=hash_bytes(checksum,v.data(),v.size()*sizeof(Complex));
        const std::string temporary=path+".tmp."+std::to_string(getpid());
        std::ofstream stream(temporary.c_str(),std::ios::binary|std::ios::trunc);
        const char magic[16]="BEM_T_RECYCLE_1";
        const std::uint64_t dimensions[3]={std::uint64_t(size_),std::uint64_t(rank()),checksum};
        stream.write(magic,16);stream.write(signature.data(),64);
        stream.write(reinterpret_cast<const char*>(dimensions),sizeof(dimensions));
        for(const auto* list: {&solutions_,&images_})for(const auto& v:*list)
            stream.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(Complex));
        stream.close();
        if(!stream || std::rename(temporary.c_str(),path.c_str())) {
            std::remove(temporary.c_str());throw std::runtime_error("could not save T recycle space");
        }
        dirty_=false;
    }
};
class ExperimentalSectorRecycleSpace {
    std::array<int,13> offsets_;
    std::vector<ExperimentalRecycleSpace> spaces_;
public:
    ExperimentalSectorRecycleSpace(const std::array<int,13>& offsets,int capacity,unsigned mask=4095):offsets_(offsets) {
        if(offsets[0]!=0 || capacity<0 || capacity>64 || mask>4095)throw std::invalid_argument("sector recycle dimensions");
        spaces_.reserve(12);
        for(int s=0;s<12;++s) {
            const int n=offsets[s+1]-offsets[s];
            if(n<0)throw std::invalid_argument("sector recycle offsets");
            spaces_.emplace_back(std::max(1,n),n && (mask&(1u<<s))?capacity:0);
        }
    }
    bool matches(const std::array<int,13>& offsets)const{return offsets==offsets_;}
    ExperimentalRecycleSpace& sector(int s){return spaces_.at(s);}
    int rank()const{int r=0;for(const auto& s:spaces_)r+=s.rank();return r;}
    double storage_gib()const{double b=0;for(const auto& s:spaces_)b+=s.storage_gib();return b;}
};

}
