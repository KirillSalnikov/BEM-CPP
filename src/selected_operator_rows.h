#ifndef BEM_SELECTED_OPERATOR_ROWS_H
#define BEM_SELECTED_OPERATOR_ROWS_H
#include <algorithm>
#include <complex>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// C=S A and C*=A* S*: selected ORIGINAL equation rows. No row matrix is
// assembled. The caller supplies the same precision/backend for A and A*.
// Column-major blocks are streamed through one physical operator workspace.
class SelectedOperatorRows {
public:
    using Complex=std::complex<double>;
    using Action=std::function<void(const Complex*,Complex*)>;
private:
    int n_;
    std::vector<int> rows_;
    Action forward_,adjoint_;
public:
    SelectedOperatorRows(int n,std::vector<int> rows,Action forward,Action adjoint)
        :n_(n),rows_(std::move(rows)),forward_(std::move(forward)),adjoint_(std::move(adjoint)) {
        if(n<1||!forward_||!adjoint_)throw std::invalid_argument("selected rows dimensions/actions");
        std::vector<bool> seen(n,false);
        for(int j:rows_){if(j<0||j>=n||seen[j])throw std::invalid_argument("selected rows must be distinct and in range");seen[j]=true;}
    }
    int dofs()const{return n_;}
    int row_count()const{return int(rows_.size());}
    const std::vector<int>& indices()const{return rows_;}
    void apply(const Complex* input,int batch,Complex* output,bool adjoint=false)const {
        if(batch<1||batch>64||!input||!output)throw std::invalid_argument("selected rows block/buffers");
        const size_t in_rows=adjoint?rows_.size():size_t(n_),out_rows=adjoint?size_t(n_):rows_.size();
        if(in_rows>std::numeric_limits<size_t>::max()/size_t(batch))throw std::length_error("selected rows block size");
        // All input is retained before any output: arbitrary overlap is safe,
        // also when the adjoint expands a short block to many physical DOFs.
        const std::vector<Complex> saved(input,input+in_rows*batch);
        std::vector<Complex> x(n_),y(n_);
        for(int c=0;c<batch;++c) {
            if(adjoint){
                std::fill(x.begin(),x.end(),Complex(0));
                for(size_t j=0;j<rows_.size();++j)x[rows_[j]]=saved[size_t(c)*in_rows+j];
                if(rows_.empty())std::fill(y.begin(),y.end(),Complex(0));else adjoint_(x.data(),y.data());
                std::copy(y.begin(),y.end(),output+size_t(c)*out_rows);
            }else if(!rows_.empty()){
                forward_(saved.data()+size_t(c)*in_rows,y.data());
                for(size_t j=0;j<rows_.size();++j)output[size_t(c)*out_rows+j]=y[rows_[j]];
            }
        }
    }
};
#endif
