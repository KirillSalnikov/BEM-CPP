#include "host_muller_correction.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using C=std::complex<double>;
using L=std::complex<long double>;
struct Entry {int column;C k1,k2_epsilon,k2_mu;};
int main() try {
    std::mt19937 rng(715);std::uniform_real_distribution<double> uniform(-1,1);
    auto random=[&](){return C(uniform(rng),uniform(rng));};
    int cases=0;double worst=0;
    for (int threads:{1,4}) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        for (int n:{1,19,73}) for (double scale:{1e-6,1.,1e6}) {
            std::vector<int> offsets(n+1,0);std::vector<Entry> entries;
            std::vector<L> dense[3];for(auto& v:dense)v.resize(n*n);
            for(int row=0;row<n;++row) {
                // Empty rows, unsorted columns and duplicate entries.
                const int count=row%7==0?0:2*n/3;
                for(int j=0;j<count;++j) {
                    const int col=rng()%n;Entry e{col,random()*scale,random()*scale,random()*scale};
                    entries.push_back(e);dense[0][row*n+col]+=L(e.k1);
                    dense[1][row*n+col]+=L(e.k2_epsilon);dense[2][row*n+col]+=L(e.k2_mu);
                }
                offsets[row+1]=entries.size();
            }
            std::vector<C> input(2*n);for(auto& v:input)v=random();
            std::array<std::vector<C>,4> outputs,initial;
            for(auto& v:initial){v.resize(n);for(auto& x:v)x=random()*scale;}
            outputs=initial;
            bem_host::correct_current_pair(n,offsets.data(),entries.data(),input.data(),
                outputs[0].data(),outputs[1].data(),outputs[2].data(),outputs[3].data());
            for(int row=0;row<n;++row)for(int block=0;block<4;++block) {
                const int kernel=block<2?0:block-1,offset=(block==1||block==2)?n:0;
                L exact=initial[block][row];long double scale_sum=std::abs(exact);
                for(int col=0;col<n;++col){const L term=dense[kernel][row*n+col]*L(input[offset+col]);exact+=term;scale_sum+=std::abs(term);}
                const double error=double(std::abs(L(outputs[block][row])-exact)/std::max(scale_sum,1e-300L));
                if(!std::isfinite(error)||error>3e-15)throw std::runtime_error("fused correction vs dense long-double contraction");
                worst=std::max(worst,error);++cases;
            }
        }
    }
    std::printf("PASS %d rows/blocks, 1 and 4 threads; max scaled error %.3e\n",cases,worst);
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
