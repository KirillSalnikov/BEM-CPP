#include "dirac_transmission.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <iomanip>
using namespace bem_dirac;

double relative_error(const V&a,const V&b){
    double e=0,n=0;for(size_t i=0;i<a.size();++i){e+=std::norm(a[i]-b[i]);n+=std::norm(b[i]);}
    return std::sqrt(e/std::max(n,1e-300));
}
template<class F> void must_reject(F f){
    bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("missing compact cache validation");
}
int main()try{
    const auto mesh=icosphere(1.,0);
    std::mt19937 gen(851);std::normal_distribution<double> normal;
    double worst=0;int cases=0;
    for(int degree:{0,1,2,3})for(int refinement:{0,1}){
        const int target=refinement?13:degree?7:1;
        Transmission full(mesh,3.,C(1.3116,.12),12,13,2.,64<<20,target,degree,false,refinement);
        Transmission compact(mesh,3.,C(1.3116,.12),12,13,2.,1,target,degree,false,refinement,true);
        const int width=full.basis_count()*8, ids[]={0,4,19},duplicate[]={0,0};
        V x(full.size()),expected(x.size()),dummy(x.size());
        for(auto&z:x)z=C(normal(gen),normal(gen));
        // Deferred construction must not allocate the full cache even with a
        // budget too small for that cache. Failed builds leave it reusable.
        if(compact.kernel_bytes()!=0)throw std::runtime_error("deferred allocation");
        must_reject([&](){compact.apply(x.data(),dummy.data());});
        must_reject([&](){compact.build_rows(2,duplicate,64<<20);});
        must_reject([&](){compact.build_rows(3,ids,1);});
        const size_t bytes=2*3*full.basis_count()*size_t(full.size()/8)*sizeof(Kernel);
        compact.build_rows(3,ids,bytes);
        if(compact.kernel_bytes()!=bytes)throw std::runtime_error("compact storage size");
        must_reject([&](){compact.build_rows(3,ids,bytes);});
        full.apply(x.data(),expected.data());
        std::vector<int> rows;for(int panel:ids)for(int b=0;b<full.basis_count();++b)rows.push_back(panel*full.basis_count()+b);
        V selected(3*width),scalar(selected.size()),simd(selected.size()),direct(selected.size());
        for(int i=0;i<3;++i)std::copy_n(expected.data()+ids[i]*width,width,selected.data()+i*width);
        compact.apply_rows_reference(x.data(),rows.size(),rows.data(),scalar.data());
        compact.apply_rows(1,x.data(),rows.size(),rows.data(),simd.data(),64);
        compact.apply_direct_panels(x.data(),3,ids,direct.data());
        for(const V*value:{&scalar,&simd,&direct})worst=std::max(worst,relative_error(*value,selected));
        for(int count:{1,2,3,8,9,16}){
            V batch(size_t(count)*full.size()),out(size_t(count)*selected.size()),oracle(out.size());
            for(auto&z:batch)z=C(normal(gen),normal(gen));
            compact.apply_rows_reference_batch(count,batch.data(),rows.size(),rows.data(),out.data());
            for(int r=0;r<count;++r){
                full.apply(batch.data()+size_t(r)*full.size(),dummy.data());
                for(int i=0;i<3;++i)std::copy_n(dummy.data()+ids[i]*width,width,oracle.data()+size_t(r)*selected.size()+i*width);
            }
            worst=std::max(worst,relative_error(out,oracle));
        }
        // Fresh direct quadrature also works on rows absent from the cache.
        const int absent=1;V fresh(width),expected_absent(width);
        compact.apply_direct_panels(x.data(),1,&absent,fresh.data());
        std::copy_n(expected.data()+width,width,expected_absent.data());
        worst=std::max(worst,relative_error(fresh,expected_absent));
        V blocks(size_t(mesh.nt())*width*width),small(size_t(3)*width*width),selected_blocks(small.size());
        full.diagonal_blocks(1,blocks.data());compact.self_blocks_selected(3,ids,small.data());
        for(int i=0;i<3;++i)std::copy_n(blocks.data()+size_t(ids[i])*width*width,width*width,selected_blocks.data()+size_t(i)*width*width);
        worst=std::max(worst,relative_error(small,selected_blocks));
        // An arbitrary coefficient submatrix must reproduce full action for
        // a sparse input, including columns from unretained source panels.
        const int columns[]={full.size()/8-1,0,full.basis_count()};
        V matrix(size_t(rows.size())*8*24),source(24),sparse(full.size()),subexpected(3*width),subactual(3*width);
        for(auto&z:source)z=C(normal(gen),normal(gen));
        compact.submatrix(rows.size(),rows.data(),3,columns,matrix.data());
        for(int j=0;j<3;++j)for(int c=0;c<8;++c)sparse[8*columns[j]+c]+=source[8*j+c];
        full.apply(sparse.data(),dummy.data());
        for(int i=0;i<3;++i)std::copy_n(dummy.data()+ids[i]*width,width,subexpected.data()+i*width);
        for(size_t i=0;i<subactual.size();++i)for(int j=0;j<24;++j)subactual[i]+=matrix[i*24+j]*source[j];
        worst=std::max(worst,relative_error(subactual,subexpected));
        const int cb=full.basis_count(),nc=3*cb*8;
        std::vector<double> basis(3*cb*cb),components(3*64);
        for(auto&v:basis){v=normal(gen);}for(auto&v:components){v=normal(gen);}
        V transformed(size_t(3*width)*nc),q(nc),lifted(full.size()),result(3*width);
        for(auto&v:q)v=C(normal(gen),normal(gen));
        compact.transformed_columns(rows.size(),rows.data(),3,ids,cb,basis.data(),components.data(),transformed.data());
        for(int j=0;j<3;++j)for(int b=0;b<cb;++b)for(int c=0;c<8;++c)
            for(int a=0;a<cb;++a)for(int r=0;r<8;++r)
                lifted[ids[j]*width+8*b+c]+=basis[(j*cb+b)*cb+a]*components[64*j+8*c+r]*q[(j*cb+a)*8+r];
        full.apply(lifted.data(),dummy.data());
        for(int i=0;i<3;++i)std::copy_n(dummy.data()+ids[i]*width,width,subexpected.data()+i*width);
        for(size_t i=0;i<result.size();++i)for(int j=0;j<nc;++j)result[i]+=transformed[i*nc+j]*q[j];
        worst=std::max(worst,relative_error(result,subexpected));
        const int missing_row=full.basis_count();
        must_reject([&](){compact.submatrix(1,&missing_row,3,columns,matrix.data());});
        const int invalid_column=full.size()/8;
        must_reject([&](){compact.submatrix(rows.size(),rows.data(),1,&invalid_column,matrix.data());});
        must_reject([&](){compact.apply_rows_reference(x.data(),1,&missing_row,scalar.data());});
        must_reject([&](){compact.apply_rows_reference_batch(2,x.data(),1,&missing_row,scalar.data());});
        for(int count:{0,17})must_reject([&](){compact.apply_rows_reference_batch(count,x.data(),rows.size(),rows.data(),scalar.data());});
        must_reject([&](){compact.apply_rows(1,x.data(),1,&missing_row,simd.data(),64);});
        must_reject([&](){compact.self_blocks_selected(1,&absent,small.data());});
        must_reject([&](){compact.apply(x.data(),dummy.data());});
        for(int count:{1,3}){
            V batch(size_t(count)*full.size()),result(batch.size());
            must_reject([&](){compact.apply_batch(count,batch.data(),result.data(),64);});
        }
        must_reject([&](){compact.cauchy(0,x.data(),dummy.data());});
        must_reject([&](){compact.prepare_mixed_cache(64<<20);});
        ++cases;
    }
    if(worst>2e-12)throw std::runtime_error("compact action disagrees with independent full construction");
    std::cout<<std::setprecision(17)<<"{\"cases\":"<<cases<<",\"max_error\":"<<worst<<"}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
