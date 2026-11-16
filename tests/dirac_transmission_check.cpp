#include "dirac_transmission.h"
#include <iostream>
#include <random>
#include <iomanip>
using namespace bem_dirac;
using A=std::array<C,8>;
A product(const A&a,const A&b){A r{};for(int i=0;i<8;++i)for(int j=0;j<8;++j){int flips=0;
    for(int bit=0;bit<3;++bit)if(i&(1<<bit))flips+=__builtin_popcount(unsigned(j&((1<<bit)-1)));
    r[i^j]+=(flips%2?-1.:1.)*a[i]*b[j];}return r;}
A vector(const Vec3&v){A a{};a[1]=v.x;a[2]=v.y;a[4]=v.z;return a;}
double rel(const V&a,const V&b){double e=0,n=0;for(size_t j=0;j<a.size();++j){e+=std::norm(a[j]-b[j]);n+=std::norm(b[j]);}return std::sqrt(e/std::max(n,1e-300));}
Vec3 rotate(const Vec3&v,int reflection){const double c=std::cos(.37),s=std::sin(.37);return Vec3(reflection*(c*v.x-s*v.y),s*v.x+c*v.y,v.z);}
V transformed(const V&x,const Transmission&before,const Transmission&after,int reflection){
    V y(x.size());for(size_t i=0;i<before.panels().size();++i)for(int basis=0;basis<before.basis_count();++basis){const auto&a=before.panels()[i];const auto&b=after.panels()[i];
        const size_t offset=8*(i*before.basis_count()+basis);
        // Reflection reverses vertices 1 and 2: xi and eta exchange, so psi1
        // and odd modes psi4, psi7, psi9 change sign; others stay fixed.
        const double basis_sign=reflection<0&&(basis==1||basis==4||basis==7||basis==9)?-1.:1.;
        y[offset]=basis_sign*x[offset];y[offset+4]=basis_sign*double(reflection)*x[offset+4];
        const Vec3 source[]={rotate(a.normal,reflection),rotate(a.tangent,reflection),rotate(a.second,reflection)};
        const Vec3 target[]={b.normal,b.tangent,b.second};
        for(int j=0;j<3;++j)for(int k=0;k<3;++k){double q=target[j].dot(source[k]);
            y[offset+1+j]+=basis_sign*reflection*q*x[offset+1+k];y[offset+5+j]+=basis_sign*q*x[offset+5+k];}}
    return y;
}
int main()try{
    std::mt19937 gen(4351);std::normal_distribution<double> rand;
    auto real=[&](){return rand(gen);};auto complex=[&](){return C(real(),real());};double error=0;
    for(int rep=0;rep<500;++rep){Panel p;p.normal=Vec3(real(),real(),real()).normalized();Vec3 t(real(),real(),real());
        p.tangent=(t-p.normal*t.dot(p.normal)).normalized();p.second=p.normal.cross(p.tangent);A h;for(auto&v:h)v=complex();
        Kernel k;k.s=complex();for(auto&v:k.d)v=complex();
        A f{};const A n=vector(p.normal),ta=vector(p.tangent),s=vector(p.second);
        std::array<A,8> basis{};basis[0][0]=1.;basis[1]=product(ta,s);basis[2]=product(s,n);basis[3]=product(n,ta);
        basis[4]=product(product(n,ta),s);basis[5]=n;basis[6]=ta;basis[7]=s;
        for(int i=0;i<8;++i)for(int j=0;j<8;++j)f[i]+=basis[j][i]*h[j];
        A kg{};kg[0]=k.s;kg[1]=-k.d[0];kg[2]=-k.d[1];kg[4]=-k.d[2];
        const auto v=product(product(kg,n),f),out=kernel_action(k,p,h.data());
        A expected{{v[0],v[6],-v[5],v[3],v[7],v[1],v[2],v[4]}};
        double e=0,d=0;for(int a=0;a<8;++a){e+=std::norm(out[a]-expected[a]);d+=std::norm(expected[a]);}
        error=std::max(error,std::sqrt(e/d));
    }
    if(error>2e-14)throw std::runtime_error("independent Clifford product mismatch");
    const auto mesh=icosphere(1.,0);Transmission op(mesh,1.,1.,12,13,2.,64<<20,7);
    V x(op.size()),y(x.size()),same;for(auto&v:x)v=complex();op.apply(x.data(),y.data());
    const double identity=rel(y,x);if(identity>1e-13)throw std::runtime_error("matched-medium identity");
    op.apply(x.data(),x.data());if(rel(x,y)>1e-13)throw std::runtime_error("aliasing");
    Transmission loss(mesh,1.,C(1.3116,.1),12,13,2.,64<<20,7);
    V zero(loss.size()),az(zero.size(),C(1));loss.apply(zero.data(),az.data());for(auto v:az)if(v!=C(0))throw std::runtime_error("zero input");
    int rejected=0;for(int mode=0;mode<3;++mode)try{
        Mesh m=mesh;if(mode==0)m.tris.resize(m.tris.size()-3);if(mode==1)for(size_t j=0;j<m.tris.size();j+=3)std::swap(m.tris[j],m.tris[j+1]);
        Transmission invalid(m,1.,mode==2?C(1.3,-.1):C(1.3),12,13,2.,64<<20);
    }catch(const std::exception&){++rejected;}
    if(rejected!=3)throw std::runtime_error("validation gates");
    double covariance=0,p1_identity=0,simd_error=0,self_block_error=0,mixed_error=0,batch_error=0;
    for(int degree:{0,1,2,3}){Transmission base(mesh,1.,C(1.3116,.1),12,13,2.,64<<20,7,degree);
      {V v(base.size()),a(v.size()),b(v.size());for(auto&z:v)z=complex();
       for(int count:{1,3,8,16})for(int tile:{64,256}){
        V inputs(size_t(count)*base.size()),expected(inputs.size()),actual(inputs.size());
        for(auto&z:inputs)z=complex();
        for(int r=0;r<count;++r)base.apply(inputs.data()+r*base.size(),expected.data()+r*base.size());
        base.apply_batch(count,inputs.data(),actual.data(),tile);batch_error=std::max(batch_error,rel(actual,expected));
        base.apply_batch(count,inputs.data(),inputs.data(),tile);batch_error=std::max(batch_error,rel(inputs,expected));
       }
       base.apply(v.data(),a.data());base.apply_simd(v.data(),b.data());simd_error=std::max(simd_error,rel(b,a));
       base.prepare_mixed_cache(64<<20);base.apply_simd(v.data(),b.data(),true);mixed_error=std::max(mixed_error,rel(b,a));
       base.apply_simd(v.data(),v.data());simd_error=std::max(simd_error,rel(v,a));
       for(int group:{1,3}){const int width=base.basis_count()*8*group,count=(base.panels().size()+group-1)/group;
        V blocks(size_t(count)*width*width);base.diagonal_blocks(group,blocks.data());
        for(int block:{0,count-1}){std::fill(v.begin(),v.end(),C(0));for(int c=0;c<width&&block*width+c<base.size();++c)v[block*width+c]=complex();
         base.apply(v.data(),a.data());V local(width),expected(width);
         for(int r=0;r<width;++r){if(block*width+r<base.size())expected[r]=a[block*width+r];
          for(int c=0;c<width;++c){const auto z=blocks[(size_t(block)*width+r)*width+c];
           if(block*width+r>=base.size()||block*width+c>=base.size())if(z!=C(double(r==c)))throw std::runtime_error("block padding identity");
           if(block*width+c<base.size())local[r]+=z*v[block*width+c];}}
         self_block_error=std::max(self_block_error,rel(local,expected));}}}
      if(degree){Transmission matched(mesh,1.,1.,12,13,2.,64<<20,7,degree);V x(matched.size()),y(x.size());for(auto&v:x)v=complex();matched.apply(x.data(),y.data());p1_identity=std::max(p1_identity,rel(y,x));}
      for(int parity:{1,-1}){Mesh moved=mesh;for(auto&v:moved.verts)v=rotate(v,parity)+Vec3(.13,.27,-.31);
        if(parity==-1)for(size_t j=0;j<moved.tris.size();j+=3)std::swap(moved.tris[j+1],moved.tris[j+2]);
        Transmission other(moved,1.,C(1.3116,.1),12,13,2.,64<<20,7,degree);
        V input(base.size()),original(base.size());for(auto&v:input)v=complex();base.apply(input.data(),original.data());
        auto moved_input=transformed(input,base,other,parity),expected=transformed(original,base,other,parity);V actual(input.size());
        other.apply(moved_input.data(),actual.data());covariance=std::max(covariance,rel(actual,expected));}
    }
    if(p1_identity>1e-13)throw std::runtime_error("P1 matched-medium identity");
    if(simd_error>2e-12||self_block_error>2e-12)throw std::runtime_error("SIMD or self block disagrees with full scalar operator");
    if(mixed_error>2e-6)throw std::runtime_error("mixed cache error");
    if(batch_error>2e-12)throw std::runtime_error("batch action disagrees with scalar operator");
    if(covariance>1e-11)throw std::runtime_error("rotation/reflection covariance");
    double near_covariance=0;
    Transmission near_base(mesh,1.,C(1.3116,.1),12,13,2.,64<<20,13,3,true,1);
    for(int parity:{1,-1}){Mesh moved=mesh;for(auto&v:moved.verts)v=rotate(v,parity)+Vec3(.13,.27,-.31);
        if(parity==-1)for(size_t j=0;j<moved.tris.size();j+=3)std::swap(moved.tris[j+1],moved.tris[j+2]);
        Transmission other(moved,1.,C(1.3116,.1),12,13,2.,64<<20,13,3,true,1);
        V input(near_base.size()),original(input.size()),actual(input.size());for(auto&v:input)v=complex();
        auto moved_input=transformed(input,near_base,other,parity);
        for(int medium:{0,1}){
            near_base.near_correction(medium,input.data(),original.data());
            other.near_correction(medium,moved_input.data(),actual.data());
            const auto expected=transformed(original,near_base,other,parity);
            near_covariance=std::max(near_covariance,rel(actual,expected));}}
    if(near_covariance>1e-11)throw std::runtime_error("refined near-correction rotation/reflection covariance");
    std::cout<<std::setprecision(17)<<"{\"clifford_cases\":500,\"clifford_relative_error\":"<<error
      <<",\"matched_medium_identity_error\":"<<identity<<",\"P1_P2_P3_matched_medium_identity_error\":"<<p1_identity<<",\"rotation_reflection_P0_P1_P2_P3_error\":"<<covariance<<",\"near52_P3_rotation_reflection_error\":"<<near_covariance
      <<",\"SIMD_FP64_error\":"<<simd_error<<",\"mixed_storage_error\":"<<mixed_error<<",\"self_block_error\":"<<self_block_error<<",\"batch_FP64_error\":"<<batch_error<<",\"rejections\":"<<rejected<<"}\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
