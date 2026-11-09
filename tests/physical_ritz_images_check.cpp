#include "physical_ritz_images.h"
#include <cstdio>
#include <limits>
#include <omp.h>
using namespace bem_tmatrix;
int main()try {
    omp_set_num_threads(2);int policies=0,rejections=0;
    for(auto key:{"BEM_TMATRIX_RITZ_IMAGE_MODEL","BEM_TMATRIX_RITZ_PRIOR","BEM_TMATRIX_SECTOR_GMRES",
                  "BEM_TMATRIX_D6H_MESH","BEM_TMATRIX_VERTICAL_PRECONDITIONER"})unsetenv(key);
    const auto default_config=bem_ritz::ImageConfig::from_environment();
    if(default_config.full_d6h || default_config.estimated_additional_bytes(100))throw std::runtime_error("default image changed");
    ++policies;
    const auto reject=[&]{bool refused=false;try{bem_ritz::ImageConfig::from_environment();}catch(const std::invalid_argument&){refused=true;}
        if(!refused)throw std::runtime_error("invalid image configuration accepted");++rejections;};
    for(auto name:{"","D6h","d6h_mixedx","originalx"}){setenv("BEM_TMATRIX_RITZ_IMAGE_MODEL",name,1);reject();}
    setenv("BEM_TMATRIX_RITZ_IMAGE_MODEL","d6h_strict",1);reject();
    setenv("BEM_TMATRIX_RITZ_PRIOR","",1);reject();setenv("BEM_TMATRIX_RITZ_PRIOR","fixture",1);
    for(auto key:{"BEM_TMATRIX_SECTOR_GMRES","BEM_TMATRIX_D6H_MESH","BEM_TMATRIX_VERTICAL_PRECONDITIONER"})setenv(key,"1",1);
    const auto config=bem_ritz::ImageConfig::from_environment();++policies;
    for(auto key:{"BEM_TMATRIX_SECTOR_GMRES","BEM_TMATRIX_D6H_MESH","BEM_TMATRIX_VERTICAL_PRECONDITIONER"}) {
        setenv(key,"0",1);reject();unsetenv(key);reject();setenv(key,"1",1);
    }
    bool overflow=false;try{config.estimated_additional_bytes(std::numeric_limits<size_t>::max());}
    catch(const std::exception&){overflow=true;}if(!overflow)throw std::runtime_error("image budget overflow accepted");++rejections;
    MullerP2BuildOptions options;options.edge_mode=MullerEdgeMode::HDivBdm1;
    const auto mesh=build_muller_p2_mesh(regular_prism(6,10./7,0,1.,0,false,true),options);
    D6hOrbit orbit;orbit.build(mesh);PrismGroupMaps group;group.build(orbit);const int n=mesh.system_dofs();
    Vector matrix(size_t(n)*n),x(n),actual(n),expected(n);size_t mixed=0,strict=0;
    for(int i=0;i<n;++i){x[i]=Complex(std::sin(.17*i),std::cos(.13*i));
        for(int j=0;j<n;++j)matrix[size_t(i)*n+j]=Complex(std::sin(.003*(i+1)*(j+3)),std::cos(.007*(i+2)*(j+1)))/double(n);}
    const Vector saved=matrix;
    const auto dense=[&](const Complex* a,Complex* b,double scale){for(int i=0;i<n;++i){Complex value=0;
        for(int j=0;j<n;++j)value+=matrix[size_t(i)*n+j]*a[j];b[i]=scale*value;}};
    const bem_streaming::Action approximate=[&](const Complex* a,Complex* b){++mixed;dense(a,b,1.01);};
    const bem_streaming::Action original=[&](const Complex* a,Complex* b){++strict;dense(a,b,1.);};
    bem_ritz::PhysicalImages raw(mesh,orbit,default_config,approximate,original);raw.apply(x.data(),actual.data());dense(x.data(),expected.data(),1.01);
    if(actual!=expected || mixed!=1 || strict || raw.mixed_actions()!=1 || raw.strict_actions() || raw.storage_bytes())
        throw std::runtime_error("default image action or accounting changed");
    bem_ritz::PhysicalImages model(mesh,orbit,config,approximate,original);model.apply(x.data(),actual.data());
    // Independent matrix-level average over the stored signed permutations.
    std::vector<int> vertical,sign;D6hOrbit::vertical_image_map(mesh,vertical,sign);
    std::fill(expected.begin(),expected.end(),Complex(0));
    for(int i=0;i<n;++i)for(int j=0;j<n;++j){Complex value=0;
        for(int v=0;v<2;++v)for(int g=0;g<12;++g){int a=group.image[g][i],b=group.image[g][j];int s=group.sign[g][i]*group.sign[g][j];
            if(v){s*=sign[a]*sign[b];a=vertical[a];b=vertical[b];}
            s*=(v?-1:1)*(g>=6?-1:1);value+=double(s)*matrix[size_t(a)*n+b]/24.;}
        expected[i]+=value*x[j];}
    double delta=0,norm=0;for(int i=0;i<n;++i){delta+=std::norm(actual[i]-expected[i]);norm+=std::norm(expected[i]);}
    const double error=std::sqrt(delta/norm);
    if(error>2e-12 || mixed!=1 || strict!=24 || model.strict_actions()!=24 || model.mixed_actions() ||
       model.storage_bytes()>config.estimated_additional_bytes(n) || matrix!=saved)
        throw std::runtime_error("strict model dispatch, accounting, storage or original matrix failed");
    const Vector averaged=expected;
    setenv("BEM_TMATRIX_RITZ_IMAGE_MODEL","strict",1);
    const auto strict_config=bem_ritz::ImageConfig::from_environment();++policies;
    bem_ritz::PhysicalImages precise(mesh,orbit,strict_config,approximate,original);
    precise.apply(x.data(),actual.data());dense(x.data(),expected.data(),1.);
    if(actual!=expected || precise.mixed_actions() || precise.strict_actions()!=1 || precise.storage_bytes() ||
       strict_config.full_d6h || !strict_config.strict_action || strict!=25 || mixed!=1)
        throw std::runtime_error("unaveraged strict image dispatch or accounting changed");
    setenv("BEM_TMATRIX_RITZ_IMAGE_MODEL","d6h_mixed",1);
    const auto mixed_config=bem_ritz::ImageConfig::from_environment();++policies;
    bem_ritz::PhysicalImages approximate_model(mesh,orbit,mixed_config,approximate,original);
    approximate_model.apply(x.data(),actual.data());delta=norm=0;
    for(int i=0;i<n;++i){delta+=std::norm(actual[i]-1.01*averaged[i]);norm+=std::norm(averaged[i]);}
    const double mixed_error=std::sqrt(delta/norm);
    if(mixed_error>2e-12 || mixed!=25 || strict!=25 || approximate_model.mixed_actions()!=24 ||
       approximate_model.strict_actions() || !mixed_config.full_d6h || mixed_config.strict_action ||
       approximate_model.storage_bytes()>mixed_config.estimated_additional_bytes(n))
        throw std::runtime_error("mixed D6h image action or accounting failed");
    for(auto name:{"strict","d6h_mixed"}) {
        setenv("BEM_TMATRIX_RITZ_IMAGE_MODEL",name,1);
        unsetenv("BEM_TMATRIX_RITZ_PRIOR");reject();setenv("BEM_TMATRIX_RITZ_PRIOR","fixture",1);
        setenv("BEM_TMATRIX_VERTICAL_PRECONDITIONER","0",1);reject();setenv("BEM_TMATRIX_VERTICAL_PRECONDITIONER","1",1);
    }
    bool refused=false;try{bem_ritz::PhysicalImages invalid(mesh,orbit,strict_config,approximate,{});}
    catch(const std::invalid_argument&){refused=true;}if(!refused)throw std::runtime_error("missing strict action accepted");
    std::printf("PASS Ritz image policy accepted%d rejected%d; raw1/strict1/model24 actions; independent errors %.3e %.3e; bytes%zu\n",
        policies,rejections,error,mixed_error,model.storage_bytes());
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
