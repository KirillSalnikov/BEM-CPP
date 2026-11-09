#include "octree.h"
#include <cassert>
#include <iostream>
int main() {
    // A concentrated cloud requests depth 7 without allocating a dense volume.
    std::vector<double> points(3*(1<<18));
    for(size_t i=0;i<points.size()/3;++i)
        for(int axis=0;axis<3;++axis)points[3*i+axis]=((i>>axis)&1)?1.:-1.;
    Octree tree;unsetenv("BEM_FMM_MAX_DEPTH");
    tree.build(points.data(),points.size()/3,points.data(),points.size()/3,1,3);
    assert(tree.max_level==6);
    setenv("BEM_FMM_MAX_DEPTH","7",1);
    tree.build(points.data(),points.size()/3,points.data(),points.size()/3,1,3);
    assert(tree.max_level==7 && tree.leaves.size()==8 && tree.nodes.size()<100);
    size_t count=0;for(int i:tree.leaves)count+=tree.nodes[i].pt_count;
    assert(count==2*points.size()/3);
    for(const char* invalid:{"0","8","bad","7x"}) {
        setenv("BEM_FMM_MAX_DEPTH",invalid,1);bool rejected=false;
        try{tree.build(points.data(),1,points.data(),1,1,3);}
        catch(const std::invalid_argument&){rejected=true;}
        assert(rejected);
    }
    unsetenv("BEM_FMM_MAX_DEPTH");std::cout<<"octree depth checks passed\n";
}
