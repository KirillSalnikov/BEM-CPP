#ifndef BEM_RWG_H
#define BEM_RWG_H

#include "mesh.h"
#include <vector>

struct RWG {
    int N;                      // number of RWG basis functions
    std::vector<int> tri_p;     // (N) plus triangle index
    std::vector<int> tri_m;     // (N) minus triangle index
    std::vector<Vec3> free_p;   // (N) free vertex of plus triangle
    std::vector<Vec3> free_m;   // (N) free vertex of minus triangle
    std::vector<double> length; // (N) edge length
    std::vector<double> area_p; // (N) area of plus triangle
    std::vector<double> area_m; // (N) area of minus triangle
};

RWG build_rwg(const Mesh& mesh);

#endif // BEM_RWG_H
