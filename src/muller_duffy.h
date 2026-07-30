#ifndef BEM_MULLER_DUFFY_H
#define BEM_MULLER_DUFFY_H

#include <vector>

enum class MullerDuffyAdjacency {
    Coincident,
    EdgeAdjacent,
    VertexAdjacent
};

struct MullerDuffyPoint {
    double test_xi;
    double test_eta;
    double trial_xi;
    double trial_eta;
    double weight;
};

std::vector<MullerDuffyPoint> muller_duffy_rule(
    int order, MullerDuffyAdjacency adjacency);

void muller_duffy_remap_shared_vertex(
    double& xi, double& eta, int local_vertex);

void muller_duffy_remap_shared_edge(
    double& xi, double& eta,
    int first_local_vertex, int second_local_vertex);

#endif
