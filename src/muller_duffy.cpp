#include "muller_duffy.h"

#include <cmath>
#include <stdexcept>
#include <utility>

// Sauter-Schwab region maps follow the open-source Bempp-cl Duffy rule
// (MIT license), rewritten here for the P2 Muller reference triangle.

namespace {

struct GaussRule {
    std::vector<double> points;
    std::vector<double> weights;
};

GaussRule gauss_legendre_unit(int order)
{
    if (order < 1)
        throw std::invalid_argument("Gauss order must be positive");
    GaussRule rule;
    rule.points.resize(order);
    rule.weights.resize(order);
    const int half = (order + 1) / 2;
    for (int i = 0; i < half; i++) {
        double root = std::cos(
            std::acos(-1.0) * (i + 0.75) / (order + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 50; iteration++) {
            double p0 = 1.0;
            double p1 = root;
            for (int degree = 2; degree <= order; degree++) {
                const double next =
                    ((2.0 * degree - 1.0) * root * p1 -
                     (degree - 1.0) * p0) /
                    degree;
                p0 = p1;
                p1 = next;
            }
            derivative =
                order * (root * p1 - p0) /
                (root * root - 1.0);
            const double delta = p1 / derivative;
            root -= delta;
            if (std::abs(delta) < 1.0e-15)
                break;
        }
        const double weight =
            1.0 / ((1.0 - root * root) *
                   derivative * derivative);
        rule.points[i] = 0.5 * (1.0 - root);
        rule.points[order - 1 - i] = 0.5 * (1.0 + root);
        rule.weights[i] = weight;
        rule.weights[order - 1 - i] = weight;
    }
    return rule;
}

void add_point(
    std::vector<MullerDuffyPoint>& points,
    double test_u, double test_v,
    double trial_u, double trial_v,
    double weight)
{
    MullerDuffyPoint point;
    // Sauter-Schwab uses (u-v,v); convert to the standard
    // reference triangle (xi,eta).
    point.test_xi = test_u - test_v;
    point.test_eta = test_v;
    point.trial_xi = trial_u - trial_v;
    point.trial_eta = trial_v;
    point.weight = weight;
    points.push_back(point);
}

} // namespace

std::vector<MullerDuffyPoint> muller_duffy_rule(
    int order, MullerDuffyAdjacency adjacency)
{
    const GaussRule gauss = gauss_legendre_unit(order);
    int regions = 0;
    switch (adjacency) {
    case MullerDuffyAdjacency::Coincident:
        regions = 6;
        break;
    case MullerDuffyAdjacency::EdgeAdjacent:
        regions = 5;
        break;
    case MullerDuffyAdjacency::VertexAdjacent:
        regions = 2;
        break;
    }
    std::vector<MullerDuffyPoint> points;
    points.reserve(
        (size_t)regions * order * order * order * order);

    for (int ix = 0; ix < order; ix++) {
        const double xsi = gauss.points[ix];
        for (int i1 = 0; i1 < order; i1++) {
            const double eta1 = gauss.points[i1];
            for (int i2 = 0; i2 < order; i2++) {
                const double eta2 = gauss.points[i2];
                for (int i3 = 0; i3 < order; i3++) {
                    const double eta3 = gauss.points[i3];
                    const double quadrature_weight =
                        gauss.weights[ix] * gauss.weights[i1] *
                        gauss.weights[i2] * gauss.weights[i3];
                    const double eta12 = eta1 * eta2;
                    const double eta123 = eta12 * eta3;

                    if (adjacency ==
                        MullerDuffyAdjacency::Coincident) {
                        const double weight =
                            quadrature_weight *
                            xsi * xsi * xsi *
                            eta1 * eta1 * eta2;
                        add_point(
                            points,
                            xsi, xsi * (1.0 - eta1 + eta12),
                            xsi * (1.0 - eta123),
                            xsi * (1.0 - eta1), weight);
                        add_point(
                            points,
                            xsi * (1.0 - eta123),
                            xsi * (1.0 - eta1),
                            xsi, xsi * (1.0 - eta1 + eta12),
                            weight);
                        add_point(
                            points,
                            xsi, xsi * (eta1 - eta12 + eta123),
                            xsi * (1.0 - eta12),
                            xsi * (eta1 - eta12), weight);
                        add_point(
                            points,
                            xsi * (1.0 - eta12),
                            xsi * (eta1 - eta12),
                            xsi, xsi * (eta1 - eta12 + eta123),
                            weight);
                        add_point(
                            points,
                            xsi * (1.0 - eta123),
                            xsi * (eta1 - eta123),
                            xsi, xsi * (eta1 - eta12), weight);
                        add_point(
                            points,
                            xsi, xsi * (eta1 - eta12),
                            xsi * (1.0 - eta123),
                            xsi * (eta1 - eta123), weight);
                    } else if (
                        adjacency ==
                        MullerDuffyAdjacency::EdgeAdjacent) {
                        const double weight =
                            quadrature_weight *
                            xsi * xsi * xsi * eta1 * eta1;
                        add_point(
                            points,
                            xsi, xsi * eta1 * eta3,
                            xsi * (1.0 - eta12),
                            xsi * eta1 * (1.0 - eta2),
                            weight);
                        add_point(
                            points,
                            xsi, xsi * eta1,
                            xsi * (1.0 - eta123),
                            xsi * eta1 * eta2 * (1.0 - eta3),
                            weight * eta2);
                        add_point(
                            points,
                            xsi * (1.0 - eta12),
                            xsi * eta1 * (1.0 - eta2),
                            xsi, xsi * eta123,
                            weight * eta2);
                        add_point(
                            points,
                            xsi * (1.0 - eta123),
                            xsi * eta12 * (1.0 - eta3),
                            xsi, xsi * eta1,
                            weight * eta2);
                        add_point(
                            points,
                            xsi * (1.0 - eta123),
                            xsi * eta1 * (1.0 - eta2 * eta3),
                            xsi, xsi * eta12,
                            weight * eta2);
                    } else {
                        const double weight =
                            quadrature_weight *
                            xsi * xsi * xsi * eta2;
                        add_point(
                            points,
                            xsi, xsi * eta1,
                            xsi * eta2, xsi * eta2 * eta3,
                            weight);
                        add_point(
                            points,
                            xsi * eta2, xsi * eta2 * eta3,
                            xsi, xsi * eta1,
                            weight);
                    }
                }
            }
        }
    }
    return points;
}

void muller_duffy_remap_shared_vertex(
    double& xi, double& eta, int local_vertex)
{
    const double old_xi = xi;
    const double old_eta = eta;
    switch (local_vertex) {
    case 0:
        return;
    case 1:
        xi = 1.0 - old_xi - old_eta;
        eta = old_eta;
        return;
    case 2:
        xi = old_xi;
        eta = 1.0 - old_xi - old_eta;
        return;
    default:
        throw std::invalid_argument("local vertex must be 0, 1, or 2");
    }
}

void muller_duffy_remap_shared_edge(
    double& xi, double& eta,
    int first_local_vertex, int second_local_vertex)
{
    if (first_local_vertex < 0 || first_local_vertex > 2 ||
        second_local_vertex < 0 || second_local_vertex > 2 ||
        first_local_vertex == second_local_vertex) {
        throw std::invalid_argument(
            "shared edge requires two distinct local vertices");
    }
    const double reference[3][2] = {
        {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}
    };
    const int third =
        3 - first_local_vertex - second_local_vertex;
    const double origin_x = reference[first_local_vertex][0];
    const double origin_y = reference[first_local_vertex][1];
    const double ax =
        reference[second_local_vertex][0] - origin_x;
    const double ay =
        reference[second_local_vertex][1] - origin_y;
    const double bx = reference[third][0] - origin_x;
    const double by = reference[third][1] - origin_y;
    const double old_xi = xi;
    const double old_eta = eta;
    xi = origin_x + ax * old_xi + bx * old_eta;
    eta = origin_y + ay * old_xi + by * old_eta;
}
