#pragma once
#include "muller_nodal.h"
#include <functional>

// Experimental exterior Cauchy projection on a planar BDM1 surface.
// exp(-i wt), G=exp(ik r)/(4 pi r), currents J=n x H, M=-n x E.
// Uses E=ik S J+(i/k) grad S div J-curl S M and its magnetic dual.
// The singular static gradient is integrated analytically; only weakly
// singular remainders use fan quadrature. The exterior jump is +I/2.
class MaxwellExteriorProjector {
public:
    using C=std::complex<double>;
    using Vector=std::vector<C>;
    using ScalarPair=std::function<void(const C*,const C*,C*,C*,C*,C*)>;
    struct Point {Vec3 position,normal;double weight;int element;double xi,eta;};
    struct Kernel {std::array<C,12> value{};}; // [1,xi,eta] x [potential,dx,dy,dz]
    struct Triangle {
        std::array<Vec3,3> vertex;
        Vec3 normal,gxi,geta,center;double jacobian,diameter;
        std::array<int,6> dof;
        std::array<std::array<Vec3,3>,6> basis;
        std::array<double,6> divergence;
    };
    MaxwellExteriorProjector(const MullerP2Mesh&,double k,int quadrature=7,int near_order=8,double near_radius=2.);
    // The scalar backend must use precisely points(), including their order,
    // and return potentials and interleaved xyz gradients for weighted charges.
    void apply(const C* currents,C* exterior_trace,const ScalarPair&);
    void direct_pair(const C*,const C*,C*,C*,C*,C*)const;
    // Convert the existing [weak -n x E, weak n x H] RHS to [J,M] coefficients.
    Vector incident(const Vector& rhs)const;
    const std::vector<Point>& points()const{return points_;}
    size_t correction_count()const{return corrections_.size();}
    size_t correction_bytes()const{return corrections_.size()*sizeof(Correction);}
    double last_mass_residual()const{return last_mass_residual_;}
    static Kernel integrated_kernel(const Triangle&,const Vec3&,double,int);
private:
    struct Correction {int element;Kernel kernel;};
    const MullerP2Mesh& mesh_;double k_;int n_,q_;
    std::vector<Point> points_;std::vector<Triangle> triangles_;
    std::vector<size_t> offsets_;std::vector<Correction> corrections_;
    std::vector<int> mass_offsets_,mass_columns_;
    std::vector<double> mass_values_,inverse_diagonal_;
    mutable double last_mass_residual_=0;
    void mass(const Vector&,Vector&)const;
    Vector solve_mass(const Vector&)const;
};
