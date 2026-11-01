// Exercise the actual MBS Splitting implementation, without a GPU or particle.
// Link against the frozen MBS checkout used to build the comparison binary.
#include "Splitting.h"
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const double nr = std::atof(argv[1]), ni = std::atof(argv[2]);
    Splitting split(false);
    split.ComputeRiParams(complex(nr, ni));
    std::cout << std::setprecision(17)
              << "incidence_deg,rp_real,rp_imag,rs_real,rs_imag,dir_x,dir_z\n";
    for (int i = 0; i <= 899; ++i) {
        const double degrees = i * .1, angle = degrees * std::acos(-1.) / 180.;
        const Point3f normal(0., 0., 1.);
        Beam incident, transmitted, reflected;
        incident.SetLight(Point3f(std::sin(angle), 0., std::cos(angle)),
                          Point3f(0., 1., 0.));
        incident.J.Fill(complex(0., 0.));
        incident.J.m11 = incident.J.m22 = complex(1., 0.);
        split.ComputeCosA(normal, incident.direction);
        if (split.IsNormalIncidence())
            split.ComputeNormalBeamParamsExternal(incident, transmitted, reflected);
        else
            split.ComputeRegularBeamParamsExternal(normal, incident, transmitted, reflected);
        // MBS rows are p,s: the normal-incidence p coefficient has the
        // opposite sign from s in this traveling-wave polarization basis.
        std::cout << degrees << ',' << real(reflected.J.m11) << ','
                  << imag(reflected.J.m11) << ',' << real(reflected.J.m22) << ','
                  << imag(reflected.J.m22) << ',' << reflected.direction.cx << ','
                  << reflected.direction.cz << '\n';
    }
}
