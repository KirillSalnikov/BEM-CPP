#pragma once

#include "geometry_lib.h"
#include "compl.hpp"
#include "Beam.h"
#include "Particle.h"
#include <vector>
#include <string>

/// Internal beam segment between two facet interactions
struct InternalBeamSegment
{
    // Beam state at entry (before Fresnel modification at exit facet)
    Matrix2x2c J;
    Point3f direction;
    Point3f polarizationBasis;
    double opticalPath;
    double front;

    // Intersection polygon on exit facet (beam cross-section)
    Point3f polyArr[MAX_VERTEX_NUM];
    int polyNVertices;

    // Facet info
    int entryFacetId;
    int exitFacetId;
    double segmentLength;

    // Exit facet plane for projection
    Point3f exitNormal;   // in_normal of exit facet
    double exitD;         // d_param of exit facet in_normal

    int nActs;            // number of internal reflections (beam generation)
};

/// Electric field accumulated at a single dipole
struct DipoleField
{
    double x, y, z;         // physical coordinates
    int ix, iy, iz;         // integer grid indices

    // Accumulated complex E-field for Y-polarization
    complex Ex_Y, Ey_Y, Ez_Y;
    // Accumulated complex E-field for X-polarization
    complex Ex_X, Ey_X, Ez_X;

    int nContributions;  // number of GO beam segments covering this dipole
    int assignedFacet;   // entry facet assigned by FillUncoveredPerFacet (-1 = none)

    DipoleField()
        : x(0), y(0), z(0), ix(0), iy(0), iz(0),
          Ex_Y(0,0), Ey_Y(0,0), Ez_Y(0,0),
          Ex_X(0,0), Ey_X(0,0), Ez_X(0,0),
          nContributions(0), assignedFacet(-1)
    {}
};

class ADDAFieldComputer
{
public:
    ADDAFieldComputer(Particle *particle, double wavelength,
                      const complex &ri, int dpl);

    /// Build cubic dipole grid inside convex particle
    void BuildDipoleGrid();

    /// Write DDSCAT6-format geometry file for ADDA (-shape read)
    void WriteGeometryFile(const std::string &filename) const;

    /// Write init field file for Y-polarization (-init_field read)
    void WriteFieldFileY(const std::string &filename) const;

    /// Write init field file for X-polarization
    void WriteFieldFileX(const std::string &filename) const;

    /// Fill uncovered dipoles with per-facet refracted plane waves.
    /// Each illuminated facet gets its own complex Snell/Fresnel coefficients.
    /// blend > 0: Fresnel edge blending with sigma = blend * sqrt(lambda_eff * depth).
    void FillUncoveredPerFacet(const Point3f &incidentDir, double blend = 0.0);

    /// Read converged ADDA internal field from coarse grid and interpolate to current grid.
    /// Used for multigrid: run ADDA at low dpl first, then use result as init for fine dpl.
    /// fieldFileY/X: ADDA IntField-Y/X output (coordinates in k*r = 2pi/lambda * r).
    /// If incidentDir is provided, applies phase-corrected interpolation:
    /// divides out dominant PW phase exp(ikn·k_inc·r) before interpolation,
    /// restores after — reduces destructive interference at coarse grids.
    void InterpolateCoarseField(const std::string &fieldFileY,
                                 const std::string &fieldFileX,
                                 const Point3f *incidentDir = nullptr);

    /// Smooth Fabry-Perot multiplicative correction.
    /// For each anti-parallel entry-exit facet pair, multiplies forward-wave field
    /// by 1/(1 - R²·exp(2ikγd)), where R is the internal Fresnel reflection coefficient,
    /// γ is the complex refraction parameter, and d is the facet separation.
    void AddSmoothFP(const Point3f &incidentDir);

    /// Add generalized analytical first reflection for ALL facet pairs.
    /// Handles TIR via complex Fresnel, no polygon clipping, no parallel restriction.
    /// Phase: exp(-2ikμ·dist_B) where μ = incDir·n̂_B + (γ_A-cosI_A)·(n̂_A·n̂_B).
    void AddGeneralReflection(const Point3f &incidentDir, double dotThreshold = -0.9);

    /// Accumulate E-field from GO-traced reflected beam segments.
    /// Uses per-dipole phase: back-projects each dipole to reflection facet,
    /// then to entry facet, computing OP = incDir·P + n*(d1+d2).
    void AccumulateReflectedBeams(const std::vector<InternalBeamSegment> &segments,
                                   const Point3f &incidentDir,
                                   int maxNActs = 1,
                                   bool incoherent = false,
                                   int minNActs = 1);

    /// Print contribution stats and compare GO field vs per-facet PW for covered dipoles
    void DiagnoseGOvsPW(const Point3f &incidentDir);

    /// Compute far-field scattering directly from GO dipole fields (no ADDA iteration).
    /// Uses volume integral with exact sinc correction for finite voxel extent.
    /// Writes Mueller matrix to file in ADDA format and prints Cext to stdout.
    void ComputeFarField(const Point3f &incidentDir,
                         double thetaMin, double thetaMax, int nTheta,
                         const std::string &filename) const;

    /// Core far-field computation: fills muellerFlat[nTheta*17] with {theta, S11..S44}
    /// and computes Cext_Y/X via optical theorem. No file I/O or diagnostics.
    /// If incidentDir is non-null, adds Fraunhofer diffraction and returns Cgeo.
    void ComputeFarFieldCore(double thetaMinDeg, double thetaMaxDeg, int nTheta,
                             std::vector<double> &muellerFlat,
                             double &CextY, double &CextX,
                             const Point3f *incidentDir = nullptr,
                             double *Cgeo_out = nullptr) const;

    /// PGOH far-field: incoherent sum of GO beam Mueller matrices + Fraunhofer diffraction.
    /// Standard IGOM approach — converts each exit beam's Jones matrix to Mueller,
    /// weights by beam cross-section, bins by scattering angle, adds circular-aperture
    /// diffraction from particle shadow. Writes Mueller matrix in ADDA format.
    void ComputePGOH(std::vector<Beam> &outBeams,
                     const Light &incidentLight,
                     double thetaMinDeg, double thetaMaxDeg, int nTheta,
                     const std::string &filename) const;

    /// Core PGOH computation: fills muellerFlat[nTheta*17] and returns Cgeo.
    /// No file I/O or diagnostics.
    void ComputePGOHCore(std::vector<Beam> &outBeams,
                          const Light &incidentLight,
                          double thetaMinDeg, double thetaMaxDeg, int nTheta,
                          std::vector<double> &muellerFlat,
                          double &Cgeo) const;

private:
    bool IsInsideParticle(const Point3f &p) const;
    bool IsInsidePolygon(const Point3f &proj, const Point3f *polyArr,
                         int nVert, const Point3f &normal) const;
    /// Minimum distance from point to nearest polygon edge (for blend mode)
    double MinEdgeDistance(const Point3f &proj, const Point3f *polyArr, int nVert) const;

    Particle *m_particle;
    double m_wavelength;
    complex m_ri;
    double m_gridSpacing;   // dipole spacing = wavelength / dpl
    double m_k;             // wave number = 2*PI / wavelength

    std::vector<DipoleField> m_dipoles;
    double m_zMax;         // Top of particle (for consistent phase reference)

    // Grid bounding box dimensions
    int m_boxX, m_boxY, m_boxZ;

public:
    bool m_noKirchhoff = false;  // skip Kirchhoff diffraction in reflections
};
