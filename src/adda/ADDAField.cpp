#include "ADDAField.h"
#include "Mueller.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std;

/// Compute Fresnel integrals C(x) = int_0^x cos(pi*t^2/2)dt, S(x) = int_0^x sin(pi*t^2/2)dt
static void FresnelCS(double x, double &outC, double &outS)
{
    double ax = fabs(x);
    if (ax < 1.6)
    {
        double x2 = ax * ax;
        double re = 0, im = 0;
        double tr = ax, ti = 0;
        double piH = M_PI / 2.0;
        for (int n = 0; n <= 20; ++n)
        {
            re += tr / (2*n + 1);
            im += ti / (2*n + 1);
            double f = piH * x2 / (n + 1);
            double tr2 = -ti * f;
            double ti2 =  tr * f;
            tr = tr2;
            ti = ti2;
        }
        outC = re;
        outS = im;
    }
    else
    {
        double f = (1.0 + 0.926*ax) / (2.0 + 1.792*ax + 3.104*ax*ax);
        double g = 1.0 / (2.0 + 4.142*ax + 3.492*ax*ax + 6.670*ax*ax*ax);
        double pix2h = M_PI * ax * ax / 2.0;
        double sn = sin(pix2h), cs = cos(pix2h);
        outC = 0.5 + f*sn - g*cs;
        outS = 0.5 - f*cs - g*sn;
    }
    if (x < 0) { outC = -outC; outS = -outS; }
}

/// Kirchhoff diffraction weight from a polygon aperture.
/// Uses Green's theorem to reduce the 2D Fresnel integral to 1D edge integrals.
/// Returns U/U0 = (-i/2) * contour_integral Q dY (normalized to 1 for infinite aperture).
static complex KirchhoffPolygonWeight(
    const double *poly_u, const double *poly_v, int nVert,
    double obs_u, double obs_v,
    double z, double lambda_eff)
{
    if (z < 1e-10 || nVert < 3)
        return complex(1.0, 0.0);

    double scale = sqrt(2.0 / (lambda_eff * z));

    static const int NG = 8;
    static const double gp[NG] = {
        0.01985507175123, 0.10166676129319, 0.23723379504184, 0.40828267875218,
        0.59171732124782, 0.76276620495816, 0.89833323870681, 0.98014492824877
    };
    static const double gw[NG] = {
        0.05061426814519, 0.11119051722669, 0.15685332293895, 0.18134189168918,
        0.18134189168918, 0.15685332293895, 0.11119051722669, 0.05061426814519
    };

    complex result(0.0, 0.0);

    for (int ei = 0; ei < nVert; ++ei)
    {
        int ej = (ei + 1) % nVert;

        double X1 = (poly_u[ei] - obs_u) * scale;
        double Y1 = (poly_v[ei] - obs_v) * scale;
        double X2 = (poly_u[ej] - obs_u) * scale;
        double Y2 = (poly_v[ej] - obs_v) * scale;

        double dY = Y2 - Y1;
        if (fabs(dY) < 1e-15) continue;
        double dX = X2 - X1;

        double edgeMaxF = fmax(fmax(fabs(X1), fabs(X2)), fmax(fabs(Y1), fabs(Y2)));
        int nSub = (edgeMaxF > 2.0) ? (int)ceil(edgeMaxF / 2.0) : 1;

        complex edgeSum(0.0, 0.0);
        for (int s = 0; s < nSub; ++s)
        {
            double t0 = (double)s / nSub;
            double t1 = (double)(s + 1) / nSub;
            double dt = t1 - t0;

            for (int g = 0; g < NG; ++g)
            {
                double t = t0 + gp[g] * dt;
                double X = X1 + t * dX;
                double Y = Y1 + t * dY;

                double C, S;
                FresnelCS(X, C, S);

                double piY2h = M_PI * Y * Y / 2.0;
                double cosY = cos(piY2h), sinY = sin(piY2h);

                double ar = 0.5 + C;
                double ai = 0.5 + S;
                double re = cosY * ar - sinY * ai;
                double im = sinY * ar + cosY * ai;

                edgeSum += complex(re, im) * (gw[g] * dt);
            }
        }

        result += edgeSum * dY;
    }

    return complex(0.0, -0.5) * result;
}

ADDAFieldComputer::ADDAFieldComputer(Particle *particle, double wavelength,
                                     const complex &ri, int dpl)
    : m_particle(particle),
      m_wavelength(wavelength),
      m_ri(ri),
      m_zMax(0),
      m_boxX(0), m_boxY(0), m_boxZ(0)
{
    m_gridSpacing = wavelength / dpl;
    m_k = 2.0 * M_PI / wavelength;
}

bool ADDAFieldComputer::IsInsideParticle(const Point3f &p) const
{
    if (!m_particle->isConcave)
    {
        // For convex particle: point inside iff on inner side of every facet
        for (int i = 0; i < m_particle->nFacets; ++i)
        {
            const Point3f &n = m_particle->facets[i].in_normal;
            double dp = (double)p.cx * n.cx + (double)p.cy * n.cy
                      + (double)p.cz * n.cz + n.d_param;
            if (dp < -1e-6)
                return false;
        }
        return true;
    }

    // For non-convex particle: ray casting along +Z axis
    // Count intersections of ray from p in +Z direction with facet triangles
    int crossings = 0;
    Point3f rayDir(0, 0, 1);

    for (int i = 0; i < m_particle->nFacets; ++i)
    {
        const Facet &f = m_particle->facets[i];
        const Point3f &n = f.ex_normal;

        // Ray-plane intersection: t = -(n·p + d) / (n·rayDir)
        double nDotRay = (double)n.cz;  // n · (0,0,1)
        if (fabs(nDotRay) < 1e-12)
            continue;  // ray parallel to facet

        double nDotP = (double)p.cx * n.cx + (double)p.cy * n.cy
                     + (double)p.cz * n.cz + n.d_param;
        double t = -nDotP / nDotRay;
        if (t < 1e-6)
            continue;  // intersection behind ray origin

        // Intersection point
        double hitZ = p.cz + t;
        Point3f hit((float)p.cx, (float)p.cy, (float)hitZ);

        // Check if hit point is inside the facet polygon
        // Triangulate facet as fan from vertex 0
        for (int j = 1; j < f.nVertices - 1; ++j)
        {
            const Point3f &v0 = f.arr[0];
            const Point3f &v1 = f.arr[j];
            const Point3f &v2 = f.arr[j + 1];

            // Barycentric test in XY plane (since ray is along Z)
            double dx1 = v1.cx - v0.cx, dy1 = v1.cy - v0.cy;
            double dx2 = v2.cx - v0.cx, dy2 = v2.cy - v0.cy;
            double dxP = hit.cx - v0.cx, dyP = hit.cy - v0.cy;

            double det = dx1 * dy2 - dx2 * dy1;
            if (fabs(det) < 1e-12)
                continue;

            double u = (dxP * dy2 - dx2 * dyP) / det;
            double v = (dx1 * dyP - dxP * dy1) / det;

            if (u >= -1e-6 && v >= -1e-6 && (u + v) <= 1.0 + 1e-6)
            {
                ++crossings;
                break;  // count each facet at most once
            }
        }
    }

    return (crossings % 2) == 1;
}

void ADDAFieldComputer::BuildDipoleGrid()
{
    // Compute bounding box of particle
    double minX = 1e30, maxX = -1e30;
    double minY = 1e30, maxY = -1e30;
    double minZ = 1e30, maxZ = -1e30;

    for (int i = 0; i < m_particle->nFacets; ++i)
    {
        for (int j = 0; j < m_particle->facets[i].nVertices; ++j)
        {
            const Point3f &v = m_particle->facets[i].arr[j];
            if (v.cx < minX) minX = v.cx;
            if (v.cx > maxX) maxX = v.cx;
            if (v.cy < minY) minY = v.cy;
            if (v.cy > maxY) maxY = v.cy;
            if (v.cz < minZ) minZ = v.cz;
            if (v.cz > maxZ) maxZ = v.cz;
        }
    }

    m_zMax = maxZ;

    // Grid dimensions (add margin of 1 cell on each side)
    m_boxX = (int)ceil((maxX - minX) / m_gridSpacing) + 2;
    m_boxY = (int)ceil((maxY - minY) / m_gridSpacing) + 2;
    m_boxZ = (int)ceil((maxZ - minZ) / m_gridSpacing) + 2;

    // Grid origin: center grid on particle center
    double centerX = (minX + maxX) / 2.0;
    double centerY = (minY + maxY) / 2.0;
    double centerZ = (minZ + maxZ) / 2.0;

    double x0 = centerX - (m_boxX - 1) * m_gridSpacing / 2.0;
    double y0 = centerY - (m_boxY - 1) * m_gridSpacing / 2.0;
    double z0 = centerZ - (m_boxZ - 1) * m_gridSpacing / 2.0;

    m_dipoles.clear();

    for (int iz = 0; iz < m_boxZ; ++iz)
    {
        for (int iy = 0; iy < m_boxY; ++iy)
        {
            for (int ix = 0; ix < m_boxX; ++ix)
            {
                Point3f p((float)(x0 + ix * m_gridSpacing),
                          (float)(y0 + iy * m_gridSpacing),
                          (float)(z0 + iz * m_gridSpacing));

                if (IsInsideParticle(p))
                {
                    DipoleField dip;
                    dip.x = p.cx;
                    dip.y = p.cy;
                    dip.z = p.cz;
                    dip.ix = ix;
                    dip.iy = iy;
                    dip.iz = iz;
                    m_dipoles.push_back(dip);
                }
            }
        }
    }

    cout << "ADDA grid: " << m_boxX << "x" << m_boxY << "x" << m_boxZ
         << ", dipoles inside: " << m_dipoles.size()
         << ", spacing: " << m_gridSpacing << " um" << endl;
}

bool ADDAFieldComputer::IsInsidePolygon(const Point3f &proj,
                                         const Point3f *polyArr, int nVert,
                                         const Point3f &normal) const
{
    // Convex polygon containment: point inside if on same side of all edges
    for (int i = 0; i < nVert; ++i)
    {
        int j = (i + 1) % nVert;
        Point3f edge = polyArr[j] - polyArr[i];
        Point3f toPoint = proj - polyArr[i];
        Point3f cross = CrossProduct(edge, toPoint);
        double dp = (double)cross.cx * normal.cx
                  + (double)cross.cy * normal.cy
                  + (double)cross.cz * normal.cz;
        if (dp < -1e-4)
            return false;
    }
    return true;
}

double ADDAFieldComputer::MinEdgeDistance(const Point3f &proj,
                                          const Point3f *polyArr, int nVert) const
{
    double minDist = 1e30;
    for (int i = 0; i < nVert; ++i)
    {
        int j = (i + 1) % nVert;
        // Edge vector and length
        double ex = polyArr[j].cx - polyArr[i].cx;
        double ey = polyArr[j].cy - polyArr[i].cy;
        double ez = polyArr[j].cz - polyArr[i].cz;
        double edgeLen2 = ex*ex + ey*ey + ez*ez;
        if (edgeLen2 < 1e-20) continue;

        // Project point onto edge line: t = dot(proj - v[i], edge) / |edge|²
        double dx = proj.cx - polyArr[i].cx;
        double dy = proj.cy - polyArr[i].cy;
        double dz = proj.cz - polyArr[i].cz;
        double t = (dx*ex + dy*ey + dz*ez) / edgeLen2;
        t = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);

        // Distance from proj to closest point on edge segment
        double px = dx - t*ex;
        double py = dy - t*ey;
        double pz = dz - t*ez;
        double dist = sqrt(px*px + py*py + pz*pz);
        if (dist < minDist) minDist = dist;
    }
    return minDist;
}

void ADDAFieldComputer::WriteGeometryFile(const string &filename) const
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "ERROR: Cannot open " << filename << " for writing" << endl;
        return;
    }

    file << "MBS-GO generated geometry; grid_spacing=" << m_gridSpacing
         << " um; wavelength=" << m_wavelength << " um" << endl;
    file << m_dipoles.size() << " = NAT" << endl;
    file << "1 0 0 = A_1 vector" << endl;
    file << "0 1 0 = A_2 vector" << endl;
    file << "1.0 1.0 1.0 = lattice spacings (d_x,d_y,d_z)/d" << endl;
    file << "JA  IX  IY  IZ ICOMP1 ICOMP2 ICOMP3" << endl;

    for (size_t i = 0; i < m_dipoles.size(); ++i)
    {
        const DipoleField &dip = m_dipoles[i];
        file << (i + 1) << " "
             << dip.ix << " " << dip.iy << " " << dip.iz << " "
             << "1 1 1" << endl;
    }

    file.close();
    cout << "Wrote geometry: " << filename << " (" << m_dipoles.size() << " dipoles)" << endl;
}

void ADDAFieldComputer::WriteFieldFileY(const string &filename) const
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "ERROR: Cannot open " << filename << " for writing" << endl;
        return;
    }

    file << scientific << setprecision(10);
    file << "x y z |E|^2 Ex.r Ex.i Ey.r Ey.i Ez.r Ez.i" << endl;

    for (size_t i = 0; i < m_dipoles.size(); ++i)
    {
        const DipoleField &dip = m_dipoles[i];
        double E2 = norm(dip.Ex_Y) + norm(dip.Ey_Y) + norm(dip.Ez_Y);

        file << dip.x << " " << dip.y << " " << dip.z << " "
             << E2 << " "
             << real(dip.Ex_Y) << " " << imag(dip.Ex_Y) << " "
             << real(dip.Ey_Y) << " " << imag(dip.Ey_Y) << " "
             << real(dip.Ez_Y) << " " << imag(dip.Ez_Y) << endl;
    }

    file.close();
    cout << "Wrote Y-pol field: " << filename << endl;
}

void ADDAFieldComputer::WriteFieldFileX(const string &filename) const
{
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "ERROR: Cannot open " << filename << " for writing" << endl;
        return;
    }

    file << scientific << setprecision(10);
    file << "x y z |E|^2 Ex.r Ex.i Ey.r Ey.i Ez.r Ez.i" << endl;

    for (size_t i = 0; i < m_dipoles.size(); ++i)
    {
        const DipoleField &dip = m_dipoles[i];
        double E2 = norm(dip.Ex_X) + norm(dip.Ey_X) + norm(dip.Ez_X);

        file << dip.x << " " << dip.y << " " << dip.z << " "
             << E2 << " "
             << real(dip.Ex_X) << " " << imag(dip.Ex_X) << " "
             << real(dip.Ey_X) << " " << imag(dip.Ey_X) << " "
             << real(dip.Ez_X) << " " << imag(dip.Ez_X) << endl;
    }

    file.close();
    cout << "Wrote X-pol field: " << filename << endl;
}

void ADDAFieldComputer::FillUncoveredPerFacet(const Point3f &incidentDir,
                                               double blend)
{
    double re_n = real(m_ri);

    double ix = (double)incidentDir.cx;
    double iy = (double)incidentDir.cy;
    double iz = (double)incidentDir.cz;

    // Collect illuminated facets with their refraction parameters
    struct FacetRefr {
        int id;
        double cosI;
        Point3f refDir;        // real refracted direction (for back-projection)
        // Complex E-field amplitude vectors for Y-pol and X-pol
        complex EY_x, EY_y, EY_z;
        complex EX_x, EX_y, EX_z;
        // Phase reference: incDir . facetCenter
        double phaseRef;
        // Complex normal wavevector component: gamma = sqrt(m^2 - sin^2 theta_i)
        complex gamma;
        // Facet inward normal (for phase computation)
        double nx, ny, nz;
        // Polygon winding sign for signed edge distances (+1 or -1)
        double windSign;
    };

    vector<FacetRefr> facetList;

    for (int f = 0; f < m_particle->nFacets; ++f)
    {
        const Facet &fac = m_particle->facets[f];
        double nx = (double)fac.in_normal.cx;
        double ny = (double)fac.in_normal.cy;
        double nz = (double)fac.in_normal.cz;
        double cosI = ix*nx + iy*ny + iz*nz;
        if (cosI < 1e-6)
            continue;

        double sinIsq = 1.0 - cosI*cosI;

        // Complex Snell's law: gamma = sqrt(m^2 - sin^2 theta_i)
        // gamma = m * cos(theta_t), where theta_t is complex for absorbing media
        complex gamma = sqrt(m_ri * m_ri - complex(sinIsq, 0));
        // Ensure Im(gamma) > 0 (wave decays into crystal)
        if (imag(gamma) < 0) gamma = -gamma;

        // Real refracted direction (for back-projection geometry only)
        double cosT_real = sqrt(fmax(0.0, 1.0 - sinIsq/(re_n*re_n)));
        double eta = 1.0 / re_n;
        double Cv = cosT_real - eta * cosI;
        double rdx = eta * ix + Cv * nx;
        double rdy = eta * iy + Cv * ny;
        double rdz = eta * iz + Cv * nz;
        double rlen = sqrt(rdx*rdx + rdy*rdy + rdz*rdz);
        rdx /= rlen; rdy /= rlen; rdz /= rlen;

        // Complex Fresnel transmission coefficients (air -> medium)
        // Ts = 2 cos_i / (cos_i + gamma),  Tp = 2 m cos_i / (m^2 cos_i + gamma)
        complex Ts = 2.0 * cosI / (cosI + gamma);
        complex Tp = 2.0 * m_ri * cosI / (m_ri * m_ri * cosI + gamma);

        // s/p polarization basis from plane of incidence (incDir, refDir)
        double sx = iy*rdz - iz*rdy;
        double sy = iz*rdx - ix*rdz;
        double sz = ix*rdy - iy*rdx;
        double slen = sqrt(sx*sx + sy*sy + sz*sz);

        complex EY_x, EY_y, EY_z;
        complex EX_x, EX_y, EX_z;

        if (slen < 1e-6)
        {
            // Normal incidence
            complex T = 0.5 * (Ts + Tp);
            EY_x = 0; EY_y = T; EY_z = 0;
            EX_x = -T; EX_y = 0; EX_z = 0;
        }
        else
        {
            sx /= slen; sy /= slen; sz /= slen;
            // p_inc = cross(s, incDir)
            double pix_ = sy*iz - sz*iy;
            double piy_ = sz*ix - sx*iz;
            double piz_ = sx*iy - sy*ix;
            double pilen = sqrt(pix_*pix_ + piy_*piy_ + piz_*piz_);
            pix_ /= pilen; piy_ /= pilen; piz_ /= pilen;
            // p_refr = cross(s, refDir)
            double prx = sy*rdz - sz*rdy;
            double pry = sz*rdx - sx*rdz;
            double prz = sx*rdy - sy*rdx;
            double prlen = sqrt(prx*prx + pry*pry + prz*prz);
            prx /= prlen; pry /= prlen; prz /= prlen;

            // Y-pol: E_inc = (0,1,0)
            double Eys = sy;
            double Eyp = piy_;
            EY_x = Ts*Eys*sx + Tp*Eyp*prx;
            EY_y = Ts*Eys*sy + Tp*Eyp*pry;
            EY_z = Ts*Eys*sz + Tp*Eyp*prz;

            // X-pol: E_inc = (-1,0,0)
            double Exs = -sx;
            double Exp_ = -pix_;
            EX_x = Ts*Exs*sx + Tp*Exp_*prx;
            EX_y = Ts*Exs*sy + Tp*Exp_*pry;
            EX_z = Ts*Exs*sz + Tp*Exp_*prz;
        }

        FacetRefr fr;
        fr.id = f;
        fr.cosI = cosI;
        fr.refDir = Point3f((float)rdx, (float)rdy, (float)rdz);
        fr.EY_x = EY_x; fr.EY_y = EY_y; fr.EY_z = EY_z;
        fr.EX_x = EX_x; fr.EX_y = EX_y; fr.EX_z = EX_z;
        fr.gamma = gamma;
        fr.nx = nx; fr.ny = ny; fr.nz = nz;
        // Phase ref: external OP to facet center
        double cx_ = (double)fac.center.cx;
        double cy_ = (double)fac.center.cy;
        double cz_ = (double)fac.center.cz;
        fr.phaseRef = ix*cx_ + iy*cy_ + iz*cz_;
        // Determine polygon winding sign: cross(edge0, center-V0) · in_normal
        {
            double e0x = fac.arr[1].cx - fac.arr[0].cx;
            double e0y = fac.arr[1].cy - fac.arr[0].cy;
            double e0z = fac.arr[1].cz - fac.arr[0].cz;
            double cv_x = fac.center.cx - fac.arr[0].cx;
            double cv_y = fac.center.cy - fac.arr[0].cy;
            double cv_z = fac.center.cz - fac.arr[0].cz;
            double wx = e0y*cv_z - e0z*cv_y;
            double wy = e0z*cv_x - e0x*cv_z;
            double wz = e0x*cv_y - e0y*cv_x;
            double wdot = wx*nx + wy*ny + wz*nz;
            fr.windSign = (wdot >= 0) ? 1.0 : -1.0;
        }
        facetList.push_back(fr);
    }

    // Sort by cosI descending (primary facet first for fallback)
    for (size_t i = 0; i < facetList.size(); ++i)
        for (size_t j = i+1; j < facetList.size(); ++j)
            if (facetList[j].cosI > facetList[i].cosI)
                swap(facetList[i], facetList[j]);

    // If all facets are steeply tilted (e.g., pyramidal top at normal incidence),
    // use incident direction for phase propagation instead of refracted direction.
    // Tilted refracted PWs from individual facets create wrong phase patterns
    // deep inside the crystal; a straight-through PW is a better approximation.
    bool pyramidMode = (!facetList.empty() && facetList[0].cosI < 0.6);

    cout << "Per-facet refraction: " << facetList.size() << " illuminated facets";
    if (!facetList.empty())
        cout << " (primary cosI=" << facetList[0].cosI << ")";
    if (pyramidMode)
        cout << " [pyramid mode: using incDir for phase]";
    cout << endl;

    int uncovered = 0;
    int perFacetFilled = 0;
    int fallbackFilled = 0;

    if (pyramidMode)
    {
        // Pyramid mode: all facets are steeply tilted, per-facet PW creates
        // wrong phase patterns. Use a single PW along incDir with
        // area-weighted average Fresnel amplitude and average gamma.
        double totalWeight = 0;
        complex avgEY_x(0,0), avgEY_y(0,0), avgEY_z(0,0);
        complex avgEX_x(0,0), avgEX_y(0,0), avgEX_z(0,0);
        double avgCx = 0, avgCy = 0, avgCz = 0;
        complex avgGamma(0,0);
        double avgNx = 0, avgNy = 0, avgNz = 0;
        for (size_t fi = 0; fi < facetList.size(); ++fi) {
            const FacetRefr &fr = facetList[fi];
            const Facet &fac = m_particle->facets[fr.id];
            double w = fr.cosI; // projected area weight
            totalWeight += w;
            avgEY_x += w * fr.EY_x; avgEY_y += w * fr.EY_y; avgEY_z += w * fr.EY_z;
            avgEX_x += w * fr.EX_x; avgEX_y += w * fr.EX_y; avgEX_z += w * fr.EX_z;
            avgCx += w * (double)fac.center.cx;
            avgCy += w * (double)fac.center.cy;
            avgCz += w * (double)fac.center.cz;
            avgGamma += w * fr.gamma;
            avgNx += w * fr.nx; avgNy += w * fr.ny; avgNz += w * fr.nz;
        }
        avgEY_x /= totalWeight; avgEY_y /= totalWeight; avgEY_z /= totalWeight;
        avgEX_x /= totalWeight; avgEX_y /= totalWeight; avgEX_z /= totalWeight;
        avgCx /= totalWeight; avgCy /= totalWeight; avgCz /= totalWeight;
        avgGamma /= totalWeight;
        avgNx /= totalWeight; avgNy /= totalWeight; avgNz /= totalWeight;
        double avgPhaseRef = ix*avgCx + iy*avgCy + iz*avgCz;
        // Average cosI for pyramid mode
        double avgCosI = ix*avgNx + iy*avgNy + iz*avgNz;

        cout << "  Pyramid mode: avg |E_Y|=" << abs(avgEY_x) << "," << abs(avgEY_y) << ","
             << abs(avgEY_z) << "), ref=(" << avgCx << "," << avgCy << "," << avgCz << ")" << endl;

        for (size_t i = 0; i < m_dipoles.size(); ++i)
        {
            DipoleField &dip = m_dipoles[i];
            if (dip.nContributions > 0)
                continue;

            ++uncovered;
            double dx_ = dip.x - avgCx, dy_ = dip.y - avgCy, dz_ = dip.z - avgCz;
            double inc_dot_dr = ix*dx_ + iy*dy_ + iz*dz_;
            double dr_n = avgNx*dx_ + avgNy*dy_ + avgNz*dz_;
            // Phase: k * [extOP + incDir·Δr + (gamma - cosI) * (n·Δr)]
            double phase_re = avgPhaseRef + inc_dot_dr + (real(avgGamma) - avgCosI) * dr_n;
            double phase_decay = imag(avgGamma) * dr_n;
            complex phase = exp_im(m_k * phase_re) * exp(-m_k * phase_decay);

            dip.Ex_Y = avgEY_x * phase;
            dip.Ey_Y = avgEY_y * phase;
            dip.Ez_Y = avgEY_z * phase;
            dip.Ex_X = avgEX_x * phase;
            dip.Ey_X = avgEX_y * phase;
            dip.Ez_X = avgEX_z * phase;
            dip.assignedFacet = facetList[0].id;
            ++perFacetFilled;
        }
    }
    else if (blend > 0 && !facetList.empty())
    {
    // Blend mode: back-project along -refractedDir, Fresnel edge blending.
    // Each dipole accumulates coherent PW contributions from ALL illuminated facets,
    // weighted by erf(signedDist / sigma) where sigma = blend * sqrt(lambda_eff * depth).
    double lambda_eff = m_wavelength / re_n;

    cout << "Blend mode: sigma_scale=" << blend << ", lambda_eff=" << lambda_eff << endl;

    for (size_t i = 0; i < m_dipoles.size(); ++i)
    {
        DipoleField &dip = m_dipoles[i];
        bool goCovered = (dip.nContributions > 0);
        if (!goCovered)
            ++uncovered;

        complex sumEY_x(0,0), sumEY_y(0,0), sumEY_z(0,0);
        complex sumEX_x(0,0), sumEX_y(0,0), sumEX_z(0,0);
        double totalWeight = 0;
        int nContribFacets = 0;
        int bestFacet = -1;
        double bestSignedDist = -1e30;

        for (size_t fi = 0; fi < facetList.size(); ++fi)
        {
            const FacetRefr &fr = facetList[fi];
            const Facet &fac = m_particle->facets[fr.id];

            // Back-project along -refractedDir to facet plane
            double rdx = (double)fr.refDir.cx;
            double rdy = (double)fr.refDir.cy;
            double rdz = (double)fr.refDir.cz;

            double dp_nr = (double)dip.x * fac.in_normal.cx
                         + (double)dip.y * fac.in_normal.cy
                         + (double)dip.z * fac.in_normal.cz
                         + fac.in_normal.d_param;
            double dp_rd = rdx * fac.in_normal.cx
                         + rdy * fac.in_normal.cy
                         + rdz * fac.in_normal.cz;
            if (dp_rd < 1e-10)
                continue;

            double t_ref = dp_nr / dp_rd;
            if (t_ref < -1e-6)
                continue;

            Point3f proj((float)(dip.x - rdx * t_ref),
                         (float)(dip.y - rdy * t_ref),
                         (float)(dip.z - rdz * t_ref));

            // Depth from facet to dipole along inward normal
            double dr_n = dp_nr;  // n̂·(dip - proj) = dp_nr since proj is on plane
            if (dr_n < 1e-10 && dr_n < m_gridSpacing * 0.1)
                continue;

            // Signed edge distance: positive inside polygon, negative outside
            double edgeDist = MinEdgeDistance(proj, fac.arr, fac.nVertices);

            Point3f polyNormal(0, 0, 0);
            if (fac.nVertices >= 3)
            {
                Point3f e1 = fac.arr[1] - fac.arr[0];
                Point3f e2 = fac.arr[2] - fac.arr[0];
                polyNormal = CrossProduct(e1, e2);
                float pnLen = (float)Length(polyNormal);
                if (pnLen > 1e-10f)
                    polyNormal = polyNormal / (double)pnLen;
            }
            bool inside = IsInsidePolygon(proj, fac.arr, fac.nVertices, polyNormal);
            double minSignedDist = inside ? edgeDist : -edgeDist;

            // Fresnel weight: smooth transition at facet edge
            // Absorption-scaled sigma: saturates at skin depth l_abs = 1/(k*Im(gamma))
            double depth_eff = fmax(dr_n, m_gridSpacing);
            double im_gamma = imag(fr.gamma);
            if (im_gamma > 1e-15) {
                double l_abs = 1.0 / (m_k * im_gamma);
                depth_eff = depth_eff / (1.0 + depth_eff / l_abs);
            }
            double sigma = blend * sqrt(lambda_eff * depth_eff);
            double weight = (sigma > 1e-15)
                ? 0.5 * (1.0 + erf(minSignedDist / sigma))
                : (inside ? 1.0 : 0.0);

            // Low threshold: shadow-zone dipoles get tiny erf weight instead of zero
            if (weight < 1e-10)
                continue;

            // Track best facet for assignedFacet
            if (minSignedDist > bestSignedDist)
            {
                bestSignedDist = minSignedDist;
                bestFacet = fr.id;
            }

            if (!goCovered)
            {
                // PW phase: same formula, independent of projection point
                double px = (double)proj.cx;
                double py = (double)proj.cy;
                double pz = (double)proj.cz;
                double dx_ = dip.x - px, dy_ = dip.y - py, dz_ = dip.z - pz;
                double extOP = ix*px + iy*py + iz*pz;
                double inc_dot_dr = ix*dx_ + iy*dy_ + iz*dz_;
                double phase_re = extOP + inc_dot_dr + (real(fr.gamma) - fr.cosI) * dr_n;
                double phase_decay = imag(fr.gamma) * dr_n;
                complex phase = exp_im(m_k * phase_re) * exp(-m_k * phase_decay);

                sumEY_x += fr.EY_x * phase * weight;
                sumEY_y += fr.EY_y * phase * weight;
                sumEY_z += fr.EY_z * phase * weight;
                sumEX_x += fr.EX_x * phase * weight;
                sumEX_y += fr.EX_y * phase * weight;
                sumEX_z += fr.EX_z * phase * weight;
                totalWeight += weight;
                ++nContribFacets;
            }
        }

        if (bestFacet >= 0)
            dip.assignedFacet = bestFacet;

        if (!goCovered && nContribFacets > 0)
        {
            // Single-facet: normalize to remove edge amplitude artifact (erf < 1 is unphysical)
            // Multi-facet: keep natural Fresnel weights (interference is physical)
            double norm = (nContribFacets == 1) ? 1.0 / fmax(totalWeight, 1e-15) : 1.0;
            dip.Ex_Y = sumEY_x * norm; dip.Ey_Y = sumEY_y * norm; dip.Ez_Y = sumEY_z * norm;
            dip.Ex_X = sumEX_x * norm; dip.Ey_X = sumEX_y * norm; dip.Ez_X = sumEX_z * norm;
            ++perFacetFilled;
        }
        else if (!goCovered && !facetList.empty())
        {
            // Fallback: primary facet PW (no blending possible)
            const FacetRefr &fr = facetList[0];
            dip.assignedFacet = fr.id;

            const Facet &fac = m_particle->facets[fr.id];
            double fb_nr = (double)dip.x * fac.in_normal.cx
                         + (double)dip.y * fac.in_normal.cy
                         + (double)dip.z * fac.in_normal.cz
                         + fac.in_normal.d_param;
            double fb_nd = ix * fac.in_normal.cx
                         + iy * fac.in_normal.cy
                         + iz * fac.in_normal.cz;
            double fb_t = (fabs(fb_nd) > 1e-10) ? -fb_nr / fb_nd : 0.0;
            double px = dip.x + ix * fb_t;
            double py = dip.y + iy * fb_t;
            double pz = dip.z + iz * fb_t;
            double extOP = ix*px + iy*py + iz*pz;
            double dx_ = dip.x - px, dy_ = dip.y - py, dz_ = dip.z - pz;
            double inc_dot_dr = ix*dx_ + iy*dy_ + iz*dz_;
            double dr_n = fr.nx*dx_ + fr.ny*dy_ + fr.nz*dz_;
            double phase_re = extOP + inc_dot_dr + (real(fr.gamma) - fr.cosI) * dr_n;
            double phase_decay = imag(fr.gamma) * dr_n;
            complex phase = exp_im(m_k * phase_re) * exp(-m_k * phase_decay);

            dip.Ex_Y = fr.EY_x * phase;
            dip.Ey_Y = fr.EY_y * phase;
            dip.Ez_Y = fr.EY_z * phase;
            dip.Ex_X = fr.EX_x * phase;
            dip.Ey_X = fr.EX_y * phase;
            dip.Ez_X = fr.EX_z * phase;
            ++fallbackFilled;
        }
    }

    cout << "Blend: per-facet=" << perFacetFilled << ", fallback=" << fallbackFilled << endl;
    }
    else
    {
    // Normal mode: per-facet assignment via back-projection along incident direction.
    // Always set assignedFacet (for FP reflection); fill field only for uncovered dipoles.
    for (size_t i = 0; i < m_dipoles.size(); ++i)
    {
        DipoleField &dip = m_dipoles[i];
        bool goCovered = (dip.nContributions > 0);
        if (!goCovered)
            ++uncovered;

        bool found = false;

        for (size_t fi = 0; fi < facetList.size(); ++fi)
        {
            const FacetRefr &fr = facetList[fi];
            const Facet &fac = m_particle->facets[fr.id];

            // Back-project dipole along -refractedDir to facet plane
            double rdx = (double)fr.refDir.cx;
            double rdy = (double)fr.refDir.cy;
            double rdz = (double)fr.refDir.cz;

            double dp_nr = (double)dip.x * fac.in_normal.cx
                         + (double)dip.y * fac.in_normal.cy
                         + (double)dip.z * fac.in_normal.cz
                         + fac.in_normal.d_param;
            double dp_rd = rdx * fac.in_normal.cx
                         + rdy * fac.in_normal.cy
                         + rdz * fac.in_normal.cz;
            if (dp_rd < 1e-10)
                continue;

            double t_ref = dp_nr / dp_rd;
            if (t_ref < -1e-6)
                continue;

            Point3f proj((float)(dip.x - rdx * t_ref),
                         (float)(dip.y - rdy * t_ref),
                         (float)(dip.z - rdz * t_ref));

            Point3f polyNormal(0, 0, 0);
            if (fac.nVertices >= 3)
            {
                Point3f e1 = fac.arr[1] - fac.arr[0];
                Point3f e2 = fac.arr[2] - fac.arr[0];
                polyNormal = CrossProduct(e1, e2);
                float pnLen = (float)Length(polyNormal);
                if (pnLen > 1e-10f)
                    polyNormal = polyNormal / (double)pnLen;
            }

            if (!IsInsidePolygon(proj, fac.arr, fac.nVertices, polyNormal))
                continue;

            // Always set assignedFacet (needed for reflection)
            dip.assignedFacet = fr.id;
            found = true;

            if (!goCovered)
            {
                // Fill field with per-facet PW using exact entry point
                double px = (double)proj.cx;
                double py = (double)proj.cy;
                double pz = (double)proj.cz;
                double extOP = ix*px + iy*py + iz*pz;
                double dx_ = dip.x - px, dy_ = dip.y - py, dz_ = dip.z - pz;
                double inc_dot_dr = ix*dx_ + iy*dy_ + iz*dz_;
                double dr_n = fr.nx*dx_ + fr.ny*dy_ + fr.nz*dz_;
                double phase_re = extOP + inc_dot_dr + (real(fr.gamma) - fr.cosI) * dr_n;
                double phase_decay = imag(fr.gamma) * dr_n;
                complex phase = exp_im(m_k * phase_re) * exp(-m_k * phase_decay);

                dip.Ex_Y = fr.EY_x * phase;
                dip.Ey_Y = fr.EY_y * phase;
                dip.Ez_Y = fr.EY_z * phase;
                dip.Ex_X = fr.EX_x * phase;
                dip.Ey_X = fr.EX_y * phase;
                dip.Ez_X = fr.EX_z * phase;
                ++perFacetFilled;
            }
            break;
        }

        if (!found && !facetList.empty())
        {
            // Fallback: use primary facet (largest cosI)
            const FacetRefr &fr = facetList[0];
            dip.assignedFacet = fr.id;

            if (!goCovered)
            {
                const Facet &fac = m_particle->facets[fr.id];
                double fb_nr = (double)dip.x * fac.in_normal.cx
                             + (double)dip.y * fac.in_normal.cy
                             + (double)dip.z * fac.in_normal.cz
                             + fac.in_normal.d_param;
                double fb_nd = ix * fac.in_normal.cx
                             + iy * fac.in_normal.cy
                             + iz * fac.in_normal.cz;
                double fb_t = (fabs(fb_nd) > 1e-10) ? -fb_nr / fb_nd : 0.0;
                double px = dip.x + ix * fb_t;
                double py = dip.y + iy * fb_t;
                double pz = dip.z + iz * fb_t;
                double extOP = ix*px + iy*py + iz*pz;
                double dx_ = dip.x - px, dy_ = dip.y - py, dz_ = dip.z - pz;
                double inc_dot_dr = ix*dx_ + iy*dy_ + iz*dz_;
                double dr_n = fr.nx*dx_ + fr.ny*dy_ + fr.nz*dz_;
                double phase_re = extOP + inc_dot_dr + (real(fr.gamma) - fr.cosI) * dr_n;
                double phase_decay = imag(fr.gamma) * dr_n;
                complex phase = exp_im(m_k * phase_re) * exp(-m_k * phase_decay);

                dip.Ex_Y = fr.EY_x * phase;
                dip.Ey_Y = fr.EY_y * phase;
                dip.Ez_Y = fr.EY_z * phase;
                dip.Ex_X = fr.EX_x * phase;
                dip.Ey_X = fr.EX_y * phase;
                dip.Ez_X = fr.EX_z * phase;
                ++fallbackFilled;
            }
        }
    }
    } // end normal mode

    cout << "Dipoles covered by GO: " << (m_dipoles.size() - uncovered)
         << ", per-facet plane wave: " << perFacetFilled
         << ", primary-facet fallback: " << fallbackFilled << endl;
}

void ADDAFieldComputer::InterpolateCoarseField(const string &fieldFileY,
                                                 const string &fieldFileX,
                                                 const Point3f *incidentDir)
{
    // Read ADDA IntField: header + lines of "x y z |E|^2 Ex.r Ex.i Ey.r Ey.i Ez.r Ez.i"
    // Coordinates in ADDA units: k*r = (2*pi/lambda) * r_physical
    struct CoarsePt {
        double x, y, z;
        complex ExY, EyY, EzY;
        complex ExX, EyX, EzX;
    };
    vector<CoarsePt> pts;

    // Read Y-pol field
    ifstream fy(fieldFileY);
    if (!fy.is_open()) { cerr << "Cannot open " << fieldFileY << endl; return; }
    string hdr; getline(fy, hdr);
    double x, y, z, e2, exr, exi, eyr, eyi, ezr, ezi;
    while (fy >> x >> y >> z >> e2 >> exr >> exi >> eyr >> eyi >> ezr >> ezi) {
        CoarsePt p;
        p.x = x; p.y = y; p.z = z;
        p.ExY = complex(exr,exi); p.EyY = complex(eyr,eyi); p.EzY = complex(ezr,ezi);
        pts.push_back(p);
    }
    fy.close();

    // Read X-pol field
    ifstream fx(fieldFileX);
    if (!fx.is_open()) { cerr << "Cannot open " << fieldFileX << endl; return; }
    getline(fx, hdr);
    size_t idx = 0;
    while (fx >> x >> y >> z >> e2 >> exr >> exi >> eyr >> eyi >> ezr >> ezi && idx < pts.size()) {
        pts[idx].ExX = complex(exr,exi); pts[idx].EyX = complex(eyr,eyi);
        pts[idx].EzX = complex(ezr,ezi); ++idx;
    }
    fx.close();

    if (pts.size() < 8) { cerr << "Too few coarse dipoles" << endl; return; }

    // Determine coarse grid from coordinates
    double xmin = pts[0].x, ymin = pts[0].y, zmin = pts[0].z;
    double xmax = xmin, ymax = ymin, zmax = zmin;
    for (size_t i = 1; i < pts.size(); ++i) {
        if (pts[i].x < xmin) xmin = pts[i].x;
        if (pts[i].y < ymin) ymin = pts[i].y;
        if (pts[i].z < zmin) zmin = pts[i].z;
        if (pts[i].x > xmax) xmax = pts[i].x;
        if (pts[i].y > ymax) ymax = pts[i].y;
        if (pts[i].z > zmax) zmax = pts[i].z;
    }

    // Grid spacing: min positive x-difference between adjacent sorted points
    double dx = 1e30;
    {
        vector<double> xs;
        for (size_t i = 0; i < pts.size(); ++i) xs.push_back(pts[i].x);
        sort(xs.begin(), xs.end());
        for (size_t i = 1; i < xs.size(); ++i) {
            double d = xs[i] - xs[i-1];
            if (d > 1e-10 && d < dx) dx = d;
        }
    }

    int nx = (int)round((xmax - xmin) / dx) + 1;
    int ny = (int)round((ymax - ymin) / dx) + 1;
    int nz = (int)round((zmax - zmin) / dx) + 1;

    cout << "Multigrid: coarse " << nx << "x" << ny << "x" << nz
         << " (" << pts.size() << " pts), spacing=" << dx << endl;

    // Build 3D grid
    int total = nx * ny * nz;
    vector<complex> exY(total), eyY(total), ezY(total);
    vector<complex> exX(total), eyX(total), ezX(total);
    vector<bool> occ(total, false);

    for (size_t i = 0; i < pts.size(); ++i) {
        int ix = (int)round((pts[i].x - xmin) / dx);
        int iy = (int)round((pts[i].y - ymin) / dx);
        int iz = (int)round((pts[i].z - zmin) / dx);
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;
        int ci = ix*ny*nz + iy*nz + iz;
        exY[ci] = pts[i].ExY; eyY[ci] = pts[i].EyY; ezY[ci] = pts[i].EzY;
        exX[ci] = pts[i].ExX; eyX[ci] = pts[i].EyX; ezX[ci] = pts[i].EzX;
        occ[ci] = true;
    }

    // Phase-corrected interpolation: divide out dominant PW phase exp(ikn * k_inc . r)
    // from coarse grid before interpolation, restore after.
    // This makes the envelope slowly varying, improving trilinear accuracy.
    bool phaseCorr = (incidentDir != nullptr);
    double pc_re_n = 0, pc_ix = 0, pc_iy = 0, pc_iz = 0;
    if (phaseCorr)
    {
        pc_re_n = real(m_ri);
        pc_ix = (double)incidentDir->cx;
        pc_iy = (double)incidentDir->cy;
        pc_iz = (double)incidentDir->cz;

        for (int jx = 0; jx < nx; ++jx)
        for (int jy = 0; jy < ny; ++jy)
        for (int jz = 0; jz < nz; ++jz)
        {
            int ci = jx*ny*nz + jy*nz + jz;
            if (!occ[ci]) continue;
            double qx = xmin + jx * dx;
            double qy = ymin + jy * dx;
            double qz = zmin + jz * dx;
            // Phase of dominant PW in ADDA coords: Re(n) * (k_inc . q)
            complex pf = exp_im(-pc_re_n * (pc_ix*qx + pc_iy*qy + pc_iz*qz));
            exY[ci] *= pf; eyY[ci] *= pf; ezY[ci] *= pf;
            exX[ci] *= pf; eyX[ci] *= pf; ezX[ci] *= pf;
        }
        cout << "Multigrid: phase correction enabled (n=" << pc_re_n << ")" << endl;
    }

    // Interpolate to fine grid. Convert fine dipole coords to ADDA units: k*r
    int nInterp = 0, nMissed = 0;
    for (size_t i = 0; i < m_dipoles.size(); ++i) {
        DipoleField &dip = m_dipoles[i];
        double qx = m_k * dip.x, qy = m_k * dip.y, qz = m_k * dip.z;

        double gx = (qx - xmin) / dx;
        double gy = (qy - ymin) / dx;
        double gz = (qz - zmin) / dx;

        int ix0 = (int)floor(gx), iy0 = (int)floor(gy), iz0 = (int)floor(gz);
        double tx = gx - ix0, ty = gy - iy0, tz = gz - iz0;

        // Trilinear interpolation (8 corners)
        double w[8] = {
            (1-tx)*(1-ty)*(1-tz), tx*(1-ty)*(1-tz),
            (1-tx)*ty*(1-tz),     tx*ty*(1-tz),
            (1-tx)*(1-ty)*tz,     tx*(1-ty)*tz,
            (1-tx)*ty*tz,         tx*ty*tz
        };
        int dix[8] = {0,1,0,1,0,1,0,1};
        int diy[8] = {0,0,1,1,0,0,1,1};
        int diz[8] = {0,0,0,0,1,1,1,1};

        complex sExY(0,0),sEyY(0,0),sEzY(0,0), sExX(0,0),sEyX(0,0),sEzX(0,0);
        double wsum = 0;
        for (int c = 0; c < 8; ++c) {
            int jx = ix0+dix[c], jy = iy0+diy[c], jz = iz0+diz[c];
            if (jx<0||jx>=nx||jy<0||jy>=ny||jz<0||jz>=nz) continue;
            int ci = jx*ny*nz + jy*nz + jz;
            if (!occ[ci]) continue;
            sExY += w[c]*exY[ci]; sEyY += w[c]*eyY[ci]; sEzY += w[c]*ezY[ci];
            sExX += w[c]*exX[ci]; sEyX += w[c]*eyX[ci]; sEzX += w[c]*ezX[ci];
            wsum += w[c];
        }

        if (wsum < 1e-15) { ++nMissed; continue; }
        double inv = 1.0 / wsum;
        dip.Ex_Y = sExY*inv; dip.Ey_Y = sEyY*inv; dip.Ez_Y = sEzY*inv;
        dip.Ex_X = sExX*inv; dip.Ey_X = sEyX*inv; dip.Ez_X = sEzX*inv;

        // Restore PW phase at fine dipole position
        if (phaseCorr)
        {
            complex pf = exp_im(pc_re_n * (pc_ix*qx + pc_iy*qy + pc_iz*qz));
            dip.Ex_Y *= pf; dip.Ey_Y *= pf; dip.Ez_Y *= pf;
            dip.Ex_X *= pf; dip.Ey_X *= pf; dip.Ez_X *= pf;
        }

        dip.nContributions = 1;
        ++nInterp;
    }

    cout << "Multigrid: interpolated " << nInterp << " / " << m_dipoles.size()
         << " fine dipoles (" << nMissed << " outside)" << endl;
}

void ADDAFieldComputer::AddSmoothFP(const Point3f &incidentDir)
{
    double ix = (double)incidentDir.cx;
    double iy = (double)incidentDir.cy;
    double iz = (double)incidentDir.cz;

    int nCorrected = 0;

    for (int fA = 0; fA < m_particle->nFacets; ++fA)
    {
        const Facet &facA = m_particle->facets[fA];
        double nAx = (double)facA.in_normal.cx;
        double nAy = (double)facA.in_normal.cy;
        double nAz = (double)facA.in_normal.cz;
        double cosI = ix*nAx + iy*nAy + iz*nAz;
        if (cosI < 1e-6) continue;

        // Find most anti-parallel exit facet
        int bestB = -1;
        double bestDot = 0;
        for (int fB = 0; fB < m_particle->nFacets; ++fB)
        {
            if (fB == fA) continue;
            double dot = nAx*(double)m_particle->facets[fB].in_normal.cx
                       + nAy*(double)m_particle->facets[fB].in_normal.cy
                       + nAz*(double)m_particle->facets[fB].in_normal.cz;
            if (dot < bestDot) { bestDot = dot; bestB = fB; }
        }
        if (bestB < 0 || bestDot > -0.9) continue;

        const Facet &facB = m_particle->facets[bestB];

        // Distance between facets along entry normal
        double dx = (double)facB.center.cx - (double)facA.center.cx;
        double dy = (double)facB.center.cy - (double)facA.center.cy;
        double dz = (double)facB.center.cz - (double)facA.center.cz;
        double dist = fabs(nAx*dx + nAy*dy + nAz*dz);

        // Complex Snell at entry facet
        double sinIsq = 1.0 - cosI * cosI;
        complex gamma = sqrt(m_ri * m_ri - complex(sinIsq, 0));
        if (imag(gamma) < 0) gamma = -gamma;

        // Internal Fresnel reflection coefficients (medium -> air)
        complex Rs_int = (gamma - cosI) / (gamma + cosI);
        complex Rp_int = (gamma - m_ri*m_ri*cosI) / (gamma + m_ri*m_ri*cosI);

        // R² for round-trip (R_entry * R_exit ≈ R² for anti-parallel facets)
        complex R2s = Rs_int * Rs_int;
        complex R2p = Rp_int * Rp_int;
        complex R2_avg = 0.5 * (R2s + R2p);

        // Round-trip phase: exp(2ikγd)
        complex phase_rt = exp_im(2.0 * m_k * real(gamma) * dist)
                         * exp(-2.0 * m_k * imag(gamma) * dist);

        // FP correction: 1 / (1 - R²·exp(2ikγd))
        complex fp_corr = 1.0 / (1.0 - R2_avg * phase_rt);

        // Apply to all dipoles assigned to this entry facet
        int nThis = 0;
        for (size_t i = 0; i < m_dipoles.size(); ++i)
        {
            if (m_dipoles[i].assignedFacet != fA) continue;
            DipoleField &dip = m_dipoles[i];
            dip.Ex_Y *= fp_corr; dip.Ey_Y *= fp_corr; dip.Ez_Y *= fp_corr;
            dip.Ex_X *= fp_corr; dip.Ey_X *= fp_corr; dip.Ez_X *= fp_corr;
            ++nThis;
        }
        if (nThis > 0)
        {
            ++nCorrected;
            cout << "  SFP facet " << fA << " -> " << bestB
                 << ": d=" << dist << ", |R|^2=" << 0.5*(norm(Rs_int)+norm(Rp_int))
                 << ", |FP|=" << abs(fp_corr)
                 << ", phase(FP)=" << arg(fp_corr)*180/M_PI << " deg"
                 << " (" << nThis << " dipoles)" << endl;
        }
    }

    cout << "Smooth FP correction applied to " << nCorrected << " facet pairs" << endl;
}

void ADDAFieldComputer::AddGeneralReflection(const Point3f &incidentDir, double dotThreshold)
{
    double re_n = real(m_ri);

    double ix = (double)incidentDir.cx;
    double iy = (double)incidentDir.cy;
    double iz = (double)incidentDir.cz;

    int nF = m_particle->nFacets;

    // Step 1: Per-entry-facet refraction + Fresnel transmission data
    struct EntryData {
        int id;
        double cosI;
        complex gamma;
        double rdx, rdy, rdz;   // refracted direction
        double nx, ny, nz;      // entry inward normal
        // Facet center (phase reference)
        double fcx, fcy, fcz;
        double phaseRef;        // incDir · center
        // Transmitted PW amplitude vectors
        complex EY_x, EY_y, EY_z;
        complex EX_x, EX_y, EX_z;
    };

    vector<EntryData> entryList;

    for (int f = 0; f < nF; ++f)
    {
        const Facet &fac = m_particle->facets[f];
        double nx = (double)fac.in_normal.cx;
        double ny = (double)fac.in_normal.cy;
        double nz = (double)fac.in_normal.cz;
        double cosI = ix*nx + iy*ny + iz*nz;
        if (cosI < 1e-6) continue;

        double sinIsq = 1.0 - cosI*cosI;
        complex gamma = sqrt(m_ri * m_ri - complex(sinIsq, 0));
        if (imag(gamma) < 0) gamma = -gamma;

        double cosT_real = sqrt(fmax(0.0, 1.0 - sinIsq/(re_n*re_n)));
        double eta = 1.0 / re_n;
        double Cv = cosT_real - eta * cosI;
        double rdx = eta*ix + Cv*nx;
        double rdy = eta*iy + Cv*ny;
        double rdz = eta*iz + Cv*nz;
        double rlen = sqrt(rdx*rdx + rdy*rdy + rdz*rdz);
        rdx /= rlen; rdy /= rlen; rdz /= rlen;

        // Fresnel transmission coefficients (air -> crystal)
        complex Ts = 2.0 * cosI / (cosI + gamma);
        complex Tp = 2.0 * m_ri * cosI / (m_ri * m_ri * cosI + gamma);

        // s/p basis from incidence plane
        double sx = iy*rdz - iz*rdy;
        double sy = iz*rdx - ix*rdz;
        double sz = ix*rdy - iy*rdx;
        double slen = sqrt(sx*sx + sy*sy + sz*sz);

        complex EY_x, EY_y, EY_z;
        complex EX_x, EX_y, EX_z;

        if (slen < 1e-6)
        {
            complex T = 0.5 * (Ts + Tp);
            EY_x = 0; EY_y = T; EY_z = 0;
            EX_x = -T; EX_y = 0; EX_z = 0;
        }
        else
        {
            sx /= slen; sy /= slen; sz /= slen;
            double pix_ = sy*iz - sz*iy;
            double piy_ = sz*ix - sx*iz;
            double piz_ = sx*iy - sy*ix;
            double pilen = sqrt(pix_*pix_ + piy_*piy_ + piz_*piz_);
            pix_ /= pilen; piy_ /= pilen; piz_ /= pilen;
            double prx = sy*rdz - sz*rdy;
            double pry = sz*rdx - sx*rdz;
            double prz = sx*rdy - sy*rdx;
            double prlen = sqrt(prx*prx + pry*pry + prz*prz);
            prx /= prlen; pry /= prlen; prz /= prlen;

            double Eys = sy;
            double Eyp = piy_;
            EY_x = Ts*Eys*sx + Tp*Eyp*prx;
            EY_y = Ts*Eys*sy + Tp*Eyp*pry;
            EY_z = Ts*Eys*sz + Tp*Eyp*prz;

            double Exs = -sx;
            double Exp_ = -pix_;
            EX_x = Ts*Exs*sx + Tp*Exp_*prx;
            EX_y = Ts*Exs*sy + Tp*Exp_*pry;
            EX_z = Ts*Exs*sz + Tp*Exp_*prz;
        }

        EntryData ed;
        ed.id = f;  ed.cosI = cosI;  ed.gamma = gamma;
        ed.rdx = rdx;  ed.rdy = rdy;  ed.rdz = rdz;
        ed.nx = nx;  ed.ny = ny;  ed.nz = nz;
        ed.fcx = (double)fac.center.cx;
        ed.fcy = (double)fac.center.cy;
        ed.fcz = (double)fac.center.cz;
        ed.phaseRef = ix*ed.fcx + iy*ed.fcy + iz*ed.fcz;
        ed.EY_x = EY_x; ed.EY_y = EY_y; ed.EY_z = EY_z;
        ed.EX_x = EX_x; ed.EX_y = EX_y; ed.EX_z = EX_z;
        entryList.push_back(ed);
    }

    // Step 2: For each (entry, exit) pair, precompute reflected PW amplitude.
    // The reflected PW = R · T · E_inc, with s/p decomposition at exit facet.
    // This is a crystal-wide PW: same amplitude vector for all dipoles.
    struct PairData {
        int entryIdx;
        int exitId;
        // Reflected amplitude vectors (constant, applied to all dipoles)
        complex rEY_x, rEY_y, rEY_z;
        complex rEX_x, rEX_y, rEX_z;
        // Phase parameters
        complex mu;
        double enx, eny, enz, ed;  // exit normal + d_param
        // Reflected direction and geometry for Kirchhoff weighting
        double reflx, refly, reflz;
        double cosI_exit;
        // Entry aperture: for tracing back from exit to entry facet
        double cosRefEntry;   // entryNormal · refractedDir
        double entry_d;       // entry facet d_param
    };

    vector<PairData> pairs;

    for (size_t ei = 0; ei < entryList.size(); ++ei)
    {
        const EntryData &ent = entryList[ei];
        for (int g = 0; g < nF; ++g)
        {
            if (g == ent.id) continue;
            const Facet &exitFac = m_particle->facets[g];
            double enx = (double)exitFac.in_normal.cx;
            double eny = (double)exitFac.in_normal.cy;
            double enz = (double)exitFac.in_normal.cz;

            // refDir must face exit facet's inner side
            double dnr = ent.rdx*enx + ent.rdy*eny + ent.rdz*enz;
            if (dnr > -1e-6) continue;

            double cosI_exit = -dnr;
            double sinI_exit_sq = 1.0 - cosI_exit * cosI_exit;

            // Complex Fresnel reflection (crystal -> air)
            complex sinT_sq = m_ri * m_ri * complex(sinI_exit_sq, 0);
            complex cosT_exit = sqrt(complex(1.0, 0) - sinT_sq);
            if (real(cosT_exit) < 0) cosT_exit = -cosT_exit;
            if (fabs(real(cosT_exit)) < 1e-10 && imag(cosT_exit) < 0)
                cosT_exit = -cosT_exit;

            complex Rs_c = (m_ri * cosI_exit - cosT_exit) / (m_ri * cosI_exit + cosT_exit);
            complex Rp_c = (cosI_exit - m_ri * cosT_exit) / (cosI_exit + m_ri * cosT_exit);

            bool isTIR = (real(sinT_sq) > 1.0 && fabs(imag(m_ri)) < 1e-10);
            if (isTIR) continue;

            // Filter by facet normal alignment (dotThreshold controls pair selection)
            double nA_dot_nB = ent.nx*enx + ent.ny*eny + ent.nz*enz;
            if (nA_dot_nB > dotThreshold) continue;

            // Skip negligible reflections
            if (fmax(abs(Rs_c), abs(Rp_c)) < 0.01) continue;

            cout << "  entry " << ent.id << " -> exit " << g
                 << "  cosI_exit=" << cosI_exit
                 << "  |Rs|=" << abs(Rs_c) << "  |Rp|=" << abs(Rp_c)
                 << "  nA·nB=" << nA_dot_nB << endl;

            // mu = incDir·n̂_B + (γ - cosI)·(n̂_A · n̂_B)
            double inc_dot_nB = ix*enx + iy*eny + iz*enz;
            complex mu = complex(inc_dot_nB, 0)
                       + (ent.gamma - complex(ent.cosI, 0)) * complex(nA_dot_nB, 0);

            // Decompose entry PW into s/p at exit facet, apply Fresnel R
            double eox = -enx, eoy = -eny, eoz = -enz;
            double sx = ent.rdy*eoz - ent.rdz*eoy;
            double sy = ent.rdz*eox - ent.rdx*eoz;
            double sz = ent.rdx*eoy - ent.rdy*eox;
            double slen = sqrt(sx*sx + sy*sy + sz*sz);

            double reflx = ent.rdx + 2.0*cosI_exit*enx;
            double refly = ent.rdy + 2.0*cosI_exit*eny;
            double reflz = ent.rdz + 2.0*cosI_exit*enz;

            PairData pd;
            pd.entryIdx = (int)ei;
            pd.exitId = g;
            pd.mu = mu;
            pd.enx = enx;  pd.eny = eny;  pd.enz = enz;
            pd.ed = exitFac.in_normal.d_param;
            pd.reflx = reflx;  pd.refly = refly;  pd.reflz = reflz;
            pd.cosI_exit = cosI_exit;
            pd.cosRefEntry = ent.nx*ent.rdx + ent.ny*ent.rdy + ent.nz*ent.rdz;
            pd.entry_d = m_particle->facets[ent.id].in_normal.d_param;

            if (slen < 1e-6)
            {
                // Normal incidence at exit: Rs = -Rp, use Rs
                complex R = Rs_c;
                pd.rEY_x = R * ent.EY_x;  pd.rEY_y = R * ent.EY_y;  pd.rEY_z = R * ent.EY_z;
                pd.rEX_x = R * ent.EX_x;  pd.rEX_y = R * ent.EX_y;  pd.rEX_z = R * ent.EX_z;
            }
            else
            {
                sx /= slen; sy /= slen; sz /= slen;
                // p_inc at exit = cross(s, refDir)
                double pv_x = sy*ent.rdz - sz*ent.rdy;
                double pv_y = sz*ent.rdx - sx*ent.rdz;
                double pv_z = sx*ent.rdy - sy*ent.rdx;
                double pl = sqrt(pv_x*pv_x + pv_y*pv_y + pv_z*pv_z);
                pv_x /= pl;  pv_y /= pl;  pv_z /= pl;
                // p_refl = cross(s, reflDir)
                double pr_x = sy*reflz - sz*refly;
                double pr_y = sz*reflx - sx*reflz;
                double pr_z = sx*refly - sy*reflx;
                double prl = sqrt(pr_x*pr_x + pr_y*pr_y + pr_z*pr_z);
                pr_x /= prl;  pr_y /= prl;  pr_z /= prl;

                // Decompose transmitted PW E-field into s/p at exit
                complex EYs = ent.EY_x*sx + ent.EY_y*sy + ent.EY_z*sz;
                complex EYp = ent.EY_x*pv_x + ent.EY_y*pv_y + ent.EY_z*pv_z;
                complex EXs = ent.EX_x*sx + ent.EX_y*sy + ent.EX_z*sz;
                complex EXp = ent.EX_x*pv_x + ent.EX_y*pv_y + ent.EX_z*pv_z;

                // Reflected amplitude = Rs*Es*s_hat + Rp*Ep*p_refl_hat
                pd.rEY_x = Rs_c*EYs*sx + Rp_c*EYp*pr_x;
                pd.rEY_y = Rs_c*EYs*sy + Rp_c*EYp*pr_y;
                pd.rEY_z = Rs_c*EYs*sz + Rp_c*EYp*pr_z;
                pd.rEX_x = Rs_c*EXs*sx + Rp_c*EXp*pr_x;
                pd.rEX_y = Rs_c*EXs*sy + Rp_c*EXp*pr_y;
                pd.rEX_z = Rs_c*EXs*sz + Rp_c*EXp*pr_z;
            }

            pairs.push_back(pd);
        }
    }

    cout << "General reflection: " << entryList.size() << " entries, "
         << pairs.size() << " pairs" << endl;

    // Precompute exit facet 2D geometry for Kirchhoff aperture weighting.
    // Each exit facet acts as a finite aperture for the reflected beam.
    struct Facet2D {
        double eu_x, eu_y, eu_z;
        double ev_x, ev_y, ev_z;
        double poly_u[MAX_VERTEX_NUM], poly_v[MAX_VERTEX_NUM];
        int nVert;
        double center_u, center_v;
        double radius;
        bool valid;
    };

    vector<Facet2D> exitData(nF);
    for (int f = 0; f < nF; ++f) exitData[f].valid = false;

    for (size_t pi = 0; pi < pairs.size(); ++pi)
    {
        int eid = pairs[pi].exitId;
        if (exitData[eid].valid) continue;

        const Facet &fac = m_particle->facets[eid];
        Facet2D &ef = exitData[eid];
        ef.valid = true;

        // Orthonormal basis on facet plane: e_u along first edge, e_v = cross(normal, e_u)
        double dx = (double)fac.arr[1].cx - (double)fac.arr[0].cx;
        double dy = (double)fac.arr[1].cy - (double)fac.arr[0].cy;
        double dz = (double)fac.arr[1].cz - (double)fac.arr[0].cz;
        double dlen = sqrt(dx*dx + dy*dy + dz*dz);
        ef.eu_x = dx/dlen; ef.eu_y = dy/dlen; ef.eu_z = dz/dlen;

        double fnx = (double)fac.in_normal.cx;
        double fny = (double)fac.in_normal.cy;
        double fnz = (double)fac.in_normal.cz;
        ef.ev_x = fny*ef.eu_z - fnz*ef.eu_y;
        ef.ev_y = fnz*ef.eu_x - fnx*ef.eu_z;
        ef.ev_z = fnx*ef.eu_y - fny*ef.eu_x;

        ef.nVert = fac.nVertices;
        double sum_u = 0, sum_v = 0;
        for (int v = 0; v < fac.nVertices; ++v)
        {
            double vx = (double)fac.arr[v].cx - (double)fac.arr[0].cx;
            double vy = (double)fac.arr[v].cy - (double)fac.arr[0].cy;
            double vz = (double)fac.arr[v].cz - (double)fac.arr[0].cz;
            ef.poly_u[v] = vx*ef.eu_x + vy*ef.eu_y + vz*ef.eu_z;
            ef.poly_v[v] = vx*ef.ev_x + vy*ef.ev_y + vz*ef.ev_z;
            sum_u += ef.poly_u[v];
            sum_v += ef.poly_v[v];
        }
        ef.center_u = sum_u / fac.nVertices;
        ef.center_v = sum_v / fac.nVertices;

        ef.radius = 0;
        for (int v = 0; v < fac.nVertices; ++v)
        {
            double du = ef.poly_u[v] - ef.center_u;
            double dv = ef.poly_v[v] - ef.center_v;
            double r = sqrt(du*du + dv*dv);
            if (r > ef.radius) ef.radius = r;
        }

        // Ensure CCW orientation (required by KirchhoffPolygonWeight / Green's theorem)
        double signedArea = 0;
        for (int v = 0; v < ef.nVert; ++v)
        {
            int v2 = (v + 1) % ef.nVert;
            signedArea += ef.poly_u[v]*ef.poly_v[v2] - ef.poly_u[v2]*ef.poly_v[v];
        }
        if (signedArea < 0)
        {
            for (int v = 0; v < ef.nVert / 2; ++v)
            {
                int v2 = ef.nVert - 1 - v;
                swap(ef.poly_u[v], ef.poly_u[v2]);
                swap(ef.poly_v[v], ef.poly_v[v2]);
            }
        }
    }

    // Precompute entry facet 2D geometry for double-aperture test.
    // The reflected beam from (entry A → exit B) only exists where the
    // transmitted beam from A actually illuminates exit facet B.
    vector<Facet2D> entryData(nF);
    for (int f = 0; f < nF; ++f) entryData[f].valid = false;

    for (size_t ei = 0; ei < entryList.size(); ++ei)
    {
        int eid = entryList[ei].id;
        if (entryData[eid].valid) continue;

        const Facet &fac = m_particle->facets[eid];
        Facet2D &af = entryData[eid];
        af.valid = true;

        double dx = (double)fac.arr[1].cx - (double)fac.arr[0].cx;
        double dy = (double)fac.arr[1].cy - (double)fac.arr[0].cy;
        double dz = (double)fac.arr[1].cz - (double)fac.arr[0].cz;
        double dlen = sqrt(dx*dx + dy*dy + dz*dz);
        af.eu_x = dx/dlen; af.eu_y = dy/dlen; af.eu_z = dz/dlen;

        double fnx = (double)fac.in_normal.cx;
        double fny = (double)fac.in_normal.cy;
        double fnz = (double)fac.in_normal.cz;
        af.ev_x = fny*af.eu_z - fnz*af.eu_y;
        af.ev_y = fnz*af.eu_x - fnx*af.eu_z;
        af.ev_z = fnx*af.eu_y - fny*af.eu_x;

        af.nVert = fac.nVertices;
        double sum_u = 0, sum_v = 0;
        for (int v = 0; v < fac.nVertices; ++v)
        {
            double vx = (double)fac.arr[v].cx - (double)fac.arr[0].cx;
            double vy = (double)fac.arr[v].cy - (double)fac.arr[0].cy;
            double vz = (double)fac.arr[v].cz - (double)fac.arr[0].cz;
            af.poly_u[v] = vx*af.eu_x + vy*af.eu_y + vz*af.eu_z;
            af.poly_v[v] = vx*af.ev_x + vy*af.ev_y + vz*af.ev_z;
            sum_u += af.poly_u[v];
            sum_v += af.poly_v[v];
        }
        af.center_u = sum_u / fac.nVertices;
        af.center_v = sum_v / fac.nVertices;

        af.radius = 0;
        for (int v = 0; v < fac.nVertices; ++v)
        {
            double du = af.poly_u[v] - af.center_u;
            double dv = af.poly_v[v] - af.center_v;
            double r = sqrt(du*du + dv*dv);
            if (r > af.radius) af.radius = r;
        }
    }

    double lambda_eff = m_wavelength / re_n;

    // Step 3: Apply Kirchhoff-weighted reflected PW to dipoles.
    // The reflected PW from (entry A → exit B) propagates through exit facet B
    // as a finite aperture. Dipoles whose back-projection along -reflDir falls
    // outside the exit facet get attenuated by Kirchhoff diffraction.
    int nKirch = 0, nDeepInterior = 0, nShadow = 0, nEntryShadow = 0;

    for (size_t pi = 0; pi < pairs.size(); ++pi)
    {
        const PairData &pd = pairs[pi];
        const EntryData &ent = entryList[pd.entryIdx];
        const Facet2D &ef = exitData[pd.exitId];
        const Facet &exitFac = m_particle->facets[pd.exitId];

        double re_gamma = real(ent.gamma);
        double im_gamma = imag(ent.gamma);
        double re_mu = real(pd.mu);
        double im_mu = imag(pd.mu);

        int nAdded = 0;
        for (size_t i = 0; i < m_dipoles.size(); ++i)
        {
            DipoleField &dip = m_dipoles[i];

            // Only apply to dipoles assigned to this entry facet
            if (dip.assignedFacet != ent.id) continue;

            // dist_B: distance from dipole to exit facet plane (must be positive)
            double dist_B = pd.enx*dip.x + pd.eny*dip.y + pd.enz*dip.z + pd.ed;
            if (dist_B < 1e-10) continue;

            // Propagation distance from exit facet to dipole along reflected direction
            double z_prop = dist_B / pd.cosI_exit;

            // Back-project dipole to exit facet plane along -reflDir
            double proj_x = dip.x - z_prop * pd.reflx;
            double proj_y = dip.y - z_prop * pd.refly;
            double proj_z = dip.z - z_prop * pd.reflz;

            // --- Entry aperture test ---
            // From P_exit, trace back along -refractedDir to entry facet A plane.
            // If P_entry is outside entry facet A, this reflected path doesn't exist.
            if (pd.cosRefEntry > 1e-10)
            {
                double dist_A = ent.nx*proj_x + ent.ny*proj_y + ent.nz*proj_z + pd.entry_d;
                double t_entry = dist_A / pd.cosRefEntry;
                if (t_entry >= 0)
                {
                    double pe_x = proj_x - t_entry * ent.rdx;
                    double pe_y = proj_y - t_entry * ent.rdy;
                    double pe_z = proj_z - t_entry * ent.rdz;

                    const Facet2D &af = entryData[ent.id];
                    const Facet &entryFac = m_particle->facets[ent.id];
                    double aox = pe_x - (double)entryFac.arr[0].cx;
                    double aoy = pe_y - (double)entryFac.arr[0].cy;
                    double aoz = pe_z - (double)entryFac.arr[0].cz;
                    double au = aox*af.eu_x + aoy*af.eu_y + aoz*af.eu_z;
                    double av = aox*af.ev_x + aoy*af.ev_y + aoz*af.ev_z;

                    // Quick bounding-circle rejection
                    double adu = au - af.center_u, adv = av - af.center_v;
                    if (sqrt(adu*adu + adv*adv) > af.radius + 0.5*m_gridSpacing)
                    {
                        ++nEntryShadow;
                        continue;
                    }

                    // Ray-casting point-in-polygon (binary inside/outside)
                    int crossings = 0;
                    for (int ei2 = 0; ei2 < af.nVert; ++ei2)
                    {
                        int ej2 = (ei2 + 1) % af.nVert;
                        double vi_v = af.poly_v[ei2], vj_v = af.poly_v[ej2];
                        if ((vi_v <= av && vj_v > av) || (vj_v <= av && vi_v > av))
                        {
                            double t_rc = (av - vi_v) / (vj_v - vi_v);
                            if (au < af.poly_u[ei2] + t_rc * (af.poly_u[ej2] - af.poly_u[ei2]))
                                ++crossings;
                        }
                    }
                    if ((crossings & 1) == 0)
                    {
                        ++nEntryShadow;
                        continue;
                    }
                }
            }
            // --- End entry aperture test ---

            // Convert projection to exit facet 2D coordinates
            double ox = proj_x - (double)exitFac.arr[0].cx;
            double oy = proj_y - (double)exitFac.arr[0].cy;
            double oz = proj_z - (double)exitFac.arr[0].cz;
            double obs_u = ox*ef.eu_x + oy*ef.eu_y + oz*ef.eu_z;
            double obs_v = ox*ef.ev_x + oy*ef.ev_y + oz*ef.ev_z;

            // Quick rejection: far outside bounding circle + Fresnel margin
            double du = obs_u - ef.center_u, dv = obs_v - ef.center_v;
            double dist2D = sqrt(du*du + dv*dv);
            double fresnelScale = sqrt(lambda_eff * z_prop);
            if (dist2D > ef.radius + 5.0 * fresnelScale)
            {
                ++nShadow;
                continue;
            }

            // Minimum distance from projection to nearest polygon edge
            double minEdgeDist = 1e30;
            for (int ei2 = 0; ei2 < ef.nVert; ++ei2)
            {
                int ej2 = (ei2 + 1) % ef.nVert;
                double eu_ = ef.poly_u[ej2] - ef.poly_u[ei2];
                double ev_ = ef.poly_v[ej2] - ef.poly_v[ei2];
                double el2 = eu_*eu_ + ev_*ev_;
                if (el2 < 1e-20) continue;
                double duu = obs_u - ef.poly_u[ei2], dvv = obs_v - ef.poly_v[ei2];
                double tc = (duu*eu_ + dvv*ev_) / el2;
                tc = (tc < 0) ? 0 : (tc > 1) ? 1 : tc;
                double ddx = duu - tc*eu_, ddy = dvv - tc*ev_;
                double d2 = sqrt(ddx*ddx + ddy*ddy);
                if (d2 < minEdgeDist) minEdgeDist = d2;
            }

            complex weight(1.0, 0.0);
            if (minEdgeDist > 5.0 * fresnelScale)
            {
                // Deep interior or deep shadow: ray-casting point-in-polygon test
                int crossings = 0;
                for (int ei2 = 0; ei2 < ef.nVert; ++ei2)
                {
                    int ej2 = (ei2 + 1) % ef.nVert;
                    double vi_v = ef.poly_v[ei2], vj_v = ef.poly_v[ej2];
                    if ((vi_v <= obs_v && vj_v > obs_v) || (vj_v <= obs_v && vi_v > obs_v))
                    {
                        double t_rc = (obs_v - vi_v) / (vj_v - vi_v);
                        if (obs_u < ef.poly_u[ei2] + t_rc * (ef.poly_u[ej2] - ef.poly_u[ei2]))
                            ++crossings;
                    }
                }
                if ((crossings & 1) == 0)
                {
                    ++nShadow;
                    continue;  // deep shadow: weight ≈ 0
                }
                ++nDeepInterior;
                // weight stays 1.0
            }
            else if (m_noKirchhoff)
            {
                // No Kirchhoff: simple point-in-polygon (sharp aperture)
                int crossings2 = 0;
                for (int ei2 = 0; ei2 < ef.nVert; ++ei2)
                {
                    int ej2 = (ei2 + 1) % ef.nVert;
                    double vi_v = ef.poly_v[ei2], vj_v = ef.poly_v[ej2];
                    if ((vi_v <= obs_v && vj_v > obs_v) || (vj_v <= obs_v && vi_v > obs_v))
                    {
                        double t_rc = (obs_v - vi_v) / (vj_v - vi_v);
                        if (obs_u < ef.poly_u[ei2] + t_rc * (ef.poly_u[ej2] - ef.poly_u[ei2]))
                            ++crossings2;
                    }
                }
                if ((crossings2 & 1) == 0)
                {
                    ++nShadow;
                    continue;
                }
                ++nDeepInterior;
                // weight stays 1.0
            }
            else
            {
                // Near edge: full Kirchhoff diffraction weight
                weight = KirchhoffPolygonWeight(
                    ef.poly_u, ef.poly_v, ef.nVert,
                    obs_u, obs_v, z_prop, lambda_eff);
                // Cap at 1.0: higher values are quadrature artifacts from distant
                // oscillating edges (despite asymptotic fix for X > 3 Fresnel units).
                double wm = abs(weight);
                if (wm > 1.0)
                    weight = weight * (1.0 / wm);
                ++nKirch;
            }

            if (norm(weight) < 1e-8) continue;

            // Entry PW phase at this dipole
            double ddx = dip.x - ent.fcx;
            double ddy = dip.y - ent.fcy;
            double ddz = dip.z - ent.fcz;
            double inc_dot_dr = ix*ddx + iy*ddy + iz*ddz;
            double nA_dot_dr = ent.nx*ddx + ent.ny*ddy + ent.nz*ddz;

            // Combined phase: entry PW + reflection
            double phase_re = ent.phaseRef + inc_dot_dr
                            + (re_gamma - ent.cosI) * nA_dot_dr
                            - 2.0 * re_mu * dist_B;
            double phase_decay = im_gamma * nA_dot_dr - 2.0 * im_mu * dist_B;

            complex phase = exp_im(m_k * phase_re);
            if (fabs(phase_decay) > 1e-15)
                phase = phase * exp(-m_k * phase_decay);

            complex wPhase = weight * phase;
            dip.Ex_Y += pd.rEY_x * wPhase;
            dip.Ey_Y += pd.rEY_y * wPhase;
            dip.Ez_Y += pd.rEY_z * wPhase;
            dip.Ex_X += pd.rEX_x * wPhase;
            dip.Ey_X += pd.rEX_y * wPhase;
            dip.Ez_X += pd.rEX_z * wPhase;
            ++nAdded;
        }

        cout << "  Pair " << ent.id << "->" << pd.exitId
             << ": applied to " << nAdded << " dipoles" << endl;
    }

    cout << "  Kirchhoff: " << nKirch << " edge, " << nDeepInterior << " interior, "
         << nShadow << " exit-shadow, " << nEntryShadow << " entry-shadow" << endl;
}

void ADDAFieldComputer::AccumulateReflectedBeams(
    const std::vector<InternalBeamSegment> &segments,
    const Point3f &incidentDir,
    int maxNActs, bool incoherent, int minNActs)
{
    double re_n = real(m_ri);
    double im_n = imag(m_ri);

    double iDirX = (double)incidentDir.cx;
    double iDirY = (double)incidentDir.cy;
    double iDirZ = (double)incidentDir.cz;

    // Incoherent mode: accumulate reflected intensity per dipole, scale at the end
    std::vector<double> reflIntY, reflIntX;
    if (incoherent)
    {
        reflIntY.assign(m_dipoles.size(), 0.0);
        reflIntX.assign(m_dipoles.size(), 0.0);
    }

    int nProcessed = 0;
    int nDipolesUpdated = 0;
    int nEntryMissed = 0;

    for (size_t si = 0; si < segments.size(); ++si)
    {
        const InternalBeamSegment &seg = segments[si];
        if (seg.nActs < minNActs || seg.nActs > maxNActs)
            continue;

        ++nProcessed;

        // Beam local polarization basis
        Point3d e_perp(seg.polarizationBasis.cx,
                       seg.polarizationBasis.cy,
                       seg.polarizationBasis.cz);
        Point3d dDir(seg.direction.cx, seg.direction.cy, seg.direction.cz);
        Point3d e_par = CrossProductD(dDir, e_perp);
        double len = LengthD(e_par);
        if (len < 1e-12) continue;
        e_par = e_par / len;

        // Compute polygon normal for containment test
        Point3f polyNormal(0, 0, 0);
        if (seg.polyNVertices >= 3)
        {
            Point3f e1 = seg.polyArr[1] - seg.polyArr[0];
            Point3f e2 = seg.polyArr[2] - seg.polyArr[0];
            polyNormal = CrossProduct(e1, e2);
            float pnLen = (float)Length(polyNormal);
            if (pnLen > 1e-10f)
                polyNormal = polyNormal / (double)pnLen;
        }

        // Reflection facet normal (in_normal points into crystal)
        const Facet &reflFacet = m_particle->facets[seg.entryFacetId];
        double nBx = (double)reflFacet.in_normal.cx;
        double nBy = (double)reflFacet.in_normal.cy;
        double nBz = (double)reflFacet.in_normal.cz;
        double nBd = reflFacet.in_normal.d_param;

        // Recover d_inc (internal direction before reflection)
        double dDotN = dDir.x * nBx + dDir.y * nBy + dDir.z * nBz;
        double dIncX = dDir.x - 2.0 * dDotN * nBx;
        double dIncY = dDir.y - 2.0 * dDotN * nBy;
        double dIncZ = dDir.z - 2.0 * dDotN * nBz;

        // Anti-parallel filter for nActs==1
        if (seg.nActs == 1)
        {
            double pcx = 0, pcy = 0, pcz = 0;
            for (int vi = 0; vi < seg.polyNVertices; ++vi)
            {
                pcx += (double)seg.polyArr[vi].cx;
                pcy += (double)seg.polyArr[vi].cy;
                pcz += (double)seg.polyArr[vi].cz;
            }
            pcx /= seg.polyNVertices;
            pcy /= seg.polyNVertices;
            pcz /= seg.polyNVertices;

            double dp_pc = pcx*nBx + pcy*nBy + pcz*nBz + nBd;
            if (fabs(dDotN) > 1e-10)
            {
                double t_bp = dp_pc / dDotN;
                pcx -= t_bp * dDir.x;
                pcy -= t_bp * dDir.y;
                pcz -= t_bp * dDir.z;
            }

            double minT = 1e30;
            int entryFacet = -1;
            for (int f = 0; f < m_particle->nFacets; ++f)
            {
                if (f == seg.entryFacetId) continue;
                const Facet &fac = m_particle->facets[f];
                double nfx = (double)fac.in_normal.cx;
                double nfy = (double)fac.in_normal.cy;
                double nfz = (double)fac.in_normal.cz;
                double nfd = fac.in_normal.d_param;

                double dp_pf = pcx*nfx + pcy*nfy + pcz*nfz + nfd;
                double dp_df = dIncX*nfx + dIncY*nfy + dIncZ*nfz;
                if (fabs(dp_df) < 1e-10) continue;

                double t_f = dp_pf / dp_df;
                if (t_f < 1e-6 || t_f >= minT) continue;

                minT = t_f;
                entryFacet = f;
            }

            if (entryFacet >= 0)
            {
                const Facet &entFac = m_particle->facets[entryFacet];
                double nAx = (double)entFac.in_normal.cx;
                double nAy = (double)entFac.in_normal.cy;
                double nAz = (double)entFac.in_normal.cz;
                double nA_dot_nB = nAx*nBx + nAy*nBy + nAz*nBz;
                if (nA_dot_nB > -0.5)
                    continue;
            }
        }

        int segUpdated = 0;
        for (size_t di = 0; di < m_dipoles.size(); ++di)
        {
            DipoleField &dip = m_dipoles[di];

            Point3f r((float)dip.x, (float)dip.y, (float)dip.z);

            // Check within segment bounds
            double t = (double)seg.direction.cx * r.cx
                     + (double)seg.direction.cy * r.cy
                     + (double)seg.direction.cz * r.cz
                     + seg.front;

            if (t < -1e-6 || t > seg.segmentLength + 1e-6)
                continue;

            // Check if dipole projects inside exit polygon
            Point3f exitNorm = seg.exitNormal;
            double dp_nr = (double)r.cx * exitNorm.cx
                         + (double)r.cy * exitNorm.cy
                         + (double)r.cz * exitNorm.cz
                         + seg.exitD;
            double dp_nd = (double)seg.direction.cx * exitNorm.cx
                         + (double)seg.direction.cy * exitNorm.cy
                         + (double)seg.direction.cz * exitNorm.cz;

            if (fabs(dp_nd) < 1e-10)
                continue;

            double t_exit = -dp_nr / dp_nd;
            Point3f r_proj(r.cx + seg.direction.cx * (float)t_exit,
                           r.cy + seg.direction.cy * (float)t_exit,
                           r.cz + seg.direction.cz * (float)t_exit);

            if (!IsInsidePolygon(r_proj, seg.polyArr, seg.polyNVertices, polyNormal))
                continue;

            // === Per-dipole phase computation ===

            double total_OP;
            double internalDist;
            double phase_decay = 0;

            if (seg.nActs == 1)
            {
                // Full back-projection for nActs=1 (exact 2-leg path)
                double dp_rn_refl = dip.x * nBx + dip.y * nBy + dip.z * nBz + nBd;
                if (fabs(dDotN) < 1e-10) continue;
                double alpha = -dp_rn_refl / dDotN;
                double Rx = dip.x + alpha * dDir.x;
                double Ry = dip.y + alpha * dDir.y;
                double Rz = dip.z + alpha * dDir.z;
                double dist_R_to_dip = fabs(alpha);

                double minT_entry = 1e30;
                int entryFacetFound = -1;
                for (int f = 0; f < m_particle->nFacets; ++f)
                {
                    if (f == seg.entryFacetId) continue;
                    const Facet &fac = m_particle->facets[f];
                    double nfx = (double)fac.in_normal.cx;
                    double nfy = (double)fac.in_normal.cy;
                    double nfz = (double)fac.in_normal.cz;
                    double nfd = fac.in_normal.d_param;

                    double dp_Rf = Rx * nfx + Ry * nfy + Rz * nfz + nfd;
                    double dp_df = dIncX * nfx + dIncY * nfy + dIncZ * nfz;
                    if (fabs(dp_df) < 1e-10) continue;

                    double t_f = dp_Rf / dp_df;
                    if (t_f < 1e-6 || t_f >= minT_entry) continue;

                    minT_entry = t_f;
                    entryFacetFound = f;
                }

                if (entryFacetFound < 0)
                {
                    ++nEntryMissed;
                    continue;
                }

                double dist_PR = minT_entry;
                double Px = Rx - dist_PR * dIncX;
                double Py = Ry - dist_PR * dIncY;
                double Pz = Rz - dist_PR * dIncZ;

                double externalOP = iDirX * Px + iDirY * Py + iDirZ * Pz;
                internalDist = dist_PR + dist_R_to_dip;
                total_OP = externalOP + re_n * internalDist;

                // Attenuation via Im(gamma) * normal distance (exact for PW)
                if (im_n > 1e-15)
                {
                    const Facet &entFac = m_particle->facets[entryFacetFound];
                    double nEx = (double)entFac.in_normal.cx;
                    double nEy = (double)entFac.in_normal.cy;
                    double nEz = (double)entFac.in_normal.cz;
                    double cosI_entry = iDirX*nEx + iDirY*nEy + iDirZ*nEz;
                    double sinIsq_e = 1.0 - cosI_entry*cosI_entry;
                    complex gamma_e = sqrt(m_ri*m_ri - complex(sinIsq_e, 0));
                    if (imag(gamma_e) < 0) gamma_e = -gamma_e;
                    // Normal distance from entry P to dipole (along entry facet normal)
                    double dr_n_e = nEx*(dip.x-Px) + nEy*(dip.y-Py) + nEz*(dip.z-Pz);
                    // Reflected beam: forward leg + return leg both decay along n̂_entry
                    // Total normal dist for 2-leg path: n̂·(R-P) + n̂·(R-D) = 2*n̂·(R-P) - n̂·(D-P)
                    double nR = nEx*Rx + nEy*Ry + nEz*Rz;
                    double nP = nEx*Px + nEy*Py + nEz*Pz;
                    double decay_dist = 2.0*(nR-nP) - dr_n_e;
                    phase_decay = imag(gamma_e) * decay_dist;
                }
            }
            else
            {
                // Simplified phase for nActs > 1
                double t_disp = dDir.x * dip.x + dDir.y * dip.y + dDir.z * dip.z + seg.front;
                internalDist = t_disp;
                total_OP = seg.opticalPath + t_disp * re_n;

                if (im_n > 1e-15)
                    phase_decay = im_n * internalDist;
            }

            complex phase = exp_im(m_k * total_OP);
            if (phase_decay > 1e-15)
            {
                phase = phase * exp(-m_k * phase_decay);
            }

            // Y-polarization: pure s-input -> (0, 1) in (p, s) basis
            complex Ep_Y = seg.J.m22 * phase;
            complex Epar_Y = seg.J.m12 * phase;
            complex dEx_Y = Ep_Y * e_perp.x + Epar_Y * e_par.x;
            complex dEy_Y = Ep_Y * e_perp.y + Epar_Y * e_par.y;
            complex dEz_Y = Ep_Y * e_perp.z + Epar_Y * e_par.z;

            // X-polarization: pure -p-input -> (-1, 0) in (p, s) basis
            complex Ep_X = -seg.J.m21 * phase;
            complex Epar_X = -seg.J.m11 * phase;
            complex dEx_X = Ep_X * e_perp.x + Epar_X * e_par.x;
            complex dEy_X = Ep_X * e_perp.y + Epar_X * e_par.y;
            complex dEz_X = Ep_X * e_perp.z + Epar_X * e_par.z;

            if (incoherent)
            {
                reflIntY[di] += norm(dEx_Y) + norm(dEy_Y) + norm(dEz_Y);
                reflIntX[di] += norm(dEx_X) + norm(dEy_X) + norm(dEz_X);
            }
            else
            {
                dip.Ex_Y += dEx_Y;
                dip.Ey_Y += dEy_Y;
                dip.Ez_Y += dEz_Y;
                dip.Ex_X += dEx_X;
                dip.Ey_X += dEy_X;
                dip.Ez_X += dEz_X;
            }

            ++segUpdated;
        }
        nDipolesUpdated += segUpdated;
    }

    // Incoherent post-processing: scale refracted field by reflected energy
    if (incoherent)
    {
        int nScaled = 0;
        for (size_t di = 0; di < m_dipoles.size(); ++di)
        {
            DipoleField &dip = m_dipoles[di];

            double I_Y = norm(dip.Ex_Y) + norm(dip.Ey_Y) + norm(dip.Ez_Y);
            if (I_Y > 1e-20 && reflIntY[di] > 1e-20)
            {
                double scale = sqrt(1.0 + reflIntY[di] / I_Y);
                dip.Ex_Y *= scale;
                dip.Ey_Y *= scale;
                dip.Ez_Y *= scale;
            }

            double I_X = norm(dip.Ex_X) + norm(dip.Ey_X) + norm(dip.Ez_X);
            if (I_X > 1e-20 && reflIntX[di] > 1e-20)
            {
                double scale = sqrt(1.0 + reflIntX[di] / I_X);
                dip.Ex_X *= scale;
                dip.Ey_X *= scale;
                dip.Ez_X *= scale;
            }

            if (reflIntY[di] > 1e-20 || reflIntX[di] > 1e-20)
                ++nScaled;
        }
        cout << "Incoherent scaling applied to " << nScaled << " dipoles" << endl;
    }

    cout << "GO reflected beams (nActs=1.." << maxNActs << ", "
         << (incoherent ? "incoherent" : "coherent") << "): "
         << nProcessed << " segments, "
         << nDipolesUpdated << " dipole updates, "
         << nEntryMissed << " entry-facet misses" << endl;
}

void ADDAFieldComputer::DiagnoseGOvsPW(const Point3f &incidentDir)
{
    // 1. Print contribution statistics
    int n0 = 0, n1 = 0, n2 = 0, n3plus = 0;
    for (size_t i = 0; i < m_dipoles.size(); ++i)
    {
        int nc = m_dipoles[i].nContributions;
        if (nc == 0) ++n0;
        else if (nc == 1) ++n1;
        else if (nc == 2) ++n2;
        else ++n3plus;
    }
    cout << "\n=== GO Field Diagnostics ===" << endl;
    cout << "Contributions: 0=" << n0 << " 1=" << n1
         << " 2=" << n2 << " 3+=" << n3plus << endl;

    // 2. For GO-covered dipoles, compute per-facet PW independently and compare
    double re_n = real(m_ri);
    double ix = (double)incidentDir.cx;
    double iy = (double)incidentDir.cy;
    double iz = (double)incidentDir.cz;



    // Build per-facet refraction data (same as FillUncoveredPerFacet)
    struct FacetInfo {
        int id;
        double cosI;
        Point3f refDir;
        double Ts, Tp;
        double EY_x, EY_y, EY_z;
        double EX_x, EX_y, EX_z;
        double phaseRef;
    };
    vector<FacetInfo> facets;

    for (int f = 0; f < m_particle->nFacets; ++f)
    {
        const Facet &fac = m_particle->facets[f];
        double nx = (double)fac.in_normal.cx;
        double ny = (double)fac.in_normal.cy;
        double nz = (double)fac.in_normal.cz;
        double cosI = ix*nx + iy*ny + iz*nz;
        if (cosI < 1e-6) continue;

        double sinIsq = 1.0 - cosI*cosI;
        double cosT = sqrt(fmax(0.0, 1.0 - sinIsq/(re_n*re_n)));
        double eta = 1.0 / re_n;
        double C = cosT - eta * cosI;
        double rdx = eta*ix + C*nx;
        double rdy = eta*iy + C*ny;
        double rdz = eta*iz + C*nz;
        double rlen = sqrt(rdx*rdx + rdy*rdy + rdz*rdz);
        rdx /= rlen; rdy /= rlen; rdz /= rlen;

        double Ts = 2.0*cosI / (cosI + re_n*cosT);
        double Tp = 2.0*cosI / (re_n*cosI + cosT);

        double sx = iy*rdz - iz*rdy;
        double sy = iz*rdx - ix*rdz;
        double sz = ix*rdy - iy*rdx;
        double slen = sqrt(sx*sx + sy*sy + sz*sz);

        double EY_x, EY_y, EY_z;
        double EX_x, EX_y, EX_z;

        if (slen < 1e-6)
        {
            double T = 0.5*(Ts+Tp);
            EY_x=0; EY_y=T; EY_z=0;
            EX_x=-T; EX_y=0; EX_z=0;
        }
        else
        {
            sx/=slen; sy/=slen; sz/=slen;
            double pix_=sy*iz-sz*iy, piy_=sz*ix-sx*iz, piz_=sx*iy-sy*ix;
            double pilen=sqrt(pix_*pix_+piy_*piy_+piz_*piz_);
            pix_/=pilen; piy_/=pilen; piz_/=pilen;
            double prx=sy*rdz-sz*rdy, pry=sz*rdx-sx*rdz, prz=sx*rdy-sy*rdx;
            double prlen=sqrt(prx*prx+pry*pry+prz*prz);
            prx/=prlen; pry/=prlen; prz/=prlen;

            double Eys=sy, Eyp=piy_;
            EY_x=Ts*Eys*sx+Tp*Eyp*prx;
            EY_y=Ts*Eys*sy+Tp*Eyp*pry;
            EY_z=Ts*Eys*sz+Tp*Eyp*prz;
            double Exs=-sx, Exp_=-pix_;
            EX_x=Ts*Exs*sx+Tp*Exp_*prx;
            EX_y=Ts*Exs*sy+Tp*Exp_*pry;
            EX_z=Ts*Exs*sz+Tp*Exp_*prz;
        }

        FacetInfo fi;
        fi.id=f; fi.cosI=cosI;
        fi.refDir=Point3f((float)rdx,(float)rdy,(float)rdz);
        fi.Ts=Ts; fi.Tp=Tp;
        fi.EY_x=EY_x; fi.EY_y=EY_y; fi.EY_z=EY_z;
        fi.EX_x=EX_x; fi.EX_y=EX_y; fi.EX_z=EX_z;
        double cx_=(double)fac.center.cx, cy_=(double)fac.center.cy, cz_=(double)fac.center.cz;
        fi.phaseRef=ix*cx_+iy*cy_+iz*cz_;
        facets.push_back(fi);
    }

    // For the first 10 GO-covered dipoles, compare GO vs PW
    int nPrinted = 0;
    double sumErr2_Y = 0, sumNorm2_Y = 0;
    double sumErr2_X = 0, sumNorm2_X = 0;
    int nCompared = 0;

    for (size_t i = 0; i < m_dipoles.size() && nCompared < 5000; ++i)
    {
        DipoleField &dip = m_dipoles[i];
        if (dip.nContributions == 0) continue;

        // Find the matching facet for this dipole (back-project)
        bool foundFacet = false;
        for (size_t fi = 0; fi < facets.size(); ++fi)
        {
            const FacetInfo &fr = facets[fi];
            const Facet &fac = m_particle->facets[fr.id];
            double dx=(double)fr.refDir.cx, dy=(double)fr.refDir.cy, dz=(double)fr.refDir.cz;

            double dp_nr = dip.x*fac.in_normal.cx + dip.y*fac.in_normal.cy
                         + dip.z*fac.in_normal.cz + fac.in_normal.d_param;
            double dp_nd = dx*fac.in_normal.cx + dy*fac.in_normal.cy + dz*fac.in_normal.cz;
            if (fabs(dp_nd) < 1e-10) continue;
            double t_back = -dp_nr / dp_nd;
            if (t_back > 1e-6) continue;

            Point3f proj((float)(dip.x+dx*t_back),(float)(dip.y+dy*t_back),(float)(dip.z+dz*t_back));
            Point3f polyNormal(0,0,0);
            if (fac.nVertices >= 3)
            {
                Point3f e1=fac.arr[1]-fac.arr[0], e2=fac.arr[2]-fac.arr[0];
                polyNormal=CrossProduct(e1,e2);
                float pnLen=(float)Length(polyNormal);
                if (pnLen > 1e-10f) polyNormal=polyNormal/(double)pnLen;
            }
            if (!IsInsidePolygon(proj, fac.arr, fac.nVertices, polyNormal)) continue;

            // Compute PW field for this dipole
            double cx_=(double)fac.center.cx, cy_=(double)fac.center.cy, cz_=(double)fac.center.cz;
            double t_int = dx*(dip.x-cx_) + dy*(dip.y-cy_) + dz*(dip.z-cz_);
            double total_OP = fr.phaseRef + re_n * t_int;
            complex phase = exp_im(m_k * total_OP);

            complex pw_Ex_Y = fr.EY_x * phase;
            complex pw_Ey_Y = fr.EY_y * phase;
            complex pw_Ez_Y = fr.EY_z * phase;
            complex pw_Ex_X = fr.EX_x * phase;
            complex pw_Ey_X = fr.EX_y * phase;
            complex pw_Ez_X = fr.EX_z * phase;

            // Accumulate error
            sumErr2_Y += norm(dip.Ex_Y - pw_Ex_Y) + norm(dip.Ey_Y - pw_Ey_Y) + norm(dip.Ez_Y - pw_Ez_Y);
            sumNorm2_Y += norm(pw_Ex_Y) + norm(pw_Ey_Y) + norm(pw_Ez_Y);
            sumErr2_X += norm(dip.Ex_X - pw_Ex_X) + norm(dip.Ey_X - pw_Ey_X) + norm(dip.Ez_X - pw_Ez_X);
            sumNorm2_X += norm(pw_Ex_X) + norm(pw_Ey_X) + norm(pw_Ez_X);
            ++nCompared;

            // Print first few
            if (nPrinted < 5)
            {
                cout << "  Dip#" << i << " nc=" << dip.nContributions
                     << " r=(" << dip.x << "," << dip.y << "," << dip.z
                     << ") facet=" << fr.id << endl;
                cout << "    GO  Y: (" << real(dip.Ex_Y) << "," << real(dip.Ey_Y) << "," << real(dip.Ez_Y)
                     << ") + i(" << imag(dip.Ex_Y) << "," << imag(dip.Ey_Y) << "," << imag(dip.Ez_Y) << ")" << endl;
                cout << "    PW  Y: (" << real(pw_Ex_Y) << "," << real(pw_Ey_Y) << "," << real(pw_Ez_Y)
                     << ") + i(" << imag(pw_Ex_Y) << "," << imag(pw_Ey_Y) << "," << imag(pw_Ez_Y) << ")" << endl;
                cout << "    GO  X: (" << real(dip.Ex_X) << "," << real(dip.Ey_X) << "," << real(dip.Ez_X)
                     << ") + i(" << imag(dip.Ex_X) << "," << imag(dip.Ey_X) << "," << imag(dip.Ez_X) << ")" << endl;
                cout << "    PW  X: (" << real(pw_Ex_X) << "," << real(pw_Ey_X) << "," << real(pw_Ez_X)
                     << ") + i(" << imag(pw_Ex_X) << "," << imag(pw_Ey_X) << "," << imag(pw_Ez_X) << ")" << endl;
                ++nPrinted;
            }
            foundFacet = true;
            break;
        }
    }

    if (nCompared > 0)
    {
        double relErr_Y = sqrt(sumErr2_Y / sumNorm2_Y);
        double relErr_X = sqrt(sumErr2_X / sumNorm2_X);
        cout << "GO vs PW comparison (" << nCompared << " dipoles):" << endl;
        cout << "  Y-pol relative error: " << relErr_Y << endl;
        cout << "  X-pol relative error: " << relErr_X << endl;
    }
    cout << "=== End Diagnostics ===\n" << endl;
}

// sinc(x) = sin(x)/x, sinc(0) = 1
static inline double sinc_f(double x)
{
    return (fabs(x) > 1e-15) ? sin(x) / x : 1.0;
}

void ADDAFieldComputer::ComputeFarFieldCore(double thetaMinDeg, double thetaMaxDeg,
                                             int nTheta,
                                             vector<double> &muellerFlat,
                                             double &CextY, double &CextX,
                                             const Point3f *incidentDir,
                                             double *Cgeo_out) const
{
    double d = m_gridSpacing;
    double k = m_k;
    double kd = k * d;

    complex m2 = m_ri * m_ri;
    complex chi_V = (d * d * d / (4.0 * M_PI)) * (m2 - 1.0);
    complex F_base = complex(0.0, -1.0) * (k * k * k) * chi_V;

    double dTheta = (nTheta > 1) ? (thetaMaxDeg - thetaMinDeg) / (nTheta - 1) : 0;

    muellerFlat.resize(nTheta * 17);

    for (int it = 0; it < nTheta; ++it)
    {
        double thetaDeg = thetaMinDeg + it * dTheta;
        double theta = thetaDeg * M_PI / 180.0;
        double sinTh = sin(theta);
        double cosTh = cos(theta);

        double nx = sinTh, ny = 0.0, nz = -cosTh;
        double epar_x = cosTh, epar_z = sinTh;

        // Exact IGT sinc correction for this scattering direction
        double eta = sinc_f(kd * nx / 2.0) * sinc_f(kd * ny / 2.0) * sinc_f(kd * nz / 2.0);
        complex F = F_base * eta;

        complex sumYx(0,0), sumYy(0,0), sumYz(0,0);
        complex sumXx(0,0), sumXy(0,0), sumXz(0,0);

        for (size_t j = 0; j < m_dipoles.size(); ++j)
        {
            const DipoleField &dip = m_dipoles[j];
            if (dip.nContributions <= 0 && dip.assignedFacet < 0)
                continue;

            double rdotn = dip.x * nx + dip.y * ny + dip.z * nz;
            complex phase = exp_im(-k * rdotn);

            sumYx += dip.Ex_Y * phase; sumYy += dip.Ey_Y * phase; sumYz += dip.Ez_Y * phase;
            sumXx += dip.Ex_X * phase; sumXy += dip.Ey_X * phase; sumXz += dip.Ez_X * phase;
        }

        sumYx = F * sumYx; sumYy = F * sumYy; sumYz = F * sumYz;
        sumXx = F * (-sumXx); sumXy = F * (-sumXy); sumXz = F * (-sumXz);

        complex ndotSY = nx * sumYx + ny * sumYy + nz * sumYz;
        complex SYx = sumYx - nx * ndotSY;
        complex SYy = sumYy - ny * ndotSY;
        complex SYz = sumYz - nz * ndotSY;

        complex ndotSX = nx * sumXx + ny * sumXy + nz * sumXz;
        complex SXx = sumXx - nx * ndotSX;
        complex SXy = sumXy - ny * ndotSX;
        complex SXz = sumXz - nz * ndotSX;

        complex S1 = SYy;
        complex S2 = epar_x * SXx + epar_z * SXz;
        complex S3 = epar_x * SYx + epar_z * SYz;
        complex S4 = SXy;

        double s1s = norm(S1), s2s = norm(S2), s3s = norm(S3), s4s = norm(S4);
        complex S2S3c = S2 * conj(S3), S1S4c = S1 * conj(S4);
        complex S2S4c = S2 * conj(S4), S1S3c = S1 * conj(S3);
        complex S1S2c = S1 * conj(S2), S3S4c = S3 * conj(S4);

        double *row = &muellerFlat[it * 17];
        row[0] = thetaDeg;
        row[1]  = 0.5 * (s1s + s2s + s3s + s4s);              // S11
        row[2]  = 0.5 * (s2s + s4s - s1s - s3s);              // S12
        row[3]  = real(S2S3c) + real(S1S4c);                   // S13
        row[4]  = imag(S1S4c) - imag(S2S3c);                   // S14
        row[5]  = 0.5 * (s2s + s3s - s1s - s4s);              // S21
        row[6]  = 0.5 * (s2s + s1s - s3s - s4s);              // S22
        row[7]  = real(S2S3c) - real(S1S4c);                   // S23
        row[8]  = -(imag(S2S3c) + imag(S1S4c));                // S24
        row[9]  = real(S2S4c) + real(S1S3c);                   // S31
        row[10] = real(S2S4c) - real(S1S3c);                   // S32
        row[11] = real(S1S2c) + real(S3S4c);                   // S33
        row[12] = imag(S1S2c) + imag(S3S4c);                   // S34
        row[13] = imag(S2S4c) - imag(S1S3c);                   // S41
        row[14] = imag(S2S4c) + imag(S1S3c);                   // S42
        row[15] = imag(S3S4c) - imag(S1S2c);                   // S43
        row[16] = real(S1S2c) - real(S3S4c);                   // S44
    }

    // Fraunhofer diffraction from particle shadow (Babinet principle)
    if (incidentDir)
    {
        double Cgeo = 0;
        Point3f incDir = *incidentDir;
        for (int f = 0; f < m_particle->nFacets; ++f)
        {
            const Facet &facet = m_particle->facets[f];
            Point3f en = facet.ex_normal;
            double cosI = -DotProduct(en, incDir) / Length(en);
            if (cosI > 0)
                Cgeo += facet.Area() * cosI;
        }
        if (Cgeo_out) *Cgeo_out = Cgeo;

        double a = sqrt(Cgeo / M_PI);
        double x = k * a;
        double dThetaRad = dTheta * M_PI / 180.0;

        for (int it = 0; it < nTheta; ++it)
        {
            double thetaDeg = thetaMinDeg + it * dTheta;
            double theta = thetaDeg * M_PI / 180.0;

            double th1 = max(0.0, theta - dThetaRad / 2.0);
            double th2 = min(M_PI, theta + dThetaRad / 2.0);

            double Sdiff2 = 0;
            const int nSub = 16;
            double wSum = 0;
            for (int is = 0; is < nSub; ++is)
            {
                double thSub = th1 + (th2 - th1) * (is + 0.5) / nSub;
                double sinSub = sin(thSub);
                double obliq = (1.0 + cos(thSub)) / 2.0;
                double uSub = x * sinSub;
                double jincSub = (uSub < 1e-10) ? 1.0 : 2.0 * j1(uSub) / uSub;
                double Sd = x * x / 2.0 * obliq * jincSub;
                Sdiff2 += Sd * Sd * sinSub;
                wSum += sinSub;
            }
            if (wSum > 0) Sdiff2 /= wSum;

            double *row = &muellerFlat[it * 17];
            row[1]  += Sdiff2;   // S11
            row[6]  += Sdiff2;   // S22
            row[11] += Sdiff2;   // S33
            row[16] += Sdiff2;   // S44
        }
    }

    // Cext via optical theorem (forward scattering, separate sum for precision)
    {
        complex sumYy(0,0), sumXx(0,0);
        double eta_fwd = sinc_f(kd / 2.0);  // n_fwd = (0,0,-1)
        complex F_fwd = F_base * eta_fwd;

        for (size_t j = 0; j < m_dipoles.size(); ++j)
        {
            const DipoleField &dip = m_dipoles[j];
            if (dip.nContributions <= 0 && dip.assignedFacet < 0)
                continue;
            complex phase = exp_im(k * dip.z);  // -k * dot(r, (0,0,-1)) = k*z
            sumYy += dip.Ey_Y * phase;
            sumXx += dip.Ex_X * phase;
        }
        complex S1_fwd = F_fwd * sumYy;
        complex S2_fwd = F_fwd * (-sumXx);

        CextY = (4.0 * M_PI / (k * k)) * real(S1_fwd);
        CextX = (4.0 * M_PI / (k * k)) * real(S2_fwd);
    }
}

void ADDAFieldComputer::ComputeFarField(const Point3f &incidentDir,
                                         double thetaMinDeg, double thetaMaxDeg,
                                         int nTheta,
                                         const string &filename) const
{
    if (m_dipoles.empty())
    {
        cerr << "ERROR: No dipoles for far-field computation" << endl;
        return;
    }

    double d = m_gridSpacing;
    double k = m_k;
    double kd = k * d;

    complex m2 = m_ri * m_ri;
    complex chi_V = (d * d * d / (4.0 * M_PI)) * (m2 - 1.0);
    double eta2 = 1.0 - (kd * kd) / 24.0;

    cout << "\n=== Far-field from GO dipole fields ===" << endl;
    cout << "  Dipoles: " << m_dipoles.size() << endl;
    cout << "  d = " << d << " um, k = " << k << " um^-1" << endl;
    cout << "  chi*V = " << real(chi_V) << " + i*" << imag(chi_V) << endl;
    cout << "  kd = " << kd << ", eta2 (IGT sinc) = " << eta2 << endl;
    cout << "  |F| = " << abs(complex(0.0, -1.0) * (k*k*k) * chi_V * eta2) << endl;

    int nActive = 0;
    for (size_t j = 0; j < m_dipoles.size(); ++j)
        if (m_dipoles[j].nContributions > 0 || m_dipoles[j].assignedFacet >= 0)
            ++nActive;
    cout << "  Active dipoles (with GO field): " << nActive << " / " << m_dipoles.size() << endl;

    // Core computation
    vector<double> mueller;
    double CextY, CextX, Cgeo;
    ComputeFarFieldCore(thetaMinDeg, thetaMaxDeg, nTheta, mueller, CextY, CextX,
                        &incidentDir, &Cgeo);

    // Write output file
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "ERROR: Cannot open " << filename << " for writing" << endl;
        return;
    }
    file << scientific << setprecision(10);
    for (int it = 0; it < nTheta; ++it)
    {
        const double *row = &mueller[it * 17];
        file << row[0];
        for (int j = 1; j < 17; ++j)
            file << " " << row[j];
        file << endl;
    }
    file.close();
    cout << "Wrote GO far-field Mueller matrix: " << filename << " (" << nTheta << " angles)" << endl;

    cout << "  Cext_Y (perp) = " << CextY << " um^2" << endl;
    cout << "  Cext_X (par)  = " << CextX << " um^2" << endl;
    cout << "  Cext_unpol    = " << 0.5 * (CextY + CextX) << " um^2" << endl;

    cout << "  Cgeo = " << Cgeo << " um^2" << endl;
    cout << "  Shadow radius a = " << sqrt(Cgeo / M_PI) << " um, x = ka = "
         << m_k * sqrt(Cgeo / M_PI) << endl;

    double Dmax = m_particle->MaximalDimention();
    double Csca_geom = M_PI * (Dmax / 2.0) * (Dmax / 2.0);
    cout << "  Qext_Y = " << CextY / Csca_geom << endl;
    cout << "  Qext_X = " << CextX / Csca_geom << endl;
}

void ADDAFieldComputer::ComputePGOHCore(std::vector<Beam> &outBeams,
                                         const Light &incidentLight,
                                         double thetaMinDeg, double thetaMaxDeg,
                                         int nTheta,
                                         std::vector<double> &muellerFlat,
                                         double &Cgeo) const
{
    double k = m_k;
    double dTheta = (nTheta > 1) ? (thetaMaxDeg - thetaMinDeg) / (nTheta - 1) : 0;
    double dThetaRad = dTheta * M_PI / 180.0;

    // Bins: each stores 16 Mueller elements weighted by beam cross-section
    struct MuellerBin {
        double s[16];
        MuellerBin() { for (int i = 0; i < 16; ++i) s[i] = 0; }
    };
    vector<MuellerBin> bins(nTheta);

    Point3f incDir = incidentLight.direction;
    Point3f polBasis = incidentLight.polarizationBasis;

    for (size_t i = 0; i < outBeams.size(); ++i)
    {
        Beam &beam = outBeams[i];

        // Beam cross-section (Handler::BeamCrossSection reimplementation)
        Point3f normal = m_particle->facets[beam.lastFacetId].ex_normal;
        double cosA = DotProduct(normal, beam.direction);
        double e = fabs(cosA);
        if (e < 1e7 * DBL_EPSILON) continue;
        double sigma = (e * beam.Area()) / Length(normal);
        if (sigma < 1e-20) continue;

        // Scattering angle theta
        double cosTheta = DotProduct(beam.direction, incDir)
                          / (Length(beam.direction) * Length(incDir));
        if (cosTheta > 1.0) cosTheta = 1.0;
        if (cosTheta < -1.0) cosTheta = -1.0;
        double thetaDeg = acos(cosTheta) * 180.0 / M_PI;

        // Rotate Jones matrix to spherical coordinate frame
        beam.RotateSpherical(-incDir, polBasis);

        // Jones -> Mueller (4x4)
        matrix m = Mueller(beam.J);

        // Rotate Mueller to reference scattering plane (xz plane)
        double bx = beam.direction.cx;
        double by = beam.direction.cy;
        double r2 = bx * bx + by * by;
        if (r2 > FLT_EPSILON)
        {
            double phi = atan2(by, bx);
            double rot = -2.0 * phi;
            RightRotateMueller(m, cos(rot), sin(rot));
        }

        // Beam diffraction spreading: delta_theta = lambda / sqrt(sigma)
        // Spread each beam across bins with Gaussian kernel
        double dthBeam = m_wavelength / sqrt(sigma);  // radians
        double sigBins = (dthBeam * 180.0 / M_PI) / dTheta; // in bin units
        if (sigBins < 0.5) sigBins = 0.5;  // minimum: half-bin
        int hw = (int)ceil(3.0 * sigBins);  // half-width: ±3 sigma

        double binCenter = (dTheta > 0) ? (thetaDeg - thetaMinDeg) / dTheta : 0;
        int iBinMin = max(0, (int)floor(binCenter) - hw);
        int iBinMax = min(nTheta - 1, (int)ceil(binCenter) + hw);

        double wSum = 0;
        for (int ib = iBinMin; ib <= iBinMax; ++ib)
        {
            double dist = ib - binCenter;
            wSum += exp(-0.5 * dist * dist / (sigBins * sigBins));
        }
        if (wSum < 1e-30) continue;

        for (int ib = iBinMin; ib <= iBinMax; ++ib)
        {
            double dist = ib - binCenter;
            double wt = exp(-0.5 * dist * dist / (sigBins * sigBins)) / wSum;
            for (int j = 0; j < 4; ++j)
                for (int l = 0; l < 4; ++l)
                    bins[ib].s[j * 4 + l] += m[j][l] * sigma * wt;
        }
    }

    // Geometric cross-section for diffraction (Babinet principle)
    Cgeo = 0;
    for (int f = 0; f < m_particle->nFacets; ++f)
    {
        const Facet &facet = m_particle->facets[f];
        Point3f en = facet.ex_normal;
        double cosI = -DotProduct(en, incDir) / Length(en);
        if (cosI > 0)
            Cgeo += facet.Area() * cosI;
    }

    double a = sqrt(Cgeo / M_PI);
    double x = k * a;

    // Fill muellerFlat: nTheta rows of 17 values {theta, S11..S44}
    muellerFlat.resize(nTheta * 17);
    for (int it = 0; it < nTheta; ++it)
    {
        double thetaDeg = thetaMinDeg + it * dTheta;
        double theta = thetaDeg * M_PI / 180.0;
        muellerFlat[it * 17] = thetaDeg;

        double th1 = max(0.0, theta - dThetaRad / 2.0);
        double th2 = min(M_PI, theta + dThetaRad / 2.0);
        double dOmega = 2.0 * M_PI * fabs(cos(th1) - cos(th2));
        if (dOmega < 1e-30) dOmega = 1e-30;

        for (int j = 0; j < 16; ++j)
            muellerFlat[it * 17 + 1 + j] = k * k * bins[it].s[j] / dOmega;

        // Fraunhofer diffraction averaged over bin solid angle
        double Sdiff2 = 0;
        {
            const int nSub = 16;
            double wSum = 0;
            for (int is = 0; is < nSub; ++is)
            {
                double thSub = th1 + (th2 - th1) * (is + 0.5) / nSub;
                double sinSub = sin(thSub);
                double obliq = (1.0 + cos(thSub)) / 2.0;
                double uSub = x * sinSub;
                double jincSub = (uSub < 1e-10) ? 1.0 : 2.0 * j1(uSub) / uSub;
                double Sd = x * x / 2.0 * obliq * jincSub;
                Sdiff2 += Sd * Sd * sinSub;
                wSum += sinSub;
            }
            if (wSum > 0) Sdiff2 /= wSum;
        }

        muellerFlat[it * 17 + 1]  += Sdiff2;   // S11
        muellerFlat[it * 17 + 6]  += Sdiff2;   // S22
        muellerFlat[it * 17 + 11] += Sdiff2;   // S33
        muellerFlat[it * 17 + 16] += Sdiff2;   // S44
    }
}

void ADDAFieldComputer::ComputePGOH(std::vector<Beam> &outBeams,
                                     const Light &incidentLight,
                                     double thetaMinDeg, double thetaMaxDeg,
                                     int nTheta,
                                     const string &filename) const
{
    vector<double> muellerFlat;
    double Cgeo;
    ComputePGOHCore(outBeams, incidentLight, thetaMinDeg, thetaMaxDeg,
                    nTheta, muellerFlat, Cgeo);

    // Write output file
    ofstream file(filename);
    if (!file.is_open())
    {
        cerr << "ERROR: Cannot open " << filename << " for writing" << endl;
        return;
    }

    file << scientific << setprecision(10);
    for (int it = 0; it < nTheta; ++it)
    {
        const double *row = &muellerFlat[it * 17];
        file << row[0];
        for (int j = 1; j < 17; ++j)
            file << " " << row[j];
        file << endl;
    }
    file.close();

    // Summary
    double Cext = 2.0 * Cgeo;

    cout << "\n=== PGOH Far-Field ===" << endl;
    cout << "  Cgeo = " << Cgeo << " um^2" << endl;
    cout << "  Cext = 2*Cgeo = " << Cext << " um^2 (extinction paradox)" << endl;
    cout << "  Shadow radius a = " << sqrt(Cgeo / M_PI) << " um, x = ka = "
         << m_k * sqrt(Cgeo / M_PI) << endl;
    cout << "Wrote PGOH Mueller matrix: " << filename
         << " (" << nTheta << " angles)" << endl;
}
