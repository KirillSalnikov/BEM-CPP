#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>
#include <float.h>
#include <string>

#include "HexagonalAggregate.h"
#include "ConcaveHexagonal.h"
#include "CertainAggregate.h"
#include "Bullet.h"
#include "BulletRosette.h"
#include "global.h"
#include "TracerGO.h"
#include "ArgPP.h"
#include "Tracks.h"
#include "HandlerTotalGO.h"
#include "HandlerTracksGO.h"
#include "Droxtal.h"
#include "ADDAField.h"

#ifdef _OUTPUT_NRG_CONV
ofstream energyFile("energy.dat", ios::out);
double SSconfined=0;
int bcount=0;
#endif

using namespace std;

/// Gauss-Legendre quadrature nodes and weights on [x1, x2]
static void gaussLegendre(int n, double x1, double x2,
                           vector<double> &x, vector<double> &w)
{
    x.resize(n);
    w.resize(n);
    int m = (n + 1) / 2;
    double xm = 0.5 * (x2 + x1);
    double xl = 0.5 * (x2 - x1);
    for (int i = 0; i < m; i++)
    {
        double z = cos(M_PI * (i + 0.75) / (n + 0.5));
        double z1, pp;
        do {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < n; j++)
            {
                double p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1);
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            z1 = z;
            z -= p1 / pp;
        } while (fabs(z - z1) > 1e-14);
        x[i] = xm - xl * z;
        x[n - 1 - i] = xm + xl * z;
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
}

/// Cube particle: 6 square facets, edge length specified
class CubeParticle : public Particle
{
public:
    CubeParticle(const complex &refrIndex, double edge)
    {
        isConcave = false;
        double radius = edge / sqrt(2.0);
        double halfH = edge / 2.0;
        Init(6, refrIndex);
        SetSymmetry(M_PI/2, M_PI/2);

        // Set all facets to 4 vertices
        for (int i = 0; i < 6; ++i)
            defaultFacets[i].nVertices = 4;

        Facet top = CreateBase(4, radius, halfH);
        Facet bot = CreateBase(4, radius, -halfH);
        defaultFacets[0] = top;
        defaultFacets[5] = bot;
        int iFacet = 0;
        CreateSideFacets(top, bot, iFacet);

        SetDefaultNormals();
        Reset();
        SetDefaultCenters();
    }
};

/// Sphere particle: UV-sphere tessellation
/// Usage: -p 6 D [nLat [nLon]]  (diameter, latitude divisions, longitude divisions)
class SphereParticle : public Particle
{
public:
    SphereParticle(const complex &refrIndex, double diameter,
                   int nLat = 10, int nLon = 20)
    {
        isConcave = false;
        double R = diameter / 2.0;

        int totalFacets = 2 * nLon + (nLat - 2) * nLon;
        if (totalFacets > MAX_FACET_NUM)
        {
            cerr << "ERROR: Sphere needs " << totalFacets
                 << " facets, max is " << MAX_FACET_NUM << endl;
            exit(1);
        }

        Init(totalFacets, refrIndex);
        SetSymmetry(M_PI, M_PI);

        // Latitude and longitude angles
        std::vector<double> theta(nLat + 1), phi(nLon + 1);
        for (int i = 0; i <= nLat; ++i)
            theta[i] = M_PI * i / nLat;
        for (int j = 0; j <= nLon; ++j)
            phi[j] = 2.0 * M_PI * j / nLon;

        auto sp = [R](double th, double ph) -> Point3f {
            return Point3f((float)(R * sin(th) * cos(ph)),
                           (float)(R * sin(th) * sin(ph)),
                           (float)(R * cos(th)));
        };

        int fi = 0;
        Point3f northPole(0, 0, (float)R);
        Point3f southPole(0, 0, (float)(-R));

        // Top cap triangles (CCW from outside → outward normal)
        for (int j = 0; j < nLon; ++j)
        {
            defaultFacets[fi].nVertices = 3;
            defaultFacets[fi].arr[0] = northPole;
            defaultFacets[fi].arr[1] = sp(theta[1], phi[j]);
            defaultFacets[fi].arr[2] = sp(theta[1], phi[j + 1]);
            ++fi;
        }

        // Middle quads: (θi,φj) → (θi+1,φj) → (θi+1,φj+1) → (θi,φj+1)
        for (int i = 1; i < nLat - 1; ++i)
        {
            for (int j = 0; j < nLon; ++j)
            {
                defaultFacets[fi].nVertices = 4;
                defaultFacets[fi].arr[0] = sp(theta[i],   phi[j]);
                defaultFacets[fi].arr[1] = sp(theta[i+1], phi[j]);
                defaultFacets[fi].arr[2] = sp(theta[i+1], phi[j+1]);
                defaultFacets[fi].arr[3] = sp(theta[i],   phi[j+1]);
                ++fi;
            }
        }

        // Bottom cap triangles (CCW from outside → outward normal)
        for (int j = 0; j < nLon; ++j)
        {
            defaultFacets[fi].nVertices = 3;
            defaultFacets[fi].arr[0] = southPole;
            defaultFacets[fi].arr[1] = sp(theta[nLat - 1], phi[j + 1]);
            defaultFacets[fi].arr[2] = sp(theta[nLat - 1], phi[j]);
            ++fi;
        }

        SetDefaultNormals();
        Reset();
        SetDefaultCenters();
    }
};

enum class ParticleType : int
{
    Hexagonal = 1,
    Bullet = 2,
    BulletRosette = 3,
    Droxtal = 4,
    Cube = 5,
    Sphere = 6,
    ConcaveHexagonal = 10,
    TiltedHexagonal = 11,
    HexagonalAggregate = 12,
    CertainAggregate = 999
};

Tracks trackGroups;

void SetArgRules(ArgPP &parser)
{
    int zero = 0;
    parser.AddRule("p", '+'); // particle (type, size, ...)
    parser.AddRule("ri", 2); // refractive index (Re and Im parts)
    parser.AddRule("n", 1); // number of internal reflection
    parser.AddRule("pf", zero, true); // particle (filename)
    parser.AddRule("rs", 1, true, "pf"); // resize particle (new size)
    parser.AddRule("fixed", '+', true); // fixed orientation (beta, gamma [, alpha])
    parser.AddRule("w", 1, true); // wavelength
    parser.AddRule("grid", '+', true); /* backscattering grid:
 (radius, Nphi, Ntheta) when 3 parameters
 (theta1, theta2, Nphi, Ntheta) when 4 parameters*/
    parser.AddRule("tr", 1, true); // file with trajectories
    parser.AddRule("all", 0, true); // calculate all trajectories
    parser.AddRule("o", 1, true); // output folder name


    parser.AddRule("forced_nonconvex", zero, true);
    parser.AddRule("forced_convex", zero, true);
    parser.AddRule("r", 1, true); // restriction ratio for small beams when intersection (100 by default)
    parser.AddRule("adda", 0, true); // ADDA mode: compute internal field and output for ADDA
    parser.AddRule("dpl", 1, true); // dipoles per wavelength for ADDA grid (default 10)
    parser.AddRule("norefl", 0, true); // skip all reflections in ADDA mode
    parser.AddRule("nokirch", 0, true); // reflections without Kirchhoff diffraction
    parser.AddRule("maxacts", 1, true); // max nActs for reflected beam segments (default: 1)
    parser.AddRule("goi", 0, true); // incoherent reflected beam accumulation (intensity sum)
    parser.AddRule("go", 0, true); // use GO-traced reflected beams instead of analytical PW reflection
    parser.AddRule("noinit", 0, true); // skip field files, output only shape (for x_0=0 start)
    parser.AddRule("gref_dot", 1, true); // n̂_A·n̂_B threshold for reflection pairs (default -0.9)
    parser.AddRule("ff", 0, true); // compute far-field from GO dipole fields (no ADDA iteration)
    parser.AddRule("pgoh", 0, true); // PGOH far-field: GO beam Mueller matrices + Fraunhofer diffraction
    parser.AddRule("blend", 1, true); // Fresnel edge blending: --blend sigma (0 = auto: 0.5)
    parser.AddRule("sfp", 0, true); // smooth Fabry-Perot: multiply field by 1/(1-R^2*exp(2ikgd))
    parser.AddRule("mg", 2, true); // multigrid: --mg IntField-Y IntField-X (coarse ADDA field)
    parser.AddRule("avg", 3, true); // orientation averaging: --avg Nbeta Ngamma Nalpha
    parser.AddRule("nf_orders", 0, true); // write per-order field files (k0, k1, ...)
}

ScatteringRange SetConus(ArgPP &parser)
{
    if (parser.GetArgNumber("grid") == 3)
    {
        double radius = parser.GetDoubleValue("grid", 0);
        int nAz = parser.GetDoubleValue("grid", 1);
        int nZen = parser.GetDoubleValue("grid", 2);
        return ScatteringRange(M_PI - DegToRad(radius), M_PI, nAz, nZen);
    }
    else if (parser.GetArgNumber("grid") == 4)
    {
        double zenStart = parser.GetDoubleValue("grid", 0);
        double zenEnd = parser.GetDoubleValue("grid", 1);

        if (zenStart < zenEnd)
        {
            int nAz = parser.GetDoubleValue("grid", 2);
            int nZen = parser.GetDoubleValue("grid", 3);
            return ScatteringRange(DegToRad(zenStart), DegToRad(zenEnd), nAz, nZen);
        }
        else
        {
            std::cerr << "ERROR!!!!!!! In \"grid\" arg 1 must be less than arg 2";
            exit(1);
        }
    }
    else
    {
        std::cerr << "ERROR!!!!!!! Wrong \"grid\" argument number";
        exit(1);
    }
}

int main(int argc, const char* argv[])
{
    RenameConsole("MBS");

    std::string additionalSummary;

    if (argc <= 1) // no arguments
    {
        cout << "No arguments. Press any key to exit..." << endl;
        getchar();
        return 1;
    }

    ArgPP args;
    SetArgRules(args);
    args.Parse(argc, argv);

    double re = args.GetDoubleValue("ri", 0);
    double im = args.GetDoubleValue("ri", 1);
    complex refrIndex = complex(re, im);

    Particle *particle = nullptr;


    additionalSummary += "Particle: ";

    if (args.IsCatched("pf"))
    {
        std::string filename = args.GetStringValue("p");
        particle = new Particle();
        particle->SetFromFile(filename);
        particle->SetRefractiveIndex(complex(refrIndex));

        double origDMax = particle->MaximalDimention();
        additionalSummary += "from file: " + filename + "\n";
        additionalSummary += "\tOriginal Dmax: " + std::to_string(origDMax);

        if (args.IsCatched("rs"))
        {
            double newSize = args.GetDoubleValue("rs", 0);
            particle->Resize(newSize);
        }

        double newDMax = particle->MaximalDimention();
        additionalSummary += ", new Dmax: " + std::to_string(newDMax)
                + ", resize factor: " + std::to_string(newDMax/origDMax) + '\n';
    }
    else
    {
        ParticleType type = (ParticleType)args.GetIntValue("p", 0);
        double height;
        double diameter;

        double sup;
        int num;

        switch (type)
        {
        case ParticleType::Hexagonal:
            height = args.GetDoubleValue("p", 1);
            diameter = args.GetDoubleValue("p", 2);
            particle = new Hexagonal(refrIndex, diameter, height);
            break;
        case ParticleType::Bullet:
            height = args.GetDoubleValue("p", 1);
            diameter = args.GetDoubleValue("p", 2);
            sup = (diameter*sqrt(3)*tan(DegToRad(62)))/4;
            particle = new Bullet(refrIndex, diameter, height, sup);
            break;
        case ParticleType::BulletRosette:
            height = args.GetDoubleValue("p", 1);
            diameter = args.GetDoubleValue("p", 2);
            sup = (args.GetArgNumber("p") == 4)
                    ? args.GetDoubleValue("p", 3)
                    : diameter*sqrt(3)*tan(DegToRad(62))/4;
            particle = new BulletRosette(refrIndex, diameter, height, sup);
            break;
        case ParticleType::Droxtal:
            sup = args.GetDoubleValue("p", 3);
            particle = new Droxtal(refrIndex, DegToRad(32.35), DegToRad(71.81), sup);
            break;
        case ParticleType::Cube:
            height = args.GetDoubleValue("p", 1); // edge length
            particle = new CubeParticle(refrIndex, height);
            break;
        case ParticleType::Sphere:
        {
            diameter = args.GetDoubleValue("p", 1); // diameter
            int nla = (args.GetArgNumber("p") >= 3) ? args.GetIntValue("p", 2) : 10;
            int nlo = (args.GetArgNumber("p") >= 4) ? args.GetIntValue("p", 3) : 20;
            particle = new SphereParticle(refrIndex, diameter, nla, nlo);
            break;
        }
        case ParticleType::ConcaveHexagonal:
            height = args.GetDoubleValue("p", 1);
            diameter = args.GetDoubleValue("p", 2);
            sup = args.GetDoubleValue("p", 3);
            particle = new ConcaveHexagonal(refrIndex, diameter, height, sup);
            break;
        case ParticleType::HexagonalAggregate:
            height = args.GetDoubleValue("p", 1);
            diameter = args.GetDoubleValue("p", 2);
            num = args.GetIntValue("p", 3);
            particle = new HexagonalAggregate(refrIndex, diameter, height, num);
            break;
        case ParticleType::CertainAggregate:
            sup = args.GetDoubleValue("p", 1);
            particle = new CertainAggregate(refrIndex, sup);
            break;
        default:
            assert(false && "ERROR! Incorrect type of particle.");
            break;
        }
    }

    additionalSummary += "\tRefractive index: " + to_string(re);

    if (fabs(im) > FLT_EPSILON)
    {
        additionalSummary +=  " + i\n";
    }

    additionalSummary += "\n";

    particle->Output("particle_for_check.dat");
    additionalSummary += "\tArea:" + to_string(particle->Area()) + "\n\n";

    int reflNum = args.GetDoubleValue("n");
    additionalSummary += "Number of secondary reflections: " + to_string(reflNum) + "\n";

    string dirName = (args.IsCatched("o")) ? args.GetStringValue("o")
                                           : "M";
    size_t pos = 0;

    while ((pos = dirName.find('%', pos)) != string::npos)
    {
        size_t start = pos;
        ++pos;
        string key;

        while (dirName[pos] != '_' && pos < dirName.size())
        {
            key += dirName[pos++];
        }

        if (key.size())
        {
            string val = args.GetStringValue(key);
            dirName.replace(start, pos-start, val);
            pos = start + val.size();
        }
    }

    double wave = args.IsCatched("w") ? args.GetDoubleValue("w") : 0;
    additionalSummary += "Wavelength (um): " + to_string(wave) + "\n";

    if (args.IsCatched("forced_nonconvex"))
    {
        particle->isConcave = true;
    }

    if (args.IsCatched("forced_convex"))
    {
        particle->isConcave = false;
    }

    if (args.IsCatched("tr"))
    {
        string trackFileName = args.GetStringValue("tr");
        trackGroups.ImportTracks(particle->nFacets, trackFileName);
        trackGroups.shouldComputeTracksOnly = !args.IsCatched("all");
    }

    int nTheta = args.GetDoubleValue("grid", 2);

    additionalSummary += "Method: Geometrical optics";

    TracerGO tracer(particle, reflNum, dirName);
    tracer.m_scattering->m_wave = wave;
    if (args.IsCatched("r"))
    {
        tracer.m_scattering->restriction = args.GetDoubleValue("r", 0);
    }
    tracer.m_summary = additionalSummary;
    HandlerGO *handler;

    if (args.IsCatched("tr"))
    {
        handler = new HandlerTracksGO(particle, &tracer.m_incidentLight, nTheta, wave);
        handler->SetTracks(&trackGroups);
    }
    else
    {
        handler = new HandlerTotalGO(particle, &tracer.m_incidentLight, nTheta, wave);
    }

    tracer.m_summary = additionalSummary;
    ScatteringRange grid = SetConus(args);
    handler->SetScatteringSphere(grid);
    handler->SetAbsorptionAccounting(im != 0);
    tracer.SetHandler(handler);

    bool addaMode = args.IsCatched("adda");

    if (addaMode && (!args.IsCatched("w") || wave < 1e-15))
    {
        cerr << "ERROR: -adda requires -w (wavelength)" << endl;
        exit(1);
    }

    if (args.IsCatched("fixed"))
    {
        additionalSummary += ", fixed orientation, ";
        double beta  = args.GetDoubleValue("fixed", 0);
        double gamma = args.GetDoubleValue("fixed", 1);
        double alpha = (args.GetArgNumber("fixed") >= 3) ? args.GetDoubleValue("fixed", 2) : 0.0;
        additionalSummary += "zenith " + to_string(beta) + "\xC2\xB0, azimuth " + to_string(gamma) + "\xC2\xB0, alpha " + to_string(alpha) + "\xC2\xB0" + "\n\n";
        cout << additionalSummary;
        tracer.m_summary = additionalSummary;

        if (addaMode)
        {
            int dpl = args.IsCatched("dpl") ? args.GetIntValue("dpl") : 10;

            // Common ADDA flags
            double blendSigma = 0.0;
            if (args.IsCatched("blend"))
            {
                blendSigma = args.GetDoubleValue("blend");
                if (blendSigma < 1e-15) blendSigma = 0.5; // --blend 0 -> auto
            }

            if (args.IsCatched("avg") && args.IsCatched("ff"))
            {
                // === Orientation-averaged far-field ===
                int nBeta  = (int)args.GetDoubleValue("avg", 0);
                int nGamma = (int)args.GetDoubleValue("avg", 1);
                int nAlpha = (int)args.GetDoubleValue("avg", 2);
                if (nAlpha < 1) nAlpha = 1;

                double thMin = args.GetDoubleValue("grid", 0);
                double thMax = args.GetDoubleValue("grid", 1);
                int nTh = (int)args.GetDoubleValue("grid", 3);

                cout << "\n=== Orientation-averaged far-field ===" << endl;
                cout << "  Quadrature: " << nBeta << " x " << nGamma << " x " << nAlpha
                     << " (beta x gamma x alpha) = " << nBeta * nGamma * nAlpha << " orientations" << endl;

                // Gauss-Legendre for cos(beta) in [-1, 1]
                vector<double> cosB, wB;
                gaussLegendre(nBeta, -1.0, 1.0, cosB, wB);

                bool doPGOH = args.IsCatched("pgoh");

                // Accumulation buffers
                vector<double> accumMueller(nTh * 17, 0.0);
                double accumCextY = 0, accumCextX = 0;
                vector<double> accumPGOH(nTh * 17, 0.0);
                double accumCgeo = 0;
                double totalWeight = 0;
                int iOri = 0;
                int totalOri = nBeta * nGamma * nAlpha;

                for (int ib = 0; ib < nBeta; ++ib)
                {
                    double betaRad = acos(cosB[ib]);
                    for (int ig = 0; ig < nGamma; ++ig)
                    {
                        double gammaRad = 2.0 * M_PI * ig / nGamma;
                        for (int ia = 0; ia < nAlpha; ++ia)
                        {
                            double alphaRad = 2.0 * M_PI * ia / nAlpha;
                            double w = wB[ib] / nGamma / nAlpha;

                            particle->Rotate(betaRad, gammaRad, alphaRad);

                            ADDAFieldComputer addaField(particle, wave, refrIndex, dpl);
                            if (args.IsCatched("nokirch"))
                                addaField.m_noKirchhoff = true;
                            addaField.BuildDipoleGrid();

                            std::vector<InternalBeamSegment> segments;
                            tracer.m_scattering->SetInternalSegmentStorage(&segments);

                            vector<Beam> outBeams;
                            tracer.m_scattering->ScatterLight(betaRad, gammaRad, outBeams);
                            if (!doPGOH) outBeams.clear();

                            if (args.IsCatched("mg"))
                                addaField.InterpolateCoarseField(args.GetStringValue("mg", 0),
                                                                  args.GetStringValue("mg", 1),
                                                                  &tracer.m_incidentLight.direction);

                            addaField.FillUncoveredPerFacet(tracer.m_incidentLight.direction,
                                                             blendSigma);

                            if (args.IsCatched("sfp"))
                                addaField.AddSmoothFP(tracer.m_incidentLight.direction);

                            if (!args.IsCatched("norefl") && !args.IsCatched("go"))
                            {
                                double grefDot = args.IsCatched("gref_dot") ? args.GetDoubleValue("gref_dot") : -0.9;
                                addaField.AddGeneralReflection(tracer.m_incidentLight.direction, grefDot);
                            }

                            tracer.m_scattering->SetInternalSegmentStorage(nullptr);

                            vector<double> mueller;
                            double cY, cX, cGeoFF;
                            Point3f ffIncDir = tracer.m_incidentLight.direction;
                            addaField.ComputeFarFieldCore(thMin, thMax, nTh, mueller, cY, cX,
                                                          &ffIncDir, &cGeoFF);

                            for (int i = 0; i < nTh * 17; ++i)
                            {
                                if (i % 17 == 0)
                                    accumMueller[i] = mueller[i]; // theta (same for all)
                                else
                                    accumMueller[i] += w * mueller[i];
                            }
                            accumCextY += w * cY;
                            accumCextX += w * cX;

                            if (doPGOH)
                            {
                                vector<double> pgohMueller;
                                double cgeo;
                                addaField.ComputePGOHCore(outBeams, tracer.m_incidentLight,
                                                          thMin, thMax, nTh, pgohMueller, cgeo);
                                for (int i = 0; i < nTh * 17; ++i)
                                {
                                    if (i % 17 == 0)
                                        accumPGOH[i] = pgohMueller[i];
                                    else
                                        accumPGOH[i] += w * pgohMueller[i];
                                }
                                accumCgeo += w * cgeo;
                            }

                            totalWeight += w;

                            ++iOri;
                            if (iOri % 10 == 0 || iOri == totalOri)
                                cout << "  " << iOri << " / " << totalOri << " orientations done\r" << flush;
                        }
                    }
                }
                cout << endl;

                // Normalize
                for (int i = 0; i < nTh * 17; ++i)
                    if (i % 17 != 0)
                        accumMueller[i] /= totalWeight;
                accumCextY /= totalWeight;
                accumCextX /= totalWeight;

                // Write averaged Mueller matrix
                string avgFile = dirName + "_mueller_avg";
                ofstream file(avgFile);
                file << scientific << setprecision(10);
                for (int it = 0; it < nTh; ++it)
                {
                    const double *row = &accumMueller[it * 17];
                    file << row[0];
                    for (int j = 1; j < 17; ++j)
                        file << " " << row[j];
                    file << endl;
                }
                file.close();
                cout << "Wrote orientation-averaged Mueller matrix: " << avgFile
                     << " (" << nTh << " angles, " << totalOri << " orientations)" << endl;

                cout << "  <Cext_Y> = " << accumCextY << " um^2" << endl;
                cout << "  <Cext_X> = " << accumCextX << " um^2" << endl;
                cout << "  <Cext>   = " << 0.5 * (accumCextY + accumCextX) << " um^2" << endl;

                double Dmax = particle->MaximalDimention();
                double Csca_geom = M_PI * (Dmax / 2.0) * (Dmax / 2.0);
                cout << "  <Qext>   = " << 0.5 * (accumCextY + accumCextX) / Csca_geom << endl;

                if (doPGOH)
                {
                    // Normalize PGOH
                    for (int i = 0; i < nTh * 17; ++i)
                        if (i % 17 != 0)
                            accumPGOH[i] /= totalWeight;
                    accumCgeo /= totalWeight;

                    string pgohFile = dirName + "_mueller_pgoh_avg";
                    ofstream pf(pgohFile);
                    pf << scientific << setprecision(10);
                    for (int it = 0; it < nTh; ++it)
                    {
                        const double *row = &accumPGOH[it * 17];
                        pf << row[0];
                        for (int j = 1; j < 17; ++j)
                            pf << " " << row[j];
                        pf << endl;
                    }
                    pf.close();
                    cout << "Wrote orientation-averaged PGOH Mueller: " << pgohFile << endl;
                    cout << "  <Cgeo> = " << accumCgeo << " um^2" << endl;
                    cout << "  <Cext_pgoh> = 2*<Cgeo> = " << 2.0 * accumCgeo << " um^2" << endl;
                }
            }
            else
            {
            // === Single-orientation ADDA mode ===
            double b = DegToRad(beta);
            double g = DegToRad(gamma);
            double a = DegToRad(alpha);
            particle->Rotate(b, g, a);

            ADDAFieldComputer addaField(particle, wave, refrIndex, dpl);
            if (args.IsCatched("nokirch"))
                addaField.m_noKirchhoff = true;
            addaField.BuildDipoleGrid();

            std::vector<InternalBeamSegment> segments;
            tracer.m_scattering->SetInternalSegmentStorage(&segments);

            vector<Beam> outBeams;
            tracer.m_scattering->ScatterLight(b, g, outBeams);
            if (!args.IsCatched("pgoh"))
                outBeams.clear();

            cout << "Captured " << segments.size() << " internal beam segments" << endl;

            if (args.IsCatched("mg"))
                addaField.InterpolateCoarseField(args.GetStringValue("mg", 0),
                                                  args.GetStringValue("mg", 1),
                                                  &tracer.m_incidentLight.direction);

            addaField.FillUncoveredPerFacet(tracer.m_incidentLight.direction,
                                             blendSigma);

            bool writeOrders = args.IsCatched("nf_orders");
            if (writeOrders)
            {
                addaField.WriteFieldFileY(dirName + "_fieldY_k0.dat");
                addaField.WriteFieldFileX(dirName + "_fieldX_k0.dat");
                cout << "Wrote order-0 field (direct transmission)" << endl;
            }

            if (args.IsCatched("sfp"))
                addaField.AddSmoothFP(tracer.m_incidentLight.direction);

            if (!args.IsCatched("norefl"))
            {
                if (args.IsCatched("go"))
                {
                    bool goIncoh = args.IsCatched("goi");
                    int maxActs = args.IsCatched("maxacts") ? (int)args.GetDoubleValue("maxacts") : 1;
                    if (writeOrders)
                    {
                        for (int k = 1; k <= maxActs; ++k)
                        {
                            addaField.AccumulateReflectedBeams(segments, tracer.m_incidentLight.direction,
                                                               k, goIncoh, k);
                            addaField.WriteFieldFileY(dirName + "_fieldY_k" + to_string(k) + ".dat");
                            addaField.WriteFieldFileX(dirName + "_fieldX_k" + to_string(k) + ".dat");
                            cout << "Wrote cumulative field through order " << k << endl;
                        }
                    }
                    else
                    {
                        addaField.AccumulateReflectedBeams(segments, tracer.m_incidentLight.direction,
                                                           maxActs, goIncoh);
                    }
                }
                else
                {
                    double grefDot = args.IsCatched("gref_dot") ? args.GetDoubleValue("gref_dot") : -0.9;
                    addaField.AddGeneralReflection(tracer.m_incidentLight.direction, grefDot);
                    if (writeOrders)
                    {
                        addaField.WriteFieldFileY(dirName + "_fieldY_k1.dat");
                        addaField.WriteFieldFileX(dirName + "_fieldX_k1.dat");
                        cout << "Wrote cumulative field through order 1 (analytical reflection)" << endl;
                    }
                }
            }

            addaField.DiagnoseGOvsPW(tracer.m_incidentLight.direction);

            bool noInit = args.IsCatched("noinit");
            addaField.WriteGeometryFile(dirName + "_shape.dat");
            if (!noInit)
            {
                addaField.WriteFieldFileY(dirName + "_fieldY.dat");
                addaField.WriteFieldFileX(dirName + "_fieldX.dat");
            }

            if (args.IsCatched("ff"))
            {
                double thMin = args.GetDoubleValue("grid", 0);
                double thMax = args.GetDoubleValue("grid", 1);
                int nTh = (int)args.GetDoubleValue("grid", 3);
                addaField.ComputeFarField(tracer.m_incidentLight.direction,
                                          thMin, thMax, nTh,
                                          dirName + "_mueller_go");
            }

            if (args.IsCatched("pgoh"))
            {
                double thMin = args.GetDoubleValue("grid", 0);
                double thMax = args.GetDoubleValue("grid", 1);
                int nTh = (int)args.GetDoubleValue("grid", 3);
                addaField.ComputePGOH(outBeams, tracer.m_incidentLight,
                                      thMin, thMax, nTh,
                                      dirName + "_mueller_pgoh");
            }

            tracer.m_scattering->SetInternalSegmentStorage(nullptr);

            cout << "\nADDA usage:" << endl;
            cout << "  adda -shape read " << dirName << "_shape.dat"
                 << " -dpl " << dpl
                 << " -m " << re << " " << im
                 << " -prop 0 0 -1";
            if (!noInit)
                cout << " -init_field read " << dirName << "_fieldY.dat"
                     << " " << dirName << "_fieldX.dat";
            cout << endl;
            } // end single-orientation
        }
        else
        {
            tracer.TraceFixed(beta, gamma);
        }
    }

    delete handler;

    cout << endl << "done" << endl;

    return 0;
}
