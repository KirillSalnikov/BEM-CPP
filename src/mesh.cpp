#include "mesh.h"
#include <map>
#include <set>
#include <utility>
#include <tuple>

Mesh icosphere(double radius, int refinements) {
    double phi = (1.0 + sqrt(5.0)) / 2.0;

    // 12 initial vertices of icosahedron (on unit sphere)
    std::vector<Vec3> verts = {
        {-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0},
        {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi},
        {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1},
    };
    // Normalize to unit sphere
    double norm0 = verts[0].norm();
    for (auto& v : verts) { v = v * (1.0 / norm0); }

    // 20 initial triangles
    std::vector<int> tris = {
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    };

    for (int ref = 0; ref < refinements; ref++) {
        std::map<std::pair<int,int>, int> edge_mid;
        std::vector<int> new_tris;

        auto get_mid = [&](int a, int b) -> int {
            auto key = std::make_pair(std::min(a,b), std::max(a,b));
            auto it = edge_mid.find(key);
            if (it != edge_mid.end()) return it->second;
            Vec3 mid = (verts[a] + verts[b]) * 0.5;
            mid = mid.normalized();
            int idx = (int)verts.size();
            verts.push_back(mid);
            edge_mid[key] = idx;
            return idx;
        };

        int ntri = (int)tris.size() / 3;
        for (int i = 0; i < ntri; i++) {
            int a = tris[3*i], b = tris[3*i+1], c = tris[3*i+2];
            int ab = get_mid(a, b);
            int bc = get_mid(b, c);
            int ca = get_mid(c, a);
            // 4 new triangles
            int t[] = {a,ab,ca, b,bc,ab, c,ca,bc, ab,bc,ca};
            new_tris.insert(new_tris.end(), t, t+12);
        }
        tris = new_tris;
    }

    // Scale to desired radius
    for (auto& v : verts) { v = v * radius; }

    Mesh m;
    m.verts = verts;
    m.tris = tris;
    return m;
}

static std::vector<double> uniform_grid(int n)
{
    std::vector<double> x(n + 1);
    for (int i = 0; i <= n; i++)
        x[i] = (double)i / n;
    x.front() = 0.0;
    x.back() = 1.0;
    return x;
}

static double dist_point_segment(const Vec3& p, const Vec3& a, const Vec3& b)
{
    Vec3 ab = b - a;
    double len2 = ab.norm2();
    if (len2 <= 0.0)
        return (p - a).norm();
    double t = (p - a).dot(ab) / len2;
    t = std::max(0.0, std::min(1.0, t));
    Vec3 q = a + ab * t;
    return (p - q).norm();
}

static double tri_quality_min_angle(const Vec3& a, const Vec3& b, const Vec3& c)
{
    double la = (b - c).norm();
    double lb = (c - a).norm();
    double lc = (a - b).norm();
    double eps = 1e-30;
    double ca = std::max(-1.0, std::min(1.0, (lb*lb + lc*lc - la*la) / (2.0 * lb * lc + eps)));
    double cb = std::max(-1.0, std::min(1.0, (lc*lc + la*la - lb*lb) / (2.0 * lc * la + eps)));
    double cc = std::max(-1.0, std::min(1.0, (la*la + lb*lb - lc*lc) / (2.0 * la * lb + eps)));
    return std::min(std::acos(ca), std::min(std::acos(cb), std::acos(cc)));
}

static void refine_marked_edges(Mesh& m, const std::set<std::pair<int,int>>& marked)
{
    std::map<std::pair<int,int>, int> mid_cache;
    auto edge_key = [](int a, int b) {
        return std::make_pair(std::min(a, b), std::max(a, b));
    };
    auto has_edge = [&](int a, int b) {
        return marked.find(edge_key(a, b)) != marked.end();
    };
    auto midpoint = [&](int a, int b) -> int {
        auto key = edge_key(a, b);
        auto it = mid_cache.find(key);
        if (it != mid_cache.end())
            return it->second;
        int id = (int)m.verts.size();
        m.verts.push_back((m.verts[a] + m.verts[b]) * 0.5);
        mid_cache[key] = id;
        return id;
    };

    std::vector<int> out;
    out.reserve(m.tris.size() * 2);
    auto add = [&](int a, int b, int c) {
        out.push_back(a);
        out.push_back(b);
        out.push_back(c);
    };

    int nt = m.nt();
    for (int t = 0; t < nt; t++) {
        int a = m.tris[3*t], b = m.tris[3*t + 1], c = m.tris[3*t + 2];
        bool mab = has_edge(a, b);
        bool mbc = has_edge(b, c);
        bool mca = has_edge(c, a);
        int nmark = (mab ? 1 : 0) + (mbc ? 1 : 0) + (mca ? 1 : 0);
        if (nmark == 0) {
            add(a, b, c);
        } else if (nmark == 3) {
            int ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            add(a, ab, ca);
            add(ab, b, bc);
            add(ca, bc, c);
            add(ab, bc, ca);
        } else if (nmark == 1) {
            if (mab) {
                int ab = midpoint(a, b);
                add(a, ab, c);
                add(ab, b, c);
            } else if (mbc) {
                int bc = midpoint(b, c);
                add(b, bc, a);
                add(bc, c, a);
            } else {
                int ca = midpoint(c, a);
                add(c, ca, b);
                add(ca, a, b);
            }
        } else {
            if (mab && mbc) {
                int ab = midpoint(a, b), bc = midpoint(b, c);
                add(ab, b, bc);
                add(a, ab, bc);
                add(a, bc, c);
            } else if (mbc && mca) {
                int bc = midpoint(b, c), ca = midpoint(c, a);
                add(bc, c, ca);
                add(b, bc, ca);
                add(b, ca, a);
            } else {
                int ca = midpoint(c, a), ab = midpoint(a, b);
                add(ca, a, ab);
                add(c, ca, ab);
                add(c, ab, b);
            }
        }
    }
    m.tris.swap(out);
}

static void refine_prism_edges(Mesh& m, const std::vector<Vec3>& poly,
                               double ztop, double zbot, int seg, int passes)
{
    std::vector<std::pair<Vec3, Vec3>> sharp_edges;
    int sides = (int)poly.size();
    for (int i = 0; i < sides; i++) {
        Vec3 a_top(poly[i].x, poly[i].y, ztop);
        Vec3 b_top(poly[(i + 1) % sides].x, poly[(i + 1) % sides].y, ztop);
        Vec3 a_bot(poly[i].x, poly[i].y, zbot);
        Vec3 b_bot(poly[(i + 1) % sides].x, poly[(i + 1) % sides].y, zbot);
        sharp_edges.push_back(std::make_pair(a_top, b_top));
        sharp_edges.push_back(std::make_pair(a_bot, b_bot));
        sharp_edges.push_back(std::make_pair(a_top, a_bot));
    }

    double side_len = (poly[1] - poly[0]).norm();
    double h = std::abs(ztop - zbot);
    double h0 = std::min(side_len, h) / std::max(1, seg);
    double band = 0.85 * h0;

    for (int pass = 0; pass < passes; pass++) {
        std::set<std::pair<int,int>> marked;
        int nt = m.nt();
        for (int t = 0; t < nt; t++) {
            int a = m.tris[3*t], b = m.tris[3*t + 1], c = m.tris[3*t + 2];
            Vec3 ctr = (m.verts[a] + m.verts[b] + m.verts[c]) * (1.0 / 3.0);
            double dmin = 1e300;
            for (const auto& e : sharp_edges)
                dmin = std::min(dmin, dist_point_segment(ctr, e.first, e.second));
            if (dmin <= band) {
                marked.insert(std::make_pair(std::min(a, b), std::max(a, b)));
                marked.insert(std::make_pair(std::min(b, c), std::max(b, c)));
                marked.insert(std::make_pair(std::min(c, a), std::max(c, a)));
            }
        }
        if (marked.empty())
            break;
        refine_marked_edges(m, marked);
    }

    double min_angle = M_PI;
    int below_25 = 0;
    for (int t = 0; t < m.nt(); t++) {
        Vec3 a, b, c;
        m.tri_verts(t, a, b, c);
        double angle = tri_quality_min_angle(a, b, c);
        min_angle = std::min(min_angle, angle);
        if (angle < 25.0 * M_PI / 180.0)
            below_25++;
    }
    printf("  [Mesh] Edge refinement: passes=%d, min_angle=%.1f deg, below25=%d/%d\n",
           passes, min_angle * 180.0 / M_PI, below_25, m.nt());
}

Mesh regular_prism(int sides, double aspect, int refinements, double equiv_radius,
                   int edge_refine) {
    if (sides < 3) sides = 3;
    if (aspect <= 0.0) aspect = 1.0;
    int seg = 1 << std::max(0, refinements);
    edge_refine = std::max(0, edge_refine);

    double Dx = 1.0;
    double Rc = (sides % 2 == 0)
        ? Dx / (2.0 * std::cos(M_PI / sides))
        : Dx / (1.0 + std::cos(M_PI / sides));
    double h = aspect * Dx;

    double base_area = 0.5 * sides * Rc * Rc * std::sin(2.0 * M_PI / sides);
    double volume = base_area * h;
    double target_volume = 4.0 * M_PI * equiv_radius * equiv_radius * equiv_radius / 3.0;
    double scale = std::pow(target_volume / volume, 1.0 / 3.0);
    Rc *= scale;
    h *= scale;

    std::vector<Vec3> poly(sides);
    for (int i = 0; i < sides; i++) {
        double ang = (2.0 * M_PI * i / sides) + M_PI / sides;
        poly[i] = Vec3(Rc * std::cos(ang), Rc * std::sin(ang), 0.0);
    }

    Mesh m;
    std::map<std::tuple<long long, long long, long long>, int> vmap;
    auto add_vertex = [&](const Vec3& p) -> int {
        const double q = 1e12;
        auto key = std::make_tuple((long long)std::llround(p.x * q),
                                   (long long)std::llround(p.y * q),
                                   (long long)std::llround(p.z * q));
        auto it = vmap.find(key);
        if (it != vmap.end()) return it->second;
        int id = (int)m.verts.size();
        m.verts.push_back(p);
        vmap[key] = id;
        return id;
    };

    auto add_tri = [&](int a, int b, int c) {
        m.tris.push_back(a);
        m.tris.push_back(b);
        m.tris.push_back(c);
    };

    auto interp = [](const Vec3& a, const Vec3& b, double t) {
        return a * (1.0 - t) + b * t;
    };

    double ztop = 0.5 * h, zbot = -0.5 * h;
    std::vector<double> u_grid = uniform_grid(seg);
    std::vector<double> z_grid = uniform_grid(seg);

    // Side surfaces.
    for (int e = 0; e < sides; e++) {
        Vec3 a = poly[e];
        Vec3 b = poly[(e + 1) % sides];
        for (int iu = 0; iu < seg; iu++) {
            double u0 = u_grid[iu];
            double u1 = u_grid[iu + 1];
            Vec3 p00 = interp(a, b, u0);
            Vec3 p10 = interp(a, b, u1);
            for (int iz = 0; iz < seg; iz++) {
                double t0 = z_grid[iz];
                double t1 = z_grid[iz + 1];
                double z0 = zbot * (1.0 - t0) + ztop * t0;
                double z1 = zbot * (1.0 - t1) + ztop * t1;
                int v00 = add_vertex(Vec3(p00.x, p00.y, z0));
                int v10 = add_vertex(Vec3(p10.x, p10.y, z0));
                int v01 = add_vertex(Vec3(p00.x, p00.y, z1));
                int v11 = add_vertex(Vec3(p10.x, p10.y, z1));
                add_tri(v00, v10, v11);
                add_tri(v00, v11, v01);
            }
        }
    }

    auto add_cap = [&](double z, bool top) {
        Vec3 center(0, 0, z);
        for (int e = 0; e < sides; e++) {
            Vec3 a(poly[e].x, poly[e].y, z);
            Vec3 b(poly[(e + 1) % sides].x, poly[(e + 1) % sides].y, z);
            std::vector<std::vector<int>> ids(seg + 1);
            for (int i = 0; i <= seg; i++) {
                ids[i].resize(seg - i + 1);
                for (int j = 0; j <= seg - i; j++) {
                    double wa = (double)i / seg;
                    double wb = (double)j / seg;
                    double wc = 1.0 - wa - wb;
                    ids[i][j] = add_vertex(a * wa + b * wb + center * wc);
                }
            }
            for (int i = 0; i < seg; i++) {
                for (int j = 0; j < seg - i; j++) {
                    int v0 = ids[i][j];
                    int v1 = ids[i + 1][j];
                    int v2 = ids[i][j + 1];
                    if (top) add_tri(v0, v1, v2);
                    else add_tri(v0, v2, v1);
                    if (j < seg - i - 1) {
                        int v3 = ids[i + 1][j + 1];
                        if (top) add_tri(v1, v3, v2);
                        else add_tri(v1, v2, v3);
                    }
                }
            }
        }
    };

    add_cap(ztop, true);
    add_cap(zbot, false);
    if (edge_refine > 0)
        refine_prism_edges(m, poly, ztop, zbot, seg, edge_refine);
    return m;
}
