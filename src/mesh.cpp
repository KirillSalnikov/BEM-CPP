#include "mesh.h"
#include <map>
#include <set>
#include <utility>
#include <tuple>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

static bool parse_obj_vertex_index(const std::string& tok, int vertex_count, int& out)
{
    const char* s = tok.c_str();
    char* end = nullptr;
    errno = 0;
    long raw = std::strtol(s, &end, 10);
    if (s == end || errno == ERANGE)
        return false;
    if (*end != '\0' && *end != '/')
        return false;
    if (raw == 0)
        return false;

    long idx = (raw > 0) ? raw - 1 : (long)vertex_count + raw;
    if (idx < 0 || idx >= vertex_count || idx > INT_MAX)
        return false;
    out = (int)idx;
    return true;
}

bool write_mesh_obj(const char* filename, const Mesh& mesh)
{
    std::ofstream stream(filename);
    if (!stream)
        return false;
    stream.precision(17);
    stream << "# BEM-CPP surface mesh: " << mesh.nv() << " vertices, "
           << mesh.nt() << " triangles\n";
    for (const Vec3& vertex : mesh.verts)
        stream << "v " << vertex.x << ' ' << vertex.y << ' ' << vertex.z << '\n';
    for (int triangle = 0; triangle < mesh.nt(); triangle++) {
        stream << "f " << mesh.tris[3 * triangle] + 1 << ' '
               << mesh.tris[3 * triangle + 1] + 1 << ' '
               << mesh.tris[3 * triangle + 2] + 1 << '\n';
    }
    return (bool)stream;
}

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

Mesh subdivide_flat(const Mesh& mesh)
{
    std::map<std::pair<int,int>, int> edge_mid;
    std::vector<Vec3> verts = mesh.verts;
    std::vector<int> new_tris;
    new_tris.reserve(mesh.tris.size() * 4);

    auto get_mid = [&](int a, int b) -> int {
        auto key = std::make_pair(std::min(a,b), std::max(a,b));
        auto it = edge_mid.find(key);
        if (it != edge_mid.end()) return it->second;
        Vec3 mid = (verts[a] + verts[b]) * 0.5;
        int idx = (int)verts.size();
        verts.push_back(mid);
        edge_mid[key] = idx;
        return idx;
    };

    int ntri = mesh.nt();
    for (int i = 0; i < ntri; i++) {
        int a = mesh.tris[3*i], b = mesh.tris[3*i+1], c = mesh.tris[3*i+2];
        int ab = get_mid(a, b);
        int bc = get_mid(b, c);
        int ca = get_mid(c, a);
        int t[] = {a,ab,ca, b,bc,ab, c,ca,bc, ab,bc,ca};
        new_tris.insert(new_tris.end(), t, t+12);
    }

    Mesh m;
    m.verts = verts;
    m.tris = new_tris;
    m.edge_refine_requested = mesh.edge_refine_requested;
    m.edge_refine_applied = mesh.edge_refine_applied;
    m.edge_refine_uniform_fallback = mesh.edge_refine_uniform_fallback;
    return m;
}

Mesh load_obj(const char* filename)
{
    Mesh m;
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: cannot open OBJ file: %s\n", filename);
        exit(1);
    }

    std::string line;
    int lineno = 0;
    while (std::getline(file, line)) {
        lineno++;
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "v") {
            double x, y, z;
            if (!(iss >> x >> y >> z)) {
                fprintf(stderr, "Error: invalid OBJ vertex at %s:%d\n", filename, lineno);
                exit(1);
            }
            m.verts.push_back(Vec3(x, y, z));
        } else if (prefix == "f") {
            std::vector<int> face_verts;
            std::string tok;
            while (iss >> tok) {
                int vi = -1;
                if (!parse_obj_vertex_index(tok, (int)m.verts.size(), vi)) {
                    fprintf(stderr, "Error: invalid OBJ face index '%s' at %s:%d\n",
                            tok.c_str(), filename, lineno);
                    exit(1);
                }
                face_verts.push_back(vi);
            }
            if ((int)face_verts.size() < 3) {
                fprintf(stderr, "Error: OBJ face with fewer than 3 vertices at %s:%d\n", filename, lineno);
                exit(1);
            }
            for (int i = 1; i + 1 < (int)face_verts.size(); i++) {
                m.tris.push_back(face_verts[0]);
                m.tris.push_back(face_verts[i]);
                m.tris.push_back(face_verts[i+1]);
            }
        }
    }

    std::vector<int> clean_tris;
    clean_tris.reserve(m.tris.size());
    int n_degenerate = 0;
    for (int i = 0; i < m.nt(); i++) {
        int v0 = m.tris[3*i], v1 = m.tris[3*i+1], v2 = m.tris[3*i+2];
        if (v0 == v1 || v1 == v2 || v0 == v2) {
            n_degenerate++;
            continue;
        }
        Vec3 e1 = m.verts[v1] - m.verts[v0];
        Vec3 e2 = m.verts[v2] - m.verts[v0];
        double area = 0.5 * e1.cross(e2).norm();
        if (area < 1e-10) {
            n_degenerate++;
            continue;
        }
        clean_tris.push_back(v0);
        clean_tris.push_back(v1);
        clean_tris.push_back(v2);
    }
    if (n_degenerate > 0) {
        m.tris.swap(clean_tris);
        fprintf(stderr, "Warning: removed %d degenerate triangles from OBJ\n", n_degenerate);
    }

    std::map<std::pair<int,int>, int> edge_count;
    for (int i = 0; i < m.nt(); i++) {
        int v[3] = {m.tris[3*i], m.tris[3*i+1], m.tris[3*i+2]};
        for (int e = 0; e < 3; e++) {
            int a = v[e], b = v[(e+1) % 3];
            auto key = std::make_pair(std::min(a,b), std::max(a,b));
            edge_count[key]++;
        }
    }
    int n_nonmanifold = 0;
    for (auto& kv : edge_count)
        if (kv.second > 2) n_nonmanifold++;
    if (n_nonmanifold > 0)
        fprintf(stderr, "Warning: %d non-manifold OBJ edges shared by >2 triangles\n", n_nonmanifold);

    double signed_vol = mesh_volume(m);
    if (signed_vol < 0.0) {
        for (int i = 0; i < m.nt(); i++)
            std::swap(m.tris[3*i + 1], m.tris[3*i + 2]);
        fprintf(stderr, "Warning: flipped OBJ triangle winding to outward normals\n");
    }

    printf("  Loaded OBJ: %d vertices, %d triangles from %s\n", m.nv(), m.nt(), filename);
    return m;
}

double mesh_volume(const Mesh& m)
{
    double vol = 0.0;
    for (int i = 0; i < m.nt(); i++) {
        Vec3 v0, v1, v2;
        m.tri_verts(i, v0, v1, v2);
        vol += v0.dot(v1.cross(v2));
    }
    return vol / 6.0;
}

double normalize_mesh(Mesh& m)
{
    double vol = std::fabs(mesh_volume(m));
    double a_eq = std::cbrt(3.0 * vol / (4.0 * M_PI));
    if (a_eq < 1e-30) {
        fprintf(stderr, "Warning: mesh volume near zero, using bounding radius\n");
        Vec3 center(0,0,0);
        for (auto& v : m.verts) center = center + v;
        center = center * (1.0 / std::max(1, m.nv()));
        double r2max = 0.0;
        for (auto& v : m.verts) {
            double r2 = (v - center).norm2();
            if (r2 > r2max) r2max = r2;
        }
        a_eq = std::sqrt(r2max);
    }
    double scale = 1.0 / a_eq;
    for (auto& v : m.verts)
        v = v * scale;
    return a_eq;
}

double mesh_dmax(const Mesh& m)
{
    double xmin=1e30, xmax=-1e30, ymin=1e30, ymax=-1e30, zmin=1e30, zmax=-1e30;
    for (auto& v : m.verts) {
        xmin = std::min(xmin, v.x); xmax = std::max(xmax, v.x);
        ymin = std::min(ymin, v.y); ymax = std::max(ymax, v.y);
        zmin = std::min(zmin, v.z); zmax = std::max(zmax, v.z);
    }
    double dx = xmax-xmin, dy = ymax-ymin, dz = zmax-zmin;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
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

static void tri_edges(const Vec3& a, const Vec3& b, const Vec3& c,
                      double& ab, double& bc, double& ca)
{
    ab = (a - b).norm();
    bc = (b - c).norm();
    ca = (c - a).norm();
}

static double tri_quality_aspect_ratio(const Vec3& a, const Vec3& b, const Vec3& c)
{
    double ab, bc, ca;
    tri_edges(a, b, c, ab, bc, ca);
    double area2 = (b - a).cross(c - a).norm();
    double perimeter = ab + bc + ca;
    if (area2 <= 1e-300 || perimeter <= 1e-300)
        return 1e300;
    double area = 0.5 * area2;
    double inradius = 2.0 * area / perimeter;
    double longest = std::max(ab, std::max(bc, ca));
    return longest / (2.0 * std::sqrt(3.0) * inradius);
}

static double percentile_sorted(const std::vector<double>& sorted, double p)
{
    if (sorted.empty())
        return 0.0;
    if (p <= 0.0)
        return sorted.front();
    if (p >= 100.0)
        return sorted.back();
    double x = (p / 100.0) * (double)(sorted.size() - 1);
    int i0 = (int)std::floor(x);
    int i1 = std::min(i0 + 1, (int)sorted.size() - 1);
    double t = x - (double)i0;
    return sorted[i0] * (1.0 - t) + sorted[i1] * t;
}

static const int kNearTouchTriangleLimit = 8192;
static const double kNearTouchCentroidRatio = 0.35;

static int near_touch_triangle_limit()
{
    const char* env = std::getenv("BEM_MESH_NEAR_TOUCH_LIMIT");
    if (!env || !*env)
        return kNearTouchTriangleLimit;
    char* end = nullptr;
    errno = 0;
    long v = std::strtol(env, &end, 10);
    if (errno != 0 || end == env || v < 0 || v > INT_MAX)
        return kNearTouchTriangleLimit;
    return (int)v;
}

MeshQualityReport analyze_mesh_quality(const Mesh& m,
                                       double min_angle_warn_deg,
                                       double max_aspect_warn)
{
    MeshQualityReport q;
    q.vertices = m.nv();
    q.triangles = m.nt();
    q.edge_refine_requested = m.edge_refine_requested;
    q.edge_refine_applied = m.edge_refine_applied;
    q.edge_refine_uniform_fallback = m.edge_refine_uniform_fallback;
    q.signed_volume = mesh_volume(m);
    q.outward_winding = q.signed_volume > 0.0;

    std::map<std::pair<int,int>, int> edge_count;
    std::map<std::pair<int,int>, std::vector<int>> edge_to_triangles;
    std::vector<double> angles;
    std::vector<double> edge_lengths;
    std::vector<Vec3> centroids;
    std::vector<double> local_size;
    std::vector<Vec3> tri_normals(std::max(0, m.nt()), Vec3(0, 0, 0));
    std::vector<double> tri_areas(std::max(0, m.nt()), 0.0);
    angles.reserve((size_t)3 * std::max(0, m.nt()));
    edge_lengths.reserve((size_t)3 * std::max(0, m.nt()));
    centroids.reserve(std::max(0, m.nt()));
    local_size.reserve(std::max(0, m.nt()));
    double area_sum = 0.0;
    q.min_area = 1e300;
    q.max_area = 0.0;
    q.min_edge = 1e300;
    q.max_edge = 0.0;
    q.max_edge_ratio = 0.0;
    q.max_aspect_ratio = 0.0;

    for (int t = 0; t < m.nt(); t++) {
        int ia = m.tris[3*t], ib = m.tris[3*t + 1], ic = m.tris[3*t + 2];
        if (ia < 0 || ib < 0 || ic < 0 || ia >= m.nv() || ib >= m.nv() || ic >= m.nv()) {
            centroids.push_back(Vec3(0, 0, 0));
            local_size.push_back(0.0);
            continue;
        }
        Vec3 a = m.verts[ia], b = m.verts[ib], c = m.verts[ic];
        Vec3 normal_raw = (b - a).cross(c - a);
        double normal_len = normal_raw.norm();
        double area = 0.5 * normal_len;
        if (area <= 1e-14)
            q.degenerate_triangles++;
        tri_normals[t] = (normal_len > 1e-300) ? normal_raw * (1.0 / normal_len) : Vec3(0, 0, 0);
        tri_areas[t] = area;
        q.min_area = std::min(q.min_area, area);
        q.max_area = std::max(q.max_area, area);
        area_sum += area;

        double ab, bc, ca;
        tri_edges(a, b, c, ab, bc, ca);
        double e_min = std::min(ab, std::min(bc, ca));
        double e_max = std::max(ab, std::max(bc, ca));
        centroids.push_back((a + b + c) * (1.0 / 3.0));
        local_size.push_back(e_min);
        if (e_min > 0.0)
            q.max_edge_ratio = std::max(q.max_edge_ratio, e_max / e_min);
        q.min_edge = std::min(q.min_edge, e_min);
        q.max_edge = std::max(q.max_edge, e_max);
        edge_lengths.push_back(ab);
        edge_lengths.push_back(bc);
        edge_lengths.push_back(ca);

        double min_angle = tri_quality_min_angle(a, b, c) * 180.0 / M_PI;
        if (min_angle < min_angle_warn_deg)
            q.skinny_triangles++;
        q.max_aspect_ratio = std::max(q.max_aspect_ratio,
                                      tri_quality_aspect_ratio(a, b, c));

        double la = bc, lb = ca, lc = ab;
        double eps = 1e-300;
        double ca0 = std::max(-1.0, std::min(1.0, (lb*lb + lc*lc - la*la) / (2.0 * lb * lc + eps)));
        double cb0 = std::max(-1.0, std::min(1.0, (lc*lc + la*la - lb*lb) / (2.0 * lc * la + eps)));
        double cc0 = std::max(-1.0, std::min(1.0, (la*la + lb*lb - lc*lc) / (2.0 * la * lb + eps)));
        angles.push_back(std::acos(ca0) * 180.0 / M_PI);
        angles.push_back(std::acos(cb0) * 180.0 / M_PI);
        angles.push_back(std::acos(cc0) * 180.0 / M_PI);

        int v[3] = {ia, ib, ic};
        for (int e = 0; e < 3; e++) {
            int u = v[e], w = v[(e + 1) % 3];
            auto key = std::make_pair(std::min(u, w), std::max(u, w));
            edge_count[key]++;
            edge_to_triangles[key].push_back(t);
        }
    }

    q.unique_edges = (int)edge_count.size();
    for (const auto& kv : edge_count) {
        if (kv.second == 1)
            q.boundary_edges++;
        else if (kv.second == 2)
            q.manifold_edges++;
        else if (kv.second > 2)
            q.nonmanifold_edges++;
    }
    const double feature_threshold_deg = 30.0;
    double feature_sum = 0.0;
    for (const auto& kv : edge_to_triangles) {
        const std::vector<int>& ts = kv.second;
        if (ts.size() != 2)
            continue;
        int t0 = ts[0], t1 = ts[1];
        double dot = tri_normals[t0].dot(tri_normals[t1]);
        dot = std::max(-1.0, std::min(1.0, dot));
        double dihedral = std::acos(dot) * 180.0 / M_PI;
        q.max_dihedral_deg = std::max(q.max_dihedral_deg, dihedral);
        if (tri_areas[t0] > 1e-300 && tri_areas[t1] > 1e-300) {
            double ar = std::max(tri_areas[t0], tri_areas[t1]) / std::min(tri_areas[t0], tri_areas[t1]);
            q.max_adjacent_area_ratio = std::max(q.max_adjacent_area_ratio, ar);
        }
        if (dihedral >= feature_threshold_deg) {
            q.feature_edges_30deg++;
            feature_sum += dihedral;
        }
    }
    if (q.feature_edges_30deg > 0)
        q.mean_feature_dihedral_deg = feature_sum / (double)q.feature_edges_30deg;
    q.feature_edge_fraction = (q.manifold_edges > 0) ?
        (double)q.feature_edges_30deg / (double)q.manifold_edges : 0.0;
    q.voxel_surface_like =
        q.feature_edges_30deg > 0 &&
        (q.feature_edge_fraction >= 0.30 ||
         (q.mean_feature_dihedral_deg >= 85.0 && q.feature_edge_fraction >= 0.10));
    q.closed = (q.boundary_edges == 0 && q.nonmanifold_edges == 0);
    int near_touch_limit = near_touch_triangle_limit();
    q.near_touch_checked = m.nt() <= near_touch_limit;
    q.near_touch_ratio = 1e300;
    q.near_touch_pairs = 0;
    q.self_panel_count = m.nt();
    q.edge_adjacent_pair_count = 0;
    q.vertex_adjacent_pair_count = 0;
    q.near_disjoint_pair_count = 0;
    if (q.near_touch_checked) {
        for (int a = 0; a < m.nt(); a++) {
            int av[3] = {m.tris[3*a], m.tris[3*a + 1], m.tris[3*a + 2]};
            for (int b = a + 1; b < m.nt(); b++) {
                int bv[3] = {m.tris[3*b], m.tris[3*b + 1], m.tris[3*b + 2]};
                int shared_vertices = 0;
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        if (av[i] == bv[j])
                            shared_vertices++;
                if (shared_vertices >= 2) {
                    q.edge_adjacent_pair_count++;
                    continue;
                }
                if (shared_vertices == 1) {
                    q.vertex_adjacent_pair_count++;
                    continue;
                }
                double denom = std::max(1e-300, std::min(local_size[a], local_size[b]));
                double ratio = (centroids[a] - centroids[b]).norm() / denom;
                q.near_touch_ratio = std::min(q.near_touch_ratio, ratio);
                if (ratio < kNearTouchCentroidRatio) {
                    q.near_touch_pairs++;
                    q.near_disjoint_pair_count++;
                }
            }
        }
        if (q.near_touch_ratio == 1e300)
            q.near_touch_ratio = 0.0;
    }
    q.taylor_duffy_candidate_count =
        q.self_panel_count +
        q.edge_adjacent_pair_count +
        q.vertex_adjacent_pair_count +
        q.near_disjoint_pair_count;
    q.mean_area = (m.nt() > 0) ? area_sum / (double)m.nt() : 0.0;
    if (q.min_area == 1e300)
        q.min_area = 0.0;
    if (q.min_edge == 1e300)
        q.min_edge = 0.0;
    if (!edge_lengths.empty()) {
        double sum = 0.0;
        for (double x : edge_lengths)
            sum += x;
        q.mean_edge = sum / (double)edge_lengths.size();
    }
    std::sort(angles.begin(), angles.end());
    q.min_angle_deg = percentile_sorted(angles, 0.0);
    q.p01_angle_deg = percentile_sorted(angles, 1.0);
    q.p05_angle_deg = percentile_sorted(angles, 5.0);
    q.median_angle_deg = percentile_sorted(angles, 50.0);

    q.pass_default_gate =
        q.closed &&
        q.degenerate_triangles == 0 &&
        q.min_angle_deg >= min_angle_warn_deg &&
        q.max_aspect_ratio <= max_aspect_warn &&
        (!q.near_touch_checked || q.near_touch_pairs == 0) &&
        q.outward_winding;

    q.requires_remesh = !q.closed || q.degenerate_triangles > 0 ||
        !q.outward_winding || q.near_disjoint_pair_count > 0 ||
        q.min_angle_deg < min_angle_warn_deg ||
        q.max_aspect_ratio > max_aspect_warn;
    if (!q.closed || q.degenerate_triangles > 0 || !q.outward_winding) {
        q.recommended_mesh_strategy = "repair_topology";
        q.recommended_mesh_action = "fix closed outward manifold mesh before solving";
        q.recommended_min_quad_order = 13;
    } else if (q.near_disjoint_pair_count > 0) {
        q.recommended_mesh_strategy = "near_singular_remesh";
        q.recommended_mesh_action = "separate or remesh near-touching nonadjacent panels before accepting result";
        q.recommended_min_quad_order = 13;
    } else if (q.min_angle_deg < min_angle_warn_deg || q.max_aspect_ratio > max_aspect_warn) {
        q.recommended_mesh_strategy = "quality_remesh";
        q.recommended_mesh_action = "improve minimum angle and aspect ratio before production solve";
        q.recommended_min_quad_order = 13;
    } else if (q.voxel_surface_like) {
        q.recommended_mesh_strategy = "cubical_edge_aware_quadrature";
        q.recommended_mesh_action = "keep the closed triangulated surface and use high-order near-edge quadrature";
        q.recommended_min_quad_order = 13;
    } else if (q.feature_edges_30deg > 0) {
        q.recommended_mesh_strategy = "edge_aware_refinement";
        q.recommended_mesh_action = "keep conforming edge-aware refinement near sharp dihedral edges";
        q.recommended_min_quad_order = 7;
    } else {
        q.recommended_mesh_strategy = "uniform_curvature_refinement";
        q.recommended_mesh_action = "uniform smooth-surface refinement is acceptable";
        q.recommended_min_quad_order = 4;
    }

    if (q.pass_default_gate) {
        q.verdict = "pass";
    } else {
        q.verdict = "warn";
        if (!q.closed || q.degenerate_triangles > 0 || !q.outward_winding)
            q.verdict = "fail";
    }
    return q;
}

void print_mesh_quality_report(const MeshQualityReport& q)
{
    printf("  Mesh quality: %s, closed=%s, outward=%s, boundary=%d, nonmanifold=%d\n",
           q.verdict.c_str(), q.closed ? "yes" : "no",
           q.outward_winding ? "yes" : "no",
           q.boundary_edges, q.nonmanifold_edges);
    printf("    angles: min=%.2f deg, p1=%.2f, p5=%.2f, median=%.2f; skinny=%d/%d\n",
           q.min_angle_deg, q.p01_angle_deg, q.p05_angle_deg, q.median_angle_deg,
           q.skinny_triangles, q.triangles);
    printf("    edges: min=%.4g, mean=%.4g, max=%.4g, max edge ratio=%.3g; max aspect=%.3g\n",
           q.min_edge, q.mean_edge, q.max_edge, q.max_edge_ratio, q.max_aspect_ratio);
    printf("    feature edges: sharp30=%d/%d (%.3g), max dihedral=%.2f deg, mean sharp=%.2f deg, max adjacent area ratio=%.3g, voxel_like=%s\n",
           q.feature_edges_30deg, q.manifold_edges, q.feature_edge_fraction,
           q.max_dihedral_deg, q.mean_feature_dihedral_deg,
           q.max_adjacent_area_ratio, q.voxel_surface_like ? "yes" : "no");
    printf("    areas: min=%.4g, mean=%.4g, max=%.4g; signed volume=%.6g\n",
           q.min_area, q.mean_area, q.max_area, q.signed_volume);
    if (q.near_touch_checked)
        printf("    near-touch: min centroid/local-edge ratio=%.3g, suspect pairs=%d\n",
               q.near_touch_ratio, q.near_touch_pairs);
    else
        printf("    near-touch: skipped for %d triangles (limit %d)\n",
               q.triangles, near_touch_triangle_limit());
    printf("    singular classes: self=%d, edge-adjacent=%d, vertex-adjacent=%d, near-disjoint=%d, Taylor-Duffy candidates=%d\n",
           q.self_panel_count, q.edge_adjacent_pair_count,
           q.vertex_adjacent_pair_count, q.near_disjoint_pair_count,
           q.taylor_duffy_candidate_count);
    printf("    mesh strategy: %s, action=%s, min quad=%d, requires remesh=%s\n",
           q.recommended_mesh_strategy.c_str(), q.recommended_mesh_action.c_str(),
           q.recommended_min_quad_order, q.requires_remesh ? "yes" : "no");
    if (q.edge_refine_requested > 0)
        printf("    edge-refine: requested=%d, applied=%d, uniform_fallback=%s\n",
               q.edge_refine_requested, q.edge_refine_applied,
               q.edge_refine_uniform_fallback ? "yes" : "no");
}

bool write_mesh_quality_json(const char* path, const MeshQualityReport& q,
                             const char* shape, double ka, int ref_or_subdiv,
                             int quad_order)
{
    std::ofstream os(path);
    if (!os)
        return false;
    os.setf(std::ios::scientific);
    os.precision(17);
    os << "{\n";
    os << "  \"shape\": \"" << (shape ? shape : "") << "\",\n";
    os << "  \"ka\": " << ka << ",\n";
    os << "  \"ref_or_subdiv\": " << ref_or_subdiv << ",\n";
    os << "  \"quad_order\": " << quad_order << ",\n";
    os << "  \"vertices\": " << q.vertices << ",\n";
    os << "  \"triangles\": " << q.triangles << ",\n";
    os << "  \"unique_edges\": " << q.unique_edges << ",\n";
    os << "  \"manifold_edges\": " << q.manifold_edges << ",\n";
    os << "  \"boundary_edges\": " << q.boundary_edges << ",\n";
    os << "  \"nonmanifold_edges\": " << q.nonmanifold_edges << ",\n";
    os << "  \"feature_edges_30deg\": " << q.feature_edges_30deg << ",\n";
    os << "  \"feature_edge_fraction\": " << q.feature_edge_fraction << ",\n";
    os << "  \"degenerate_triangles\": " << q.degenerate_triangles << ",\n";
    os << "  \"skinny_triangles\": " << q.skinny_triangles << ",\n";
    os << "  \"closed\": " << (q.closed ? "true" : "false") << ",\n";
    os << "  \"outward_winding\": " << (q.outward_winding ? "true" : "false") << ",\n";
    os << "  \"voxel_surface_like\": " << (q.voxel_surface_like ? "true" : "false") << ",\n";
    os << "  \"pass_default_gate\": " << (q.pass_default_gate ? "true" : "false") << ",\n";
    os << "  \"verdict\": \"" << q.verdict << "\",\n";
    os << "  \"signed_volume\": " << q.signed_volume << ",\n";
    os << "  \"min_area\": " << q.min_area << ",\n";
    os << "  \"mean_area\": " << q.mean_area << ",\n";
    os << "  \"max_area\": " << q.max_area << ",\n";
    os << "  \"min_edge\": " << q.min_edge << ",\n";
    os << "  \"mean_edge\": " << q.mean_edge << ",\n";
    os << "  \"max_edge\": " << q.max_edge << ",\n";
    os << "  \"max_edge_ratio\": " << q.max_edge_ratio << ",\n";
    os << "  \"max_aspect_ratio\": " << q.max_aspect_ratio << ",\n";
    os << "  \"max_dihedral_deg\": " << q.max_dihedral_deg << ",\n";
    os << "  \"mean_feature_dihedral_deg\": " << q.mean_feature_dihedral_deg << ",\n";
    os << "  \"max_adjacent_area_ratio\": " << q.max_adjacent_area_ratio << ",\n";
    os << "  \"near_touch_checked\": " << (q.near_touch_checked ? "true" : "false") << ",\n";
    os << "  \"near_touch_ratio\": " << q.near_touch_ratio << ",\n";
    os << "  \"near_touch_pairs\": " << q.near_touch_pairs << ",\n";
    os << "  \"self_panel_count\": " << q.self_panel_count << ",\n";
    os << "  \"edge_adjacent_pair_count\": " << q.edge_adjacent_pair_count << ",\n";
    os << "  \"vertex_adjacent_pair_count\": " << q.vertex_adjacent_pair_count << ",\n";
    os << "  \"near_disjoint_pair_count\": " << q.near_disjoint_pair_count << ",\n";
    os << "  \"taylor_duffy_candidate_count\": " << q.taylor_duffy_candidate_count << ",\n";
    os << "  \"recommended_min_quad_order\": " << q.recommended_min_quad_order << ",\n";
    os << "  \"recommended_mesh_strategy\": \"" << q.recommended_mesh_strategy << "\",\n";
    os << "  \"recommended_mesh_action\": \"" << q.recommended_mesh_action << "\",\n";
    os << "  \"requires_remesh\": " << (q.requires_remesh ? "true" : "false") << ",\n";
    os << "  \"edge_refine_requested\": " << q.edge_refine_requested << ",\n";
    os << "  \"edge_refine_applied\": " << q.edge_refine_applied << ",\n";
    os << "  \"edge_refine_uniform_fallback\": " << (q.edge_refine_uniform_fallback ? "true" : "false") << ",\n";
    os << "  \"min_angle_deg\": " << q.min_angle_deg << ",\n";
    os << "  \"p01_angle_deg\": " << q.p01_angle_deg << ",\n";
    os << "  \"p05_angle_deg\": " << q.p05_angle_deg << ",\n";
    os << "  \"median_angle_deg\": " << q.median_angle_deg << "\n";
    os << "}\n";
    return true;
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

static double marked_refinement_min_angle(
    const Vec3& a, const Vec3& b, const Vec3& c,
    bool mab, bool mbc, bool mca)
{
    const Vec3 ab = (a + b) * 0.5;
    const Vec3 bc = (b + c) * 0.5;
    const Vec3 ca = (c + a) * 0.5;
    double minimum = M_PI;
    auto child = [&](const Vec3& p0, const Vec3& p1, const Vec3& p2) {
        minimum = std::min(minimum, tri_quality_min_angle(p0, p1, p2));
    };

    const int nmark = (mab ? 1 : 0) + (mbc ? 1 : 0) + (mca ? 1 : 0);
    if (nmark == 1) {
        if (mab) {
            child(a, ab, c);
            child(ab, b, c);
        } else if (mbc) {
            child(b, bc, a);
            child(bc, c, a);
        } else {
            child(c, ca, b);
            child(ca, a, b);
        }
    } else if (nmark == 2) {
        if (mab && mbc) {
            child(ab, b, bc);
            child(a, ab, bc);
            child(a, bc, c);
        } else if (mbc && mca) {
            child(bc, c, ca);
            child(b, bc, ca);
            child(b, ca, a);
        } else {
            child(ca, a, ab);
            child(c, ca, ab);
            child(c, ab, b);
        }
    } else {
        minimum = tri_quality_min_angle(a, b, c);
    }
    return minimum;
}

static double refine_prism_edges(Mesh& m, const std::vector<Vec3>& poly,
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
    double tol = 1e-8 * std::max(side_len, h);

    auto on_sharp_edge = [&](const Vec3& a, const Vec3& b) {
        for (const auto& e : sharp_edges) {
            if (dist_point_segment(a, e.first, e.second) <= tol &&
                dist_point_segment(b, e.first, e.second) <= tol)
                return true;
        }
        return false;
    };

    for (int pass = 0; pass < passes; pass++) {
        std::set<std::pair<int,int>> marked;
        int nt = m.nt();
        for (int t = 0; t < nt; t++) {
            int a = m.tris[3*t], b = m.tris[3*t + 1], c = m.tris[3*t + 2];
            bool touches_sharp =
                on_sharp_edge(m.verts[a], m.verts[b]) ||
                on_sharp_edge(m.verts[b], m.verts[c]) ||
                on_sharp_edge(m.verts[c], m.verts[a]);
            if (touches_sharp) {
                marked.insert(std::make_pair(std::min(a, b), std::max(a, b)));
                marked.insert(std::make_pair(std::min(b, c), std::max(b, c)));
                marked.insert(std::make_pair(std::min(c, a), std::max(c, a)));
            }
        }
        if (marked.empty())
            break;

        // Extend the red-refined band only where a green transition would
        // create a poor triangle. This preserves a local edge layer without
        // accepting the skinny elements produced by an arbitrary bisection.
        const double min_transition_angle = 25.0 * M_PI / 180.0;
        bool closure_changed = true;
        while (closure_changed) {
            closure_changed = false;
            for (int t = 0; t < nt; t++) {
                int a = m.tris[3*t], b = m.tris[3*t + 1],
                    c = m.tris[3*t + 2];
                auto edge_key = [](int p, int q) {
                    return std::make_pair(std::min(p, q), std::max(p, q));
                };
                bool mab = marked.find(edge_key(a, b)) != marked.end();
                bool mbc = marked.find(edge_key(b, c)) != marked.end();
                bool mca = marked.find(edge_key(c, a)) != marked.end();
                int nmark =
                    (mab ? 1 : 0) + (mbc ? 1 : 0) + (mca ? 1 : 0);
                if ((nmark == 1 || nmark == 2) &&
                    marked_refinement_min_angle(
                        m.verts[a], m.verts[b], m.verts[c],
                        mab, mbc, mca) < min_transition_angle) {
                    closure_changed |= marked.insert(edge_key(a, b)).second;
                    closure_changed |= marked.insert(edge_key(b, c)).second;
                    closure_changed |= marked.insert(edge_key(c, a)).second;
                }
            }
        }
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
    return min_angle * 180.0 / M_PI;
}

double refine_feature_edges(
    Mesh& mesh, double feature_angle_degrees, int passes)
{
    passes = std::max(0, passes);
    mesh.edge_refine_requested = passes;
    mesh.edge_refine_applied = 0;
    mesh.edge_refine_uniform_fallback = false;
    const double threshold =
        std::max(0.0, std::min(180.0, feature_angle_degrees)) *
        M_PI / 180.0;
    const auto edge_key = [](int a, int b) {
        return std::make_pair(std::min(a, b), std::max(a, b));
    };

    for (int pass = 0; pass < passes; pass++) {
        const int triangle_count = mesh.nt();
        std::vector<Vec3> normals(triangle_count);
        std::map<std::pair<int, int>, std::vector<int>> adjacency;
        for (int triangle = 0; triangle < triangle_count; triangle++) {
            const int a = mesh.tris[3 * triangle];
            const int b = mesh.tris[3 * triangle + 1];
            const int c = mesh.tris[3 * triangle + 2];
            const Vec3 raw =
                (mesh.verts[b] - mesh.verts[a]).cross(
                    mesh.verts[c] - mesh.verts[a]);
            const double length = raw.norm();
            normals[triangle] =
                length > 1.0e-300 ? raw * (1.0 / length) : Vec3();
            adjacency[edge_key(a, b)].push_back(triangle);
            adjacency[edge_key(b, c)].push_back(triangle);
            adjacency[edge_key(c, a)].push_back(triangle);
        }

        std::set<std::pair<int, int>> feature_edges;
        for (const auto& entry : adjacency) {
            if (entry.second.size() != 2)
                continue;
            double cosine = normals[entry.second[0]].dot(
                normals[entry.second[1]]);
            cosine = std::max(-1.0, std::min(1.0, cosine));
            if (std::acos(cosine) >= threshold)
                feature_edges.insert(entry.first);
        }
        if (feature_edges.empty())
            break;

        std::set<std::pair<int, int>> marked;
        for (int triangle = 0; triangle < triangle_count; triangle++) {
            const int a = mesh.tris[3 * triangle];
            const int b = mesh.tris[3 * triangle + 1];
            const int c = mesh.tris[3 * triangle + 2];
            const bool touches_feature =
                feature_edges.count(edge_key(a, b)) != 0 ||
                feature_edges.count(edge_key(b, c)) != 0 ||
                feature_edges.count(edge_key(c, a)) != 0;
            if (!touches_feature)
                continue;
            marked.insert(edge_key(a, b));
            marked.insert(edge_key(b, c));
            marked.insert(edge_key(c, a));
        }

        const double min_transition_angle = 25.0 * M_PI / 180.0;
        bool closure_changed = true;
        while (closure_changed) {
            closure_changed = false;
            for (int triangle = 0; triangle < triangle_count; triangle++) {
                const int a = mesh.tris[3 * triangle];
                const int b = mesh.tris[3 * triangle + 1];
                const int c = mesh.tris[3 * triangle + 2];
                const bool mab = marked.count(edge_key(a, b)) != 0;
                const bool mbc = marked.count(edge_key(b, c)) != 0;
                const bool mca = marked.count(edge_key(c, a)) != 0;
                const int count =
                    (mab ? 1 : 0) + (mbc ? 1 : 0) + (mca ? 1 : 0);
                if ((count == 1 || count == 2) &&
                    marked_refinement_min_angle(
                        mesh.verts[a], mesh.verts[b], mesh.verts[c],
                        mab, mbc, mca) < min_transition_angle) {
                    closure_changed |= marked.insert(edge_key(a, b)).second;
                    closure_changed |= marked.insert(edge_key(b, c)).second;
                    closure_changed |= marked.insert(edge_key(c, a)).second;
                }
            }
        }
        refine_marked_edges(mesh, marked);
        mesh.edge_refine_applied++;
    }

    double minimum_angle = M_PI;
    int below_25 = 0;
    for (int triangle = 0; triangle < mesh.nt(); triangle++) {
        Vec3 a, b, c;
        mesh.tri_verts(triangle, a, b, c);
        const double angle = tri_quality_min_angle(a, b, c);
        minimum_angle = std::min(minimum_angle, angle);
        if (angle < 25.0 * M_PI / 180.0)
            below_25++;
    }
    std::printf(
        "  [Mesh] Generic feature-edge refinement: requested=%d, "
        "applied=%d, threshold=%.1f deg, min_angle=%.1f deg, "
        "below25=%d/%d\n",
        passes, mesh.edge_refine_applied, feature_angle_degrees,
        minimum_angle * 180.0 / M_PI, below_25, mesh.nt());
    return minimum_angle * 180.0 / M_PI;
}

Mesh regular_prism(int sides, double aspect, int refinements, double equiv_radius,
                   int edge_refine, bool mirror_symmetric_sides) {
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
    double side_len = (poly[1] - poly[0]).norm();
    int side_zseg = std::max(seg, (int)std::ceil((h / side_len) * seg));
    std::vector<double> u_grid = uniform_grid(seg);
    std::vector<double> z_grid = uniform_grid(side_zseg);

    // Side surfaces.
    for (int e = 0; e < sides; e++) {
        Vec3 a = poly[e];
        Vec3 b = poly[(e + 1) % sides];
        for (int iu = 0; iu < seg; iu++) {
            double u0 = u_grid[iu];
            double u1 = u_grid[iu + 1];
            Vec3 p00 = interp(a, b, u0);
            Vec3 p10 = interp(a, b, u1);
            for (int iz = 0; iz < side_zseg; iz++) {
                double t0 = z_grid[iz];
                double t1 = z_grid[iz + 1];
                double z0 = zbot * (1.0 - t0) + ztop * t0;
                double z1 = zbot * (1.0 - t1) + ztop * t1;
                int v00 = add_vertex(Vec3(p00.x, p00.y, z0));
                int v10 = add_vertex(Vec3(p10.x, p10.y, z0));
                int v01 = add_vertex(Vec3(p00.x, p00.y, z1));
                int v11 = add_vertex(Vec3(p10.x, p10.y, z1));
                const bool forward_diagonal =
                    !mirror_symmetric_sides || (e % 2 == 0);
                if (forward_diagonal) {
                    add_tri(v00, v10, v11);
                    add_tri(v00, v11, v01);
                } else {
                    add_tri(v00, v10, v01);
                    add_tri(v10, v11, v01);
                }
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
    m.edge_refine_requested = edge_refine;
    m.edge_refine_applied = 0;
    m.edge_refine_uniform_fallback = false;
    if (edge_refine > 0) {
        Mesh base = m;
        double min_angle = refine_prism_edges(m, poly, ztop, zbot, seg, edge_refine);
        const double min_allowed_angle = 25.0;
        if (min_angle < min_allowed_angle) {
            m = base;
            for (int pass = 0; pass < edge_refine; pass++)
                m = subdivide_flat(m);
            m.edge_refine_requested = edge_refine;
            m.edge_refine_applied = 0;
            m.edge_refine_uniform_fallback = true;
            MeshQualityReport fallback_q = analyze_mesh_quality(m);
            printf("  [Mesh] Edge refinement rejected: min_angle=%.1f deg < %.1f deg; "
                   "using uniform fallback passes=%d, min_angle=%.1f deg\n",
                   min_angle, min_allowed_angle, edge_refine,
                   fallback_q.min_angle_deg);
        } else {
            m.edge_refine_requested = edge_refine;
            m.edge_refine_applied = edge_refine;
            m.edge_refine_uniform_fallback = false;
        }
    }
    return m;
}

//======================================================================================================================

Mesh structured_cube(int refinements, double equiv_radius)
{
    const int cells = 1 << std::max(0, refinements);
    const double target_volume =
        4.0 * M_PI * equiv_radius * equiv_radius * equiv_radius / 3.0;
    const double edge = std::cbrt(target_volume);
    const double half = 0.5 * edge;

    Mesh mesh;
    std::map<std::tuple<int, int, int>, int> vertex_ids;
    auto vertex = [&](int ix, int iy, int iz) {
        const std::tuple<int, int, int> key(ix, iy, iz);
        const auto found = vertex_ids.find(key);
        if (found != vertex_ids.end())
            return found->second;
        const double scale = edge / static_cast<double>(cells);
        const int id = static_cast<int>(mesh.verts.size());
        mesh.verts.push_back(Vec3(
            -half + scale * ix,
            -half + scale * iy,
            -half + scale * iz));
        vertex_ids[key] = id;
        return id;
    };
    auto triangle = [&](int a, int b, int c) {
        mesh.tris.push_back(a);
        mesh.tris.push_back(b);
        mesh.tris.push_back(c);
    };
    auto face = [&](int face_index) {
        std::vector<int> ids((cells + 1) * (cells + 1));
        auto at = [&](int i, int j) -> int& {
            return ids[i * (cells + 1) + j];
        };
        for (int i = 0; i <= cells; i++) {
            for (int j = 0; j <= cells; j++) {
                switch (face_index) {
                    case 0: at(i, j) = vertex(cells, i, j); break; // +x
                    case 1: at(i, j) = vertex(0, j, i); break;     // -x
                    case 2: at(i, j) = vertex(j, cells, i); break; // +y
                    case 3: at(i, j) = vertex(i, 0, j); break;     // -y
                    case 4: at(i, j) = vertex(i, j, cells); break; // +z
                    default: at(i, j) = vertex(j, i, 0); break;    // -z
                }
            }
        }
        for (int i = 0; i < cells; i++) {
            for (int j = 0; j < cells; j++) {
                const int v00 = at(i, j);
                const int v10 = at(i + 1, j);
                const int v11 = at(i + 1, j + 1);
                const int v01 = at(i, j + 1);
                if ((i + j) % 2 == 0) {
                    triangle(v00, v10, v11);
                    triangle(v00, v11, v01);
                } else {
                    triangle(v00, v10, v01);
                    triangle(v10, v11, v01);
                }
            }
        }
    };

    for (int face_index = 0; face_index < 6; face_index++)
        face(face_index);
    return mesh;
}
