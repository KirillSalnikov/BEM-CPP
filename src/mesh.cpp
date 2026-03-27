#include "mesh.h"
#include <map>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

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

Mesh subdivide_flat(const Mesh& mesh) {
    std::map<std::pair<int,int>, int> edge_mid;
    std::vector<Vec3> verts = mesh.verts;
    std::vector<int> new_tris;

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
    return m;
}

Mesh hex_prism(double aspect_ratio, int refinements) {
    // Hexagonal prism with unit equivalent-sphere radius.
    // aspect_ratio = H / Dx, where Dx = inscribed diameter = a*sqrt(3) (same as ADDA)
    // a = circumscribed radius, Ri = a*cos(pi/6) = a*sqrt(3)/2, Dx = 2*Ri = a*sqrt(3)
    // H = aspect_ratio * Dx = aspect_ratio * a * sqrt(3)
    // Volume: (3*sqrt(3)/2) * a^2 * H = (4/3)*pi
    //   => (9/2) * AR * a^3 = (4/3)*pi  =>  a = ((8*pi)/(27*AR))^(1/3)
    double a = std::pow(8.0 * M_PI / (27.0 * aspect_ratio), 1.0/3.0);
    double H = aspect_ratio * a * std::sqrt(3.0);

    // 14 vertices: top center, top hex (6), bottom center, bottom hex (6)
    std::vector<Vec3> verts(14);
    verts[0] = Vec3(0, 0, H/2);   // top center
    verts[7] = Vec3(0, 0, -H/2);  // bottom center
    for (int i = 0; i < 6; i++) {
        double angle = 2.0 * M_PI * i / 6.0;
        double x = a * std::cos(angle);
        double y = a * std::sin(angle);
        verts[1 + i] = Vec3(x, y,  H/2);   // top hex vertex
        verts[8 + i] = Vec3(x, y, -H/2);   // bottom hex vertex
    }

    std::vector<int> tris;
    tris.reserve(24 * 3);

    for (int i = 0; i < 6; i++) {
        int i1 = 1 + i;
        int i2 = 1 + (i + 1) % 6;
        int j1 = 8 + i;
        int j2 = 8 + (i + 1) % 6;

        // Top face: outward normal +z (CCW from above)
        tris.push_back(0); tris.push_back(i1); tris.push_back(i2);

        // Bottom face: outward normal -z (CCW from below = CW from above)
        tris.push_back(7); tris.push_back(j2); tris.push_back(j1);

        // Side face: two triangles with outward normals
        tris.push_back(i1); tris.push_back(j1); tris.push_back(j2);
        tris.push_back(i1); tris.push_back(j2); tris.push_back(i2);
    }

    Mesh m;
    m.verts = verts;
    m.tris = tris;

    for (int r = 0; r < refinements; r++)
        m = subdivide_flat(m);

    printf("  Hex prism: AR=%.2f, a=%.4f, H=%.4f, %d tris\n",
           aspect_ratio, a, H, m.nt());
    return m;
}

Mesh load_obj(const char* filename) {
    Mesh m;
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Error: cannot open OBJ file: %s\n", filename);
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            m.verts.push_back(Vec3(x, y, z));
        } else if (prefix == "f") {
            // Support "f v1 v2 v3" and "f v1/vt1/vn1 v2/... v3/..."
            std::vector<int> face_verts;
            std::string tok;
            while (iss >> tok) {
                int vi = std::atoi(tok.c_str()) - 1;  // OBJ is 1-based
                face_verts.push_back(vi);
            }
            // Triangulate polygon fan
            for (int i = 1; i + 1 < (int)face_verts.size(); i++) {
                m.tris.push_back(face_verts[0]);
                m.tris.push_back(face_verts[i]);
                m.tris.push_back(face_verts[i+1]);
            }
        }
    }

    printf("  Loaded OBJ: %d vertices, %d triangles from %s\n",
           m.nv(), m.nt(), filename);
    return m;
}
