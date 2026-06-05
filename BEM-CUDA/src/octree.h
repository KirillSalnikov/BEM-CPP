#ifndef BEM_OCTREE_H
#define BEM_OCTREE_H

#include "types.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unordered_map>

struct OctreeNode {
    double center[3];
    double half_size;
    int level;
    int index;
    int parent;         // -1 if root
    int children[8];    // -1 if no child
    bool is_leaf;

    // Grid coordinates for O(1) neighbor lookup
    int gx, gy, gz;

    // Points range in sorted array
    int pt_start, pt_count;

    // Flat near-list and far-list indices (into Octree::near_list / far_list)
    int near_start, near_count;
    int far_start,  far_count;
};

struct Octree {
    std::vector<OctreeNode> nodes;
    std::vector<int>        leaves;     // indices into nodes[]
    std::vector<int>        near_list;  // flat array of node indices
    std::vector<int>        far_list;   // flat array of node indices

    // Sorted point permutation: sorted_idx[i] = original point index
    std::vector<int> sorted_idx;

    // Sorted point positions (interleaved xyz)
    std::vector<double> sorted_pts;  // (N*3)

    int max_level;
    int N_total;  // number of points
    int N_tgt;    // first N_tgt points are targets, rest are sources

    // Level -> node index ranges
    struct LevelRange { int start, count; };
    std::vector<LevelRange> level_ranges;  // indexed by level

    // Node indices at each level (for iteration)
    std::vector<std::vector<int>> level_nodes;

    void build(const double* targets, int n_tgt,
               const double* sources, int n_src,
               int max_leaf = 64) {
        N_tgt = n_tgt;
        N_total = n_tgt + n_src;

        // Merge all points
        std::vector<double> all_pts(N_total * 3);
        memcpy(all_pts.data(), targets, n_tgt * 3 * sizeof(double));
        memcpy(all_pts.data() + n_tgt * 3, sources, n_src * 3 * sizeof(double));

        // Bounding box
        double pmin[3] = {1e30, 1e30, 1e30};
        double pmax[3] = {-1e30, -1e30, -1e30};
        for (int i = 0; i < N_total; i++) {
            for (int d = 0; d < 3; d++) {
                double v = all_pts[i*3+d];
                if (v < pmin[d]) pmin[d] = v;
                if (v > pmax[d]) pmax[d] = v;
            }
        }
        double center[3], hs = 0;
        for (int d = 0; d < 3; d++) {
            center[d] = 0.5 * (pmin[d] + pmax[d]);
            double ext = 0.5 * (pmax[d] - pmin[d]);
            if (ext > hs) hs = ext;
        }
        hs *= 1.001;  // small margin

        // Compute uniform depth
        int uniform_depth = 2;
        {
            double ratio = (double)N_total / max_leaf;
            if (ratio > 1.0)
                uniform_depth = std::max(2, (int)std::ceil(std::log(ratio) / std::log(8.0)));
            if (uniform_depth > 6) uniform_depth = 6;
        }

        // Initialize index array
        sorted_idx.resize(N_total);
        for (int i = 0; i < N_total; i++) sorted_idx[i] = i;

        // Build root
        nodes.clear();
        // Reserve enough for max depth 6: 1 + 8 + 64 + 512 + 4096 + 32768 + 262144
        int max_nodes = 1;
        for (int d = 0; d <= uniform_depth; d++) max_nodes += (1 << (3*d));
        nodes.reserve(max_nodes + 1);
        OctreeNode root;
        root.center[0] = center[0]; root.center[1] = center[1]; root.center[2] = center[2];
        root.half_size = hs;
        root.level = 0;
        root.index = 0;
        root.parent = -1;
        for (int i = 0; i < 8; i++) root.children[i] = -1;
        root.is_leaf = true;
        root.pt_start = 0;
        root.pt_count = N_total;
        root.gx = root.gy = root.gz = 0;
        root.near_start = root.near_count = 0;
        root.far_start = root.far_count = 0;
        nodes.push_back(root);

        // Recursive subdivision
        subdivide(0, all_pts.data(), uniform_depth);

        // Sort points by leaf for coalesced GPU access
        // Build final sorted arrays
        sorted_pts.resize(N_total * 3);
        std::vector<int> new_idx(N_total);

        int pos = 0;
        for (int li = 0; li < (int)nodes.size(); li++) {
            if (!nodes[li].is_leaf) continue;
            int start = nodes[li].pt_start;
            int count = nodes[li].pt_count;
            nodes[li].pt_start = pos;
            for (int j = 0; j < count; j++) {
                int orig = sorted_idx[start + j];
                new_idx[pos] = orig;
                sorted_pts[pos*3]   = all_pts[orig*3];
                sorted_pts[pos*3+1] = all_pts[orig*3+1];
                sorted_pts[pos*3+2] = all_pts[orig*3+2];
                pos++;
            }
        }
        sorted_idx = new_idx;

        // Collect leaves and levels
        max_level = 0;
        for (int i = 0; i < (int)nodes.size(); i++) {
            if (nodes[i].is_leaf)
                leaves.push_back(i);
            if (nodes[i].level > max_level)
                max_level = nodes[i].level;
        }

        level_nodes.resize(max_level + 1);
        for (int i = 0; i < (int)nodes.size(); i++)
            level_nodes[nodes[i].level].push_back(i);

        // Build interaction lists
        build_interaction_lists();

        printf("  [Octree] %d nodes, %d leaves, depth=%d, %d points\n",
               (int)nodes.size(), (int)leaves.size(), max_level, N_total);
    }

    // Get target point indices in a leaf (indices into sorted array)
    void leaf_targets(int leaf_idx, int& start, int& count) const {
        const OctreeNode& leaf = nodes[leaf_idx];
        start = -1; count = 0;
        for (int i = leaf.pt_start; i < leaf.pt_start + leaf.pt_count; i++) {
            if (sorted_idx[i] < N_tgt) {
                if (start < 0) start = i;
                count++;
            }
        }
        if (start < 0) start = leaf.pt_start;
    }

    // Get source point indices in a leaf (indices into sorted array)
    void leaf_sources(int leaf_idx, int& start, int& count) const {
        const OctreeNode& leaf = nodes[leaf_idx];
        start = -1; count = 0;
        for (int i = leaf.pt_start; i < leaf.pt_start + leaf.pt_count; i++) {
            if (sorted_idx[i] >= N_tgt) {
                if (start < 0) start = i;
                count++;
            }
        }
        if (start < 0) start = leaf.pt_start;
    }

private:
    void subdivide(int node_idx, const double* pts, int uniform_depth) {
        if (nodes[node_idx].level >= uniform_depth) return;

        // Copy node data to locals BEFORE any push_back (avoids reference invalidation)
        double node_center[3] = {nodes[node_idx].center[0], nodes[node_idx].center[1], nodes[node_idx].center[2]};
        int node_level = nodes[node_idx].level;
        int node_pt_start = nodes[node_idx].pt_start;
        int node_pt_count = nodes[node_idx].pt_count;
        int node_gx = nodes[node_idx].gx;
        int node_gy = nodes[node_idx].gy;
        int node_gz = nodes[node_idx].gz;

        nodes[node_idx].is_leaf = false;
        double hs = nodes[node_idx].half_size / 2.0;

        // Classify points into octants
        int octant_counts[8] = {};
        std::vector<int> octant_pts[8];
        for (int i = node_pt_start; i < node_pt_start + node_pt_count; i++) {
            int idx = sorted_idx[i];
            int oct = 0;
            if (pts[idx*3]   >= node_center[0]) oct |= 1;
            if (pts[idx*3+1] >= node_center[1]) oct |= 2;
            if (pts[idx*3+2] >= node_center[2]) oct |= 4;
            octant_pts[oct].push_back(idx);
            octant_counts[oct]++;
        }

        // Reorder sorted_idx by octant
        int pos = node_pt_start;
        int octant_starts[8];
        for (int o = 0; o < 8; o++) {
            octant_starts[o] = pos;
            for (int j = 0; j < (int)octant_pts[o].size(); j++)
                sorted_idx[pos++] = octant_pts[o][j];
        }

        // Create children (all 8 for uniform tree)
        for (int o = 0; o < 8; o++) {
            OctreeNode child;
            child.center[0] = node_center[0] + ((o & 1) ? hs : -hs);
            child.center[1] = node_center[1] + ((o & 2) ? hs : -hs);
            child.center[2] = node_center[2] + ((o & 4) ? hs : -hs);
            child.half_size = hs;
            child.level = node_level + 1;
            child.index = (int)nodes.size();
            child.parent = node_idx;
            for (int i = 0; i < 8; i++) child.children[i] = -1;
            child.is_leaf = true;
            child.pt_start = octant_starts[o];
            child.pt_count = octant_counts[o];
            child.gx = 2 * node_gx + ((o & 1) ? 1 : 0);
            child.gy = 2 * node_gy + ((o & 2) ? 1 : 0);
            child.gz = 2 * node_gz + ((o & 4) ? 1 : 0);
            child.near_start = child.near_count = 0;
            child.far_start = child.far_count = 0;

            nodes[node_idx].children[o] = child.index;
            nodes.push_back(child);
            subdivide(child.index, pts, uniform_depth);
        }
    }

    void build_interaction_lists() {
        near_list.clear();
        far_list.clear();

        // Hash function for (gx, gy, gz) grid coordinates
        struct GridHash {
            size_t operator()(int64_t k) const { return std::hash<int64_t>()(k); }
        };
        auto grid_key = [](int gx, int gy, int gz) -> int64_t {
            return ((int64_t)(gx + 100000) * 200001LL + (gy + 100000)) * 200001LL + (gz + 100000);
        };

        for (int level = 0; level <= max_level; level++) {
            const std::vector<int>& lvl = level_nodes[level];
            int n_lvl = (int)lvl.size();
            if (n_lvl == 0) continue;

            // Build spatial hash map for this level: grid_key -> node_index
            std::unordered_map<int64_t, int, GridHash> grid_map;
            grid_map.reserve(n_lvl * 2);
            for (int ii = 0; ii < n_lvl; ii++) {
                int ni = lvl[ii];
                grid_map[grid_key(nodes[ni].gx, nodes[ni].gy, nodes[ni].gz)] = ni;
            }

            for (int ii = 0; ii < n_lvl; ii++) {
                int ni = lvl[ii];
                OctreeNode& node_i = nodes[ni];
                int ix = node_i.gx, iy = node_i.gy, iz = node_i.gz;

                // Near list: check 3^3-1 = 26 neighbor positions (leaves only)
                if (node_i.is_leaf) {
                    node_i.near_start = (int)near_list.size();
                    node_i.near_count = 0;
                    for (int dx = -1; dx <= 1; dx++)
                    for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        auto it = grid_map.find(grid_key(ix+dx, iy+dy, iz+dz));
                        if (it != grid_map.end() && nodes[it->second].is_leaf) {
                            near_list.push_back(it->second);
                            node_i.near_count++;
                        }
                    }
                }

                // Far list (interaction list): children of parent's neighbors
                // that are NOT self-neighbors. Up to 6^3 - 3^3 = 189 candidates.
                node_i.far_start = (int)far_list.size();
                node_i.far_count = 0;

                if (level > 0) {
                    // Parent grid coord via integer floor division
                    int pix = (ix >= 0) ? ix / 2 : (ix - 1) / 2;
                    int piy = (iy >= 0) ? iy / 2 : (iy - 1) / 2;
                    int piz = (iz >= 0) ? iz / 2 : (iz - 1) / 2;

                    // Enumerate children of parent's 3^3 neighborhood
                    for (int dpx = -1; dpx <= 1; dpx++)
                    for (int dpy = -1; dpy <= 1; dpy++)
                    for (int dpz = -1; dpz <= 1; dpz++) {
                        int ppx = pix + dpx, ppy = piy + dpy, ppz = piz + dpz;
                        // 8 children of this parent-level cell
                        for (int cx = 0; cx <= 1; cx++)
                        for (int cy = 0; cy <= 1; cy++)
                        for (int cz = 0; cz <= 1; cz++) {
                            int jx = 2*ppx + cx, jy = 2*ppy + cy, jz = 2*ppz + cz;
                            // Skip self and neighbors
                            if (std::abs(jx-ix) <= 1 && std::abs(jy-iy) <= 1 && std::abs(jz-iz) <= 1)
                                continue;
                            auto it = grid_map.find(grid_key(jx, jy, jz));
                            if (it != grid_map.end()) {
                                far_list.push_back(it->second);
                                node_i.far_count++;
                            }
                        }
                    }
                }
            }
        }

        int total_near = 0, total_far = 0;
        for (int li : leaves) total_near += nodes[li].near_count;
        for (auto& nd : nodes) total_far += nd.far_count;
        printf("  [Octree] Near pairs: %d, Far (M2L) pairs: %d\n", total_near, total_far);
    }

    static double max_abs_diff(const double a[3], const double b[3]) {
        double d = 0;
        for (int i = 0; i < 3; i++) {
            double v = std::fabs(a[i] - b[i]);
            if (v > d) d = v;
        }
        return d;
    }
};

#endif // BEM_OCTREE_H
