#include "precond.h"
#include "bem_fmm.h"
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <string>

// ============================================================
// Spatial hash key for 3D integer cell coordinates
// ============================================================
struct CellKey {
    int ix, iy, iz;
    bool operator==(const CellKey& o) const {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

struct CellKeyHash {
    size_t operator()(const CellKey& c) const {
        size_t h = (size_t)c.ix * 73856093u ^ (size_t)c.iy * 19349663u ^ (size_t)c.iz * 83492791u;
        return h;
    }
};

// ============================================================
// Assemble near-field sparse matrix (used by ILU0)
// ============================================================
static void assemble_near_field(NearFieldPrecond& P, BemFmmOperator& op, double radius_mult)
{
    Timer timer;
    int N = P.N;
    int N2 = P.N2;
    int Nq = op.Nq;

    // Step 1: Compute RWG centers and average extent
    std::vector<double> centers(N * 3);
    double avg_extent = 0;

    for (int m = 0; m < N; m++) {
        double cx_val = 0, cy_val = 0, cz_val = 0;
        double max_x = -1e30, min_x = 1e30;
        double max_y = -1e30, min_y = 1e30;
        double max_z = -1e30, min_z = 1e30;
        for (int q = 0; q < Nq; q++) {
            double px = op.qpts_p[m*Nq*3 + q*3];
            double py = op.qpts_p[m*Nq*3 + q*3+1];
            double pz = op.qpts_p[m*Nq*3 + q*3+2];
            double mx = op.qpts_m[m*Nq*3 + q*3];
            double my = op.qpts_m[m*Nq*3 + q*3+1];
            double mz = op.qpts_m[m*Nq*3 + q*3+2];
            cx_val += px + mx;
            cy_val += py + my;
            cz_val += pz + mz;
            max_x = std::max(max_x, std::max(px, mx));
            min_x = std::min(min_x, std::min(px, mx));
            max_y = std::max(max_y, std::max(py, my));
            min_y = std::min(min_y, std::min(py, my));
            max_z = std::max(max_z, std::max(pz, mz));
            min_z = std::min(min_z, std::min(pz, mz));
        }
        double inv2Nq = 1.0 / (2 * Nq);
        centers[m*3]   = cx_val * inv2Nq;
        centers[m*3+1] = cy_val * inv2Nq;
        centers[m*3+2] = cz_val * inv2Nq;
        double ext = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
        avg_extent += ext;
    }
    avg_extent /= N;

    // Compute bounding box of RWG centers to set geometry-based minimum cell size
    double bb_min[3] = {1e30, 1e30, 1e30};
    double bb_max[3] = {-1e30, -1e30, -1e30};
    for (int m = 0; m < N; m++) {
        for (int d = 0; d < 3; d++) {
            bb_min[d] = std::min(bb_min[d], centers[m*3+d]);
            bb_max[d] = std::max(bb_max[d], centers[m*3+d]);
        }
    }
    double bb_max_dim = std::max({bb_max[0]-bb_min[0], bb_max[1]-bb_min[1], bb_max[2]-bb_min[2]});

    // Cell size: max of element-based and geometry-based
    // Element-based shrinks with refinement; geometry-based stays constant
    // ensuring consistent near-field coverage (~15%) regardless of mesh density
    double cell_size_elem = radius_mult * avg_extent;
    double cell_size_geom = bb_max_dim / 5.0;
    double cell_size = std::max(cell_size_elem, cell_size_geom);
    if (cell_size < 1e-15) cell_size = 1e-15;
    double inv_cell = 1.0 / cell_size;

    printf("  [Precond] Avg RWG extent=%.4e, cell_size=%.4e (elem=%.4e, geom=%.4e)\n",
           avg_extent, cell_size, cell_size_elem, cell_size_geom);

    // Step 2: Spatial hashing
    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> cell_map;
    cell_map.reserve(N);

    std::vector<CellKey> rwg_cell(N);
    for (int m = 0; m < N; m++) {
        CellKey key;
        key.ix = (int)std::floor(centers[m*3]   * inv_cell);
        key.iy = (int)std::floor(centers[m*3+1] * inv_cell);
        key.iz = (int)std::floor(centers[m*3+2] * inv_cell);
        rwg_cell[m] = key;
        cell_map[key].push_back(m);
    }

    printf("  [Precond] %d spatial cells\n", (int)cell_map.size());

    // Step 3: Build near-field pair list
    std::vector<std::vector<int>> nf_lists(N);

    for (int m = 0; m < N; m++) {
        std::unordered_set<int> nf_set;
        nf_set.insert(m);

        CellKey& ck = rwg_cell[m];
        for (int dix = -1; dix <= 1; dix++)
            for (int diy = -1; diy <= 1; diy++)
                for (int diz = -1; diz <= 1; diz++) {
                    CellKey nk = {ck.ix + dix, ck.iy + diy, ck.iz + diz};
                    auto it = cell_map.find(nk);
                    if (it != cell_map.end())
                        for (int n_idx : it->second)
                            nf_set.insert(n_idx);
                }

        for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++)
            nf_set.insert(op.corr_col_idx[jc]);

        nf_lists[m].assign(nf_set.begin(), nf_set.end());
        std::sort(nf_lists[m].begin(), nf_lists[m].end());
    }

    // Stats
    {
        long long total_nf = 0;
        int min_nf = N, max_nf = 0;
        for (int m = 0; m < N; m++) {
            int sz = (int)nf_lists[m].size();
            total_nf += sz;
            min_nf = std::min(min_nf, sz);
            max_nf = std::max(max_nf, sz);
        }
        printf("  [Precond] Near-field pairs per RWG: min=%d, max=%d, avg=%.1f\n",
               min_nf, max_nf, (double)total_nf / N);
        printf("  [Precond] Total NxN near-field entries: %lld\n", total_nf);
        printf("  [Precond] Coverage: %.1f%% of full N×N matrix\n",
               100.0 * total_nf / ((long long)N * N));
    }

    printf("  [Precond] Near-field detection: %.2fs\n", timer.elapsed_s());

    // Step 4: Build 2N×2N CSR
    Timer t_assemble;

    long long nnz_total = 0;
    for (int m = 0; m < N; m++)
        nnz_total += 2 * (long long)nf_lists[m].size();
    nnz_total *= 2;

    P.csr_row_ptr.resize(N2 + 1);
    P.csr_col_idx.resize(nnz_total);
    P.csr_val.assign(nnz_total, cdouble(0));
    P.diag_ptr.resize(N2);

    P.csr_row_ptr[0] = 0;
    for (int i = 0; i < N2; i++) {
        int m = (i < N) ? i : i - N;
        int nnz_row = 2 * (int)nf_lists[m].size();
        P.csr_row_ptr[i + 1] = P.csr_row_ptr[i] + nnz_row;
    }

    for (int i = 0; i < N2; i++) {
        int m = (i < N) ? i : i - N;
        int base = P.csr_row_ptr[i];
        int nnz_half = (int)nf_lists[m].size();
        for (int j = 0; j < nnz_half; j++)
            P.csr_col_idx[base + j] = nf_lists[m][j];
        for (int j = 0; j < nnz_half; j++)
            P.csr_col_idx[base + nnz_half + j] = nf_lists[m][j] + N;

        P.diag_ptr[i] = -1;
        for (int j = P.csr_row_ptr[i]; j < P.csr_row_ptr[i + 1]; j++) {
            if (P.csr_col_idx[j] == i) {
                P.diag_ptr[i] = j;
                break;
            }
        }
        if (P.diag_ptr[i] < 0)
            fprintf(stderr, "  [Precond] ERROR: diagonal not found for row %d\n", i);
    }

    printf("  [Precond] 2N×2N CSR: %lld nonzeros (%.1f nnz/row avg)\n",
           nnz_total, (double)nnz_total / N2);

    // Step 5: Assemble entries via quadrature
    double inv4pi = 1.0 / (4.0 * M_PI);
    cdouble k_vals[2] = {op.k_ext, op.k_int};
    cdouble eta_e = op.eta_ext, eta_i = op.eta_int;

    // Lookup: nf_lists[m] -> position
    std::vector<std::unordered_map<int,int>> nf_pos(N);
    for (int m = 0; m < N; m++) {
        nf_pos[m].reserve(nf_lists[m].size());
        for (int j = 0; j < (int)nf_lists[m].size(); j++)
            nf_pos[m][nf_lists[m][j]] = j;
    }

    auto csr_idx = [&](int row, int col) -> int {
        int m = (row < N) ? row : row - N;
        int base = P.csr_row_ptr[row];
        int nnz_half = (int)nf_lists[m].size();

        if (col < N) {
            auto it = nf_pos[m].find(col);
            if (it != nf_pos[m].end()) return base + it->second;
        } else {
            auto it = nf_pos[m].find(col - N);
            if (it != nf_pos[m].end()) return base + nnz_half + it->second;
        }
        return -1;
    };

    for (int m = 0; m < N; m++) {
        for (int jn = 0; jn < (int)nf_lists[m].size(); jn++) {
            int n_idx = nf_lists[m][jn];

            cdouble L_vals_k[2] = {0, 0};
            cdouble K_vals_k[2] = {0, 0};

            for (int hm = 0; hm < 2; hm++) {
                const double* qm = (hm == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
                const double* fm = (hm == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
                double dm = (hm == 0) ? op.div_p[m] : op.div_m[m];
                const double* jwm = (hm == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

                for (int hn = 0; hn < 2; hn++) {
                    const double* qn = (hn == 0) ? &op.qpts_p[n_idx * Nq * 3] : &op.qpts_m[n_idx * Nq * 3];
                    const double* fn = (hn == 0) ? &op.f_p[n_idx * Nq * 3] : &op.f_m[n_idx * Nq * 3];
                    double dn = (hn == 0) ? op.div_p[n_idx] : op.div_m[n_idx];
                    const double* jwn = (hn == 0) ? &op.jw_p[n_idx * Nq] : &op.jw_m[n_idx * Nq];

                    for (int qi = 0; qi < Nq; qi++) {
                        double rx = qm[qi*3], ry = qm[qi*3+1], rz = qm[qi*3+2];
                        double fxm = fm[qi*3], fym = fm[qi*3+1], fzm = fm[qi*3+2];
                        double wm_val = jwm[qi];

                        for (int qj = 0; qj < Nq; qj++) {
                            double dx = rx - qn[qj*3];
                            double dy = ry - qn[qj*3+1];
                            double dz = rz - qn[qj*3+2];
                            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                            double wn_val = jwn[qj];
                            double ww = wm_val * wn_val;

                            double fxn = fn[qj*3], fyn = fn[qj*3+1], fzn = fn[qj*3+2];
                            double f_dot = fxm*fxn + fym*fyn + fzm*fzn;

                            for (int ki = 0; ki < 2; ki++) {
                                cdouble kv = k_vals[ki];
                                cdouble ik = cdouble(0, 1) * kv;
                                cdouble iok = cdouble(0, 1) / kv;

                                if (R > 1e-12) {
                                    cdouble G = std::exp(ik * R) * inv4pi / R;
                                    L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G * ww;

                                    cdouble gG = G * (ik - 1.0/R) / R;
                                    double cross_x = dy*fzn - dz*fyn;
                                    double cross_y = dz*fxn - dx*fzn;
                                    double cross_z = dx*fyn - dy*fxn;
                                    K_vals_k[ki] += gG * (fxm*cross_x + fym*cross_y + fzm*cross_z) * ww;
                                } else {
                                    cdouble G0 = ik * inv4pi;
                                    L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G0 * ww;
                                }
                            }
                        }
                    }
                }
            }

            // Add singular corrections
            for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++) {
                if (op.corr_col_idx[jc] == n_idx) {
                    L_vals_k[0] += op.corr_L_ext_val[jc];
                    K_vals_k[0] += op.corr_K_ext_val[jc];
                    L_vals_k[1] += op.corr_L_int_val[jc];
                    K_vals_k[1] += op.corr_K_int_val[jc];
                }
            }

            // PMCHWT 2x2 block entries
            cdouble A_mn = eta_e * L_vals_k[0] + eta_i * L_vals_k[1];     // JJ
            cdouble B_mn = -(K_vals_k[0] + K_vals_k[1]);                  // JM
            cdouble C_mn = K_vals_k[0] + K_vals_k[1];                     // MJ
            cdouble D_mn = L_vals_k[0] / eta_e + L_vals_k[1] / eta_i;    // MM

            int idx_jj = csr_idx(m, n_idx);
            if (idx_jj >= 0) P.csr_val[idx_jj] = A_mn;

            int idx_jm = csr_idx(m, n_idx + N);
            if (idx_jm >= 0) P.csr_val[idx_jm] = B_mn;

            int idx_mj = csr_idx(m + N, n_idx);
            if (idx_mj >= 0) P.csr_val[idx_mj] = C_mn;

            int idx_mm = csr_idx(m + N, n_idx + N);
            if (idx_mm >= 0) P.csr_val[idx_mm] = D_mn;
        }

        if (m > 0 && m % (N / 10 + 1) == 0)
            printf("  [Precond] Assembly: %d/%d RWG rows (%.0f%%)\n",
                   m, N, 100.0 * m / N);
    }

    printf("  [Precond] Assembly done: %.2fs\n", t_assemble.elapsed_s());

    // Diagnostic: print diagonal statistics
    double dmin = 1e30, dmax = 0, dmean = 0;
    for (int i = 0; i < N2; i++) {
        double d = std::abs(P.csr_val[P.diag_ptr[i]]);
        dmin = std::min(dmin, d);
        dmax = std::max(dmax, d);
        dmean += d;
    }
    dmean /= N2;
    printf("  [Precond] Diagonal |Z_ii|: min=%.4e, max=%.4e, mean=%.4e, ratio=%.1f\n",
           dmin, dmax, dmean, dmax/dmin);
}

// ============================================================
// ILU(0) factorization
// ============================================================
static void do_ilu0(NearFieldPrecond& P)
{
    Timer t_ilu;
    int N2 = P.N2;

    // Build per-row hash maps for column lookup
    std::vector<std::unordered_map<int,int>> row_col_map(N2);
    for (int i = 0; i < N2; i++) {
        int rs = P.csr_row_ptr[i], re = P.csr_row_ptr[i + 1];
        row_col_map[i].reserve(re - rs);
        for (int j = rs; j < re; j++)
            row_col_map[i][P.csr_col_idx[j]] = j;
    }

    printf("  [Precond] Starting ILU(0) factorization...\n");

    int zero_diag_count = 0;

    for (int i = 0; i < N2; i++) {
        int rs = P.csr_row_ptr[i];
        int re = P.csr_row_ptr[i + 1];

        for (int p = rs; p < re; p++) {
            int k = P.csr_col_idx[p];
            if (k >= i) break;

            cdouble akk = P.csr_val[P.diag_ptr[k]];
            if (std::abs(akk) < 1e-30) {
                zero_diag_count++;
                akk = cdouble(1e-15);
            }
            cdouble a_ik = P.csr_val[p] / akk;
            P.csr_val[p] = a_ik;

            for (int q = p + 1; q < re; q++) {
                int j = P.csr_col_idx[q];
                auto it = row_col_map[k].find(j);
                if (it != row_col_map[k].end())
                    P.csr_val[q] -= a_ik * P.csr_val[it->second];
            }
        }

        if (std::abs(P.csr_val[P.diag_ptr[i]]) < 1e-30) {
            zero_diag_count++;
            P.csr_val[P.diag_ptr[i]] = cdouble(1e-15);
        }

        if (i > 0 && i % (N2 / 10 + 1) == 0)
            printf("  [Precond] ILU(0): %d/%d rows (%.0f%%)\n",
                   i, N2, 100.0 * i / N2);
    }

    if (zero_diag_count > 0)
        printf("  [Precond] WARNING: %d near-zero diagonal entries\n", zero_diag_count);

    printf("  [Precond] ILU(0) factorization done: %.2fs\n", t_ilu.elapsed_s());
}

// Default max RWG per block for Block-Jacobi.
// Blocks larger than this are split by longest-axis bisection so that dense LU stays tractable.
// 2B×2B dense LU: B=1500 → 3000×3000 → 144 MB per block, O(27G) flops.
static const int DEFAULT_MAX_BLOCK_RWG = 1500;

// ============================================================
// Block-Jacobi: spatial cell blocking with dense LU per block
// ============================================================
static void do_block_jacobi(NearFieldPrecond& P, BemFmmOperator& op, double radius_mult,
                            int max_block_rwg, int overlap_layers)
{
    Timer timer;
    int N = P.N;
    int Nq = op.Nq;

    // Step 1: Compute RWG centers
    std::vector<double> centers(N * 3);
    double avg_extent = 0;
    for (int m = 0; m < N; m++) {
        double cx_val = 0, cy_val = 0, cz_val = 0;
        double max_x = -1e30, min_x = 1e30;
        double max_y = -1e30, min_y = 1e30;
        double max_z = -1e30, min_z = 1e30;
        for (int q = 0; q < Nq; q++) {
            double px = op.qpts_p[m*Nq*3 + q*3];
            double py = op.qpts_p[m*Nq*3 + q*3+1];
            double pz = op.qpts_p[m*Nq*3 + q*3+2];
            double mx_v = op.qpts_m[m*Nq*3 + q*3];
            double my_v = op.qpts_m[m*Nq*3 + q*3+1];
            double mz_v = op.qpts_m[m*Nq*3 + q*3+2];
            cx_val += px + mx_v;
            cy_val += py + my_v;
            cz_val += pz + mz_v;
            max_x = std::max(max_x, std::max(px, mx_v));
            min_x = std::min(min_x, std::min(px, mx_v));
            max_y = std::max(max_y, std::max(py, my_v));
            min_y = std::min(min_y, std::min(py, my_v));
            max_z = std::max(max_z, std::max(pz, mz_v));
            min_z = std::min(min_z, std::min(pz, mz_v));
        }
        double inv2Nq = 1.0 / (2 * Nq);
        centers[m*3]   = cx_val * inv2Nq;
        centers[m*3+1] = cy_val * inv2Nq;
        centers[m*3+2] = cz_val * inv2Nq;
        double ext = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
        avg_extent += ext;
    }
    avg_extent /= N;

    // Bounding box for geometry-based cell size
    double bb_min[3] = {1e30, 1e30, 1e30};
    double bb_max[3] = {-1e30, -1e30, -1e30};
    for (int m = 0; m < N; m++) {
        for (int d = 0; d < 3; d++) {
            bb_min[d] = std::min(bb_min[d], centers[m*3+d]);
            bb_max[d] = std::max(bb_max[d], centers[m*3+d]);
        }
    }
    double bb_max_dim = std::max({bb_max[0]-bb_min[0], bb_max[1]-bb_min[1], bb_max[2]-bb_min[2]});

    // For blockj: larger cells = bigger blocks = better preconditioning.
    // Target ~10-30 blocks total (each block 50-300 RWG).
    // Use radius_mult to scale: default 2.0 → ~10-20 blocks.
    double cell_size = bb_max_dim / (2.5 / radius_mult);
    if (cell_size < avg_extent) cell_size = avg_extent;
    if (cell_size < 1e-15) cell_size = 1e-15;
    double inv_cell = 1.0 / cell_size;

    // Step 2: Hash RWG into spatial cells -> blocks
    // Offset by bb_min so cell boundaries align with bounding box edge
    // (avoids splitting across origin for centered geometries)
    std::unordered_map<CellKey, std::vector<int>, CellKeyHash> cell_map;
    cell_map.reserve(N);
    for (int m = 0; m < N; m++) {
        CellKey key;
        key.ix = (int)std::floor((centers[m*3]   - bb_min[0]) * inv_cell);
        key.iy = (int)std::floor((centers[m*3+1] - bb_min[1]) * inv_cell);
        key.iz = (int)std::floor((centers[m*3+2] - bb_min[2]) * inv_cell);
        cell_map[key].push_back(m);
    }

    // Convert to block list
    std::vector<std::vector<int>> block_list;
    block_list.reserve(cell_map.size());
    for (auto& kv : cell_map) {
        auto& members = kv.second;
        std::sort(members.begin(), members.end());
        block_list.push_back(std::move(members));
    }

    // Step 2b: Adaptive splitting — bisect blocks exceeding max_block_rwg
    {
        bool did_split = true;
        int split_rounds = 0;
        while (did_split) {
            did_split = false;
            std::vector<std::vector<int>> new_list;
            new_list.reserve(block_list.size());
            for (auto& blk : block_list) {
                if ((int)blk.size() <= max_block_rwg) {
                    new_list.push_back(std::move(blk));
                    continue;
                }
                did_split = true;
                // Find longest axis of this block's RWG centers
                double bmin[3] = {1e30,1e30,1e30}, bmax[3] = {-1e30,-1e30,-1e30};
                for (int m : blk) {
                    for (int d = 0; d < 3; d++) {
                        bmin[d] = std::min(bmin[d], centers[m*3+d]);
                        bmax[d] = std::max(bmax[d], centers[m*3+d]);
                    }
                }
                int ax = 0;
                double max_range = bmax[0] - bmin[0];
                for (int d = 1; d < 3; d++) {
                    double rng = bmax[d] - bmin[d];
                    if (rng > max_range) { max_range = rng; ax = d; }
                }
                double mid = 0.5 * (bmin[ax] + bmax[ax]);

                std::vector<int> left, right;
                for (int m : blk) {
                    if (centers[m*3+ax] < mid)
                        left.push_back(m);
                    else
                        right.push_back(m);
                }
                // Degenerate case: all on one side
                if (left.empty() || right.empty()) {
                    new_list.push_back(std::move(blk));
                } else {
                    new_list.push_back(std::move(left));
                    new_list.push_back(std::move(right));
                }
            }
            block_list = std::move(new_list);
            split_rounds++;
            if (split_rounds > 20) break;  // safety
        }
    }

    // Populate P block arrays
    P.n_blocks = (int)block_list.size();
    P.block_sizes.resize(P.n_blocks);
    P.block_rwg.resize(P.n_blocks);
    P.rwg_block.resize(N);
    P.rwg_local.resize(N);

    for (int b = 0; b < P.n_blocks; b++) {
        P.block_rwg[b] = std::move(block_list[b]);
        P.block_sizes[b] = (int)P.block_rwg[b].size();
        for (int j = 0; j < P.block_sizes[b]; j++) {
            P.rwg_block[P.block_rwg[b][j]] = b;
            P.rwg_local[P.block_rwg[b][j]] = j;
        }
    }

    // Step 2c: Compute overlap (RAS) — extend each block with neighboring RWGs
    P.overlap_layers = overlap_layers;
    P.block_sizes_ext.resize(P.n_blocks);
    P.block_rwg_ext.resize(P.n_blocks);

    if (overlap_layers > 0) {
        // Use distance-based overlap: overlap_dist scales with element size (avg_extent)
        // Each layer adds ~2 element widths of overlap around each block
        double overlap_dist = overlap_layers * avg_extent * 2.0;

        for (int bi = 0; bi < P.n_blocks; bi++) {
            const auto& own = P.block_rwg[bi];
            int B_own = P.block_sizes[bi];

            // Compute bounding box of this block
            double bmin[3] = {1e30,1e30,1e30}, bmax[3] = {-1e30,-1e30,-1e30};
            for (int m : own) {
                for (int d = 0; d < 3; d++) {
                    bmin[d] = std::min(bmin[d], centers[m*3+d]);
                    bmax[d] = std::max(bmax[d], centers[m*3+d]);
                }
            }
            // Expand bbox by overlap_dist
            for (int d = 0; d < 3; d++) {
                bmin[d] -= overlap_dist;
                bmax[d] += overlap_dist;
            }

            // Find all RWGs from other blocks within expanded bbox
            std::vector<int> overlap_rwg;
            for (int m = 0; m < N; m++) {
                if (P.rwg_block[m] == bi) continue; // skip own
                double cx_v = centers[m*3], cy_v = centers[m*3+1], cz_v = centers[m*3+2];
                if (cx_v >= bmin[0] && cx_v <= bmax[0] &&
                    cy_v >= bmin[1] && cy_v <= bmax[1] &&
                    cz_v >= bmin[2] && cz_v <= bmax[2]) {
                    overlap_rwg.push_back(m);
                }
            }
            std::sort(overlap_rwg.begin(), overlap_rwg.end());

            // Extended block: own first, then overlap
            P.block_rwg_ext[bi].resize(B_own + (int)overlap_rwg.size());
            for (int j = 0; j < B_own; j++)
                P.block_rwg_ext[bi][j] = own[j];
            for (int j = 0; j < (int)overlap_rwg.size(); j++)
                P.block_rwg_ext[bi][B_own + j] = overlap_rwg[j];
            P.block_sizes_ext[bi] = (int)P.block_rwg_ext[bi].size();
        }

        // Print overlap stats
        int min_ext = N, max_ext = 0, total_ovl = 0;
        for (int bi = 0; bi < P.n_blocks; bi++) {
            int ext = P.block_sizes_ext[bi];
            min_ext = std::min(min_ext, ext);
            max_ext = std::max(max_ext, ext);
            total_ovl += ext - P.block_sizes[bi];
        }
        printf("  [BlockJ-RAS] overlap_layers=%d, overlap_dist=%.4e\n",
               overlap_layers, overlap_dist);
        printf("  [BlockJ-RAS] Extended sizes: min=%d, max=%d (own: min=%d, max=%d)\n",
               min_ext, max_ext,
               *std::min_element(P.block_sizes.begin(), P.block_sizes.end()),
               *std::max_element(P.block_sizes.begin(), P.block_sizes.end()));
        printf("  [BlockJ-RAS] Total overlap RWGs: %d (%.1f%% extra)\n",
               total_ovl, 100.0 * total_ovl / N);
    } else {
        // No overlap: extended = own
        for (int bi = 0; bi < P.n_blocks; bi++) {
            P.block_rwg_ext[bi] = P.block_rwg[bi];
            P.block_sizes_ext[bi] = P.block_sizes[bi];
        }
    }

    // Stats
    int min_bs = N, max_bs = 0;
    long long total_dense = 0;
    for (int i = 0; i < P.n_blocks; i++) {
        min_bs = std::min(min_bs, P.block_sizes[i]);
        max_bs = std::max(max_bs, P.block_sizes[i]);
        int bs2 = 2 * P.block_sizes_ext[i]; // LU size uses extended
        total_dense += (long long)bs2 * bs2;
    }

    printf("  [BlockJ] %d blocks, sizes: min=%d, max=%d, avg=%.1f RWG\n",
           P.n_blocks, min_bs, max_bs, (double)N / P.n_blocks);
    printf("  [BlockJ] Total dense storage: %.1f MB\n", total_dense * 16.0 / 1e6);
    printf("  [BlockJ] cell_size=%.4e, avg_extent=%.4e\n", cell_size, avg_extent);

    printf("  [BlockJ] Spatial hashing: %.2fs\n", timer.elapsed_s());

    // Step 3: Assemble + factorize each block
    Timer t_assemble;
    double inv4pi = 1.0 / (4.0 * M_PI);
    cdouble k_vals[2] = {op.k_ext, op.k_int};
    cdouble eta_e = op.eta_ext, eta_i = op.eta_int;

    P.block_lu.resize(P.n_blocks);
    P.block_piv.resize(P.n_blocks);

    int blocks_done = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < P.n_blocks; bi++) {
        int B_ext = P.block_sizes_ext[bi]; // extended size (own + overlap)
        int B2 = 2 * B_ext;
        const auto& rwg = P.block_rwg_ext[bi]; // extended RWG list

        // Assemble 2B_ext×2B_ext dense block (column-major)
        std::vector<cdouble> blk(B2 * B2, cdouble(0));

        for (int im = 0; im < B_ext; im++) {
            int m = rwg[im];
            for (int jn = 0; jn < B_ext; jn++) {
                int n_idx = rwg[jn];

                cdouble L_vals_k[2] = {0, 0};
                cdouble K_vals_k[2] = {0, 0};

                for (int hm = 0; hm < 2; hm++) {
                    const double* qm = (hm == 0) ? &op.qpts_p[m*Nq*3] : &op.qpts_m[m*Nq*3];
                    const double* fm = (hm == 0) ? &op.f_p[m*Nq*3] : &op.f_m[m*Nq*3];
                    double dm = (hm == 0) ? op.div_p[m] : op.div_m[m];
                    const double* jwm = (hm == 0) ? &op.jw_p[m*Nq] : &op.jw_m[m*Nq];

                    for (int hn = 0; hn < 2; hn++) {
                        const double* qn = (hn == 0) ? &op.qpts_p[n_idx*Nq*3] : &op.qpts_m[n_idx*Nq*3];
                        const double* fn = (hn == 0) ? &op.f_p[n_idx*Nq*3] : &op.f_m[n_idx*Nq*3];
                        double dn = (hn == 0) ? op.div_p[n_idx] : op.div_m[n_idx];
                        const double* jwn = (hn == 0) ? &op.jw_p[n_idx*Nq] : &op.jw_m[n_idx*Nq];

                        for (int qi = 0; qi < Nq; qi++) {
                            double rx = qm[qi*3], ry = qm[qi*3+1], rz = qm[qi*3+2];
                            double fxm = fm[qi*3], fym = fm[qi*3+1], fzm = fm[qi*3+2];
                            double wm_val = jwm[qi];

                            for (int qj = 0; qj < Nq; qj++) {
                                double dx = rx - qn[qj*3];
                                double dy = ry - qn[qj*3+1];
                                double dz = rz - qn[qj*3+2];
                                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                                double wn_val = jwn[qj];
                                double ww = wm_val * wn_val;

                                double fxn = fn[qj*3], fyn = fn[qj*3+1], fzn = fn[qj*3+2];
                                double f_dot = fxm*fxn + fym*fyn + fzm*fzn;

                                for (int ki = 0; ki < 2; ki++) {
                                    cdouble kv = k_vals[ki];
                                    cdouble ik = cdouble(0, 1) * kv;
                                    cdouble iok = cdouble(0, 1) / kv;

                                    if (R > 1e-12) {
                                        cdouble G = std::exp(ik * R) * inv4pi / R;
                                        L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G * ww;
                                        cdouble gG = G * (ik - 1.0/R) / R;
                                        double cross_x = dy*fzn - dz*fyn;
                                        double cross_y = dz*fxn - dx*fzn;
                                        double cross_z = dx*fyn - dy*fxn;
                                        K_vals_k[ki] += gG * (fxm*cross_x + fym*cross_y + fzm*cross_z) * ww;
                                    } else {
                                        cdouble G0 = ik * inv4pi;
                                        L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G0 * ww;
                                    }
                                }
                            }
                        }
                    }
                }

                // Singular corrections
                for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m+1]; jc++) {
                    if (op.corr_col_idx[jc] == n_idx) {
                        L_vals_k[0] += op.corr_L_ext_val[jc];
                        K_vals_k[0] += op.corr_K_ext_val[jc];
                        L_vals_k[1] += op.corr_L_int_val[jc];
                        K_vals_k[1] += op.corr_K_int_val[jc];
                    }
                }

                // PMCHWT 2×2 block entries (column-major: [row + col*B2])
                cdouble A_mn = eta_e * L_vals_k[0] + eta_i * L_vals_k[1];    // JJ
                cdouble B_mn = -(K_vals_k[0] + K_vals_k[1]);                 // JM
                cdouble C_mn = K_vals_k[0] + K_vals_k[1];                    // MJ
                cdouble D_mn = L_vals_k[0] / eta_e + L_vals_k[1] / eta_i;   // MM

                blk[im         + jn         * B2] = A_mn;  // JJ block
                blk[im         + (jn+B_ext) * B2] = B_mn;  // JM block
                blk[(im+B_ext) + jn         * B2] = C_mn;  // MJ block
                blk[(im+B_ext) + (jn+B_ext) * B2] = D_mn;  // MM block
            }
        }

        // LU factorize with partial pivoting
        std::vector<int> piv(B2);
        for (int k = 0; k < B2; k++) {
            // Find pivot
            int max_idx = k;
            double max_val = std::abs(blk[k + k * B2]);
            for (int i = k + 1; i < B2; i++) {
                double v = std::abs(blk[i + k * B2]);
                if (v > max_val) { max_val = v; max_idx = i; }
            }
            piv[k] = max_idx;

            if (max_idx != k) {
                for (int j = 0; j < B2; j++)
                    std::swap(blk[k + j * B2], blk[max_idx + j * B2]);
            }

            cdouble akk = blk[k + k * B2];
            if (std::abs(akk) < 1e-30) akk = cdouble(1e-15);

            for (int i = k + 1; i < B2; i++) {
                cdouble factor = blk[i + k * B2] / akk;
                blk[i + k * B2] = factor;
                for (int j = k + 1; j < B2; j++)
                    blk[i + j * B2] -= factor * blk[k + j * B2];
            }
        }

        P.block_lu[bi] = std::move(blk);
        P.block_piv[bi] = std::move(piv);

        #pragma omp atomic
        blocks_done++;

        if (blocks_done % std::max(1, P.n_blocks / 5) == 0) {
            #pragma omp critical
            printf("  [BlockJ] Assembled+factored %d/%d blocks (%.0f%%)\n",
                   blocks_done, P.n_blocks, 100.0 * blocks_done / P.n_blocks);
        }
    }

    printf("  [BlockJ] Assembly+factorize: %.2fs\n", t_assemble.elapsed_s());
}

// ============================================================
// Build preconditioner
// ============================================================
void NearFieldPrecond::build(BemFmmOperator& op, PrecondMode pm, double radius_mult,
                            int max_block_rwg, int overlap)
{
    Timer timer;
    N = op.N;
    N2 = 2 * N;
    mode = pm;
    gpu_ready = false;
    d_workspace = nullptr;
    d_blk_B_ext = nullptr;

    const char* mode_name[] = {"NONE", "ILU0", "BLOCKJ"};
    printf("  [Precond] Building %s preconditioner (N=%d, system_size=%d, radius=%.1f%s)...\n",
           mode_name[mode], N, N2, radius_mult,
           overlap > 0 ? (", overlap=" + std::to_string(overlap)).c_str() : "");

    if (mode == PREC_NONE) return;

    if (mode == PREC_BLOCKJ) {
        do_block_jacobi(*this, op, radius_mult, max_block_rwg, overlap);
        upload_to_gpu();
        printf("  [Precond] %s preconditioner built: %.2fs total\n",
               mode_name[mode], timer.elapsed_s());
        return;
    }

    // Assemble near-field sparse matrix (needed for ILU0)
    assemble_near_field(*this, op, radius_mult);

    // ILU(0) factorization
    if (mode == PREC_ILU0)
        do_ilu0(*this);

    printf("  [Precond] %s preconditioner built: %.2fs total\n",
           mode_name[mode], timer.elapsed_s());
}

// ============================================================
// Apply preconditioner: z = M^{-1} * r
// ============================================================
void NearFieldPrecond::apply(const cdouble* r, cdouble* z) const
{
    switch (mode) {
        case PREC_NONE:
            // Identity: z = r
            for (int i = 0; i < N2; i++)
                z[i] = r[i];
            break;

        case PREC_ILU0:
            // ILU forward/backward solve
            for (int i = 0; i < N2; i++)
                z[i] = r[i];

            // Forward solve: L * y = r (L has unit diagonal)
            for (int i = 0; i < N2; i++) {
                int rs = csr_row_ptr[i];
                int dp = diag_ptr[i];
                for (int p = rs; p < dp; p++)
                    z[i] -= csr_val[p] * z[csr_col_idx[p]];
            }

            // Backward solve: U * z = y
            for (int i = N2 - 1; i >= 0; i--) {
                int dp = diag_ptr[i];
                int re = csr_row_ptr[i + 1];
                for (int p = dp + 1; p < re; p++)
                    z[i] -= csr_val[p] * z[csr_col_idx[p]];
                z[i] /= csr_val[dp];
            }
            break;

        case PREC_BLOCKJ:
        {
            if (gpu_ready) {
                // GPU path: upload r, kernel, download z
                CUDA_CHECK(cudaMemcpy(d_buf_r, r, N2 * sizeof(cuDoubleComplex),
                                      cudaMemcpyHostToDevice));
                apply_gpu(d_buf_r, d_buf_z);
                CUDA_CHECK(cudaMemcpy(z, d_buf_z, N2 * sizeof(cuDoubleComplex),
                                      cudaMemcpyDeviceToHost));
            } else {
                // CPU fallback
                for (int i = 0; i < N2; i++)
                    z[i] = cdouble(0);

                #pragma omp parallel for schedule(dynamic)
                for (int bi = 0; bi < n_blocks; bi++) {
                    int B_own = block_sizes[bi];
                    int B_ext = block_sizes_ext[bi];
                    int B2 = 2 * B_ext;
                    const auto& rwg_ext = block_rwg_ext[bi];
                    const auto& lu = block_lu[bi];
                    const auto& piv = block_piv[bi];

                    // Gather extended block from r
                    std::vector<cdouble> tmp(B2);
                    for (int j = 0; j < B_ext; j++) {
                        tmp[j]       = r[rwg_ext[j]];
                        tmp[j+B_ext] = r[rwg_ext[j] + N];
                    }
                    // Permute
                    for (int k = 0; k < B2; k++) {
                        if (piv[k] != k)
                            std::swap(tmp[k], tmp[piv[k]]);
                    }
                    // Forward solve
                    for (int i = 1; i < B2; i++) {
                        for (int j = 0; j < i; j++)
                            tmp[i] -= lu[i + j * B2] * tmp[j];
                    }
                    // Backward solve
                    for (int i = B2 - 1; i >= 0; i--) {
                        for (int j = i + 1; j < B2; j++)
                            tmp[i] -= lu[i + j * B2] * tmp[j];
                        tmp[i] /= lu[i + i * B2];
                    }
                    // Scatter only own RWGs (RAS: restricted)
                    for (int j = 0; j < B_own; j++) {
                        z[block_rwg[bi][j]]     = tmp[j];
                        z[block_rwg[bi][j] + N] = tmp[j+B_ext];
                    }
                }
            }
            break;
        }
    }
}

// ============================================================
// GPU Block-Jacobi apply kernel
//
// Each CUDA block handles one preconditioner block using a single warp
// (32 threads).  Warp-synchronous execution — no __syncthreads needed.
// LU data is stored row-major on GPU for coalesced memory access.
// ============================================================

// RAS Block-Jacobi kernel:
// - Gathers B_ext values (own + overlap)
// - Solves 2*B_ext system via LU
// - Scatters only B_own values (restricted Additive Schwarz)
__global__ void blockj_apply_kernel(
    const cuDoubleComplex* __restrict__ r,
    cuDoubleComplex* __restrict__ z,
    const cuDoubleComplex* __restrict__ lu_flat,
    const int* __restrict__ piv_flat,
    const int* __restrict__ rwg_flat,
    const int* __restrict__ blk_B,
    const int* __restrict__ blk_B_ext,
    const int* __restrict__ lu_off,
    const int* __restrict__ piv_off,
    const int* __restrict__ rwg_off,
    cuDoubleComplex* __restrict__ workspace,
    int max_B2,
    int n_blocks, int N)
{
    int bi = blockIdx.x;
    if (bi >= n_blocks) return;

    int B_own = blk_B[bi];
    int B_ext = blk_B_ext[bi];
    int B2 = 2 * B_ext;
    int tid = threadIdx.x;       // 0..31

    const cuDoubleComplex* lu  = lu_flat  + lu_off[bi];
    const int*             piv = piv_flat + piv_off[bi];
    const int*             rwg = rwg_flat + rwg_off[bi];

    // Working vector in global memory (no shared memory limit)
    cuDoubleComplex* tmp = workspace + bi * max_B2;

    // Gather: load from global r (extended block)
    for (int j = tid; j < B_ext; j += 32) {
        tmp[j]       = r[rwg[j]];
        tmp[j+B_ext] = r[rwg[j] + N];
    }
    __syncwarp();

    // Permute (lane 0 only)
    if (tid == 0) {
        for (int k = 0; k < B2; k++) {
            int pk = piv[k];
            if (pk != k) {
                cuDoubleComplex t = tmp[k];
                tmp[k]  = tmp[pk];
                tmp[pk] = t;
            }
        }
    }
    __syncwarp();

    // Forward solve: L * y = Pr
    for (int i = 1; i < B2; i++) {
        double sr = 0.0, si = 0.0;
        for (int j = tid; j < i; j += 32) {
            cuDoubleComplex Lij = lu[i * B2 + j];  // row-major
            cuDoubleComplex tj  = tmp[j];
            sr += Lij.x * tj.x - Lij.y * tj.y;
            si += Lij.x * tj.y + Lij.y * tj.x;
        }
        // Warp shuffle reduction
        for (int s = 16; s > 0; s >>= 1) {
            sr += __shfl_down_sync(0xFFFFFFFF, sr, s);
            si += __shfl_down_sync(0xFFFFFFFF, si, s);
        }
        if (tid == 0) {
            tmp[i].x -= sr;
            tmp[i].y -= si;
        }
        __syncwarp();
    }

    // Backward solve: U * z = y
    for (int i = B2 - 1; i >= 0; i--) {
        double sr = 0.0, si = 0.0;
        for (int j = i + 1 + tid; j < B2; j += 32) {
            cuDoubleComplex Uij = lu[i * B2 + j];
            cuDoubleComplex tj  = tmp[j];
            sr += Uij.x * tj.x - Uij.y * tj.y;
            si += Uij.x * tj.y + Uij.y * tj.x;
        }
        for (int s = 16; s > 0; s >>= 1) {
            sr += __shfl_down_sync(0xFFFFFFFF, sr, s);
            si += __shfl_down_sync(0xFFFFFFFF, si, s);
        }
        if (tid == 0) {
            tmp[i].x -= sr;
            tmp[i].y -= si;
            cuDoubleComplex d = lu[i * B2 + i];
            double denom = d.x * d.x + d.y * d.y;
            double tr = tmp[i].x, ti_v = tmp[i].y;
            tmp[i].x = (tr * d.x + ti_v * d.y) / denom;
            tmp[i].y = (ti_v * d.x - tr * d.y) / denom;
        }
        __syncwarp();
    }

    // RAS scatter: write only own RWGs (first B_own), discard overlap results
    for (int j = tid; j < B_own; j += 32) {
        z[rwg[j]]     = tmp[j];
        z[rwg[j] + N] = tmp[j+B_ext];
    }
}

// ============================================================
// Upload Block-Jacobi data to GPU
// ============================================================
void NearFieldPrecond::upload_to_gpu()
{
    if (mode != PREC_BLOCKJ || n_blocks == 0) return;

    Timer timer;

    // Compute flat offsets and total sizes
    std::vector<int> h_lu_off(n_blocks), h_piv_off(n_blocks), h_rwg_off(n_blocks);
    std::vector<int> h_blk_B(n_blocks), h_blk_B_ext(n_blocks);
    long long total_lu = 0, total_piv = 0, total_rwg = 0;
    max_B2 = 0;

    for (int bi = 0; bi < n_blocks; bi++) {
        int B_own = block_sizes[bi];
        int B_ext = block_sizes_ext[bi];
        int B2 = 2 * B_ext;
        h_blk_B[bi]     = B_own;
        h_blk_B_ext[bi] = B_ext;
        h_lu_off[bi]  = (int)total_lu;
        h_piv_off[bi] = (int)total_piv;
        h_rwg_off[bi] = (int)total_rwg;
        total_lu  += (long long)B2 * B2;
        total_piv += B2;
        total_rwg += B_ext;  // store extended RWG indices
        if (B2 > max_B2) max_B2 = B2;
    }

    // Check GPU memory
    size_t lu_bytes  = total_lu  * sizeof(cuDoubleComplex);
    size_t piv_bytes = total_piv * sizeof(int);
    size_t rwg_bytes = total_rwg * sizeof(int);
    size_t meta_bytes = n_blocks * 4 * sizeof(int) + n_blocks * sizeof(int); // blk_B, blk_B_ext, lu_off, piv_off, rwg_off
    size_t buf_bytes  = 2 * N2 * sizeof(cuDoubleComplex);
    size_t total_bytes = lu_bytes + piv_bytes + rwg_bytes + meta_bytes + buf_bytes;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (total_bytes > free_mem * 0.8) {
        printf("  [BlockJ-GPU] Skipping GPU upload: need %.0f MB, free %.0f MB\n",
               total_bytes / 1e6, free_mem / 1e6);
        gpu_ready = false;
        return;
    }

    // Build flat host arrays, transposing LU from column-major to row-major
    std::vector<cuDoubleComplex> h_lu_flat(total_lu);
    std::vector<int> h_piv_flat(total_piv);
    std::vector<int> h_rwg_flat(total_rwg);

    for (int bi = 0; bi < n_blocks; bi++) {
        int B_ext = block_sizes_ext[bi];
        int B2 = 2 * B_ext;
        int off_lu  = h_lu_off[bi];
        int off_piv = h_piv_off[bi];
        int off_rwg = h_rwg_off[bi];

        // Transpose LU: col-major → row-major
        const auto& lu = block_lu[bi];
        for (int i = 0; i < B2; i++) {
            for (int j = 0; j < B2; j++) {
                cdouble v = lu[i + j * B2];  // col-major: (row=i, col=j)
                h_lu_flat[off_lu + i * B2 + j] = make_cuDoubleComplex(v.real(), v.imag());
            }
        }

        const auto& piv = block_piv[bi];
        for (int k = 0; k < B2; k++)
            h_piv_flat[off_piv + k] = piv[k];

        // Store extended RWG indices (own first, then overlap)
        const auto& rwg_ext = block_rwg_ext[bi];
        for (int j = 0; j < B_ext; j++)
            h_rwg_flat[off_rwg + j] = rwg_ext[j];
    }

    // Allocate and upload to GPU
    CUDA_CHECK(cudaMalloc(&d_lu_flat,  lu_bytes));
    CUDA_CHECK(cudaMalloc(&d_piv_flat, piv_bytes));
    CUDA_CHECK(cudaMalloc(&d_rwg_flat, rwg_bytes));
    CUDA_CHECK(cudaMalloc(&d_blk_B,    n_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blk_B_ext, n_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lu_off,   n_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_piv_off,  n_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rwg_off,  n_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_buf_r,    N2 * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_buf_z,    N2 * sizeof(cuDoubleComplex)));

    CUDA_CHECK(cudaMemcpy(d_lu_flat,  h_lu_flat.data(),  lu_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_piv_flat, h_piv_flat.data(), piv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rwg_flat, h_rwg_flat.data(), rwg_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_blk_B,    h_blk_B.data(),   n_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_blk_B_ext, h_blk_B_ext.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lu_off,   h_lu_off.data(),  n_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_piv_off,  h_piv_off.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rwg_off,  h_rwg_off.data(), n_blocks * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate global memory workspace for LU solve (replaces shared memory)
    size_t ws_bytes = (size_t)n_blocks * max_B2 * sizeof(cuDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_workspace, ws_bytes));
    total_bytes += ws_bytes;

    gpu_ready = true;
    printf("  [BlockJ-GPU] Uploaded %d blocks to GPU: LU=%.1f MB, workspace=%.1f MB, total=%.1f MB (%.2fs)\n",
           n_blocks, lu_bytes / 1e6, ws_bytes / 1e6, total_bytes / 1e6, timer.elapsed_s());
}

// ============================================================
// GPU Block-Jacobi apply: z = M^{-1} * r  (device pointers)
// ============================================================
void NearFieldPrecond::apply_gpu(const cuDoubleComplex* d_r, cuDoubleComplex* d_z) const
{
    if (!gpu_ready) return;

    // Zero output
    CUDA_CHECK(cudaMemset(d_z, 0, N2 * sizeof(cuDoubleComplex)));

    blockj_apply_kernel<<<n_blocks, 32>>>(
        d_r, d_z,
        d_lu_flat, d_piv_flat, d_rwg_flat,
        d_blk_B, d_blk_B_ext, d_lu_off, d_piv_off, d_rwg_off,
        d_workspace, max_B2,
        n_blocks, N);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// Free GPU Block-Jacobi data
// ============================================================
void NearFieldPrecond::cleanup_gpu()
{
    if (!gpu_ready) return;
    cudaFree(d_workspace); d_workspace = nullptr;
    cudaFree(d_lu_flat);   d_lu_flat  = nullptr;
    cudaFree(d_piv_flat);  d_piv_flat = nullptr;
    cudaFree(d_rwg_flat);  d_rwg_flat = nullptr;
    cudaFree(d_blk_B);     d_blk_B    = nullptr;
    cudaFree(d_blk_B_ext); d_blk_B_ext = nullptr;
    cudaFree(d_lu_off);    d_lu_off   = nullptr;
    cudaFree(d_piv_off);   d_piv_off  = nullptr;
    cudaFree(d_rwg_off);   d_rwg_off  = nullptr;
    cudaFree(d_buf_r);     d_buf_r    = nullptr;
    cudaFree(d_buf_z);     d_buf_z    = nullptr;
    gpu_ready = false;
}
