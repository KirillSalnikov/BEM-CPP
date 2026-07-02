#include "precond.h"
#include "bem_fmm.h"
#include "gpu_select.h"
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <limits>

namespace {
__global__ void precond_split_complex_kernel(const double2* in, double* out_re, double* out_im, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double2 v = in[i];
    out_re[i] = v.x;
    out_im[i] = v.y;
}

__global__ void precond_pack_complex_kernel(const double* in_re, const double* in_im, double2* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = make_double2(in_re[i], in_im[i]);
}

__device__ inline void c_mul(double ar, double ai, double br, double bi, double& cr, double& ci)
{
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

__device__ inline void c_div(double ar, double ai, double br, double bi, double& cr, double& ci)
{
    double den = br * br + bi * bi;
    cr = (ar * br + ai * bi) / den;
    ci = (ai * br - ar * bi) / den;
}

__global__ void precond_schwarz_kernel(
    int n_blocks, int N,
    const int* offsets, const int* ids, const int* piv,
    const double* lu_re, const double* lu_im,
    const double* weight,
    const double* r_re, const double* r_im,
    double* z_re, double* z_im)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    const int max_dim = 32;
    int off = offsets[b];
    int next = offsets[b + 1];
    int nb = next - off;
    int nd = 2 * nb;
    double xr[max_dim], xi[max_dim];

    for (int i = 0; i < nb; i++) {
        int id = ids[off + i];
        xr[2*i] = r_re[id];
        xi[2*i] = r_im[id];
        xr[2*i + 1] = r_re[N + id];
        xi[2*i + 1] = r_im[N + id];
    }

    int lu_base = b * max_dim * max_dim;
    int piv_base = b * max_dim;
    for (int k = 0; k < nd; k++) {
        int p = piv[piv_base + k];
        if (p != k) {
            double tr = xr[k], ti = xi[k];
            xr[k] = xr[p]; xi[k] = xi[p];
            xr[p] = tr; xi[p] = ti;
        }
        for (int i = k + 1; i < nd; i++) {
            int a = lu_base + i * max_dim + k;
            double mr, mi;
            c_mul(lu_re[a], lu_im[a], xr[k], xi[k], mr, mi);
            xr[i] -= mr;
            xi[i] -= mi;
        }
    }
    for (int i = nd - 1; i >= 0; i--) {
        double sr = xr[i], si = xi[i];
        for (int j = i + 1; j < nd; j++) {
            int a = lu_base + i * max_dim + j;
            double mr, mi;
            c_mul(lu_re[a], lu_im[a], xr[j], xi[j], mr, mi);
            sr -= mr;
            si -= mi;
        }
        int diag = lu_base + i * max_dim + i;
        c_div(sr, si, lu_re[diag], lu_im[diag], xr[i], xi[i]);
    }

    for (int i = 0; i < nb; i++) {
        int id = ids[off + i];
        double w = weight[id];
        atomicAdd(&z_re[id], xr[2*i] / w);
        atomicAdd(&z_im[id], xi[2*i] / w);
        atomicAdd(&z_re[N + id], xr[2*i + 1] / w);
        atomicAdd(&z_im[N + id], xi[2*i + 1] / w);
    }
}

__global__ void precond_near_matvec_kernel(
    int N,
    const int* row_ptr, const int* col_idx,
    const double* diag_re, const double* diag_im,
    const double* near_re, const double* near_im,
    const double* x_re, const double* x_im,
    double* y_re, double* y_im)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= N) return;

    double xm_r = x_re[m], xm_i = x_im[m];
    double xN_r = x_re[N + m], xN_i = x_im[N + m];
    double yr, yi, tr, ti;

    int db = 4 * m;
    c_mul(diag_re[db], diag_im[db], xm_r, xm_i, yr, yi);
    c_mul(diag_re[db + 1], diag_im[db + 1], xN_r, xN_i, tr, ti);
    yr += tr; yi += ti;

    double yNr, yNi;
    c_mul(diag_re[db + 2], diag_im[db + 2], xm_r, xm_i, yNr, yNi);
    c_mul(diag_re[db + 3], diag_im[db + 3], xN_r, xN_i, tr, ti);
    yNr += tr; yNi += ti;

    for (int jc = row_ptr[m]; jc < row_ptr[m + 1]; jc++) {
        int n = col_idx[jc];
        double xn_r = x_re[n], xn_i = x_im[n];
        double xNn_r = x_re[N + n], xNn_i = x_im[N + n];
        int nb = 4 * jc;
        c_mul(near_re[nb], near_im[nb], xn_r, xn_i, tr, ti);
        yr += tr; yi += ti;
        c_mul(near_re[nb + 1], near_im[nb + 1], xNn_r, xNn_i, tr, ti);
        yr += tr; yi += ti;
        c_mul(near_re[nb + 2], near_im[nb + 2], xn_r, xn_i, tr, ti);
        yNr += tr; yNi += ti;
        c_mul(near_re[nb + 3], near_im[nb + 3], xNn_r, xNn_i, tr, ti);
        yNr += tr; yNi += ti;
    }
    y_re[m] = yr; y_im[m] = yi;
    y_re[N + m] = yNr; y_im[N + m] = yNi;
}

__global__ void precond_residual_kernel(
    const double* r_re, const double* r_im,
    const double* Az_re, const double* Az_im,
    double* err_re, double* err_im, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    err_re[i] = r_re[i] - Az_re[i];
    err_im[i] = r_im[i] - Az_im[i];
}

__global__ void precond_axpy_kernel(double* z_re, double* z_im,
                                    const double* corr_re, const double* corr_im,
                                    double omega, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    z_re[i] += omega * corr_re[i];
    z_im[i] += omega * corr_im[i];
}

} // namespace

static bool lu_factor_small(std::vector<cdouble>& A, std::vector<int>& piv, int n)
{
    piv.resize(n);
    for (int k = 0; k < n; k++) {
        int p = k;
        double best = std::abs(A[k*n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = std::abs(A[i*n + k]);
            if (v > best) {
                best = v;
                p = i;
            }
        }
        if (best < 1e-24)
            return false;
        piv[k] = p;
        if (p != k) {
            for (int j = 0; j < n; j++)
                std::swap(A[k*n + j], A[p*n + j]);
        }
        cdouble diag = A[k*n + k];
        for (int i = k + 1; i < n; i++) {
            cdouble f = A[i*n + k] / diag;
            A[i*n + k] = f;
            for (int j = k + 1; j < n; j++)
                A[i*n + j] -= f * A[k*n + j];
        }
    }
    return true;
}

static void lu_solve_small(const std::vector<cdouble>& LU, const std::vector<int>& piv,
                           const cdouble* b, cdouble* x, int n)
{
    for (int i = 0; i < n; i++)
        x[i] = b[i];
    for (int k = 0; k < n; k++) {
        int p = piv[k];
        if (p != k)
            std::swap(x[k], x[p]);
        for (int i = k + 1; i < n; i++)
            x[i] -= LU[i*n + k] * x[k];
    }
    for (int i = n - 1; i >= 0; i--) {
        cdouble s = x[i];
        for (int j = i + 1; j < n; j++)
            s -= LU[i*n + j] * x[j];
        x[i] = s / LU[i*n + i];
    }
}

static int find_csr_col(const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
                        int row, int col)
{
    for (int jc = row_ptr[row]; jc < row_ptr[row + 1]; jc++) {
        if (col_idx[jc] == col)
            return jc;
    }
    return -1;
}

static void push_unique(std::vector<int>& ids, int id)
{
    if (std::find(ids.begin(), ids.end(), id) == ids.end())
        ids.push_back(id);
}

void NearFieldPrecond::build(BemFmmOperator& op)
{
    Timer timer;
    N = op.N;
    N2 = 2 * N;
    int Nq = op.Nq;

    printf("  [Precond] Building 2x2 block Jacobi preconditioner...\n");

    double inv4pi = 1.0 / (4.0 * M_PI);
    cdouble k_vals[2] = {op.k_ext, op.k_int};
    cdouble eta_e = op.eta_ext, eta_i = op.eta_int;

    richardson_sweeps = std::max(0, bem_env_int("BEM_PREC_SWEEPS", richardson_sweeps));
    richardson_omega = bem_env_double("BEM_PREC_OMEGA", richardson_omega);

    block_schwarz = bem_env_flag_enabled("BEM_PREC_BLOCK");
    max_block_basis = std::max(2, bem_env_int("BEM_PREC_BLOCK_SIZE", max_block_basis));
    max_block_basis = std::min(max_block_basis, 16);

    int near_degree = block_schwarz ? max_block_basis : 0;
    near_degree = std::max(0, bem_env_int("BEM_PREC_NEAR", near_degree));

    blk_inv.resize(4 * N);
    diag_blk.resize(4 * N);
    if (block_schwarz && near_degree > 0) {
        std::vector<double> centers((size_t)N * 3, 0.0);
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < N; m++) {
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int q = 0; q < Nq; q++) {
                const double* qp = &op.qpts_p[(m * Nq + q) * 3];
                const double* qm = &op.qpts_m[(m * Nq + q) * 3];
                cx += qp[0] + qm[0];
                cy += qp[1] + qm[1];
                cz += qp[2] + qm[2];
            }
            double inv = 0.5 / (double)Nq;
            centers[(size_t)m * 3 + 0] = cx * inv;
            centers[(size_t)m * 3 + 1] = cy * inv;
            centers[(size_t)m * 3 + 2] = cz * inv;
        }

        std::vector<std::vector<int>> rows(N);
        #pragma omp parallel for schedule(dynamic, 16)
        for (int m = 0; m < N; m++) {
            int keep = std::min(std::max(1, near_degree), N);
            std::vector<double> best_d2(keep, std::numeric_limits<double>::infinity());
            std::vector<int> best_id(keep, -1);
            double mx = centers[(size_t)m * 3 + 0];
            double my = centers[(size_t)m * 3 + 1];
            double mz = centers[(size_t)m * 3 + 2];
            for (int n = 0; n < N; n++) {
                if (n == m)
                    continue;
                double dx = mx - centers[(size_t)n * 3 + 0];
                double dy = my - centers[(size_t)n * 3 + 1];
                double dz = mz - centers[(size_t)n * 3 + 2];
                double d2 = dx*dx + dy*dy + dz*dz;
                int worst = 0;
                for (int i = 1; i < keep; i++)
                    if (best_d2[i] > best_d2[worst])
                        worst = i;
                if (d2 < best_d2[worst]) {
                    best_d2[worst] = d2;
                    best_id[worst] = n;
                }
            }
            std::vector<int> order(keep);
            for (int i = 0; i < keep; i++)
                order[i] = i;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return best_d2[a] < best_d2[b];
            });

            std::vector<int> ids;
            ids.reserve((size_t)keep + (size_t)(op.corr_row_ptr[m + 1] - op.corr_row_ptr[m]) + 1);
            push_unique(ids, m);
            for (int idx : order)
                if (best_id[idx] >= 0)
                    push_unique(ids, best_id[idx]);
            for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++)
                push_unique(ids, op.corr_col_idx[jc]);
            rows[m].swap(ids);
        }

        near_row_ptr.assign(N + 1, 0);
        for (int m = 0; m < N; m++)
            near_row_ptr[m + 1] = near_row_ptr[m] + (int)rows[m].size();
        near_col_idx.resize(near_row_ptr[N]);
        for (int m = 0; m < N; m++)
            std::copy(rows[m].begin(), rows[m].end(), near_col_idx.begin() + near_row_ptr[m]);
        printf("  [Precond] Expanded near graph: degree=%d nnz=%zu (%.1f per row)\n",
               near_degree, near_col_idx.size(), near_col_idx.empty() ? 0.0 : (double)near_col_idx.size() / (double)N);
    } else {
        near_row_ptr = op.corr_row_ptr;
        near_col_idx = op.corr_col_idx;
    }
    near_blk.assign(4 * near_col_idx.size(), cdouble(0));

    // For each RWG m, compute diagonal L(m,m) and K(m,m) entries
    #pragma omp parallel for schedule(dynamic, 16)
    for (int m = 0; m < N; m++) {
        cdouble L_vals_k[2] = {0, 0};
        cdouble K_vals_k[2] = {0, 0};

        // Sum over 4 half-pair combos: (p,p), (p,m), (m,p), (m,m), source=target=m
        for (int hm = 0; hm < 2; hm++) {
            const double* qm = (hm == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
            const double* fm = (hm == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
            double dm = (hm == 0) ? op.div_p[m] : op.div_m[m];
            const double* jwm = (hm == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

            for (int hn = 0; hn < 2; hn++) {
                const double* qn = (hn == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
                const double* fn = (hn == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
                double dn = (hn == 0) ? op.div_p[m] : op.div_m[m];
                const double* jwn = (hn == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

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
                                double cx = dy*fzn - dz*fyn;
                                double cy = dz*fxn - dx*fzn;
                                double cz = dx*fyn - dy*fxn;
                                K_vals_k[ki] += gG * (fxm*cx + fym*cy + fzm*cz) * ww;
                            } else {
                                cdouble G0 = ik * inv4pi;
                                L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G0 * ww;
                            }
                        }
                    }
                }
            }
        }

        // Add singular corrections for m=m
        for (int jc = op.corr_row_ptr[m]; jc < op.corr_row_ptr[m + 1]; jc++) {
            if (op.corr_col_idx[jc] == m) {
                L_vals_k[0] += op.corr_L_ext_val[jc];
                K_vals_k[0] += op.corr_K_ext_val[jc];
                L_vals_k[1] += op.corr_L_int_val[jc];
                K_vals_k[1] += op.corr_K_int_val[jc];
            }
        }

        // Assemble 2x2 PMCHWT block
        cdouble Ksum_mm = K_vals_k[0] + op.int_op_sign * K_vals_k[1] + op.k_identity;
        cdouble A_mm = eta_e * L_vals_k[0] + op.int_op_sign * eta_i * L_vals_k[1]; // eta*L
        cdouble B_mm = -Ksum_mm / op.unknown_m_scale;                 // -K/sM
        cdouble C_mm = op.row_h_scale * Ksum_mm;                      // rH*K
        cdouble D_mm = op.row_h_scale *
                       (L_vals_k[0] / eta_e + op.int_op_sign * L_vals_k[1] / eta_i) /
                       op.unknown_m_scale;                            // rH*L/(eta*sM)

        diag_blk[4*m + 0] = A_mm;
        diag_blk[4*m + 1] = B_mm;
        diag_blk[4*m + 2] = C_mm;
        diag_blk[4*m + 3] = D_mm;

        // Invert 2x2 block
        cdouble det = A_mm * D_mm - B_mm * C_mm;
        if (std::abs(det) < 1e-30) det = cdouble(1e-30);
        cdouble inv_det = cdouble(1.0) / det;

        blk_inv[4*m + 0] =  D_mm * inv_det;
        blk_inv[4*m + 1] = -B_mm * inv_det;
        blk_inv[4*m + 2] = -C_mm * inv_det;
        blk_inv[4*m + 3] =  A_mm * inv_det;
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (int m = 0; m < N; m++) {
        for (int jc = near_row_ptr[m]; jc < near_row_ptr[m + 1]; jc++) {
            int n = near_col_idx[jc];
            if (n == m)
                continue;

            cdouble L_vals_k[2] = {0, 0};
            cdouble K_vals_k[2] = {0, 0};
            int corr_pos = find_csr_col(op.corr_row_ptr, op.corr_col_idx, m, n);
            if (corr_pos >= 0) {
                L_vals_k[0] = op.corr_L_ext_val[corr_pos];
                K_vals_k[0] = op.corr_K_ext_val[corr_pos];
                L_vals_k[1] = op.corr_L_int_val[corr_pos];
                K_vals_k[1] = op.corr_K_int_val[corr_pos];
            }

            for (int hm = 0; hm < 2; hm++) {
                const double* qm = (hm == 0) ? &op.qpts_p[m * Nq * 3] : &op.qpts_m[m * Nq * 3];
                const double* fm = (hm == 0) ? &op.f_p[m * Nq * 3] : &op.f_m[m * Nq * 3];
                double dm = (hm == 0) ? op.div_p[m] : op.div_m[m];
                const double* jwm = (hm == 0) ? &op.jw_p[m * Nq] : &op.jw_m[m * Nq];

                for (int hn = 0; hn < 2; hn++) {
                    const double* qn = (hn == 0) ? &op.qpts_p[n * Nq * 3] : &op.qpts_m[n * Nq * 3];
                    const double* fn = (hn == 0) ? &op.f_p[n * Nq * 3] : &op.f_m[n * Nq * 3];
                    double dn = (hn == 0) ? op.div_p[n] : op.div_m[n];
                    const double* jwn = (hn == 0) ? &op.jw_p[n * Nq] : &op.jw_m[n * Nq];

                    for (int qi = 0; qi < Nq; qi++) {
                        double rx = qm[qi*3], ry = qm[qi*3+1], rz = qm[qi*3+2];
                        double fxm = fm[qi*3], fym = fm[qi*3+1], fzm = fm[qi*3+2];
                        double wm_val = jwm[qi];

                        for (int qj = 0; qj < Nq; qj++) {
                            double dx = rx - qn[qj*3];
                            double dy = ry - qn[qj*3+1];
                            double dz = rz - qn[qj*3+2];
                            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                            double ww = wm_val * jwn[qj];
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
                                    double cx = dy*fzn - dz*fyn;
                                    double cy = dz*fxn - dx*fzn;
                                    double cz = dx*fyn - dy*fxn;
                                    K_vals_k[ki] += gG * (fxm*cx + fym*cy + fzm*cz) * ww;
                                } else {
                                    cdouble G0 = ik * inv4pi;
                                    L_vals_k[ki] += (ik * f_dot - iok * dm * dn) * G0 * ww;
                                }
                            }
                        }
                    }
                }
            }

            cdouble Lext = L_vals_k[0];
            cdouble Lint = L_vals_k[1];
            cdouble Ksum = K_vals_k[0] + op.int_op_sign * K_vals_k[1];

            near_blk[4*jc + 0] = eta_e * Lext + op.int_op_sign * eta_i * Lint;
            near_blk[4*jc + 1] = -Ksum / op.unknown_m_scale;
            near_blk[4*jc + 2] = op.row_h_scale * Ksum;
            near_blk[4*jc + 3] = op.row_h_scale *
                                 (Lext / eta_e + op.int_op_sign * Lint / eta_i) /
                                 op.unknown_m_scale;
        }
    }

    size_t report_block_count = 0;
    blocks.clear();
    max_block_dim = 0;
    block_weight.assign(N, 0.0);
    if (block_schwarz) {
        blocks.reserve(N);
        for (int m = 0; m < N; m++) {
            LocalBlock blk;
            for (int jc = near_row_ptr[m]; jc < near_row_ptr[m + 1]; jc++) {
                if ((int)blk.ids.size() >= max_block_basis)
                    break;
                blk.ids.push_back(near_col_idx[jc]);
            }
            if (std::find(blk.ids.begin(), blk.ids.end(), m) == blk.ids.end()) {
                if ((int)blk.ids.size() >= max_block_basis)
                    blk.ids.back() = m;
                else
                    blk.ids.push_back(m);
            }
            std::sort(blk.ids.begin(), blk.ids.end());
            blk.ids.erase(std::unique(blk.ids.begin(), blk.ids.end()), blk.ids.end());

            int nb = (int)blk.ids.size();
            int nd = 2 * nb;
            blk.lu.assign(nd * nd, cdouble(0));
            for (int a = 0; a < nb; a++) {
                int row = blk.ids[a];
                for (int b = 0; b < nb; b++) {
                    int col = blk.ids[b];
                    cdouble A(0), B(0), C(0), D(0);
                    if (row == col) {
                        A = diag_blk[4*row + 0];
                        B = diag_blk[4*row + 1];
                        C = diag_blk[4*row + 2];
                        D = diag_blk[4*row + 3];
                    } else {
                        int pos = find_csr_col(near_row_ptr, near_col_idx, row, col);
                        if (pos >= 0) {
                            A = near_blk[4*pos + 0];
                            B = near_blk[4*pos + 1];
                            C = near_blk[4*pos + 2];
                            D = near_blk[4*pos + 3];
                        }
                    }
                    blk.lu[(2*a)   * nd + (2*b)]   = A;
                    blk.lu[(2*a)   * nd + (2*b+1)] = B;
                    blk.lu[(2*a+1) * nd + (2*b)]   = C;
                    blk.lu[(2*a+1) * nd + (2*b+1)] = D;
                }
            }

            if (lu_factor_small(blk.lu, blk.piv, nd)) {
                for (int id : blk.ids)
                    block_weight[id] += 1.0;
                max_block_dim = std::max(max_block_dim, nd);
                blocks.push_back(std::move(blk));
            }
        }
        for (double& w : block_weight)
            if (w == 0.0) w = 1.0;
        report_block_count = blocks.size();
        if (bem_env_flag_enabled("BEM_PREC_GPU", true))
            upload_device();
    }

    printf("  [Precond] Block Jacobi built: %.2fs", timer.elapsed_s());
    if (richardson_sweeps > 0)
        printf(" + %d near sweeps (omega=%.2f)", richardson_sweeps, richardson_omega);
    if (block_schwarz)
        printf(" + Schwarz blocks=%zu max_basis=%d", report_block_count, max_block_basis);
    if (device_ready)
        printf(" + GPU apply");
    printf("\n");
}

void NearFieldPrecond::apply_block_inv(const cdouble* r, cdouble* z) const
{
    // z[m] = inv_A*r[m] + inv_B*r[N+m]
    // z[N+m] = inv_C*r[m] + inv_D*r[N+m]
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < N; m++) {
        cdouble rm = r[m], rNm = r[N + m];
        z[m]     = blk_inv[4*m+0] * rm + blk_inv[4*m+1] * rNm;
        z[N + m] = blk_inv[4*m+2] * rm + blk_inv[4*m+3] * rNm;
    }
}

void NearFieldPrecond::apply_near(const cdouble* x, cdouble* y) const
{
    std::fill(y, y + N2, cdouble(0));
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < N; m++) {
        cdouble xm = x[m], xNm = x[N + m];

        cdouble ym = diag_blk[4*m+0] * xm + diag_blk[4*m+1] * xNm;
        cdouble yNm = diag_blk[4*m+2] * xm + diag_blk[4*m+3] * xNm;

        for (int jc = near_row_ptr[m]; jc < near_row_ptr[m + 1]; jc++) {
            int n = near_col_idx[jc];
            cdouble xn = x[n], xNn = x[N + n];
            ym  += near_blk[4*jc+0] * xn + near_blk[4*jc+1] * xNn;
            yNm += near_blk[4*jc+2] * xn + near_blk[4*jc+3] * xNn;
        }

        y[m] = ym;
        y[N + m] = yNm;
    }
}

void NearFieldPrecond::apply(const cdouble* r, cdouble* z) const
{
    if (block_schwarz) {
        if (device_ready)
            apply_block_schwarz_cuda(r, z);
        else
            apply_block_schwarz(r, z);
        return;
    }

    apply_block_inv(r, z);
    if (richardson_sweeps <= 0)
        return;

    bool reuse_workspace = bem_env_flag_enabled("BEM_PREC_REUSE_WORKSPACE", true);
    std::vector<cdouble> local_Az, local_err, local_corr;
    cdouble* Az;
    cdouble* err;
    cdouble* corr;
    if (reuse_workspace) {
        tmp_Az.resize(N2);
        tmp_err.resize(N2);
        tmp_corr.resize(N2);
        Az = tmp_Az.data();
        err = tmp_err.data();
        corr = tmp_corr.data();
    } else {
        local_Az.resize(N2);
        local_err.resize(N2);
        local_corr.resize(N2);
        Az = local_Az.data();
        err = local_err.data();
        corr = local_corr.data();
    }
    for (int sweep = 0; sweep < richardson_sweeps; sweep++) {
        apply_near(z, Az);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N2; i++)
            err[i] = r[i] - Az[i];
        apply_block_inv(err, corr);
        cdouble omega(richardson_omega, 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N2; i++)
            z[i] += omega * corr[i];
    }
}

void NearFieldPrecond::apply_pair(const cdouble* r1, const cdouble* r2,
                                  cdouble* z1, cdouble* z2) const
{
    apply(r1, z1);
    apply(r2, z2);
}

void NearFieldPrecond::apply_block_schwarz(const cdouble* r, cdouble* z) const
{
    std::fill(z, z + N2, cdouble(0));
    std::vector<cdouble> rhs(max_block_dim), sol(max_block_dim);

    for (const LocalBlock& blk : blocks) {
        int nb = (int)blk.ids.size();
        int nd = 2 * nb;
        for (int i = 0; i < nb; i++) {
            int id = blk.ids[i];
            rhs[2*i] = r[id];
            rhs[2*i + 1] = r[N + id];
        }
        lu_solve_small(blk.lu, blk.piv, rhs.data(), sol.data(), nd);
        for (int i = 0; i < nb; i++) {
            int id = blk.ids[i];
            z[id] += sol[2*i] / block_weight[id];
            z[N + id] += sol[2*i + 1] / block_weight[id];
        }
    }
}

void NearFieldPrecond::upload_device()
{
    cleanup_device();
    if (!block_schwarz || blocks.empty() || max_block_dim > 32)
        return;

    device_block_count = (int)blocks.size();
    std::vector<int> offsets(device_block_count + 1, 0);
    for (int b = 0; b < device_block_count; b++)
        offsets[b + 1] = offsets[b] + (int)blocks[b].ids.size();
    device_ids_count = offsets[device_block_count];
    device_lu_count = device_block_count * 32 * 32;

    std::vector<int> flat_ids(device_ids_count);
    std::vector<int> flat_piv(device_block_count * 32, 0);
    std::vector<double> flat_lu_re(device_lu_count, 0.0), flat_lu_im(device_lu_count, 0.0);
    for (int b = 0; b < device_block_count; b++) {
        const LocalBlock& blk = blocks[b];
        int nb = (int)blk.ids.size();
        int nd = 2 * nb;
        for (int i = 0; i < nb; i++)
            flat_ids[offsets[b] + i] = blk.ids[i];
        for (int i = 0; i < nd; i++)
            flat_piv[b * 32 + i] = blk.piv[i];
        for (int i = 0; i < nd; i++) {
            for (int j = 0; j < nd; j++) {
                cdouble v = blk.lu[i * nd + j];
                int dst = b * 32 * 32 + i * 32 + j;
                flat_lu_re[dst] = v.real();
                flat_lu_im[dst] = v.imag();
            }
        }
    }

    CUDA_CHECK(cudaMalloc(&d_block_offsets, (device_block_count + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_ids, device_ids_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_piv, device_block_count * 32 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_lu_re, device_lu_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_lu_im, device_lu_count * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_weight, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_complex, N2 * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_z_complex, N2 * sizeof(double2)));
    CUDA_CHECK(cudaMalloc(&d_r_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_z_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Az_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Az_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_corr_re, N2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_corr_im, N2 * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_block_offsets, offsets.data(), (device_block_count + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_ids, flat_ids.data(), device_ids_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_piv, flat_piv.data(), device_block_count * 32 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_lu_re, flat_lu_re.data(), device_lu_count * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_lu_im, flat_lu_im.data(), device_lu_count * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_weight, block_weight.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    device_near_nnz = (int)near_col_idx.size();
    std::vector<double> diag_re(4 * N), diag_im(4 * N);
    for (int i = 0; i < 4 * N; i++) {
        diag_re[i] = diag_blk[i].real();
        diag_im[i] = diag_blk[i].imag();
    }
    std::vector<double> near_re(4 * device_near_nnz), near_im(4 * device_near_nnz);
    for (int i = 0; i < 4 * device_near_nnz; i++) {
        near_re[i] = near_blk[i].real();
        near_im[i] = near_blk[i].imag();
    }
    CUDA_CHECK(cudaMalloc(&d_diag_re, 4 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_diag_im, 4 * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_near_re, 4 * device_near_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_near_im, 4 * device_near_nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_near_row_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_near_col_idx, device_near_nnz * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_diag_re, diag_re.data(), 4 * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_diag_im, diag_im.data(), 4 * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_re, near_re.data(), 4 * device_near_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_im, near_im.data(), 4 * device_near_nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_row_ptr, near_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_near_col_idx, near_col_idx.data(), device_near_nnz * sizeof(int), cudaMemcpyHostToDevice));

    device_ready = true;

    // After a successful GPU upload the Schwarz preconditioner is applied
    // entirely from device memory. Drop host mirrors so long orientation runs
    // do not carry duplicate LU blocks and near-field CSR data.
    if (!bem_env_flag_enabled("BEM_PREC_KEEP_HOST")) {
        std::vector<cdouble>().swap(blk_inv);
        std::vector<cdouble>().swap(diag_blk);
        std::vector<int>().swap(near_row_ptr);
        std::vector<int>().swap(near_col_idx);
        std::vector<cdouble>().swap(near_blk);
        std::vector<LocalBlock>().swap(blocks);
        std::vector<double>().swap(block_weight);
    }
}

void NearFieldPrecond::cleanup_device()
{
    cudaFree(d_block_offsets); d_block_offsets = nullptr;
    cudaFree(d_block_ids); d_block_ids = nullptr;
    cudaFree(d_block_piv); d_block_piv = nullptr;
    cudaFree(d_block_lu_re); d_block_lu_re = nullptr;
    cudaFree(d_block_lu_im); d_block_lu_im = nullptr;
    cudaFree(d_block_weight); d_block_weight = nullptr;
    cudaFree(d_r_complex); d_r_complex = nullptr;
    cudaFree(d_z_complex); d_z_complex = nullptr;
    cudaFree(d_r_re); d_r_re = nullptr;
    cudaFree(d_r_im); d_r_im = nullptr;
    cudaFree(d_z_re); d_z_re = nullptr;
    cudaFree(d_z_im); d_z_im = nullptr;
    cudaFree(d_Az_re); d_Az_re = nullptr;
    cudaFree(d_Az_im); d_Az_im = nullptr;
    cudaFree(d_err_re); d_err_re = nullptr;
    cudaFree(d_err_im); d_err_im = nullptr;
    cudaFree(d_corr_re); d_corr_re = nullptr;
    cudaFree(d_corr_im); d_corr_im = nullptr;
    cudaFree(d_diag_re); d_diag_re = nullptr;
    cudaFree(d_diag_im); d_diag_im = nullptr;
    cudaFree(d_near_re); d_near_re = nullptr;
    cudaFree(d_near_im); d_near_im = nullptr;
    cudaFree(d_near_row_ptr); d_near_row_ptr = nullptr;
    cudaFree(d_near_col_idx); d_near_col_idx = nullptr;
    device_ready = false;
    device_block_count = 0;
    device_ids_count = 0;
    device_lu_count = 0;
    device_near_nnz = 0;
}

void NearFieldPrecond::apply_block_schwarz_cuda(const cdouble* r, cdouble* z) const
{
    int block = 256;
    int grid_vec = (N2 + block - 1) / block;
    CUDA_CHECK(cudaMemcpy(d_r_complex, r, N2 * sizeof(double2), cudaMemcpyHostToDevice));
    precond_split_complex_kernel<<<grid_vec, block>>>(d_r_complex, d_r_re, d_r_im, N2);
    CUDA_CHECK(cudaGetLastError());
    apply_block_schwarz_cuda_device(d_r_re, d_r_im, d_z_re, d_z_im);

    if (richardson_sweeps > 0) {
        int grid_N = (N + block - 1) / block;
        for (int sweep = 0; sweep < richardson_sweeps; sweep++) {
            precond_near_matvec_kernel<<<grid_N, block>>>(
                N, d_near_row_ptr, d_near_col_idx,
                d_diag_re, d_diag_im, d_near_re, d_near_im,
                d_z_re, d_z_im, d_Az_re, d_Az_im);
            CUDA_CHECK(cudaGetLastError());
            precond_residual_kernel<<<grid_vec, block>>>(
                d_r_re, d_r_im, d_Az_re, d_Az_im, d_err_re, d_err_im, N2);
            CUDA_CHECK(cudaGetLastError());
            apply_block_schwarz_cuda_device(d_err_re, d_err_im, d_corr_re, d_corr_im);
            precond_axpy_kernel<<<grid_vec, block>>>(
                d_z_re, d_z_im, d_corr_re, d_corr_im, richardson_omega, N2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    precond_pack_complex_kernel<<<grid_vec, block>>>(d_z_re, d_z_im, d_z_complex, N2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(z, d_z_complex, N2 * sizeof(double2), cudaMemcpyDeviceToHost));
}

void NearFieldPrecond::apply_block_schwarz_cuda_device(const double* in_re, const double* in_im,
                                                       double* out_re, double* out_im) const
{
    CUDA_CHECK(cudaMemset(out_re, 0, N2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(out_im, 0, N2 * sizeof(double)));
    int block = 256;
    int grid_blocks = (device_block_count + block - 1) / block;
    precond_schwarz_kernel<<<grid_blocks, block>>>(
        device_block_count, N,
        d_block_offsets, d_block_ids, d_block_piv,
        d_block_lu_re, d_block_lu_im,
        d_block_weight,
        in_re, in_im,
        out_re, out_im);
    CUDA_CHECK(cudaGetLastError());
}
