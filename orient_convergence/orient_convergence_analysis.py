#!/usr/bin/env python3
"""
Orientation convergence analysis for BEM-CUDA.

Reads per-orient binary file (.per_orient) from a FULL resolution run,
subsamples every 2nd/4th/8th orientation, and compares Mueller M11.

Usage:
  python3 orient_convergence_analysis.py result.json.per_orient [...]
"""
import json
import sys
import os
import struct
import numpy as np

def load_per_orient(bin_path):
    """Load per-orient binary: header(4 ints) + weights(n) + mueller(n*16*ntheta)."""
    with open(bin_path, 'rb') as f:
        header = struct.unpack('4i', f.read(16))
        n_orient, ntheta, n_alpha, n_beta = header
        weights = np.frombuffer(f.read(n_orient * 8), dtype=np.float64)
        mueller = np.frombuffer(f.read(n_orient * 16 * ntheta * 8), dtype=np.float64)
        mueller = mueller.reshape(n_orient, 16, ntheta)
    return n_orient, ntheta, n_alpha, n_beta, weights, mueller

def load_json_mueller(json_path):
    """Load averaged Mueller from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    # mueller: 4x4xntheta in JSON
    M_raw = data["mueller"]
    ntheta = data["ntheta"]
    M = np.zeros((16, ntheta))
    for i in range(4):
        for j in range(4):
            M[i*4+j] = M_raw[i][j]
    theta = np.array(data["theta"])
    return M, theta, data

def subsample_average(weights, mueller, step):
    """Average Mueller using every `step`-th orientation."""
    indices = np.arange(0, len(weights), step)
    w_sub = weights[indices]
    m_sub = mueller[indices]
    # Renormalize weights (they should sum to ~1 for proper average)
    w_sum = w_sub.sum()
    w_full_sum = weights.sum()
    scale = w_full_sum / w_sum  # compensate for missing orientations
    M_avg = np.zeros_like(mueller[0])
    for i, idx in enumerate(indices):
        M_avg += w_sub[i] * m_sub[i]
    M_avg *= scale  # approximate: scale to match full weight sum
    return M_avg, len(indices)

def subsample_by_grid(weights, mueller, n_alpha, n_beta, div):
    """
    Subsample by taking every div-th point in alpha and beta dimensions.
    Orientations are stored in order: alpha varies fastest.
    """
    alpha_idx = np.arange(0, n_alpha, div)
    beta_idx = np.arange(0, n_beta, div)

    indices = []
    for ib in beta_idx:
        for ia in alpha_idx:
            idx = ib * n_alpha + ia
            if idx < len(weights):
                indices.append(idx)

    indices = np.array(indices)
    w_sub = weights[indices]
    m_sub = mueller[indices]

    w_sum = w_sub.sum()
    w_full_sum = weights.sum()
    scale = w_full_sum / w_sum

    M_avg = np.einsum('i,ijk->jk', w_sub, m_sub) * scale
    return M_avg, len(indices)

def rel_error(M_test, M_ref):
    """Relative error metrics for Mueller M11."""
    m11_test = M_test[0]  # M11 is element (0,0)
    m11_ref = M_ref[0]

    # Exclude forward scattering (first few degrees)
    mask = np.arange(len(m11_ref)) > 2

    diff = np.abs(m11_test[mask] - m11_ref[mask])
    ref_abs = np.abs(m11_ref[mask])

    rel_err = diff / np.maximum(ref_abs, 1e-30)
    max_re = np.max(rel_err)
    avg_re = np.mean(rel_err)

    # Csca: integral of M11 * sin(theta) d(theta)
    theta_deg = np.linspace(0, 180, len(m11_ref))
    theta_rad = np.deg2rad(theta_deg)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    csca_test = _trapz(m11_test * np.sin(theta_rad), theta_rad)
    csca_ref = _trapz(m11_ref * np.sin(theta_rad), theta_rad)
    csca_re = abs(csca_test - csca_ref) / abs(csca_ref)

    return max_re, avg_re, csca_re

def analyze(bin_path):
    """Analyze orientation convergence from per-orient binary."""
    json_path = bin_path.replace('.per_orient', '')
    ka_label = os.path.basename(json_path).replace('.json', '')

    print(f"\n=== {ka_label} ===")

    n_orient, ntheta, n_alpha, n_beta, weights, mueller = load_per_orient(bin_path)
    print(f"  Full: {n_alpha}x{n_beta}={n_orient} orient, {ntheta} theta points")

    # Full average (reference)
    M_full = np.einsum('i,ijk->jk', weights, mueller)

    # Subsample by grid: div=2,4,8 (take every div-th in alpha and beta)
    print(f"\n  {'Divisor':<10} {'Orient':>8} {'Alpha':>6} {'Beta':>6} {'M11 maxRE':>12} {'M11 avgRE':>12} {'Csca RE':>12}")
    print(f"  {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

    for div in [8, 4, 2, 1]:
        M_sub, n_sub = subsample_by_grid(weights, mueller, n_alpha, n_beta, div)
        na_sub = len(range(0, n_alpha, div))
        nb_sub = len(range(0, n_beta, div))

        if div == 1:
            print(f"  {'full':<10} {n_sub:>8} {na_sub:>6} {nb_sub:>6} {'(reference)':>12} {'(reference)':>12} {'(reference)':>12}")
        else:
            max_re, avg_re, csca_re = rel_error(M_sub, M_full)
            print(f"  {'/' + str(div):<10} {n_sub:>8} {na_sub:>6} {nb_sub:>6} {max_re:>12.4e} {avg_re:>12.4e} {csca_re:>12.4e}")

    # Also try intermediate divisors: 3, 5, 6
    print(f"\n  Extra divisors:")
    for div in [3, 5, 6]:
        if div >= n_alpha or div >= n_beta:
            continue
        M_sub, n_sub = subsample_by_grid(weights, mueller, n_alpha, n_beta, div)
        na_sub = len(range(0, n_alpha, div))
        nb_sub = len(range(0, n_beta, div))
        max_re, avg_re, csca_re = rel_error(M_sub, M_full)
        print(f"  {'/' + str(div):<10} {n_sub:>8} {na_sub:>6} {nb_sub:>6} {max_re:>12.4e} {avg_re:>12.4e} {csca_re:>12.4e}")

def main():
    if len(sys.argv) < 2:
        # Auto-find .per_orient files
        import glob
        search_dirs = [".", os.path.expanduser("~/orient_convergence")]
        files = []
        for d in search_dirs:
            files.extend(glob.glob(os.path.join(d, "*.per_orient")))
        if not files:
            print("Usage: python3 orient_convergence_analysis.py <file.json.per_orient> [...]")
            sys.exit(1)
    else:
        files = sys.argv[1:]

    for f in sorted(files):
        analyze(f)

if __name__ == "__main__":
    main()
