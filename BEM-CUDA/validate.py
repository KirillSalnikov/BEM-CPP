#!/usr/bin/env python3
"""Compare BEM-CUDA output with Python BEM reference."""
import sys, json, os
sys.path.insert(0, os.path.expanduser('~/BEM'))
import numpy as np
import bem_core as bc

def main():
    ka = 1.0
    n_re, n_im = 1.3116, 0.0
    ref = 2
    ntheta = 181

    k_ext = ka
    m = complex(n_re, n_im)
    k_int = k_ext * m
    eta_ext = 1.0
    eta_int = 1.0 / abs(m)

    print(f"Python BEM reference: ka={ka}, m={m}, ref={ref}")

    # Mesh
    verts, tris = bc.icosphere(radius=1.0, refinements=ref)
    print(f"  Mesh: {len(verts)} verts, {len(tris)} tris")

    # RWG
    rwg = bc.build_rwg(verts, tris)
    N = rwg['N']
    print(f"  RWG: {N} basis functions")

    # PMCHWT
    Z, L_ext, K_ext = bc.assemble_pmchwt(rwg, verts, tris, k_ext, k_int,
                                          eta_ext, eta_int, quad_order=7, parallel=False)

    theta_arr = np.linspace(0, np.pi, ntheta)
    M = bc.compute_mueller_matrix(rwg, verts, tris, k_ext, eta_ext, theta_arr,
                                   Z=Z, k_hat=np.array([0,0,1.0]),
                                   sM=-1, quad_order=7)

    # Save Python result
    np.savez('python_ref.npz', theta=theta_arr, mueller=M)

    # Load CUDA result
    cuda_file = 'test_single.json'
    if not os.path.exists(cuda_file):
        print(f"CUDA result {cuda_file} not found")
        return

    with open(cuda_file) as f:
        cuda = json.load(f)

    M_cuda = np.array(cuda['mueller'])  # (4, 4, ntheta)
    theta_cuda = np.array(cuda['theta']) * np.pi / 180  # to radians

    # Compare M11 (most important)
    print(f"\n  M11 comparison (theta=0..180):")
    print(f"  {'theta':>6s}  {'Python':>12s}  {'CUDA':>12s}  {'RelErr':>10s}")

    angles = [0, 30, 60, 90, 120, 150, 180]
    for a in angles:
        idx_p = np.argmin(np.abs(theta_arr - np.radians(a)))
        idx_c = np.argmin(np.abs(theta_cuda - np.radians(a)))
        mp = M[0,0,idx_p]
        mc = M_cuda[0][0][idx_c]
        re = abs(mp - mc) / max(abs(mp), 1e-30)
        print(f"  {a:>6d}  {mp:>12.6f}  {mc:>12.6f}  {re:>10.4f}")

    # Overall relative error
    M11_py = M[0,0,:]
    M11_cu = np.array(M_cuda[0][0])
    mask = np.abs(M11_py) > 1e-10
    rel_err = np.abs(M11_py[mask] - M11_cu[mask]) / np.abs(M11_py[mask])
    print(f"\n  M11 mean relative error: {np.mean(rel_err):.4f}")
    print(f"  M11 max  relative error: {np.max(rel_err):.4f}")

if __name__ == '__main__':
    main()
