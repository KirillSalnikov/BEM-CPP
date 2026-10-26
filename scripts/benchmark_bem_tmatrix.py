#!/usr/bin/env python3
"""Build a BEM T matrix and analytically average the complete Mueller curve.

Dense CPU proof of concept. It measures common BEM assembly/LU, T-specific
setup, and analytical SO(3) averaging separately. No GPU speedup is implied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from tmatrix_reference import (modes, transverse_harmonics, tangent_frame,
                               analytic_mueller, numerical_so3_mueller)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_mie import mie_mueller, mie_coefficients


def relative(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["sphere", "prism", "tetra"], default="sphere")
    parser.add_argument("--ref", type=int, default=1)
    parser.add_argument("--ka", type=float, default=1.)
    parser.add_argument("--ri", type=float, default=1.5)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--angular-band", type=int, default=0)
    parser.add_argument("--ntheta", type=int, default=19)
    parser.add_argument("--quad", type=int, choices=[4,7,13], default=7)
    parser.add_argument("--duffy-order", type=int, default=4)
    parser.add_argument("--skip-cubature", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT/"runs/tmatrix_reference")
    args = parser.parse_args()
    if not 1 <= args.order <= 8 or args.ntheta < 2:
        parser.error("this reference supports orders 1..8 and ntheta >= 2")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    order = args.order
    band = args.angular_band or (2*order+4)
    mu, w = np.polynomial.legendre.leggauss(band)
    phi = np.arange(2*band)*np.pi/band
    z, azimuth = np.meshgrid(mu, phi, indexing="ij")
    directions = np.stack((np.sqrt(1-z*z)*np.cos(azimuth),
                           np.sqrt(1-z*z)*np.sin(azimuth), z), axis=-1).reshape(-1, 3)
    weights = np.repeat(w, 2*band)*np.pi/band
    quadrature_points = len(directions)
    # These directions are excluded from both projection quadratures, providing
    # independent incident/outgoing directions for the reconstruction check.
    rng = np.random.default_rng(193)
    checks = rng.normal(size=(9, 3))
    checks /= np.linalg.norm(checks, axis=1)[:, None]
    directions = np.vstack((directions, checks, [[0., 0., 1.]]))
    weights = np.concatenate((weights, np.zeros(10)))
    _, _, et, ep = tangent_frame(directions)
    grid = np.concatenate((directions, et, ep), axis=1)
    grid.tofile(out/"grid.bin")
    prefix = out/"operator"
    command = [str(ROOT/"bin/muller_tmatrix_probe"), args.shape, str(args.ref),
               str(args.ka), str(args.ri), str(out/"grid.bin"), str(prefix), str(args.quad), str(args.duffy_order)]
    signature = hashlib.sha256(grid.tobytes() +
        json.dumps([args.shape,args.ref,args.ka,args.ri,args.quad,args.duffy_order]).encode() +
        (ROOT/"bin/muller_tmatrix_probe").read_bytes()).hexdigest()
    cache_path = out/"bridge.signature"
    cached = cache_path.exists() and cache_path.read_text().strip() == signature
    print("Reusing dense BEM reference matrices" if cached else
          "Building dense BEM reference matrices", flush=True)
    start = time.perf_counter()
    if not cached:
        with (out/"bridge.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        cache_path.write_text(signature+"\n")
    bridge_wall = time.perf_counter()-start
    meta = json.loads(prefix.with_suffix('.json').read_text())
    n = meta["dofs"]
    channel_count = 2*len(directions)
    a = np.fromfile(str(prefix)+".A.bin", dtype=np.complex128).reshape(n,n)
    b = np.fromfile(str(prefix)+".B.bin", dtype=np.complex128).reshape(n,channel_count)
    radiation = np.fromfile(str(prefix)+".D.bin", dtype=np.complex128).reshape(channel_count,n)
    print(f"BEM: {n} unknowns; T: {len(modes(order))} channels", flush=True)
    start = time.perf_counter()
    lu = lu_factor(a)
    lu_seconds = time.perf_counter()-start
    start = time.perf_counter()
    v = transverse_harmonics(directions, order)
    ww = np.repeat(weights, 2)
    gram_error = relative(v.conj().T @ (ww[:, None]*v), np.eye(v.shape[1]))
    modal_rhs = b @ (ww[:, None]*v)
    project_radiation = v.conj().T @ (ww[:, None]*radiation)
    projection_seconds = time.perf_counter()-start
    start = time.perf_counter()
    currents = lu_solve(lu, modal_rhs)
    modal_solve_seconds = time.perf_counter()-start
    start = time.perf_counter()
    kernel = project_radiation @ currents
    phase = np.array([1j**(l-p) for p,l,m in modes(order)])
    t_matrix = -1j*args.ka/(4*np.pi)*phase[:,None]*kernel*phase[None,:].conj()
    transform_seconds = time.perf_counter()-start
    solve_residual = relative(a @ currents, modal_rhs)
    # Same BEM operator, independent plane waves and observations; no modal RHS
    # construction or spherical projection enters the direct reference solve.
    check_start = 2*quadrature_points
    start = time.perf_counter()
    direct_currents = lu_solve(lu, b[:, check_start:])
    check_solve_seconds = time.perf_counter()-start
    direct_field = radiation[check_start:] @ direct_currents
    modal_field = v[check_start:] @ kernel @ v[check_start:].conj().T
    reconstruction_error = relative(modal_field, direct_field)
    theta = np.linspace(0,180,args.ntheta)
    start = time.perf_counter()
    mueller = args.ka**2*analytic_mueller(kernel, order, theta)
    analytic_seconds = time.perf_counter()-start
    numerical_seconds = None
    cubature_error = None
    if not args.skip_cubature:
        print("Checking analytical SO(3) mean against independent cubature", flush=True)
        start = time.perf_counter()
        numerical = args.ka**2*numerical_so3_mueller(kernel, order, theta)
        numerical_seconds = time.perf_counter()-start
        cubature_error = relative(mueller, numerical)
        if cubature_error > 1e-10:
            raise RuntimeError(f"SO(3) averaging mismatch: {cubature_error}")
    csca = 2*np.pi/args.ka**2*np.linalg.norm(t_matrix)**2
    cext = -2*np.pi/args.ka**2*np.trace(t_matrix).real
    zz,ww = np.polynomial.legendre.leggauss(2*order+1)
    integrated_mueller = args.ka**2*analytic_mueller(kernel,order,np.rad2deg(np.arccos(zz)))
    integrated_csca = 2*np.pi/args.ka**2*np.dot(ww,integrated_mueller[:,0,0])
    integrated_trace_error = float(abs(integrated_csca/csca-1))
    if integrated_trace_error > 1e-11:
        raise RuntimeError("Mueller angular integral disagrees with T-matrix trace")
    mie = None
    if args.shape == "sphere":
        target = np.moveaxis(np.asarray(mie_mueller(theta.tolist(), args.ri, args.ka)), -1, 0)
        aa, bb, last = mie_coefficients(args.ri,args.ka)
        mie_csca = 2*np.pi/args.ka**2*sum((2*l+1)*(abs(aa[l])**2+abs(bb[l])**2) for l in range(1,last+1))
        mie = {"full_mueller_relative_error": relative(mueller,target),
               "tmatrix_relative_error": relative(t_matrix, np.diag([-bb[l] if p==0 else -aa[l] for p,l,m in modes(order)])),
               "csca": mie_csca, "csca_relative_error": float(abs(csca/mie_csca-1)),
               "cext_relative_error": float(abs(cext/mie_csca-1))}
    np.savez(out/"tmatrix.npz", T=t_matrix, kernel=kernel, modes=np.asarray(modes(order)),
             theta=theta, mueller=mueller)
    t_specific = projection_seconds+modal_solve_seconds+transform_seconds
    # Solves only: the actual direct path also has RHS and radiation costs.
    # This estimate is diagnostic, not a production GPU break-even benchmark.
    per_plane_solve = check_solve_seconds/(channel_count-check_start)
    report = {
        "base_commit":"aa66bfe1bc84b0288dfd731f68b2ca1fab05424c",
        "implementation":"dense CPU reference; full analytic uniform SO(3) Mueller average",
        "bridge_cached":cached,
        "shape":args.shape,"ka":args.ka,"ri":args.ri,"ref":args.ref,
        "bounding_radius":meta["bounding_radius"],"quad":args.quad,"duffy_order":args.duffy_order,
        "order":order,"multipole_channels":len(modes(order)),"system_dofs":n,
        "quadrature_directions":quadrature_points,"theta_angles":args.ntheta,
        "timing":{"bridge_wall_s":bridge_wall,"assembly_s":meta["assembly_s"],
                  "radiation_matrix_s":meta["radiation_s"],"plane_rhs_matrix_s":meta["rhs_s"],
                  "common_lu_s":lu_seconds,"modal_projection_s":projection_seconds,
                  "modal_solve_s":modal_solve_seconds,"transform_s":transform_seconds,
                  "t_specific_after_bridge_and_lu_s":t_specific,
                  "analytic_so3_s":analytic_seconds,"numerical_t_so3_s":numerical_seconds,
                  "direct_check_solve_s":check_solve_seconds},
        "accuracy":{"radiation_matrix_vs_library":meta["radiation_check_relative_error"],
                    "vector_harmonic_gram_error":gram_error,"modal_true_residual":solve_residual,
                    "new_direction_farfield_relative_error":reconstruction_error,
                    "analytic_vs_cubature_full_mueller":cubature_error,
                    "integrated_mueller_vs_trace":integrated_trace_error,
                    "mie":mie},
        "cross_sections":{"scattering":float(csca),"extinction":float(cext),
                          "energy_balance_relative_error":float(abs(cext-csca)/csca)},
        "cost_estimate":{"direct_backsolve_s_per_plane_wave":per_plane_solve,
                         "t_specific_solve_equivalents":t_specific/per_plane_solve,
                         "scope":"CPU diagnostic; bridge cost excluded here and reported separately; no GPU speedup estimate"},
        "theta":theta.tolist(),"mueller":mueller.tolist(),
    }
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/f"summary_L{order}.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:val for k,val in report.items() if k not in ("mueller","theta")},indent=2),flush=True)
    print(out/"summary.json",flush=True)


if __name__ == "__main__":
    main()
