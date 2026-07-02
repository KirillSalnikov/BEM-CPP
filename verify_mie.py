#!/usr/bin/env python3
"""Run BEM-CUDA on a sphere and compare its Mueller matrix against Mie theory."""
import argparse
import cmath
import json
import math
import os
import subprocess
import sys


ALL_MUELLER = [(f"M{i + 1}{j + 1}", i, j) for i in range(4) for j in range(4)]
MAIN_GATE_MUELLER = [("M11", 0, 0), ("M12", 0, 1), ("M21", 1, 0),
                     ("M34", 2, 3), ("M43", 3, 2), ("M44", 3, 3)]


def riccati_psi(nmax, z):
    psi = [0j] * (nmax + 1)
    psi[0] = cmath.sin(z)
    if nmax >= 1:
        psi[1] = cmath.sin(z) / z - cmath.cos(z)
    for n in range(1, nmax):
        psi[n + 1] = (2 * n + 1) * psi[n] / z - psi[n - 1]
    return psi


def riccati_chi(nmax, z):
    chi = [0j] * (nmax + 1)
    chi[0] = cmath.cos(z)
    if nmax >= 1:
        chi[1] = cmath.cos(z) / z + cmath.sin(z)
    for n in range(1, nmax):
        chi[n + 1] = (2 * n + 1) * chi[n] / z - chi[n - 1]
    return chi


def mie_coefficients(m, x):
    nstop = int(round(x + 4.0 * x ** (1.0 / 3.0) + 2.0))
    nmx = max(nstop + 16, int(abs(m * x)) + 16)
    mx = m * x

    D = [0j] * (nmx + 1)
    for n in range(nmx, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)

    psi = riccati_psi(nstop, x)
    chi = riccati_chi(nstop, x)
    xi = [psi[n] - 1j * chi[n] for n in range(nstop + 1)]

    a = [0j] * (nstop + 1)
    b = [0j] * (nstop + 1)
    for n in range(1, nstop + 1):
        dn = D[n]
        anx = n / x
        a_num = (dn / m + anx) * psi[n] - psi[n - 1]
        a_den = (dn / m + anx) * xi[n] - xi[n - 1]
        b_num = (m * dn + anx) * psi[n] - psi[n - 1]
        b_den = (m * dn + anx) * xi[n] - xi[n - 1]
        a[n] = a_num / a_den
        b[n] = b_num / b_den
    return a, b, nstop


def mie_m11(theta_deg, m, x):
    mueller = mie_mueller(theta_deg, m, x)
    return mueller[0][0]


def mie_amplitudes(theta_deg, m, x):
    a, b, nstop = mie_coefficients(m, x)
    s1_out = []
    s2_out = []
    for th in theta_deg:
        mu = math.cos(math.radians(th))
        pi_nm1 = 0.0
        pi_n = 1.0
        s1 = 0j
        s2 = 0j
        for n in range(1, nstop + 1):
            tau_n = n * mu * pi_n - (n + 1) * pi_nm1
            coef = (2 * n + 1) / (n * (n + 1))
            s1 += coef * (a[n] * pi_n + b[n] * tau_n)
            s2 += coef * (a[n] * tau_n + b[n] * pi_n)
            pi_np1 = ((2 * n + 1) * mu * pi_n - (n + 1) * pi_nm1) / n
            pi_nm1, pi_n = pi_n, pi_np1
        s1_out.append(s1)
        s2_out.append(s2)
    return s1_out, s2_out


def mie_mueller(theta_deg, m, x):
    s1, s2 = mie_amplitudes(theta_deg, m, x)
    ntheta = len(theta_deg)
    out = [[[0.0 for _ in range(ntheta)] for _ in range(4)] for _ in range(4)]
    for t in range(ntheta):
        as1 = abs(s1[t]) ** 2
        as2 = abs(s2[t]) ** 2
        s1s2c = s1[t] * s2[t].conjugate()
        out[0][0][t] = 0.5 * (as1 + as2)
        out[0][1][t] = 0.5 * (as2 - as1)
        out[1][0][t] = out[0][1][t]
        out[1][1][t] = out[0][0][t]
        out[2][2][t] = s1s2c.real
        out[2][3][t] = -s1s2c.imag
        out[3][2][t] = s1s2c.imag
        out[3][3][t] = s1s2c.real
    return out


def run_solver(args):
    cmd = [
        args.exe,
        "--ka", str(args.ka),
        "--ri", str(args.n_re), str(args.n_im),
        "--ref", str(args.ref),
        "--ntheta", str(args.ntheta),
        "--quad", str(args.quad),
        "--single",
        "--out", args.out,
    ]
    if args.fmm:
        cmd.extend([
            "--fmm",
            "--fmm-digits", str(args.fmm_digits),
            "--gmres-tol", str(args.gmres_tol),
            "--gmres-restart", str(args.gmres_restart),
            "--max-leaf", str(args.max_leaf),
        ])
    if args.no_prec:
        cmd.append("--no-prec")

    env = os.environ.copy()
    if args.cuda_lib:
        env["LD_LIBRARY_PATH"] = args.cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def mueller_component(mueller, i, j):
    shape = getattr(mueller, "shape", None)
    if shape is not None:
        shape = tuple(shape)
        if len(shape) == 3 and shape[0] == 4 and shape[1] == 4:
            return [float(v) for v in mueller[i, j, :]]
        if len(shape) == 2 and shape[0] == 16:
            return [float(v) for v in mueller[4 * i + j, :]]
        if len(shape) == 2 and shape[1] == 16:
            return [float(row[4 * i + j]) for row in mueller]
    if (len(mueller) == 4 and isinstance(mueller[0], list) and
            len(mueller[0]) == 4 and isinstance(mueller[0][0], list)):
        return [float(v) for v in mueller[i][j]]
    if len(mueller) == 16 and isinstance(mueller[0], list):
        return [float(v) for v in mueller[4 * i + j]]
    if mueller and isinstance(mueller[0], list) and len(mueller[0]) == 16:
        return [float(row[4 * i + j]) for row in mueller]
    raise ValueError("unknown mueller layout; expected 4x4xN, 16xN, or Nx16")


def component_floor2_errors(theta, bem_mueller, ref_mueller):
    bem_norm = max(abs(mueller_component(bem_mueller, 0, 0)[0]), 1e-300)
    ref_norm = max(abs(ref_mueller[0][0][0]), 1e-300)
    out = {}
    for label, i, j in ALL_MUELLER:
        bem = [v / bem_norm for v in mueller_component(bem_mueller, i, j)]
        ref = [v / ref_norm for v in ref_mueller[i][j]]
        errs = [
            abs(b - r) / max(abs(r), 0.02)
            for b, r in zip(bem, ref)
        ]
        out[label] = sum(errs) / max(len(theta), 1)
    return out


def compare(out_file, n_re, n_im, ka):
    with open(out_file) as f:
        data = json.load(f)
    theta = data["theta"]
    bem_mueller = data["mueller"]
    mie = mie_mueller(theta, complex(n_re, n_im), ka)
    errors = component_floor2_errors(theta, bem_mueller, mie)
    worst_name, worst_error = max(errors.items(), key=lambda item: item[1])
    main_errors = [errors[name] for name, _, _ in MAIN_GATE_MUELLER]
    failed_main_10 = [name for name, _, _ in MAIN_GATE_MUELLER if errors[name] > 0.10]
    failed_all_20 = [name for name, _, _ in ALL_MUELLER if errors[name] > 0.20]

    bem_m11 = mueller_component(bem_mueller, 0, 0)
    mie_m11_ref = mie[0][0]
    bem_norm = max(abs(bem_m11[0]), 1e-300)
    mie_norm = max(abs(mie_m11_ref[0]), 1e-300)

    print("\nMie check: full Mueller matrix")
    print("  normalization = each matrix divided by M11(theta=0)")
    print(f"  max main floor2 err = {max(main_errors):.4e}")
    print(f"  worst component     = {worst_name} ({worst_error:.4e})")
    if failed_main_10:
        print(f"  failed main >10%    = {','.join(failed_main_10)}")
    if failed_all_20:
        print(f"  failed all >20%     = {','.join(failed_all_20)}")
    print("\n  theta       BEM M11       Mie M11      rel err")
    for target in (0, 30, 60, 90, 120, 150, 180):
        idx = min(range(len(theta)), key=lambda i: abs(theta[i] - target))
        b = bem_m11[idx] / bem_norm
        m = mie_m11_ref[idx] / mie_norm
        err = abs(b - m) / max(abs(m), 0.02)
        print(f"  {theta[idx]:6.1f}  {b:12.5e}  {m:12.5e}  {err:9.3e}")
    return {
        "max_main_floor2": max(main_errors),
        "worst_component": worst_name,
        "worst_component_error": worst_error,
        "failed_main_10pct": failed_main_10,
        "failed_all_20pct": failed_all_20,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", default="bin/bem_cuda")
    p.add_argument("--out", default="mie_check.json")
    p.add_argument("--ka", type=float, default=1.0)
    p.add_argument("--ri", dest="n_re", type=float, default=1.3116)
    p.add_argument("--n-im", type=float, default=0.0)
    p.add_argument("--ref", type=int, default=2)
    p.add_argument("--ntheta", type=int, default=181)
    p.add_argument("--quad", type=int, default=7)
    p.add_argument("--fmm", action="store_true")
    p.add_argument("--fmm-digits", type=int, default=3)
    p.add_argument("--gmres-tol", type=float, default=1e-4)
    p.add_argument("--gmres-restart", type=int, default=100)
    p.add_argument("--max-leaf", type=int, default=64)
    p.add_argument("--no-prec", action="store_true")
    p.add_argument("--cuda-lib", default="")
    p.add_argument("--skip-run", action="store_true")
    args = p.parse_args()

    if not args.skip_run:
        run_solver(args)
    compare(args.out, args.n_re, args.n_im, args.ka)


if __name__ == "__main__":
    sys.exit(main())
