#!/usr/bin/env python3
"""Run BEM-CUDA on a sphere and compare M11 against Mie theory."""
import argparse
import cmath
import json
import math
import os
import subprocess
import sys


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
    a, b, nstop = mie_coefficients(m, x)
    out = []
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
        out.append(0.5 * (abs(s1) ** 2 + abs(s2) ** 2))
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


def compare(out_file, n_re, n_im, ka):
    with open(out_file) as f:
        data = json.load(f)
    theta = data["theta"]
    bem = data["mueller"][0][0]
    mie = mie_m11(theta, complex(n_re, n_im), ka)

    num = sum(b * m for b, m in zip(bem, mie))
    den = sum(m * m for m in mie)
    scale = num / den if den > 0 else 1.0
    mie_scaled = [scale * v for v in mie]

    rel = [
        abs(b - m) / max(abs(b), 1e-30)
        for b, m in zip(bem, mie_scaled)
        if abs(b) > 1e-12
    ]
    mean_rel = sum(rel) / len(rel)
    max_rel = max(rel)
    i_max = max(range(len(bem)), key=lambda i: abs(bem[i] - mie_scaled[i]) / max(abs(bem[i]), 1e-30))

    print("\nMie check: M11")
    print(f"  scale(BEM/Mie) = {scale:.8e}")
    print(f"  mean rel err   = {mean_rel:.4e}")
    print(f"  max  rel err   = {max_rel:.4e} at theta={theta[i_max]:.2f} deg")
    print("\n  theta       BEM M11       Mie M11      rel err")
    for target in (0, 30, 60, 90, 120, 150, 180):
        idx = min(range(len(theta)), key=lambda i: abs(theta[i] - target))
        err = abs(bem[idx] - mie_scaled[idx]) / max(abs(bem[idx]), 1e-30)
        print(f"  {theta[idx]:6.1f}  {bem[idx]:12.5e}  {mie_scaled[idx]:12.5e}  {err:9.3e}")
    return mean_rel, max_rel


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
