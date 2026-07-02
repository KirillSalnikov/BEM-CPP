#!/usr/bin/env python3
"""Benchmark BEM-CUDA against ADDA for spheres and compare M11 with Mie."""
import argparse
import json
import os
import re
import subprocess
import time

from verify_mie import mie_m11


def run(cmd, env=None, timeout=None, check=True):
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, env=env, check=check,
                              timeout=timeout)
        return proc.stdout, time.time() - t0, proc.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        return stdout, time.time() - t0, None, True


def rel_stats(theta, values, ka, n_re, n_im):
    mie = mie_m11(theta, complex(n_re, n_im), ka)
    scale = sum(v * m for v, m in zip(values, mie)) / sum(m * m for m in mie)
    scaled = [scale * m for m in mie]
    rel = [abs(v - m) / max(abs(v), 1e-30) for v, m in zip(values, scaled) if abs(v) > 1e-12]
    return scale, sum(rel) / len(rel), max(rel)


def rel_stats_against(ref_values, values):
    scale = sum(v * r for v, r in zip(values, ref_values)) / sum(r * r for r in ref_values)
    scaled = [scale * r for r in ref_values]
    rel = [abs(v - r) / max(abs(v), 1e-30) for v, r in zip(values, scaled) if abs(v) > 1e-12]
    l2_num = sum((v - r) * (v - r) for v, r in zip(values, scaled))
    l2_den = sum(v * v for v in values)
    l2 = (l2_num / max(l2_den, 1e-300)) ** 0.5
    return scale, l2, sum(rel) / len(rel), max(rel)


def run_bem(args, ka, out_dir):
    out_file = os.path.join(out_dir, f"bem_ka{ka:g}.json")
    cmd = [
        args.bem_exe,
        "--ka", str(ka),
        "--ri", str(args.n_re), str(args.n_im),
        "--shape", args.shape,
        "--ref", str(args.bem_ref),
        "--ntheta", str(args.ntheta),
        "--single",
        "--solver", args.bem_solver,
        "--out", out_file,
    ]
    if args.bem_quad is not None:
        cmd += ["--quad", str(args.bem_quad)]
    if args.fmm_digits is not None:
        cmd += ["--fmm-digits", str(args.fmm_digits)]
    if args.gmres_tol is not None:
        cmd += ["--gmres-tol", str(args.gmres_tol)]
    if args.max_leaf is not None:
        cmd += ["--max-leaf", str(args.max_leaf)]
    if args.gmres_restart is not None:
        cmd += ["--gmres-restart", str(args.gmres_restart)]
    if args.bem_accurate:
        cmd += ["--accurate"]
    if args.shape == "hex_prism":
        cmd += ["--prism-aspect", str(args.prism_aspect)]
        if args.edge_refine is not None:
            cmd += ["--edge-refine", str(args.edge_refine)]
    if args.scat_plane:
        cmd += ["--scat-plane", args.scat_plane]
    env = os.environ.copy()
    if args.cuda_lib:
        env["LD_LIBRARY_PATH"] = args.cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
    stdout, elapsed, rc, timed_out = run(cmd, env=env, timeout=args.bem_timeout)
    if timed_out:
        raise RuntimeError(f"BEM timed out after {args.bem_timeout}s for ka={ka:g}")
    if rc != 0:
        raise RuntimeError(f"BEM failed with rc={rc} for ka={ka:g}\n{stdout[-4000:]}")
    data = json.load(open(out_file))
    total_s = data.get("timing", {}).get("total_s", elapsed)
    solve_s = data.get("timing", {}).get("solve_s", 0.0)
    m = re.search(r"Both converged, (\d+) matvec", stdout)
    matvecs = int(m.group(1)) if m else None
    theta = data["theta"]
    m11 = data["mueller"][0][0]
    scale, mean_err, max_err = rel_stats(theta, m11, ka, args.n_re, args.n_im)
    return {
        "method": "BEM-FMM",
        "ka": ka,
        "ref": args.bem_ref,
        "quad": args.bem_quad,
        "time_s": total_s,
        "solve_s": solve_s,
        "matvecs": matvecs,
        "m11_mie_scale": scale,
        "m11_mean_err": mean_err,
        "m11_max_err": max_err,
        "theta": theta,
        "m11": m11,
    }


def parse_adda_mueller(path):
    theta = []
    m11 = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("theta"):
                continue
            parts = line.split()
            theta.append(float(parts[0]))
            m11.append(float(parts[1]))
    return theta, m11


def run_adda(args, ka, dpl, out_dir):
    run_dir = os.path.join(out_dir, f"adda_ka{ka:g}_dpl{dpl:g}")
    adda_shape = ["sphere"]
    if args.shape == "hex_prism":
        adda_shape = ["prism", "6", str(args.prism_aspect)]
    cmd = [
        args.adda_exe,
        "-dir", run_dir,
        "-shape",
    ] + adda_shape + [
        "-eq_rad", str(ka),
        "-m", str(args.n_re), str(args.n_im),
        "-dpl", str(dpl),
        "-ntheta", str(args.ntheta),
        "-scat_matr", "muel",
        "-sym", "auto",
        "-eps", str(args.adda_eps),
        "-iter", args.adda_iter,
    ]
    stdout, elapsed, rc, timed_out = run(cmd, timeout=args.adda_timeout, check=False)
    log_path = os.path.join(run_dir, "log")
    log = open(log_path).read() if os.path.exists(log_path) else stdout
    wall = re.search(r"Total wall time:\s+([0-9.]+)", log)
    occ = re.search(r"Total number of occupied dipoles:\s+(\d+)", log)
    iters = re.search(r"Total number of iterations:\s+(\d+)", log)
    dpl_eff = re.search(r"Dipoles/lambda:\s+([0-9.]+)", log)
    res_matches = re.findall(r"RE\s*=\s*([0-9.eE+-]+)", log)
    residual = float(res_matches[-1]) if res_matches else None
    mueller_path = os.path.join(run_dir, "mueller")
    if timed_out or rc != 0 or not os.path.exists(mueller_path):
        return {
            "method": "ADDA",
            "ka": ka,
            "dpl": dpl,
            "dpl_eff": float(dpl_eff.group(1)) if dpl_eff else dpl,
            "dipoles": int(occ.group(1)) if occ else None,
            "iterations": int(iters.group(1)) if iters else None,
            "residual": residual,
            "time_s": float(wall.group(1)) if wall else elapsed,
            "status": "timeout" if timed_out else f"failed:{rc}",
        }
    theta, m11 = parse_adda_mueller(mueller_path)
    scale, mean_err, max_err = rel_stats(theta, m11, ka, args.n_re, args.n_im)
    return {
        "method": "ADDA",
        "ka": ka,
        "dpl": dpl,
        "dpl_eff": float(dpl_eff.group(1)) if dpl_eff else dpl,
        "dipoles": int(occ.group(1)) if occ else None,
        "iterations": int(iters.group(1)) if iters else None,
        "residual": residual,
        "time_s": float(wall.group(1)) if wall else elapsed,
        "status": "ok",
        "m11_mie_scale": scale,
        "m11_mean_err": mean_err,
        "m11_max_err": max_err,
        "theta": theta,
        "m11": m11,
    }


def compact(row):
    if row["method"] == "BEM-FMM":
        detail = f"ref={row['ref']} q={row['quad']} mv={row['matvecs']}"
    else:
        detail = f"dpl={row['dpl']} eff={row['dpl_eff']:.1f} dip={row['dipoles']} it={row['iterations']}"
    base = (f"{row['method']:7s} ka={row['ka']:5.2f} {detail:28s} "
            f"time={row['time_s']:8.3f}s")
    if row.get("status") and row["status"] != "ok":
        residual = row.get("residual")
        res = f" residual={residual:.3e}" if residual is not None else ""
        return f"{base} {row['status']}{res}"
    return f"{base} mean={row['m11_mean_err']:.3e} max={row['m11_max_err']:.3e}"


def compact_prism_diff(bem, adda):
    if adda.get("status") and adda["status"] != "ok":
        return (f"BEM-vs-ADDA ka={bem['ka']:5.2f} ref={bem['ref']} vs dpl={adda['dpl']} "
                f"skipped ({adda['status']})")
    scale, l2_err, mean_err, max_err = rel_stats_against(adda["m11"], bem["m11"])
    return (f"BEM-vs-ADDA ka={bem['ka']:5.2f} ref={bem['ref']} vs dpl={adda['dpl']} "
            f"scale={scale:.4e} l2={l2_err:.3e} mean_rel={mean_err:.3e} max_rel={max_err:.3e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bem-exe", default="./bin/bem_cuda_fmm")
    p.add_argument("--adda-exe", default=os.path.expanduser("~/adda/src/seq/adda"))
    p.add_argument("--cuda-lib", default="")
    p.add_argument("--out-dir", default="bench_bem_adda")
    p.add_argument("--ka", nargs="+", type=float, default=[1, 2, 4, 8])
    p.add_argument("--adda-dpl", nargs="+", type=float, default=[32, 64])
    p.add_argument("--n-re", type=float, default=1.3116)
    p.add_argument("--n-im", type=float, default=0.0)
    p.add_argument("--shape", choices=["sphere", "hex_prism"], default="sphere")
    p.add_argument("--prism-aspect", type=float, default=1.0,
                   help="For hex_prism: h/Dx, same convention as ADDA prism")
    p.add_argument("--edge-refine", type=int, default=None,
                   help="For hex_prism BEM mesh: conforming local edge-refinement passes; omit for solver auto")
    p.add_argument("--ntheta", type=int, default=61)
    p.add_argument("--scat-plane", choices=["yz", "xz"], default="yz",
                   help="BEM single-orient scattering plane; yz matches ADDA default")
    p.add_argument("--bem-ref", type=int, default=2)
    p.add_argument("--bem-quad", type=int, default=None,
                   help="BEM quadrature order; omit to use bem_cuda auto defaults")
    p.add_argument("--bem-solver", choices=["auto", "fmm", "spfft", "pfft"], default="auto")
    p.add_argument("--bem-accurate", action="store_true",
                   help="Use bem_cuda --accurate conservative hex_prism defaults")
    p.add_argument("--fmm-digits", type=int, default=None)
    p.add_argument("--gmres-tol", type=float, default=None)
    p.add_argument("--gmres-restart", type=int, default=None)
    p.add_argument("--max-leaf", type=int, default=None)
    p.add_argument("--bem-timeout", type=float, default=None,
                   help="Per-BEM-run timeout in seconds")
    p.add_argument("--adda-eps", type=float, default=5.0,
                   help="ADDA residual exponent: epsilon=10^(-value)")
    p.add_argument("--adda-iter", default="qmr")
    p.add_argument("--adda-timeout", type=float, default=None,
                   help="Per-ADDA-run timeout in seconds; timed-out rows stay in summary")
    p.add_argument("--skip-bem", action="store_true")
    p.add_argument("--skip-adda", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for ka in args.ka:
        print(f"\n=== ka={ka:g} ===", flush=True)
        bem_row = None
        adda_rows = []
        if not args.skip_bem:
            row = run_bem(args, ka, args.out_dir)
            rows.append(row)
            bem_row = row
            print(compact(row), flush=True)
        if not args.skip_adda:
            for dpl in args.adda_dpl:
                row = run_adda(args, ka, dpl, args.out_dir)
                rows.append(row)
                adda_rows.append(row)
                print(compact(row), flush=True)
        if args.shape == "hex_prism" and bem_row:
            for adda_row in adda_rows:
                print(compact_prism_diff(bem_row, adda_row), flush=True)

        summary = [{k: v for k, v in r.items() if k not in ("theta", "m11")} for r in rows]
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\nSummary")
    for row in rows:
        print(compact(row))


if __name__ == "__main__":
    main()
