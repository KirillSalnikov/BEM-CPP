#!/usr/bin/env python3
"""Benchmark hexagonal-prism orientation scaling for BEM-CUDA and ADDA-OCL.

The benchmark uses explicit Euler-angle files so both solvers see the same
alpha, beta, gamma grid.  ADDA-OCL is run as one process per orientation and
scheduled across the selected GPUs.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bemcuda.gpu_guard import parse_gpu_csv, select_free_gpus


def parse_gpus(spec):
    return parse_gpu_csv(spec)


def parse_levels(spec):
    levels = []
    for item in spec.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parts = item.replace("x", " ").split()
        if len(parts) != 3:
            raise SystemExit(f"bad level '{item}', expected Na x Nb x Ng")
        levels.append(tuple(int(x) for x in parts))
    if not levels:
        raise SystemExit("no levels selected")
    return levels


def orientation_grid(na, nb, ng):
    rows = []
    for ia in range(na):
        alpha = 360.0 * ia / na
        for ib in range(nb):
            beta = 90.0 if nb == 1 else 180.0 * (ib + 0.5) / nb
            for ig in range(ng):
                gamma = 360.0 * ig / ng
                rows.append((alpha, beta, gamma, 1.0))
    return rows


def rotation_matrix(alpha_deg, beta_deg, gamma_deg):
    a = math.radians(alpha_deg)
    b = math.radians(beta_deg)
    g = math.radians(gamma_deg)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    rz1 = ((ca, -sa, 0.0), (sa, ca, 0.0), (0.0, 0.0, 1.0))
    ry = ((cb, 0.0, sb), (0.0, 1.0, 0.0), (-sb, 0.0, cb))
    rz2 = ((cg, -sg, 0.0), (sg, cg, 0.0), (0.0, 0.0, 1.0))

    def mm(x, y):
        return tuple(tuple(sum(x[i][k] * y[k][j] for k in range(3)) for j in range(3)) for i in range(3))

    return mm(rz1, mm(ry, rz2))


def nearest_orientation_order(rows):
    if len(rows) <= 2:
        return rows
    rotations = [rotation_matrix(a, b, g) for a, b, g, _ in rows]
    used = [False] * len(rows)
    ordered = []
    cur = 0
    for _ in range(len(rows)):
        ordered.append(rows[cur])
        used[cur] = True
        best = None
        best_d = None
        r0 = rotations[cur]
        for j, r1 in enumerate(rotations):
            if used[j]:
                continue
            d = sum((r0[i][k] - r1[i][k]) ** 2 for i in range(3) for k in range(3))
            if best_d is None or d < best_d:
                best = j
                best_d = d
        if best is None:
            break
        cur = best
    return ordered


def write_orient_file(path, na, nb, ng):
    rows = nearest_orientation_order(orientation_grid(na, nb, ng))
    with open(path, "w") as f:
        f.write("# alpha_deg beta_deg gamma_deg weight; nearest-neighbor order for BEM warm start\n")
        for row in rows:
            f.write("%.12g %.12g %.12g %.12g\n" % row)
    return rows


def read_bem_wall(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return float(data["timing"]["total_s"])


def run_checked(cmd, log_path, cwd=None, env=None):
    with open(log_path, "w") as log:
        t0 = time.time()
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    return rc, time.time() - t0


def run_bem(args, out_dir, level_name, orient_file, n_orient):
    out_json = out_dir / f"bem_{level_name}.json"
    work_dir = out_dir / f"bem_{level_name}_parts"
    log_path = out_dir / f"bem_{level_name}.log"
    if out_json.exists() and not args.force:
        return read_bem_wall(out_json), "cached"

    cmd = [
        sys.executable, str(args.bem_queue),
        "--exe", str(args.bem_exe),
        "--out", str(out_json),
        "--work-dir", str(work_dir),
        "--gpus", ",".join(args.selected_gpus),
        "--chunk-size", str(min(args.bem_chunk_size, n_orient)),
        "--ka", str(args.ka),
        "--ri", str(args.ri_real), str(args.ri_imag),
        "--shape", "hex_prism",
        "--prism-aspect", str(args.prism_aspect),
        "--ref", str(args.ref),
        "--quad", str(args.quad),
        "--ntheta", str(args.ntheta),
        "--scat-plane", args.scat_plane,
        "--orient-file", str(orient_file),
        "--solver", args.bem_solver,
        "--omp-threads", str(args.omp_threads),
    ]
    if args.bem_system:
        cmd += ["--system", args.bem_system]
    if args.fmm_digits:
        cmd += ["--fmm-digits", str(args.fmm_digits)]
    if args.gmres_tol:
        cmd += ["--gmres-tol", str(args.gmres_tol)]
    if args.gmres_restart:
        cmd += ["--gmres-restart", str(args.gmres_restart)]
    if args.max_leaf:
        cmd += ["--max-leaf", str(args.max_leaf)]
    if args.no_prec:
        cmd += ["--no-prec"]
    if args.bem_orient_warm_start:
        cmd += ["--orient-warm-start", args.bem_orient_warm_start]
    if args.bem_orient_recycle:
        cmd += ["--orient-recycle", str(args.bem_orient_recycle)]

    env = os.environ.copy()
    env.setdefault("BEM_FAST_REORTH_OFF", "1")
    rc, wall = run_checked(cmd, log_path, env=env)
    if rc != 0:
        raise SystemExit(f"BEM failed for {level_name}; see {log_path}")
    return read_bem_wall(out_json), "ran"


def adda_command(args, out_dir, gpu, alpha, beta, gamma):
    return [
        str(args.adda_exe),
        "-gpu", str(gpu),
        "-dir", str(out_dir),
        "-shape", "prism", "6", str(args.prism_aspect),
        "-m", str(args.ri_real), str(args.ri_imag),
        "-dpl", str(args.adda_dpl),
        "-eps", str(args.adda_eps),
        "-orient", "%.12g" % alpha, "%.12g" % beta, "%.12g" % gamma,
        "-ntheta", str(args.ntheta),
        "-scat_matr", "muel",
        "-sym", "no",
        "-eq_rad", str(args.ka),
    ]


def run_adda(args, out_dir, level_name, rows, gpus):
    level_dir = out_dir / f"adda_{level_name}"
    summary_path = level_dir / "orientation_times.csv"
    if summary_path.exists() and not args.force:
        with open(summary_path, "r") as f:
            records = list(csv.DictReader(f))
        wall = max(float(r["end_s"]) for r in records) if records else 0.0
        total = sum(float(r["wall_s"]) for r in records)
        return wall, total, "cached"

    level_dir.mkdir(parents=True, exist_ok=True)
    pending = list(enumerate(rows))
    running = []
    done = []
    t0 = time.time()

    def launch(gpu, index, row):
        run_dir = level_dir / f"orient_{index:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "adda.log"
        cmd = adda_command(args, run_dir, gpu, row[0], row[1], row[2])
        env = os.environ.copy()
        log = open(log_path, "w")
        start = time.time()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        return {
            "gpu": gpu,
            "index": index,
            "angles": row,
            "log": log,
            "log_path": log_path,
            "proc": proc,
            "start": start,
        }

    for gpu in gpus:
        if not pending:
            break
        index, row = pending.pop(0)
        running.append(launch(gpu, index, row))

    while running:
        time.sleep(0.5)
        for item in list(running):
            rc = item["proc"].poll()
            if rc is None:
                continue
            item["log"].close()
            running.remove(item)
            end = time.time()
            if rc != 0:
                raise SystemExit(f"ADDA failed for {level_name} orient {item['index']}; see {item['log_path']}")
            done.append({
                "index": item["index"],
                "gpu": item["gpu"],
                "alpha": item["angles"][0],
                "beta": item["angles"][1],
                "gamma": item["angles"][2],
                "start_s": item["start"] - t0,
                "end_s": end - t0,
                "wall_s": end - item["start"],
            })
            if pending:
                index, row = pending.pop(0)
                running.append(launch(item["gpu"], index, row))

    done.sort(key=lambda r: r["index"])
    with open(summary_path, "w", newline="") as f:
        fields = ["index", "gpu", "alpha", "beta", "gamma", "start_s", "end_s", "wall_s"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in done:
            writer.writerow(row)
    wall = time.time() - t0
    total = sum(float(r["wall_s"]) for r in done)
    return wall, total, "ran"


def write_summary(path, rows):
    fields = [
        "level", "na", "nb", "ng", "n_orient",
        "bem_wall_s", "adda_wall_s", "adda_total_gpu_s",
        "bem_orient_per_wall_s", "adda_orient_per_wall_s",
        "adda_over_bem_wall", "bem_status", "adda_status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_summary(csv_path, png_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped: {exc}", file=sys.stderr)
        return

    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    x = [int(r["n_orient"]) for r in rows]
    bem = [float(r["bem_wall_s"]) for r in rows]
    adda = [float(r["adda_wall_s"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=180)
    ax.plot(x, bem, "o-", label="BEM-CUDA")
    ax.plot(x, adda, "s-", label="ADDA-OCL")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Euler orientations")
    ax.set_ylabel("Wall time, s")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.set_title("Hexagonal prism orientation-scaling benchmark")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/hex_euler_scaling_20260623")
    p.add_argument("--levels", default="1x1x1,2x2x2,4x2x2,4x4x2,4x4x4")
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--allow-compute-share", action="store_true",
                   help="Allow starting while nvidia-smi reports existing CUDA compute processes")
    p.add_argument("--bem-exe", default="./bin/bem_cuda_fmm")
    p.add_argument("--bem-queue", default="./run_orient_queue.py")
    p.add_argument("--adda-exe", default=str(Path.home() / "adda/src/ocl/adda_ocl"))
    p.add_argument("--ka", type=float, default=10.0)
    p.add_argument("--ri-real", type=float, default=1.3116)
    p.add_argument("--ri-imag", type=float, default=0.0)
    p.add_argument("--prism-aspect", type=float, default=1.5)
    p.add_argument("--ref", type=int, default=3)
    p.add_argument("--quad", type=int, default=4)
    p.add_argument("--ntheta", type=int, default=181)
    p.add_argument("--scat-plane", default="yz")
    p.add_argument("--bem-solver", default="fmm")
    p.add_argument("--bem-system", default="")
    p.add_argument("--fmm-digits", default="3")
    p.add_argument("--gmres-tol", default="1e-2")
    p.add_argument("--gmres-restart", default="120")
    p.add_argument("--max-leaf", default="128")
    p.add_argument("--no-prec", action="store_true")
    p.add_argument("--bem-chunk-size", type=int, default=8)
    p.add_argument("--bem-orient-warm-start", choices=["zero", "previous", "recycle"], default="zero")
    p.add_argument("--bem-orient-recycle", type=int, default=8)
    p.add_argument("--omp-threads", type=int, default=8)
    p.add_argument("--adda-dpl", type=int, default=20)
    p.add_argument("--adda-eps", default="5")
    p.add_argument("--bem-only", action="store_true")
    p.add_argument("--adda-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.bem_only and args.adda_only:
        raise SystemExit("--bem-only and --adda-only are mutually exclusive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpus = select_free_gpus(args.gpus, args.allow_compute_share)
    args.selected_gpus = gpus
    levels = parse_levels(args.levels)
    summary = []

    for na, nb, ng in levels:
        level_name = f"{na}x{nb}x{ng}"
        orient_file = out_dir / f"orient_{level_name}.txt"
        rows = write_orient_file(orient_file, na, nb, ng)
        n_orient = len(rows)
        print(f"[{level_name}] {n_orient} orientations", flush=True)

        bem_wall = math.nan
        bem_status = "skipped"
        if not args.adda_only:
            bem_wall, bem_status = run_bem(args, out_dir, level_name, orient_file, n_orient)
            print(f"[{level_name}] BEM {bem_wall:.3f}s ({bem_status})", flush=True)

        adda_wall = math.nan
        adda_total = math.nan
        adda_status = "skipped"
        if not args.bem_only:
            adda_wall, adda_total, adda_status = run_adda(args, out_dir, level_name, rows, gpus)
            print(f"[{level_name}] ADDA {adda_wall:.3f}s wall, {adda_total:.3f}s gpu-s ({adda_status})", flush=True)

        row = {
            "level": level_name,
            "na": na,
            "nb": nb,
            "ng": ng,
            "n_orient": n_orient,
            "bem_wall_s": bem_wall,
            "adda_wall_s": adda_wall,
            "adda_total_gpu_s": adda_total,
            "bem_orient_per_wall_s": bem_wall / n_orient if not math.isnan(bem_wall) else math.nan,
            "adda_orient_per_wall_s": adda_wall / n_orient if not math.isnan(adda_wall) else math.nan,
            "adda_over_bem_wall": adda_wall / bem_wall if not math.isnan(adda_wall) and not math.isnan(bem_wall) else math.nan,
            "bem_status": bem_status,
            "adda_status": adda_status,
        }
        summary.append(row)
        write_summary(out_dir / "summary.csv", summary)
        if not args.bem_only and not args.adda_only:
            plot_summary(out_dir / "summary.csv", out_dir / "scaling.png")

    print(f"Wrote {out_dir / 'summary.csv'}")
    if not args.bem_only and not args.adda_only:
        print(f"Wrote {out_dir / 'scaling.png'}")


if __name__ == "__main__":
    main()
