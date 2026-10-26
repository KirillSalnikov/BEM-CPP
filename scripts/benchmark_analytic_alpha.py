#!/usr/bin/env python3
"""Compare direct, spectral, and Parseval averaging on identical BEM currents.

Creates its own converged-current checkpoints; no private runs are required.
Stage timings exclude operator setup and are not whole-solver speedups.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def flatten(data):
    return [v for row in data["mueller"] for column in row for v in column]


def difference(a, b):
    delta = [x - y for x, y in zip(a, b)]
    return {
        "relative_frobenius": math.sqrt(sum(x*x for x in delta) / sum(x*x for x in b)),
        "max_over_global_peak": max(map(abs, delta)) / max(map(abs, b)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/analytic_alpha_benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--alpha", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--field-samples", default="auto",
                        help="Existing spectral sampler: auto or an explicit sample count")
    parser.add_argument("--grid-check", action="store_true",
                        help="Also compare full 256 x 3 x 4 orientation averages")
    parser.add_argument("--cases", nargs="+", choices=["prism", "prism25", "sphere", "asymmetric"],
                        default=["prism", "prism25", "sphere", "asymmetric"])
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 128 for n in args.alpha):
        parser.error("positive repeats and alpha >= 128 required")
    if args.field_samples != "auto" and (
            not args.field_samples.isdigit() or
            not 4 <= int(args.field_samples) <= min(args.alpha) - 4):
        parser.error("field samples must be auto or in [4, min(alpha)-4]")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    obj = out / "asymmetric_tetrahedron.obj"
    obj.write_text("v 1.2 0 0\nv -0.8 0.3 0.1\nv 0.1 1.0 0.2\nv 0.2 0.1 1.5\n"
                   "f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n")
    cases = {
        "prism": (["--shape", "prism", "--sides", "6", "--aspect", "1"], 5, 2),
        "prism25": (["--shape", "prism", "--sides", "6", "--aspect", "1"], 25, 4),
        "sphere": (["--shape", "sphere"], 10, 3),
        "asymmetric": (["--obj", str(obj)], 8, 3),
    }
    env_base = os.environ.copy()
    for name in ("BEM_FARFIELD_SPECTRAL_ALPHA", "BEM_FARFIELD_ANALYTIC_ALPHA",
                 "BEM_FARFIELD_REPLAY_CHECKPOINT"):
        env_base.pop(name, None)
    env_base["OMP_NUM_THREADS"] = str(args.threads)
    rows = []
    records = []
    grid_checks = []
    for case in args.cases:
        shape_args, ka, ref = cases[case]
        directory = out / case
        directory.mkdir(exist_ok=True)
        command = [str(ROOT / "bin/muller_nodal_fmm_demo_fp32"), *shape_args,
                   "--ref", str(ref), "--ka", str(ka), "--ri", "1.5",
                   "--edge-mode", "hdiv", "--quad", "7", "--duffy-order", "4",
                   "--digits", "5", "--max-leaf", "64", "--fmm-near-radius", "3",
                   "--tol", "1e-5", "--gmres-restart", "100", "--max-iters", "500",
                   "--mbj-only", "--mbj-nodes", "50", "--orient-symmetry-order", "1",
                   "--orient-zero-start", "--ntheta", "181", "--no-dense-validation",
                   "--near-correction-cache", str(directory / "operator.near"),
                   "--mbj-cache", str(directory / "mbj.cache")]

        def run(label, alpha, mode=None, grid=None):
            env = env_base.copy()
            replay = mode is not None and grid is None
            if replay:
                env["BEM_FARFIELD_REPLAY_CHECKPOINT"] = str(directory / "seed.json.orient.checkpoint")
            if mode in ("spectral", "analytic"):
                env["BEM_FARFIELD_SPECTRAL_ALPHA"] = args.field_samples
            if mode == "analytic":
                env["BEM_FARFIELD_ANALYTIC_ALPHA"] = "1"
            path = directory / (label + ".json")
            beta, gamma = grid or (1, 1)
            cmd = command + ["--orient-average", str(alpha), str(beta), str(gamma), "--out", str(path)]
            if mode:
                cmd += ["--no-checkpoint"]
            start = time.monotonic()
            with (directory / (label + ".log")).open("w") as log:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f"solver failed, see {directory / (label + '.log')}")
            data = json.loads(path.read_text())
            if mode == "analytic" and data["orientation"]["alpha_average_method"] != "fourier_parseval":
                raise RuntimeError("analytic alpha mode was not activated")
            if replay and "[farfield replay]" not in (directory / (label + ".log")).read_text():
                raise RuntimeError("checkpoint replay did not run")
            if replay and data["iterations"]["total"] != 0:
                raise RuntimeError("replay changed the seed currents; tighten the seed solve")
            records.append({"case": case, "label": label, "mode": mode,
                            "wall_s": time.monotonic()-start, "timing": data["timing"],
                            "orientation": data["orientation"], "path": str(path)})
            return data

        # Reusing an already completed seed is safe; all replays below run anew.
        print(f"{case}: converging seed currents", flush=True)
        seed = run("seed", 256)
        for alpha in args.alpha:
            times = {mode: [] for mode in ("direct", "spectral", "analytic")}
            data = {}
            # Round-robin order reduces systematic temperature/drift bias.
            for repeat in range(args.repeats):
                for mode in times:
                    label = f"alpha{alpha}_{mode}_{repeat}"
                    data[mode] = run(label, alpha, mode)
                    times[mode].append(data[mode]["timing"]["farfield_s"])
            median = {k: statistics.median(v) for k, v in times.items()}
            exact = difference(flatten(data["analytic"]), flatten(data["spectral"]))
            truncation = difference(flatten(data["analytic"]), flatten(data["direct"]))
            if exact["relative_frobenius"] > 1e-11:
                raise RuntimeError(f"Parseval disagrees with spectral reconstruction: {exact}")
            row = {
                "case": case, "ka": ka, "ref": ref, "alpha": alpha,
                "system_dofs": seed["system_dofs"],
                "field_samples": data["analytic"]["orientation"]["farfield_alpha_samples"],
                "direct_farfield_s": median["direct"],
                "spectral_farfield_s": median["spectral"],
                "analytic_farfield_s": median["analytic"],
                "speedup_vs_spectral": median["spectral"] / median["analytic"],
                "speedup_vs_direct": median["direct"] / median["analytic"],
                "relative_error_vs_spectral": exact["relative_frobenius"],
                "relative_error_vs_direct": truncation["relative_frobenius"],
                "max_error_over_peak_vs_direct": truncation["max_over_global_peak"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.grid_check:
            spectral = run("grid_spectral", 256, "spectral", (3, 4))
            analytic = run("grid_analytic", 256, "analytic", (3, 4))
            error = difference(flatten(analytic), flatten(spectral))
            if error["relative_frobenius"] > 1e-7:
                raise RuntimeError(f"full orientation grid disagrees: {error}")
            grid_checks.append({"case": case, "grid": [256, 3, 4],
                                "error": error, "spectral_timing": spectral["timing"],
                                "analytic_timing": analytic["timing"],
                                "spectral_iterations": spectral["iterations"],
                                "analytic_iterations": analytic["iterations"]})
            print("grid check: " + json.dumps(grid_checks[-1]), flush=True)
    report = {"base_commit": "aa66bfe1bc84b0288dfd731f68b2ca1fab05424c",
              "threads": args.threads, "repeats": args.repeats,
              "field_samples_setting": args.field_samples,
              "scope": "one base orientation, two fixed converged currents; far-field stage only",
              "surface_accuracy": "not established by this same-discretization comparison",
              "rows": rows, "grid_checks": grid_checks, "runs": records}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
