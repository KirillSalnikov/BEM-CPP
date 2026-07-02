#!/usr/bin/env python3
"""Run a chunked BEM-vs-raw-ADDA comparison for an OBJ particle."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bemcuda.gpu_guard import assert_gpus_free


PHYSICAL = {
    "S11": (0, 0),
    "S12": (0, 1),
    "S22": (1, 1),
    "S33": (2, 2),
    "S34": (2, 3),
    "S44": (3, 3),
}


def run(cmd, *, env=None, stdout=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    return subprocess.check_call([str(x) for x in cmd], env=env,
                                 stdout=stdout,
                                 stderr=subprocess.STDOUT if stdout else None)


def split_counts(total, parts):
    base = total // parts
    rem = total % parts
    out = []
    start = 0
    for i in range(parts):
        count = base + (1 if i < rem else 0)
        out.append((start, count))
        start += count
    return out


def infer_beta_order(adda):
    text = str(adda)
    matches = re.findall(r"(?:^|[_/])(?:discrete_)?a\d+b(\d+)g\d+(?:$|[_/])", text)
    if matches:
        return matches[-1]
    matches = re.findall(r"(?:^|[_/])b(\d+)g\d+(?:$|[_/])", text)
    if matches:
        return matches[-1]
    return "0"


def load_mueller(path):
    with Path(path).open() as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    if mueller.shape == (4, 4, len(theta)):
        pass
    elif mueller.shape == (len(theta), 4, 4):
        mueller = np.moveaxis(mueller, 0, -1)
    elif mueller.shape == (16, len(theta)):
        mueller = mueller.reshape(4, 4, len(theta))
    elif mueller.shape == (len(theta), 16):
        mueller = np.moveaxis(mueller.reshape(len(theta), 4, 4), 0, -1)
    else:
        raise ValueError(f"unsupported Mueller shape in {path}: {mueller.shape}")
    return theta, mueller


def curve_error(prev_path, curr_path, elements):
    theta_a, mu_a = load_mueller(prev_path)
    theta_b, mu_b = load_mueller(curr_path)
    if len(theta_a) != len(theta_b) or np.max(np.abs(theta_a - theta_b)) > 1e-9:
        raise ValueError("theta grids differ between adaptive levels")
    s11_a0 = mu_a[0, 0, 0]
    s11_b0 = mu_b[0, 0, 0]
    if s11_a0 == 0.0 or s11_b0 == 0.0:
        raise ValueError("S11(0) is zero; cannot normalize adaptive convergence")
    out = {}
    total = 0.0
    for name in elements:
        i, j = PHYSICAL[name]
        a = mu_a[i, j, :] / s11_a0
        b = mu_b[i, j, :] / s11_b0
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)
        err = float(np.sqrt(np.mean(((b - a) / scale) ** 2)))
        out[name] = err
        total += err
    out["score"] = total / max(1, len(elements))
    out["max"] = max(out[name] for name in elements)
    out["scale_change"] = float(abs(s11_b0 / s11_a0 - 1.0))
    return out


def make_bem_cmd(args, orient_file, start, count, out_json):
    cmd = [
        args.bin,
        "--solver", args.solver,
        "--obj", args.obj,
        "--subdiv", args.subdiv,
        "--ka", f"{args.ka:g}",
        "--ri", args.ri[0], args.ri[1],
        "--orient-file", orient_file,
        "--orient-start", str(start),
        "--orient-count", str(count),
        "--ntheta", args.ntheta,
        "--out", out_json,
    ]
    accurate_obj = args.accurate or not args.fast_obj
    if accurate_obj:
        cmd.append("--accurate")

    system = args.system if args.system is not None else (None if accurate_obj else "pmchwt")
    quad = args.quad if args.quad is not None else (None if accurate_obj else "4")
    gmres_tol = args.gmres_tol if args.gmres_tol is not None else (None if accurate_obj else "2e-2")
    gmres_restart = args.gmres_restart if args.gmres_restart is not None else (None if accurate_obj else "100")
    fmm_digits = args.fmm_digits if args.fmm_digits is not None else (None if accurate_obj else "3")
    max_leaf = args.max_leaf if args.max_leaf is not None else (None if accurate_obj else "96")

    if system is not None:
        cmd.extend(["--system", system])
    if quad is not None:
        cmd.extend(["--quad", quad])
    if gmres_tol is not None:
        cmd.extend(["--gmres-tol", gmres_tol])
    if gmres_restart is not None:
        cmd.extend(["--gmres-restart", gmres_restart])
    if fmm_digits is not None:
        cmd.extend(["--fmm-digits", fmm_digits])
    if max_leaf is not None:
        cmd.extend(["--max-leaf", max_leaf])
    if args.extra:
        extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
        cmd.extend(extra)
    return cmd


def launch_chunks(args, out, orient_file, chunk_specs, gpus):
    processes = []
    for gpu, (start, count) in zip(gpus, chunk_specs):
        if count <= 0:
            continue
        chunk_dir = out / f"chunk_{start}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_json = chunk_dir / "bem.json"
        cmd = make_bem_cmd(args, orient_file, start, count, chunk_json)
        log = chunk_dir / "run.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("BEM_NO_AUTO_MGPU", "1")
        if not args.chunk_project:
            env["BEM_NO_ORIENT_PROJECT"] = "1"
        if args.orient_progress:
            env["BEM_ORIENT_PROGRESS"] = args.orient_progress
        if args.gmres_verbose:
            env["BEM_GMRES_VERBOSE"] = "1"
        if args.gmres_max_cycles:
            env["BEM_GMRES_MAX_CYCLES"] = args.gmres_max_cycles
        if args.dry_run:
            print(f"CUDA_VISIBLE_DEVICES={gpu}", "+", " ".join(str(x) for x in cmd), ">", log)
            continue
        handle = log.open("w")
        print(f"+ CUDA_VISIBLE_DEVICES={gpu} {' '.join(str(x) for x in cmd)} > {log}", flush=True)
        processes.append((subprocess.Popen([str(x) for x in cmd], env=env, stdout=handle,
                                           stderr=subprocess.STDOUT), handle))
    failed = False
    for proc, handle in processes:
        rc = proc.wait()
        handle.close()
        if rc != 0:
            failed = True
    if failed:
        raise SystemExit("one or more BEM chunks failed")


def combine_chunks(out, starts, dry_run, output_name="combined.json", project_random=False):
    combine_cmd = ["python3", "scripts/combine_bem_mueller.py"]
    for start in starts:
        combine_cmd.extend(["--input", "orient", out / f"chunk_{start}" / "bem.json"])
    if project_random:
        combine_cmd.append("--project-random")
    combine_cmd.extend(["--out", out / output_name])
    if dry_run:
        print("+", " ".join(str(x) for x in combine_cmd))
    else:
        run(combine_cmd)
    return out / output_name


def final_compare_and_plot(args, out, adda, combined, title_suffix=""):
    compare_cmd = [
        "python3", "scripts/compare_mueller.py",
        "--bem", combined,
        "--adda", adda,
        "--beta-order", args.beta_order,
    ]
    if args.dry_run:
        print("+", " ".join(str(x) for x in compare_cmd))
    else:
        with (out / "compare.txt").open("w") as f:
            run(compare_cmd, stdout=f)

    if args.plot:
        plot_cmd = [
            "python3", "scripts/plot_bem_raw_adda.py",
            "--bem", combined,
            "--adda", adda,
            "--beta-order", args.beta_order,
            "--log-big",
            "--title", f"OBJ ka{args.ka:g}{title_suffix}",
            "--out", out / "bem_vs_raw_adda_logbig.png",
        ]
        if args.dry_run:
            print("+", " ".join(str(x) for x in plot_cmd))
        else:
            try:
                run(plot_cmd)
            except subprocess.CalledProcessError as exc:
                print(f"WARNING: plot command failed with exit code {exc.returncode}; "
                      "comparison data was still written", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ka", required=True, type=float)
    parser.add_argument("--adda", required=True, help="Raw ADDA directory with */log and */mueller")
    parser.add_argument("--obj", required=True, help="OBJ path understood by bem_cuda_fmm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bin", default="bin/bem_cuda_fmm")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--allow-compute-share", action="store_true",
                        help="Allow starting while nvidia-smi reports existing CUDA compute processes")
    parser.add_argument("--ri", nargs=2, default=["1.6", "0.002"])
    parser.add_argument("--orient-count", type=int, default=0,
                        help="Default: use all orientations found in --adda")
    parser.add_argument("--ntheta", default="181")
    parser.add_argument("--quad", default=None)
    parser.add_argument("--subdiv", default="0")
    parser.add_argument("--solver", default="fmm")
    parser.add_argument("--accurate", action="store_true",
                        help="Use conservative OBJ defaults; now the default unless --fast-obj is set")
    parser.add_argument("--fast-obj", action="store_true",
                        help="Reproduce the old fast OBJ defaults: pmchwt, quad4, digits3, tol2e-2")
    parser.add_argument("--system", default=None)
    parser.add_argument("--gmres-tol", default=None)
    parser.add_argument("--gmres-restart", default=None)
    parser.add_argument("--fmm-digits", default=None)
    parser.add_argument("--max-leaf", default=None)
    parser.add_argument("--beta-order", default="auto",
                        help="Gauss-Legendre beta order for ADDA/BEM weights; auto infers from a*b*g directory name; 0 means equal weights")
    parser.add_argument("--euler-transform", default="identity",
                        choices=["identity", "inverse", "swap-ag", "neg-ag"],
                        help="Transform ADDA Euler angles before passing them to BEM")
    parser.add_argument("--orient-order", default="path", choices=["path", "nested"],
                        help="Order extracted ADDA orientations; nested is better for adaptive prefixes")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Extra arguments passed to bem_cuda_fmm after --")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-final-project", action="store_true",
                        help="Do not apply random-orientation Mueller projection after combining chunks")
    parser.add_argument("--chunk-project", action="store_true",
                        help="Apply projection inside each BEM chunk; kept only for reproducing old runs")
    parser.add_argument("--adaptive", action="store_true",
                        help="ADDA-like cumulative orientation averaging: run in chunks and stop when BEM average stabilizes")
    parser.add_argument("--adaptive-chunk", type=int, default=0,
                        help="Orientations added per adaptive level; default is one full GPU batch")
    parser.add_argument("--min-levels", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--max-tol", type=float, default=0.06)
    parser.add_argument("--scale-tol", type=float, default=0.02)
    parser.add_argument("--elements", default="S11,S12,S22,S33,S34,S44")
    parser.add_argument("--orient-progress", default="16",
                        help="Set BEM_ORIENT_PROGRESS for solver progress logging; empty disables")
    parser.add_argument("--gmres-verbose", action="store_true",
                        help="Set BEM_GMRES_VERBOSE=1 for per-iteration residual logging")
    parser.add_argument("--gmres-max-cycles", default="",
                        help="Set BEM_GMRES_MAX_CYCLES to cap restart cycles")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adda = Path(args.adda)
    if args.beta_order == "auto":
        args.beta_order = infer_beta_order(adda)
        print(f"Inferred beta order: {args.beta_order}", flush=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("no GPUs selected")
    assert_gpus_free(gpus, args.allow_compute_share)

    orient_file = out / "orientations.txt"
    extract_cmd = [
        "python3", "scripts/extract_adda_orientations.py", str(adda),
        "--beta-order", args.beta_order,
        "--euler-transform", args.euler_transform,
        "--order", args.orient_order,
        "--out", orient_file,
    ]
    if args.dry_run:
        print("+", " ".join(str(x) for x in extract_cmd))
    else:
        run(extract_cmd)

    if args.orient_count > 0:
        total_orient = args.orient_count
    else:
        if args.dry_run:
            total_orient = len(list(adda.glob("*/log")))
            if total_orient == 0 and (adda / "log").exists():
                total_orient = 1
        else:
            with orient_file.open() as f:
                total_orient = sum(1 for line in f if line.strip() and not line.lstrip().startswith("#"))
        if total_orient <= 0:
            raise SystemExit("no orientations found")

    elements = [item.strip() for item in args.elements.split(",") if item.strip()]
    unknown = [name for name in elements if name not in PHYSICAL]
    if unknown:
        raise SystemExit("unknown Mueller elements: " + ", ".join(unknown))

    if args.adaptive:
        adaptive_chunk = args.adaptive_chunk if args.adaptive_chunk > 0 else len(gpus)
        project_final = (not args.no_final_project and total_orient > 1)
        prev_combined = None
        accepted = None
        levels = []
        completed_starts = []
        for level, level_start in enumerate(range(0, total_orient, adaptive_chunk), start=1):
            level_count = min(adaptive_chunk, total_orient - level_start)
            level_specs = [
                (level_start + start, count)
                for start, count in split_counts(level_count, min(len(gpus), level_count))
            ]
            launch_chunks(args, out, orient_file, level_specs, gpus[:len(level_specs)])
            completed_starts.extend(start for start, _ in level_specs)
            combined = combine_chunks(out, completed_starts, args.dry_run,
                                      f"combined_{level_start + level_count:04d}.json",
                                      project_random=project_final)
            rec = {
                "level": level,
                "included_orient": level_start + level_count,
                "new_orient": level_count,
                "combined": str(combined),
                "chunk_starts": list(completed_starts),
            }
            if not args.dry_run and prev_combined is not None:
                err = curve_error(prev_combined, combined, elements)
                rec["change_from_previous"] = err
                print(
                    "adaptive level %d: included=%d/%d score=%.4g max=%.4g scale=%.4g" %
                    (level, level_start + level_count, total_orient, err["score"],
                     err["max"], err["scale_change"]),
                    flush=True,
                )
                if (level >= args.min_levels and err["score"] <= args.tol and
                        err["max"] <= args.max_tol and err["scale_change"] <= args.scale_tol):
                    rec["accepted"] = True
                    accepted = combined
                    levels.append(rec)
                    break
            levels.append(rec)
            prev_combined = combined
        if accepted is None:
            accepted = prev_combined
        manifest = {
            "mode": "adaptive_raw_adda_orientations",
            "total_orient_available": total_orient,
            "adaptive_chunk": adaptive_chunk,
            "min_levels": args.min_levels,
            "tol": args.tol,
            "max_tol": args.max_tol,
            "scale_tol": args.scale_tol,
            "elements": elements,
            "orient_order": args.orient_order,
            "accepted": str(accepted) if accepted is not None else None,
            "levels": levels,
        }
        if not args.dry_run:
            with (out / "adaptive_manifest.json").open("w") as f:
                json.dump(manifest, f, indent=2)
                f.write("\n")
            if accepted is not None and accepted.name != "combined.json":
                final = combine_chunks(out, completed_starts, False, "combined.json",
                                       project_random=project_final)
            else:
                final = accepted
            final_compare_and_plot(args, out, adda, final, " adaptive")
        else:
            print("adaptive manifest:", manifest)
        return

    chunk_specs = split_counts(total_orient, len(gpus))
    launch_chunks(args, out, orient_file, chunk_specs, gpus)
    combined = combine_chunks(out, [start for start, _ in chunk_specs], args.dry_run,
                              project_random=(not args.no_final_project and total_orient > 1))
    if not args.dry_run:
        final_compare_and_plot(args, out, adda, combined)
    else:
        final_compare_and_plot(args, out, adda, combined)


if __name__ == "__main__":
    main()
