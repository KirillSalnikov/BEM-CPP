#!/usr/bin/env python3
"""Run a chunked BEM-vs-raw-ADDA hex-prism comparison.

This is the reproducible driver for the validated hex-prism workflow:
1. extract exact ADDA orientations and beta weights from raw ADDA logs;
2. run BEM chunks across the requested GPUs;
3. combine chunk Mueller matrices;
4. compare against raw ADDA, and optionally plot.

Run this script on the machine that has `bin/bem_cuda_fmm` and CUDA GPUs.
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bemcuda.gpu_guard import assert_gpus_free


def run(cmd, *, cwd=None, env=None, stdout=None):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    return subprocess.check_call([str(x) for x in cmd], cwd=cwd, env=env, stdout=stdout,
                                 stderr=subprocess.STDOUT if stdout else None)


def run_capture(cmd, out_path):
    print("+", " ".join(str(x) for x in cmd), ">", out_path, flush=True)
    proc = subprocess.run([str(x) for x in cmd], universal_newlines=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = proc.stdout
    if proc.stderr:
        text += proc.stderr
    Path(out_path).write_text(text)
    sys.stdout.write(text)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=text)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ka", required=True, type=float)
    parser.add_argument("--adda", required=True, help="Raw ADDA directory with */log and */mueller")
    parser.add_argument("--out", required=True)
    parser.add_argument("--bin", default="bin/bem_cuda_fmm")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--allow-compute-share", action="store_true",
                        help="Allow starting while nvidia-smi reports existing CUDA compute processes")
    parser.add_argument("--ri", nargs=2, default=["1.3116", "0"])
    parser.add_argument("--prism-aspect", default="1.5")
    parser.add_argument("--ref", default="auto", help="auto or explicit integer")
    parser.add_argument("--orient-count", type=int, default=64)
    parser.add_argument("--orient-order", default="path", choices=["path", "nested"],
                        help="Order ADDA orientations before chunking. path preserves ADDA's own averaging order")
    parser.add_argument("--ntheta", default="181")
    parser.add_argument("--quad", default=None,
                        help="Override BEM triangle quadrature. By default the binary uses guarded auto settings.")
    parser.add_argument("--beta-order", default="8")
    parser.add_argument("--mode", choices=["fast", "adda-compare", "unsafe-fast"], default="adda-compare",
                        help="fast/adda-compare use guarded BEM auto-accuracy; unsafe-fast reproduces old aggressive timing settings")
    parser.add_argument("--no-final-project", action="store_true",
                        help="Do not apply random-orientation Mueller projection after combining chunks")
    parser.add_argument("--chunk-project", action="store_true",
                        help="Apply random-orientation projection inside each chunk; only for reproducing old runs")
    parser.add_argument("--extra", nargs=argparse.REMAINDER,
                        help="Extra arguments passed to bem_cuda_fmm after --")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    adda = Path(args.adda)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("no GPUs selected")
    assert_gpus_free(gpus, args.allow_compute_share)

    orient_file = out / "orientations_beta_w.txt"
    extract_cmd = [
        "python3", "scripts/extract_adda_orientations.py", str(adda),
        "--beta-order", args.beta_order,
        "--order", args.orient_order,
        "--out", orient_file,
    ]
    if args.dry_run:
        print("+", " ".join(str(x) for x in extract_cmd))
    else:
        run(extract_cmd)

    chunk_specs = split_counts(args.orient_count, len(gpus))
    processes = []
    for gpu, (start, count) in zip(gpus, chunk_specs):
        chunk_dir = out / f"chunk_{start}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.bin,
            "--ka", f"{args.ka:g}",
            "--ri", args.ri[0], args.ri[1],
            "--shape", "hex_prism",
            "--prism-aspect", args.prism_aspect,
            "--orient-file", orient_file,
            "--orient-start", str(start),
            "--orient-count", str(count),
            "--ntheta", args.ntheta,
            "--out", chunk_dir / "bem.json",
        ]
        if args.quad is not None:
            cmd.extend(["--quad", args.quad])
        if args.mode in ("fast", "adda-compare"):
            cmd.append("--adda-compare")
        if args.ref != "auto":
            cmd.extend(["--ref", args.ref])
        if args.extra:
            extra = args.extra[1:] if args.extra and args.extra[0] == "--" else args.extra
            cmd.extend(extra)

        log = chunk_dir / "run.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("BEM_NO_AUTO_MGPU", "1")
        if args.mode == "unsafe-fast":
            env["BEM_HEX_UNSAFE_FAST"] = "1"
        if not args.chunk_project:
            env["BEM_NO_ORIENT_PROJECT"] = "1"
        if args.dry_run:
            print(f"CUDA_VISIBLE_DEVICES={gpu}", "+", " ".join(str(x) for x in cmd), ">", log)
            continue
        f = log.open("w")
        print(f"+ CUDA_VISIBLE_DEVICES={gpu} {' '.join(str(x) for x in cmd)} > {log}", flush=True)
        processes.append((subprocess.Popen([str(x) for x in cmd], env=env, stdout=f,
                                           stderr=subprocess.STDOUT), f))

    failed = False
    for proc, handle in processes:
        rc = proc.wait()
        handle.close()
        if rc != 0:
            failed = True
    if failed:
        raise SystemExit("one or more BEM chunks failed")
    combine_cmd = ["python3", "scripts/combine_bem_mueller.py"]
    for start, _ in chunk_specs:
        combine_cmd.extend(["--input", "orient", out / f"chunk_{start}" / "bem.json"])
    if not args.no_final_project and args.orient_count > 1:
        combine_cmd.append("--project-random")
    combine_cmd.extend(["--out", out / "combined.json"])
    if args.dry_run:
        print("+", " ".join(str(x) for x in combine_cmd))
    else:
        run(combine_cmd)

    compare_cmd = [
        "python3", "scripts/compare_mueller.py",
        "--bem", out / "combined.json",
        "--adda", adda,
        "--beta-order", args.beta_order,
    ]
    if args.dry_run:
        print("+", " ".join(str(x) for x in compare_cmd))
    else:
        run_capture(compare_cmd, out / "compare.txt")

    if args.plot:
        plot_base = [
            "python3", "scripts/plot_bem_raw_adda.py",
            "--bem", out / "combined.json",
            "--adda", adda,
            "--beta-order", args.beta_order,
            "--log-big",
            "--title", f"hex prism ka{args.ka:g} {args.mode}",
        ]
        plot_cmds = [
            plot_base + ["--out", out / "bem_vs_raw_adda_norm_logbig.png"],
            plot_base + ["--raw", "--out", out / "bem_vs_raw_adda_raw_logbig.png"],
        ]
        for plot_cmd in plot_cmds:
            if args.dry_run:
                print("+", " ".join(str(x) for x in plot_cmd))
            else:
                try:
                    run(plot_cmd)
                except subprocess.CalledProcessError as exc:
                    print(f"WARNING: plot command failed with exit code {exc.returncode}; "
                          "comparison data was still written", flush=True)


if __name__ == "__main__":
    main()
