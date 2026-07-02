#!/usr/bin/env python3
"""Run explicit orientation averaging over multiple GPUs and reduce JSON output."""
import argparse
import json
import os
import shutil
import subprocess
import time


def split_ranges(n, k):
    k = max(1, min(k, n))
    base = n // k
    rem = n % k
    out = []
    start = 0
    for i in range(k):
        cnt = base + (1 if i < rem else 0)
        if cnt > 0:
            out.append((start, cnt))
        start += cnt
    return out


def compute_apps(gpu, nvidia_smi="nvidia-smi"):
    if shutil.which(nvidia_smi) is None:
        return ""
    try:
        out = subprocess.check_output([
            nvidia_smi,
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ], universal_newlines=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""
    return "\n".join(line.strip() for line in out.splitlines() if line.strip())


def assert_gpus_not_busy(gpus, allow_compute_share=False, nvidia_smi="nvidia-smi"):
    if allow_compute_share:
        return
    busy = []
    for gpu in gpus:
        apps = compute_apps(gpu, nvidia_smi=nvidia_smi)
        if apps:
            busy.append("gpu=%s compute_apps=%s" % (gpu, apps.replace("\n", "; ")))
    if busy:
        raise SystemExit("GPU_BUSY " + " ".join(busy))


def auto_gpus(limit):
    nvidia_smi = os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi")
    excluded = {
        item.strip()
        for item in os.environ.get("BEM_EXCLUDE_GPUS", "3").split(",")
        if item.strip()
    }
    try:
        out = subprocess.check_output([
            nvidia_smi,
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ], universal_newlines=True)
    except Exception:
        return [str(i) for i in range(limit) if str(i) not in excluded]

    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx, util, mem = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if idx in excluded:
            continue
        if os.environ.get("BEM_ALLOW_COMPUTE_SHARE", "0") != "1" and compute_apps(idx, nvidia_smi=nvidia_smi):
            continue
        rows.append((mem, util, int(idx), idx))
    rows.sort()
    return [idx for _, _, _, idx in rows[:limit]]


def parse_gpus(spec):
    if spec.startswith("auto"):
        limit = 999
        if ":" in spec:
            limit = int(spec.split(":", 1)[1])
        return auto_gpus(limit)
    return [g.strip() for g in spec.split(",") if g.strip()]


def add_mueller(dst, src):
    for i in range(len(dst)):
        for j in range(len(dst[i])):
            for k in range(len(dst[i][j])):
                dst[i][j][k] += src[i][j][k]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", default="./bin/bem_cuda_fmm")
    p.add_argument("--out", default="orient_mgpu.json")
    p.add_argument("--work-dir", default="orient_mgpu_parts")
    p.add_argument("--gpus", default="0,1,2",
                   help="Comma list, or auto:N for N least busy GPUs")
    p.add_argument("--min-orient-per-gpu", type=int, default=16,
                   help="Shrink GPU count so each process has at least this many orientations; 0 disables")
    p.add_argument("--omp-threads", type=int, default=8,
                   help="CPU OpenMP threads per GPU worker; 0 leaves the environment unchanged")
    p.add_argument("--ka", required=True)
    p.add_argument("--ri", nargs=2, default=["1.3116", "0"])
    p.add_argument("--shape", choices=["sphere", "hex_prism", "obj"], default="sphere")
    p.add_argument("--obj", default=None,
                   help="OBJ mesh for --shape obj; bem_cuda normalizes it to unit equal-volume radius")
    p.add_argument("--subdiv", default="0",
                   help="Flat midpoint subdivisions for --shape obj")
    p.add_argument("--prism-aspect", default="1")
    p.add_argument("--edge-refine", default="auto")
    p.add_argument("--ref", default="3")
    p.add_argument("--quad", default=None)
    p.add_argument("--ntheta", default="181")
    p.add_argument("--scat-plane", choices=["yz", "xz"], default="yz")
    p.add_argument("--orient", nargs=3, default=["8", "8", "1"])
    p.add_argument("--alpha-avg", default="1",
                   help="Average alpha/phi in far-field only; use with --orient 1 NB NG")
    p.add_argument("--oldauto", choices=["2"], default=None,
                   help="MBS-fast orientation preset. oldauto=2 maps to Nphi=600, Ntheta=181")
    p.add_argument("--solver", choices=["auto", "dense", "fmm", "spfft", "pfft"], default="auto")
    p.add_argument("--accurate", action="store_true",
                   help="Use bem_cuda --accurate conservative defaults; default for --shape obj unless --fast-obj is set")
    p.add_argument("--fast-obj", action="store_true",
                   help="Reproduce old OBJ wrapper defaults without --accurate")
    p.add_argument("--fmm-digits", default=None)
    p.add_argument("--gmres-tol", default=None)
    p.add_argument("--gmres-restart", default=None)
    p.add_argument("--max-leaf", default=None)
    p.add_argument("--no-prec", action="store_true")
    p.add_argument("--cuda-lib", default="")
    p.add_argument("--allow-compute-share", action="store_true",
                   help="allow using GPUs that already have CUDA compute processes")
    args = p.parse_args()

    if args.oldauto == "2":
        args.orient = ["600", "181", "1"]
        args.ntheta = "181"

    gpus = parse_gpus(args.gpus)
    n_orient = int(args.orient[0]) * int(args.orient[1]) * int(args.orient[2])
    if args.min_orient_per_gpu > 0 and gpus:
        useful = max(1, n_orient // args.min_orient_per_gpu)
        gpus = gpus[:max(1, min(len(gpus), useful))]
    if not gpus:
        raise SystemExit("no GPUs selected")
    assert_gpus_not_busy(
        gpus,
        allow_compute_share=args.allow_compute_share or os.environ.get("BEM_ALLOW_COMPUTE_SHARE", "0") == "1",
        nvidia_smi=os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi"),
    )
    ranges = split_ranges(n_orient, len(gpus))
    os.makedirs(args.work_dir, exist_ok=True)

    procs = []
    t0 = time.time()
    for idx, (start, count) in enumerate(ranges):
        part_out = os.path.join(args.work_dir, "part_%02d.json" % idx)
        cmd = [
            args.exe,
            "--ka", args.ka,
            "--ri", args.ri[0], args.ri[1],
            "--shape", args.shape,
            "--ref", args.ref,
            "--ntheta", args.ntheta,
            "--scat-plane", args.scat_plane,
            "--orient", args.orient[0], args.orient[1], args.orient[2],
            "--orient-start", str(start),
            "--orient-count", str(count),
            "--out", part_out,
            "--force-orient",
            "--solver", args.solver,
        ]
        if args.shape == "obj":
            if not args.obj:
                raise SystemExit("--shape obj requires --obj")
            cmd += ["--obj", args.obj, "--subdiv", args.subdiv]
        if int(args.alpha_avg) > 1:
            cmd += ["--alpha-avg", args.alpha_avg]
        if args.quad is not None:
            cmd += ["--quad", args.quad]
        if args.fmm_digits is not None:
            cmd += ["--fmm-digits", args.fmm_digits]
        if args.gmres_tol is not None:
            cmd += ["--gmres-tol", args.gmres_tol]
        if args.gmres_restart is not None:
            cmd += ["--gmres-restart", args.gmres_restart]
        if args.max_leaf is not None:
            cmd += ["--max-leaf", args.max_leaf]
        if args.shape == "hex_prism":
            cmd += ["--prism-aspect", args.prism_aspect]
            if args.edge_refine != "auto":
                cmd += ["--edge-refine", args.edge_refine]
        if args.no_prec:
            cmd.append("--no-prec")
        accurate_obj = args.shape == "obj" and not args.fast_obj
        if args.accurate or accurate_obj:
            cmd.append("--accurate")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[idx % len(gpus)]
        env.setdefault("BEM_NO_AUTO_MGPU", "1")
        if args.omp_threads > 0:
            env.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
            env.setdefault("OMP_PROC_BIND", "close")
            env.setdefault("OMP_PLACES", "cores")
        if args.cuda_lib:
            env["LD_LIBRARY_PATH"] = args.cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        log = open(os.path.join(args.work_dir, "part_%02d.log" % idx), "w")
        print("GPU %s: orientations [%d, %d)" % (env["CUDA_VISIBLE_DEVICES"], start, start + count), flush=True)
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, part_out))

    for proc, log, _ in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            raise SystemExit("chunk failed with exit code %d" % rc)

    wall = time.time() - t0
    parts = [json.load(open(path)) for _, _, path in procs]
    result = parts[0]
    for part in parts[1:]:
        add_mueller(result["mueller"], part["mueller"])

    result["timing"] = {
        "assembly_s": max(p["timing"]["assembly_s"] for p in parts),
        "solve_s": max(p["timing"]["solve_s"] for p in parts),
        "farfield_s": max(p["timing"]["farfield_s"] for p in parts),
        "total_s": wall,
    }
    result["mgpu"] = {
        "gpus": gpus[:len(ranges)],
        "chunks": [{"start": s, "count": c} for s, c in ranges],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote %s, wall %.2fs" % (args.out, wall))


if __name__ == "__main__":
    main()
