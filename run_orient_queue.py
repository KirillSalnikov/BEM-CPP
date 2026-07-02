#!/usr/bin/env python3
"""Run orientation averaging as a bounded-memory GPU work queue."""
import argparse
import json
import os
import shutil
import subprocess
import time


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


def parse_gpus(spec):
    return [g.strip() for g in spec.split(",") if g.strip()]


def add_mueller(dst, src):
    for i in range(len(dst)):
        for j in range(len(dst[i])):
            for k in range(len(dst[i][j])):
                dst[i][j][k] += src[i][j][k]


def spread_order(n):
    """Return chunk indices interleaved across the full queue.

    This keeps completed early chunks representative of the whole orientation
    grid, while each chunk itself remains a normal contiguous solver range.
    """
    if n <= 0:
        return []
    left = 0
    right = n - 1
    out = []
    while left <= right:
        out.append(left)
        left += 1
        if left <= right:
            out.append(right)
            right -= 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", default="./bin/bem_cuda_fmm")
    p.add_argument("--out", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--gpus", default=os.environ.get("BEM_ORIENT_GPUS", "0,1,2"))
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--omp-threads", type=int, default=8,
                   help="CPU OpenMP threads per GPU worker; 0 leaves the environment unchanged")
    p.add_argument("--orient-warm-start", choices=["zero", "previous", "recycle"], default=None,
                   help="Initial guess policy inside each orientation chunk")
    p.add_argument("--orient-recycle", type=int, default=None,
                   help="History length for --orient-warm-start recycle")
    p.add_argument("--ka", required=True)
    p.add_argument("--ri", nargs=2, default=["1.3116", "0"])
    p.add_argument("--shape", choices=["sphere", "hex_prism", "obj"], default="hex_prism")
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
    p.add_argument("--orient", nargs=3, default=["600", "181", "1"])
    p.add_argument("--orient-file", default=None,
                   help="Explicit alpha beta gamma [weight] grid in degrees; chunks are taken by line index")
    p.add_argument("--alpha-avg", default="1",
                   help="Average alpha/phi in far-field only; use with --orient 1 NB NG")
    p.add_argument("--oldauto", choices=["2"], default=None)
    p.add_argument("--solver", choices=["auto", "fmm", "spfft", "pfft"], default="fmm")
    p.add_argument("--accurate", action="store_true",
                   help="Pass --accurate to bem_cuda_fmm; default for --shape obj unless --fast-obj is set")
    p.add_argument("--fast-obj", action="store_true",
                   help="Reproduce old OBJ wrapper defaults without --accurate")
    p.add_argument("--system", default=None,
                   help="Linear system passed to bem_cuda_fmm, e.g. pmchwt or muller2-balanced")
    p.add_argument("--fmm-digits", default=None)
    p.add_argument("--gmres-tol", default=None)
    p.add_argument("--gmres-restart", default=None)
    p.add_argument("--max-leaf", default=None)
    p.add_argument("--no-prec", action="store_true")
    p.add_argument("--cuda-lib", default="")
    p.add_argument("--allow-compute-share", action="store_true",
                   help="allow using GPUs that already have CUDA compute processes")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore existing part_*.json files and recompute every chunk")
    p.add_argument("--chunk-order", choices=["sequential", "spread"], default="sequential",
                   help="launch order for chunks; spread improves early partial diagnostics")
    args = p.parse_args()

    if args.oldauto == "2":
        args.orient = ["600", "181", "1"]
        args.ntheta = "181"

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("no GPUs selected")
    assert_gpus_not_busy(
        gpus,
        allow_compute_share=args.allow_compute_share or os.environ.get("BEM_ALLOW_COMPUTE_SHARE", "0") == "1",
        nvidia_smi=os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi"),
    )
    os.makedirs(args.work_dir, exist_ok=True)

    if args.orient_file:
        with open(args.orient_file, "r") as f:
            n_orient = sum(1 for line in f
                           if line.strip() and not line.lstrip().startswith("#"))
    else:
        n_orient = int(args.orient[0]) * int(args.orient[1]) * int(args.orient[2])
    chunks = []
    for start in range(0, n_orient, args.chunk_size):
        chunks.append((start, min(args.chunk_size, n_orient - start)))

    running = []
    finished = []
    skipped = []
    pending = []
    for idx, (start, count) in enumerate(chunks):
        part_out = os.path.join(args.work_dir, "part_%04d.json" % idx)
        reusable = False
        if not args.no_resume and os.path.exists(part_out):
            try:
                with open(part_out, "r") as f:
                    part = json.load(f)
                reusable = (
                    int(part.get("orient_start", -1)) == start
                    and int(part.get("orient_count", -1)) == count
                    and int(part.get("orient_total", n_orient)) == n_orient
                    and "theta" in part
                    and "mueller" in part
                )
            except Exception:
                reusable = False
        if reusable:
            print("SKIP existing chunk %d orientations [%d, %d)" %
                  (idx, start, start + count), flush=True)
            item = {"gpu": "reuse", "index": idx, "out": part_out}
            finished.append(item)
            skipped.append({"index": idx, "start": start, "count": count, "out": part_out})
        else:
            pending.append((idx, start, count))
    if args.chunk_order == "spread":
        order = {idx: pos for pos, idx in enumerate(spread_order(len(chunks)))}
        pending.sort(key=lambda item: order.get(item[0], item[0]))

    next_pending = 0
    t0 = time.time()

    def launch(gpu, chunk_index, start, count):
        part_out = os.path.join(args.work_dir, "part_%04d.json" % chunk_index)
        part_log = os.path.join(args.work_dir, "part_%04d.log" % chunk_index)
        cmd = [
            args.exe,
            "--ka", args.ka,
            "--ri", args.ri[0], args.ri[1],
            "--shape", args.shape,
            "--ref", args.ref,
            "--ntheta", args.ntheta,
            "--scat-plane", args.scat_plane,
            "--out", part_out,
            "--force-orient",
            "--solver", args.solver,
        ]
        if args.orient_file:
            cmd += ["--orient-file", args.orient_file]
        else:
            cmd += ["--orient", args.orient[0], args.orient[1], args.orient[2]]
        cmd += ["--orient-start", str(start), "--orient-count", str(count)]
        if args.shape == "obj":
            if not args.obj:
                raise SystemExit("--shape obj requires --obj")
            cmd += ["--obj", args.obj, "--subdiv", args.subdiv]
        if int(args.alpha_avg) > 1:
            cmd += ["--alpha-avg", args.alpha_avg]
        accurate_obj = args.shape == "obj" and not args.fast_obj
        if args.accurate or accurate_obj:
            cmd.append("--accurate")
        quad = args.quad if args.quad is not None else ("4" if args.shape != "obj" else None)
        if quad:
            cmd += ["--quad", quad]
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
        if args.system is not None:
            cmd += ["--system", args.system]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("BEM_NO_AUTO_MGPU", "1")
        if args.omp_threads > 0:
            env.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
            env.setdefault("OMP_PROC_BIND", "close")
            env.setdefault("OMP_PLACES", "cores")
        if args.orient_warm_start:
            env["BEM_ORIENT_WARM_START"] = args.orient_warm_start
        if args.orient_recycle is not None:
            env["BEM_ORIENT_RECYCLE"] = str(args.orient_recycle)
        if args.cuda_lib:
            env["LD_LIBRARY_PATH"] = args.cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        log = open(part_log, "w")
        print("GPU %s: chunk %d orientations [%d, %d)" %
              (gpu, chunk_index, start, start + count), flush=True)
        return {
            "gpu": gpu,
            "index": chunk_index,
            "out": part_out,
            "log": log,
            "proc": subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env),
        }

    while next_pending < len(pending) and len(running) < len(gpus):
        idx, s, c = pending[next_pending]
        running.append(launch(gpus[len(running)], idx, s, c))
        next_pending += 1

    while running:
        time.sleep(1)
        for item in list(running):
            rc = item["proc"].poll()
            if rc is None:
                continue
            item["log"].close()
            running.remove(item)
            if rc != 0:
                raise SystemExit("chunk %d failed with exit code %d" % (item["index"], rc))
            finished.append(item)
            if next_pending < len(pending):
                idx, s, c = pending[next_pending]
                running.append(launch(item["gpu"], idx, s, c))
                next_pending += 1

    finished.sort(key=lambda x: x["index"])
    if not finished:
        raise SystemExit("no chunks were produced")
    parts = [json.load(open(item["out"])) for item in finished]
    result = parts[0]
    for part in parts[1:]:
        add_mueller(result["mueller"], part["mueller"])

    result["timing"] = {
        "assembly_s": max(p["timing"]["assembly_s"] for p in parts),
        "solve_s": sum(p["timing"]["solve_s"] for p in parts) / len(gpus),
        "farfield_s": max(p["timing"]["farfield_s"] for p in parts),
        "total_s": time.time() - t0,
    }
    result["mgpu_queue"] = {
        "gpus": gpus,
        "chunk_size": args.chunk_size,
        "orient_file": args.orient_file,
        "chunks": [{"start": s, "count": c} for s, c in chunks],
        "chunk_order": args.chunk_order,
        "reused_chunks": skipped,
        "orient_warm_start": args.orient_warm_start or os.environ.get("BEM_ORIENT_WARM_START", "zero"),
        "orient_recycle": args.orient_recycle if args.orient_recycle is not None else os.environ.get("BEM_ORIENT_RECYCLE"),
    }
    result["orient_start"] = 0
    result["orient_count"] = n_orient
    result["orient_total"] = n_orient
    result["orientation_weight_sum"] = sum(float(p.get("orientation_weight_sum", 0.0)) for p in parts)
    for key in ("gmres_matvecs", "gmres_converged_systems", "gmres_nonconverged_systems",
                "gmres_stagnation_stops", "gmres_numerical_breakdowns",
                "gmres_restored_best_iterates", "gmres_max_cycle_exhaustions"):
        if key in parts[0]:
            result[key] = sum(int(p.get(key, 0)) for p in parts)
    if "gmres_max_final_relres" in parts[0]:
        result["gmres_max_final_relres"] = max(float(p.get("gmres_max_final_relres", 0.0)) for p in parts)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote %s, wall %.2fs" % (args.out, result["timing"]["total_s"]))


if __name__ == "__main__":
    main()
