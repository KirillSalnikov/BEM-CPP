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


def log_has_memory_failure(path):
    patterns = (
        "out of memory",
        "cudaErrorMemoryAllocation",
        "memory allocation",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "bad_alloc",
    )
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read().lower()
    except Exception:
        return False
    return any(p.lower() in text for p in patterns)


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


def parse_orientation_angles(line, bg_mode):
    vals = line.split("#", 1)[0].split()
    if bg_mode:
        if len(vals) < 2:
            return None
        return float(vals[0]), float(vals[1])
    if len(vals) < 3:
        return None
    return float(vals[1]), float(vals[2])


def spatial_order_indices(indices, orient_lines, bg_mode):
    if orient_lines is None:
        return list(indices)
    coords = []
    for idx in indices:
        try:
            ang = parse_orientation_angles(orient_lines[idx], bg_mode)
        except Exception:
            ang = None
        if ang is None:
            coords.append((idx, None))
        else:
            beta, gamma = ang
            coords.append((idx, (beta % 360.0, gamma % 360.0)))
    if any(ang is None for _, ang in coords):
        return list(indices)
    coords_by_idx = {idx: ang for idx, ang in coords}
    remaining = dict(coords_by_idx)
    start = min(remaining.items(), key=lambda item: (item[1][0], item[1][1], item[0]))[0]
    ordered = [start]
    cur = start
    del remaining[cur]

    def dist2(a, b):
        db = abs(a[0] - b[0])
        dg = abs(a[1] - b[1])
        dg = min(dg, 360.0 - dg)
        return db * db + dg * dg

    while remaining:
        cur_ang = coords_by_idx[cur]
        nxt = min(remaining, key=lambda idx: (dist2(cur_ang, remaining[idx]), remaining[idx][0], remaining[idx][1], idx))
        ordered.append(nxt)
        cur = nxt
        del remaining[cur]
    return ordered


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
    p.add_argument("--orient-bg-file", default=None,
                   help="Explicit beta gamma [weight] grid in degrees; compatible with --alpha-avg")
    p.add_argument("--active-indices-file", default=None,
                   help="Optional zero-based orientation indices to compute from an orientation file; "
                        "requires --chunk-size 1 and preserves part_INDEX.json names")
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
    p.add_argument("--krylov",
                   choices=["gmres", "bicgstab", "bcgstab", "bicgstab-rr",
                            "bicgstab_rr", "bcgstab-rr", "bcgstab_rr",
                            "cgs", "cgs-rr", "cgs_rr", "auto"],
                   default=None,
                   help="FMM iterative method passed to bem_cuda_fmm")
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
    if args.orient_file and args.orient_bg_file:
        raise SystemExit("use either --orient-file or --orient-bg-file, not both")
    if args.active_indices_file and not (args.orient_file or args.orient_bg_file):
        raise SystemExit("--active-indices-file requires --orient-file or --orient-bg-file")
    warm_mode = args.orient_warm_start or os.environ.get("BEM_ORIENT_WARM_START", "")
    if warm_mode == "recycle" and os.environ.get("BEM_ORIENT_KEEP_CHUNK_SIZE", "0") != "1":
        hist = args.orient_recycle
        if hist is None:
            try:
                hist = int(os.environ.get("BEM_ORIENT_RECYCLE", "4"))
            except ValueError:
                hist = 4
        default_min_chunk = max(32, 4 * max(1, hist))
        min_chunk = max(1, int(os.environ.get("BEM_ORIENT_RECYCLE_MIN_CHUNK", str(default_min_chunk))))
        if args.chunk_size < min_chunk:
            print(
                "BEM recycle warm-start: raising chunk size from %d to %d "
                "(set BEM_ORIENT_KEEP_CHUNK_SIZE=1 to keep the requested value)" %
                (args.chunk_size, min_chunk),
                flush=True,
            )
            args.chunk_size = min_chunk

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("no GPUs selected")
    assert_gpus_not_busy(
        gpus,
        allow_compute_share=args.allow_compute_share or os.environ.get("BEM_ALLOW_COMPUTE_SHARE", "0") == "1",
        nvidia_smi=os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi"),
    )
    os.makedirs(args.work_dir, exist_ok=True)

    orient_line_file = args.orient_bg_file or args.orient_file
    orient_bg_mode = bool(args.orient_bg_file)
    orient_lines = None
    if orient_line_file:
        with open(orient_line_file, "r") as f:
            orient_lines = [line for line in f
                            if line.strip() and not line.lstrip().startswith("#")]
        n_orient = len(orient_lines)
    else:
        n_orient = int(args.orient[0]) * int(args.orient[1]) * int(args.orient[2])
    def part_is_reusable(path, start, count, total):
        if args.no_resume or not os.path.exists(path):
            return False
        try:
            with open(path, "r") as f:
                part = json.load(f)
            return (
                int(part.get("orient_start", -1)) == start
                and int(part.get("orient_count", -1)) == count
                and int(part.get("orient_total", total)) == total
                and "theta" in part
                and "mueller" in part
            )
        except Exception:
            return False

    chunks = []
    skipped = []
    all_active_indices = None
    if args.active_indices_file:
        with open(args.active_indices_file, "r") as f:
            active_indices = []
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                active_indices.append(int(line))
        if not active_indices:
            raise SystemExit(f"no active indices in {args.active_indices_file}")
        for idx in active_indices:
            if idx < 0 or idx >= n_orient:
                raise SystemExit(f"active orientation index {idx} outside [0,{n_orient})")
        all_active_indices = list(active_indices)
        missing_active = [
            idx for idx in active_indices
            if not part_is_reusable(os.path.join(args.work_dir, "part_%04d.json" % idx), idx, 1, n_orient)
        ]
        missing_active = spatial_order_indices(missing_active, orient_lines, orient_bg_mode)
        missing_set = set(missing_active)
        for idx in active_indices:
            if idx not in missing_set:
                skipped.append({
                    "index": idx,
                    "start": idx,
                    "count": 1,
                    "active": [idx],
                    "out": os.path.join(args.work_dir, "part_%04d.json" % idx),
                })
        chunk_size = args.chunk_size
        tail_chunk_env = os.environ.get("BEM_ORIENT_TAIL_CHUNK_SIZE", "")
        if tail_chunk_env:
            try:
                tail_chunk_size = max(1, int(tail_chunk_env))
            except ValueError:
                tail_chunk_size = chunk_size
            tail_threshold = max(
                chunk_size,
                len(gpus) * chunk_size * int(os.environ.get("BEM_ORIENT_TAIL_THRESHOLD_CHUNKS", "2")),
            )
            if len(missing_active) <= tail_threshold and tail_chunk_size < chunk_size:
                print(
                    "BEM orientation tail balance: reducing chunk size from %d to %d for %d remaining active orientations" %
                    (chunk_size, tail_chunk_size, len(missing_active)),
                    flush=True,
                )
                chunk_size = tail_chunk_size
        for seq, first in enumerate(range(0, len(missing_active), chunk_size)):
            group = missing_active[first:first + chunk_size]
            chunks.append({"index": group[0], "seq": seq, "start": 0, "count": len(group), "active": group})
    else:
        for seq, start in enumerate(range(0, n_orient, args.chunk_size)):
            chunks.append({"index": seq, "seq": seq, "start": start, "count": min(args.chunk_size, n_orient - start), "active": None})

    running = []
    finished = []
    pending = []
    for chunk in chunks:
        idx, start, count, active = chunk["index"], chunk["start"], chunk["count"], chunk["active"]
        reusable = False
        if active is not None and not args.active_indices_file:
            missing = [
                ai for ai in active
                if not part_is_reusable(os.path.join(args.work_dir, "part_%04d.json" % ai), ai, 1, n_orient)
            ]
            reusable = not missing
            part_out = os.path.join(args.work_dir, "group_%04d.json" % idx)
            if missing:
                chunk = dict(chunk)
                chunk["active"] = missing
                chunk["count"] = len(missing)
                chunk["index"] = missing[0]
                active = missing
                idx = missing[0]
                part_out = os.path.join(args.work_dir, "group_%04d.json" % idx)
        else:
            part_out = os.path.join(args.work_dir, "part_%04d.json" % idx)
            reusable = part_is_reusable(part_out, start, count, n_orient)
        if reusable:
            label = active if active is not None else list(range(start, start + count))
            print("SKIP existing chunk %d orientations %s" % (idx, label[:6]), flush=True)
            item = {"gpu": "reuse", "index": idx, "out": part_out, "active": active}
            finished.append(item)
            skipped.append({"index": idx, "start": start, "count": count, "active": active, "out": part_out})
        else:
            if chunk["active"] is not None and len(chunk["active"]) != count:
                done = count - len(chunk["active"])
                print("RESUME partial chunk %d: skip %d existing, compute %d missing" %
                      (idx, done, len(chunk["active"])), flush=True)
            pending.append(chunk)
    if args.chunk_order == "spread":
        order = {seq: pos for pos, seq in enumerate(spread_order(len(chunks)))}
        pending.sort(key=lambda item: order.get(item.get("seq", item["index"]), item.get("seq", item["index"])))

    next_pending = 0
    t0 = time.time()

    def write_active_group_files(chunk_index, active):
        bg_path = os.path.join(args.work_dir, "group_%04d.bg" % chunk_index)
        idx_path = os.path.join(args.work_dir, "group_%04d.idx" % chunk_index)
        if orient_lines is None:
            raise SystemExit("active grouped chunks require an orientation line file")
        with open(bg_path, "w") as f:
            for ai in active:
                f.write(orient_lines[ai])
                if not orient_lines[ai].endswith("\n"):
                    f.write("\n")
        with open(idx_path, "w") as f:
            for ai in active:
                f.write("%d\n" % ai)
        return bg_path, idx_path

    def launch(gpu, chunk, low_memory=False, attempt=0):
        chunk_index, start, count, active = chunk["index"], chunk["start"], chunk["count"], chunk["active"]
        part_out = os.path.join(args.work_dir, "group_%04d.json" % chunk_index) if active is not None else os.path.join(args.work_dir, "part_%04d.json" % chunk_index)
        part_log = os.path.join(args.work_dir, "group_%04d.log" % chunk_index) if active is not None else os.path.join(args.work_dir, "part_%04d.log" % chunk_index)
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
        if active is not None:
            bg_path, idx_path = write_active_group_files(chunk_index, active)
            cmd += ["--orient-bg-file", bg_path,
                    "--orient-split-dir", args.work_dir,
                    "--orient-split-indices", idx_path,
                    "--orient-split-total", str(n_orient)]
        elif args.orient_file:
            cmd += ["--orient-file", args.orient_file]
        elif args.orient_bg_file:
            cmd += ["--orient-bg-file", args.orient_bg_file]
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
        if args.krylov is not None:
            cmd += ["--krylov", args.krylov]
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
        if low_memory:
            env["BEM_FMM_NO_BATCH4"] = "1"
            env["BEM_FMM_BATCH4"] = "0"
            env["BEM_FMM_ALLOC_BATCH4"] = "0"
            env.setdefault("BEM_FF_TARGET_MB", "256")
            env.setdefault("BEM_RHS_TARGET_MB", "256")
        log = open(part_log, "w")
        label = active if active is not None else list(range(start, start + count))
        suffix = " low-memory retry" if low_memory else ""
        print("GPU %s: chunk %d%s orientations %s" %
              (gpu, chunk_index, suffix, label[:8]), flush=True)
        return {
            "gpu": gpu,
            "index": chunk_index,
            "chunk": chunk,
            "attempt": attempt,
            "low_memory": low_memory,
            "out": part_out,
            "active": active,
            "log": log,
            "proc": subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env),
        }

    while next_pending < len(pending) and len(running) < len(gpus):
        running.append(launch(gpus[len(running)], pending[next_pending]))
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
                if (not item.get("low_memory")) and log_has_memory_failure(item["log"].name):
                    failed_log = item["log"].name + ".failed_mem_attempt%d" % item.get("attempt", 0)
                    try:
                        os.replace(item["log"].name, failed_log)
                    except OSError:
                        pass
                    print("GPU %s: chunk %d failed with memory error; retrying in low-memory mode" %
                          (item["gpu"], item["index"]), flush=True)
                    running.append(launch(item["gpu"], item["chunk"], low_memory=True,
                                          attempt=item.get("attempt", 0) + 1))
                    continue
                raise SystemExit("chunk %d failed with exit code %d" % (item["index"], rc))
            finished.append(item)
            if next_pending < len(pending):
                running.append(launch(item["gpu"], pending[next_pending]))
                next_pending += 1

    finished.sort(key=lambda x: x["index"])
    if not finished:
        raise SystemExit("no chunks were produced")
    part_paths = []
    if all_active_indices is not None:
        part_paths = [os.path.join(args.work_dir, "part_%04d.json" % ai) for ai in all_active_indices]
    else:
        for chunk in chunks:
            active = chunk["active"]
            if active is not None:
                part_paths.extend(os.path.join(args.work_dir, "part_%04d.json" % ai) for ai in active)
            else:
                part_paths.append(os.path.join(args.work_dir, "part_%04d.json" % chunk["index"]))
    parts = [json.load(open(path)) for path in part_paths if os.path.exists(path)]
    if not parts:
        raise SystemExit("no reusable part JSON files were produced")
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
        "orient_bg_file": args.orient_bg_file,
        "chunks": [{"index": c["index"], "seq": c.get("seq"), "start": c["start"], "count": c["count"], "active": c["active"]} for c in chunks],
        "chunk_order": args.chunk_order,
        "active_indices_file": args.active_indices_file,
        "low_memory_retry_enabled": True,
        "low_memory_retry_env": {
            "BEM_FMM_NO_BATCH4": "1",
            "BEM_FF_TARGET_MB": "256",
            "BEM_RHS_TARGET_MB": "256",
        },
        "reused_chunks": skipped,
        "orient_warm_start": args.orient_warm_start or os.environ.get("BEM_ORIENT_WARM_START", "zero"),
        "orient_recycle": args.orient_recycle if args.orient_recycle is not None else os.environ.get("BEM_ORIENT_RECYCLE"),
    }
    result["orient_start"] = 0
    result["orient_count"] = len(all_active_indices) if all_active_indices is not None else n_orient
    result["orient_total"] = n_orient
    if all_active_indices is not None:
        result["active_orient_count"] = len(all_active_indices)
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
