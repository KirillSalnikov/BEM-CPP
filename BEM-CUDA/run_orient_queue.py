#!/usr/bin/env python3
"""Run orientation averaging as a bounded-memory GPU work queue."""
import argparse
import json
import os
import subprocess
import time


def parse_gpus(spec):
    return [g.strip() for g in spec.split(",") if g.strip()]


def add_mueller(dst, src):
    for i in range(len(dst)):
        for j in range(len(dst[i])):
            for k in range(len(dst[i][j])):
                dst[i][j][k] += src[i][j][k]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", default="./bin/bem_cuda_fmm")
    p.add_argument("--out", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4")
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--omp-threads", type=int, default=8,
                   help="CPU OpenMP threads per GPU worker; 0 leaves the environment unchanged")
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
    p.add_argument("--quad", default="4")
    p.add_argument("--ntheta", default="181")
    p.add_argument("--scat-plane", choices=["yz", "xz"], default="yz")
    p.add_argument("--orient", nargs=3, default=["600", "181", "1"])
    p.add_argument("--alpha-avg", default="1",
                   help="Average alpha/phi in far-field only; use with --orient 1 NB NG")
    p.add_argument("--oldauto", choices=["2"], default=None)
    p.add_argument("--solver", choices=["auto", "fmm", "spfft", "pfft"], default="fmm")
    p.add_argument("--fmm-digits", default=None)
    p.add_argument("--gmres-tol", default=None)
    p.add_argument("--gmres-restart", default=None)
    p.add_argument("--max-leaf", default=None)
    p.add_argument("--no-prec", action="store_true")
    p.add_argument("--cuda-lib", default="")
    args = p.parse_args()

    if args.oldauto == "2":
        args.orient = ["600", "181", "1"]
        args.ntheta = "181"

    gpus = parse_gpus(args.gpus)
    if not gpus:
        raise SystemExit("no GPUs selected")
    os.makedirs(args.work_dir, exist_ok=True)

    n_orient = int(args.orient[0]) * int(args.orient[1]) * int(args.orient[2])
    chunks = []
    for start in range(0, n_orient, args.chunk_size):
        chunks.append((start, min(args.chunk_size, n_orient - start)))

    running = []
    finished = []
    next_chunk = 0
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
        if args.quad:
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
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if args.omp_threads > 0:
            env.setdefault("OMP_NUM_THREADS", str(args.omp_threads))
            env.setdefault("OMP_PROC_BIND", "close")
            env.setdefault("OMP_PLACES", "cores")
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

    while next_chunk < len(chunks) and len(running) < len(gpus):
        s, c = chunks[next_chunk]
        running.append(launch(gpus[len(running)], next_chunk, s, c))
        next_chunk += 1

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
            if next_chunk < len(chunks):
                s, c = chunks[next_chunk]
                running.append(launch(item["gpu"], next_chunk, s, c))
                next_chunk += 1

    finished.sort(key=lambda x: x["index"])
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
        "chunks": [{"start": s, "count": c} for s, c in chunks],
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("Wrote %s, wall %.2fs" % (args.out, result["timing"]["total_s"]))


if __name__ == "__main__":
    main()
