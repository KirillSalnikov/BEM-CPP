#!/usr/bin/env python3
"""Run hex-prism BEM size sweep with oldauto orientation preset."""
import argparse
import json
import math
import os
import subprocess
import time


def choose_mesh(ka):
    # Conservative enough to increase surface resolution with size, but capped
    # because oldauto=2 is already very expensive.
    if ka < 20:
        return {"ref": 4, "quad": 4, "accurate": False}
    if ka < 35:
        return {"ref": 4, "quad": 7, "accurate": False}
    return {"ref": 5, "quad": 4, "accurate": False}


def run(cmd, env=None):
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, env=env, check=True)
    return proc.stdout, time.time() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", default="./bin/bem_cuda_fmm")
    p.add_argument("--wrapper", default="./run_orient_mgpu.py")
    p.add_argument("--out-dir", default="hex_oldauto_sweep")
    p.add_argument("--gpus", default="auto:3")
    p.add_argument("--lambda-um", type=float, default=0.532)
    p.add_argument("--n-re", type=float, default=1.3116)
    p.add_argument("--n-im", type=float, default=0.0)
    p.add_argument("--sizes-um", nargs="+", type=float,
                   default=[1.0, 3.25, 5.5, 7.75, 10.0])
    p.add_argument("--prism-aspect", default="1")
    p.add_argument("--oldauto", default="2", choices=["2"])
    p.add_argument("--min-orient-per-gpu", default="16")
    p.add_argument("--accurate", action="store_true",
                   help="Force bem_cuda --accurate for all sizes")
    p.add_argument("--pilot", action="store_true",
                   help="Use a tiny orientation grid for timing sanity instead of oldauto")
    p.add_argument("--cuda-lib", default="")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    env = os.environ.copy()
    if args.cuda_lib:
        env["LD_LIBRARY_PATH"] = args.cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    for d_um in args.sizes_um:
        ka = math.pi * d_um / args.lambda_um
        mesh = choose_mesh(ka)
        if args.accurate:
            mesh["accurate"] = True
            mesh["quad"] = None

        tag = f"D{d_um:g}_ka{ka:.4g}".replace(".", "p")
        out_json = os.path.join(args.out_dir, f"{tag}.json")
        work_dir = os.path.join(args.out_dir, f"{tag}_parts")

        cmd = [
            args.wrapper,
            "--exe", args.exe,
            "--gpus", args.gpus,
            "--min-orient-per-gpu", args.min_orient_per_gpu,
            "--ka", f"{ka:.12g}",
            "--ri", str(args.n_re), str(args.n_im),
            "--shape", "hex_prism",
            "--prism-aspect", args.prism_aspect,
            "--ref", str(mesh["ref"]),
            "--out", out_json,
            "--work-dir", work_dir,
        ]
        if args.pilot:
            cmd += ["--orient", "4", "3", "1", "--ntheta", "31"]
        else:
            cmd += ["--oldauto", args.oldauto]
        if mesh["quad"] is not None:
            cmd += ["--quad", str(mesh["quad"])]
        if mesh["accurate"]:
            cmd.append("--accurate")
        if args.cuda_lib:
            cmd += ["--cuda-lib", args.cuda_lib]

        print(f"\n=== D={d_um:g} um, lambda={args.lambda_um:g} um, ka={ka:.4f}, "
              f"ref={mesh['ref']}, quad={mesh['quad'] or 'auto'} ===",
              flush=True)
        stdout, elapsed = run(cmd, env=env)
        print(stdout, end="", flush=True)

        data = json.load(open(out_json))
        row = {
            "D_um": d_um,
            "lambda_um": args.lambda_um,
            "ka": ka,
            "ref": mesh["ref"],
            "quad": mesh["quad"],
            "accurate": mesh["accurate"],
            "time_s": data.get("timing", {}).get("total_s", elapsed),
            "timing": data.get("timing", {}),
            "mgpu": data.get("mgpu", {}),
            "out": out_json,
        }
        rows.append(row)
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(rows, f, indent=2)

    print("\nSummary")
    for r in rows:
        print(f"D={r['D_um']:6.3f}um ka={r['ka']:8.3f} ref={r['ref']} "
              f"quad={r['quad'] or 'auto'} time={r['time_s']:.2f}s")


if __name__ == "__main__":
    main()
