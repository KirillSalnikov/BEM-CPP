#!/usr/bin/env python3
"""Run one BEM candidate on epyc1 and score it against MBS.

This is intentionally narrow: it standardizes the Greek-particle benchmark
used for ADDA dpl25 comparisons so mesh/solver experiments are reproducible.
"""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from greek_profiles import select_greek_profile


DEFAULT_REMOTE = "kirill_epyc@172.16.1.168"
DEFAULT_REMOTE_DIR = "/home/kirill_epyc/BEM-CUDA"
DEFAULT_MBS_DIR = "/home/user/cluster/BEM-CPP/greek/ADDA_for_PO_comparison/refr_1_6__0_002"


def format_ax(value):
    return f"{float(value):.12g}"


def default_mbs_path(mbs_dir, ka):
    return str(Path(mbs_dir) / f"A_x={format_ax(ka)}_refr_1_6__0_002.dat")


def run(cmd, **kwargs):
    print("+", " ".join(shlex.quote(str(c)) for c in cmd))
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--obj", help="Remote OBJ path, relative to remote repo")
    parser.add_argument("--auto-greek-profile", action="store_true",
                        help="select the current validated Greek-particle mesh profile from --ka")
    parser.add_argument("--out", required=True, help="Remote/local JSON path, relative to repo")
    parser.add_argument("--ka", default="5.645")
    parser.add_argument("--ri", nargs=2, default=["1.6", "0.002"])
    parser.add_argument("--orient", nargs=3, default=["95", "65", "20"])
    parser.add_argument("--quad", default=None,
                        help="quadrature order; default is 7 for accurate OBJ mode")
    parser.add_argument("--ntheta", default="19")
    parser.add_argument("--solver", default="fmm")
    parser.add_argument("--system", default=None,
                        help="linear system; default is bem_cuda_fmm accurate OBJ auto-balanced policy")
    parser.add_argument("--fmm-digits", default=None,
                        help="FMM digits; default is 6 for accurate OBJ mode")
    parser.add_argument("--gmres-tol", default=None,
                        help="GMRES tolerance; default is 5e-4 for accurate OBJ mode")
    parser.add_argument("--gmres-restart", default=None,
                        help="GMRES restart; default is 500 for accurate OBJ mode")
    parser.add_argument("--max-leaf", default=None,
                        help="FMM leaf size; default is 128 for accurate OBJ mode")
    parser.add_argument("--fast-obj", action="store_true",
                        help="allow the old fast OBJ policy instead of conservative accurate defaults")
    parser.add_argument("--cuda-devices",
                        help="set CUDA_VISIBLE_DEVICES on the remote run, e.g. 1 or 0,1,2")
    parser.add_argument("--wait-gpu-free", action="store_true",
                        help="wait on the remote host until the first selected GPU is below util/memory thresholds")
    parser.add_argument("--wait-gpu-util", default="20",
                        help="GPU utilization threshold for --wait-gpu-free, percent")
    parser.add_argument("--wait-gpu-mem", default="1200",
                        help="GPU memory threshold for --wait-gpu-free, MiB")
    parser.add_argument("--wait-gpu-interval", default="60",
                        help="poll interval for --wait-gpu-free, seconds")
    parser.add_argument("--allow-compute-share", action="store_true",
                        help="allow starting while nvidia-smi reports existing CUDA compute processes")
    parser.add_argument("--mbs", help="MBS/ADDA reference table; defaults to A_x=<ka> in --mbs-dir")
    parser.add_argument("--mbs-dir", default=DEFAULT_MBS_DIR)
    parser.add_argument("--theta-max", default="180")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip remote run when local --out already exists; still score it")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mbs_path = args.mbs or default_mbs_path(args.mbs_dir, args.ka)
    if not Path(mbs_path).is_file():
        raise FileNotFoundError(f"reference table not found for ka={args.ka}: {mbs_path}")
    obj_path = args.obj
    if args.auto_greek_profile:
        profile, extrapolated = select_greek_profile(args.ka)
        obj_path = profile.mesh
        print(
            f"Auto Greek profile: {obj_path} "
            f"({'extrapolated' if extrapolated else 'validated'}; {profile.note})"
        )
    if not obj_path:
        raise ValueError("--obj is required unless --auto-greek-profile is set")

    remote_out = args.out
    local_out = Path(args.out)
    local_out.parent.mkdir(parents=True, exist_ok=True)

    cuda_env = (
        "export CUDA_HOME=$HOME/cuda-12.2/usr/local/cuda-12.2; "
        "export PATH=$CUDA_HOME/bin:$PATH; "
        "export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}; "
    )
    accurate_obj = not args.fast_obj
    quad = args.quad if args.quad is not None else ("7" if accurate_obj else None)
    fmm_digits = args.fmm_digits if args.fmm_digits is not None else ("6" if accurate_obj else None)
    gmres_tol = args.gmres_tol if args.gmres_tol is not None else ("5e-4" if accurate_obj else None)
    gmres_restart = args.gmres_restart if args.gmres_restart is not None else ("500" if accurate_obj else None)
    max_leaf = args.max_leaf if args.max_leaf is not None else ("128" if accurate_obj else None)

    bem_cmd = [
        "./bin/bem_cuda_fmm",
        "--solver", args.solver,
        "--obj", obj_path,
        "--ka", args.ka,
        "--ri", args.ri[0], args.ri[1],
        "--orient", args.orient[0], args.orient[1], args.orient[2],
        "--ntheta", args.ntheta,
        "--out", remote_out,
    ]
    if accurate_obj:
        bem_cmd.append("--accurate")
    else:
        bem_cmd.append("--fast-obj")
    if args.system is not None:
        bem_cmd.extend(["--system", args.system])
    if quad is not None:
        bem_cmd.extend(["--quad", quad])
    if fmm_digits is not None:
        bem_cmd.extend(["--fmm-digits", fmm_digits])
    if gmres_tol is not None:
        bem_cmd.extend(["--gmres-tol", gmres_tol])
    if gmres_restart is not None:
        bem_cmd.extend(["--gmres-restart", gmres_restart])
    if max_leaf is not None:
        bem_cmd.extend(["--max-leaf", max_leaf])
    cuda_visible = ""
    if args.cuda_devices:
        cuda_visible = (
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_devices)}; "
            "export BEM_NO_AUTO_MGPU=${BEM_NO_AUTO_MGPU:-1}; "
            "export BEM_GMRES_VERBOSE=${BEM_GMRES_VERBOSE:-1}; "
        )
    nvidia_smi = shlex.quote(os.environ.get("BEM_NVIDIA_SMI", "nvidia-smi"))
    selected_gpus = [g.strip() for g in (args.cuda_devices or "0").split(",") if g.strip()]
    gpu_list = " ".join(shlex.quote(g) for g in selected_gpus)
    compute_check = ""
    if not args.allow_compute_share:
        compute_check = (
            f"apps=$({nvidia_smi} -i \"$gpu\" --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | head -n 3 || true); "
            "if [ -n \"$apps\" ]; then "
            "ok=0; wait_msg=\"$wait_msg gpu=$gpu compute_apps=$(printf '%s' \"$apps\" | tr '\\n' ';')\"; fi; "
        )
    gpu_preflight = ""
    if not args.wait_gpu_free and not args.allow_compute_share:
        gpu_preflight = (
            f"for gpu in {gpu_list}; do "
            f"apps=$({nvidia_smi} -i \"$gpu\" --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | head -n 3 || true); "
            "if [ -n \"$apps\" ]; then "
            "echo \"GPU_BUSY gpu=$gpu compute_apps=$(printf '%s' \"$apps\" | tr '\\n' ';')\" >&2; exit 3; fi; "
            "done; "
        )
    wait_gpu = ""
    if args.wait_gpu_free:
        wait_gpu = (
            "while true; do "
            f"status=$({nvidia_smi} --query-gpu=index,memory.used,utilization.gpu "
            "--format=csv,noheader,nounits); "
            "ok=1; wait_msg=''; "
            f"for gpu in {gpu_list}; do "
            "line=$(printf '%s\n' \"$status\" | awk -F, -v g=\"$gpu\" "
            "'$1+0 == g+0 {gsub(/ /,\"\",$2); gsub(/ /,\"\",$3); print $2\" \"$3; exit}'); "
            "mem=$(printf '%s' \"$line\" | awk '{print $1}'); "
            "util=$(printf '%s' \"$line\" | awk '{print $2}'); "
            "if [ -z \"$mem\" ] || [ -z \"$util\" ] || "
            f"[ \"$mem\" -gt {shlex.quote(args.wait_gpu_mem)} ] || "
            f"[ \"$util\" -gt {shlex.quote(args.wait_gpu_util)} ]; then "
            "ok=0; wait_msg=\"$wait_msg gpu=$gpu mem=${mem:-unknown}MiB util=${util:-unknown}%\"; fi; "
            f"{compute_check}"
            "done; "
            "if [ \"$ok\" -eq 1 ]; then "
            f"echo \"[$(date)] GPU wait satisfied: gpus={','.join(selected_gpus)}\"; break; fi; "
            "echo \"[$(date)] Waiting for GPUs:"
            "${wait_msg}\"; "
            f"sleep {shlex.quote(args.wait_gpu_interval)}; "
            "done; "
        )
    remote_cmd = (
        f"cd {shlex.quote(args.remote_dir)} && {cuda_env}{cuda_visible}{gpu_preflight}{wait_gpu}"
        "mkdir -p " + shlex.quote(str(Path(remote_out).parent)) + " && "
        "BEM_ORIENT_PROGRESS=100000 " + " ".join(shlex.quote(x) for x in bem_cmd)
    )
    if args.dry_run:
        print(remote_cmd)
        return

    if args.skip_existing and local_out.is_file():
        print(f"Using existing local result: {local_out}")
    else:
        run(["ssh", args.remote, remote_cmd])
        run([
            "rsync", "-az",
            f"{args.remote}:{args.remote_dir}/{remote_out}",
            str(local_out),
        ])
    score = subprocess.check_output([
        "scripts/score_mbs.py",
        "--bem", str(local_out),
        "--mbs", mbs_path,
        "--theta-max", args.theta_max,
    ], text=True)
    print(score, end="")

    with local_out.open("r") as f:
        data = json.load(f)
    timing = data.get("timing", {})
    if timing:
        print("Timing summary:", " ".join(f"{k}={v}" for k, v in timing.items()))


if __name__ == "__main__":
    main()
