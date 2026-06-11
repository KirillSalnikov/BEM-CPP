#!/usr/bin/env python3
"""Run one BEM candidate on epyc1 and score it against MBS.

This is intentionally narrow: it standardizes the Greek-particle benchmark
used for ADDA dpl25 comparisons so mesh/solver experiments are reproducible.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

from greek_profiles import select_greek_profile


DEFAULT_REMOTE = "kirill_epyc@172.16.0.212"
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
    parser.add_argument("--quad", default="4")
    parser.add_argument("--ntheta", default="19")
    parser.add_argument("--solver", default="dense")
    parser.add_argument("--system", default="pmchwt")
    parser.add_argument("--mbs", help="MBS/ADDA reference table; defaults to A_x=<ka> in --mbs-dir")
    parser.add_argument("--mbs-dir", default=DEFAULT_MBS_DIR)
    parser.add_argument("--theta-max", default="180")
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
    bem_cmd = [
        "./bin/bem_cuda_fmm",
        "--solver", args.solver,
        "--system", args.system,
        "--obj", obj_path,
        "--ka", args.ka,
        "--ri", args.ri[0], args.ri[1],
        "--orient", args.orient[0], args.orient[1], args.orient[2],
        "--quad", args.quad,
        "--ntheta", args.ntheta,
        "--out", remote_out,
    ]
    remote_cmd = (
        f"cd {shlex.quote(args.remote_dir)} && {cuda_env}"
        "mkdir -p " + shlex.quote(str(Path(remote_out).parent)) + " && "
        "BEM_ORIENT_PROGRESS=100000 " + " ".join(shlex.quote(x) for x in bem_cmd)
    )
    if args.dry_run:
        print(remote_cmd)
        return

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
