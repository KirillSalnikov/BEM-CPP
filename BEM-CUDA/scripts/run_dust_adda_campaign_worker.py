#!/usr/bin/env python3
"""Run one GPU lane of the ka>=30 dust/ADDA validation campaign."""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


TERMINAL = {"complete", "not_converged", "failed"}


def load_json(path):
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    tmp.replace(path)


def material_dir(case):
    return "refr_1_6__{}".format(str(case.get("ri_im", 0)).replace(".", "_"))


def case_tag(case):
    return "ka{}".format(str(case["ka"]).replace(".", "p"))


def accepted_path(root, manifest):
    value = manifest.get("accepted")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def metric_value(root, bem, adda):
    output = subprocess.check_output(
        ["python3", "scripts/summarize_bem_adda_m11.py", "--bem", str(bem),
         "--adda", str(root / adda)],
        cwd=str(root), universal_newlines=True,
    )
    for line in output.splitlines():
        if line.startswith("m11_integral_rel_l2: "):
            return float(line.split(": ", 1)[1])
    raise RuntimeError("M11 metric missing for {}".format(bem))


def wait_for_final(root, final_root, case, interval):
    directory = final_root / material_dir(case) / case_tag(case)
    manifest_path = directory / "adaptive_nested_bg_manifest.json"
    while True:
        manifest = load_json(manifest_path)
        if manifest and manifest.get("status") in TERMINAL:
            result = {
                "status": manifest.get("status"),
                "manifest": str(manifest_path),
                "orientation_converged": manifest.get("status") == "complete",
                "time": time.time(),
            }
            bem = accepted_path(root, manifest)
            if bem and bem.is_file():
                result["accepted"] = str(bem)
                result["m11_l2"] = metric_value(root, bem, case["adda"])
                result["m11_pass"] = result["m11_l2"] <= 0.10
            else:
                result["m11_pass"] = False
            return result
        time.sleep(interval)


def launch_profile(root, config, case, log_path):
    env = os.environ.copy()
    env.update({
        "ROOT": str(root),
        "RUN_ROOT": str(root / "runs/dust_adda_ka30plus_profiles_20260710"),
        "MANIFEST": str(root / config["orientation_manifest"]),
        "GPU": str(case["gpu"]),
        "KA": str(case["ka"]),
        "RI_IM": str(case.get("ri_im", 0)),
        "MESH": case["mesh"],
        "MAX_LEAF": str(case["max_leaf"]),
        "NTHETA": "181",
        "ADDA": case["adda"],
    })
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab", buffering=0)
    return subprocess.Popen(
        ["bash", "scripts/run_dust_adda_4gpu_goal.sh"], cwd=str(root), env=env,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
    )


def profile_is_active(root, case_dir):
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", "--", "adaptive_nested_bg_orient_queue.py"],
            universal_newlines=True,
        )
    except subprocess.CalledProcessError:
        return False
    expected = case_dir.resolve()
    for line in output.splitlines():
        try:
            command = line.split(" ", 1)[1]
            tokens = shlex.split(command)
            value = tokens[tokens.index("--out-dir") + 1]
        except (IndexError, ValueError):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.resolve() == expected:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True, choices=range(4))
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()
    config = load_json(args.config)
    if not config:
        raise SystemExit("cannot read {}".format(args.config))
    root = Path(config["root"])
    final_root = root / config["final_root"]
    state_root = final_root / "campaign_state"
    cases = [case for case in config["cases"] if case["gpu"] == args.gpu]

    for case in cases:
        case_id = case.get("id", case["tag"])
        state_path = state_root / (case_id + ".json")
        existing = load_json(state_path)
        if existing and existing.get("terminal"):
            continue
        promotion_path = final_root / "promotion" / (case_id + ".json")
        promotion = load_json(promotion_path)
        if not promotion:
            profile_dir = root / case["profile_dir"]
            process = None
            while not promotion:
                profile_bem = list(profile_dir.glob("level01_*/bem.json"))
                if (not profile_bem and process is None
                        and not profile_is_active(root, profile_dir)):
                    log = final_root / "profile_logs" / (case_id + ".log")
                    process = launch_profile(root, config, case, log)
                if process is not None and process.poll() is not None:
                    # A normal profile is terminated by the promoter. Give its
                    # atomic record one polling interval to become visible.
                    time.sleep(args.interval)
                    promotion = load_json(promotion_path)
                    process = None
                else:
                    time.sleep(args.interval)
                    promotion = load_json(promotion_path)

        if not promotion.get("profile_pass"):
            atomic_json(state_path, {
                "case": case_id, "terminal": True, "status": "profile_l2_failed",
                "profile_l2": promotion.get("profile_l2"), "time": time.time(),
            })
            continue

        result = wait_for_final(root, final_root, case, args.interval)
        result.update({"case": case_id, "terminal": True,
                       "profile_l2": promotion.get("profile_l2")})
        atomic_json(state_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
