#!/usr/bin/env python3
"""Summarize ADDA-like Jmax adaptive BEM orientation sweep results."""

import argparse
import json
import re
from pathlib import Path


def fmt(x, nd=4):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}g}"
    except Exception:
        return str(x)


def effective_orientation_count(nodes):
    if not isinstance(nodes, dict):
        return "-"
    try:
        return int(nodes.get("alpha", 0)) * int(nodes.get("beta", 0)) * int(nodes.get("gamma", 0))
    except Exception:
        return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    args = ap.parse_args()
    root = Path(args.run_root)
    rows = []
    seen_cases = set()
    manifests = list(root.glob("ka*/adaptive_jmax_manifest.json"))
    manifests += list(root.glob("ka*/pilot/adaptive_jmax_manifest.json"))
    for manifest in sorted(manifests):
        try:
            data = json.loads(manifest.read_text())
        except Exception as exc:
            rows.append((manifest.parent.name, "bad_manifest", str(exc), "", "", "", "", ""))
            continue
        case_dir = manifest.parent if manifest.parent.name.startswith("ka") else manifest.parent.parent
        final_manifest = case_dir / "adaptive_final_manifest.json"
        final_bem = case_dir / "final_quality" / "bem.json"
        levels = data.get("levels", [])
        accepted = data.get("accepted") or ""
        last = levels[-1] if levels else {}
        accepted_rec = None
        for rec in levels:
            if rec.get("accepted"):
                accepted_rec = rec
                break
        if accepted_rec is None:
            accepted_rec = last
        change = last.get("change_from_previous", {}) if isinstance(last, dict) else {}
        status = "accepted" if accepted else "running"
        if final_bem.exists():
            status = "final_done"
        elif final_manifest.exists():
            status = "final_manifest"
        elif accepted:
            status = "pilot_done"
        seen_cases.add(case_dir.name)
        rows.append((
            case_dir.name,
            status,
            "pilot",
            len(levels),
            accepted,
            accepted_rec.get("J", {}) if isinstance(accepted_rec, dict) else {},
            accepted_rec.get("N", {}) if isinstance(accepted_rec, dict) else {},
            "-",
            effective_orientation_count(accepted_rec.get("N", {}) if isinstance(accepted_rec, dict) else {}),
            fmt(change.get("score")),
            fmt(change.get("max")),
            fmt(change.get("scale_change")),
        ))

    level_re = re.compile(r"level\d+_Ja(\d+)_Jb(\d+)_Jg(\d+)_a(\d+)_b(\d+)_g(\d+)")
    orient_re = re.compile(r"Orient\s+(\d+)/(\d+)\s+done")
    for case_dir in sorted(root.glob("ka*")):
        if case_dir.name in seen_cases:
            continue
        latest = None
        phase = "pilot"
        for candidate in list((case_dir / "pilot").glob("level*")) + list((case_dir / "final_quality").glob("parts")):
            if candidate.is_dir():
                mtime = max((p.stat().st_mtime for p in candidate.glob("**/*") if p.is_file()), default=candidate.stat().st_mtime)
                if latest is None or mtime > latest[0]:
                    latest = (mtime, candidate)
        if latest is None:
            continue
        path = latest[1]
        if "final_quality" in path.parts:
            phase = "final"
            level = {}
            nodes = {}
        else:
            match = level_re.search(path.name)
            level = {"alpha": int(match.group(1)), "beta": int(match.group(2)), "gamma": int(match.group(3))} if match else {}
            nodes = {"alpha": int(match.group(4)), "beta": int(match.group(5)), "gamma": int(match.group(6))} if match else {}
        done = 0
        total = 0
        for log in path.glob("parts/part_*.log") if phase == "pilot" else path.glob("part_*.log"):
            text = log.read_text(errors="ignore")
            matches = orient_re.findall(text)
            if matches:
                d, t = matches[-1]
                done += int(d)
                total += int(t)
            elif "Orientation chunk:" in text:
                chunk_match = re.search(r"Orientation chunk: start=\d+ count=(\d+) of", text)
                if chunk_match:
                    total += int(chunk_match.group(1))
        if phase == "pilot" and nodes:
            total = int(nodes.get("beta", 0)) * int(nodes.get("gamma", 0))
        progress = "%d/%d" % (done, total) if total else "-"
        if phase == "pilot" and nodes:
            alpha = int(nodes.get("alpha", 0))
            effective_progress = "%d/%d" % (done * alpha, total * alpha)
        else:
            effective_progress = "-"
        rows.append((case_dir.name, "running", phase, "", "", level, nodes, progress, effective_progress, "-", "-", "-"))

    print("ka,status,phase,levels,accepted,J,N,solve_progress,effective_orient_progress,score,max,scale")
    for row in rows:
        print(",".join(str(x).replace(",", ";") for x in row))


if __name__ == "__main__":
    main()
