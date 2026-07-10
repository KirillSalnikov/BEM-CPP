#!/usr/bin/env python3
"""Generate nested beta/gamma grids for reusable BEM orientation averaging.

The master grid is the finest ADDA-style beta/gamma grid.  Each coarser level
is represented as a weight file with the same line count as the master grid and
zero weights for inactive master points.  This lets BEM solve every master
orientation at most once and recombine different J-level averages cheaply.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_orientation_helpers():
    helper_path = Path(__file__).resolve().parent / "generate_adda_avg_orientations.py"
    spec = importlib.util.spec_from_file_location("generate_adda_avg_orientations", helper_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def bg_rows(alpha, beta, gamma):
    full_alpha_range = abs((alpha.max_deg - alpha.min_deg) - 360.0) < 1e-10
    rows = {}
    for b, wb in zip(beta.values, beta.weights):
        pole = full_alpha_range and (
            abs(b - beta.min_deg) < 1e-10 or abs(b - beta.max_deg) < 1e-10
        )
        for g, wg in zip(gamma.values, gamma.weights):
            gout = 0.0 if pole else g
            key = (round(b, 12), round(gout, 12))
            rows[key] = rows.get(key, 0.0) + wb * wg
    total = sum(rows.values())
    if total <= 0.0:
        raise RuntimeError("zero beta/gamma weight sum")
    return [(b, g, w / total) for (b, g), w in rows.items()]


def axis_from_template(mod, template_text, name, j, cos_space):
    import re

    repl = re.sub(rf"(^\s*Jmin\s*=\s*)\d+", rf"\g<1>{j}", template_text, flags=re.M)
    repl = re.sub(rf"(^\s*Jmax\s*=\s*)\d+", rf"\g<1>{j}", repl, flags=re.M)
    return mod.parse_axis(repl, name, cos_space=cos_space)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True,
                    help="ADDA-style avg parameter file with alpha/beta/gamma sections")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--jmin-beta", type=int, default=2)
    ap.add_argument("--jmax-beta", type=int, default=5)
    ap.add_argument("--jmin-gamma", type=int, default=2)
    ap.add_argument("--jmax-gamma", type=int, default=5)
    ap.add_argument("--j-alpha", type=int, default=8)
    args = ap.parse_args()

    if args.jmax_beta < args.jmin_beta or args.jmax_gamma < args.jmin_gamma:
        raise SystemExit("Jmax must be >= Jmin")

    mod = load_orientation_helpers()
    text = args.template.read_text()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = axis_from_template(mod, text, "alpha", args.j_alpha, cos_space=False)
    beta_master = axis_from_template(mod, text, "beta", args.jmax_beta, cos_space=True)
    gamma_master = axis_from_template(mod, text, "gamma", args.jmax_gamma, cos_space=False)
    master = bg_rows(alpha, beta_master, gamma_master)
    master_index = {(round(b, 12), round(g, 12)): i for i, (b, g, _w) in enumerate(master)}

    master_file = out_dir / f"master_bg_Jb{args.jmax_beta}_Jg{args.jmax_gamma}.txt"
    with master_file.open("w") as f:
        f.write("# beta_deg gamma_deg weight\n")
        f.write("# finest nested beta/gamma grid; use with --alpha-avg\n")
        for b, g, w in master:
            f.write(f"{b:.17g} {g:.17g} {w:.17g}\n")

    levels = []
    nsteps = max(args.jmax_beta - args.jmin_beta, args.jmax_gamma - args.jmin_gamma)
    for step in range(nsteps + 1):
        jb = min(args.jmax_beta, args.jmin_beta + step)
        jg = min(args.jmax_gamma, args.jmin_gamma + step)
        beta = axis_from_template(mod, text, "beta", jb, cos_space=True)
        gamma = axis_from_template(mod, text, "gamma", jg, cos_space=False)
        rows = bg_rows(alpha, beta, gamma)
        weights = [0.0] * len(master)
        active = []
        for b, g, w in rows:
            key = (round(b, 12), round(g, 12))
            if key not in master_index:
                raise RuntimeError(f"level Jb={jb} Jg={jg} point is not nested in master: {key}")
            idx = master_index[key]
            weights[idx] = w
            active.append(idx)
        level_file = out_dir / f"level_Jb{jb}_Jg{jg}_weights.txt"
        active_file = out_dir / f"level_Jb{jb}_Jg{jg}_active_indices.txt"
        with level_file.open("w") as f:
            f.write("# beta_deg gamma_deg weight\n")
            for (b, g, _mw), w in zip(master, weights):
                f.write(f"{b:.17g} {g:.17g} {w:.17g}\n")
        with active_file.open("w") as f:
            for idx in sorted(active):
                f.write(f"{idx}\n")
        levels.append({
            "J": {"beta": jb, "gamma": jg},
            "N": {"beta": len(beta.values), "gamma": len(gamma.values)},
            "active_count": len(active),
            "file": str(level_file),
            "active_indices_file": str(active_file),
        })

    manifest = {
        "mode": "nested_bg_manifest",
        "template": str(args.template),
        "J": {"alpha": args.j_alpha, "beta_master": args.jmax_beta, "gamma_master": args.jmax_gamma},
        "N": {"alpha": len(alpha.values), "master_bg": len(master)},
        "master_file": str(master_file),
        "levels": levels,
    }
    manifest_path = out_dir / "nested_bg_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {manifest_path}")
    print(f"master_bg={len(master)}")
    for level in levels:
        print(f"Jb={level['J']['beta']} Jg={level['J']['gamma']} active={level['active_count']}")


if __name__ == "__main__":
    main()
