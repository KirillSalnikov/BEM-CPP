#!/usr/bin/env python3
"""Compare BEM JSON Mueller output against ADDA mueller files or tables."""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


ALL_COMPONENTS = {
    f"S{i + 1}{j + 1}": (i, j, 1 + 4 * i + j, 2 + 4 * i + j)
    for i in range(4)
    for j in range(4)
}

DEFAULT_COMPONENT_NAMES = ("S11", "S12", "S22", "S33", "S34", "S44")
COMPONENTS = {name: ALL_COMPONENTS[name] for name in DEFAULT_COMPONENT_NAMES}

ORIENT_RE = re.compile(
    r"-orient\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def bem_component(mueller, theta, i, j):
    m = np.asarray(mueller)
    n = len(theta)
    if m.shape == (4, 4, n):
        return m[i, j]
    if m.shape == (n, 4, 4):
        return m[:, i, j]
    if m.shape == (16, n):
        return m[4 * i + j]
    if m.shape == (n, 16):
        return m[:, 4 * i + j]
    raise ValueError(f"unknown BEM mueller shape: {m.shape}")


def load_bem(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.asarray(data["theta"]), np.asarray(data["mueller"]), data.get("timing", {})


def load_adda_average(path, beta_order):
    files = sorted(glob.glob(str(Path(path) / "*" / "mueller")))
    if Path(path).is_file():
        files = [str(path)]
    if not files:
        raise FileNotFoundError(f"no ADDA mueller files under {path}")
    missing_logs = [name for name in files if not (Path(name).parent / "log").exists()]
    if missing_logs:
        sample = ", ".join(missing_logs[:3])
        raise FileNotFoundError(
            "ADDA comparison requires raw ADDA directories with both mueller and log; "
            f"missing log for {len(missing_logs)} mueller files, e.g. {sample}"
        )

    beta_weights = None
    if beta_order:
        nodes, weights = np.polynomial.legendre.leggauss(beta_order)
        betas = np.degrees(np.arccos(nodes))
        beta_weights = [(float(b), float(w)) for b, w in zip(betas, weights)]

    samples = []
    theta_union = []
    for file_name in files:
        data = np.loadtxt(file_name, skiprows=1)
        weight = 1.0
        if beta_weights is not None:
            match = re.search(r"_b([0-9p]+)_g", file_name)
            if match:
                beta = float(match.group(1).replace("p", "."))
            else:
                log_text = (Path(file_name).parent / "log").read_text(errors="replace")
                orient_match = ORIENT_RE.search(log_text)
                if not orient_match:
                    raise ValueError(f"cannot parse beta from {file_name}")
                beta = float(orient_match.group(2))
            weight = min(beta_weights, key=lambda item: abs(item[0] - beta))[1]
        samples.append((data, weight))
        theta_union.append(data[:, 0])
    theta = np.unique(np.concatenate(theta_union))
    acc = np.zeros((len(theta), samples[0][0].shape[1]), dtype=float)
    acc[:, 0] = theta
    wsum = 0.0
    for data, weight in samples:
        acc[:, 1:] += weight * np.column_stack([
            np.interp(theta, data[:, 0], data[:, col])
            for col in range(1, data.shape[1])
        ])
        wsum += weight
    acc[:, 1:] /= wsum
    return acc, len(files)


def load_mbs_table(path):
    return np.loadtxt(path, skiprows=1)


def load_named_table(path):
    table = np.genfromtxt(path, names=True)
    if table.dtype.names is None or "theta" not in table.dtype.names:
        raise ValueError(f"{path} is not a named theta/Sij table")
    return table


def parse_stokes_signs(text):
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three signs for Q,U,V, e.g. -1,1,-1")
    out = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid sign {part!r}") from exc
        if value not in (-1.0, 1.0):
            raise argparse.ArgumentTypeError("Stokes basis transform supports only +/-1 signs")
        out.append(value)
    return tuple(out)


def apply_stokes_signs(mueller, theta, out_signs, in_signs):
    m = np.asarray(mueller, dtype=float)
    n = len(theta)
    if m.shape == (4, 4, n):
        data = m.copy()
        layout = "44n"
    elif m.shape == (n, 4, 4):
        data = np.moveaxis(m, 0, -1).copy()
        layout = "n44"
    elif m.shape == (16, n):
        data = m.reshape(4, 4, n).copy()
        layout = "16n"
    elif m.shape == (n, 16):
        data = np.moveaxis(m.reshape(n, 4, 4), 0, -1).copy()
        layout = "n16"
    else:
        raise ValueError(f"unknown BEM mueller shape: {m.shape}")

    left = np.diag((1.0, *out_signs))
    right = np.diag((1.0, *in_signs))
    data = np.einsum("ab,bcT,cd->adT", left, data, right)

    if layout == "44n":
        return data
    if layout == "n44":
        return np.moveaxis(data, -1, 0)
    if layout == "16n":
        return data.reshape(16, n)
    return np.moveaxis(data, -1, 0).reshape(n, 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", required=True)
    parser.add_argument("--adda", help="ADDA mueller file or directory with */mueller files")
    parser.add_argument("--mbs", help="MBS/converted table")
    parser.add_argument("--adda-table",
                        help="ADDA_for_PO_comparison table with theta and S11..S44 columns")
    parser.add_argument("--beta-order", type=int, default=0,
                        help="Use Gauss-Legendre beta weights for ADDA directory averaging")
    parser.add_argument("--raw", action="store_true",
                        help="Compare raw Mueller values instead of normalizing by S11(theta=0)")
    parser.add_argument("--elements", default=",".join(DEFAULT_COMPONENT_NAMES),
                        help="Comma-separated Mueller elements, e.g. S11,S12 or all")
    parser.add_argument("--component-floor", type=float, default=0.0,
                        help="Floor for normalized reference magnitude in relative RMS error")
    parser.add_argument("--bem-stokes-out", type=parse_stokes_signs, default=(1.0, 1.0, 1.0),
                        metavar="Q,U,V",
                        help="Signed output Stokes-basis transform for BEM, e.g. 1,-1,-1")
    parser.add_argument("--bem-stokes-in", type=parse_stokes_signs, default=(1.0, 1.0, 1.0),
                        metavar="Q,U,V",
                        help="Signed input Stokes-basis transform for BEM, e.g. -1,-1,1")
    args = parser.parse_args()

    ref_modes = sum(bool(x) for x in (args.adda, args.mbs, args.adda_table))
    if ref_modes != 1:
        raise SystemExit("provide exactly one of --adda, --mbs, or --adda-table")

    theta, bem_mueller, timing = load_bem(args.bem)
    if args.bem_stokes_out != (1.0, 1.0, 1.0) or args.bem_stokes_in != (1.0, 1.0, 1.0):
        bem_mueller = apply_stokes_signs(
            bem_mueller, theta, args.bem_stokes_out, args.bem_stokes_in)
    if args.adda:
        ref, ref_count = load_adda_average(args.adda, args.beta_order)
        ref_kind = f"ADDA ({ref_count} files)"
        ref_col = lambda name: ALL_COMPONENTS[name][2]
        ref_value = lambda name, th: np.interp(th, ref[:, 0], ref[:, ref_col(name)])
    elif args.mbs:
        ref = load_mbs_table(args.mbs)
        ref_kind = "MBS table"
        ref_col = lambda name: ALL_COMPONENTS[name][3]
        ref_value = lambda name, th: np.interp(th, ref[:, 0], ref[:, ref_col(name)])
    else:
        ref = load_named_table(args.adda_table)
        ref_kind = "ADDA table"
        ref_value = lambda name, th: np.interp(th, ref["theta"], ref[name])

    bem_s11 = bem_component(bem_mueller, theta, 0, 0)
    ref_s11 = ref_value("S11", theta)
    bem_norm = 1.0 if args.raw else bem_s11[0]
    ref_norm = 1.0 if args.raw else ref_s11[0]

    print(f"BEM: {args.bem}")
    print(f"Reference: {ref_kind}")
    if timing:
        print("Timing:", " ".join(f"{k}={v}" for k, v in timing.items()))
    if not args.raw:
        print(f"Scale BEM/REF S11(0): {bem_s11[0] / ref_s11[0]:.8g}")
    if args.bem_stokes_out != (1.0, 1.0, 1.0) or args.bem_stokes_in != (1.0, 1.0, 1.0):
        print(
            "BEM Stokes transform: "
            f"out QUV={args.bem_stokes_out}, in QUV={args.bem_stokes_in}"
        )

    if args.elements.strip().lower() == "all":
        names = list(ALL_COMPONENTS)
    else:
        names = [x.strip().upper() for x in args.elements.split(",") if x.strip()]
    unknown = [name for name in names if name not in ALL_COMPONENTS]
    if unknown:
        raise SystemExit("unknown Mueller elements: " + ", ".join(unknown))

    score = 0.0
    for name in names:
        i, j, _, _ = ALL_COMPONENTS[name]
        y = bem_component(bem_mueller, theta, i, j) / bem_norm
        r = ref_value(name, theta) / ref_norm
        if args.component_floor > 0.0:
            den = np.sqrt(np.mean(np.maximum(np.abs(r), args.component_floor) ** 2))
            err = np.sqrt(np.mean((y - r) ** 2)) / max(den, 1e-300)
        else:
            err = np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-300)
        score += err
        print(f"{name}: {err:.6g}")
    print(f"score{len(names)}: {score:.6g}")


if __name__ == "__main__":
    main()
