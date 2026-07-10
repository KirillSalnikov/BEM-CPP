#!/usr/bin/env python3
"""Summarize M11 agreement between BEM JSON and ADDA mueller/table output."""

import argparse
import json
from pathlib import Path

import numpy as np


def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def load_bem(path: Path):
    with path.open() as f:
        data = json.load(f)
    theta = np.asarray(data["theta"], dtype=float)
    mueller = np.asarray(data["mueller"], dtype=float)
    if mueller.shape == (4, 4, len(theta)):
        pass
    elif mueller.shape == (len(theta), 4, 4):
        mueller = np.moveaxis(mueller, 0, -1)
    elif mueller.shape == (16, len(theta)):
        mueller = mueller.reshape(4, 4, len(theta))
    elif mueller.shape == (len(theta), 16):
        mueller = np.moveaxis(mueller.reshape(len(theta), 4, 4), 0, -1)
    else:
        raise ValueError(f"unsupported BEM Mueller shape: {mueller.shape}")
    return theta, mueller, data


def load_adda(path: Path, theta: np.ndarray):
    table = np.genfromtxt(path, names=True)
    if table.dtype.names is None:
        raise ValueError(f"{path} must have a header row")
    names = {name.lower(): name for name in table.dtype.names}
    if "theta" not in names or "s11" not in names:
        raise ValueError(f"{path} must contain theta and S11/s11 columns")
    ref_theta = np.asarray(table[names["theta"]], dtype=float)
    ref_m11 = np.asarray(table[names["s11"]], dtype=float)
    return np.interp(theta, ref_theta, ref_m11), ref_theta


def rel_l2(a: np.ndarray, b: np.ndarray):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def weighted_rel_l2(a: np.ndarray, b: np.ndarray, weight: np.ndarray, x: np.ndarray):
    num = trapz((a - b) ** 2 * weight, x)
    den = trapz(b ** 2 * weight, x)
    return float(np.sqrt(max(num, 0.0) / max(den, 1e-300)))


def angular_grid_metrics(theta: np.ndarray, ref_theta: np.ndarray):
    same_grid = (
        theta.shape == ref_theta.shape
        and bool(np.allclose(theta, ref_theta, rtol=0.0, atol=1e-10))
    )
    bem_max_step = float(np.max(np.diff(theta))) if theta.size > 1 else 0.0
    adda_min_step = float(np.min(np.diff(ref_theta))) if ref_theta.size > 1 else 0.0
    covers_reference = (
        theta.size > 1
        and ref_theta.size > 1
        and theta[0] <= ref_theta[0] + 1e-10
        and theta[-1] >= ref_theta[-1] - 1e-10
    )
    resolves_reference = covers_reference and bem_max_step <= adda_min_step + 1e-9
    return {
        "bem_ntheta": int(theta.size),
        "adda_ntheta": int(ref_theta.size),
        "bem_max_angle_step_degrees": bem_max_step,
        "adda_min_angle_step_degrees": adda_min_step,
        "adda_max_angle_step_degrees": (
            float(np.max(np.diff(ref_theta))) if ref_theta.size > 1 else 0.0
        ),
        # Numeric values keep the line-oriented report parser unambiguous.
        "angular_grid_matches_reference": 1.0 if same_grid else 0.0,
        "angular_grid_resolves_reference": 1.0 if resolves_reference else 0.0,
    }


def m11_metrics(theta_deg: np.ndarray, bem_m11: np.ndarray, ref_m11: np.ndarray):
    theta_rad = np.deg2rad(theta_deg)
    weight = np.sin(theta_rad)
    bem_i = trapz(bem_m11 * weight, theta_rad)
    ref_i = trapz(ref_m11 * weight, theta_rad)
    bem_integral_norm = bem_m11 / bem_i
    ref_integral_norm = ref_m11 / ref_i
    metrics = {
        "m11_forward_rel_l2": rel_l2(bem_m11 / bem_m11[0], ref_m11 / ref_m11[0]),
        # Random-orientation scattering is integrated over solid angle, so the
        # curve norm must use the same sin(theta) dtheta measure.
        "m11_integral_rel_l2": weighted_rel_l2(
            bem_integral_norm, ref_integral_norm, weight, theta_rad
        ),
        "m11_integral_rel_l2_unweighted_points": rel_l2(
            bem_integral_norm, ref_integral_norm
        ),
        "raw_forward_ratio": float(bem_m11[0] / ref_m11[0]),
        "raw_integral_ratio": float(bem_i / ref_i),
        "shape_integral_over_forward": float((bem_i / bem_m11[0]) / (ref_i / ref_m11[0])),
    }
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80), (80, 120), (120, 180)):
        mask = (theta_deg >= lo) & (theta_deg <= hi)
        if int(np.count_nonzero(mask)) < 2:
            continue
        bem_band = trapz(bem_m11[mask] * weight[mask], theta_rad[mask])
        ref_band = trapz(ref_m11[mask] * weight[mask], theta_rad[mask])
        metrics[f"raw_band_ratio_{lo}_{hi}"] = bem_band / ref_band if abs(ref_band) > 1e-300 else float("nan")
        metrics[f"ref_integral_fraction_{lo}_{hi}"] = ref_band / ref_i if abs(ref_i) > 1e-300 else float("nan")
        metrics[f"bem_integral_fraction_{lo}_{hi}"] = bem_band / bem_i if abs(bem_i) > 1e-300 else float("nan")
        metrics[f"integral_ratio_contribution_{lo}_{hi}"] = bem_band / ref_i if abs(ref_i) > 1e-300 else float("nan")
        bshape = bem_m11[mask] / bem_m11[0]
        rshape = ref_m11[mask] / ref_m11[0]
        metrics[f"shape_band_ratio_{lo}_{hi}"] = (
            trapz(bshape * weight[mask], theta_rad[mask]) /
            trapz(rshape * weight[mask], theta_rad[mask])
        )
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bem", type=Path, required=True)
    ap.add_argument("--adda", type=Path, required=True,
                    help="ADDA mueller file or named theta/S11 table")
    ap.add_argument("--unweight-bem", action="store_true",
                    help="Divide BEM Mueller matrix by orientation_weight_sum; use for single weighted orientation chunks")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    theta, mueller, meta = load_bem(args.bem)
    if args.unweight_bem:
        orient_weight = float(meta.get("orientation_weight_sum", 1.0) or 1.0)
        if abs(orient_weight) < 1e-300:
            raise ValueError(f"{args.bem} has zero orientation_weight_sum")
        mueller = mueller / orient_weight
    ref_m11, ref_theta = load_adda(args.adda, theta)
    metrics = m11_metrics(theta, mueller[0, 0], ref_m11)
    metrics.update(angular_grid_metrics(theta, ref_theta))
    metrics["bem_unweighted"] = bool(args.unweight_bem)
    metrics["bem_orientation_weight_sum"] = float(meta.get("orientation_weight_sum", np.nan))
    metrics["total_s"] = float(meta.get("timing", {}).get("total_s", np.nan))
    metrics["solve_s"] = float(meta.get("timing", {}).get("solve_s", np.nan))
    metrics["orient_count"] = int(meta.get("orient_count", meta.get("orient_total", 0)) or 0)
    metrics["alpha_avg"] = int(meta.get("alpha_avg", 1) or 1)
    if args.csv:
        keys = list(metrics)
        print(",".join(keys))
        print(",".join(str(metrics[k]) for k in keys))
    else:
        for key, value in metrics.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
