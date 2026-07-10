#!/usr/bin/env python3
"""Plot BEM Mueller JSON against a tabular ADDA A_x=...dat reference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ADDA_COLS = [
    "S11", "S12", "S13", "S14",
    "S21", "S22", "S23", "S24",
    "S31", "S32", "S33", "S34",
    "S41", "S42", "S43", "S44",
]


def trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def load_bem(path):
    with path.open() as f:
        meta = json.load(f)
    theta = np.asarray(meta["theta"], dtype=float)
    mueller = np.asarray(meta["mueller"], dtype=float)
    ntheta = len(theta)
    if mueller.shape == (4, 4, ntheta):
        pass
    elif mueller.shape == (ntheta, 4, 4):
        mueller = np.moveaxis(mueller, 0, -1)
    elif mueller.shape == (16, ntheta):
        mueller = mueller.reshape(4, 4, ntheta)
    elif mueller.shape == (ntheta, 16):
        mueller = np.moveaxis(mueller.reshape(ntheta, 4, 4), 0, -1)
    else:
        raise ValueError(f"unsupported BEM Mueller shape: {mueller.shape}")
    if np.nanmax(theta) <= np.pi + 1e-6:
        theta = np.rad2deg(theta)
    return theta, mueller, meta


def load_adda_table(path):
    table = np.genfromtxt(path, names=True)
    if table.dtype.names is None:
        raise ValueError(f"{path} has no header")
    names = {name.lower(): name for name in table.dtype.names}
    if "theta" not in names:
        raise ValueError(f"{path} has no theta column")
    theta = np.asarray(table[names["theta"]], dtype=float)
    mueller = np.zeros((4, 4, len(theta)), dtype=float)
    for idx, col in enumerate(ADDA_COLS):
        key = names.get(col.lower())
        if key is None:
            raise ValueError(f"{path} has no {col} column")
        mueller[idx // 4, idx % 4] = np.asarray(table[key], dtype=float)
    return theta, mueller


def interpolate_mueller(theta_src, mueller, theta_dst):
    out = np.zeros((4, 4, len(theta_dst)), dtype=float)
    for i in range(4):
        for j in range(4):
            out[i, j] = np.interp(theta_dst, theta_src, mueller[i, j])
    return out


def m11_integral_l2(theta_deg, bem_m11, adda_m11):
    theta_rad = np.deg2rad(theta_deg)
    weight = np.sin(theta_rad)
    bem_i = trapz(bem_m11 * weight, theta_rad)
    adda_i = trapz(adda_m11 * weight, theta_rad)
    bem_n = bem_m11 / max(abs(bem_i), 1e-300)
    adda_n = adda_m11 / max(abs(adda_i), 1e-300)
    num = trapz((bem_n - adda_n) ** 2 * weight, theta_rad)
    den = trapz(adda_n ** 2 * weight, theta_rad)
    return float(np.sqrt(num / max(den, 1e-300))), bem_n, adda_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", type=Path, required=True)
    parser.add_argument("--adda", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--title", default="BEM против ADDA")
    args = parser.parse_args()

    theta_b, bem, meta = load_bem(args.bem)
    theta_a, adda = load_adda_table(args.adda)
    bem_i = interpolate_mueller(theta_b, bem, theta_a)
    rel_l2, bem_m11_n, adda_m11_n = m11_integral_l2(theta_a, bem_i[0, 0], adda[0, 0])

    fig, axes = plt.subplots(4, 4, figsize=(15.8, 11.2), sharex=True)
    fig.suptitle(args.title, fontsize=16, y=0.985)
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == 0 and j == 0:
                ax.plot(theta_a, adda_m11_n, color="#1f77b4", lw=2.0, label="ADDA")
                ax.plot(theta_a, bem_m11_n, color="#d62728", lw=1.8, ls="--", label="BEM")
                ax.set_yscale("log")
                ax.set_title(r"$M_{11}$, нормировка на интеграл", fontsize=10)
                ax.text(
                    0.03,
                    0.05,
                    f"L2 = {100.0 * rel_l2:.1f}%",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.85),
                )
                ax.legend(loc="upper right", fontsize=9)
            else:
                adda_den = np.where(np.abs(adda[0, 0]) > 1e-300, adda[0, 0], np.nan)
                bem_den = np.where(np.abs(bem_i[0, 0]) > 1e-300, bem_i[0, 0], np.nan)
                ya = adda[i, j] / adda_den
                yb = bem_i[i, j] / bem_den
                ax.plot(theta_a, ya, color="#1f77b4", lw=1.5)
                ax.plot(theta_a, yb, color="#d62728", lw=1.5, ls="--")
                ax.set_title(rf"$M_{{{i + 1}{j + 1}}}/M_{{11}}$", fontsize=10)
                finite = np.isfinite(ya) & np.isfinite(yb)
                if np.count_nonzero(finite) > 2:
                    lo = np.nanpercentile(np.r_[ya[finite], yb[finite]], 1)
                    hi = np.nanpercentile(np.r_[ya[finite], yb[finite]], 99)
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        pad = 0.12 * (hi - lo)
                        ax.set_ylim(lo - pad, hi + pad)
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.set_xlim(0, 180)
            if i == 3:
                ax.set_xlabel("Угол рассеяния, градусы")
            if j == 0:
                ax.set_ylabel("Значение")

    orient_count = meta.get("orient_count", meta.get("orient_total", "н/д"))
    alpha_avg = meta.get("alpha_avg", 1)
    fig.text(
        0.5,
        0.012,
        f"BEM: {orient_count} узлов beta/gamma x {alpha_avg} alpha. "
        "Для M11 показана форма, нормированная на угловой интеграл; остальные элементы делены на M11.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.965])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    if args.pdf:
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf)
    print(args.out)
    if args.pdf:
        print(args.pdf)
    print(f"m11_integral_rel_l2: {rel_l2}")


if __name__ == "__main__":
    raise SystemExit(main())
