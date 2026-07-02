#!/usr/bin/env python3
"""Independent PMCHWT block-forming checks.

The CUDA code computes L/K operators elsewhere.  This script audits the final
2N x 2N block contract:

    [ eta_e L_e + eta_i L_i      -(K_e + K_i) ]
    [ K_e + K_i                  L_e/eta_e + L_i/eta_i ]

It is intentionally small and independent of the C++ implementation so sign
mistakes in fast paths are caught before comparing physics.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def form_pmchwt_numpy(le, ke, li, ki, eta_e, eta_i):
    le = np.asarray(le, dtype=np.complex128)
    ke = np.asarray(ke, dtype=np.complex128)
    li = np.asarray(li, dtype=np.complex128)
    ki = np.asarray(ki, dtype=np.complex128)
    if le.shape != ke.shape or le.shape != li.shape or le.shape != ki.shape:
        raise ValueError("L/K shapes differ")
    if le.ndim != 2 or le.shape[0] != le.shape[1]:
        raise ValueError("operators must be square matrices")
    n = le.shape[0]
    z = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    ksum = ke + ki
    z[:n, :n] = eta_e * le + eta_i * li
    z[:n, n:] = -ksum
    z[n:, :n] = ksum
    z[n:, n:] = le / eta_e + li / eta_i
    return z


def block_scales(system, refractive_index, eta_i, experimental_nform=False):
    scales = {
        "unknown_m_scale": 1.0,
        "row_h_scale": 1.0 + 0j,
        "int_op_sign": 1.0,
        "k_identity": 0.0,
        "n_form": False,
        "n_form_eps_int": 1.0,
        "n_form_m_identity": 0.0,
    }
    if system in {"muller", "muller-balanced"} or experimental_nform:
        scales["int_op_sign"] = -1.0
    if system in {"balanced", "muller-balanced", "muller2-balanced"}:
        scales["unknown_m_scale"] = abs(refractive_index)
        scales["row_h_scale"] = eta_i
    if experimental_nform:
        eps_int = abs(refractive_index) ** 2
        scales.update(
            unknown_m_scale=1.0,
            row_h_scale=1.0 + 0j,
            int_op_sign=-1.0,
            k_identity=-1.0,
            n_form=True,
            n_form_eps_int=eps_int,
            n_form_m_identity=-0.5 * (1.0 + eps_int),
        )
    return scales


def form_general_system_numpy(le, ke, li, ki, eta_e, eta_i, scales):
    le = np.asarray(le, dtype=np.complex128)
    ke = np.asarray(ke, dtype=np.complex128)
    li = np.asarray(li, dtype=np.complex128)
    ki = np.asarray(ki, dtype=np.complex128)
    if le.shape != ke.shape or le.shape != li.shape or le.shape != ki.shape:
        raise ValueError("L/K shapes differ")
    if le.ndim != 2 or le.shape[0] != le.shape[1]:
        raise ValueError("operators must be square matrices")
    n = le.shape[0]
    z = np.zeros((2 * n, 2 * n), dtype=np.complex128)
    int_sign = scales["int_op_sign"]
    inv_m = 1.0 / scales["unknown_m_scale"]
    row_h = scales["row_h_scale"]
    ksum = ke + int_sign * ki + np.eye(n, dtype=np.complex128) * scales["k_identity"]
    z[:n, :n] = eta_e * le + int_sign * eta_i * li
    z[:n, n:] = -ksum * inv_m
    z[n:, :n] = row_h * ksum
    z[n:, n:] = row_h * (le / eta_e + int_sign * li / eta_i) * inv_m
    return z


def self_test(seed: int = 1234, n: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    mats = [rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) for _ in range(4)]
    eta_e = 1.0 + 0j
    eta_i = 0.7 + 0.15j
    z = form_pmchwt_numpy(*mats, eta_e, eta_i)
    ksum = mats[1] + mats[3]
    report = {
        "n": n,
        "offdiag_antisymmetry_l2": float(np.linalg.norm(z[:n, n:] + z[n:, :n])),
        "top_left_l2": float(np.linalg.norm(z[:n, :n] - (eta_e * mats[0] + eta_i * mats[2]))),
        "bottom_right_l2": float(np.linalg.norm(z[n:, n:] - (mats[0] / eta_e + mats[2] / eta_i))),
        "ksum_l2": float(np.linalg.norm(z[n:, :n] - ksum)),
    }
    refractive_index = 1.6 + 0.2j
    eta_m = 1.0 / refractive_index
    systems = {}
    for name, experimental in [
        ("pmchwt", False),
        ("balanced", False),
        ("muller", False),
        ("muller-balanced", False),
        ("muller2-balanced", True),
    ]:
        scales = block_scales(name, refractive_index, eta_m, experimental)
        zg = form_general_system_numpy(*mats, eta_e, eta_m, scales)
        inv_m = 1.0 / scales["unknown_m_scale"]
        ksum = mats[1] + scales["int_op_sign"] * mats[3] + np.eye(n) * scales["k_identity"]
        systems[name] = {
            "top_left_l2": float(np.linalg.norm(zg[:n, :n] - (eta_e * mats[0] + scales["int_op_sign"] * eta_m * mats[2]))),
            "top_right_l2": float(np.linalg.norm(zg[:n, n:] + inv_m * ksum)),
            "bottom_left_l2": float(np.linalg.norm(zg[n:, :n] - scales["row_h_scale"] * ksum)),
            "bottom_right_l2": float(np.linalg.norm(zg[n:, n:] - scales["row_h_scale"] * inv_m * (mats[0] / eta_e + scales["int_op_sign"] * mats[2] / eta_m))),
            "unknown_m_scale": float(scales["unknown_m_scale"]),
            "int_op_sign": float(scales["int_op_sign"]),
            "k_identity": float(scales["k_identity"]),
            "n_form": bool(scales["n_form"]),
        }
    report["systems"] = systems
    l2_values = [report[k] for k in report if k.endswith("_l2")]
    for system in systems.values():
        l2_values.extend(v for k, v in system.items() if k.endswith("_l2"))
    report["pass"] = all(v < 1e-12 for v in l2_values)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.self_test:
        raise SystemExit("currently use --self-test; file-based C++ dumps can be added when needed")
    report = self_test(args.seed, args.n)
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
