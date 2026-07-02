#!/usr/bin/env python3
"""Audit BEM/ADDA Mueller files for sign, shape and physical consistency."""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_mueller  # noqa: E402


COMPONENTS = getattr(compare_mueller, "ALL_COMPONENTS", compare_mueller.COMPONENTS)


def parse_component_names(spec: str) -> List[str]:
    requested = [x.strip().upper() for x in spec.split(",") if x.strip()]
    if any(name == "ALL" for name in requested):
        return list(COMPONENTS.keys())
    names = []
    for name in requested:
        if name.startswith("M") and len(name) == 3 and name[1:].isdigit():
            name = "S" + name[1:]
        names.append(name)
    unknown = [name for name in names if name not in COMPONENTS]
    if unknown:
        raise SystemExit(f"unknown Mueller component(s): {', '.join(unknown)}")
    return names


def load_bem_mueller(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    theta, mueller, meta = compare_mueller.load_bem(path)
    theta = np.asarray(theta, dtype=float)
    m = np.asarray(mueller, dtype=float)
    n = len(theta)
    if m.shape == (4, 4, n):
        pass
    elif m.shape == (n, 4, 4):
        m = np.moveaxis(m, 0, -1)
    elif m.shape == (16, n):
        m = m.reshape(4, 4, n)
    elif m.shape == (n, 16):
        m = np.moveaxis(m.reshape(n, 4, 4), 0, -1)
    else:
        raise ValueError(f"unsupported Mueller shape in {path}: {m.shape}")
    return theta, m, meta


def relative_l2(y: np.ndarray, r: np.ndarray) -> float:
    return float(np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-300))


PAULI_STOKES = (
    np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
)


def cloude_min_eigenvalues(m: np.ndarray, *, scale_floor: float) -> np.ndarray:
    """Return normalized minimum eigenvalue of the Mueller coherency matrix.

    A physically realizable Mueller matrix has a positive semidefinite Cloude
    coherency matrix.  We normalize by M11 to make the threshold independent of
    absolute scattering scale.
    """
    mins = []
    for k in range(m.shape[2]):
        scale = max(abs(float(m[0, 0, k])), scale_floor)
        h = np.zeros((4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                h += float(m[i, j, k]) * np.kron(PAULI_STOKES[i], PAULI_STOKES[j].conjugate())
        h *= 0.25 / scale
        h = 0.5 * (h + h.conjugate().T)
        mins.append(float(np.linalg.eigvalsh(h)[0]))
    return np.asarray(mins, dtype=float)


def audit_file(path: Path, *, m11_floor: float, negative_tol: float, physical_tol: float) -> dict:
    theta, m, meta = load_bem_mueller(path)
    finite = bool(np.isfinite(m).all())
    m11 = m[0, 0]
    m11_scale = max(float(np.nanmax(np.abs(m11))), 1e-300)
    min_m11 = float(np.nanmin(m11))
    negative_fraction = float(np.mean(m11 < -negative_tol * m11_scale))
    tiny_m11_fraction = float(np.mean(np.abs(m11) < m11_floor * m11_scale))

    # Physically necessary polarization bound for passive scattering:
    # |M_ij| <= M_11 is a useful numerical sanity check after normalization.
    bound_violations = {}
    denom = np.maximum(np.abs(m11), m11_floor * m11_scale)
    for name, (i, j, _, _) in COMPONENTS.items():
        if name == "S11":
            continue
        ratio = np.abs(m[i, j]) / denom
        bound_violations[name] = float(np.nanmax(ratio))

    polarizance = np.sqrt(m[1, 0] ** 2 + m[2, 0] ** 2 + m[3, 0] ** 2) / denom
    diattenuation = np.sqrt(m[0, 1] ** 2 + m[0, 2] ** 2 + m[0, 3] ** 2) / denom
    max_abs_over_m11 = max(bound_violations.values(), default=0.0)
    cloude_min = cloude_min_eigenvalues(m, scale_floor=m11_floor * m11_scale)
    cloude_fail = cloude_min < -physical_tol

    return {
        "path": str(path),
        "finite": finite,
        "theta_count": int(len(theta)),
        "min_m11": min_m11,
        "negative_m11_fraction": negative_fraction,
        "tiny_m11_fraction": tiny_m11_fraction,
        "max_abs_over_m11": bound_violations,
        "max_abs_over_m11_all": float(max_abs_over_m11),
        "max_polarizance": float(np.nanmax(polarizance)),
        "max_diattenuation": float(np.nanmax(diattenuation)),
        "min_cloude_eigenvalue": float(np.nanmin(cloude_min)),
        "negative_cloude_fraction": float(np.mean(cloude_fail)),
        "physical_tolerance": float(physical_tol),
        "timing": meta.get("timing", {}),
    }


def reference_errors(
    bem_path: Path,
    *,
    adda: Optional[Path],
    mbs: Optional[Path],
    beta_order: int,
    raw: bool,
) -> dict:
    theta, bem, _ = load_bem_mueller(bem_path)
    if adda is not None:
        ref, count = compare_mueller.load_adda_average(adda, beta_order)
        ref_kind = f"ADDA:{count}"
        col = lambda name: COMPONENTS[name][2]
    elif mbs is not None:
        ref = compare_mueller.load_mbs_table(mbs)
        ref_kind = "MBS"
        col = lambda name: COMPONENTS[name][3]
    else:
        return {}

    bem_s11 = bem[0, 0]
    ref_s11 = np.interp(theta, ref[:, 0], ref[:, col("S11")])
    bem_norm = 1.0 if raw else bem_s11[0]
    ref_norm = 1.0 if raw else ref_s11[0]
    out = {"reference": ref_kind, "raw": raw, "scale_bem_over_ref_s11_0": float(bem_s11[0] / ref_s11[0])}
    errs = {}
    for name, (i, j, _, _) in COMPONENTS.items():
        y = bem[i, j] / bem_norm
        r = np.interp(theta, ref[:, 0], ref[:, col(name)]) / ref_norm
        errs[name] = relative_l2(y, r)
    out["relative_l2"] = errs
    out["score_all"] = float(sum(errs.values()))
    return out


def self_test() -> None:
    tmp = {
        "theta": [0.0],
        "mueller": np.zeros((4, 4, 1)).tolist(),
    }
    # Ideal diagonal amplitude S1=S2=1 gives nonnegative intensity and unit
    # linear polarization block.  This is a structural sanity test for indexing.
    m = np.zeros((4, 4, 1))
    m[0, 0, 0] = 1.0
    m[1, 1, 0] = 1.0
    m[2, 2, 0] = 1.0
    m[3, 3, 0] = 1.0
    tmp["mueller"] = m.tolist()
    path = Path("/tmp/bemcuda_mueller_audit_selftest.json")
    path.write_text(json.dumps(tmp))
    audit = audit_file(path, m11_floor=1e-12, negative_tol=1e-12, physical_tol=1e-12)
    assert audit["finite"]
    assert audit["negative_m11_fraction"] == 0.0
    assert audit["max_polarizance"] == 0.0
    assert audit["max_diattenuation"] == 0.0
    assert audit["min_cloude_eigenvalue"] >= -1e-12

    bad = m.copy()
    bad[1, 0, 0] = 1.5
    tmp["mueller"] = bad.tolist()
    path.write_text(json.dumps(tmp))
    audit = audit_file(path, m11_floor=1e-12, negative_tol=1e-12, physical_tol=1e-12)
    assert audit["max_polarizance"] > 1.0
    assert audit["negative_cloude_fraction"] == 1.0

    bad = m.copy()
    bad[0, 1, 0] = 1.5
    tmp["mueller"] = bad.tolist()
    path.write_text(json.dumps(tmp))
    audit = audit_file(path, m11_floor=1e-12, negative_tol=1e-12, physical_tol=1e-12)
    assert audit["max_diattenuation"] > 1.0
    assert audit["negative_cloude_fraction"] == 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bem", type=Path, help="BEM JSON output")
    parser.add_argument("--adda", type=Path, help="ADDA mueller file or raw directory")
    parser.add_argument("--mbs", type=Path, help="MBS table")
    parser.add_argument("--beta-order", type=int, default=0)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--m11-floor", type=float, default=1e-12)
    parser.add_argument("--negative-tol", type=float, default=1e-10)
    parser.add_argument("--physical-tol", type=float, default=1e-8,
                        help="Tolerance for passive Mueller constraints")
    parser.add_argument("--max-abs-over-m11", type=float, default=1.000001,
                        help="Fail if any |Mij|/M11 exceeds this value")
    parser.add_argument("--require-cloude-physical", action="store_true",
                        help="Fail if the Cloude coherency matrix is not positive semidefinite")
    parser.add_argument("--max-l2", type=float, default=None,
                        help="Fail if any requested reference L2 error is above this")
    parser.add_argument("--elements", default="S11,S12,S22,S33,S34,S44")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("self-test: ok")
        return 0
    if args.bem is None:
        raise SystemExit("--bem is required unless --self-test is used")
    if args.adda is not None and args.mbs is not None:
        raise SystemExit("provide at most one of --adda and --mbs")

    report = audit_file(args.bem, m11_floor=args.m11_floor, negative_tol=args.negative_tol,
                        physical_tol=args.physical_tol)
    ref = reference_errors(args.bem, adda=args.adda, mbs=args.mbs,
                           beta_order=args.beta_order, raw=args.raw)
    if ref:
        names = parse_component_names(args.elements)
        ref["selected_score"] = float(sum(ref["relative_l2"][name] for name in names))
        report["reference_errors"] = ref

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["finite"] or report["negative_m11_fraction"] > 0.0:
        return 2
    if report["max_abs_over_m11_all"] > args.max_abs_over_m11:
        return 2
    if report["max_polarizance"] > 1.0 + args.physical_tol:
        return 2
    if report["max_diattenuation"] > 1.0 + args.physical_tol:
        return 2
    if args.require_cloude_physical and report["negative_cloude_fraction"] > 0.0:
        return 2
    if args.max_l2 is not None and ref:
        for name in parse_component_names(args.elements):
            if ref["relative_l2"][name] > args.max_l2:
                return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
