#!/usr/bin/env python3
"""Independent CPU PMCHWT reference for tiny meshes.

This is a diagnostic reference, not a production-accuracy solver.  It uses
one centroid point per RWG half support, so it is useful for checking block
signs, indexing, scaling, conditioning trends and dense/FMM regressions on
small meshes without touching CUDA.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from operator_block_audit import block_scales, form_general_system_numpy, form_pmchwt_numpy


def tetra_mesh():
    verts = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    tris = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
    ], dtype=int)
    return verts, tris


def load_obj(path: Path):
    verts, faces = [], []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("v "):
            _, x, y, z, *rest = line.split()
            verts.append((float(x), float(y), float(z)))
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            if len(idx) == 3:
                faces.append(tuple(idx))
            elif len(idx) > 3:
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def triangle_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def build_rwg(verts, tris):
    edges = {}
    for ti, tri in enumerate(tris):
        for e in range(3):
            u, v = int(tri[e]), int(tri[(e + 1) % 3])
            key = tuple(sorted((u, v)))
            free = int(tri[(e + 2) % 3])
            orient = 1 if (u, v) == key else -1
            edges.setdefault(key, []).append((ti, free, orient))

    out = []
    for (u, v), adj in sorted(edges.items()):
        if len(adj) != 2:
            continue
        (tp, fp, _), (tm, fm, _) = adj
        ap = triangle_area(*(verts[tris[tp]]))
        am = triangle_area(*(verts[tris[tm]]))
        out.append({
            "edge": (u, v),
            "tri_p": tp,
            "tri_m": tm,
            "free_p": fp,
            "free_m": fm,
            "length": float(np.linalg.norm(verts[u] - verts[v])),
            "area_p": ap,
            "area_m": am,
        })
    return out


def half_sample(rwg, verts, tris, sign):
    rows = []
    for item in rwg:
        ti = item["tri_p"] if sign > 0 else item["tri_m"]
        free = item["free_p"] if sign > 0 else item["free_m"]
        area = item["area_p"] if sign > 0 else item["area_m"]
        length = item["length"]
        tri = verts[tris[ti]]
        rc = tri.mean(axis=0)
        f = sign * length / (2.0 * area) * (rc - verts[free])
        div = sign * length / area
        rows.append((rc, f, area, div, ti))
    return rows


def assemble_lk_centroid(rwg, verts, tris, k):
    n = len(rwg)
    halves = [half_sample(rwg, verts, tris, +1), half_sample(rwg, verts, tris, -1)]
    L = np.zeros((n, n), dtype=np.complex128)
    K = np.zeros((n, n), dtype=np.complex128)
    inv4pi = 1.0 / (4.0 * np.pi)
    for sgn_t, test_half in enumerate(halves):
        test_sign = 1.0 if sgn_t == 0 else -1.0
        for sgn_s, source_half in enumerate(halves):
            source_sign = 1.0 if sgn_s == 0 else -1.0
            support_sign = test_sign * source_sign
            for i, (rt, ft, wt, divt, tit) in enumerate(test_half):
                for j, (rs, fs, ws, divs, tis) in enumerate(source_half):
                    dr = rt - rs
                    R = np.linalg.norm(dr)
                    if R < 1e-12:
                        # Centroid rule cannot resolve singular self terms.
                        continue
                    G = np.exp(1j * k * R) * inv4pi / R
                    jw = wt * ws
                    Lvec = np.dot(ft, fs) * G * jw
                    Lscl = G * jw
                    L[i, j] += support_sign * (1j * k * Lvec - (1j / k) * divt * divs * Lscl)
                    gradG = G * (1j * k - 1.0 / R) / R * dr
                    K[i, j] += support_sign * np.dot(np.cross(ft, gradG), fs) * jw
    return L, K


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path)
    parser.add_argument("--ka", type=float, default=1.0)
    parser.add_argument("--ri", nargs=2, type=float, default=(1.3116, 0.0))
    parser.add_argument("--eta-ext", type=complex, default=1.0 + 0j)
    parser.add_argument("--system", default="pmchwt",
                        choices=("pmchwt", "balanced", "muller", "muller-balanced", "muller2-balanced"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.obj:
        verts, tris = load_obj(args.obj)
    else:
        verts, tris = tetra_mesh()
    rwg = build_rwg(verts, tris)
    nrel = complex(args.ri[0], args.ri[1])
    k_ext = args.ka
    k_int = args.ka * nrel
    eta_ext = args.eta_ext
    eta_int = eta_ext / nrel
    le, ke = assemble_lk_centroid(rwg, verts, tris, k_ext)
    li, ki = assemble_lk_centroid(rwg, verts, tris, k_int)
    z = form_pmchwt_numpy(le, ke, li, ki, eta_ext, eta_int)
    s = np.linalg.svd(z, compute_uv=False)
    system_reports = {}
    for name in ("pmchwt", "balanced", "muller", "muller-balanced", "muller2-balanced"):
        experimental_nform = (name == "muller2-balanced")
        scales = block_scales(name, nrel, eta_int, experimental_nform)
        zg = form_general_system_numpy(le, ke, li, ki, eta_ext, eta_int, scales)
        sg = np.linalg.svd(zg, compute_uv=False)
        n = len(rwg)
        coupling_l2 = float(np.linalg.norm(zg[n:, :n] + scales["row_h_scale"] * scales["unknown_m_scale"] * zg[:n, n:]))
        system_reports[name] = {
            "matrix_shape": list(zg.shape),
            "condition_estimate": float(sg[0] / max(sg[-1], 1e-300)),
            "min_singular_value": float(sg[-1]),
            "max_singular_value": float(sg[0]),
            "unknown_m_scale": float(scales["unknown_m_scale"]),
            "row_h_scale": [float(np.real(scales["row_h_scale"])), float(np.imag(scales["row_h_scale"]))],
            "int_op_sign": float(scales["int_op_sign"]),
            "k_identity": float(scales["k_identity"]),
            "n_form": bool(scales["n_form"]),
            "coupling_antisymmetry_l2": coupling_l2,
        }
    selected = system_reports[args.system]
    report = {
        "vertices": int(len(verts)),
        "triangles": int(len(tris)),
        "rwg": int(len(rwg)),
        "matrix_shape": list(z.shape),
        "selected_system": args.system,
        "centroid_rule": True,
        "singular_terms_skipped": True,
        "condition_estimate": selected["condition_estimate"],
        "min_singular_value": selected["min_singular_value"],
        "max_singular_value": selected["max_singular_value"],
        "offdiag_antisymmetry_l2": float(np.linalg.norm(z[:len(rwg), len(rwg):] + z[len(rwg):, :len(rwg)])),
        "systems": system_reports,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
