#!/usr/bin/env python3
"""Benchmark mass and Calderon-preconditioned Maxwell PMCHWT in Bempp-cl."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ka", required=True, type=float)
    parser.add_argument("--ri", nargs=2, type=float, default=(2.3, 0.0), metavar=("RE", "IM"))
    parser.add_argument("--method", choices=("mass", "calderon"), required=True)
    parser.add_argument("--polarization", choices=("x", "y", "both"), default="both")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--restart", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=80)
    parser.add_argument(
        "--quad",
        type=int,
        default=5,
        help="Bempp triangle rule order (order 5 has 7 points, matching BEM-CPP --quad 7)",
    )
    parser.add_argument("--fmm-order", type=int, default=5)
    parser.add_argument("--fmm-depth", type=int, default=4)
    parser.add_argument("--fmm-ncrit", type=int, default=400)
    return parser.parse_args()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def make_rhs(bempp, operator, k_ext: float, polarization: str):
    px = 1.0 if polarization == "x" else 0.0
    py = 1.0 if polarization == "y" else 0.0

    @bempp.complex_callable
    def tangential_trace(point, normal, domain_index, result):
        phase = np.exp(1j * k_ext * point[2])
        result[0] = py * normal[2] * phase
        result[1] = -px * normal[2] * phase
        result[2] = (px * normal[1] - py * normal[0]) * phase

    @bempp.complex_callable
    def neumann_trace(point, normal, domain_index, result):
        phase = np.exp(1j * k_ext * point[2])
        # (k_hat x E) x n for k_hat=(0,0,1).
        result[0] = px * normal[2] * phase
        result[1] = py * normal[2] * phase
        result[2] = -(px * normal[0] + py * normal[1]) * phase

    return [
        bempp.GridFunction(
            space=operator.range_spaces[0],
            dual_space=operator.dual_to_range_spaces[0],
            fun=tangential_trace,
        ),
        bempp.GridFunction(
            space=operator.range_spaces[1],
            dual_space=operator.dual_to_range_spaces[1],
            fun=neumann_trace,
        ),
    ]


def main() -> int:
    args = parse_args()
    started = time.perf_counter()

    import bempp_cl.api as bempp
    import exafmm
    from bempp_cl.api.assembly.blocked_operator import coefficients_from_grid_functions_list
    from bempp_cl.api.integration.triangle_gauss import get_number_of_quad_points
    from bempp_cl.api.operators.boundary.maxwell import multitrace_operator

    parameters = bempp.DefaultParameters()
    parameters.quadrature.regular = args.quad
    parameters.quadrature.singular = args.quad
    parameters.fmm.expansion_order = args.fmm_order
    parameters.fmm.depth = args.fmm_depth
    parameters.fmm.ncrit = args.fmm_ncrit
    # Bempp-cl 0.4.2 Maxwell FMM evaluators read these values from the global
    # parameter object even when operator construction receives a local one.
    bempp.GLOBAL_PARAMETERS.quadrature.regular = args.quad
    bempp.GLOBAL_PARAMETERS.quadrature.singular = args.quad
    bempp.GLOBAL_PARAMETERS.fmm.expansion_order = args.fmm_order
    bempp.GLOBAL_PARAMETERS.fmm.depth = args.fmm_depth
    bempp.GLOBAL_PARAMETERS.fmm.ncrit = args.fmm_ncrit

    m = complex(*args.ri)
    eps_r = m * m
    k_ext = args.ka
    k_int = k_ext * m

    print(f"Importing exact mesh: {args.mesh}", flush=True)
    grid = bempp.import_grid(str(args.mesh))
    mesh_seconds = time.perf_counter() - started
    print(
        f"Grid: {grid.number_of_vertices} vertices, {grid.number_of_elements} triangles",
        flush=True,
    )

    common = dict(
        parameters=parameters,
        assembler="fmm",
        precision="double",
    )
    assembly_started = time.perf_counter()
    print("Creating RWG/SNC PMCHWT operator...", flush=True)
    ext_rwg = multitrace_operator(grid, k_ext, space_type="all_rwg", **common)
    int_rwg = multitrace_operator(
        grid,
        k_int,
        epsilon_r=eps_r,
        mu_r=1.0,
        space_type="all_rwg",
        **common,
    )
    operator_rwg = ext_rwg + int_rwg
    discrete_rwg = operator_rwg.strong_form()

    discrete_bc = None
    if args.method == "calderon":
        print("Creating dual BC/RBC PMCHWT operator...", flush=True)
        ext_bc = multitrace_operator(grid, k_ext, space_type="all_bc", **common)
        int_bc = multitrace_operator(
            grid,
            k_int,
            epsilon_r=eps_r,
            mu_r=1.0,
            space_type="all_bc",
            **common,
        )
        discrete_bc = (ext_bc + int_bc).strong_form()
    assembly_seconds = time.perf_counter() - assembly_started

    dofs = sum(space.global_dof_count for space in operator_rwg.domain_spaces)
    result = {
        "status": "running",
        "implementation": "bempp-cl",
        "bempp_version": bempp.__version__,
        "exafmm_version": getattr(exafmm, "__version__", "0.1.1"),
        "python_version": platform.python_version(),
        "mesh": str(args.mesh.resolve()),
        "vertices": grid.number_of_vertices,
        "triangles": grid.number_of_elements,
        "system_size": dofs,
        "ka": args.ka,
        "ri": list(args.ri),
        "epsilon_r": [eps_r.real, eps_r.imag],
        "method": args.method,
        "formulation": "PMCHWT strong form" if args.method == "mass" else "dual-space strong Calderon PMCHWT",
        "spaces": "RWG/BC with SNC/RBC stable dual pairings",
        "tol": args.tol,
        "restart": args.restart,
        "max_cycles": args.max_cycles,
        "quadrature_order": args.quad,
        "quadrature_points_per_triangle": int(get_number_of_quad_points(args.quad)),
        "fmm_expansion_order": args.fmm_order,
        "fmm_depth": args.fmm_depth,
        "fmm_ncrit": args.fmm_ncrit,
        "mesh_seconds": mesh_seconds,
        "assembly_seconds": assembly_seconds,
        "polarizations": {},
    }
    write_json(args.out, result)

    requested = ("x", "y") if args.polarization == "both" else (args.polarization,)
    for polarization in requested:
        print(f"Building incident-field projections for E_{polarization}...", flush=True)
        rhs_functions = make_rhs(bempp, operator_rwg, k_ext, polarization)
        rhs = coefficients_from_grid_functions_list(rhs_functions)
        rhs_norm = float(np.linalg.norm(rhs))
        counts = {"rwg_actions": 0, "bc_actions": 0}

        def apply_rwg(vector):
            counts["rwg_actions"] += 1
            return discrete_rwg @ vector

        if args.method == "mass":
            solve_rhs = rhs

            def apply_system(vector):
                return apply_rwg(vector)

        else:
            assert discrete_bc is not None

            def apply_bc(vector):
                counts["bc_actions"] += 1
                return discrete_bc @ vector

            solve_rhs = apply_bc(rhs)

            def apply_system(vector):
                return apply_bc(apply_rwg(vector))

        linear_operator = LinearOperator(
            shape=(dofs, dofs),
            dtype=np.complex128,
            matvec=apply_system,
        )
        residual_history = []

        def callback(relative_residual):
            value = float(relative_residual)
            residual_history.append(value)
            count = len(residual_history)
            if count <= 5 or count % 10 == 0:
                print(f"  [{polarization} {count}] preconditioned residual={value:.6e}", flush=True)

        solve_started = time.perf_counter()
        solution, info = gmres(
            linear_operator,
            solve_rhs,
            rtol=args.tol,
            atol=0.0,
            restart=args.restart,
            maxiter=args.max_cycles,
            callback=callback,
            callback_type="pr_norm",
        )
        solve_seconds = time.perf_counter() - solve_started
        true_residual = float(np.linalg.norm(apply_rwg(solution) - rhs) / rhs_norm)
        entry = {
            "info": int(info),
            "converged_preconditioned_system": info == 0,
            "iterations": len(residual_history),
            "preconditioned_final_relres": residual_history[-1] if residual_history else None,
            "true_original_relres": true_residual,
            "solve_seconds": solve_seconds,
            "operator_actions": counts.copy(),
            "residual_history": residual_history,
        }
        result["polarizations"][polarization] = entry
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write_json(args.out, result)
        print(
            f"E_{polarization}: iterations={entry['iterations']}, "
            f"true residual={true_residual:.6e}, solve={solve_seconds:.2f}s",
            flush=True,
        )

    result["status"] = "complete"
    result["total_seconds"] = time.perf_counter() - started
    result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.out, result)
    print(f"Results written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
