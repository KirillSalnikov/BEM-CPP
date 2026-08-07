#!/usr/bin/env python3
"""Run a resumable, profiled BEM convergence study from a JSON matrix."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "studies" / "bem_convergence_20260805" / "study_config.json"
NON_FACTOR_KEYS = {"extra_args", "tags", "repeats", "name"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def archive_previous_attempt(case_directory: Path) -> Path | None:
    """Preserve profiling/provenance before an incomplete case is resumed."""
    status_path = case_directory / "status.json"
    profile_path = case_directory / "profile"
    if not status_path.is_file() and not profile_path.exists():
        return None
    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
    if status.get("state") == "complete":
        return None
    attempts = case_directory / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    index = 0
    while (attempts / f"attempt_{index:03d}").exists():
        index += 1
    destination = attempts / f"attempt_{index:03d}"
    destination.mkdir()
    for name in ("profile", "case.json", "status.json", "result.json"):
        source = case_directory / name
        if source.exists():
            shutil.move(str(source), str(destination / name))
    return destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def expand_sweep(sweep: dict[str, Any]) -> Iterator[dict[str, Any]]:
    factor_names: list[str] = []
    factor_values: list[list[Any]] = []
    fixed: dict[str, Any] = {}
    for key, value in sweep.items():
        if key in NON_FACTOR_KEYS or not isinstance(value, list):
            fixed[key] = value
        else:
            if not value:
                raise ValueError(f"empty factor {key!r}")
            factor_names.append(key)
            factor_values.append(value)
    products = itertools.product(*factor_values) if factor_names else [()]
    for product in products:
        yield {**fixed, **dict(zip(factor_names, product))}


def expand_config(config: dict[str, Any], selected: set[str] | None = None) -> list[dict[str, Any]]:
    global_defaults = config.get("defaults", {})
    cases: list[dict[str, Any]] = []
    for phase in config.get("phases", []):
        phase_name = phase["name"]
        if selected and phase_name not in selected:
            continue
        if not phase.get("enabled", True):
            continue
        phase_defaults = phase.get("defaults", {})
        for sweep in phase.get("sweeps", []):
            for expanded in expand_sweep(sweep):
                merged = {
                    **global_defaults,
                    **phase_defaults,
                    **expanded,
                    "phase": phase_name,
                    "phase_description": phase.get("description", ""),
                }
                repeats = int(merged.pop("repeats", 1))
                if repeats < 1:
                    raise ValueError(f"{phase_name}: repeats must be positive")
                base_identity = {
                    key: value for key, value in merged.items()
                    if key not in {"phase_description", "name"}
                }
                base_id = hashlib.sha256(
                    json.dumps(
                        base_identity, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()[:12]
                for repeat in range(repeats):
                    cases.append({
                        **merged,
                        "base_id": base_id,
                        "repeat": repeat,
                        "repeat_count": repeats,
                        "cache_state_expected": "cold" if repeat == 0 else "warm",
                    })
    identities: dict[tuple[str, str, int], str] = {}
    for case in cases:
        key = (case["phase"], case["base_id"], case["repeat"])
        label = str(case.get("name") or case_slug(case))
        previous = identities.get(key)
        if previous is not None:
            raise ValueError(
                "duplicate physical case in study matrix: "
                f"phase={case['phase']} base_id={case['base_id']} "
                f"repeat={case['repeat']} labels={previous!r},{label!r}"
            )
        identities[key] = label
    return cases


def number_slug(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p").replace("+", "")


def case_slug(case: dict[str, Any]) -> str:
    explicit = case.get("name")
    if explicit:
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(explicit)).strip("_")
    else:
        prefix = "_".join((
            str(case["shape"]),
            f"ka{number_slug(case['ka'])}",
            f"m{number_slug(case['ri'])}",
            f"r{case['ref']}",
        ))
    return f"{prefix}_{case['base_id']}"


def case_run_directory(case: dict[str, Any], runs_root: Path) -> Path:
    return (
        runs_root / case["phase"] / case_slug(case)
        / f"repeat_{case['repeat']:02d}"
    )


def completed_case(case: dict[str, Any], runs_root: Path) -> bool:
    directory = case_run_directory(case, runs_root)
    status_path = directory / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            return False
        valid, _ = validate_result(case, directory / "result.json")
        if not valid:
            return False
        manifest_path = directory / "case.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        stored_identity = manifest.get("operator_cache_identity")
        if stored_identity and stored_identity != operator_cache_identity(case):
            return False
        if case.get("setup_only", False):
            cache_directory = (
                runs_root / "_cache" / "operators" / operator_cache_id(case)
            )
            if not all(
                (cache_directory / name).is_file()
                for name in ("operator.near", "mbj.cache")
            ):
                return False
        return True
    except (OSError, json.JSONDecodeError):
        return False


def select_cases(
    cases: list[dict[str, Any]], selectors: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select exact case slugs or base IDs while preserving matrix order."""
    if not selectors:
        return cases, []
    requested = set(selectors)
    matched: set[str] = set()
    selected: list[dict[str, Any]] = []
    for case in cases:
        identities = {str(case["base_id"]), case_slug(case)}
        if identities & requested:
            selected.append(case)
            matched.update(identities & requested)
    return selected, sorted(requested - matched)


def binary_path(case: dict[str, Any]) -> Path:
    precision = case.get("precision", "fp64")
    if precision == "fp64":
        default = ROOT / "bin" / "muller_nodal_fmm_demo"
        configured = case.get("binary_fp64")
    elif precision in {"mixed", "fp32-near"}:
        default = ROOT / "bin" / "muller_nodal_fmm_demo_fp32"
        configured = case.get("binary_mixed")
    else:
        raise ValueError(f"unsupported precision {precision!r}")
    path = Path(configured).expanduser() if configured else default
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def operator_cache_identity(case: dict[str, Any]) -> dict[str, Any]:
    binary = binary_path(case)
    identity: dict[str, Any] = {
        "schema_version": 1,
        "binary_sha256": file_sha256(binary) if binary.is_file() else None,
        "shape": case["shape"],
        "ref": int(case["ref"]),
        "ka": float(case["ka"]),
        "ri": float(case["ri"]),
        "edge_mode": case.get("edge_mode", "smooth"),
        "quad": int(case.get("quad", 13)),
        "duffy_order": int(case.get("duffy_order", 6)),
        "digits": int(case.get("digits", 7)),
        "digits_cap": int(case.get("digits_cap", 7)),
        "max_leaf": int(case.get("max_leaf", 64)),
        "near_radius": int(case.get("near_radius", 4)),
        "precision": case.get("precision", "fp64"),
        "near_precision": case.get("near_precision"),
        "mbj_nodes": int(case.get("mbj_nodes", 50)),
        "mbj_overlap": int(case.get("mbj_overlap", 0)),
        "cache_variant": case.get("cache_variant"),
        "extra_args": list(case.get("extra_args", [])),
    }
    if case["shape"] in {"prism", "cube"}:
        identity.update({
            "sides": int(case.get("sides", 6)),
            "aspect": float(case.get("aspect", 1.0)),
        })
    elif case["shape"] == "obj":
        obj = Path(case["obj"]).expanduser()
        if not obj.is_absolute():
            obj = ROOT / obj
        obj = obj.resolve()
        identity.update({
            "obj_sha256": file_sha256(obj) if obj.is_file() else None,
        })
    return identity


def operator_cache_id(case: dict[str, Any]) -> str:
    encoded = json.dumps(
        operator_cache_identity(case), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_solver_command(
    case: dict[str, Any], case_directory: Path, cache_directory: Path
) -> list[str]:
    command = [
        str(binary_path(case)),
        "--shape", str(case["shape"]),
        "--ref", str(case["ref"]),
        "--ka", str(case["ka"]),
        "--ri", str(case["ri"]),
        "--edge-mode", str(case.get("edge_mode", "smooth")),
        "--quad", str(case.get("quad", 13)),
        "--duffy-order", str(case.get("duffy_order", 6)),
        "--digits", str(case.get("digits", 7)),
        "--max-leaf", str(case.get("max_leaf", 64)),
        "--fmm-near-radius", str(case.get("near_radius", 4)),
        "--tol", f"{float(case.get('tolerance', 1.0e-7)):.17g}",
        "--max-iters", str(case.get("max_iterations", 2000)),
        "--gmres-restart", str(case.get("gmres_restart", 100)),
        "--mbj-only",
        "--mbj-nodes", str(case.get("mbj_nodes", 50)),
        "--mbj-overlap", str(case.get("mbj_overlap", 0)),
        "--mbj-coarse-rank", str(case.get("mbj_coarse_rank", 0)),
        "--no-dense-validation",
        "--near-correction-cache", str(cache_directory / "operator.near"),
        "--mbj-cache", str(cache_directory / "mbj.cache"),
        "--out", str(case_directory / "result.json"),
    ]
    shape = case["shape"]
    if shape == "prism":
        command += [
            "--sides", str(case.get("sides", 6)),
            "--aspect", str(case.get("aspect", 1.0)),
        ]
    elif shape == "obj":
        obj = Path(case["obj"]).expanduser()
        if not obj.is_absolute():
            obj = ROOT / obj
        command += ["--obj", str(obj.resolve())]
    if case.get("precision", "fp64") == "fp64" or case.get("near_precision") == "fp64":
        command.append("--fmm-near-fp64")
    else:
        command.append("--fmm-near-fp32")
    if case.get("setup_only", False):
        command.append("--setup-only")
    else:
        command += [
            "--iteration-log", str(case_directory / "iterations.csv"),
            "--checkpoint", str(case_directory / "checkpoint"),
        ]
        if case.get("physical_check", True):
            command += [
                "--physical-check",
                "--ntheta", str(case.get("ntheta", 181)),
            ]
    command.extend(str(value) for value in case.get("extra_args", []))
    return command


def seed_solver_checkpoints(
    case: dict[str, Any], case_directory: Path
) -> list[str]:
    configured = case.get("checkpoint_seed")
    if not configured:
        return []
    source_base = Path(str(configured)).expanduser()
    if not source_base.is_absolute():
        source_base = (ROOT / source_base).resolve()
    target_base = case_directory / "checkpoint"
    copied: list[str] = []
    for source in sorted(source_base.parent.glob(source_base.name + ".*.bin")):
        if not source.is_file() or source.stat().st_size == 0:
            continue
        suffix = source.name[len(source_base.name):]
        target = Path(str(target_base) + suffix)
        if target.exists():
            continue
        shutil.copy2(source, target)
        copied.append(target.name)
    if not copied:
        raise FileNotFoundError(
            f"checkpoint seed has no usable stage files: {source_base}"
        )
    return copied


def validate_result(case: dict[str, Any], path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, [f"cannot read result JSON: {error}"]
    checks = {
        "shape": case["shape"],
        "refinements": int(case["ref"]),
        "fmm_digits_requested": int(case.get("digits", 7)),
        "fmm_near_radius": int(case.get("near_radius", 4)),
    }
    for key, expected in checks.items():
        if data.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {data.get(key)!r}")
    if abs(float(data.get("ka", float("nan"))) - float(case["ka"])) > 1.0e-12:
        errors.append("ka mismatch")
    if abs(float(data.get("ri", float("nan"))) - float(case["ri"])) > 1.0e-12:
        errors.append("ri mismatch")
    if int(data.get("system_dofs", 0)) <= 0:
        errors.append("system_dofs must be positive")
    for key in (
        "mesh_vertices", "mesh_triangles", "surface_current_dofs",
        "p2_nodes_per_shortest_wavelength",
    ):
        if key not in data:
            errors.append(f"missing {key}")
    if not case.get("setup_only", False):
        mbj = data.get("mbj") or {}
        if int(mbj.get("coarse_rank", 0)) != int(
            case.get("mbj_coarse_rank", 0)
        ):
            errors.append("mbj.coarse_rank mismatch")
        residual = mbj.get("fmm_residual")
        if residual is None or not isinstance(residual, (int, float)):
            errors.append("missing mbj.fmm_residual")
        elif residual > 5.0 * float(case.get("tolerance", 1.0e-7)):
            errors.append(
                f"verified residual {residual:.3e} exceeds 5*tolerance"
            )
        if case.get("physical_check", True):
            physical = data.get("physical")
            if not isinstance(physical, dict):
                errors.append("missing physical result")
            elif len(physical.get("theta_degrees", [])) != int(case.get("ntheta", 181)):
                errors.append("unexpected theta grid")
    return not errors, errors


@contextmanager
def gpu_lock(index: int) -> Iterator[None]:
    path = Path(f"/tmp/bem-cpp-convergence-gpu{index}.lock")
    with path.open("w", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"GPU {index} is already reserved by another study") from error
        stream.write(f"pid={os.getpid()} started={utc_now()}\n")
        stream.flush()
        yield


@contextmanager
def cache_lock(cache_directory: Path) -> Iterator[None]:
    path = cache_directory / ".lock"
    with path.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(f"pid={os.getpid()} started={utc_now()}\n")
        stream.flush()
        yield


def run_case(
    case: dict[str, Any], runs_root: Path, config_path: Path,
    interval: float, gpu: int, dry_run: bool,
) -> str:
    slug = case_slug(case)
    case_directory = case_run_directory(case, runs_root)
    cache_identity = operator_cache_identity(case)
    cache_directory = (
        runs_root / "_cache" / "operators" / operator_cache_id(case)
    )
    case_directory.mkdir(parents=True, exist_ok=True)
    cache_directory.mkdir(parents=True, exist_ok=True)
    status_path = case_directory / "status.json"
    if completed_case(case, runs_root):
        print(f"SKIP complete {case['phase']}/{slug} repeat={case['repeat']}")
        return "skipped"
    command = build_solver_command(case, case_directory, cache_directory)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "case": case,
        "command": command,
        "command_shell": shlex.join(command),
        "case_directory": str(case_directory),
        "cache_directory": str(cache_directory),
        "operator_cache_identity": cache_identity,
    }
    if dry_run:
        print(f"DRY {case['phase']}/{slug} repeat={case['repeat']}")
        print("  " + shlex.join(command))
        return "dry"
    archived = archive_previous_attempt(case_directory)
    if archived is not None:
        print(f"ARCHIVE previous attempt: {archived}")
    seeded_checkpoints = seed_solver_checkpoints(case, case_directory)
    if seeded_checkpoints:
        manifest["seeded_checkpoints"] = seeded_checkpoints
        print("SEED checkpoints: " + ",".join(seeded_checkpoints))
    atomic_json(case_directory / "case.json", manifest)
    binary = binary_path(case)
    if not binary.is_file():
        raise FileNotFoundError(f"solver binary not found: {binary}")
    atomic_json(status_path, {"state": "running", "started_at_utc": utc_now()})
    profiler = [
        sys.executable, str(ROOT / "scripts" / "profile_command.py"),
        "--output", str(case_directory / "profile"),
        "--cwd", str(ROOT),
        "--interval", str(interval),
        "--gpu", str(gpu),
        "--env", f"OMP_NUM_THREADS={case.get('threads', 16)}",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu}",
        "--env", f"BEM_MULLER_FMM_DIGITS_CAP={case.get('digits_cap', 7)}",
        "--",
        *command,
    ]
    print(f"RUN {case['phase']}/{slug} repeat={case['repeat']}")
    try:
        with cache_lock(cache_directory):
            completed = subprocess.run(profiler, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        started_at = json.loads(
            status_path.read_text(encoding="utf-8")
        )["started_at_utc"]
        checkpoints = sorted(
            path.name for path in case_directory.glob("checkpoint*.bin")
            if path.is_file() and path.stat().st_size > 0
        )
        atomic_json(status_path, {
            "state": "interrupted",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "return_code": 130,
            "checkpoints": checkpoints,
        })
        print(
            f"INTERRUPTED {case['phase']}/{slug}; "
            f"checkpoints={','.join(checkpoints) or 'none'}"
        )
        return "interrupted"
    valid, errors = validate_result(case, case_directory / "result.json")
    state = "complete" if completed.returncode == 0 and valid else "failed"
    atomic_json(status_path, {
        "state": state,
        "started_at_utc": json.loads(status_path.read_text())["started_at_utc"],
        "finished_at_utc": utc_now(),
        "return_code": completed.returncode,
        "validation_errors": errors,
    })
    if state != "complete":
        print(f"FAIL {case['phase']}/{slug}: {'; '.join(errors) or completed.returncode}")
        return "failed"
    print(f"DONE {case['phase']}/{slug}")
    return "complete"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument(
        "--case", action="append", default=[], metavar="SLUG_OR_ID",
        help="run an exact case slug or base ID shown by --list; repeatable",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    all_cases = expand_config(config)
    selected = set(args.phase) or None
    cases = expand_config(config, selected)
    cases, unmatched = select_cases(cases, args.case)
    if unmatched:
        parser.error("unknown --case selector(s): " + ", ".join(unmatched))
    if args.max_cases is not None:
        if args.max_cases < 0:
            parser.error("--max-cases must be non-negative")
        cases = cases[:args.max_cases]
    if args.list:
        for index, case in enumerate(cases, start=1):
            print(
                f"{index:4d} {case['phase']:24s} {case_slug(case)} "
                f"repeat={case['repeat']}"
            )
        print(f"cases: {len(cases)}")
        return 0
    runs_root = (
        args.runs_root.expanduser().resolve()
        if args.runs_root
        else ROOT / config.get("runs_root", "runs/convergence_study_20260805")
    )
    memory_gates: dict[str, list[dict[str, Any]]] = {}
    for candidate in all_cases:
        if candidate.get("setup_only", False):
            memory_gates.setdefault(operator_cache_id(candidate), []).append(candidate)
    summary = {
        "complete": 0, "skipped": 0, "failed": 0, "blocked": 0,
        "interrupted": 0, "dry": 0,
    }
    with gpu_lock(args.gpu):
        for case in cases:
            gates = memory_gates.get(operator_cache_id(case), [])
            if (
                gates
                and not case.get("setup_only", False)
                and not args.dry_run
                and not completed_case(case, runs_root)
                and not any(completed_case(gate, runs_root) for gate in gates)
            ):
                print(
                    f"BLOCK {case['phase']}/{case_slug(case)}: "
                    "matching setup-only memory gate is not complete"
                )
                summary["blocked"] += 1
                continue
            outcome = run_case(
                case, runs_root, config_path, args.interval, args.gpu, args.dry_run
            )
            summary[outcome] += 1
            if outcome == "interrupted":
                break
            if outcome == "failed" and args.fail_fast:
                break
    print("summary: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
    if summary["interrupted"]:
        return 130
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
