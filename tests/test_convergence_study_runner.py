#!/usr/bin/env python3
"""Unit checks for the convergence-study matrix and command builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_convergence_study", ROOT / "scripts" / "run_convergence_study.py"
)
assert SPEC and SPEC.loader
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def main() -> int:
    config = {
        "defaults": {
            "ri": 1.3,
            "precision": "fp64",
            "quad": 13,
            "digits": 7,
        },
        "phases": [{
            "name": "mesh",
            "defaults": {"shape": "sphere", "physical_check": True},
            "sweeps": [{"ka": [2, 5], "ref": [1, 2], "repeats": 2}],
        }],
    }
    cases = STUDY.expand_config(config)
    assert len(cases) == 8
    assert cases[0]["cache_state_expected"] == "cold"
    assert cases[1]["cache_state_expected"] == "warm"
    assert cases[0]["base_id"] == cases[1]["base_id"]
    assert cases[2]["base_id"] != cases[0]["base_id"]
    duplicate_config = {
        "defaults": {"shape": "sphere", "ka": 2, "ref": 1},
        "phases": [{
            "name": "duplicate",
            "sweeps": [{"name": "first"}, {"name": "second"}],
        }],
    }
    with pytest.raises(ValueError, match="duplicate physical case"):
        STUDY.expand_config(duplicate_config)
    cache_id = STUDY.operator_cache_id(cases[0])
    assert STUDY.operator_cache_id({
        **cases[0], "phase": "memory_gates", "setup_only": True,
        "tolerance": 1.0e-2,
    }) == cache_id
    assert STUDY.operator_cache_id({
        **cases[0], "near_radius": 6,
    }) != cache_id
    assert STUDY.operator_cache_id({
        **cases[0], "cache_variant": "independent-cold-audit",
    }) != cache_id
    assert STUDY.operator_cache_id({
        **cases[0], "mbj_coarse_rank": 20,
    }) == cache_id
    selected, unmatched = STUDY.select_cases(cases, [cases[0]["base_id"]])
    assert len(selected) == 2
    assert not unmatched
    slug = STUDY.case_slug(cases[2])
    selected, unmatched = STUDY.select_cases(cases, [slug, "missing"])
    assert [STUDY.case_slug(case) for case in selected] == [slug, slug]
    assert unmatched == ["missing"]
    with tempfile.TemporaryDirectory() as temporary:
        case_dir = Path(temporary) / "case"
        cache_dir = Path(temporary) / "cache"
        command = STUDY.build_solver_command(cases[0], case_dir, cache_dir)
    assert option(command, "--shape") == "sphere"
    assert option(command, "--ka") == "2"
    assert option(command, "--ref") == "1"
    assert option(command, "--digits") == "7"
    assert option(command, "--mbj-coarse-rank") == "0"
    assert "--fmm-near-fp64" in command
    assert "--physical-check" in command
    assert "--checkpoint" in command
    assert "--setup-only" not in command

    setup = {**cases[0], "setup_only": True}
    with tempfile.TemporaryDirectory() as temporary:
        command = STUDY.build_solver_command(
            setup, Path(temporary) / "case", Path(temporary) / "cache"
        )
    assert "--setup-only" in command
    assert "--physical-check" not in command
    assert "--checkpoint" not in command

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source" / "checkpoint.MBJ.bin"
        source.parent.mkdir()
        source.write_bytes(b"checkpoint")
        destination = root / "destination"
        destination.mkdir()
        copied = STUDY.seed_solver_checkpoints(
            {"checkpoint_seed": str(source.parent / "checkpoint")},
            destination,
        )
        assert copied == ["checkpoint.MBJ.bin"]
        assert (destination / "checkpoint.MBJ.bin").read_bytes() == b"checkpoint"

    with tempfile.TemporaryDirectory() as temporary:
        runs_root = Path(temporary)
        gate = {**cases[0], "phase": "gate", "setup_only": True}
        case_directory = STUDY.case_run_directory(gate, runs_root)
        case_directory.mkdir(parents=True)
        (case_directory / "status.json").write_text(
            json.dumps({"state": "complete"}), encoding="utf-8"
        )
        (case_directory / "result.json").write_text(json.dumps({
            "shape": "sphere", "refinements": 1, "ka": 2.0, "ri": 1.3,
            "fmm_digits_requested": 7, "fmm_near_radius": 4,
            "system_dofs": 8, "mesh_vertices": 4, "mesh_triangles": 4,
            "surface_current_dofs": 4,
            "p2_nodes_per_shortest_wavelength": 2.0,
        }), encoding="utf-8")
        identity = STUDY.operator_cache_identity(gate)
        (case_directory / "case.json").write_text(
            json.dumps({"operator_cache_identity": identity}), encoding="utf-8"
        )
        assert not STUDY.completed_case(gate, runs_root)
        cache_directory = (
            runs_root / "_cache" / "operators" / STUDY.operator_cache_id(gate)
        )
        cache_directory.mkdir(parents=True)
        (cache_directory / "operator.near").write_bytes(b"near")
        (cache_directory / "mbj.cache").write_bytes(b"mbj")
        assert STUDY.completed_case(gate, runs_root)
        (case_directory / "case.json").write_text(json.dumps({
            "operator_cache_identity": {**identity, "binary_sha256": "stale"},
        }), encoding="utf-8")
        assert not STUDY.completed_case(gate, runs_root)

    with tempfile.TemporaryDirectory() as temporary:
        case_directory = Path(temporary) / "case"
        (case_directory / "profile").mkdir(parents=True)
        (case_directory / "profile" / "resources.json").write_text("{}")
        (case_directory / "status.json").write_text(json.dumps({
            "state": "interrupted",
        }))
        (case_directory / "case.json").write_text("{}")
        (case_directory / "iterations.csv").write_text("keep\n")
        archived = STUDY.archive_previous_attempt(case_directory)
        assert archived == case_directory / "attempts" / "attempt_000"
        assert (archived / "profile" / "resources.json").is_file()
        assert (archived / "status.json").is_file()
        assert not (case_directory / "profile").exists()
        assert (case_directory / "iterations.csv").read_text() == "keep\n"
    print("convergence study runner: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
