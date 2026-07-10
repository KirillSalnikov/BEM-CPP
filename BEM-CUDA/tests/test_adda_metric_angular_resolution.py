#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_bem_adda_m11", ROOT / "scripts" / "summarize_bem_adda_m11.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coarse_grid_is_not_final_quality():
    adda = np.r_[np.arange(0.0, 2.0, 0.1), np.arange(2.0, 180.0, 4.0), 180.0]
    metrics = MODULE.angular_grid_metrics(np.linspace(0.0, 180.0, 181), adda)
    assert metrics["angular_grid_resolves_reference"] == 0.0


def test_dense_grid_resolves_adaptive_adda_grid():
    adda = np.r_[np.arange(0.0, 2.0, 0.1), np.arange(2.0, 180.0, 4.0), 180.0]
    metrics = MODULE.angular_grid_metrics(np.linspace(0.0, 180.0, 1801), adda)
    assert metrics["angular_grid_matches_reference"] == 0.0
    assert metrics["angular_grid_resolves_reference"] == 1.0


if __name__ == "__main__":
    test_coarse_grid_is_not_final_quality()
    test_dense_grid_resolves_adaptive_adda_grid()
    print("PASS: ADDA angular-resolution quality gate")
