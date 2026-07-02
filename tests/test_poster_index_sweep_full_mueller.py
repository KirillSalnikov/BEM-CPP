#!/usr/bin/env python3
"""Poster index sweep must gate sphere/Mie rows by full Mueller accuracy."""

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verify_mie import mie_mueller  # noqa: E402
import poster_a0.make_assets as make_assets  # noqa: E402


def write_adda_mueller(path: Path, theta, mueller) -> None:
    names = [f"s{i}{j}" for i in range(1, 5) for j in range(1, 5)]
    lines = ["theta " + " ".join(names)]
    for t_idx, angle in enumerate(theta):
        values = [mueller[i][j][t_idx] for i in range(4) for j in range(4)]
        lines.append(f"{angle:.12g} " + " ".join(f"{value:.16e}" for value in values))
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    theta = [0.0, 30.0, 90.0, 150.0, 180.0]
    ka = 5.0
    n_re = 1.5
    mu = mie_mueller(theta, complex(n_re, 0.0), ka)
    norm = max(abs(mu[0][0][0]), 1.0)
    for t in range(len(theta)):
        mu[2][3][t] += 0.3 * norm

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        sweep = tmp_dir / "sweep"
        sweep.mkdir()
        adda = tmp_dir / "adda"
        adda_run = adda / "ka5_n1p5"
        adda_run.mkdir(parents=True)
        assets = tmp_dir / "assets"
        assets.mkdir()
        result = {
            "theta": theta,
            "mueller": mu,
            "ka": ka,
            "ri": [n_re, 0.0],
            "refinements": 3,
            "timing": {"total_s": 1.0, "solve_s": 0.5},
            "gmres_matvecs": 10,
            "gmres_nonconverged_systems": 0,
            "gmres_max_final_relres": 1e-4,
        }
        (sweep / "ka5_n1p5_ref3.json").write_text(json.dumps(result))
        write_adda_mueller(adda_run / "mueller", theta, mu)
        (adda / "summary.csv").write_text(
            "case,dir,status,ka,n,dpl,ntheta,time_s,note\n"
            f"ka5_n1p5,{adda_run},measured,{ka},{n_re},20,{len(theta)},2.0,\n"
        )

        old_out = make_assets.OUT
        make_assets.OUT = assets
        try:
            make_assets.make_index_sweep(sweep_dirs=[("strict", sweep)], adda_dir=adda)
        finally:
            make_assets.OUT = old_out

        table = pd.read_csv(assets / "table_index_sweep.csv")
        assert len(table) == 1, table
        row = table.iloc[0]
        assert row["status"] == "FAIL", row
        assert bool(row["pass10_shape_l2"]) is True, row
        assert bool(row["pass10_full_mueller"]) is False, row
        assert row["worst_component"] == "M34", row
        coverage = pd.read_csv(assets / "table_index_sweep_coverage.csv")
        cov = coverage[(coverage["ka"].eq(ka)) & (coverage["n"].eq(n_re))].iloc[0]
        assert cov["status"] == "PENDING", cov
        assert bool(cov["any_pass"]) is False, cov
        selected = pd.read_csv(assets / "table_index_sweep_selected.csv")
        assert selected.empty, selected
        adda_table = pd.read_csv(assets / "table_index_sweep_adda_ocl.csv")
        assert len(adda_table) == 1, adda_table
        adda_row = adda_table.iloc[0]
        assert adda_row["status"] == "FAIL", adda_row
        assert adda_row["worst_component"] == "M34", adda_row
        assert "M34" in str(adda_row["failed_all_20pct"]), adda_row
        assert float(adda_row["max_main_floor2"]) > 0.10, adda_row

    print("poster index sweep full mueller: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
