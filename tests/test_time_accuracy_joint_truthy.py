from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "poster_a0"))

import make_assets  # noqa: E402


def csv_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def test_time_accuracy_joint_does_not_treat_false_strings_as_pass(tmp_path):
    old_out = make_assets.OUT
    make_assets.OUT = Path(tmp_path)
    try:
        pd.DataFrame([
            {
                "shape": "сфера",
                "ka": 5.0,
                "mesh_ref": 4,
                "mesh_label": "ref4",
                "bem_over_adda_ocl": 0.5,
                "gmres_convergence_status": "converged",
                "gmres_ok": "True",
                "gmres_nonconverged_systems": 0,
                "gmres_stagnation_stops": 0,
                "gmres_max_final_relres": 1e-4,
                "source": "runs/example/sphere_ka5_ref4.json",
            }
        ]).to_csv(tmp_path / "table_bem_vs_adda_ocl_same_shape.csv", index=False)

        pd.DataFrame([
            {
                "shape": "сфера",
                "ka": 5.0,
                "mesh_ref": 4,
                "mesh_label": "ref4",
                "M11": 0.01,
                "M12": 0.01,
                "M34": 0.01,
                "score16": 0.01,
                "max_pol15": 0.01,
                "full16_pass_10pct": "True",
                "pol15_pass_20pct": "False",
                "mean_pol15_floor2": 0.01,
                "max_pol15_floor2": 0.01,
                "mean_main_floor2": 0.01,
                "max_main_floor2": 0.01,
                "full16_floor2_pass_20pct": "True",
                "pol15_floor2_pass_20pct": "False",
                "main_floor2_pass_5pct": "True",
                "full16_floor2_pass_5pct": "True",
                "pol15_floor2_pass_5pct": "False",
            }
        ]).to_csv(tmp_path / "table_adda_ocl_accuracy_summary.csv", index=False)

        make_assets.make_time_accuracy_joint()

        joint = pd.read_csv(tmp_path / "table_time_accuracy_joint.csv")
        assert len(joint) == 1
        row = joint.iloc[0]
        assert csv_bool(row["bem_faster"])
        assert csv_bool(row["main_floor2_pass_5pct"])
        assert not csv_bool(row["pol15_floor2_pass_5pct"])
        assert not csv_bool(row["pol15_floor2_pass_20pct"])
        assert not csv_bool(row["fast_and_m11_10_pol15_floor2_20pct"])
        assert not csv_bool(row["fast_and_accuracy_5pct"])
    finally:
        make_assets.OUT = old_out


if __name__ == "__main__":
    with TemporaryDirectory() as tmp:
        test_time_accuracy_joint_does_not_treat_false_strings_as_pass(Path(tmp))
