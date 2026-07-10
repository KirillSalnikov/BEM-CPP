#!/usr/bin/env python3

import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "report_dust_adda_ka30_goal", ROOT / "scripts" / "report_dust_adda_ka30_goal.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main():
    material = {"directory": "refr_1_6__0", "ri": [1.6, 0.0]}
    case = {"ka": 30.25, "mesh": "dust.obj", "triangles": 100}
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        case_dir = run_root / material["directory"] / "ka30p25"
        level = case_dir / "level03_Jb4_Jg4"
        level.mkdir(parents=True)
        accepted = level / "bem.json"
        accepted.write_text("{}\n")
        (case_dir / "adaptive_nested_bg_manifest.json").write_text(json.dumps({
            "accepted": str(accepted), "converged": True, "accepted_level": 3,
            "accepted_active_count": 242,
        }))
        (case_dir / "level_m11_vs_adda.json").write_text(json.dumps({
            "levels": {level.name: {
                "bem": str(accepted), "angular_grid_resolves_reference": True,
                "metrics": {"m11_integral_rel_l2": 0.08, "total_s": 12.0},
            }}
        }))
        record = MODULE.case_record(run_root, material, case, 0.10)
        assert record["status"] == "pass", record
        assert record["passed"] is True
        assert record["m11_weighted_l2"] == 0.08

        missing = MODULE.case_record(run_root, material, dict(case, ka=33.28), 0.10)
        assert missing["status"] == "missing", missing
        assert missing["passed"] is False

    print("PASS: final goal report requires convergence, dense angles, and L2 gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
