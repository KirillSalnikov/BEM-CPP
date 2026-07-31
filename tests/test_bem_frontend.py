#!/usr/bin/env python3
"""Regression tests for the user-facing bem launcher."""

from pathlib import Path
import json
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BEM = ROOT / "bem"


def invoke(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(BEM), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == expected, completed.stdout + completed.stderr
    return completed


def plan(*arguments: str) -> dict:
    completed = invoke(*arguments, "--dry-run", "--json")
    return json.loads(completed.stdout)


def synthetic_result(path: Path, scale: float = 1.0, residual: float = 5e-6) -> None:
    mueller = [
        [[scale * (1.0 if i == j else 0.01) for _ in range(2)] for j in range(4)]
        for i in range(4)
    ]
    path.write_text(
        json.dumps(
            {
                "software_version": "test",
                "solver": "test_muller",
                "tolerance": 1e-5,
                "mbj": {"fmm_residual": residual},
                "physical": {
                    "theta_degrees": [0.0, 180.0],
                    "mueller": mueller,
                },
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    profiles = json.loads(invoke("presets", "--json").stdout)
    assert set(profiles) == {"quick", "standard", "strict"}
    assert profiles["standard"]["tolerance"] == 1e-5
    assert profiles["strict"]["mixed_precision"] is False

    standard = plan(
        "run", "--shape", "prism", "--ka", "25", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-standard-plan",
    )
    assert standard["inputs"]["refinement"] == 5
    assert standard["inputs"]["sides"] == 6
    assert standard["quality"] == "standard"
    assert "--pfft-fgmres" in standard["command"]
    assert "--mbj-only" in standard["command"]
    assert "hdiv" in standard["command"]
    assert standard["effective_parameters"]["max_leaf"] == 32
    assert standard["effective_parameters"]["solver"] == "fmm_pfft_fgmres"

    small = plan(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-small-plan",
    )
    assert small["effective_parameters"]["max_leaf"] == 128
    assert small["effective_parameters"]["solver"] == "fmm_mbj"
    assert "--pfft-fgmres" not in small["command"]

    below_threshold = plan(
        "run", "--shape", "prism", "--ka", "9.99", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-threshold-plan",
    )
    assert below_threshold["effective_parameters"]["solver"] == "fmm_mbj"

    average = plan(
        "average", "--shape", "prism", "--ka", "25", "--ri", "1.3",
        "--quality", "quick", "--alpha", "256", "--beta", "4", "--gamma", "4",
        "--out", "/tmp/bem-frontend-average-plan",
    )
    orient = average["command"].index("--orient-average")
    assert average["command"][orient + 1:orient + 4] == ["256", "4", "4"]
    symmetry = average["command"].index("--orient-symmetry-order")
    assert average["command"][symmetry + 1] == "6"

    standard_average = plan(
        "average", "--shape", "prism", "--ka", "10", "--ri", "1.3",
        "--alpha", "8", "--beta", "4", "--gamma", "4",
        "--out", "/tmp/bem-frontend-standard-average-plan",
    )
    assert "--pfft-fgmres" in standard_average["command"]
    assert "--orient-paired-gpu-gmres" not in standard_average["command"]

    strict = plan(
        "run", "--shape", "sphere", "--ka", "1", "--ri", "1.3",
        "--quality", "strict", "--out", "/tmp/bem-frontend-strict-plan",
    )
    assert strict["kind"] == "strict_suite"
    assert [child["inputs"]["refinement"] for child in strict["children"]] == [2, 3]
    assert all("--fmm-near-fp64" in child["command"] for child in strict["children"])

    missing_obj = invoke(
        "run", "--shape", "obj", "--obj", "/dev/null", "--ka", "1", "--ri", "1.3",
        "--dry-run", expected=2,
    )
    assert "specify --ref" in missing_obj.stderr

    with tempfile.TemporaryDirectory(prefix="bem-frontend-test.") as directory:
        root = Path(directory)
        result = root / "result.json"
        reference = root / "reference.json"
        synthetic_result(result)
        synthetic_result(reference)
        report = json.loads(
            invoke("validate", str(result), "--reference", str(reference), "--json").stdout
        )
        assert report["comparison"]["passes"] is True
        synthetic_result(result, residual=3e-5)
        failed = invoke("validate", str(result), "--json", expected=1)
        assert "exceeds 2*tolerance" in failed.stdout

    print("bem frontend: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
