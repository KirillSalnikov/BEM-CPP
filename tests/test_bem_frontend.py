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


def command_value(data: dict, option: str) -> str:
    index = data["command"].index(option)
    return data["command"][index + 1]


def synthetic_result(
    path: Path,
    scale: float = 1.0,
    residual: float = 5e-6,
    theta: list[float] | None = None,
) -> None:
    theta = theta or [0.0, 180.0]
    mueller = [
        [[scale * (1.0 if i == j else 0.01) for _ in theta] for j in range(4)]
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
                    "theta_degrees": theta,
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
    assert small["inputs"]["refinement"] == 2
    assert small["effective_parameters"]["solver"] == "fmm_mbj"
    assert "--pfft-fgmres" not in small["command"]

    larger = plan(
        "run", "--shape", "sphere", "--ka", "20", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-large-mesh-plan",
    )
    finer = plan(
        "run", "--shape", "sphere", "--ka", "20", "--ri", "1.3",
        "--points-per-wavelength", "16",
        "--out", "/tmp/bem-frontend-finer-mesh-plan",
    )
    assert larger["inputs"]["refinement"] == 5
    assert finer["inputs"]["refinement"] == 6
    assert larger["inputs"]["refinement_selection"] == "automatic"

    overrides = plan(
        "run", "--shape", "prism", "--ka", "25", "--ri", "1.3",
        "--ref", "3", "--solver", "fmm", "--tol", "2e-6",
        "--quad", "13", "--duffy-order", "7", "--digits", "6",
        "--ntheta", "91", "--max-iters", "777", "--gmres-restart", "64",
        "--max-leaf", "48", "--mbj-nodes", "72", "--mbj-overlap", "4",
        "--out", "/tmp/bem-frontend-overrides-plan",
    )
    assert overrides["inputs"]["refinement"] == 3
    assert overrides["inputs"]["refinement_selection"] == "explicit"
    assert overrides["effective_parameters"]["solver"] == "fmm_mbj"
    assert "--pfft-fgmres" not in overrides["command"]
    assert command_value(overrides, "--tol") == "2.0e-06"
    assert command_value(overrides, "--quad") == "13"
    assert command_value(overrides, "--digits") == "6"
    assert command_value(overrides, "--ntheta") == "91"
    assert command_value(overrides, "--mbj-nodes") == "72"
    assert command_value(overrides, "--mbj-cache").endswith("/mbj72.cache")
    for option in ("--tol", "--quad", "--digits", "--ntheta", "--mbj-nodes"):
        assert overrides["command"].count(option) == 1

    pfft_overrides = plan(
        "run", "--shape", "prism", "--ka", "25", "--ri", "1.3",
        "--solver", "pfft", "--pfft-inner-tol", "0.08",
        "--pfft-inner-iters", "12", "--pfft-outer-restart", "40",
        "--pfft-order", "3", "--pfft-correction-radius", "1.5",
        "--pfft-grid-safety", "0.9",
        "--out", "/tmp/bem-frontend-pfft-overrides-plan",
    )
    assert command_value(pfft_overrides, "--pfft-inner-tol") == "0.08"
    assert command_value(pfft_overrides, "--pfft-inner-iters") == "12"
    assert command_value(pfft_overrides, "--pfft-outer-restart") == "40"
    assert command_value(pfft_overrides, "--pfft-order") == "3"
    assert command_value(pfft_overrides, "--pfft-correction-radius") == "1.5"
    assert command_value(pfft_overrides, "--pfft-grid-safety") == "0.9"
    assert pfft_overrides["effective_parameters"]["pfft_order"] == 3

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
    assert "--orient-adaptive" not in average["command"]
    assert average["effective_parameters"]["orientation"]["mode"] == "fixed"

    adaptive_average = plan(
        "average", "--shape", "sphere", "--ka", "3", "--ri", "1.3",
        "--out", "/tmp/bem-frontend-adaptive-average-plan",
    )
    adaptive_index = adaptive_average["command"].index("--orient-adaptive")
    assert adaptive_average["command"][adaptive_index + 1:adaptive_index + 3] == [
        "2", "4"
    ]
    assert "--orient-parts-dir" in adaptive_average["command"]
    assert adaptive_average["effective_parameters"]["orientation"]["mode"] == "adaptive"
    assert adaptive_average["runtime"]["environment"] == {
        "BEM_MULLER_GPU_ASSEMBLY": "1"
    }

    adaptive_overrides = plan(
        "average", "--shape", "sphere", "--ka", "3", "--ri", "1.3",
        "--quality", "quick", "--adaptive-levels", "2", "5",
        "--adaptive-m11-tol", "0.004", "--adaptive-integral-tol", "0.005",
        "--adaptive-component-tol", "0.03", "--orient-warm-max-angle", "12",
        "--orient-recycle-rank", "3", "--orient-zero-start",
        "--out", "/tmp/bem-frontend-adaptive-overrides-plan",
    )
    adaptive = adaptive_overrides["effective_parameters"]["orientation"]
    assert adaptive["minimum_level"] == 2 and adaptive["maximum_level"] == 5
    assert adaptive["m11_tolerance"] == 0.004
    assert adaptive["integral_tolerance"] == 0.005
    assert adaptive["component_tolerance"] == 0.03
    assert adaptive["warm_start"] is False
    assert adaptive["warm_start_max_angle_degrees"] == 12
    assert adaptive["recycle_rank"] == 3

    conflict = invoke(
        "average", "--shape", "sphere", "--ka", "3", "--ri", "1.3",
        "--beta", "8", "--adaptive-levels", "1", "3", "--dry-run",
        expected=2,
    )
    assert "cannot be combined" in conflict.stderr

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
    assert [child["inputs"]["refinement_selection"] for child in strict["children"]] == [
        "automatic", "strict_fine"
    ]
    assert all("--fmm-near-fp64" in child["command"] for child in strict["children"])
    assert all("--pfft-fgmres" not in child["command"] for child in strict["children"])

    strict_normal = plan(
        "run", "--shape", "prism", "--ka", "10", "--ri", "1.5",
        "--quality", "strict", "--out", "/tmp/bem-frontend-strict-normal-plan",
    )
    assert all("--pfft-fgmres" in child["command"] for child in strict_normal["children"])
    assert all(
        child["effective_parameters"]["solver"] == "fmm_pfft_fgmres"
        for child in strict_normal["children"]
    )

    strict_average = plan(
        "average", "--shape", "prism", "--ka", "10", "--ri", "1.5",
        "--quality", "strict", "--out", "/tmp/bem-frontend-strict-average-plan",
    )
    for child in strict_average["children"]:
        adaptive_index = child["command"].index("--orient-adaptive")
        assert child["command"][adaptive_index + 1:adaptive_index + 3] == ["2", "5"]
        assert "--no-orient-paired-gpu-gmres" in child["command"]

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
        assert report["comparison"]["reference_interpolated"] is False
        synthetic_result(result, theta=[0.0, 90.0, 180.0])
        interpolated = json.loads(
            invoke("validate", str(result), "--reference", str(reference), "--json").stdout
        )
        assert interpolated["comparison"]["passes"] is True
        assert interpolated["comparison"]["reference_interpolated"] is True
        synthetic_result(result, residual=3e-5)
        failed = invoke("validate", str(result), "--json", expected=1)
        assert "exceeds 2*tolerance" in failed.stdout
        synthetic_result(result)
        document = json.loads(result.read_text(encoding="utf-8"))
        document["adaptive"] = {"enabled": True, "converged": False}
        result.write_text(json.dumps(document), encoding="utf-8")
        adaptive_failed = invoke("validate", str(result), "--json", expected=1)
        assert "without satisfying convergence" in adaptive_failed.stdout

    print("bem frontend: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
