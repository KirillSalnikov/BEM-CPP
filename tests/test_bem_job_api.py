#!/usr/bin/env python3
"""Smoke tests for the Python BEM-CUDA job API."""

import json
from pathlib import Path
import sys
import tempfile
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bemcuda import BemJob, Geometry, Material, MeshQuality, OrientationGrid, SolverOptions
from bemcuda.job import write_orientation_file


def require_contains(cmd: List[str], *tokens: str) -> None:
    text = "\0".join(cmd)
    for token in tokens:
        assert token in cmd or token in text, f"missing {token!r} in {cmd!r}"


def without_out_value(cmd: List[str]) -> List[str]:
    idx = cmd.index("--out")
    return cmd[:idx] + ["--out", "<out>"] + cmd[idx + 2:]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        mesh_report = tmp / "mesh.json"
        out = tmp / "hex.json"
        job = BemJob(
            ka=5.0,
            shape="hex_prism",
            ref=2,
            prism_aspect=1.5,
            edge_refine=1,
            out=out,
            mesh_quality=MeshQuality(strict=True, report=mesh_report),
            asm_gpu_list=(2, 4),
            lu_gpu_list=(2, 4),
            ff_gpu_list=(4,),
            extra_args=("--adda-compare",),
        )
        cmd = job.command()
        require_contains(
            cmd,
            "--shape", "hex_prism",
            "--ref", "2",
            "--prism-aspect", "1.5",
            "--edge-refine", "1",
            "--mesh-quality-report", str(mesh_report),
            "--mesh-quality-strict",
            "--adda-compare",
            "--single",
        )
        assert "BEM_ASM_GPU_LIST" not in cmd
        assert "--system" not in cmd
        assert job.environment() == {
            "BEM_ASM_GPU_LIST": "2,4",
            "BEM_LU_GPU_LIST": "2,4",
            "BEM_FF_GPU_LIST": "4",
        }

        config = tmp / "job.json"
        job.to_json(config)
        payload = json.loads(config.read_text())
        assert payload["ka"] == 5.0
        assert payload["shape"] == "hex_prism"
        assert payload["system"] is None
        assert payload["mesh_quality"]["strict"] is True
        assert payload["mesh_quality"]["report"] == str(mesh_report)
        assert payload["asm_gpu_list"] == [2, 4]
        assert payload["lu_gpu_list"] == [2, 4]
        assert payload["ff_gpu_list"] == [4]

        orient_file = tmp / "orient.txt"
        write_orientation_file(orient_file, [(0, 0, 0, 1), (90, 45, 10, 2)])
        assert orient_file.read_text().count("\n") == 2
        avg_job = BemJob(
            ka=3.0,
            single=False,
            orient=(4, 5, 6),
            alpha_avg=8,
            out=tmp / "avg.json",
            mesh_quality=MeshQuality(strict=False),
        )
        avg_cmd = avg_job.command()
        require_contains(avg_cmd, "--orient", "4", "5", "6", "--alpha-avg", "8")
        assert "--single" not in avg_cmd
        assert "--mesh-quality-strict" not in avg_cmd

        file_job = BemJob(
            ka=3.0,
            single=False,
            orient_file=orient_file,
            out=tmp / "file.json",
            mesh_quality=MeshQuality(strict=False),
        )
        file_cmd = file_job.command()
        require_contains(file_cmd, "--orient-file", str(orient_file))
        assert "--orient" not in file_cmd

        obj_job = BemJob(
            ka=4.0,
            obj=tmp / "dust.obj",
            subdiv=2,
            out=tmp / "obj.json",
            no_prec=True,
        )
        obj_cmd = obj_job.command()
        require_contains(
            obj_cmd,
            "--obj", str(tmp / "dust.obj"),
            "--subdiv", "2",
            "--accurate",
            "--fmm-digits", "6",
            "--gmres-tol", "0.0005",
            "--gmres-restart", "500",
            "--no-prec",
        )
        assert "--shape" not in obj_cmd

        fast_obj_job = BemJob(
            ka=4.0,
            obj=tmp / "dust.obj",
            out=tmp / "fast_obj.json",
            extra_args=("--fast-obj",),
        )
        fast_obj_cmd = fast_obj_job.command()
        require_contains(fast_obj_cmd, "--fast-obj", "--fmm-digits", "5", "--gmres-tol", "0.001")
        assert "--accurate" not in fast_obj_cmd

        pmchwt_job = BemJob(ka=2.0, system="pmchwt", out=tmp / "pmchwt.json")
        require_contains(pmchwt_job.command(), "--system", "pmchwt")

        structured_job = BemJob(
            ka=12.0,
            material=Material(refractive_index=(1.6, 0.002)),
            geometry=Geometry(
                shape="hex_prism",
                ref=4,
                prism_aspect=1.25,
                edge_refine=2,
            ),
            orientations=OrientationGrid(
                single=False,
                counts=(1, 45, 90),
                alpha_avg=360,
                start=10,
                count=25,
            ),
            solver_options=SolverOptions(
                backend="fmm",
                system="muller2-balanced",
                quad=13,
                fmm_digits=6,
                gmres_tol=2e-4,
                gmres_restart=600,
                max_leaf=96,
                no_prec=True,
            ),
            ntheta=181,
            out=tmp / "structured.json",
            asm_gpu_list=(0,),
            extra_args=("--accurate",),
        )
        structured_cmd = structured_job.command()
        require_contains(
            structured_cmd,
            "--ri", "1.6", "0.002",
            "--shape", "hex_prism",
            "--ref", "4",
            "--prism-aspect", "1.25",
            "--edge-refine", "2",
            "--orient", "1", "45", "90",
            "--alpha-avg", "360",
            "--orient-start", "10",
            "--orient-count", "25",
            "--system", "muller2-balanced",
            "--quad", "13",
            "--fmm-digits", "6",
            "--gmres-tol", "0.0002",
            "--gmres-restart", "600",
            "--max-leaf", "96",
            "--no-prec",
        )
        manifest = structured_job.manifest()
        assert manifest["material"]["refractive_index"] == [1.6, 0.002]
        assert manifest["geometry"]["edge_refine"] == 2
        assert manifest["orientations"]["count"] == 25
        assert manifest["solver_options"]["system"] == "muller2-balanced"
        assert manifest["effective_solver_options"]["gmres_tol"] == 2e-4
        assert manifest["environment"] == {"BEM_ASM_GPU_LIST": "0"}
        assert len(manifest["semantic_id"]) == 16

        restored = BemJob.from_dict(manifest)
        assert restored.command() == structured_cmd
        assert restored.environment() == structured_job.environment()
        assert restored.semantic_id() == structured_job.semantic_id()
        moved_output = BemJob.from_dict(dict(manifest, out=str(tmp / "other.json")))
        assert moved_output.semantic_id() == structured_job.semantic_id()
        moved_report_payload = dict(manifest)
        moved_report_payload["mesh_quality"] = dict(
            manifest["mesh_quality"], report=str(tmp / "other_mesh.json")
        )
        assert BemJob.from_dict(moved_report_payload).semantic_id() == structured_job.semantic_id()
        changed_tol_payload = dict(manifest)
        changed_tol_payload["solver_options"] = dict(manifest["solver_options"], gmres_tol=1e-3)
        assert BemJob.from_dict(changed_tol_payload).semantic_id() != structured_job.semantic_id()

        obj_default = BemJob(ka=4.0, obj=tmp / "dust.obj", out=tmp / "obj_default.json")
        obj_explicit = BemJob(
            ka=4.0,
            obj=tmp / "dust.obj",
            out=tmp / "obj_explicit.json",
            solver_options=SolverOptions(fmm_digits=6, gmres_tol=5e-4, gmres_restart=500),
        )
        assert without_out_value(obj_default.command()) == without_out_value(obj_explicit.command())
        assert obj_default.semantic_id() == obj_explicit.semantic_id()
        assert obj_default.manifest()["effective_solver_options"]["fmm_digits"] == 6

    print("bem job api: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
