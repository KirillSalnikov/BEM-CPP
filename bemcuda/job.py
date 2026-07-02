#!/usr/bin/env python3
"""Command builder for BEM-CUDA production runs.

This module keeps run parameters explicit and serializable.  It deliberately
has no dependency beyond the Python standard library so it can run on the old
Python interpreter installed on the V100 server.
"""

import json
import hashlib
import os
from pathlib import Path
import subprocess


def _repo_root():
    return Path(__file__).resolve().parents[1]


class MeshQuality:
    """Mesh gate thresholds used before a solve."""

    def __init__(self, strict=True, report=None):
        self.strict = strict
        self.report = report

    def to_dict(self):
        return {
            "strict": self.strict,
            "report": None if self.report is None else str(self.report),
        }


class Material:
    """Optical material parameters passed to the executable."""

    def __init__(self, refractive_index=(1.3116, 0.0)):
        self.refractive_index = tuple(refractive_index)

    def to_dict(self):
        return {"refractive_index": list(self.refractive_index)}

    @classmethod
    def from_dict(cls, payload):
        return cls(refractive_index=payload.get("refractive_index", (1.3116, 0.0)))


class Geometry:
    """Shape or OBJ mesh description."""

    def __init__(
        self,
        shape="sphere",
        ref=3,
        obj=None,
        subdiv=0,
        prism_aspect=1.0,
        edge_refine=None,
    ):
        self.shape = shape
        self.ref = ref
        self.obj = obj
        self.subdiv = subdiv
        self.prism_aspect = prism_aspect
        self.edge_refine = edge_refine

    def to_dict(self):
        return {
            "shape": self.shape,
            "ref": self.ref,
            "obj": None if self.obj is None else str(self.obj),
            "subdiv": self.subdiv,
            "prism_aspect": self.prism_aspect,
            "edge_refine": self.edge_refine,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            shape=payload.get("shape", "sphere"),
            ref=payload.get("ref", 3),
            obj=payload.get("obj"),
            subdiv=payload.get("subdiv", 0),
            prism_aspect=payload.get("prism_aspect", 1.0),
            edge_refine=payload.get("edge_refine"),
        )


class OrientationGrid:
    """Single orientation, tensor grid, or explicit orientation file."""

    def __init__(
        self,
        single=True,
        counts=(1, 1, 1),
        file=None,
        alpha_avg=1,
        start=None,
        count=None,
    ):
        self.single = single
        self.counts = tuple(counts)
        self.file = file
        self.alpha_avg = alpha_avg
        self.start = start
        self.count = count

    def to_dict(self):
        return {
            "single": self.single,
            "counts": list(self.counts),
            "file": None if self.file is None else str(self.file),
            "alpha_avg": self.alpha_avg,
            "start": self.start,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            single=payload.get("single", True),
            counts=payload.get("counts", (1, 1, 1)),
            file=payload.get("file"),
            alpha_avg=payload.get("alpha_avg", 1),
            start=payload.get("start"),
            count=payload.get("count"),
        )


class SolverOptions:
    """Numerical solver options that affect accuracy or convergence."""

    def __init__(
        self,
        backend="fmm",
        system=None,
        quad=7,
        fmm_digits=5,
        gmres_tol=1e-3,
        gmres_restart=200,
        max_leaf=128,
        no_prec=False,
    ):
        self.backend = backend
        self.system = system
        self.quad = quad
        self.fmm_digits = fmm_digits
        self.gmres_tol = gmres_tol
        self.gmres_restart = gmres_restart
        self.max_leaf = max_leaf
        self.no_prec = no_prec

    def to_dict(self):
        return {
            "backend": self.backend,
            "system": self.system,
            "quad": self.quad,
            "fmm_digits": self.fmm_digits,
            "gmres_tol": self.gmres_tol,
            "gmres_restart": self.gmres_restart,
            "max_leaf": self.max_leaf,
            "no_prec": self.no_prec,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            backend=payload.get("backend", "fmm"),
            system=payload.get("system"),
            quad=payload.get("quad", 7),
            fmm_digits=payload.get("fmm_digits", 5),
            gmres_tol=payload.get("gmres_tol", 1e-3),
            gmres_restart=payload.get("gmres_restart", 200),
            max_leaf=payload.get("max_leaf", 128),
            no_prec=payload.get("no_prec", False),
        )


class RunResult:
    """Completed external process plus parsed JSON payload when available."""

    def __init__(self, command, returncode, stdout, stderr, output, data=None):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.output = output
        self.data = data


class BemJob:
    """One BEM-CUDA run with parameters matching the executable CLI."""

    def __init__(
        self,
        ka,
        shape="sphere",
        ri=(1.3116, 0.0),
        ref=3,
        out=Path("result.json"),
        binary=None,
        obj=None,
        subdiv=0,
        prism_aspect=1.0,
        edge_refine=None,
        orient=(1, 1, 1),
        orient_file=None,
        alpha_avg=1,
        ntheta=181,
        single=True,
        solver="fmm",
        system=None,
        quad=7,
        fmm_digits=5,
        gmres_tol=1e-3,
        gmres_restart=200,
        max_leaf=128,
        no_prec=False,
        mesh_quality=None,
        asm_gpu_list=(),
        lu_gpu_list=(),
        ff_gpu_list=(),
        extra_args=(),
        material=None,
        geometry=None,
        orientations=None,
        solver_options=None,
        orient_start=None,
        orient_count=None,
    ):
        if material is not None:
            ri = material.refractive_index
        if geometry is not None:
            shape = geometry.shape
            ref = geometry.ref
            obj = geometry.obj
            subdiv = geometry.subdiv
            prism_aspect = geometry.prism_aspect
            edge_refine = geometry.edge_refine
        if orientations is not None:
            single = orientations.single
            orient = orientations.counts
            orient_file = orientations.file
            alpha_avg = orientations.alpha_avg
            orient_start = orientations.start
            orient_count = orientations.count
        if solver_options is not None:
            solver = solver_options.backend
            system = solver_options.system
            quad = solver_options.quad
            fmm_digits = solver_options.fmm_digits
            gmres_tol = solver_options.gmres_tol
            gmres_restart = solver_options.gmres_restart
            max_leaf = solver_options.max_leaf
            no_prec = solver_options.no_prec

        self.ka = ka
        self.shape = shape
        self.ri = tuple(ri)
        self.ref = ref
        self.out = out
        self.binary = binary if binary is not None else _repo_root() / "bin" / "bem_cuda_fmm"
        self.obj = obj
        self.subdiv = subdiv
        self.prism_aspect = prism_aspect
        self.edge_refine = edge_refine
        self.orient = tuple(orient)
        self.orient_file = orient_file
        self.alpha_avg = alpha_avg
        self.orient_start = orient_start
        self.orient_count = orient_count
        self.ntheta = ntheta
        self.single = single
        self.solver = solver
        self.system = system
        self.quad = quad
        self.fmm_digits = fmm_digits
        self.gmres_tol = gmres_tol
        self.gmres_restart = gmres_restart
        self.max_leaf = max_leaf
        self.no_prec = no_prec
        self.mesh_quality = mesh_quality if mesh_quality is not None else MeshQuality()
        self.asm_gpu_list = tuple(asm_gpu_list)
        self.lu_gpu_list = tuple(lu_gpu_list)
        self.ff_gpu_list = tuple(ff_gpu_list)
        self.extra_args = tuple(extra_args)

    def command(self):
        effective = self.effective_solver_options()
        cmd = [
            str(self.binary),
            "--ka", str(self.ka),
            "--ri", str(self.ri[0]), str(self.ri[1]),
            "--ntheta", str(self.ntheta),
            "--solver", effective.backend,
            "--quad", str(effective.quad),
            "--fmm-digits", str(effective.fmm_digits),
            "--gmres-tol", str(effective.gmres_tol),
            "--gmres-restart", str(effective.gmres_restart),
            "--max-leaf", str(effective.max_leaf),
            "--out", str(self.out),
        ]
        if self.system is not None:
            cmd += ["--system", self.system]
        if self.obj is not None:
            cmd += ["--obj", str(self.obj), "--subdiv", str(self.subdiv)]
            if "--fast-obj" not in self.extra_args:
                cmd.append("--accurate")
        else:
            cmd += ["--shape", self.shape, "--ref", str(self.ref)]
            if self.shape in {"hex_prism", "prism6"}:
                cmd += ["--prism-aspect", str(self.prism_aspect)]
                if self.edge_refine is not None:
                    cmd += ["--edge-refine", str(self.edge_refine)]
        if self.single:
            cmd.append("--single")
        else:
            if self.orient_file is not None:
                cmd += ["--orient-file", str(self.orient_file)]
            else:
                cmd += ["--orient"] + [str(v) for v in self.orient]
            if self.alpha_avg != 1:
                cmd += ["--alpha-avg", str(self.alpha_avg)]
            if self.orient_start is not None:
                cmd += ["--orient-start", str(self.orient_start)]
            if self.orient_count is not None:
                cmd += ["--orient-count", str(self.orient_count)]
        if effective.no_prec:
            cmd.append("--no-prec")
        if self.mesh_quality.report is not None:
            cmd += ["--mesh-quality-report", str(self.mesh_quality.report)]
        if self.mesh_quality.strict:
            cmd.append("--mesh-quality-strict")
        cmd += list(self.extra_args)
        return cmd

    def environment(self):
        """Environment variables required for this job's explicit GPU policy."""

        env = {}
        if self.asm_gpu_list:
            env["BEM_ASM_GPU_LIST"] = ",".join(str(v) for v in self.asm_gpu_list)
        if self.lu_gpu_list:
            env["BEM_LU_GPU_LIST"] = ",".join(str(v) for v in self.lu_gpu_list)
        if self.ff_gpu_list:
            env["BEM_FF_GPU_LIST"] = ",".join(str(v) for v in self.ff_gpu_list)
        return env

    def to_dict(self):
        return {
            "ka": self.ka,
            "shape": self.shape,
            "ri": list(self.ri),
            "ref": self.ref,
            "out": str(self.out),
            "binary": str(self.binary),
            "obj": None if self.obj is None else str(self.obj),
            "subdiv": self.subdiv,
            "prism_aspect": self.prism_aspect,
            "edge_refine": self.edge_refine,
            "orient": list(self.orient),
            "orient_file": None if self.orient_file is None else str(self.orient_file),
            "alpha_avg": self.alpha_avg,
            "orient_start": self.orient_start,
            "orient_count": self.orient_count,
            "ntheta": self.ntheta,
            "single": self.single,
            "solver": self.solver,
            "system": self.system,
            "quad": self.quad,
            "fmm_digits": self.fmm_digits,
            "gmres_tol": self.gmres_tol,
            "gmres_restart": self.gmres_restart,
            "max_leaf": self.max_leaf,
            "no_prec": self.no_prec,
            "mesh_quality": self.mesh_quality.to_dict(),
            "asm_gpu_list": list(self.asm_gpu_list),
            "lu_gpu_list": list(self.lu_gpu_list),
            "ff_gpu_list": list(self.ff_gpu_list),
            "extra_args": list(self.extra_args),
            "material": self.material().to_dict(),
            "geometry": self.geometry().to_dict(),
            "orientations": self.orientations().to_dict(),
            "solver_options": self.solver_options().to_dict(),
        }

    def to_json(self, path):
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n")

    def material(self):
        return Material(self.ri)

    def geometry(self):
        return Geometry(
            shape=self.shape,
            ref=self.ref,
            obj=self.obj,
            subdiv=self.subdiv,
            prism_aspect=self.prism_aspect,
            edge_refine=self.edge_refine,
        )

    def orientations(self):
        return OrientationGrid(
            single=self.single,
            counts=self.orient,
            file=self.orient_file,
            alpha_avg=self.alpha_avg,
            start=self.orient_start,
            count=self.orient_count,
        )

    def solver_options(self):
        return SolverOptions(
            backend=self.solver,
            system=self.system,
            quad=self.quad,
            fmm_digits=self.fmm_digits,
            gmres_tol=self.gmres_tol,
            gmres_restart=self.gmres_restart,
            max_leaf=self.max_leaf,
            no_prec=self.no_prec,
        )

    def effective_solver_options(self):
        """Solver options after BEM-CUDA's guarded OBJ defaults are applied."""

        opts = self.solver_options()
        fast_obj = self.obj is not None and "--fast-obj" in self.extra_args
        if self.obj is not None and not fast_obj:
            if opts.fmm_digits == 5:
                opts.fmm_digits = 6
            if opts.gmres_tol == 1e-3:
                opts.gmres_tol = 5e-4
            if opts.gmres_restart == 200:
                opts.gmres_restart = 500
        return opts

    def manifest(self):
        """Full reproducibility manifest: parameters, command and environment."""

        payload = self.to_dict()
        payload["effective_solver_options"] = self.effective_solver_options().to_dict()
        payload["command"] = self.command()
        payload["environment"] = self.environment()
        payload["semantic_id"] = self.semantic_id()
        return payload

    def semantic_id(self):
        """Stable short hash for comparing intended runs in queues and plots."""

        payload = {
            "ka": self.ka,
            "material": self.material().to_dict(),
            "geometry": self.geometry().to_dict(),
            "orientations": self.orientations().to_dict(),
            "ntheta": self.ntheta,
            "solver_options": self.effective_solver_options().to_dict(),
            "mesh_quality": {"strict": self.mesh_quality.strict},
            "asm_gpu_list": list(self.asm_gpu_list),
            "lu_gpu_list": list(self.lu_gpu_list),
            "ff_gpu_list": list(self.ff_gpu_list),
            "extra_args": list(self.extra_args),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_dict(cls, payload):
        material = Material.from_dict(payload.get("material", payload))
        geometry = Geometry.from_dict(payload.get("geometry", payload))
        orientations_payload = payload.get("orientations")
        if orientations_payload is None:
            orientations_payload = {
                "single": payload.get("single", True),
                "counts": payload.get("orient", (1, 1, 1)),
                "file": payload.get("orient_file"),
                "alpha_avg": payload.get("alpha_avg", 1),
                "start": payload.get("orient_start"),
                "count": payload.get("orient_count"),
            }
        solver_payload = payload.get("solver_options")
        if solver_payload is None:
            solver_payload = {
                "backend": payload.get("solver", "fmm"),
                "system": payload.get("system"),
                "quad": payload.get("quad", 7),
                "fmm_digits": payload.get("fmm_digits", 5),
                "gmres_tol": payload.get("gmres_tol", 1e-3),
                "gmres_restart": payload.get("gmres_restart", 200),
                "max_leaf": payload.get("max_leaf", 128),
                "no_prec": payload.get("no_prec", False),
            }
        mesh_payload = payload.get("mesh_quality", {})
        mesh_quality = MeshQuality(
            strict=mesh_payload.get("strict", True),
            report=mesh_payload.get("report"),
        )
        return cls(
            ka=payload["ka"],
            out=Path(payload.get("out", "result.json")),
            binary=payload.get("binary"),
            material=material,
            geometry=geometry,
            orientations=OrientationGrid.from_dict(orientations_payload),
            solver_options=SolverOptions.from_dict(solver_payload),
            ntheta=payload.get("ntheta", 181),
            mesh_quality=mesh_quality,
            asm_gpu_list=payload.get("asm_gpu_list", ()),
            lu_gpu_list=payload.get("lu_gpu_list", ()),
            ff_gpu_list=payload.get("ff_gpu_list", ()),
            extra_args=payload.get("extra_args", ()),
        )

    def run(self, cwd=None, env=None, check=True):
        cwd = cwd or _repo_root()
        merged_env = os.environ.copy()
        merged_env.update(self.environment())
        if env:
            merged_env.update(env)
        cmd = self.command()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=merged_env,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        data = None
        output = self.out if self.out.is_absolute() else cwd / self.out
        if output.exists():
            data = json.loads(output.read_text())
        result = RunResult(cmd, proc.returncode, proc.stdout, proc.stderr, output, data)
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
            )
        return result


def write_orientation_file(path, rows):
    """Write alpha beta gamma [weight] rows in degrees for ``--orient-file``."""

    with path.open("w") as f:
        for row in rows:
            f.write(" ".join("{:.17g}".format(float(v)) for v in row) + "\n")
