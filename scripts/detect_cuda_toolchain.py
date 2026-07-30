#!/usr/bin/env python3
"""Detect a usable CUDA toolkit for BEM-CUDA builds."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional


def parse_version_major(text: str) -> Optional[int]:
    for token in text.replace("-", ".").split("."):
        if token.isdigit():
            return int(token)
    return None


def detect_host_compiler() -> dict:
    candidates = []
    env_cxx = os.environ.get("CXX")
    if env_cxx:
        candidates.append(env_cxx)
    candidates.extend(["g++", "gcc"])

    seen = set()
    for name in candidates:
        path = shutil.which(name)
        if not path or path in seen:
            continue
        seen.add(path)
        proc = subprocess.run([path, "-dumpfullversion", "-dumpversion"],
                              universal_newlines=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              check=False)
        version = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
        return {
            "path": path,
            "version": version,
            "major": parse_version_major(version),
        }
    return {"path": None, "version": None, "major": None}


def nvcc_release_major_minor(version_text: Optional[str]) -> Optional[tuple]:
    if not version_text:
        return None
    marker = "release "
    pos = version_text.find(marker)
    if pos < 0:
        return None
    tail = version_text[pos + len(marker):].split(",", 1)[0].strip()
    parts = tail.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return int(parts[0]), int(parts[1])


def nvcc_supported_gcc_major(version_text: Optional[str]) -> Optional[int]:
    release = nvcc_release_major_minor(version_text)
    if release is None:
        return None
    major, minor = release
    if major < 12:
        return None
    if major == 12 and minor <= 2:
        return 12
    if major == 12:
        return 13
    return None


def host_compiler_compatibility(version_text: Optional[str], host: dict) -> dict:
    max_gcc = nvcc_supported_gcc_major(version_text)
    host_major = host.get("major")
    supported = None
    reason = None
    if max_gcc is not None and host_major is not None:
        supported = host_major <= max_gcc
        if not supported:
            reason = f"CUDA nvcc supports GCC <= {max_gcc}, current host compiler is GCC {host_major}"
    return {
        "host_compiler": host,
        "nvcc_max_gcc_major": max_gcc,
        "supported": supported,
        "reason": reason,
    }


def candidate_roots() -> List[Path]:
    roots = []
    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home:
        roots.append(Path(env_home))
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        roots.append(Path(nvcc_path).resolve().parents[1])
    conda_root = Path.home() / "anaconda3"
    repo_root = Path(__file__).resolve().parents[1]
    roots.extend([repo_root / ".cuda-local", Path.cwd() / ".cuda-local"])
    for p in (Path("/usr/local/cuda"), Path("/opt/cuda"), conda_root):
        roots.append(p)
    envs = conda_root / "envs"
    if envs.exists():
        for nvcc in sorted(envs.glob("*/bin/nvcc")):
            roots.append(nvcc.resolve().parents[1])
    pkgs = Path.home() / "anaconda3" / "pkgs"
    if pkgs.exists():
        for nvcc in sorted(pkgs.glob("cuda-nvcc-tools-*/bin/nvcc")):
            roots.append(nvcc.resolve().parents[1])
        for dev in sorted(pkgs.glob("cuda-nvcc-dev_linux-*/targets/x86_64-linux")):
            roots.append(dev.resolve().parents[1])
    out = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def inspect_root(root: Path) -> dict:
    nvcc = root / "bin" / "nvcc"
    include_dirs = [
        root / "targets" / "x86_64-linux" / "include",
        root / "include",
    ]
    lib_dirs = [
        root / "targets" / "x86_64-linux" / "lib",
        root / "targets" / "x86_64-linux" / "lib64",
        root / "lib" / "x86_64-linux-gnu",
        root / "lib64",
        root / "lib",
    ]
    has_cuda_runtime = any((d / "cuda_runtime.h").exists() for d in include_dirs)
    has_cucomplex = any((d / "cuComplex.h").exists() for d in include_dirs)
    has_cudart = any(list(d.glob("libcudart.so*")) for d in lib_dirs if d.exists())
    version = None
    if nvcc.exists():
        proc = subprocess.run([str(nvcc), "--version"], universal_newlines=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        version = (proc.stdout + proc.stderr).strip()
    host_compat = host_compiler_compatibility(version, detect_host_compiler())
    missing = []
    if not nvcc.exists():
        missing.append("nvcc")
    if not has_cuda_runtime:
        missing.append("cuda_runtime.h")
    if not has_cucomplex:
        missing.append("cuComplex.h")
    if not has_cudart:
        missing.append("libcudart.so")
    usable = not missing
    return {
        "root": str(root),
        "nvcc": str(nvcc),
        "nvcc_exists": nvcc.exists(),
        "include_dirs": [str(d) for d in include_dirs if d.exists()],
        "lib_dirs": [str(d) for d in lib_dirs if d.exists()],
        "has_cuda_runtime_h": has_cuda_runtime,
        "has_cuComplex_h": has_cucomplex,
        "has_libcudart": has_cudart,
        "missing": missing,
        "usable_for_bem_cuda": usable,
        "host_compiler_compatibility": host_compat,
        "usable_for_local_build": usable and host_compat["supported"] is not False,
        "version": version,
    }


def classify_runtime(dev_nodes: List[str], libcuda_lines: List[str],
                     nvidia_smi: Optional[str] = None,
                     nvidia_smi_result: Optional[dict] = None) -> dict:
    has_device = any(Path(node).name in {"nvidiactl", "nvidia0"} for node in dev_nodes)
    has_libcuda = bool(libcuda_lines)
    smi_ok = bool(nvidia_smi_result and nvidia_smi_result["returncode"] == 0)
    ready = has_device and has_libcuda and (smi_ok or nvidia_smi is None)
    missing = []
    if not has_device:
        missing.append("/dev/nvidia*")
    if not has_libcuda:
        missing.append("libcuda.so")
    if nvidia_smi and not smi_ok:
        missing.append("working nvidia-smi")
    return {
        "ready_for_cuda_runtime": ready,
        "dev_nodes": dev_nodes,
        "has_nvidia_device": has_device,
        "nvidia_smi": nvidia_smi,
        "nvidia_smi_result": nvidia_smi_result,
        "libcuda": libcuda_lines,
        "has_libcuda": has_libcuda,
        "missing": missing,
    }


def detect_runtime() -> dict:
    dev_nodes = sorted(str(p) for p in Path("/dev").glob("nvidia*"))
    nvidia_smi = shutil.which("nvidia-smi")
    nvidia_smi_result = None
    if nvidia_smi:
        proc = subprocess.run([nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                              universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        nvidia_smi_result = {
            "command": [nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

    libcuda_lines: List[str] = []
    try:
        proc = subprocess.run(["ldconfig", "-p"], universal_newlines=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0:
            libcuda_lines = [line.strip() for line in proc.stdout.splitlines() if "libcuda.so" in line]
    except FileNotFoundError:
        pass
    if not libcuda_lines:
        for root in (Path("/usr/lib64"), Path("/usr/lib"), Path("/lib64"), Path("/lib")):
            if root.exists():
                libcuda_lines.extend(str(p) for p in sorted(root.glob("libcuda.so*")))

    return classify_runtime(dev_nodes, libcuda_lines, nvidia_smi, nvidia_smi_result)


def build_recommendation(usable: Optional[dict]) -> Optional[dict]:
    if not usable:
        return {
            "conda_env_file": "environment.cuda.yml",
            "create_command": "conda env create -f environment.cuda.yml",
            "activate_command": "conda activate bem-cuda-toolchain",
            "rerun": "make cuda-toolchain-check",
        }
    if usable.get("usable_for_local_build") is False:
        compat = usable.get("host_compiler_compatibility", {})
        max_gcc = compat.get("nvcc_max_gcc_major")
        if max_gcc:
            return {
                "reason": compat.get("reason"),
                "install_host_compiler": f"install gcc-{max_gcc} g++-{max_gcc} or use a CUDA toolkit that supports the current compiler",
                "make_command": f"CXX=g++-{max_gcc} CUDA_HOME={usable['root']} make fmm-only",
                "bootstrap_cuda_12_8": "python3 scripts/bootstrap_cuda_toolchain.py --out .cuda-local --force",
                "rerun": "make cuda-toolchain-check",
            }
        return {
            "reason": "CUDA toolkit found, but local build compatibility could not be verified",
            "rerun": "make cuda-toolchain-check",
        }
    return None


def select_toolkit(inspected: List[dict]) -> Optional[dict]:
    local = next((item for item in inspected
                  if item["usable_for_bem_cuda"] and item.get("usable_for_local_build")), None)
    if local:
        return local
    return next((item for item in inspected if item["usable_for_bem_cuda"]), None)


def summary_text(report: dict) -> str:
    selected = report.get("selected")
    if not selected:
        rec = report.get("recommendation") or {}
        return (
            "CUDA toolkit unavailable: nvcc/cuda_runtime.h/cuComplex.h/libcudart.so were not found together.\n"
            f"Recommended: {rec.get('create_command', 'install a complete CUDA toolkit')}"
        )
    if not selected.get("usable_for_local_build"):
        rec = report.get("recommendation") or {}
        reason = rec.get("reason") or "selected CUDA toolkit is not usable for local build"
        action = rec.get("install_host_compiler") or rec.get("make_command") or "fix CUDA host compiler compatibility"
        return f"CUDA local build unavailable: {reason}\nRecommended: {action}"
    return f"CUDA toolkit ok for local build: {selected['root']}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--print-env", action="store_true",
                        help="Print shell exports for the first usable toolkit")
    parser.add_argument("--summary", action="store_true",
                        help="Print a short human-readable status instead of the full JSON report")
    parser.add_argument("--require-local-build", action="store_true",
                        help="Return non-zero unless the selected toolkit can be built with the current host compiler")
    parser.add_argument("--require-runtime", action="store_true",
                        help="Return non-zero unless CUDA toolkit and NVIDIA driver/runtime are usable")
    args = parser.parse_args()

    inspected = [inspect_root(root) for root in candidate_roots()]
    usable = select_toolkit(inspected)
    runtime = detect_runtime()
    report = {
        "usable": usable is not None,
        "selected": usable,
        "candidates": inspected,
        "runtime": runtime,
        "runtime_ready": (usable is not None) and runtime["ready_for_cuda_runtime"],
        "recommendation": build_recommendation(usable),
    }
    text = json.dumps(report, indent=2)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    if args.print_env:
        if not usable:
            return 2
        print(f"export CUDA_HOME={usable['root']}")
        print('export PATH="$CUDA_HOME/bin:$PATH"')
        print('export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"')
    elif args.summary:
        print(summary_text(report))
    else:
        print(text)
    if args.require_local_build and (not usable or not usable.get("usable_for_local_build")):
        return 4
    if args.require_runtime and not report["runtime_ready"]:
        return 3
    return 0 if usable else 2


if __name__ == "__main__":
    raise SystemExit(main())
