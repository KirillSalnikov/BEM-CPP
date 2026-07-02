#!/usr/bin/env python3
"""Assemble a local CUDA toolkit tree from split conda CUDA packages."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def latest_pkg(pkgs: Path, pattern: str) -> Optional[Path]:
    matches = sorted(p for p in pkgs.glob(pattern) if p.is_dir())
    return matches[-1] if matches else None


def symlink_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def symlink_tree_files(src: Path, dst: Path) -> int:
    count = 0
    for item in sorted(src.rglob("*")):
        if item.is_dir():
            continue
        rel = item.relative_to(src)
        symlink_file(item, dst / rel)
        count += 1
    return count


def discover_components(pkgs: Path) -> dict:
    nvcc_tools = latest_pkg(pkgs, "cuda-nvcc-tools-12.8*")
    nvcc_dev = latest_pkg(pkgs, "cuda-nvcc-dev_linux-64-12.8*")
    nvvm_tools = latest_pkg(pkgs, "cuda-nvvm-tools-12.8*")
    crt_tools = latest_pkg(pkgs, "cuda-crt-tools-12.8*")
    cudart_dev = latest_pkg(pkgs, "cuda-cudart-dev_linux-64-12.8*")
    cudart_runtime = latest_pkg(pkgs, "cuda-cudart_linux-64-12.8*")
    cudart_static = latest_pkg(pkgs, "cuda-cudart-static_linux-64-12.8*")
    crt_dev = latest_pkg(pkgs, "cuda-crt-dev_linux-64-12.8*")
    return {
        "nvcc_tools": nvcc_tools,
        "nvcc_dev": nvcc_dev,
        "nvvm_tools": nvvm_tools,
        "crt_tools": crt_tools,
        "cudart_dev": cudart_dev,
        "cudart_runtime": cudart_runtime,
        "cudart_static": cudart_static,
        "crt_dev": crt_dev,
    }


def validate_components(components: dict) -> list:
    missing = []
    if not components["nvcc_tools"] or not (components["nvcc_tools"] / "bin" / "nvcc").exists():
        missing.append("cuda-nvcc-tools-12.8/bin/nvcc")
    nvcc_dev_target = components["nvcc_dev"] / "targets" / "x86_64-linux" if components["nvcc_dev"] else None
    if not nvcc_dev_target or not (nvcc_dev_target / "include" / "fatbinary_section.h").exists():
        missing.append("cuda-nvcc-dev_linux-64-12.8/targets/x86_64-linux/include/fatbinary_section.h")
    if not components["nvvm_tools"] or not (components["nvvm_tools"] / "nvvm" / "bin" / "cicc").exists():
        missing.append("cuda-nvvm-tools-12.8/nvvm/bin/cicc")
    if not components["crt_tools"] or not (components["crt_tools"] / "bin" / "crt" / "link.stub").exists():
        missing.append("cuda-crt-tools-12.8/bin/crt/link.stub")
    cudart_target = components["cudart_dev"] / "targets" / "x86_64-linux" if components["cudart_dev"] else None
    if not cudart_target or not (cudart_target / "include" / "cuda_runtime.h").exists():
        missing.append("cuda-cudart-dev_linux-64-12.8/targets/x86_64-linux/include/cuda_runtime.h")
    if not cudart_target or not (cudart_target / "include" / "cuComplex.h").exists():
        missing.append("cuda-cudart-dev_linux-64-12.8/targets/x86_64-linux/include/cuComplex.h")
    runtime_target = components["cudart_runtime"] / "targets" / "x86_64-linux" if components["cudart_runtime"] else None
    if not runtime_target or not any((runtime_target / "lib").glob("libcudart.so.*.*.*")):
        missing.append("cuda-cudart_linux-64-12.8/targets/x86_64-linux/lib/libcudart.so.<version>")
    static_target = components["cudart_static"] / "targets" / "x86_64-linux" if components["cudart_static"] else None
    if not static_target or not (static_target / "lib" / "libcudadevrt.a").exists():
        missing.append("cuda-cudart-static_linux-64-12.8/targets/x86_64-linux/lib/libcudadevrt.a")
    if not static_target or not (static_target / "lib" / "libcudart_static.a").exists():
        missing.append("cuda-cudart-static_linux-64-12.8/targets/x86_64-linux/lib/libcudart_static.a")
    crt_target = components["crt_dev"] / "targets" / "x86_64-linux" if components["crt_dev"] else None
    if not crt_target or not (crt_target / "include" / "crt" / "host_config.h").exists():
        missing.append("cuda-crt-dev_linux-64-12.8/targets/x86_64-linux/include/crt/host_config.h")
    return missing


def assemble(out: Path, components: dict, force: bool = False) -> dict:
    missing = validate_components(components)
    if missing:
        raise SystemExit("missing CUDA 12.8 conda components: " + ", ".join(missing))
    if out.exists() and force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    bin_dir = out / "bin"
    include_dir = out / "targets" / "x86_64-linux" / "include"
    lib_dir = out / "targets" / "x86_64-linux" / "lib"
    bin_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    nvcc_bin = components["nvcc_tools"] / "bin"
    for name in ("nvcc", "nvcc.profile", "cudafe++", "ptxas", "nvlink", "fatbinary", "bin2c"):
        src = nvcc_bin / name
        if src.exists():
            symlink_file(src, bin_dir / name)
    symlink_tree_files(components["crt_tools"] / "bin" / "crt", bin_dir / "crt")

    nvvm_dst = out / "nvvm"
    if nvvm_dst.exists() or nvvm_dst.is_symlink():
        if nvvm_dst.is_dir() and not nvvm_dst.is_symlink():
            shutil.rmtree(nvvm_dst)
        else:
            nvvm_dst.unlink()
    nvvm_dst.symlink_to(components["nvvm_tools"] / "nvvm")
    target_nvvm = out / "targets" / "x86_64-linux" / "nvvm"
    if target_nvvm.exists() or target_nvvm.is_symlink():
        if target_nvvm.is_dir() and not target_nvvm.is_symlink():
            shutil.rmtree(target_nvvm)
        else:
            target_nvvm.unlink()
    target_nvvm.symlink_to(Path("..") / ".." / "nvvm")

    cudart_target = components["cudart_dev"] / "targets" / "x86_64-linux"
    nvcc_dev_target = components["nvcc_dev"] / "targets" / "x86_64-linux"
    crt_target = components["crt_dev"] / "targets" / "x86_64-linux"
    include_count = symlink_tree_files(cudart_target / "include", include_dir)
    include_count += symlink_tree_files(nvcc_dev_target / "include", include_dir)
    include_count += symlink_tree_files(crt_target / "include", include_dir)
    runtime_target = components["cudart_runtime"] / "targets" / "x86_64-linux"
    actual_cudart = sorted((runtime_target / "lib").glob("libcudart.so.*.*.*"))[-1]
    symlink_file(actual_cudart, lib_dir / actual_cudart.name)
    symlink_file(Path(actual_cudart.name), lib_dir / "libcudart.so.12")
    symlink_file(Path("libcudart.so.12"), lib_dir / "libcudart.so")
    static_target = components["cudart_static"] / "targets" / "x86_64-linux"
    symlink_file(static_target / "lib" / "libcudadevrt.a", lib_dir / "libcudadevrt.a")
    symlink_file(static_target / "lib" / "libcudart_static.a", lib_dir / "libcudart_static.a")
    lib_count = 5

    manifest = {
        "out": str(out),
        "components": {k: str(v) for k, v in components.items()},
        "include_files": include_count,
        "lib_files": lib_count,
    }
    (out / "bem_cuda_toolchain_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkgs", type=Path, default=Path.home() / "anaconda3" / "pkgs")
    parser.add_argument("--out", type=Path, default=Path(".cuda-local"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    components = discover_components(args.pkgs)
    manifest = assemble(args.out, components, force=args.force)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
