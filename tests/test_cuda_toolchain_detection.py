#!/usr/bin/env python3
"""Unit checks for CUDA toolkit/runtime detection."""

from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.detect_cuda_toolchain import (
    build_recommendation,
    classify_runtime,
    host_compiler_compatibility,
    inspect_root,
    nvcc_release_major_minor,
    nvcc_supported_gcc_major,
    parse_version_major,
    select_toolkit,
    summary_text,
)
from scripts.bootstrap_cuda_toolchain import discover_components, validate_components


def main() -> int:
    missing = classify_runtime([], [])
    assert missing["ready_for_cuda_runtime"] is False
    assert missing["missing"] == ["/dev/nvidia*", "libcuda.so"]

    no_smi = classify_runtime(
        ["/dev/nvidiactl", "/dev/nvidia0"],
        ["libcuda.so.1 (libc6,x86-64) => /usr/lib/libcuda.so.1"],
    )
    assert no_smi["ready_for_cuda_runtime"] is True
    assert no_smi["missing"] == []

    broken_smi = classify_runtime(
        ["/dev/nvidiactl", "/dev/nvidia0"],
        ["libcuda.so.1 (libc6,x86-64) => /usr/lib/libcuda.so.1"],
        "/usr/bin/nvidia-smi",
        {"returncode": 1, "stdout": "", "stderr": "driver error"},
    )
    assert broken_smi["ready_for_cuda_runtime"] is False
    assert broken_smi["missing"] == ["working nvidia-smi"]

    ok_smi = classify_runtime(
        ["/dev/nvidiactl", "/dev/nvidia0"],
        ["libcuda.so.1 (libc6,x86-64) => /usr/lib/libcuda.so.1"],
        "/usr/bin/nvidia-smi",
        {"returncode": 0, "stdout": "Tesla V100, 535.183.01", "stderr": ""},
    )
    assert ok_smi["ready_for_cuda_runtime"] is True
    assert ok_smi["missing"] == []
    assert parse_version_major("13.3.0") == 13
    assert nvcc_release_major_minor("Cuda compilation tools, release 12.2, V12.2.140") == (12, 2)
    assert nvcc_supported_gcc_major("Cuda compilation tools, release 12.2, V12.2.140") == 12
    incompatible = host_compiler_compatibility(
        "Cuda compilation tools, release 12.2, V12.2.140",
        {"path": "/usr/bin/g++", "version": "13.3.0", "major": 13},
    )
    assert incompatible["supported"] is False
    assert "GCC <= 12" in incompatible["reason"]
    compatible = host_compiler_compatibility(
        "Cuda compilation tools, release 12.2, V12.2.140",
        {"path": "/usr/bin/g++-12", "version": "12.4.0", "major": 12},
    )
    assert compatible["supported"] is True
    rec = build_recommendation({
        "root": "/cuda-12.2",
        "usable_for_local_build": False,
        "host_compiler_compatibility": incompatible,
    })
    assert "g++-12" in rec["make_command"]
    assert "bootstrap_cuda_12_8" in rec
    assert rec["reason"] == incompatible["reason"]
    assert build_recommendation(None)["conda_env_file"] == "environment.cuda.yml"
    selected = select_toolkit([
        {"usable_for_bem_cuda": True, "usable_for_local_build": False, "root": "/cuda-12.2"},
        {"usable_for_bem_cuda": True, "usable_for_local_build": True, "root": "/cuda-12.8"},
    ])
    assert selected["root"] == "/cuda-12.8"
    fallback = select_toolkit([
        {"usable_for_bem_cuda": False, "usable_for_local_build": False, "root": "/broken"},
        {"usable_for_bem_cuda": True, "usable_for_local_build": False, "root": "/cuda-12.2"},
    ])
    assert fallback["root"] == "/cuda-12.2"
    assert "CUDA local build unavailable" in summary_text({
        "selected": {"root": "/cuda-12.2", "usable_for_local_build": False},
        "recommendation": {"reason": "bad compiler", "install_host_compiler": "install g++-12"},
    })
    assert "CUDA toolkit ok" in summary_text({
        "selected": {"root": "/cuda-12.8", "usable_for_local_build": True},
        "recommendation": None,
    })
    proc = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts" / "detect_cuda_toolchain.py"),
            "--require-local-build",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 4), proc.stdout + proc.stderr

    with tempfile.TemporaryDirectory() as tmp:
        assert validate_components(discover_components(Path(tmp))) != []
        root = Path(tmp)
        (root / "bin").mkdir()
        nvcc = root / "bin" / "nvcc"
        nvcc.write_text("#!/usr/bin/env bash\necho nvcc test\n")
        nvcc.chmod(0o755)
        (root / "include").mkdir()
        (root / "include" / "cuda_runtime.h").write_text("\n")
        (root / "include" / "cuComplex.h").write_text("\n")
        (root / "lib" / "x86_64-linux-gnu").mkdir(parents=True)
        (root / "lib" / "x86_64-linux-gnu" / "libcudart.so").write_text("\n")
        debian_root = inspect_root(root)
        assert debian_root["usable_for_bem_cuda"] is True, debian_root
        assert "host_compiler_compatibility" in debian_root, debian_root
        assert "usable_for_local_build" in debian_root, debian_root
        assert str(root / "lib" / "x86_64-linux-gnu") in debian_root["lib_dirs"], debian_root

    print("cuda toolchain detection: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
