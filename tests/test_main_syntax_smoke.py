#!/usr/bin/env python3
"""Syntax-check main.cpp without requiring a full cuFFT installation."""

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        detect_json = tmp / "cuda_toolchain.json"
        proc = subprocess.run(
            ["python3", "scripts/detect_cuda_toolchain.py", "--json-out", str(detect_json)],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout
        data = json.loads(detect_json.read_text())
        include_dirs = (data.get("selected") or {}).get("include_dirs") or []
        assert include_dirs, "no CUDA include dirs detected for main.cpp syntax smoke"

        # main.cpp only needs cuFFT type names through pfft headers in this smoke.
        (tmp / "cufft.h").write_text(
            "#ifndef CUFFT_H\n"
            "#define CUFFT_H\n"
            "typedef int cufftHandle;\n"
            "typedef struct { double x, y; } cufftDoubleComplex;\n"
            "typedef struct { float x, y; } cufftComplex;\n"
            "#endif\n"
        )

        cmd = ["g++", "-std=c++11", "-Isrc", "-I" + str(tmp)]
        for inc in include_dirs:
            cmd.append("-I" + inc)
        cmd += ["-fsyntax-only", "src/main.cpp"]
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout

    print("main syntax smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
