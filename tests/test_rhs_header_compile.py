#!/usr/bin/env python3
"""Compile-check public RHS header after adding CUDA workspace types."""

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
        selected = data.get("selected") or {}
        include_dirs = selected.get("include_dirs") or []
        assert include_dirs, "no CUDA include dirs detected for rhs.h compile check"

        source = tmp / "rhs_header_check.cpp"
        source.write_text('#include "rhs.h"\nint main(){ return 0; }\n')
        cmd = ["g++", "-std=c++11", "-Isrc"]
        for inc in include_dirs:
            cmd.append("-I" + inc)
        cmd += ["-fsyntax-only", str(source)]
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout

    print("rhs header compile: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
