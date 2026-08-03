#!/usr/bin/env python3
"""Keep release documentation aligned with the public command-line interface."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_DOCUMENTS = ("README.md", "MANUAL.md", "MANUAL.tex")


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    version = read("VERSION").strip()
    documents = {name: read(name) for name in PRIMARY_DOCUMENTS}

    assert f"Current release: `{version}`" in documents["README.md"]
    assert f"release `{version}`" in documents["MANUAL.md"]
    assert f"Версия программы: {version}" in documents["MANUAL.tex"]
    assert f"reference/v{version}/small_sphere.json" in documents["README.md"]
    assert (ROOT / f"reference/v{version}/small_sphere.json").is_file()

    joined = "\n".join(documents.values())
    assert "/path/to/BEM-CUDA" not in joined
    assert "BEM-CUDA" not in documents["README.md"]
    assert "BEM-CUDA" not in documents["MANUAL.md"]
    assert "BEM-CUDA" not in documents["MANUAL.tex"]
    assert "Архивные растровые графики" in documents["MANUAL.tex"]

    manual_assets = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "manual_assets").glob("*.csv")
    )
    assert "BEM-CUDA" not in manual_assets
    assert "poster_a0/assets" not in documents["MANUAL.tex"]

    assert "./bem run --shape prism --ka 25 --ri 1.3" in documents["README.md"]
    assert "./bem run --shape prism --ka 25 --ri 1.3" in documents["MANUAL.tex"]
    assert "scripts/release_audit.sh --gpu" in documents["MANUAL.tex"]

    documented = set(re.findall(r"--[a-z][a-z0-9-]*", joined))
    implementation = "\n".join(
        read(name) for name in ("bem", "tools/muller_nodal_fmm_demo.cpp", "src/main.cpp")
    )
    implemented = set(re.findall(r"--[a-z][a-z0-9-]*", implementation))
    documentation_tool_options = {
        "--host",
        "--keep-going",
        "--keep-intermediates",
        "--keep-logs",
    }
    unmatched = documented - implemented - documentation_tool_options
    assert not unmatched, sorted(unmatched)

    assert (ROOT / "MANUAL.pdf").stat().st_size > 1_000_000
    print(f"documentation freshness: {version}, ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
