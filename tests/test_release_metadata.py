#!/usr/bin/env python3
"""Check release metadata and reject machine-local paths."""

from pathlib import Path
import json
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> int:
    version = read("VERSION").strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version), version

    required = [
        "LICENSE",
        "NOTICE",
        "CITATION.cff",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "README.md",
    ]
    for relative in required:
        assert (ROOT / relative).is_file(), relative

    citation = read("CITATION.cff")
    assert f"version: {version}" in citation
    assert "license: MIT" in citation
    assert "repository-code: https://github.com/KirillSalnikov/BEM-CPP" in citation
    assert f"## [{version}]" in read("CHANGELOG.md")
    reference = json.loads(read(f"reference/v{version}/small_sphere.json"))
    assert reference["software_version"] == version
    assert reference["observed_on_release_machine"][
        "m11_solid_angle_relative_l2"
    ] <= reference["acceptance"]["maximum_m11_solid_angle_relative_l2"]

    makefile = read("Makefile")
    driver = read("tools/muller_nodal_fmm_demo.cpp")
    assert "cat VERSION" in makefile
    assert 'std::strcmp(argv[i], "--help")' in driver
    assert 'std::strcmp(argv[i], "--version")' in driver
    assert "unknown option or missing value" in driver
    assert "software_version" in driver
    assert (ROOT / "scripts/release_audit.sh").stat().st_mode & 0o111
    assert (ROOT / "scripts/package_release.sh").stat().st_mode & 0o111
    assert (ROOT / "bem").stat().st_mode & 0o111
    launcher = read("bem")
    for command in ("run", "average", "resume", "validate", "presets"):
        assert f'add_parser("{command}"' in launcher

    checked_suffixes = {".md", ".py", ".sh", ".cpp", ".cu", ".h", ".yml"}
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    for relative in tracked:
        path = ROOT / relative
        if path.suffix not in checked_suffixes and path.name != "Makefile":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "/" + "home/kirill" not in text, str(path.relative_to(ROOT))
        assert "/" + "home/user" not in text, str(path.relative_to(ROOT))

    print(f"release metadata: {version}, ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
