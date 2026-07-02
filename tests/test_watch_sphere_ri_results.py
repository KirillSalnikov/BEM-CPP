#!/usr/bin/env python3

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_watch(tmp: Path) -> subprocess.CompletedProcess:
    fake_bin = tmp / "bin"
    fake_bin.mkdir()
    (fake_bin / "rsync").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fake_bin / "pdflatex").write_text("#!/usr/bin/env bash\nexit 0\n")
    for tool in ["rsync", "pdflatex"]:
        (fake_bin / tool).chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "MAX_POLLS": "1",
        "POLL_S": "0",
        "REMOTE": "unused@example",
        "REMOTE_ROOT": "/unused",
        "STRICT_DIR": "strict",
        "FALLBACK_DIR": "fallback",
    })
    return subprocess.run(
        ["bash", str(ROOT / "scripts/watch_sphere_ri_results.sh")],
        cwd=tmp,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )


def write_stub_project(tmp: Path, table: str, coverage: str) -> None:
    assets = tmp / "poster_a0" / "assets"
    assets.mkdir(parents=True)
    (assets / "table_index_sweep.csv").write_text(table)
    (assets / "table_index_sweep_coverage.csv").write_text(coverage)
    (tmp / "poster_a0" / "make_assets.py").write_text("#!/usr/bin/env python3\n")
    (tmp / "poster_a0" / "validate_poster.py").write_text("#!/usr/bin/env python3\n")
    (tmp / "scripts").mkdir()
    (tmp / "scripts" / "summarize_sphere_ri_sweep.py").write_text("#!/usr/bin/env python3\n")


def test_watch_rejects_coverage_pass_without_full_mueller_source() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_stub_project(
            tmp,
            "ka,n,pass10_full_mueller\n"
            "5,1.5,False\n",
            "ka,n,any_pass,status\n"
            "5,1.5,True,PASS\n",
        )
        proc = run_watch(tmp)
        assert proc.returncode == 2, (proc.stdout, proc.stderr)
        assert "BAD_COVERAGE ka=5:n=1.5" in proc.stdout


def test_watch_accepts_only_full_mueller_complete_required_set() -> None:
    rows = []
    cov = []
    for ka in (5.0, 10.0, 15.0):
        for n in (1.5, 3.0, 4.5, 6.0):
            rows.append(f"{ka:g},{n:g},True")
            cov.append(f"{ka:g},{n:g},True,PASS")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_stub_project(
            tmp,
            "ka,n,pass10_full_mueller\n" + "\n".join(rows) + "\n",
            "ka,n,any_pass,status\n" + "\n".join(cov) + "\n",
        )
        proc = run_watch(tmp)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert "full-Mueller evidence" in proc.stdout


if __name__ == "__main__":
    test_watch_rejects_coverage_pass_without_full_mueller_source()
    test_watch_accepts_only_full_mueller_complete_required_set()
    print("watch sphere ri results: ok")
