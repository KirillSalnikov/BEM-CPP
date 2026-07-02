#!/usr/bin/env python3

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_accuracy_matrix_15_results.sh"


def make_repo(tmp_path: Path, audit_rc: int) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, repo / "scripts" / SCRIPT.name)
    (repo / "scripts" / SCRIPT.name).chmod(0o755)
    audit = repo / "scripts" / "audit_accuracy_matrix_15.py"
    audit.write_text(
        "#!/usr/bin/env python3\n"
        "print('fake audit ran')\n"
        f"raise SystemExit({audit_rc})\n"
    )
    audit.chmod(0o755)
    return repo


def make_fake_path(tmp_path: Path) -> Path:
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    ssh = fakebin / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "cmd=\"${@: -1}\"\n"
        "printf 'SSH_CMD %s\\n' \"$cmd\" >> \"$FAKE_LOG\"\n"
        "case \"$cmd\" in\n"
        "  *'printf BEM_REMOTE_OK'*) printf BEM_REMOTE_OK; exit 0 ;;\n"
        "  *'printf '\\''%s\\\\n'\\'' /remote/repo'*) printf '/remote/repo\\n'; exit 0 ;;\n"
        "  *'test -d /remote/repo && printf'*) printf '/remote/repo\\n'; exit 0 ;;\n"
        "  *'runs/adda_ocl_benchmark_ext/dust_ka15_m1p6_dpl20_scaled'*) exit 1 ;;\n"
        "  *'test -d'*) exit 0 ;;\n"
        "  *'===PROCS==='*) printf '===PROCS===\\n===GPU===\\nnvidia-smi-missing\\n===FILES===\\n===AUDIT===\\nREMOTE_AUDIT_RC=4\\n'; exit 0 ;;\n"
        "esac\n"
        "exit 0\n"
    )
    ssh.chmod(0o755)
    rsync = fakebin / "rsync"
    rsync.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'RSYNC %s\\n' \"$*\" >> \"$FAKE_LOG\"\n"
        "dest=\"${@: -1}\"\n"
        "mkdir -p \"$dest\"\n"
        "exit 0\n"
    )
    rsync.chmod(0o755)
    return fakebin


def run_fetch(repo: Path, fakebin: Path, log_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["FAKE_LOG"] = str(log_path)
    return subprocess.run(
        [
            str(repo / "scripts" / SCRIPT.name),
            "--hosts", "fakehost",
            "--remote-repo", "/remote/repo",
            "--local-repo", str(repo),
            *args,
        ],
        cwd=str(repo),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fakebin = make_fake_path(tmp_path)
        log_path = tmp_path / "calls.log"

        strict_repo = make_repo(tmp_path, audit_rc=4)
        proc = run_fetch(strict_repo, fakebin, log_path)
        assert proc.returncode == 4, proc.stdout
        assert "FETCH_HOSTS fakehost" in proc.stdout, proc.stdout
        assert "REMOTE_AUDIT_RC=4" in proc.stdout, proc.stdout
        assert "fake audit ran" in proc.stdout, proc.stdout
        assert "FETCH_AUDIT_RC 4" in proc.stdout, proc.stdout

        best_repo = make_repo(tmp_path / "best", audit_rc=4)
        proc = run_fetch(best_repo, fakebin, log_path, "--audit-best-effort")
        assert proc.returncode == 0, proc.stdout
        assert "FETCH_AUDIT_RC 4" in proc.stdout, proc.stdout

        no_audit_repo = make_repo(tmp_path / "noaudit", audit_rc=4)
        proc = run_fetch(no_audit_repo, fakebin, log_path, "--no-audit")
        assert proc.returncode == 0, proc.stdout
        assert "FETCH_AUDIT_RC" not in proc.stdout, proc.stdout

    print("fetch accuracy matrix results: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
