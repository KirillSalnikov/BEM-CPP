import importlib.util
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_out_dir_matches_absolute_campaign_path(tmp_path):
    promoter = load("campaign_promoter", "scripts/promote_dust_profile_to_final.py")
    worker = load("campaign_worker", "scripts/run_dust_adda_campaign_worker.py")
    relative = Path("case") / "profile"
    expected = tmp_path / relative
    expected.mkdir(parents=True)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)",
         "adaptive_nested_bg_orient_queue.py", "--out-dir", str(relative)],
        cwd=str(tmp_path),
    )
    try:
        time.sleep(0.1)
        command = Path("/proc/{}/cmdline".format(process.pid)).read_bytes()
        command = command.replace(b"\0", b" ").decode()
        assert promoter.process_references_directory(process.pid, command, expected)
        assert worker.profile_is_active(tmp_path, expected)
        assert not worker.profile_is_active(tmp_path, tmp_path / "other")
    finally:
        process.terminate()
        process.wait(timeout=5)
