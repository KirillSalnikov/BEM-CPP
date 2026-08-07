#!/usr/bin/env python3
"""Smoke test for reproducible resource profiling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "profile"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "profile_command.py"),
                "--output", str(output),
                "--cwd", str(ROOT),
                "--interval", "0.1",
                "--gpu", "-1",
                "--",
                sys.executable, "-c",
                (
                    "import time; x=bytearray(2_000_000); "
                    "end=time.perf_counter()+0.4; value=0; "
                    "exec('while time.perf_counter()<end:\\n value+=1')"
                ),
            ],
            check=True,
        )
        resources = json.loads(
            (output / "resources.json").read_text(encoding="utf-8")
        )
        assert resources["return_code"] == 0
        assert resources["wall_time_s"] >= 0.2
        assert resources["sample_count"] >= 2
        assert resources["gnu_time"]["time_max_rss_kib"] > 0
        assert resources["aggregates"]["process_rss_bytes"]["maximum"] > 0
        assert resources["aggregates"]["process_cpu_percent"]["maximum"] > 10
        assert resources["hardware"]["swap_total_bytes"] >= 0
        assert resources["aggregates"]["system_swap_used_bytes"]["maximum"] >= 0
        assert resources["aggregates"]["disk_free_bytes"]["minimum"] > 0
        assert (output / "resource_samples.csv").is_file()
        assert (output / "stdout.log").is_file()

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "interrupted"
        process = subprocess.Popen([
            sys.executable,
            str(ROOT / "scripts" / "profile_command.py"),
            "--output", str(output),
            "--cwd", str(ROOT),
            "--interval", "0.1",
            "--gpu", "-1",
            "--",
            sys.executable, "-c", "import time; time.sleep(10)",
        ])
        samples = output / "resource_samples.csv"
        deadline = time.monotonic() + 4.0
        rows: list[dict[str, str]] = []
        while time.monotonic() < deadline:
            if samples.exists():
                with samples.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) >= 2:
                    break
            time.sleep(0.05)
        assert len(rows) >= 2, "resource samples were not streamed"
        process.terminate()
        assert process.wait(timeout=5) == 128 + signal.SIGTERM
        resources = json.loads(
            (output / "resources.json").read_text(encoding="utf-8")
        )
        assert resources["interrupted_signal"] == signal.SIGTERM
        assert resources["sample_count"] >= 2
    print("profile command: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
