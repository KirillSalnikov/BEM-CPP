#!/usr/bin/env bash
set -euo pipefail

cd /home/kirill_epyc/BEM-CUDA

src=runs/adda_greek_dpl25/discrete_a4b2g8
out=runs/adda_greek_dpl25/discrete_a4b2g8_ntheta180
mkdir -p "$out"

python3 - <<'PY'
from pathlib import Path
import re
import shlex

src = Path("runs/adda_greek_dpl25/discrete_a4b2g8")
cmd_file = Path("/tmp/adda_greek_ntheta180_cmds.sh")
orient_re = re.compile(r"-orient\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")
lines = []
for log in sorted(src.glob("*/log")):
    text = log.read_text(errors="replace")
    match = orient_re.search(text)
    if not match:
        raise SystemExit(f"no orient in {log}")
    alpha, beta, gamma = match.groups()
    name = log.parent.name
    out_dir = Path("runs/adda_greek_dpl25/discrete_a4b2g8_ntheta180") / name
    if (out_dir / "mueller").exists() and (out_dir / "log").exists():
        continue
    cmd = [
        "/home/kirill_epyc/adda/src/seq/adda",
        "-dir", str(out_dir),
        "-shape", "read", "runs/adda_greek_dpl25/greek_ka5p71_dpl25.shape",
        "-m", "1.6", "0.002",
        "-dpl", "25",
        "-ntheta", "180",
        "-scat_matr", "muel",
        "-orient", alpha, beta, gamma,
        "-eps", "5",
        "-iter", "qmr",
        "-sym", "no",
    ]
    lines.append(" ".join(shlex.quote(x) for x in cmd))

cmd_file.write_text("\n".join(lines) + ("\n" if lines else ""))
print(f"commands {len(lines)} written to {cmd_file}", flush=True)
PY

if [[ -s /tmp/adda_greek_ntheta180_cmds.sh ]]; then
    xargs -a /tmp/adda_greek_ntheta180_cmds.sh -I{} -P 8 bash -lc '{} >/dev/null 2>&1'
fi

done_count=$(find "$out" -mindepth 2 -maxdepth 2 -name mueller | wc -l)
echo "done mueller files: $done_count"
