"""Compare the scheduled physical T columns with their independent ka80 baseline."""
import json
import subprocess
import time
from pathlib import Path
import numpy as np

here=Path(__file__).resolve().parent
base=here.parent/"muller_ka80_week_20261116"/"first_modal_pair_cpudone"
spec=json.loads((here/"pilot_spec.json").read_text())
started=time.monotonic()
subprocess.run(spec["command"],check=True,cwd=here.parents[1])
elapsed=time.monotonic()-started
out=here/"ka80_pair"
errors=[]
for i in spec["indices"]:
    name=f"{i:06d}.npy"
    x,y=[np.load(d/"columns"/name) for d in (out,base)]
    errors.append(float(np.linalg.norm(x-y)/np.linalg.norm(y)))
def metrics(folder):
    s=json.loads((folder/"columns"/f"{spec['indices'][0]:06d}.json").read_text())["status"]
    return {k:s[k] for k in ("total_worker_s","strict_residual_x","strict_residual_y","compressed_fft_action_calls","incident_field_s")}
a,b=metrics(out),metrics(base)
fft=json.loads((out/"external_fft.json").read_text())
control=json.loads((base.parent/"first_modal_pilot_cpudone.json").read_text())
residuals=[a["strict_residual_x"],a["strict_residual_y"]]
result=dict(candidate=a,baseline=b,column_errors=errors,speedup=b["total_worker_s"]/a["total_worker_s"],
            wall_seconds=elapsed,baseline_wall_seconds=control["elapsed_s"],
            wall_speedup=control["elapsed_s"]/elapsed,
            rotation_calls={k:fft.get(k,0) for k in ("folded_rotation_calls","mixed_rotation_calls")},
            passed=all(np.isfinite(x) and x<=1e-3 for x in errors) and
                   all(np.isfinite(x) and x<=2e-6 for x in residuals))
(here/"ka80_pair_comparison.json").write_text(json.dumps(result,indent=2)+"\n")
