"""Run under the ka80 GPU lease. Test the same vectors before enabling a modal pilot."""
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
B=HERE.parent/"muller_ka80_week_20261116"
ROOT=HERE.parents[1]
CASE=B/"fft_ka80"
sys.path.insert(0,str(ROOT/"scripts"))
from compressed_fft_service import load_operator, validate_manifest
from screen import project
from bounded_bem_run import atomic_json


def main():
    # The numeric cutoff must pass on unrestricted AND single-rotation inputs.
    cpu=json.loads((HERE/"ka80_ref7_cpu.json").read_text())
    cuts=[cut for cut in (.0001,.0005,.001) if len([r for r in cpu if r["cut"]==cut])==4 and
          max(r["relative_native_error"] for r in cpu if r["cut"]==cut)<4e-4]
    if not cuts:raise RuntimeError("No validated rank schedule")
    cut=max(cuts)
    screens=json.loads((B/"spectral_traffic_screen.json").read_text())
    ranks={i:[max(1,r) for r in next(t for t in rec["rows"] if t["tol"]==cut)["batch_ranks"]]
           for i,rec in screens.items()}
    plan=HERE/"rank_plan.json"
    atomic_json(plan,dict(format=1,batch=8,assets_sha256=hashlib.sha256((CASE/"service_assets.json").read_bytes()).hexdigest(),ranks=ranks))
    keys=("BEM_TMATRIX_FFT_SINGLE_ROTATION","BEM_TMATRIX_FFT_RANK_PLAN","BEM_TMATRIX_FFT_PINNED_BUDGET_MIB")
    config=dict(zip(keys,["1",str(plan),"12288"]))
    configs={"baseline":{},"ranked":config}
    transverse_receipt=HERE/"transverse_2_7_8_1245_g32.json"
    if transverse_receipt.exists():
        proof=json.loads(transverse_receipt.read_text())
        controls={(r["input"],r["m"]):r for r in proof["rows"]}
        physical=(("smooth",0),("smooth",-1),("random",1),("random",-1))
        holdouts=(("holdout",3),("holdout",-1))
        if (cut==.0005 and all(k in controls and controls[k]["native_error"]<4e-4 for k in physical)
                and all(k in controls and controls[k]["relative_change"]<3e-4 for k in holdouts)):
            schedule=json.loads(plan.read_text())
            options=proof["parameters"].copy()
            groups=options.pop("groups")
            schedule["transverse"]={str(k):options for k in groups}
            new_plan=HERE/"rank_plan_transverse.json"
            atomic_json(new_plan,schedule)
            configs["transverse"]={**config,keys[1]:str(new_plan)}
    if "transverse" in configs:
        direct_plan=HERE/"rank_plan_direct.json"
        atomic_json(direct_plan,{**schedule,"direct_panels":True})
        configs["direct"]={**config,keys[1]:str(direct_plan)}
    result=dict(cutoff=cut,trials=[])
    vectors=[]
    def measure(op,x,y):
        times=[]
        for j in range(3):
            torch.cuda.synchronize()
            t=time.perf_counter()
            with torch.inference_mode():z=op(x)
            torch.cuda.synchronize()
            if j:times.append(time.perf_counter()-t)
        e=float(torch.linalg.vector_norm(z-y)/torch.linalg.vector_norm(y))
        return dict(seconds=min(times),samples=times,native_error=e)
    subprocess.run([sys.executable,'-m','pytest','-q',
                    'tests/test_bounded_spectral_storage.py::BoundedStorageTests::test_pinned_rank_panels_repeat_on_shared_slots'],
                   cwd=ROOT,check=True)
    if "transverse" in configs:
        check=subprocess.run([sys.executable,'-m','pytest','-q',
                              'tests/test_transverse_fft_interpolation.py::TransverseTests::test_guarded_gpu_action'],
                             cwd=ROOT)
        if check.returncode:
            result['transverse_unit_test_returncode']=check.returncode
            del configs['transverse']
            configs.pop('direct',None)
    if "direct" in configs:
        smoke=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_spectral_panel_kernel.py'],cwd=ROOT)
        if smoke.returncode:
            result["direct_unit_test_returncode"]=smoke.returncode
            del configs["direct"]
    validate_manifest(CASE)
    def evaluate(method,settings):
        for k in keys:os.environ.pop(k,None)
        os.environ.update(settings)
        op=load_operator(CASE)
        if not vectors:
            import pickle
            from prism_fft_rotation import rotation_tables
            with (CASE/"bound_operator.pkl").open("rb") as f:bound=pickle.load(f)
            perm,sign,_=rotation_tables(bound)
            del bound
            for label,m in [("random",1),("smooth",0),("random",-1),("smooth",-1)]:
                x,y=[np.fromfile(B/"native_ka80"/(n+".bin"),complex) for n in (label,"current_depth6_"+label)]
                if m>=0:x,y=project(x,perm,sign,m),project(y,perm,sign,m)
                vectors.append((label,m,x.astype(np.complex64),y.astype(np.complex64)))
        row=dict(method=method,candidate=bool(settings),stream_bytes=op.stream_transfer_bytes,
                 pinned_coefficients=op.pinned_coefficient_bytes,inputs=[])
        for label,m,x,y in vectors:
            r=measure(op,torch.as_tensor(x,device="cuda"),torch.as_tensor(y,device="cuda"))
            row["inputs"].append(dict(input=label,m=m,**r))
        return row
    for method,settings in configs.items():
        try:
            result["trials"].append(evaluate(method,settings))
        except (RuntimeError,MemoryError,ImportError) as error:
            if method not in ("transverse","direct"):raise
            result[method+"_failure"]=str(error)
        atomic_json(HERE/"ka80_ref7_gpu.json",result)
        gc.collect()
        torch.cuda.empty_cache()
    trials={t["method"]:t["inputs"] for t in result["trials"]}
    base=trials["baseline"]
    eligible=[name for name in trials if name!="baseline" and
              all(r["native_error"]<=5e-4 for r in trials[name])]
    if not eligible:
        result["passed_for_pilot"]=False
        atomic_json(HERE/"ka80_ref7_gpu.json",result)
        return
    def modal_seconds(name):
        return sum(r["seconds"] for r in trials[name] if r["m"]>=0)
    winner=eligible[0]
    for name in eligible[1:]:
        if (modal_seconds(name)*1.03<modal_seconds(winner) and
                all(new["seconds"]<=1.1*old["seconds"] for new,old in zip(trials[name][2:],trials[winner][2:]))):
            winner=name
    result["selected_method"]=winner
    new=trials[winner]
    config=configs[winner].copy()
    result["mode_action_speedup"]=sum(t["seconds"] for t in base[:2])/sum(t["seconds"] for t in new[:2])
    passed=all(t["native_error"]<=5e-4 for t in new) and result["mode_action_speedup"]>1.2
    result["passed_for_pilot"]=passed
    atomic_json(HERE/"ka80_ref7_gpu.json",result)
    if not passed:return
    runtime=HERE/"runtime"
    shutil.copytree(B/"runtime_v2",runtime,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("build_fmm_tmatrix.py","compressed_fft_service.py","symmetric_galerkin_fft.py","bounded_spectral_storage.py",
                 "spectral_panel_kernel.py","transverse_fft_interpolation.py","prism_fft_rotation.py","packed_current_host.py","packed_current_cpu.cpp",
                 "tmatrix_incident.py","tmatrix_c6.py","tmatrix_pipeline.py"):
        shutil.copy2(ROOT/"scripts"/name,runtime/"scripts"/name)
    # Keep all T indexing, right hand sides and native checker identical to the control.
    receipt=json.loads((B/"first_modal_pilot_cpudone.json").read_text())
    out=HERE/"ka80_pair"
    cmd=receipt["command"].copy()
    incident=json.loads((HERE/"ka80_incident_single.json").read_text())
    baseline,trial=incident["runs"][:2]
    if trial["relative_error"]<2e-12 and trial["setup_s"]+trial["prepare_s"]<baseline["prepare_s"]:
        cmd[cmd.index("--no-incident-cache")]="--separable-incident-cache"
        config["BEM_INCIDENT_THREADS"]=str(incident["threads"])
    cmd[1]=str(runtime/"scripts"/"build_fmm_tmatrix.py")
    cmd[cmd.index("--out")+1]=str(out)
    env={k:v for k,v in os.environ.items() if not k.startswith("BEM_")}
    env.update(receipt["environment"],**config)
    env["LD_LIBRARY_PATH"]=str(runtime/"lib")+os.pathsep+env.get("LD_LIBRARY_PATH","")
    # Release screen CUDA memory and its context before the near-capacity solve.
    atomic_json(HERE/"pilot_spec.json",dict(command=cmd,env_config=config,indices=receipt["indices"]))
    os.execve(sys.executable,[sys.executable,str(HERE/"pilot_result.py")],env)


if __name__=="__main__":main()
