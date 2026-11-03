#!/usr/bin/env python3
"""Plot raw GPU telemetry from either thermal report; Python + Matplotlib."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('report',type=Path);a=p.parse_args()
d=json.loads(a.report.read_text());intervals=[]
if 'native' in d:
    samples=d['process']['telemetry'];n=d['native'];t0=min(s['time'] for s in samples)
    intervals.append((n['warmup_start_unix_s'],n['warmup_start_unix_s']+n['warmup_s'],'warmup'))
    intervals += [(r['start_unix_s'],r['start_unix_s']+r['wall_s'],'measured') for r in n['trials']]
    title='RTX 5080 · BEM prepared GPU operator · 1536 matvecs/repeat'
else:
    runs=d['cases'][0]['runs'];samples=[s for r in runs for s in r['telemetry']];t0=min(s['time'] for s in samples)
    intervals=[(min(s['time'] for s in r['telemetry']),max(s['time'] for s in r['telemetry']),'warmup' if r['warmup'] else 'measured') for r in runs]
    title='RTX 5080 · MBS full Mueller curve · 2048 orientations × 2048 azimuths'
fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
for ax,key,label in zip(axes,['utilization.gpu','temperature.gpu','power.draw'],['GPU load (%)','GPU temperature (°C)','GPU power (W)']):
    ax.plot([(s['time']-t0)/60 for s in samples],[s.get(key,float('nan')) for s in samples],lw=1,color='#173f70')
    used=set()
    for start,end,kind in intervals:
        ax.axvspan((start-t0)/60,(end-t0)/60,color='#8cb8e8' if kind=='warmup' else '#efbb67',alpha=.25,label=kind if kind not in used else None);used.add(kind)
    ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0].set_ylim(0,105);axes[0].legend(loc='lower right');axes[0].set_title(title)
axes[-1].set_xlabel('Elapsed time (minutes)')
for suffix in ['png','svg']:
    destination=a.report.with_name('thermal.'+suffix)
    fig.savefig(destination,dpi=160)
    if suffix=='svg':destination.write_text('\n'.join(line.rstrip() for line in destination.read_text().splitlines())+'\n')
