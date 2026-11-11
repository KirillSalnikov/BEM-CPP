"""Read-only survey of independently stored +/-m columns, including C6 packets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

B=Path(__file__).resolve().parent;R=B.parents[1]
sys.path.insert(0,str(R/'scripts'))
from tmatrix_reference import modes
from tmatrix_symmetry import vertical_mirror_plan,c2x_partner_column

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path,default=B/'saved_pairs_audit.json');a=p.parse_args()
    manifest=json.loads((a.source/'manifest.json').read_text());order=manifest['parameters']['order']
    modal=modes(order);flip,sign,_,_=vertical_mirror_plan(order);rows=[]
    for i,mode in enumerate(modal):
        if mode[2]<=0:continue
        j=int(flip[i]);paths=[a.source/'columns'/f'{index:06d}.npy' for index in (i,j)]
        if not all(path.exists() and path.with_suffix('.json').exists() for path in paths):continue
        metadata=[json.loads(path.with_suffix('.json').read_text()) for path in paths]
        if any(d.get('derived_by') not in (None,'C6_packed_RHS') for d in metadata):continue
        x,y=[np.load(path,allow_pickle=False) for path in paths]
        error=float(np.linalg.norm(c2x_partner_column(x,i,flip,sign)-y)/max(np.linalg.norm(y),1e-300))
        rows.append(dict(source=i,target=j,mode=mode,relative_error=error))
    errors=[r['relative_error'] for r in rows]
    summary=dict(pairs=len(rows),max_error=max(errors),quantiles=np.quantile(errors,[.5,.9,.99]).tolist(),
        worst=sorted(rows,key=lambda r:-r['relative_error'])[:10],source=str(a.source.resolve()),pairs_detail=rows)
    a.out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pairs_detail','worst')},indent=2))

if __name__=='__main__':main()
