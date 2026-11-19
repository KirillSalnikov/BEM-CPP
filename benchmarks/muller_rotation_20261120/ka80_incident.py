"""Compare emitted, normalized FP64 RHS payloads on the full worker quadrature.

Generate the positions with bin/prism_packet_rhs 7 1.6495721976846451 PREFIX positions
and pass PREFIX.positions.bin. Use --packed for both mirror codes across 12 modes.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
from tmatrix_pipeline import IncidentPreparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('positions', type=Path)
    parser.add_argument('--packed', action='store_true')
    args = parser.parse_args()
    points = np.memmap(args.positions, dtype=np.float64, mode='r').reshape(-1,3)
    assert len(points) == 9864192
    modes = [(p,80,m) for m in range(6 if args.packed else 1) for p in (0,1)]
    indices = list(range(len(modes)))
    key = 'packed' if args.packed else 'single'
    fields = np.empty((2,len(points),6), np.complex128)
    reference = np.empty_like(fields)
    report = dict(points=len(points),mode=key,modes=modes,threads=int(os.environ.get('BEM_INCIDENT_THREADS','1')),runs=[])
    for new in (False,True):
        start = time.perf_counter()
        prep = IncidentPreparation(points,modes,80,fields,cache=new,separable=new)
        setup = time.perf_counter()-start
        repeats = 2 if new else 1
        for j in range(repeats):
            scales, combined, seconds = prep.prepare(prep.key(key,indices,args.packed))
            if not new:
                reference[:]=fields
                old_scales=np.asarray(scales)
                old_combined=combined
            max_error=diff=norm=0.
            for start in range(0,len(points),32768):
                sl=slice(start,start+32768)
                delta=fields[:,sl]-reference[:,sl]
                max_error=max(max_error,float(np.abs(delta).max()))
                diff+=np.vdot(delta,delta).real
                norm+=np.vdot(reference[:,sl],reference[:,sl]).real
            err=float(np.sqrt(diff/norm))
            scale_error=float(np.max(np.abs(scales/old_scales-1)))
            combined_error=float(np.max(np.abs(combined/old_combined-1))) if args.packed else 0.
            record=dict(candidate=new,setup_s=setup,prepare_s=seconds,repeat=j,relative_error=err,
                        max_absolute_error=max_error,scale_error=scale_error,combined_scale_error=combined_error)
            if new:
                cache=prep.cache
                record.update(unique_radii=len(cache.z),unique_theta=len(cache.theta),
                              retained_cache_bytes=sum(a.nbytes for k,a in vars(cache).items()
                                                       if k!='positions' and isinstance(a,np.ndarray)))
            print(record,flush=True)
            report['runs'].append(record)
            (HERE/f'ka80_incident_{key}.json').write_text(json.dumps(report,indent=2)+'\n')
            if not all(np.isfinite(x) and x<2e-12 for x in (err,max_error,scale_error,combined_error)):
                raise RuntimeError('incident-field equivalence failed')
        prep.close()


if __name__=='__main__':
    main()
