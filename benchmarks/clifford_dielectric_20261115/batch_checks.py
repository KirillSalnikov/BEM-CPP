"""Full-space FP64 mirror audits, completed before writing any partner column."""
import time
import numpy as np


def audit_mirror_batch(d,mirror,requests,tol):
    """requests: (source index, partner index, partner mode, sign, incident, h)."""
    if not requests:return {}
    if len(requests)>16 or not 0<tol<1:raise ValueError('mirror audit batch controls')
    incidents=[];densities=[];metadata=[]
    for source,partner,mode,sign,f,h in requests:
        t=time.perf_counter();pf=d.modal_incident(mode);rhs_s=time.perf_counter()-t
        t=time.perf_counter();ph=sign*mirror.apply(h)
        norm=np.linalg.norm(pf)
        lift_error=float(np.linalg.norm(sign*mirror.apply(f)-pf)/norm) if norm else float('inf')
        if not np.isfinite(lift_error) or lift_error>2e-11:raise RuntimeError('mirror incident transformation failed')
        incidents.append(pf);densities.append(ph)
        metadata.append(dict(seconds=time.perf_counter()-t,iterations=0,info=0,
            history=[],mirror_from=source,mirror_lift_error=lift_error,rhs_s=rhs_s,
            far_projection_s=0.,reference_batch_size=len(requests),
            residual_scope=('full-space strict-epsilon FMM mirror audit plus exact near corrections; direct quadrature audited' if getattr(d,'_matrix_free',False) else
                'independent full-space FP64 mirror audit; orbit reconstruction separately audited')))
    t=time.perf_counter();packed=None
    if d.reference_checks=='packed':
        from packed_reference import apply_packed
        g=d._compact_geometry
        if g is None:raise ValueError('packed mirror checks require audited cyclic geometry')
        modes=[request[2] for request in requests];reduced=g.restrict(np.asarray(densities))
        for x,y,mode in zip(reduced,densities,modes):
            error=np.linalg.norm(g.expand_mode(x,mode)-y)/max(np.linalg.norm(y),1e-300)
            if not np.isfinite(error) or error>2e-11:raise RuntimeError('mirror density is outside its claimed sector')
        values,packed=apply_packed(g,reduced,modes)
        checked=np.asarray([g.expand_mode(v,mode) for v,mode in zip(values,modes)])
    else:checked=d.apply_reference_batch(np.asarray(densities))
    elapsed=time.perf_counter()-t
    result={}
    for (source,partner,*_),pf,value,stats in zip(requests,incidents,checked,metadata):
        t=time.perf_counter();rhs=(pf.reshape(-1,8)*(2*d.scaling[2])).ravel()
        norm=np.linalg.norm(rhs);defect=np.linalg.norm(value-rhs)
        residual=float(defect/norm) if norm else (0. if defect==0 else float('inf'))
        if not np.isfinite(residual) or residual>1.01*tol:raise RuntimeError(f'full FP64 mirror residual failed at column {partner}: {residual}')
        stats.update(residual=residual,strict_history=[residual],full_reference_s=elapsed/len(requests))
        if packed is not None:stats.update(packed_reference=packed,residual_scope=packed['scope'])
        stats['seconds']+=time.perf_counter()-t+elapsed/len(requests)
        result[source]=(partner,stats)
    return result
