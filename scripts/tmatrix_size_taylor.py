"""Local T(k) continuation from differentiated BEM equations at one k0.

All series contain derivative/j!. A single LU factorization is reused for
the moment equations. This requires A/B/radiation derivatives and currents;
a saved far-field T0 alone is not sufficient. No off-center T is fitted.
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve


def moments(a, b):
    if len(a)!=len(b) or a.ndim!=3 or b.ndim!=3 or a.shape[1]!=a.shape[2] or b.shape[1]!=a.shape[1]:
        raise ValueError('incompatible operator/RHS Taylor coefficients')
    factor=lu_factor(a[0],check_finite=True)
    x=[]
    for j in range(len(a)):
        rhs=b[j].copy()
        for i in range(1,j+1):rhs-=a[i]@x[j-i]
        x.append(lu_solve(factor,rhs,check_finite=True))
    return np.array(x),factor


def polynomial(coefficients, delta):
    out=np.zeros_like(coefficients[0])
    for coefficient in coefficients[::-1]:out=delta*out+coefficient
    return out


def radiation_moments(d, x, ka, phase):
    kernel=[];tmatrix=[]
    for j in range(len(x)):
        kernel.append(sum(d[i]@x[j-i] for i in range(j+1)))
        value=ka*kernel[j]+(kernel[j-1] if j else 0)
        tmatrix.append(-1j/(4*np.pi)*phase[:,None]*value*phase[None,:].conj())
    return np.array(tmatrix)


def matrix_pade(coefficients, denominator_degree=2, scale=.5):
    """One scalar denominator for every complex T entry; moments only.

    Least-squares matching preserves linear symmetry relations. Denominator
    coefficients are solved by SVD, not normal equations or elementwise Padé.
    """
    n=len(coefficients)-1;m=denominator_degree;p=n-m
    if m<1 or p<0 or scale<=0:raise ValueError('invalid Pade degrees/scale')
    t=np.array([v*scale**i for i,v in enumerate(coefficients)])
    left=np.vstack([np.column_stack([t[k-j].reshape(-1) for j in range(1,m+1)]) for k in range(p+1,n+1)])
    right=-np.concatenate([t[k].reshape(-1) for k in range(p+1,n+1)])
    solution,_,rank,singular=np.linalg.lstsq(left,right,rcond=1e-12)
    q=np.r_[1+0j,solution]
    numerator=np.array([sum(q[j]*t[k-j] for j in range(min(k,m)+1)) for k in range(p+1)])
    info=dict(numerator_degree=p,denominator_degree=m,rank=int(rank),scale=scale,
              relative_moment_mismatch=float(np.linalg.norm(left@solution-right)/max(np.linalg.norm(right),1e-300)),
              denominator_real=q.real.tolist(),denominator_imag=q.imag.tolist())
    return numerator,q,info


def evaluate_pade(model,delta):
    numerator,q,info=model;z=delta/info['scale']
    denominator=np.polynomial.polynomial.polyval(z,q)
    bound=np.sum(abs(q)*abs(z)**np.arange(len(q)))
    if abs(denominator)<1e-10*max(bound,1):raise ValueError('Pade denominator near a pole')
    return polynomial(numerator,z)/denominator


def pade_interval_guard(model, interval, relative_clearance=.05, denominator_floor=1e-8):
    """Conservative diagnostic over a real ka segment, not just its endpoint.

    Roots are model poles, not evidence of physical resonances. A small
    numerator at a root flags a possible removable/Froissart pole; we still
    reject a nearby pole instead of cancelling it without physical controls.
    Stationary points of |q(x)|^2 catch interior denominator minima. This is a
    floating-point diagnostic, not an interval-arithmetic certificate.
    """
    numerator, q, info = model
    q = np.asarray(q, complex)
    numerator = np.asarray(numerator, complex)
    scale = float(info['scale'])
    ends = np.asarray(interval, float)
    if (ends.shape != (2,) or not np.isfinite(ends).all() or ends[0] > ends[1]
            or not np.isfinite(scale) or scale <= 0 or not 0 <= relative_clearance < 1
            or not 0 < denominator_floor < 1 or q.ndim != 1 or len(q) < 2
            or not np.isfinite(q).all() or not np.isfinite(numerator).all() or q[0] != 1):
        raise ValueError('invalid Pade interval/model')
    zlo, zhi = ends / scale
    roots = np.polynomial.polynomial.polyroots(q)
    if not np.isfinite(roots).all():
        raise ValueError('nonfinite Pade poles')
    roots_info = []
    for root in roots:
        delta = scale * root
        nearest = np.clip(delta.real, ends[0], ends[1])
        distance = float(abs(delta - nearest))
        bound = sum(np.linalg.norm(value) * abs(root)**j for j, value in enumerate(numerator))
        numerator_ratio = float(np.linalg.norm(polynomial(numerator, root)) / max(bound, 1e-300))
        roots_info.append(dict(delta_real=float(delta.real), delta_imag=float(delta.imag),
                               distance_to_segment=distance, numerator_relative_value=numerator_ratio,
                               possible_removable_pole=bool(numerator_ratio < 1e-8)))
    # q(x) conjugate(q(x)) is real on the real parameter axis.
    squared = np.polynomial.polynomial.polymul(q, q.conj()).real
    critical = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(squared))
    locations = [float(zlo), float(zhi)]
    for root in critical:
        if abs(root.imag) <= 1e-7 * max(1., abs(root.real)) and zlo <= root.real <= zhi:
            locations.append(float(root.real))
    values = [float(abs(np.polynomial.polynomial.polyval(x, q))) for x in locations]
    bound = float(sum(abs(q[j]) * max(abs(zlo), abs(zhi))**j for j in range(len(q))))
    minimum = min(values)
    clearance = relative_clearance * max(abs(ends[0]), abs(ends[1]), ends[1] - ends[0])
    reasons = []
    if info.get('rank') != len(q) - 1:
        reasons.append('rank_deficient_denominator_fit')
    if any(p['distance_to_segment'] <= clearance for p in roots_info):
        reasons.append('model_pole_near_segment')
    if minimum <= denominator_floor * max(bound, 1.):
        reasons.append('small_denominator_inside_segment')
    return dict(passed=not reasons, reasons=reasons, interval=ends.tolist(), poles=roots_info,
                relative_clearance=relative_clearance, clearance=clearance,
                minimum_denominator_abs=minimum, denominator_bound=bound,
                normalized_minimum=minimum / max(bound, 1.),
                checked_delta_locations=[scale * x for x in locations],
                scope='Numerical model-pole diagnostic; independent physical controls remain required.')


def evaluate_pade_checked(model, delta, interval=None):
    interval = (min(0., delta), max(0., delta)) if interval is None else interval
    if not interval[0] <= delta <= interval[1]:
        raise ValueError('Pade target outside checked segment')
    report = pade_interval_guard(model, interval)
    if not report['passed']:
        raise ValueError('Pade segment rejected: ' + ', '.join(report['reasons']))
    return evaluate_pade(model, delta)
