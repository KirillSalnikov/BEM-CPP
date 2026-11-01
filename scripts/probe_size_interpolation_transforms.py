#!/usr/bin/env python3
"""Compare size-interpolation coordinates using only saved direct T matrices.

Diagnostic only: never launches a Maxwell solve or changes a production model.
Every transform is inverted before exact SO(3) averaging. There is no projection
onto a lossless/symmetric matrix and no interpolation of the Mueller outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve
from scipy.special import spherical_jn, spherical_yn

from bounded_bem_run import atomic_json
from study_size_interpolation_step import polynomial
from study_tmatrix_cutoff import cutoff_indices, error_metrics
from tmatrix_reference import modes
from tmatrix_size_interpolation import fit_aaa
from tmatrix_so3_mstm import analytic_mueller_tmatrix

ROOT = Path(__file__).resolve().parents[1]


def right_solve(a, b):
    """Return a @ inv(b), without explicitly forming an inverse."""
    return solve(b.T, a.T, check_finite=True).T


class Coordinates:
    def __init__(self, kind, order, center, central_t, radius=1.):
        self.kind, self.center, self.radius = kind, center, radius
        self.degree = np.array([l for p, l, m in modes(order)])
        self.identity = np.eye(len(self.degree), dtype=complex)
        self.s0 = self.identity + 2*central_t

    def factors(self, ka):
        if self.kind == 'power':
            return (ka/self.center)**(self.degree + .5)
        if self.kind.startswith('hankel'):
            def h(k):
                z = k*self.radius
                return np.sqrt(k)*(spherical_jn(self.degree, z) + 1j*spherical_yn(self.degree, z))
            d = h(self.center)/h(ka)
            return d/abs(d) if self.kind == 'hankel_phase' else d
        if self.kind == 'contrast_phase':
            return np.full(len(self.degree), np.exp(1j*(1.3116-1)*self.radius*(ka-self.center)))
        return np.ones(len(self.degree))

    def forward(self, t, ka):
        if self.kind in ('cayley', 'relative_cayley'):
            s = self.identity + 2*t
            base = self.s0 if self.kind == 'relative_cayley' else self.identity
            return right_solve(s-base, s+base)
        d = self.factors(ka)
        return np.ascontiguousarray(t/d[:, None]/d[None, :])

    def inverse(self, value, ka):
        if self.kind in ('cayley', 'relative_cayley'):
            base = self.s0 if self.kind == 'relative_cayley' else self.identity
            return .5*(solve(self.identity-value, (self.identity+value)@base)-self.identity)
        d = self.factors(ka)
        return value*d[:, None]*d[None, :]


def load_case(ka0):
    folder = ROOT/f'benchmarks/interpolation_step_20261031/ka{ka0:g}'
    plan = json.loads((folder/'plan.json').read_text())
    refs = json.loads((folder/'references.json').read_text())
    samples, means = {}, {}
    for r in refs:
        source = Path(r['T'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != r['sha256']:
            raise RuntimeError('reference T hash mismatch')
        samples[r['ka']] = np.load(source)
        with np.load(folder/f'direct_ka{r["ka"]:.8f}/average.npz') as data:
            theta = data['theta_deg']; means[r['ka']] = data['mueller']
    return dict(center=ka0, order=plan['order'], samples=samples, means=means,
                theta=theta, references=refs, plan=plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wide', action='store_true')
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    cases = [load_case(ka0) for ka0 in (3., 6.)]
    # Physical circumscribed radius in equal-volume-radius units.
    aspect = 10/7; rc = 1/np.sqrt(3); height = aspect
    scale = ((4*np.pi/3)/(3*np.sqrt(3)/2*rc*rc*height))**(1/3)
    radius = np.hypot(rc, height/2)*scale
    configurations = [('raw', 1.), ('power', 1.), ('contrast_phase', radius),
                      ('hankel', 1.), ('hankel', radius),
                      ('hankel_phase', radius), ('cayley', 1.), ('relative_cayley', 1.)]
    if args.wide:
        # Common explicit mode mapping, never a leading square slice.
        order = max(c['order'] for c in cases); samples = {}; means = {}; refs = []
        for c in cases:
            index = cutoff_indices(order, c['order'])
            for ka, t in c['samples'].items():
                padded = np.zeros((2*order*(order+2),)*2, complex)
                padded[np.ix_(index, index)] = t; samples[ka] = padded
            means.update(c['means']); refs += c['references']
        cases = [dict(center=4., order=order, samples=samples, means=means,
                      theta=cases[0]['theta'], references=refs,
                      plan={'note': 'ka2-4 L12 zero-padded to L16; ka5-7 L16. No known controls in (4,5).'})]
    definition = dict(configurations=configurations, methods=['polynomial', 'AAA'],
                      new_Maxwell_solves=0, radius=radius, wide=args.wide,
                      tolerance=.01, analytic_SO3=True, independent_training_and_controls=True,
                      metric='max all16 abs(delta Mij)/direct M11; also Csca and Cext',
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(out/'definition.json', definition)
    all_records = []
    for case in cases:
        center = case['center']; samples = case['samples']; theta = case['theta']
        nodes = np.array([2., 2.5, 3., 3.5, 4., 5., 6., 6.5, 7.] if args.wide
                         else [center+x for x in (-1., -.5, 0., .5, 1.)])
        controls = ([2.25, 2.75, 3.25, 3.75, 5.25, 5.5, 5.75, 6.25, 6.75] if args.wide
                    else ((nodes[:-1]+nodes[1:])/2).tolist())
        if set(nodes)&set(controls): raise RuntimeError('validation leakage')
        folder = out/f'ka{center:g}'; folder.mkdir(exist_ok=True)
        atomic_json(folder/'references.json', case['references'])
        atomic_json(folder/'plan.json', dict(nodes=nodes.tolist(), controls=controls, source=case['plan']))
        for kind, r in configurations:
            coords = Coordinates(kind, case['order'], center, samples[center], r)
            start = time.perf_counter()
            values = [np.ascontiguousarray(coords.forward(samples[k], k)) for k in nodes]
            roundtrip = max(float(np.linalg.norm(coords.inverse(v,k)-samples[k])/np.linalg.norm(samples[k]))
                            for k,v in zip(nodes, values))
            if roundtrip > 1e-9: raise RuntimeError(f'inexact coordinate transform: {kind} {roundtrip}')
            coordinate_seconds = time.perf_counter()-start
            for method in ('polynomial', 'AAA'):
                name = f'{kind}_r{r:.6f}_{method}'; target = folder/name; target.mkdir(exist_ok=True)
                start = time.perf_counter(); info = {}
                if method == 'AAA': model, info = fit_aaa(nodes, values, tolerance=1e-6)
                fit_seconds = time.perf_counter()-start
                def evaluate(k):
                    v = model.evaluate(k) if method == 'AAA' else polynomial(nodes, values, k)
                    return coords.inverse(v, k)
                checks = []
                for k in controls:
                    pred = evaluate(k)
                    mean, meta = analytic_mueller_tmatrix(pred, case['order'], theta, precision='fp64', threads=2, cyclic_order=1)
                    assert meta['Euler_quadrature_nodes'] == 0
                    direct = samples[k]
                    csca = float(abs(np.vdot(pred,pred).real/np.vdot(direct,direct).real-1))
                    cext = float(abs(np.trace(pred).real/np.trace(direct).real-1))
                    check = dict(ka=k, **error_metrics(mean, case['means'][k], theta),
                                 T_relative_error=float(np.linalg.norm(pred-direct)/np.linalg.norm(direct)),
                                 Csca_relative_error=csca, Cext_relative_error=cext)
                    check['acceptance_error'] = max(check['maximum_all16_error_over_reference_M11'], csca, cext)
                    checks.append(check)
                    np.savez_compressed(target/f'ka{k:.8f}.npz', theta_deg=theta, predicted=mean, direct=case['means'][k])
                record = dict(center=center, name=name, kind=kind, radius=r, method=method,
                              nodes=nodes.tolist(), checks=checks, roundtrip_error=roundtrip,
                              coordinate_seconds=coordinate_seconds, fit_seconds=fit_seconds, diagnostics=info,
                              maximum_error=max(c['acceptance_error'] for c in checks),
                              maximum_mueller_error=max(c['maximum_all16_error_over_reference_M11'] for c in checks))
                atomic_json(target/'report.json', record); all_records.append(record)
                atomic_json(out/'progress.json', all_records)
                print(json.dumps(dict(center=center, name=name, maximum_percent=100*record['maximum_error'])), flush=True)
    atomic_json(out/'report.json', dict(definition=definition, records=all_records, status='complete'))


if __name__ == '__main__':
    main()
