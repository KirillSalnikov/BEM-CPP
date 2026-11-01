#!/usr/bin/env python3
"""Small-prism aspect study: analytic SO(3), modal tails, and mesh changes.

Mesh differences are convergence diagnostics, not absolute-error certificates.
The modal envelope uses the circumscribing radius, while ka uses equal volume.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from run_large_tmatrix import solver_profile
from study_tmatrix_cutoff import analyze, atomic_json, error_metrics, stable_minimum, cutoff_indices
from tmatrix_so3_mstm import analytic_mueller_tmatrix

ROOT = Path(__file__).resolve().parents[1]


def geometry(aspect):
    width = (8*math.pi/(3*math.sqrt(3)*aspect))**(1/3)
    return {'height_over_width': aspect, 'width_over_a': width,
            'height_over_a': aspect*width,
            'circumscribing_radius_over_a': width*math.sqrt(1/3+aspect**2/4)}


def case_name(ka, aspect):
    return f'ka{ka:g}_aspect{aspect:g}'.replace('.', 'p')


def execute(case, ref, out, packed=True):
    suffix = '' if packed else '_unpacked'
    run = out/case['name']/f'ref{ref}_L{case["reference_order"]}{suffix}'
    run.mkdir(parents=True, exist_ok=True)
    summary_path = run/'summary.json'
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        if s.get('complete'):
            p, op = s['parameters'], s['operator']
            cached_aspect = p.get('prism_aspect', op.get('prism_height_to_width_across_flats', 1.))
            matches = (p['shape']=='prism' and p['ref']==ref
                       and p['order']==case['reference_order']
                       and abs(p['ka']-case['ka'])<1e-12
                       and abs(p['ri']-1.3116)<1e-12 and p.get('ri_imag',0.)==0.
                       and abs(cached_aspect-case['height_over_width'])<1e-12
                       and op['prism_dihedral_mesh'] and op['duffy_order']==6
                       and abs(op['radius']-case['circumscribing_radius_over_a'])<1e-10)
            if not matches:
                raise RuntimeError(f'cached solution has different physics/mesh: {run}')
            return run, s
    if packed and any(not json.loads(p.read_text())['passed']
                      for p in (run/'c6_audits').glob('*.json')):
        return execute(case, ref, out, packed=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith('BEM_')}
    env.update(solver_profile(ref))
    # Long prisms add vertical cells: ref alone no longer predicts tree depth.
    # Match regular_prism's D6h cell counts and the worker's depth estimator.
    segments = 2**ref
    z_segments = max(segments, math.ceil(math.sqrt(3)*case['height_over_width']*segments))
    z_segments += z_segments % 2
    triangles = 12*segments*(z_segments+segments)
    ratio = 2*7*triangles/int(env['BEM_TMATRIX_MAX_LEAF'])
    predicted_depth = min(6, max(1, math.ceil(math.log(ratio, 8))))
    env['BEM_FMM_BANDED_SPLIT_DEPTH'] = str(max(1, predicted_depth-2))
    env.update(OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='1',
               BEM_MULLER_GPU_ASSEMBLY='0', BEM_FMM_MULTILEVEL_GPU_ASSEMBLY='0',
               BEM_HOST_GMRES_STRICT_INTERVAL='2',
               BEM_TMATRIX_SECTOR_GMRES='1', BEM_TMATRIX_ORBIT_NODES='50',
               BEM_TMATRIX_VERTICAL_PRECONDITIONER='1',
               BEM_TMATRIX_SECTOR_RECYCLE_RANK='8')
    command = [sys.executable, str(ROOT/'scripts/build_fmm_tmatrix.py'),
               '--shape', 'prism', '--ka', str(case['ka']), '--ri', '1.3116',
               '--wavelength', '.532', '--prism-aspect', str(case['height_over_width']),
               '--ref', str(ref), '--order', str(case['reference_order']),
               '--tol', '2e-9', '--restart', '64', '--max-iters', '800',
               '--c2-reuse', '--fast-mlfma',
               '--incident-prefetch', '--ntheta', '1801', '--so3-precision', 'fp64',
               '--so3-threads', '4', '--out', str(run)]
    if packed:
        command += ['--c6-pack', '--mirror-pack']
    atomic_json(run/'launch.json', {'command': command,
                'environment': {k:v for k,v in env.items() if k.startswith('BEM_') or k.endswith('NUM_THREADS')},
                'started_unix': time.time()})
    print(json.dumps({'event': 'start', 'case': case['name'], 'ref': ref,
                      'order': case['reference_order']}), flush=True)
    with (run/'driver.log').open('a') as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    if process.returncode:
        if packed and 'C6 packed RHS audit failed' in (run/'driver.log').read_text():
            print(json.dumps({'event': 'packing_audit_failed_fallback',
                              'case': case['name'], 'ref': ref}), flush=True)
            return execute(case, ref, out, packed=False)
        raise subprocess.CalledProcessError(process.returncode, command)
    s = json.loads(summary_path.read_text())
    if not s.get('complete'):
        raise RuntimeError(f'incomplete T matrix: {run}')
    print(json.dumps({'event': 'complete', 'case': case['name'], 'ref': ref,
                      'wall_s': s.get('end_to_end_s')}), flush=True)
    return run, s


def average(run, order=None):
    data = json.loads((run/'average.json').read_text())
    # Keep the stored, absolute Mueller normalization for mesh comparisons.
    theta = np.asarray(data['theta'])
    if order is None or order == data['order']:
        return theta, np.moveaxis(np.asarray(data['mueller']), -1, 0)
    indices = cutoff_indices(data['order'], order)
    t = np.load(run/'tmatrix.npy', mmap_mode='r')[np.ix_(indices, indices)]
    mean, _ = analytic_mueller_tmatrix(t, order, theta, precision='fp64',
                                       threads=4, cyclic_order=6)
    return theta, mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ka', type=float, nargs='+', default=[2., 5.])
    parser.add_argument('--aspects', type=float, nargs='+', default=[.2, 1., 5.])
    parser.add_argument('--max-ref', type=int, default=4)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    cases = []
    for ka in args.ka:
        for aspect in args.aspects:
            g = geometry(aspect)
            kr = ka*g['circumscribing_radius_over_a']
            cases.append({'name': case_name(ka, aspect), 'ka': ka, **g,
                          'k_circumscribing_radius': kr,
                          'reference_order': math.ceil(kr+4*kr**(1/3)+4),
                          'envelope_order': math.ceil(kr+4*kr**(1/3)+4)})
    report = {'ri': 1.3116, 'wavelength_um': .532, 'shape': 'hexagonal prism',
              'ka_definition': 'equal-volume sphere radius', 'cases': cases,
              'metric': 'max_theta_ij abs(delta_Mij)/M11_finer',
              'limitations': ['Mesh differences are not absolute errors.',
                              'Cutoff errors are relative to the finite reference L.',
                              'Analytic uniform SO(3); theta is the output scattering angle.']}
    atomic_json(out/'plan.json', report)
    for case in cases:
        runs = []
        case['mesh_runs'] = []
        case['mesh_comparisons'] = []
        for ref in range(2, args.max_ref+1):
            run, summary = execute(case, ref, out)
            runs.append(run)
            case['mesh_runs'].append({'ref': ref, 'run': str(run),
                    'order': summary['parameters']['order'],
                    'wall_s': summary.get('end_to_end_s'),
                    'system_dofs': summary['operator']['system_dofs'],
                    'maximum_strict_residual': summary.get('maximum_strict_residual'),
                    'new_direction_field_error': summary.get('new_direction_field_error'),
                    'energy_balance_relative_error': abs(summary['extinction_cross_section']/summary['scattering_cross_section']-1),
                    'analytic_reciprocity_defect': summary['analytic_so3']['reciprocity_relative_defect']})
            if len(runs)>1:
                common_L = min(r['order'] for r in case['mesh_runs'][-2:])
                theta, fine = average(runs[-1], common_L)
                coarse_theta, coarse = average(runs[-2], common_L)
                if not np.array_equal(theta, coarse_theta):
                    raise RuntimeError('mesh output-angle grids differ')
                comparison = {'coarse_ref': ref-1, 'fine_ref': ref, 'common_order': common_L,
                              **error_metrics(coarse, fine, theta)}
                case['mesh_comparisons'].append(comparison)
                print(json.dumps({'event': 'mesh_comparison', 'case': case['name'],
                                  **comparison}), flush=True)
                atomic_json(out/'progress.json', report)
                if comparison['maximum_all16_error_over_reference_M11'] < .01:
                    break
            if ref == 2:
                pilot = analyze(run, out/case['name']/'cutoff_coarse', ntheta=1801, threads=4)
                selected = min(case['envelope_order'], stable_minimum(pilot['records'], 1e-4)+2)
                # Preserve completed finer results on resume.
                existing = [json.loads(p.read_text()) for p in (out/case['name']).glob('ref3_L*/summary.json')]
                completed_orders = [s['parameters']['order'] for s in existing if s.get('complete')]
                if completed_orders:
                    selected = max(completed_orders)
                case['reference_order'] = selected
                case['coarse_tail_at_refined_order'] = next(r['acceptance_error'] for r in pilot['records'] if r['order']==selected)
                print(json.dumps({'event': 'refined_mesh_order', 'case': case['name'],
                                  'order': selected, 'coarse_tail': case['coarse_tail_at_refined_order']}), flush=True)
        cutoff = analyze(runs[-1], out/case['name']/'cutoff', ntheta=1801, threads=4)
        case['cutoff_minima'] = cutoff['minima_relative_to_reference']
        case['cutoff_reference_run'] = str(runs[-1])
        atomic_json(out/'progress.json', report)
    atomic_json(out/'summary.json', report)
    print(json.dumps({'event': 'finished', 'summary': str(out/'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
