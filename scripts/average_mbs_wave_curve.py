#!/usr/bin/env python3
"""Numerical SO(3) audit of near-forward M11 with a scalar wave correction.

Uses the exact D6h geometry symmetry to fold incident directions, and explicit
uniform outgoing azimuth integration for the remaining Euler rotation. Mueller
intensities are averaged AFTER coherent Jones addition, never amplitudes.
"""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/d6h_20261026/mbs_wave_correction/orientation_curve'
DIAG = ROOT / 'benchmarks/d6h_20261026/forward_diagnosis'
sys.path.insert(0, str(ROOT / 'experiments/MBS-fast-wave'))
from waveoptics.prism import Parameters, exit_field_difference

KA = 20.
THETA = np.linspace(0, 30, 61)
NPHI = 64
PHI = np.arange(NPHI)*360/NPHI
ENV = {k: v for k, v in os.environ.items() if not k.startswith(('MBS_', 'BEM_', 'ADDA_'))}
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
COMMAND = json.loads((OUT.parent / 'ka20_cuda_gamma30/command.json').read_text())


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def folded_quadrature(label):
    data = np.load(DIAG / f'forward_quadrature_{label}.npz')
    folded = {}
    for direction, weight in zip(data['directions'], data['weights']):
        beta = float(np.degrees(np.arccos(direction[2])))
        gamma = float((210-np.degrees(np.arctan2(direction[1], direction[0]))) % 60)
        key = (round(min(beta, 180-beta), 10), round(min(gamma, 60-gamma), 10))
        folded[key] = folded.get(key, 0.)+float(weight)
    np.testing.assert_allclose(sum(folded.values()), 1, atol=1e-12)
    return folded


def case_path(key):
    return OUT / 'cases' / f'b{key[0]:.10f}_g{key[1]:.10f}'


def native(key):
    case = case_path(key)
    case.mkdir(parents=True, exist_ok=True)
    cached = case / 'native.npz'
    if cached.exists():
        return key, np.load(cached)['jones']
    command = COMMAND.copy()
    position = command.index('--fixed-orientation')
    command[position+1:position+3] = [str(key[0]), str(key[1])]
    for flag, value in [('--phi-points', str(NPHI)), ('--theta-grid-file', str(OUT/'theta.txt')),
                        ('--max-reflections', '14'), ('--threads', '1'), ('--output', str(case/'native'))]:
        command[command.index(flag)+1] = value
    atomic_json(case/'command.json', command)
    with (case/'native.log').open('w') as log:
        result = subprocess.run(command, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'MBS failed in {case}')
    table = np.loadtxt(case/'native/native_jones.dat', skiprows=1)
    np.testing.assert_allclose(table[:, 0], np.repeat(THETA, NPHI), atol=1e-10)
    np.testing.assert_allclose(table[:, 1], np.tile(PHI, len(THETA)), atol=1e-10)
    jones = (table[:, 2::2]+1j*table[:, 3::2]).reshape(len(THETA), NPHI, 2, 2)
    mueller = np.loadtxt(case/'native/native.dat', skiprows=1)[:, 3].reshape(len(THETA), NPHI)
    np.testing.assert_allclose(np.sum(abs(jones)**2, axis=(-2, -1))/2, mueller, rtol=1e-11, atol=1e-10)
    np.savez_compressed(cached, theta_deg=THETA, phi_deg=PHI, jones=jones)
    # Text output is reproducible from command.json and is much larger than the checkpoint.
    (case/'native/native_jones.dat').unlink()
    (case/'native/native.dat').unlink()
    return key, jones


def radiation_tables():
    x = (np.arange(256)-128+.5)*8/256
    theta = np.deg2rad(np.repeat(THETA, NPHI))
    phi = np.deg2rad(np.tile(PHI, len(THETA)))
    px = np.exp(-1j*KA*np.sin(theta[:, None])*np.cos(phi[:, None])*x[None, :]).astype(np.complex64)
    py = np.exp(-1j*KA*np.sin(theta[:, None])*np.sin(phi[:, None])*x[None, :]).astype(np.complex64)
    # Native Jones output has lab incident axes and scattering-plane outgoing axes.
    # A scalar Huygens wave is delta_f * R(phi), not delta_f * I for phi != 0.
    # At exactly theta=0 the native code deliberately uses fixed lab axes.
    polar = np.zeros((len(theta), 2, 2))
    angle = np.where(theta == 0, 0., phi)
    polar[:, 0, 0] = polar[:, 1, 1] = np.cos(angle)
    polar[:, 0, 1] = -np.sin(angle)
    polar[:, 1, 0] = np.sin(angle)
    return theta, px, py, polar.reshape(len(THETA), NPHI, 2, 2)


def correct(key, jones, tables):
    case = case_path(key)
    checkpoint = case/'corrected.npz'
    if checkpoint.exists():
        return
    theta, px, py, polar = tables
    started = time.perf_counter()
    p = Parameters(KA, beta_deg=key[0], gamma_deg=key[1], precision='fp32')
    _, field, metadata = exit_field_difference(p, backend='cuda')
    integral = np.sum((py @ field)*px, axis=1, dtype=np.complex128)*(8/256)**2
    phase = np.exp(.5j*KA*metadata['propagation_length']*(1-np.cos(theta)))
    delta = (-1j*KA/(2*np.pi)*(1+np.cos(theta))/2*phase*integral).reshape(len(THETA), NPHI)
    corrected = jones+delta[:, :, None, None]*polar
    baseline_m11 = KA**2/2*np.sum(abs(jones)**2, axis=(-2, -1))
    corrected_m11 = KA**2/2*np.sum(abs(corrected)**2, axis=(-2, -1))
    np.testing.assert_allclose(corrected_m11[0], corrected_m11[0, 0], rtol=1e-9)
    np.savez_compressed(checkpoint, theta_deg=THETA, phi_deg=PHI, delta_f=delta,
                        baseline_M11=baseline_m11, corrected_M11=corrected_m11)
    atomic_json(case/'wave.json', {**metadata, 'radiation_and_save_seconds': time.perf_counter()-started})


def aggregate(label, quadrature):
    base = np.zeros((len(THETA), NPHI))
    corrected = np.zeros_like(base)
    for key, weight in quadrature.items():
        case = np.load(case_path(key)/'corrected.npz')
        base += weight*case['baseline_M11']
        corrected += weight*case['corrected_M11']
    np.savez_compressed(OUT/f'mean_{label}.npz', theta_deg=THETA,
                        baseline_M11=base.mean(axis=1), corrected_M11=corrected.mean(axis=1),
                        baseline_phi32_M11=base[:, ::2].mean(axis=1),
                        corrected_phi32_M11=corrected[:, ::2].mean(axis=1),
                        baseline_phi16_M11=base[:, ::4].mean(axis=1),
                        corrected_phi16_M11=corrected[:, ::4].mean(axis=1))
    return {'label': label, 'folded_directions': len(quadrature), 'phi_points': NPHI,
            'baseline_forward': float(base[0].mean()), 'corrected_forward': float(corrected[0].mean()),
            'max_phi32_vs_64_relative_change': float(np.max(abs(corrected[:, ::2].mean(axis=1)/corrected.mean(axis=1)-1)))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT/'theta.txt', THETA)
    started = time.perf_counter()
    tables = radiation_tables()
    reports = []
    for label in ['45_16', '89_30']:
        quad = folded_quadrature(label)
        tasks = [key for key in quad if not (case_path(key)/'corrected.npz').exists()]
        with cf.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(native, key) for key in tasks]
            for count, future in enumerate(cf.as_completed(futures), 1):
                key, jones = future.result()
                correct(key, jones, tables)
                if count % 25 == 0 or count == len(tasks):
                    status = {'complete': False, 'stage': label, 'stage_completed': len(quad)-len(tasks)+count,
                              'stage_total': len(quad), 'elapsed_seconds': time.perf_counter()-started}
                    atomic_json(OUT/'progress.json', status)
                    print(json.dumps(status), flush=True)
        report = aggregate(label, quad)
        reports.append(report)
        atomic_json(OUT/'quadrature.json', {'scope': 'numerical SO(3) M11 in 0..30 degrees',
                                           'reflection_depth': 14, 'geometry_symmetry': 'D6h', 'rows': reports})
        print(json.dumps(report), flush=True)
    # Verify actual native and corrected azimuth means under both mirror operations
    # and a 60-degree body rotation, independently of the quadrature folding.
    symmetry = [(37., 17.), (143., 17.), (37., -17.), (37., 77.)]
    for key in symmetry:
        _, jones = native(key)
        correct(key, jones, tables)
    reference = np.load(case_path(symmetry[0])/'corrected.npz')
    symmetry_errors = {}
    for key in symmetry[1:]:
        other = np.load(case_path(key)/'corrected.npz')
        symmetry_errors[str(key)] = {name: float(np.max(abs(other[name].mean(axis=1)/reference[name].mean(axis=1)-1)))
                                    for name in ['baseline_M11', 'corrected_M11']}
    atomic_json(OUT/'symmetry_validation.json', symmetry_errors)
    atomic_json(OUT/'progress.json', {'complete': True, 'elapsed_seconds': time.perf_counter()-started,
                                      'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print('complete', time.perf_counter()-started, symmetry_errors, flush=True)


if __name__ == '__main__':
    main()
