#!/usr/bin/env python3
"""Compare the actual MBS external splitter with complex-interface Fresnel coefficients.

This CPU diagnostic does not solve a finite particle or validate diffraction.
The caller supplies the same frozen sources used for its MBS executable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresnel(m, theta):
    cosine = np.cos(theta)
    q = np.sqrt(m*m-np.sin(theta)**2+0j)
    q = np.where((q.imag < 0) | ((q.imag == 0) & (q.real < 0)), -q, q)
    return (m*m*cosine-q)/(m*m*cosine+q), (cosine-q)/(cosine+q)


def mueller(p, s):
    result = np.zeros((len(p), 4, 4))
    pp, ss = abs(p)**2, abs(s)**2
    cross = p*np.conj(s)
    result[:, 0, 0] = result[:, 1, 1] = .5*(pp+ss)
    result[:, 0, 1] = result[:, 1, 0] = .5*(pp-ss)
    result[:, 2, 2] = result[:, 3, 3] = cross.real
    result[:, 2, 3] = cross.imag
    result[:, 3, 2] = -cross.imag
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mbs-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index-real', type=float, default=1.3116)
    parser.add_argument('--index-imag', type=float, default=10.)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = args.mbs_root.resolve()/'src'
    files = ['Splitting.cpp', 'Beam.cpp', 'geometry/Polygon.cpp', 'geometry/geometry_lib.cpp',
             'geometry/intrinsic/intrinsics.cpp', 'math/compl.cpp', 'math/JonesMatrix.cpp',
             'math/matrix.cpp', 'math/PhysMtr.cpp', 'common/common.cpp',
             'common/Matrix4x4.cpp', 'common/MullerMatrix.cpp']
    sources = [ROOT/'tools/mbs_fresnel_probe.cpp']+[src/p for p in files]+sorted((src/'bigint').glob('*.cc'))
    includes = ['', 'math', 'geometry', 'geometry/intrinsic', 'geometry/sse',
                'particle', 'scattering', 'common', 'bigint']
    command = ['g++', '-O2', '-std=gnu++11', '-march=native', '-fopenmp',
               '-ffunction-sections', '-fdata-sections']+list(map(str, sources))
    command += ['-I'+str(src/p) for p in includes]+['-Wl,--gc-sections', '-o', str(out/'mbs_fresnel_probe')]
    with (out/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    commit = subprocess.check_output(['git', '-C', str(src.parent), 'rev-parse', 'HEAD'], text=True).strip()
    receipt = dict(command=command, mbs_commit=commit,
        source_sha256={str(p): digest(p) for p in sources},
        binary_sha256=digest(out/'mbs_fresnel_probe'), gpu_used=False)
    (out/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
    reports = []
    for loss in sorted({0., 1., args.index_imag}):
        path = out/f'fresnel_kappa{loss:g}.csv'
        with path.open('w') as stream:
            subprocess.run([str(out/'mbs_fresnel_probe'), str(args.index_real), str(loss)],
                           stdout=stream, check=True, env=dict(os.environ, OMP_NUM_THREADS='1'))
        values = np.loadtxt(path, delimiter=',', skiprows=1)
        theta = np.radians(values[:, 0])
        rp, rs = values[:, 1]+1j*values[:, 2], values[:, 3]+1j*values[:, 4]
        ep, es = fresnel(complex(args.index_real, loss), theta)
        candidate, reference = mueller(rp, rs), mueller(ep, es)
        errors = abs(candidate-reference)/reference[:, 0, 0, None, None]
        worst = np.unravel_index(errors.argmax(), errors.shape)
        direction_error = np.max(np.hypot(values[:, 5]-np.sin(theta), values[:, 6]+np.cos(theta)))
        normal_error = max(abs(rp[0]-ep[0]), abs(rs[0]-es[0]))
        phase = np.degrees(np.angle((rp/rs)/(ep/es)))
        report = dict(index_real=args.index_real, index_imag=loss,
            normal_reflectance_exact=float(abs(ep[0])**2),
            normal_coefficient_absolute_error=float(normal_error),
            maximum_specular_direction_vector_error=float(direction_error),
            maximum_absolute_complex_coefficient_error=float(max(max(abs(rp-ep)), max(abs(rs-es)))),
            maximum_Mueller_error_over_exact_M11=float(errors[worst]),
            worst_component=f'M{worst[1]+1}{worst[2]+1}', worst_angle_deg=float(values[worst[0], 0]),
            maximum_relative_p_s_phase_error_deg=float(max(abs(phase))), selected_angles=[])
        for deg in [0., 30., 45., 60., 75., 80., 85., 89.]:
            i = int(round(deg*10))
            report['selected_angles'].append(dict(angle_deg=deg,
                exact_Rp=float(abs(ep[i])**2), mbs_Rp=float(abs(rp[i])**2),
                Rp_relative_error_percent=float(100*(abs(rp[i]/ep[i])**2-1)),
                exact_Rs=float(abs(es[i])**2), mbs_Rs=float(abs(rs[i])**2),
                Rs_relative_error_percent=float(100*(abs(rs[i]/es[i])**2-1)),
                relative_p_s_phase_error_deg=float(phase[i]),
                maximum_Mueller_error_percent=float(100*errors[i].max())))
        if loss == 0.:
            assert report['maximum_absolute_complex_coefficient_error'] < 1e-12
        assert normal_error < 1e-12 and direction_error < 1e-12
        reports.append(report)
        if loss == args.index_imag:
            np.savez_compressed(out/'absorbing_interface.npz', theta_deg=values[:, 0],
                exact_jones=np.stack([ep, es], axis=-1), mbs_jones=np.stack([rp, rs], axis=-1),
                exact_mueller=reference, mbs_mueller=candidate, errors_over_exact_M11=errors)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
            for ax, a, b, label in [(axes[0, 0], ep, rp, 'p'), (axes[0, 1], es, rs, 's')]:
                ax.plot(values[:, 0], abs(a)**2, label='Complex Fresnel', color='#1765a3')
                ax.plot(values[:, 0], abs(b)**2, '--', label='MBS-fast splitter', color='#d75c22')
                ax.set(ylabel=f'Reflected power R{label}', ylim=(0, 1.02))
                ax.legend()
            axes[1, 0].plot(values[:, 0], phase)
            axes[1, 0].set(ylabel='Error of relative p/s phase, degrees')
            axes[1, 1].plot(values[:, 0], 100*errors.max(axis=(1, 2)))
            axes[1, 1].set(ylabel='Max all-16 error / exact M11, %')
            for ax in axes.flat:
                ax.set(xlim=(0, 90), xlabel='Incidence from facet normal, degrees')
                ax.grid(alpha=.25)
            fig.suptitle(f'Flat-interface diagnostic: m={args.index_real}+{loss:g}i\n'
                         'Actual MBS external splitter; not a finite-particle BEM comparison')
            for ext in ('png', 'pdf'):
                fig.savefig(out/f'fresnel_audit.{ext}', dpi=160)
            plt.close(fig)
    result = dict(mbs_commit=commit, tests=reports,
        oracle='q=sqrt(m^2-sin(theta)^2), passive branch Im(q)>=0; complex Fresnel coefficients',
        oracle_source='https://arxiv.org/html/1603.02720v5#S3',
        scope='CPU calls to actual Splitting methods. Lossless and normal-incidence controls passed. '
              'No finite-particle BEM solve, no diffraction or fullauto validation.',
        production_sources_modified=False)
    (out/'fresnel_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
