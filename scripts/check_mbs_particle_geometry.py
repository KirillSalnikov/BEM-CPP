"""Compare BEM shapes with the actual generators in an external MBS-fast tree."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.spatial import cKDTree
from mbs_particle_geometry import builtin, prepare, read_particle

ROOT = Path(__file__).resolve().parents[1]
SPECS = ([1, 2, 1], [2, 2, 1], [3, 2, 1], [4, 1],
         [10, 2, 1, 20], [12, 2, 1, 2], [999, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mbs-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    source = args.mbs_root.resolve() / 'src'
    names = ['Particle', 'Hexagonal', 'Bullet', 'BulletRosette', 'Droxtal',
             'ConcaveHexagonal', 'HexagonalAggregate', 'CertainAggregate']
    files = [source / 'particle' / (name + '.cpp') for name in names]
    files += [source / name for name in ('math/compl.cpp', 'geometry/Polygon.cpp',
              'geometry/Facet.cpp', 'geometry/geometry_lib.cpp',
              'geometry/intrinsic/intrinsics.cpp', 'common/common.cpp')]
    include = [source / name for name in ('particle', 'math', 'geometry',
                                         'geometry/intrinsic', 'common')]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix='mbs_geometry_oracle_') as temporary:
        executable = Path(temporary) / 'oracle'
        command = [args.cxx, '-std=c++17', '-O2', '-ffunction-sections',
                   '-fdata-sections', '-Wl,--gc-sections']
        command += ['-I' + str(path) for path in include]
        command += [str(ROOT / 'tools/export_mbs_particle_geometry.cpp')]
        command += [str(path) for path in files]
        command += ['-o', str(executable)]
        subprocess.run(command, check=True)
        for spec in SPECS:
            output = args.out / f'type{spec[0]}.particle'
            native = subprocess.check_output([str(executable), str(spec[0]), str(output)], text=True)
            fields = native.split()
            volume = float(fields[fields.index('volume') + 1])
            a, b = np.vstack(builtin(spec)), np.vstack(read_particle(output))
            scale = np.linalg.norm(np.ptp(a, axis=0))
            vertex_error = max(cKDTree(a).query(b)[0].max(),
                               cKDTree(b).query(a)[0].max()) / scale
            _, _, report = prepare(spec)
            row = dict(particle=spec, vertex_relative_error=float(vertex_error),
                       volume_relative_error=abs(report['volume'] / volume - 1),
                       mbs_volume=volume, bem_volume=report['volume'],
                       symmetry=report['symmetry']['order'])
            rows.append(row)
    result = dict(cases=rows, source_sha256={str(path.relative_to(source)):
                  hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    (args.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    if any(max(row['vertex_relative_error'], row['volume_relative_error']) > 1e-12 for row in rows):
        raise SystemExit('MBS geometry mismatch; see comparison.json')
    print(f'PASS: {len(rows)} native MBS geometries; {args.out / "comparison.json"}')


if __name__ == '__main__':
    main()
