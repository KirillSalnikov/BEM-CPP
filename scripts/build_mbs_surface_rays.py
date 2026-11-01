#!/usr/bin/env python3
"""Build the experimental surface exporter against a separate MBS-fast checkout.

The MBS checkout and its production executable are never modified. This exporter
uses protected ScatteringConvex APIs from MBS-fast 1df49a0; later API changes may
require updating tools/mbs_surface_rays.cpp.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mbs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cxx', default='g++')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.mbs_root.resolve() / 'src'
    output = args.output.resolve()
    files = [
        'Beam.cpp', 'Splitting.cpp', 'Tracks.cpp', 'CutoffStatistics.cpp',
        'scattering/Scattering.cpp', 'scattering/ScatteringConvex.cpp',
        'particle/Particle.cpp', 'particle/Hexagonal.cpp',
        'geometry/geometry_lib.cpp', 'geometry/Intersection.cpp',
        'geometry/Facet.cpp', 'geometry/Polygon.cpp',
        'geometry/intrinsic/intrinsics.cpp', 'math/compl.cpp',
        'math/matrix.cpp', 'math/PhysMtr.cpp', 'math/JonesMatrix.cpp',
        'common/common.cpp', 'common/Matrix4x4.cpp', 'common/MullerMatrix.cpp',
    ]
    sources = [root / 'tools/mbs_surface_rays.cpp']
    sources += [source / name for name in files] + sorted((source / 'bigint').glob('*.cc'))
    if any(not path.is_file() for path in sources):
        parser.error('MBS source files are missing; supply a compatible complete checkout')
    includes = ['', 'math', 'geometry', 'geometry/intrinsic', 'geometry/sse',
                'particle', 'scattering', 'common', 'bigint']
    command = [args.cxx, '-O3', '-march=native', '-fopenmp', '-std=gnu++11']
    command += list(map(str, sources))
    command += ['-I' + str(source / path) for path in includes]
    command += ['-o', str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    git = subprocess.run(['git', '-C', str(source.parent), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True)
    receipt = {'command': command, 'mbs_commit': git.stdout.strip() if git.returncode == 0 else None,
               'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
               'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(output.suffix + '.build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
