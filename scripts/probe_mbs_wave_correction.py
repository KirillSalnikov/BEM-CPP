#!/usr/bin/env python3
"""Experimental scalar paraxial correction to axial prism PO, not a Maxwell solver.

Subtract the straight propagation result on the identical raster before adding
the correction to MBS. This retains MBS's geometry and Fresnel/reflection terms.
The mean reference index (1+n)/2 reproduces the exact relative axial phase in
the two media. The approximation omits backward waves and vector edge coupling.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def correction(ka, n, height, radius, grid, steps, width, workers=1,
               interface_fresnel=False, round_trips=2, propagator='paraxial'):
    start = time.perf_counter()
    dx = width / grid
    x = (np.arange(grid) - grid / 2 + .5) * dx
    xx, yy = np.meshgrid(x, x, indexing='xy')
    # Hexagon with vertices at azimuth 0,60,...; edge distance sqrt(3)*R/2.
    mask = np.ones((grid, grid), dtype=bool)
    for azimuth in np.arange(6) * np.pi / 3 + np.pi / 6:
        mask &= xx * np.cos(azimuth) + yy * np.sin(azimuth) <= np.sqrt(3) * radius / 2
    dz = height / steps
    phase_half = np.exp(.5j * ka * (n - 1) * dz * mask)
    q = 2 * np.pi * fftfreq(grid, d=dx)
    q2 = q[:, None]**2 + q[None, :]**2
    if propagator == 'wide-angle':
        reference_k = ka * (1 + n) / 2
        propagation = np.exp(1j * dz * (np.sqrt(reference_k**2 - q2 + 0j) - reference_k))
    else:
        propagation = np.exp(-1j * dz * q2 / (ka * (1 + n)))
    def propagate(field):
        wave = field.copy()
        for _ in range(steps):
            wave *= phase_half
            wave = ifft2(fft2(wave, workers=workers) * propagation, workers=workers)
            wave *= phase_half
        return wave
    if interface_fresnel:
        r = (n - 1) / (n + 1)
        entry = 1 + (2 / (1 + n) - 1) * mask
        exit_transmission = 1 + (2 * n / (1 + n) - 1) * mask
        reflection = r * mask
        exit_field = propagate(entry.astype(np.complex128))
        wave = exit_transmission * exit_field
        ray_ratio = r**2 * np.exp(2j * ka * n * height)
        ray_transmission = (1 - r**2) * np.exp(1j * ka * (n - 1) * height)
        ray_sum = ray_transmission
        for order in range(1, round_trips + 1):
            entrance_reflected = reflection * propagate(reflection * exit_field)
            exit_field = np.exp(2j * ka * height) * propagate(entrance_reflected)
            wave += exit_transmission * exit_field
            ray_sum += ray_transmission * ray_ratio**order
        straight = 1 + (ray_sum - 1) * mask
    else:
        wave = propagate(np.ones((grid, grid), dtype=np.complex128))
        straight = np.exp(1j * ka * (n - 1) * height * mask)
    delta_f = -1j * ka / (2 * np.pi) * dx**2 * np.sum(wave - straight)
    return delta_f, {'grid': grid, 'steps': steps, 'width': width,
                     'elapsed_s': time.perf_counter() - start,
                     'raster_area': float(mask.sum() * dx**2),
                     'propagator': propagator,
                     'interface_fresnel': interface_fresnel, 'round_trips': round_trips if interface_fresnel else 0,
                     'exit_plane_mean_power_minus_one': float(np.mean(abs(wave)**2) - 1),
                     'edge_perturbation_max': float(max(np.max(abs(wave[0] - 1)), np.max(abs(wave[-1] - 1)),
                                                        np.max(abs(wave[:, 0] - 1)), np.max(abs(wave[:, -1] - 1))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, default=256)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--width', type=float, default=8.)
    parser.add_argument('--ka', type=float, nargs='+', default=[20, 40, 60])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--interface-fresnel', action='store_true')
    parser.add_argument('--round-trips', type=int, default=2)
    parser.add_argument('--propagator', choices=['paraxial', 'wide-angle'], default='paraxial')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ref = json.loads((root / 'benchmarks/d6h_20261026/forward_diagnosis/axial_model_diagnosis.json').read_text())
    height = ref['geometry']['height']
    radius = height / np.sqrt(3)
    records = []
    for ka in args.ka:
        row = next(row for row in ref['rows'] if row['ka'] == ka)
        base, exact = complex(*row['MBS_f']), complex(*row['BEM_f'])
        delta, metadata = correction(ka, 1.3, height, radius, args.grid, args.steps, args.width, args.workers,
                                     args.interface_fresnel, args.round_trips, args.propagator)
        improved = base + delta
        result = {'ka': ka, **metadata, 'MBS_f': [base.real, base.imag],
                  'BEM_f': [exact.real, exact.imag], 'delta_f': [delta.real, delta.imag],
                  'corrected_f': [improved.real, improved.imag],
                  'baseline_M11_error': abs(base / exact)**2 - 1,
                  'corrected_M11_error': abs(improved / exact)**2 - 1,
                  'corrected_complex_relative_error': abs(improved - exact) / abs(exact)}
        records.append(result)
        print(json.dumps(result), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'experimental': True, 'scope': 'axial forward scalar amplitude only',
                                       'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
