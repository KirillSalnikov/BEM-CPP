#!/usr/bin/env python3
"""Measure a type-3 cuFINUFFT far-field surrogate at BEM production sizes."""

import argparse
import json
import math
import time

import torch

import cufinufft


def synchronize():
    torch.cuda.synchronize()


parser = argparse.ArgumentParser()
parser.add_argument("--sources", type=int, default=236544)
parser.add_argument("--alpha", type=int, default=256)
parser.add_argument("--theta", type=int, default=73)
parser.add_argument("--transforms", type=int, default=6)
parser.add_argument("--ka", type=float, default=30.0)
parser.add_argument("--eps", type=float, default=1.0e-6)
parser.add_argument("--output", default="")
args = parser.parse_args()

torch.manual_seed(1701)
device = torch.device("cuda")
real_dtype = torch.float64
complex_dtype = torch.complex128

# Coordinates have the same bounded scale as the normalized BEM particle.
points = 2.0 * torch.rand(
    (3, args.sources), device=device, dtype=real_dtype
) - 1.0
strengths = torch.randn(
    (args.transforms, args.sources), device=device, dtype=complex_dtype
)

alpha = torch.arange(args.alpha, device=device, dtype=real_dtype)
alpha *= 2.0 * math.pi / args.alpha
theta = torch.linspace(
    0.0, math.pi, args.theta, device=device, dtype=real_dtype
)
aa, tt = torch.meshgrid(alpha, theta, indexing="ij")
target_x = args.ka * torch.sin(tt).ravel() * torch.cos(aa).ravel()
target_y = args.ka * torch.sin(tt).ravel() * torch.sin(aa).ravel()
target_z = args.ka * torch.cos(tt).ravel()

synchronize()
start = time.perf_counter()
plan = cufinufft.Plan(
    3, 3, args.transforms, args.eps, -1, "complex128"
)
plan.setpts(
    points[0], points[1], points[2],
    target_x, target_y, target_z,
)
synchronize()
setup_seconds = time.perf_counter() - start

# The first call includes lazy CUDA work. Report it separately from the median.
start = time.perf_counter()
result = plan.execute(strengths)
synchronize()
first_seconds = time.perf_counter() - start

execution_seconds = []
for _ in range(5):
    start = time.perf_counter()
    result = plan.execute(strengths)
    synchronize()
    execution_seconds.append(time.perf_counter() - start)
execution_seconds.sort()

# Check a small exact subset without materializing the full dense phase matrix.
check_sources = min(4096, args.sources)
check_targets = min(32, target_x.numel())
phase = (
    points[0, :check_sources, None] * target_x[None, :check_targets]
    + points[1, :check_sources, None] * target_y[None, :check_targets]
    + points[2, :check_sources, None] * target_z[None, :check_targets]
)
direct = torch.sum(
    strengths[0, :check_sources, None] * torch.exp(-1j * phase), dim=0
)
subset = cufinufft.nufft3d3(
    points[0, :check_sources],
    points[1, :check_sources],
    points[2, :check_sources],
    strengths[0, :check_sources],
    target_x[:check_targets],
    target_y[:check_targets],
    target_z[:check_targets],
    eps=args.eps,
    isign=-1,
)
synchronize()
relative_error = float(
    torch.max(torch.abs(subset - direct))
    / torch.max(torch.abs(direct))
)

summary = {
    "cufinufft_version": cufinufft.__version__,
    "sources": args.sources,
    "targets": int(target_x.numel()),
    "transforms": args.transforms,
    "ka": args.ka,
    "requested_epsilon": args.eps,
    "plan_and_points_s": setup_seconds,
    "first_execution_s": first_seconds,
    "median_reused_execution_s": execution_seconds[2],
    "subset_relative_error": relative_error,
    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
}
print(json.dumps(summary, indent=2))
if args.output:
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
