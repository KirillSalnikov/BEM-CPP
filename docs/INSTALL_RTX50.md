# RTX 50 / Blackwell installer (Linux and WSL2)

Run from an existing checkout:

```bash
python3 scripts/install_rtx50.py --install-deps
```

The optional `--install-deps` flag installs host compiler/MPI/Fortran/Python
packages through apt and may ask for sudo. It never installs a display driver.
Omit it when host dependencies are already available.

The installer resolves the latest **stable** NVIDIA CUDA release from the
official archive, downloads the necessary redistributions over HTTPS, verifies
their sizes and SHA256 hashes against NVIDIA's release manifest, and installs
the toolkit under `~/.local/share/cuda/X.Y.Z`. Existing system toolkits remain
available. Original component licenses are preserved under the toolkit.

On 2026-11-03 the latest stable toolkit was **13.4.3**. To pin this version:

```bash
python3 scripts/install_rtx50.py --cuda-version 13.4.3 --jobs 4
```

To use an existing toolkit instead of downloading one:

```bash
python3 scripts/install_rtx50.py --cuda-home /usr/local/cuda --cxx g++-14
```

Native `sm_120` requires CUDA 12.8 or later. The installer detects compute
capability using `nvidia-smi`, including `/usr/lib/wsl/lib/nvidia-smi`, and
rejects a different architecture. RTX 50 and RTX PRO Blackwell GPUs with
compute capability 12.0 are supported; this is not a B200/`sm_100` profile.

On WSL, install/update the NVIDIA driver **in Windows**. Never install a Linux
NVIDIA display driver inside WSL. Toolkit installation does not update the
Windows driver. A compiled kernel is executed before project compilation to
check the actual driver/runtime/device combination.

Host prerequisites: GNU Make, Git, Python >=3.10 and a CUDA-compatible GNU C++
compiler. MBS needs Open MPI and currently selects GCC 12–14 because its legacy
global `complex` type conflicts with a new GCC 15 forward declaration. BEM also
needs gfortran, BLAS/LAPACK, FFTW, and NumPy/SciPy (`sph_harm_y` is required).
Pass `--cxx /path/to/g++-14` for a non-system compiler. BEM creates a private
Python environment when the supplied Python lacks its analysis requirements;
`--python /path/to/venv/bin/python` can reuse an existing suitable environment.

## Builds and outputs

MBS installs CPU MPI/OpenMP and both precise FP64 and mixed FP32 CUDA profiles.
It enables native CPU instructions, host LTO, native `sm_120`, and preserves
FP64 phase construction/trigonometry. Fast math and reduced/consumer phase
profiles are not silently enabled.

BEM installs the mixed-precision Muller solver, T-matrix FMM server, CPU dense
reference, and FP64/FP32 analytic SO(3) tools. CUDA compilation uses C++17 for
current CCCL and the build handles the CUDA 13 managed-memory advice API.

Executables are available under:

- `~/.local/opt/mbs-fast/bin/`: `mbs-po-cpu`, `mbs-po-gpu-fp64`, `mbs-po-gpu-fp32`;
- `~/.local/opt/bem-cpp/bin/`: `bem`, `bem-gpu`, `bem-tmatrix-server`,
  `bem-so3-fp64`, `bem-so3-fp32`, `bem-cpu-reference`.

Use `--prefix /other/install/directory` to change the destination. Source
snapshots and build caches are isolated by source hash, revision, toolkit,
compiler and architecture; old object files are not reused across profiles.
Each installation records exact commands, versions, source hash and device
probe results in `installation.json`. Per-stage logs stay in its build folder.

For a dry run without installations, downloads or builds:

```bash
python3 scripts/install_rtx50.py --plan --cuda-home /usr/local/cuda --cxx g++-14
python3 tests/test_install_rtx50.py
```

Native compilation and a passing GPU probe establish compatibility, not that
all possible kernel optimizations have been implemented. No Tensor Core/TF32
substitution or broad numerical-accuracy guarantee is made. Validate precision,
mesh/orientation convergence and performance on representative particles.

References: [NVIDIA toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive),
[CUDA redistribution manifests](https://developer.download.nvidia.com/compute/cuda/redist/),
[GPU compute capability](https://developer.nvidia.com/cuda/gpus),
[NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
