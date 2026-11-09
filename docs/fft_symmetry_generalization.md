# Checked FFT acceleration and general surface symmetries

The ka=40 production continuation now uses a compressed FFT inner action in
the native T worker, with an independently checked FMM residual. On two
matched C6 audit packets, including radiation, time decreased from 790.26 s
to 201.72 s (3.92×). The largest column-norm difference from saved results was
6.72e-6. These measurements use the SciIce hexagonal column, ref=6, L=65,
m=1.3116, wavelength 0.532 µm, H/Dflat=1.6495721976846451, tolerance 2e-6.
They exclude one-time preparation and SO(3) averaging, and are not a measured
speedup of the entire T build or of another shape.

## What transfers to other shapes

| Component | Geometric requirement | Current status |
|---|---|---|
| Accurate residual replacement around a compressed inner operator | None; accurate action must be available | `checked_residual_solver.py`, native GMRES |
| Double-single near kernel | Supported real-wavenumber flat neighbor-list branch | Opt-in `BEM_FMM_P2P_DOUBLE_SINGLE=1`; unsupported cases retain FP64 |
| CSR storage for the local correction/mass | None | Shared Torch implementation, COO/CSR tests |
| Frequency-wise low-rank factors | A usable regular-face spectral block | Existing generic cross-face builder; ranks require validation for each geometry/material/size |
| Point-group reuse of spectral blocks | Finite orthogonal group preserving the actual triangulated mesh | Generalized and tested below |
| C6/modal RHS packing, D6h sector inverse and warm starts | Existing D6h prism mesh | Still specialized; arbitrary-group block reuse does not turn a noncommuting group into independent C6 sectors |

Double-single uses two FP32 components for improved precision, not the full
exponent range or mantissa of FP64. Its ka=40 accuracy measurements and exact
fallback conditions remain necessary; a different particle does not inherit
the same timing or a guarantee of 1% Mueller accuracy.

## General point-group implementation

`scripts/surface_symmetry.py` accepts a finite list of 3×3 orthogonal matrices,
or closes a group from generators. The center is explicit. It checks identity,
uniqueness, closure and orthogonality. Before reusing blocks it verifies a
bijection of triangle centers, all triangle vertices, outward normals and
the face partition. A symmetry of the smooth shape alone is insufficient:
the triangulation must preserve it too. Invalid groups or meshes fail early.

`RegularSurfaceGeometry` reads native BDM arrays and explicitly supplied
lattice steps. Planar faces are grouped using both normals and plane offsets;
distinct parallel planes of a nonconvex particle are not merged. A common
lattice is verified separately for each pair. Masks and signed BDM incidence
retain the actual finite face. The number of faces is unrestricted by the
old eight-face prism layout. Unsupported lattices raise an error; use the
ordinary FMM solver for those meshes.

`SymmetricGalerkinFFT(..., group=group)` builds one physical block per orbit of
ordered face pairs. It supports proper and improper transformations and
noncommuting groups. Reflection signs distinguish the two current families:
the K1 block changes sign under an improper orthogonal transformation, while
K2ε and K2μ do not. Source and target signed maps therefore differ. This is
block reuse, not an assumed simultaneous diagonalization of all symmetries.
Near corrections and mass are added in the original global basis.

Example for native exported arrays and a known lattice:

```python
import json
import numpy as np
from regular_surface_geometry import RegularSurfaceGeometry
from surface_symmetry import PointGroup
from symmetric_galerkin_fft import SymmetricGalerkinFFT

fixture = json.load(open("patterns.json"))
axes = np.load("face_axes.npy")  # (face_count, 2, 3), first-triangle order
geometry = RegularSurfaceGeometry("native_export", fixture, axes)
group = PointGroup.from_generators([
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
    np.diag([-1, 1, 1]),
], center=(0, 0, 0))  # Oh, only valid for an invariant mesh
operator = SymmetricGalerkinFFT(geometry, group=group)
operator.load_correction()  # native sparse near correction and 13-point mass
# inner = operator.torch_operator("cuda", dtype=torch.complex64)
# Accept solutions only with an independent original-operator residual.
```

Identity alone is valid for an asymmetric particle. Continuous axial symmetry
is usable here only through a finite subgroup preserved by the discrete mesh.
Arbitrary curved or irregular meshes do not become convolution grids merely
because they possess a point symmetry.

The production socket/driver path remains restricted to the validated built-in
D6h prism. The general-group operator is a Python CPU/Torch library interface,
with independent CPU physics tests; a full native T run for another shape
has not been validated or enabled by removing the production guard.

## Independent checks on other shapes

`tests/test_surface_symmetry_fft.py` builds CPU-only native BDM/Green exporters
from source, generates closed OBJ meshes and compares random-vector actions
against the independently accumulated C++ kernel matrix. It also checks a
full absorbing tetrahedron solve against native dense BEM including singular
integration and mass. Small validation matrices are never a large-run backend.

| Geometry / chosen group | Face pairs | Independent blocks |
|---|---:|---:|
| Cube / Oh (48 operations) | 36 | 3 |
| Cube / Cs (one mirror) | 36 | 26 |
| Cube / C4 | 36 | 12 |
| Cube / C1 | 36 | 36 |
| Tetrahedron / Td (24 operations), translated, absorbing | 16 | 2 |
| Skew tetrahedron / C1 | 16 | 16 |
| Nonconvex L-shaped column / C2v | 64 | 26 |

These are block counts, not total-memory or runtime speedups. Initial native
regular-action discrepancies are below 1e-14. The absorbing tetrahedron's
full checked residual is 1.40e-14 and current difference is 2.41e-14.
The tests additionally exercise rotated symmetry axes, the Torch action on
CPU in both precisions, invalid group closure and incompatible geometry.
They establish algebraic/operator correctness on these small meshes, not
mesh convergence or large-ka scattering accuracy for the other shapes.

Run without touching the active GPU calculation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 tests/test_surface_symmetry_fft.py
python3 tests/test_external_fft_action.py
python3 tests/test_tmatrix_checkpoint_migration.py
```

PyTorch is optional for the native CPU/group checks and required for the
Torch/streamed FFT paths (`requirements-fft.txt`). Torch-specific tests are
reported as skipped when it is absent; the independent C++ physics checks
still run. CUDA execution on the other shapes was not part of these tests.

For a new shape: validate its mesh and local quadrature, check the proposed
group, compare the compressed action and several checked solves with the
original operator, then measure complete preparation/solve/radiation time.
Reuse of T columns for general groups requires the corresponding vector-wave
representations and independent phase/covariance audits; it is separate work
from the face-pair reduction implemented here.

## Bounded storage for the SciIce ref=7 mesh

At ka=40 the same SciIce aspect gives 758,784 triangles and 4,552,704
unknowns at ref=7. This mesh is not obtained by doubling the rounded ref=6
axial cell count. `muller_fft_geometry_export` exports the actual native mesh;
the face lattice and signed BDM maps are checked against those arrays.

The geometry-only exporter can be built without allocating the FMM:

```bash
make bin/muller_fft_geometry_export
mkdir -p exported_geometry
bin/muller_fft_geometry_export 7 40 1.6495721976846451 exported_geometry
```

This diagnostic exports SciIce metadata with m=1.3116. The native near
correction and the prepared spectral blocks are separate required assets.

The optional `execution.json` in a prepared FFT case now selects bounded
storage. Its contents and the packed local arrays are included in the asset
hash manifest. The measured 24 GiB GPU configuration is:

```json
{
  "local_storage": "host_packed",
  "spectral_budget_MiB": 2242,
  "frequency_batch": 8,
  "compact_maps": true
}
```

`spectral_budget_MiB` bounds resident coefficients, not the total CUDA
allocation. Maps, double transfer buffers, FFT scratch, the native FMM and
the preconditioner also consume VRAM. Cross factors retain the existing
scaled FP16 storage and FP32 arithmetic. Two pinned buffers overlap host
transfers with products; no rank, FFT grid, or symmetry is removed.
The full operator need not fit on the GPU at once, but the frequently read
host coefficients still need enough RAM to avoid disk paging.

The local correction/mass stores three complex coefficients with shared CSR
indices, representing `[[A,B],[C,-A]]`. `host_packed` applies this block on
the CPU while the GPU evaluates the regular FFT blocks. On the actual ref=7
arrays its CPU action takes 0.069–0.071 s and differs from an independently
assembled SciPy CSR action by 1.47e-7 in relative vector norm. It releases
about 3.45 GiB of GPU coefficient storage. The CPU code uses OpenMP and
native vector instructions without `-ffast-math`.

For the T driver, `--no-incident-cache --release-incident-buffer` releases
the already transmitted incident payload during each solve. The next packet
regenerates both fields. `--release-incident-buffer` is incompatible with
`--incident-prefetch`; these flags propagate through `run_large_tmatrix.py`
and its resume command. The independent strict residual gate is unchanged.

Sector GMRES now keeps separate Arnoldi buffers and allocates them only for
active sectors, materializing vectors as iterations advance. A later strict
residual may activate any previously inactive sector. For restart=64, the
old unconditional ref=7 basis was 4.41 GiB; an RHS exciting one of twelve
equal sectors reserves only one twelfth of that storage. Fully populated
packed RHSs can still need the full basis. CPU controls retain the original
iteration counts and strict residual tolerances, including cross-sector
perturbations, recycling, and activation after a residual replacement.

An additional experimental `"symbol_storage": "half"` setting applies
scaled FP16 storage to dense parallel-face symbols too, with FP32 products.
It is separate from lossless compact maps and packed CSR: it changes the
approximate inner operator and must pass the native startup and final
residual checks. The default remains `"full"`.

The coexistence audit (`benchmarks/ref7_memory_20261109/combined_ref7_report.json`)
measured 20.75 GiB peak total GPU use, 2.08 s median warm FFT action, and
0.0145% relative action error on a smooth input against the original FMM.
Those are operator measurements, not a Mueller error or a complete T-build
speedup. In particular, the bounded ref=6 action is slower than the fully
resident action (0.614 s versus about 0.080 s); keep resident storage when it
fits. A tested deeper FMM tree was rejected because it changed the control
operator by 1.65e-5; the accepted ref=7 configuration retains depth six.

### Modal cutoff of the completed ka=40 reference

An analytic SO(3) check of the completed SciIce ref=6 T matrix (L=65)
retains both rows and columns with degree at most L. The output angle grid
has step 0.025 degrees. Relative M11 error at 180 degrees and the maximum
absolute error of any Mueller entry divided by reference M11 are:

| L | M11 at 180 degrees | All entries, all angles |
|---:|---:|---:|
| 55 | 5.611% | 6.832% |
| 57 | 1.491% | 2.212% |
| 58 | 0.768% | 1.286% |
| 59 | 0.553% | 0.709% |
| 61 | 0.213% | 0.213% |
| 63 | 0.077% | 0.077% |

In the tested range, L=58 passes 1% throughout the 175–180 degree cone,
and L=59 passes the full-angle metric. L=61 provides two extra orders.
The reference is L=65, not an infinite-order solution. These tests do not
establish surface-mesh convergence or accuracy of a ref=7 T matrix. Results
and individual analytic averages are in
`benchmarks/ref7_memory_20261109/ka40_saved_T_L_controls/`.
