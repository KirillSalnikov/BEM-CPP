# Checked FFT acceleration and general surface symmetries

The ka=40 production continuation now uses a compressed FFT inner action in
the native T worker, with an independently checked FMM residual. On two
matched C6 audit packets, including radiation, time decreased from 790.26 s
to 201.72 s (3.92×). The largest column-norm difference from saved results was
6.72e-6. These measurements use the SciIce hexagonal column, ref=6, L=65,
m=1.3116, wavelength 0.532 µm, H/Dflat=1.6495721976846451, tolerance 2e-6.
They exclude one-time preparation and SO(3) averaging, and are not a measured
speedup of the entire unfinished T matrix or of another shape.

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
