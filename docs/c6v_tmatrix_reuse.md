# Vertical reflection in the T builder

The bullet has C6v symmetry: sixfold rotations and vertical reflection,
without C2x or horizontal reflection. Previously the geometry detector
reported this correctly, but the T builder used only C6 packing.

`build_fmm_tmatrix.py --particle 2 1 1` now automatically enables vertical
reflection reuse. `--vertical-reuse` requests it explicitly;
`--particle-symmetry off` retains the independent reference path.
The xz mirror is checked in the actual geometry. This option is separate
from `--mirror-pack`, which requires horizontal reflection.

In the repository's vector-wave convention, reflection in xz acts as

    U |p,l,m> = (-1)^(p+m+1) |p,l,-m>.

E transforms as a polar vector and H as an axial vector. The transformation
applies to both input and output indices. It maps m to -m within each
polarization; it does not identify the two polarizations with each other.
No horizontal-reflection selection rule is imposed on a bullet.

Before reconstructing negative-m columns, the builder compares positive
and negative controls at low, intermediate and maximum incident degree,
for both polarizations. The default relative column tolerance is 1e-4.
Failure stops the build before accepting symmetry-derived columns.
`vertical_audit.json` records the numerical comparison. Derived columns
carry `derived_by: sigma_v_xz`, their source index, and inherited C6 packet
provenance. Their residual is inherited, not independently measured.
Restart validation checks the exact signed permutation and the parent.

For L=52 there are 5616 full columns but 2860 nonnegative-m representatives,
plus independent negative-m controls. C6 packets still hold at most six
sectors and use two phase codes. The saving approaches twofold in the
number of solved modal columns; it is not a measured twofold wall-time gain.
Already accepted direct columns are retained, including negative-m ones.

Solver migration can add this scheduling optimization with the same
physical operator, worker, incident fields, projection and numerical gates.
It copies committed column arrays without changes and verifies packet hashes.
Existing C6 audits can be retained only with the existing explicit
identical-worker/input proof. The new vertical covariance audit is always
performed. Disabling a symmetry cannot import descendants that require it.

Validation on the saved ka=20/ref=5 bullet data compared 1749 pairs with
maximum relative column difference 1.44098e-4 (0.01441%). Three pairs exceed
1e-4; this survey is not a pass of the builder's separate control gate.
The median difference is 1.04715e-5. These are same-mesh checks, not a mesh
convergence certificate or an error bound for every Mueller angle.

Tests verify the field transformation, a C6v operator without C2x,
independent-versus-reconstructed complete T matrices, refusal of wrong
covariance, restart, and checksum-preserving migration from C6-only data.
The reduced-quadrature FGMRES experiment remains separate; this scheduling
change does not select that experimental inverse in the production worker.
