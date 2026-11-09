SciIce column comparison ka=20,25,30,35,40. Reverse cone theta=175..180 degrees.
MBS: coherent native fullauto 1%, including depth and independent final checks.
BEM: existing complete T matrices and analytic SO(3), no new linear solves.
All particles: n=1.3116, lambda=0.532 um, D_across_vertices/H=0.7.
Old full-angle 4% campaign stopped and preserved; no 4% result is labelled 1%.
CPU queue: 20,25 then 30,35 (two workers). GPU: 40.
BEM_MBS_backcone_1pct.pdf updates atomically after each accepted MBS result.
status.json records ready sizes. Pending sizes contain only the existing BEM curve.
Errors and their integrals are restricted to 175..180 degrees; errors use BEM as reference.
M11 is dimensional, other plotted elements use each method's own M11.
MBS 1% is a convergence tolerance, not a promise of 1% agreement with BEM.
Rebuild: python3 build.py

The updated SciIce + BEM + MBS report is in
benchmarks/scice_2025_comparison_20261103/bem_vs_scice_backscatter.pdf.
That newer report uses MBS in the error denominator, as requested separately;
this original comparison retains its explicitly labelled BEM denominator.
The saved comparison NPZ/CSV files contain Mueller data, not T matrices.
Rebuilding from the native sources requires the original run directories;
their paths and hashes are recorded in the comparison JSON files.
