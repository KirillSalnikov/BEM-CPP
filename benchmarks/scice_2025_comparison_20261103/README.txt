Updated 2026-11-09: matched SciIce columns, ka=20,25,30,35,40.
The directory date records the original comparison; the current report uses
D_across_vertices/H=0.7 for all five BEM and MBS calculations.

bem_vs_scice_backscatter.pdf contains ten pages: M11 and LDR versus size,
backscatter differences, angular comparisons over 175..180 degrees, maxima,
and integral errors for the Mueller elements. The new errors use MBS in the
denominator; formulas are printed on the plots. MBS is coherent fullauto 1%.

Five direct BEM T matrices were averaged analytically over SO(3). No BEM size
interpolation was used. Only published SciIce M11 and M22 were linearly
interpolated for the comparison at the five BEM sizes; LDR was then formed.
The original ScIce-2025 table is unchanged; source DOI and hashes are in
summary.json. The local derived database and full-angle Mueller NPZ files are
in data/external/scice_2025/BEM_extension_20261109/.

Report source: scripts/report_scice_bem_mbs.py (run from the repository root).
Regeneration verifies original run hashes and requires those external run
directories and the published SciIce table. Giant T matrices are not in Git.
Compact comparison inputs are in exports/scice_bem_mbs_cone5_1pct_20261109/.
The completed-data validation and PDF digest are in validation.json.
No ref=7 solution is included; its separate T pilot is still pending.
