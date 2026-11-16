"""Exact array-size model from generated meshes; excludes FMM/ILU/T workspace."""
import os,sys,json
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[1]
os.environ.setdefault('BEM_DIRAC_LIBRARY',str(B/'libdirac_hierarchy.so'))
sys.path.insert(0,str(R/'benchmarks/clifford_dielectric_20261115'))
from solve import lib
out=[]
for ka,ref in [(20,3),(40,4),(80,5),(80,7)]:
    p=lib.dirac_create_geometry(1,ref,1.6495721976846451,ka,1.3116,0.,24,1,13,3,1)
    if not p:raise RuntimeError(lib.dirac_error().decode())
    full=lib.dirac_size(p);nb=lib.dirac_basis_count(p);lib.dirac_destroy(p)
    panels=full//(8*nb);nf=full//12;nc=panels//12*24
    row=dict(ka=ka,ref=ref,panels=panels,full_dofs=full,full_P1_LU_bytes=16*nc*nc+4*nc,
        all_panel_LU_bytes=panels*(16*(8*nb)**2+4*8*nb))
    row['capped_global']={str(cap):dict(columns=8*3*min(cap,panels//12),
        LU_bytes=16*(8*3*min(cap,panels//12))**2+4*(8*3*min(cap,panels//12)),
        W_bytes=16*nf*(8*3*min(cap,panels//12))) for cap in [128,256,512]}
    out.append(row)
(B/'capacity.json').write_text(json.dumps(dict(scope='Array-size model only. Excludes sparse ILU, near cache, FMM workspace, Krylov/T arrays and temporary buffers. No accuracy or full-fit claim.',cases=out),indent=2)+'\n')
print('capacity.json written')
