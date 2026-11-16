"""Dense T-driver capacity check, before allocating any interaction matrix.

Array sizes are exact; the workspace reserve is a conservative planning model,
not a process-wide allocator limit. The FMM backend has a separate memory model.
"""
import psutil
from solve import lib


def sector_fmm_memory_plan(a):
    """Bound persistent arrays; reserve (not a hard bound) FMM3D tree workspace."""
    if a.target_order!=13:raise ValueError('sector FMM requires target13')
    p=lib.dirac_create_geometry(int(a.shape=='prism'),a.ref,a.aspect,a.ka,
        complex(a.index).real,complex(a.index).imag,a.order,1,a.target_order,a.degree,a.near_subdiv)
    if not p:raise RuntimeError(lib.dirac_error().decode())
    pair_upper=None
    try:
        full=lib.dirac_size(p);nb=lib.dirac_basis_count(p)
        if getattr(a,'coarse_factorization','dense')!='dense' and a.sector_coarse_basis:
            import ctypes as ct
            import numpy as np
            from scipy.spatial import cKDTree
            from solve import address
            panels=full//(8*nb);vertices=np.empty((panels,3,3));geometry=np.empty((panels,13))
            lib.dirac_vertices.argtypes=[ct.c_void_p,ct.c_void_p]
            lib.dirac_vertices(p,address(vertices));lib.dirac_geometry(p,address(geometry))
            diam=np.max(np.linalg.norm(vertices-np.roll(vertices,1,axis=1),axis=2))
            counts=cKDTree(geometry[:,:3]).query_ball_point(geometry[:,:3],a.coarse_near_radius*diam*(1+1e-12),return_length=True)
            group=12 if a.symmetry=='c6hv' else 6
            pair_upper=(int(counts.sum())+group-1)//group
    finally:lib.dirac_destroy(p)
    panels=full//(8*nb);group=12 if a.symmetry=='c6hv' else 6;fine=full//group
    factors=panels*(16*(8*nb)**2+4*(8*nb));nc=panels//group*8*a.sector_coarse_basis
    states=1 if a.coarse_cache_policy=='sector' else group
    coarse=states*(16*nc*nc+4*nc) if nc else 0
    transforms=group*panels*(8*(nb*nb+64)+8)
    coarse_work=max(3*16*nc*a.sector_coarse_basis*8*8,16*(14*full+20*fine))+transforms if nc else 0
    sparse_info={}
    if pair_upper is not None:
        nnz=pair_upper*(8*a.sector_coarse_basis)**2
        one=int((20*nnz+8*nc)*a.coarse_ilu_fill)+16*nc+8
        coarse_work=32*nnz+2*one+16*(14*full+20*fine)+(32<<20)+transforms+16*pair_upper+16*panels
        global_bytes=0
        if a.coarse_factorization=='hierarchical':
            ng=8*a.coarse_aggregate_basis*min(a.coarse_aggregate_groups,panels//group)
            rows=fine if a.coarse_global_scope=='full' else nc
            global_bytes=16*(rows*ng+ng**2)+4*ng
            coarse_work+=16*(2*ng**2+rows*640)+16384*panels
        coarse=states*(one+global_bytes)
        sparse_info=dict(factorization=a.coarse_factorization,near_pair_upper=pair_upper,
            factor_storage_reserve=one,bounded_global_correction_bytes=global_bytes,
            dense_P1_Ac_bytes=16*ng**2 if a.coarse_factorization=='hierarchical' and ng==nc and a.sector_coarse_basis==3 else 0)
    if coarse+coarse_work+factors//group>a.coarse_budget_gib*(1<<30):
        raise MemoryError('coarse LU and workspace exceed budget; no near cache allocated')
    if a.kernel_budget_gib<=0:raise ValueError('near cache budget must be positive')
    near_cap=int(a.kernel_budget_gib*(1<<30))
    modes=2*a.L*(a.L+2);t_arrays=2*16*modes*modes
    # The native near builder enforces near_cap. FMM3D internal allocations
    # depend on ka, epsilon, packet and the adaptive tree; measure actual RSS.
    work=16*(66*a.batch_size*fine+20*a.batch_size*full)+2*factors+2*transforms+(512<<20)
    fmm_work=max(2<<30,16*13*panels*8*a.fmm_packet*16)
    iterative=near_cap+factors+coarse+coarse_work+t_arrays+work+fmm_work
    dm=(2*a.L+3)*3*a.L*2*a.L*2*16
    averaging=t_arrays+16*modes*modes+2*dm*(a.so3_threads+1)+(256<<20) if a.so3=='mstm' and not getattr(a,'no_average',False) else None
    estimate=max(iterative,averaging or 0)
    budget=min(int(a.ram_budget_gib*(1<<30)),max(0,psutil.virtual_memory().available-(2<<30)))
    if estimate>budget:raise MemoryError(f'planned sector FMM peak {estimate/2**30:.2f} GiB exceeds RAM budget {budget/2**30:.2f} GiB; no near cache allocated')
    return dict(panels=panels,full_dofs=full,sector_dofs=fine,near_cache_cap_bytes=near_cap,dense_far_bytes=0,
        stored_fine_W_bytes=0,coarse_resident_bytes=coarse,coarse_workspace_bytes=coarse_work,panel_factors_bytes=factors,sparse_coarse=sparse_info,
        T_arrays_bytes=t_arrays,solver_workspace_reserve_bytes=work,FMM_workspace_reserve_bytes=fmm_work,
        solve_phase_estimate_bytes=iterative,SO3_phase_estimate_bytes=averaging,
        estimated_peak_bytes=estimate,effective_budget_bytes=budget,
        scope='Near cache cap and selected coarse factorization accounted before allocation. FMM/ILU workspace is a reserve, not a hard allocator limit; measure RSS. No spatial/L convergence certificate.')


def dense_memory_plan(a):
    p=lib.dirac_create_geometry(int(a.shape=='prism'),a.ref,a.aspect,a.ka,
        complex(a.index).real,complex(a.index).imag,a.order,1,a.target_order,a.degree,a.near_subdiv)
    if not p:raise RuntimeError(lib.dirac_error().decode())
    try:
        full=lib.dirac_size(p);nb=lib.dirac_basis_count(p)
    finally:lib.dirac_destroy(p)
    panels=full//(8*nb);group=12 if a.symmetry=='c6hv' else 6 if a.symmetry!='none' else 1
    fine=full//group;coefficient=full//8
    kernel=128*coefficient*coefficient//(12 if a.dense_cache=='orbits12' else 1)
    if kernel>a.kernel_budget_gib*(1<<30):raise MemoryError(f'dense operator alone needs {kernel/2**30:.3f} GiB; kernel budget exceeded')
    panel_factors=panels*(16*(8*nb)**2+4*(8*nb)) if a.block_panels else 0
    coarse=coarse_workspace=0
    if a.sector_coarse_basis:
        nc=(panels//group)*8*a.sector_coarse_basis
        states=1 if a.coarse_cache_policy=='sector' else group
        coarse=states*(16*(fine*nc+nc*nc)+4*nc)
        coarse_workspace=max(3*16*fine*8*a.sector_coarse_basis*8,
                             16*(12*full+12*fine+3*nc+8*nb*nc))
        if coarse+coarse_workspace+panel_factors//group>a.coarse_budget_gib*(1<<30):
            raise MemoryError('coarse cache and workspace exceed budget; no dense matrix allocated')
    modes=2*a.L*(a.L+2)
    t_arrays=2*16*modes*modes
    # Restart-60 Krylov vectors, simultaneous full-space input/output/lifts,
    # geometry transforms, panel-factor assembly and Python/native overhead.
    workspace=16*(66*a.batch_size*fine+20*a.batch_size*full)+2*panel_factors+(512<<20)
    iterative=kernel+panel_factors+coarse+coarse_workspace+t_arrays+workspace
    # MSTM releases the BEM operator first; FP64 T plus shared/thread tensors.
    dm=(2*a.L+3)*3*a.L*2*a.L*2*16
    averaging=t_arrays+16*modes*modes+2*dm*(a.so3_threads+1)+(256<<20) if a.so3=='mstm' and not getattr(a,'no_average',False) else None
    estimate=max(iterative,averaging or 0)
    budget=min(int(a.ram_budget_gib*(1<<30)),max(0,psutil.virtual_memory().available-(2<<30)))
    result=dict(panels=panels,full_dofs=full,sector_dofs=fine,kernel_bytes=kernel,
        coarse_resident_bytes=coarse,coarse_workspace_bytes=coarse_workspace,
        panel_factors_bytes=panel_factors,T_arrays_bytes=t_arrays,solver_workspace_reserve_bytes=workspace,
        solve_phase_estimate_bytes=iterative,SO3_phase_estimate_bytes=averaging,
        estimated_peak_bytes=estimate,effective_budget_bytes=budget,
        scope='Planning estimate including temporary reserve; actual RSS must be measured. Legacy SO3 workspace excluded unless mstm.')
    if estimate>budget:raise MemoryError(f'planned dense T peak {estimate/2**30:.2f} GiB exceeds RAM budget {budget/2**30:.2f} GiB; no dense matrix allocated')
    return result
