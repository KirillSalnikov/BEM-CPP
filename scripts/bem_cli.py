"""Unified, inspectable entry point; existing expert drivers remain available."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DIRAC = ROOT / 'benchmarks/clifford_dielectric_20261115'
GIB = 2**30
METHODS = {
    'muller': 'Мюллер P2/BDM1: GPU/FMM, все встроенные формы и OBJ, фиксированное падение и T/SO(3).',
    'dirac': 'Дирак: экспериментальный CPU FP64, сфера/призма, фиксированное падение и T/SO(3).',
    'pmchwt': 'PMCHWT/RWG: прежний GPU/FMM, фиксированное падение; построение T здесь не подключено.',
}


def register(subparsers):
    p = subparsers.add_parser('solve', help='единый запуск: метод, частица, фиксированное падение или T/SO(3)')
    p.add_argument('--method', choices=METHODS, default='muller', help='формулировка; default: muller')
    p.add_argument('--task', choices=('fixed', 'tmatrix', 'average'), default='fixed',
                   help='fixed: две поляризации; tmatrix: сохранить T; average: T + аналитическое SO(3)')
    geo = p.add_mutually_exclusive_group()
    geo.add_argument('--shape', choices=('sphere', 'prism', 'tetra'), default=None)
    geo.add_argument('--particle', type=float, nargs='+', metavar='TYPE_PARAMS', help='типы MBS 1,2,3,4,10,12,999')
    geo.add_argument('--particle-file', type=Path)
    geo.add_argument('--obj', type=Path, help='замкнутая треугольная поверхность')
    p.add_argument('--aspect', type=float, default=10/7, help='призма: H / ширина между гранями (default: 10/7)')
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument('--ka', type=float, help='2*pi*a_eq/lambda')
    size.add_argument('--radius', type=float, help='радиус равновеликой сферы, мкм')
    p.add_argument('--wavelength', type=float, default=.532, help='длина волны в среде с n=1, мкм')
    p.add_argument('--index', default='1.3116', help='m=n+i*kappa, например 1.3116+0.1i')
    p.add_argument('--orientation', type=float, nargs=3, default=None, metavar=('ALPHA','BETA','GAMMA'),
                   help='fixed: активное вращение частицы Rz(alpha) Ry(beta) Rz(gamma), градусы')
    p.add_argument('--symmetry', choices=('auto','none','c6','c6v','d6h'), default='auto',
                   help='явная группа проверяется по геометрии; none отключает сокращения')
    p.add_argument('--accuracy-percent', type=float, default=1., help='целевая точность для начального подбора; не сертификат сходимости')
    p.add_argument('--ref', type=int, help='эксперт: переопределить начальное измельчение')
    p.add_argument('--L', type=int, help='эксперт: переопределить угловой порядок T')
    p.add_argument('--tol', type=float, help='эксперт: невязка линейной системы')
    p.add_argument('--backend', choices=('auto','dense','fmm-sector'), default='auto', help='только Дирак')
    p.add_argument('--global-patches', type=int, help='Дирак/FMM: глобальные P1-участки, 2..512 (не L)')
    p.add_argument('--batch-size', type=int, help='Дирак: одновременно решаемые столбцы, 1..16')
    p.add_argument('--ram-gib', type=float, help='бюджет суммы RSS дерева расчёта; автоматически оставляется резерв ОС')
    p.add_argument('--threads', type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument('--ntheta', type=int, default=181, help='углы рассеяния 0..180, не ориентации')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--dry-run', action='store_true', help='показать план без сборки и расчёта')
    p.add_argument('--no-build', action='store_true', help='не собирать отсутствующие/устаревшие библиотеки')
    p.set_defaults(handler=execute)
    methods = subparsers.add_parser('methods', help='возможности методов')
    methods.set_defaults(handler=lambda _: show_methods())
    wizard = subparsers.add_parser('wizard', help='пошагово составить и запустить расчёт')
    wizard.set_defaults(handler=lambda _: interactive(p))
    resume = subparsers.add_parser('continue', help='продолжить расчёт, созданный bem solve')
    resume.add_argument('directory', type=Path)
    resume.set_defaults(handler=continue_run)


def show_methods():
    for name, description in METHODS.items():
        print(f'{name:8s} {description}')
    print('FMM, FFT и dense — способы применения/хранения оператора, не отдельные уравнения.\n'
          'Старый эксперимент RWG muller2 не заменяет основной решатель Мюллера.')
    return 0


def finite_positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name}: требуется конечное положительное число')
    return value


def index_value(text):
    try:
        m = complex(str(text).replace(' ', '').replace('i', 'j'))
    except ValueError as exc:
        raise ValueError('--index: например 1.3116 или 1.3116+0.1i') from exc
    if not math.isfinite(m.real) or not math.isfinite(m.imag) or m.real <= 0 or m.imag < 0:
        raise ValueError('Для текущего интерфейса нужны Re(m)>0, Im(m)>=0 и конечные компоненты')
    return m


def geometry(a):
    """Bound the enclosing radius and initial edge length; detect actual mesh groups."""
    import numpy as np
    shape = a.shape or 'prism'
    sym = dict(c6_z=False, vertical_mirror=False, horizontal_mirror=False)
    if a.particle is not None or a.particle_file is not None:
        from mbs_particle_geometry import prepare
        v, t, info = prepare(a.particle, a.particle_file)
        radius = info['equivalent_radius']
        center = np.asarray(info['symmetry']['center'])
        v = (v-center)/radius
        sym = info['solver_symmetries']
        shape = 'particle'
        return shape, float(np.max(np.linalg.norm(v,axis=1))), float(np.max(np.linalg.norm(v[t]-np.roll(v[t],1,axis=1),axis=2))), len(t), sym
    if a.obj:
        import trimesh
        mesh = trimesh.load(a.obj, force='mesh', process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
            raise ValueError('OBJ должен быть замкнутым с согласованными наружными нормалями')
        v, t = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        from surface_symmetry import PointGroup
        from mbs_particle_geometry import solver_symmetries
        detected = PointGroup.detect(v,t).summary()
        sym = solver_symmetries({'symmetry':detected})
        radius = (3*mesh.volume/(4*np.pi))**(1/3)
        # A conservative diameter bound also covers the native centering convention.
        extent = float(np.linalg.norm(np.ptp(v,axis=0))/radius)
        edge = float(np.max(np.linalg.norm(v[t]-np.roll(v[t],1,axis=1),axis=2))/radius)
        return 'obj', extent, edge, len(t), sym
    if shape == 'prism':
        width = (8*math.pi/(3*math.sqrt(3)*a.aspect))**(1/3)
        radius = width/math.sqrt(3)
        height = a.aspect*width
        sym = dict(c6_z=True,vertical_mirror=True,horizontal_mirror=True)
        return shape, math.hypot(radius,height/2), math.hypot(radius,height), 48, sym
    if shape == 'tetra':
        edge = (8*math.sqrt(2)*math.pi)**(1/3)
        return shape, edge*math.sqrt(6)/4, edge, 4, sym
    return shape, 1., 1.052, 20, sym


def resolve(a):
    import psutil
    m = index_value(a.index)
    finite_positive(a.wavelength,'wavelength'); finite_positive(a.aspect,'aspect')
    ka = finite_positive(a.ka if a.ka is not None else 2*math.pi*finite_positive(a.radius,'radius')/a.wavelength,'ka')
    accuracy = finite_positive(a.accuracy_percent,'accuracy-percent')/100
    if accuracy >= 1:raise ValueError('--accuracy-percent должен быть меньше 100')
    if not 1 <= a.threads <= (os.cpu_count() or 1):raise ValueError('Некорректное число потоков')
    if not 2 <= a.ntheta <= 100001:raise ValueError('--ntheta: 2..100001')
    if a.orientation is not None and (a.task != 'fixed' or not all(math.isfinite(x) for x in a.orientation)):
        raise ValueError('--orientation: три конечных угла только для --task fixed; полная T задаётся в системе частицы')
    if a.method != 'dirac' and (a.backend != 'auto' or a.global_patches is not None or a.batch_size is not None):
        raise ValueError('--backend/--global-patches/--batch-size относятся к Дираку')
    shape, enclosing, edge, base_panels, detected = geometry(a)
    if a.method == 'dirac' and shape not in ('sphere','prism'):
        raise ValueError('Дирак пока поддерживает только --shape sphere|prism; для этой формы выберите muller')
    if a.method == 'pmchwt' and a.task != 'fixed':
        raise ValueError('PMCHWT: T/SO(3) не подключены; выберите muller или dirac')
    if a.method == 'pmchwt' and shape == 'tetra':
        raise ValueError('PMCHWT: для тетраэдра используйте --obj')
    symmetry = a.symmetry
    if symmetry == 'auto':
        symmetry = ('d6h' if detected.get('c6_z') and detected.get('horizontal_mirror') and detected.get('vertical_mirror')
                    else 'c6v' if detected.get('c6_z') and detected.get('vertical_mirror')
                    else 'c6' if detected.get('c6_z') else 'none')
    required = {'c6':['c6_z'],'c6v':['c6_z','vertical_mirror'],'d6h':['c6_z','horizontal_mirror','vertical_mirror']}.get(symmetry,[])
    if any(not detected.get(k) for k in required):
        raise ValueError(f'Геометрия не имеет заявленной {symmetry} относительно осей тела')
    if a.method == 'pmchwt' and a.symmetry not in ('auto','none'):
        raise ValueError('PMCHWT fixed: симметрийные сокращения этого интерфейса не подключены')
    # P3 has higher spatial order than BDM1/RWG: ref numbers are not interchangeable.
    degree = 3 if ka >= 5 else 2
    ppw = (3./(degree+1) if a.method == 'dirac' else 4.) * max(1.,(.01/accuracy)**.25)
    ref = a.ref if a.ref is not None else max(1, math.ceil(math.log2(max(1., edge*ka*max(1.,abs(m))*ppw/(2*math.pi)))))
    if not 0 <= ref <= 9:raise ValueError('Требуется ref вне 0..9; задайте другой размер/сетку явно')
    kr = ka*enclosing
    order = a.L if a.L is not None else max(3,math.ceil(kr+4*kr**(1/3)+2+max(0.,math.log10(.01/accuracy))))
    if a.task == 'fixed':
        if a.L is not None:raise ValueError('--L относится к T, а не к fixed')
        order = 1
    if not 1 <= order <= (128 if a.method == 'dirac' else 256):raise ValueError('L вне диапазона выбранного метода')
    tol = a.tol if a.tol is not None else min(1e-8 if a.method == 'dirac' or ka<=1 else 1e-5,accuracy/1000)
    if not math.isfinite(tol) or not 1e-12 <= tol < 1:raise ValueError('--tol: от 1e-12 до 1 (не включая 1)')
    available = psutil.virtual_memory().available/GIB
    ram = min(a.ram_gib or max(.25, available-4), max(.25,available-2))
    if a.ram_gib is not None:finite_positive(a.ram_gib,'ram-gib')
    panels = base_panels*4**ref
    nb = (degree+1)*(degree+2)//2
    group = 12 if symmetry == 'd6h' else 6 if symmetry in ('c6','c6v') else 1
    dense_gib = 128*(panels*nb)**2/(12 if symmetry == 'd6h' else 1)/GIB
    backend = a.backend
    if a.method == 'dirac' and backend == 'auto':
        backend = 'dense' if dense_gib < .35*ram else 'fmm-sector'
    if a.method == 'dirac' and backend == 'fmm-sector' and (shape != 'prism' or symmetry not in ('c6v','d6h')):
        raise ValueError('Для большой задачи Дирака нужен fmm-sector с c6v/d6h призмой; уменьшите размер или выберите Мюллера')
    patches = a.global_patches or (512 if ka > 25 else 256)
    if not 2 <= patches <= 512:raise ValueError('--global-patches: 2..512')
    batch = a.batch_size or (2 if ka > 25 else 8)
    if not 1 <= batch <= 16:raise ValueError('--batch-size: 1..16')
    modes = 2*order*(order+2)
    output = (a.out or ROOT/'runs'/f'{a.method}_{a.task}_ka{ka:g}_{time.strftime("%Y%m%d_%H%M%S")}').expanduser().resolve()
    parent=output
    while not parent.exists():parent=parent.parent
    disk = shutil.disk_usage(parent).free/GIB
    disk_required = (6*16*modes*modes/GIB+1) if a.task != 'fixed' else .1
    if disk < disk_required+2:raise ValueError(f'Мало диска: нужно {disk_required+2:.2f} GiB с резервом, доступно {disk:.2f}')
    return dict(schema=1,method=a.method,task=a.task,shape=shape,ka=ka,index=[m.real,m.imag],
        wavelength_um=a.wavelength,radius_um=ka*a.wavelength/(2*math.pi),aspect=a.aspect,
        orientation_degrees=a.orientation or [0.,0.,0.],symmetry_requested=a.symmetry,symmetry_selected=symmetry,
        geometry_symmetries=detected,ref=ref,L=order,degree=degree,tolerance=tol,
        backend=backend if a.method=='dirac' else 'fmm',global_patches=patches,batch_size=batch,
        threads=a.threads,ntheta=a.ntheta,ram_budget_gib=ram,available_ram_gib=available,
        disk_free_gib=disk,disk_planning_gib=disk_required,out=str(output),
        estimated_base_mesh_panels=panels,dense_Dirac_operator_estimate_gib=dense_gib if a.method=='dirac' else None,
        accuracy_target=accuracy,accuracy_certified=False,
        parameter_selection='Initial wavelength/enclosing-sphere estimates; ref and L convergence must be checked separately.',
        memory_scope='Dirac native preflight plus process-tree RSS limit; FMM/ILU reserves are estimates, not allocator bounds.',
        particle=a.particle,particle_file=str(a.particle_file.resolve()) if a.particle_file else None,
        obj=str(a.obj.resolve()) if a.obj else None)


def command(plan, resume=False):
    p=plan; out=Path(p['out'])/'result'; py=sys.executable
    m=complex(*p['index']); env={'OMP_NUM_THREADS':str(p['threads']),'OPENBLAS_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE'}
    if p['method']=='dirac':
        module='dirac_plane.py' if p['task']=='fixed' else 'tmatrix.py'
        program=ROOT/'scripts'/module if p['task']=='fixed' else DIRAC/module
        cmd=[py,str(program),'--shape',p['shape'],'--ka',str(p['ka']),'--index',str(m),'--aspect',str(p['aspect']),
             '--ref',str(p['ref']),'--degree',str(p['degree']),'--order','24','--target-order','13','--near-subdiv','1',
             '--tol',str(p['tolerance']),'--backend',p['backend'],'--ram-budget-gib',str(p['ram_budget_gib']),
             '--kernel-budget-gib',str(max(.01,.40*p['ram_budget_gib']) if p['backend']=='dense'
                                        else min(8.,max(.01,.15*p['ram_budget_gib']))),'--out',str(out)]
        if p['task']=='fixed':
            cmd+=['--orientation',*map(str,p['orientation_degrees']),'--ntheta',str(p['ntheta']),
                  '--symmetry',p['symmetry_selected']]
        else:
            sym={'d6h':'c6hv'}.get(p['symmetry_selected'],p['symmetry_selected'])
            cmd+=['--L',str(p['L']),'--block-panels','1','--batch-size',str(p['batch_size']),
                  '--symmetry',sym,'--so3','mstm','--so3-threads',str(min(4,p['threads'])),
                  '--matrix-budget-gib',str(max(.05,.25*p['ram_budget_gib'])),
                  '--coarse-budget-gib',str(max(.02,.7*p['ram_budget_gib'])),'--ntheta',str(p['ntheta'])]
            if p['task']=='tmatrix':cmd+=['--no-average']
            if resume:cmd+=['--resume']
            if p['backend']=='dense':
                cmd+=['--dense-kernel','simd']
                if sym=='c6hv':cmd+=['--dense-cache','orbits12']
                if sym!='none':cmd+=['--sector-coarse-basis','3']
            else:
                cmd+=['--sector-coarse-basis','3','--coarse-factorization','hierarchical','--coarse-global-scope','full',
                      '--coarse-aggregate-groups',str(p['global_patches']),'--coarse-aggregate-basis','3',
                      '--fmm-eps',str(min(1e-8,p['tolerance'])),'--fmm-strict-eps',str(min(1e-12,p['tolerance']/100)),
                      '--fmm-packet','1']
        env['BEM_DIRAC_LIBRARY']=str(ROOT/'build/libdirac.so')
        return cmd,env
    if p['method']=='muller':
        cmd=[py,str(ROOT/'scripts/build_fmm_tmatrix.py'),'--ka',str(p['ka']),'--ri',str(m.real),'--ri-imag',str(m.imag),
             '--wavelength',str(p['wavelength_um']),'--ref',str(p['ref']),'--order',str(p['L']),
             '--tol',str(p['tolerance']),'--max-iters','1000','--ntheta',str(p['ntheta']),
             '--so3-precision','fp64','--so3-threads',str(min(4,p['threads'])),'--release-incident-buffer','--out',str(out)]
        if p['particle'] is not None:cmd+=['--particle',*map(str,p['particle'])]
        elif p['particle_file']:cmd+=['--particle-file',p['particle_file']]
        elif p['obj']:cmd+=['--obj',p['obj'],'--obj-symmetry']
        elif p['shape']=='prism' and p['symmetry_requested']=='auto':
            cmd+=['--particle','1',str(p['aspect']*math.sqrt(3)/2),'1']
        else:cmd+=['--shape',p['shape'],'--prism-aspect',str(p['aspect'])]
        # Explicit policy must not be broadened by the particle driver's auto dispatch.
        cmd+=['--particle-symmetry','auto' if p['symmetry_requested']=='auto' else 'off']
        sym=p['symmetry_selected']
        if p['task']=='fixed':cmd+=['--plane-only','--no-average','--plane-euler',*map(str,p['orientation_degrees'])]
        else:
            if p['task']=='tmatrix':cmd+=['--no-average']
            if sym!='none':cmd+=['--c6-pack']
            if sym=='d6h':cmd+=['--c2-reuse']
            elif sym=='c6v':cmd+=['--vertical-reuse']
            if sym=='d6h':cmd+=['--mirror-pack']
        if sym!='d6h':env['BEM_TMATRIX_SECTOR_GMRES']='0'
        elif sym=='d6h':
            env.update(BEM_TMATRIX_D6H_MESH='1',BEM_TMATRIX_SECTOR_GMRES='1',
                       BEM_TMATRIX_VERTICAL_PRECONDITIONER='1',BEM_TMATRIX_ORBIT_NODES='50')
        env.update(BEM_TMATRIX_PRISM_ASPECT=str(p['aspect']),BEM_TMATRIX_RESIDENT_MEMORY_MB=str(max(64,min(2048,int(p['ram_budget_gib']*64)))))
        if p['ka']<=1:env.update(BEM_FARFIELD_FAST_PHASE='0',BEM_FARFIELD_FP32_ACCUM='0')
        if p['obj'] and sym=='d6h':env['BEM_TMATRIX_GEOMETRY_D6H']='1'
        return cmd,env
    cmd=[str(ROOT/'bin/bem_cuda_fmm'),'--solver','fmm','--system','pmchwt','--ka',str(p['ka']),'--ri',str(m.real),str(m.imag),
         '--ref',str(p['ref']),'--fmm-digits',str(min(8,max(5,math.ceil(-math.log10(p['tolerance']))))),
         '--prec','auto','--krylov','gpu-gmres','--gmres-tol',str(p['tolerance']),'--scat-plane','xz','--force-orient',
         '--ntheta',str(p['ntheta']),'--orient-file',str(Path(p['out'])/'orientation.txt'),'--out',str(out/'result.json')]
    if p['shape'] in ('sphere','prism'):cmd+=['--shape','sphere' if p['shape']=='sphere' else 'hex_prism','--prism-aspect',str(p['aspect'])]
    else:cmd+=['--obj',p['obj'] or str(Path(p['out'])/'particle.obj'),'--subdiv',str(p['ref'])]
    return cmd,env


def ensure_build(p,no_build):
    builds=[]
    if p['method']=='dirac':
        output=ROOT/'build/libdirac.so'
        inputs=[ROOT/'src/dirac_transmission.cpp',ROOT/'src/dirac_transmission.h',ROOT/'src/mesh.cpp',DIRAC/'binding.cpp']
        if not output.exists() or any(x.stat().st_mtime>output.stat().st_mtime for x in inputs):
            builds.append([sys.executable,str(DIRAC/'build.py'),'--fast-complex','--output',str(output)])
        if p['backend']=='fmm-sector' and not (ROOT/'benchmarks/clifford_fmm_20261115/FMM3D/lib/libfmm3d.so').exists():
            builds.insert(0,[sys.executable,str(ROOT/'benchmarks/clifford_fmm_20261115/build.py'),'--jobs',str(min(8,p['threads']))])
    else:
        target='bin/muller_tmatrix_fmm_server' if p['method']=='muller' else 'bin/bem_cuda_fmm'
        if not (ROOT/target).exists():builds.append(['make',target,'CXX=g++-12','CUDA_HOME=/usr','-j2'])
    if p['task']=='average' and not (ROOT/'bin/tmatrix_so3_mstm').exists():
        builds.append(['make','bin/tmatrix_so3_mstm'])
    if no_build and builds:raise ValueError('Нужна сборка: '+shlex.join(builds[0])+' (уберите --no-build)')
    for cmd in builds:subprocess.run(cmd,cwd=ROOT,check=True)


def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def run_plan(p,no_build=False,resume=False):
    out=Path(p['out'])
    if not resume and (out/'calculation.json').exists():raise ValueError(f'Каталог занят; используйте ./bem continue {out}')
    ensure_build(p,no_build)
    out.mkdir(parents=True,exist_ok=True)
    (out/'result').mkdir(exist_ok=True)
    if p['method']=='pmchwt':
        (out/'orientation.txt').write_text(' '.join(map(str,p['orientation_degrees']))+' 1\n')
        if p['shape']=='particle':
            from mbs_particle_geometry import prepare,write_obj
            v,t,_=prepare(p['particle'],Path(p['particle_file']) if p['particle_file'] else None)
            write_obj(out/'particle.obj',v,t)
        elif p['shape']=='tetra':raise ValueError('PMCHWT: используйте OBJ вместо tetra')
    cmd,settings=command(p,resume)
    # Reproducibility: a previous experiment's BEM_* environment must not change this plan.
    env={k:v for k,v in os.environ.items() if not k.startswith('BEM_')}
    env.update(settings)
    if not resume:write_json(out/'calculation.json',p)
    write_json(out/'launch.json',dict(command=cmd,environment=settings,resume=resume))
    from bounded_bem_run import run
    def progress(state):
        detail=''
        try:
            v=json.loads((out/'result/progress.json').read_text())
            if 'completed_columns' in v:
                detail=f"; столбцы {v['completed_columns']}/{v.get('total_columns',v.get('mode_count','?'))}"
        except (OSError,ValueError):pass
        print(f"Выполняется: {state['elapsed_s']/60:.1f} мин; ОЗУ {state['rss_gib']:.2f} GiB{detail}",flush=True)
    monitor=out/('monitor_resume_'+str(time.time_ns()) if resume else 'monitor')
    state=run(cmd,monitor,env=env,
              rss_limit_gib=p['ram_budget_gib'],available_gib=2,disk_reserve_gib=2,progress_callback=progress)
    result=dict(state=state['state'],method=p['method'],task=p['task'],accuracy_certified=False,
                target_accuracy_percent=100*p['accuracy_target'],result_directory=str(out/'result'),
                reason=state.get('reason'),peak_rss_gib=state.get('peak_rss_gib'),elapsed_s=state.get('elapsed_s'))
    if state['state']=='complete':
        import numpy as np
        from scattering_frame import save_mueller
        if p['task']=='average' and p['method']=='dirac':
            with np.load(out/'result/solution.npz') as z:save_mueller(out/'result',z['theta'],z['mueller'])
        elif p['task']=='average' and p['method']=='muller':
            data=json.loads((out/'result/average.json').read_text())
            save_mueller(out/'result',np.asarray(data['theta']),np.moveaxis(data['mueller'],-1,0))
    write_json(out/'result_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if state['state']!='complete':print('Подробности ошибки: '+str(monitor/'worker.log'),file=sys.stderr)
    return 0 if state['state']=='complete' else 1


def execute(a):
    try:
        p=resolve(a);cmd,env=command(p)
        print(json.dumps(p,ensure_ascii=False,indent=2),flush=True)
        print('Команда: '+shlex.join(cmd),flush=True)
        if a.dry_run:return 0
        return run_plan(p,a.no_build)
    except (ValueError,OSError,RuntimeError,subprocess.CalledProcessError) as error:
        print('bem: '+str(error),file=sys.stderr);return 2


def continue_run(a):
    try:
        p=json.loads((a.directory/'calculation.json').read_text())
        p['out']=str(a.directory.resolve())
        if p.get('schema')!=1:raise ValueError('Неизвестная версия calculation.json')
        summary=a.directory/'result_summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('state')=='complete':
            result=a.directory/'result'
            required=([result/'tmatrix.npy'] if p['task']!='fixed' else
                      [result/('result.json' if p['method']=='pmchwt' else 'mueller.npz')])
            if p['task']=='average':required.append(result/'mueller.npz')
            if all(path.is_file() and path.stat().st_size>0 for path in required):
                print('Расчёт уже завершён. Сохранённые результаты: '+str(result.resolve()))
                print('Повторный расчёт и повторная проверка столбцов не запускались.')
                return 0
        return run_plan(p,resume=True)
    except (ValueError,OSError,RuntimeError,subprocess.CalledProcessError) as error:
        print('bem: '+str(error),file=sys.stderr);return 2


def interactive(parser):
    if not sys.stdin.isatty():
        print('wizard требует терминал; используйте ./bem solve --help',file=sys.stderr);return 2
    show_methods()
    def ask(label,default):return input(f'{label} [{default}]: ').strip() or default
    method=ask('Метод','muller');task=ask('Расчёт fixed/tmatrix/average','average')
    shape=ask('Форма sphere/prism/tetra/mbs/obj','prism')
    argv=['--method',method,'--task',task]
    if shape=='mbs':argv+=['--particle',*ask('Тип MBS и параметры (например bullet: 2 2 1)','2 2 1').split()]
    elif shape=='obj':argv+=['--obj',ask('Путь к OBJ','particle.obj')]
    else:argv+=['--shape',shape]
    argv+=['--ka',ask('ka','3'),'--index',ask('Показатель n+i*kappa','1.3116'),
           '--wavelength',ask('Длина волны, мкм','0.532'),
           '--aspect',ask('Для призмы H/ширина между гранями','1.4285714285714286'),
           '--symmetry',ask('Симметрия auto/none/c6/c6v/d6h','auto')]
    if task=='fixed':argv+=['--orientation',*ask('Ориентация alpha beta gamma, градусы','0 0 0').split()]
    argv+=['--out',ask('Каталог результата','runs/example')]
    args=parser.parse_args(argv)
    print('Повторяемый запуск: ./bem solve '+shlex.join(argv))
    return execute(args)
