"""MBS-fast particle parameters, closed BEM surfaces and measured symmetry.

Dimensions follow MBS-fast: hexagon D is corner-to-corner, L is the prismatic
body length, droxtal SCALE is its construction radius, fixed aggregate SCALE
is Dmax. BEM ka remains the equal-volume sphere size parameter.
"""
import argparse
from collections import defaultdict,deque
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from surface_symmetry import PointGroup

ROOT=Path(__file__).resolve().parents[1]
NAMES={1:'hexagonal',2:'bullet',3:'bullet_rosette',4:'droxtal',
       10:'concave_hexagonal',12:'two_column_aggregate',999:'fixed_aggregate'}


def ring(radius,z):
    a=np.arange(6)*np.pi/3
    return np.c_[radius*np.cos(a),radius*np.sin(a),np.full(6,z)]


def sides(a,b):
    return [np.array([a[i],a[(i+1)%6],b[(i+1)%6],b[i]]) for i in range(6)]


def column(length,diameter,cap=None,cavity=None):
    a,b=ring(diameter/2,length/2),ring(diameter/2,-length/2)
    faces=sides(a,b)
    if cavity is not None:
        depth=diameter/2*np.tan(np.deg2rad(cavity))
        for polygon,z in ((a,length/2-depth),(b,-length/2+depth)):
            faces.extend(np.array([polygon[i],polygon[(i+1)%6],[0,0,z]]) for i in range(6))
    elif cap is not None:
        faces.append(b)
        faces.extend(np.array([a[i],a[(i+1)%6],[0,0,length/2+cap]]) for i in range(6))
    else:faces.extend((a,b))
    return faces


def builtin(spec):
    values=np.asarray(spec,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('Finite particle parameters required')
    kind=int(values[0]);p=values[1:]
    if values[0]!=kind or kind not in NAMES:raise ValueError('Particle type must be 1, 2, 3, 4, 10, 12 or 999')
    arity={1:(2,),2:(2,),3:(2,3),4:(1,3),10:(3,),12:(3,),999:(1,)}
    if len(p) not in arity[kind]:raise ValueError(f'Wrong number of parameters for {NAMES[kind]}')
    if kind in (1,2,3,10,12) and np.any(p[:2]<=0):raise ValueError('L and D must be positive')
    if kind in (1,2,3,10,12):length,diameter=p[:2]
    if kind==1:return column(length,diameter)
    if kind in (2,3):
        cap=p[2] if len(p)==3 else diameter*np.sqrt(3)*np.tan(np.deg2rad(62))/4
        if cap<=0:raise ValueError('Cap height must be positive')
        faces=column(length,diameter,cap=cap)
        if kind==2:return faces
        moved=[f-np.array([0,0,length/2+cap+1]) for f in faces]
        result=[]
        for beta,alpha in ((0,0),(90,0),(180,0),(270,0),(90,90),(90,270)):
            b,a=np.deg2rad([beta,alpha]);c,s=np.cos(b),np.sin(b)
            ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            c,s=np.cos(a),np.sin(a);rz=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            result.extend(f@(rz@ry).T for f in moved)
        return result
    if kind==4:
        radius=p[-1]
        if radius<=0:raise ValueError('Droxtal radius must be positive')
        a,b=np.deg2rad([32.35,71.81])
        levels=[ring(radius*np.sin(t),sign*radius*np.cos(t))
                for t,sign in ((a,1),(b,1),(b,-1),(a,-1))]
        return [levels[0],levels[-1]]+[f for i in range(3) for f in sides(levels[i],levels[i+1])]
    if kind==10:
        angle=p[2]
        if not 0<angle<np.rad2deg(np.arctan(length/diameter)):raise ValueError('Cavity tips must remain distinct: 0 < angle < atan(L/D)')
        return column(length,diameter,cavity=angle)
    if kind==12:
        if p[2]!=2:raise ValueError('MBS-fast type 12 defines exactly two columns')
        offset=length/2+diameter/2+1;faces=column(length,diameter)
        return [f+[0,0,offset] for f in faces]+[f[:,[0,2,1]]+[0,offset,0] for f in faces]
    if p[0]<=0:raise ValueError('Aggregate Dmax must be positive')
    data=json.loads((ROOT/'reference/particles/mbs_fixed_aggregate.json').read_text())
    faces=[np.array(f,float) for f in data['facets']]
    scale=p[0]/np.max(pdist(np.vstack(faces)))
    return [f*scale for f in faces]


def read_particle(path):
    """Read native MBS polygon files; header symmetry is never trusted."""
    lines=Path(path).read_text().splitlines();headers=[];faces=[];current=[]
    for line in lines:
        text=line.split('#',1)[0].strip()
        if len(headers)<3:
            if text:headers.append(text)
            continue
        if not text:
            if current:faces.append(np.array(current,float));current=[]
        else:
            values=[float(x) for x in text.split()]
            if len(values)!=3:raise ValueError('Particle vertex must contain three coordinates')
            current.append(values)
    if current:faces.append(np.array(current,float))
    if len(headers)!=3 or not faces:raise ValueError('Missing particle headers or facets')
    for text in headers:
        if not np.isfinite([float(x) for x in text.split()]).all():raise ValueError('Nonfinite particle header')
    return faces


def triangulate(facets):
    """Centroid fans preserve polygon rotations and mirrors on convex facets."""
    points=[];triangles=[];keys={}
    scale=max(np.linalg.norm(np.ptp(np.vstack(facets),axis=0)),1e-100)
    def vertex(p):
        key=tuple(np.rint(np.asarray(p)/scale*1e11).astype(np.int64))
        if key not in keys:keys[key]=len(points);points.append(p)
        return keys[key]
    for polygon in facets:
        polygon=np.asarray(polygon,float)
        if polygon.ndim!=2 or polygon.shape[1]!=3 or len(polygon)<3 or not np.isfinite(polygon).all():raise ValueError('Invalid particle facet')
        normal=np.cross(polygon[1]-polygon[0],polygon[2]-polygon[0]);norm=np.linalg.norm(normal)
        if norm<1e-14*scale**2:raise ValueError('Degenerate facet')
        normal/=norm
        if np.max(abs((polygon-polygon[0])@normal))>1e-8*scale:raise ValueError('Nonplanar facet')
        turns=np.cross(np.roll(polygon,-1,axis=0)-polygon,np.roll(polygon,-2,axis=0)-np.roll(polygon,-1,axis=0))@normal
        if np.min(turns)<-1e-10*scale**2:raise ValueError('Facet must be convex; subdivide concave polygons before import')
        ids=[vertex(p) for p in polygon]
        if len(ids)==3:triangles.append(ids)
        else:
            mid=vertex(polygon.mean(axis=0))
            triangles.extend([ids[i],ids[(i+1)%len(ids)],mid] for i in range(len(ids)))
    vertices=np.array(points,float);triangles=np.array(triangles,int)
    edges=defaultdict(list)
    for t,row in enumerate(triangles):
        for a,b in zip(row,np.roll(row,-1)):
            edges[tuple(sorted((a,b)))].append((t,1 if a<b else -1))
    if any(len(v)!=2 for v in edges.values()):raise ValueError('BEM needs a closed manifold: every edge must have two incident triangles')
    neighbors=defaultdict(list)
    for (a,sa),(b,sb) in edges.values():neighbors[a].append((b,-sa*sb));neighbors[b].append((a,-sa*sb))
    signs={};components=[]
    for seed in range(len(triangles)):
        if seed in signs:continue
        signs[seed]=1;queue=deque([seed]);component=[]
        while queue:
            a=queue.popleft();component.append(a)
            for b,factor in neighbors[a]:
                expected=signs[a]*factor
                if b in signs:
                    if signs[b]!=expected:raise ValueError('Nonorientable boundary')
                else:signs[b]=expected;queue.append(b)
        components.append(component)
    for t,sign in signs.items():
        if sign<0:triangles[t]=triangles[t,[0,2,1]]
    volumes=[]
    for component in components:
        v=vertices[triangles[component]]
        volume=np.sum(np.einsum('ij,ij->i',v[:,0],np.cross(v[:,1],v[:,2])))/6
        if abs(volume)<1e-14*scale**3:raise ValueError('Zero-volume component')
        if volume<0:triangles[component]=triangles[component][:,[0,2,1]]
        volumes.append(abs(volume))
    return vertices,triangles,dict(components=len(components),component_volumes=volumes,
        volume=sum(volumes),triangles=len(triangles),vertices=len(vertices),closed=True,
        equivalent_radius=float((3*sum(volumes)/(4*np.pi))**(1/3)),dmax=float(np.max(pdist(vertices))))


def write_obj(path,vertices,triangles):
    text=''.join('v '+' '.join(format(x,'.17g') for x in p)+'\n' for p in vertices)
    text+=''.join('f '+' '.join(str(int(i)+1) for i in t)+'\n' for t in triangles)
    Path(path).write_text(text)


def solver_symmetries(report):
    matrices=np.asarray(report['symmetry']['matrices']);tolerance=report['symmetry']['tolerance']
    def contains(R):return bool(np.any(np.max(abs(matrices-R),axis=(1,2))<tolerance))
    a=np.pi/3;c,s=np.cos(a),np.sin(a)
    c6=contains(np.array([[c,-s,0],[s,c,0],[0,0,1]]))
    horizontal=contains(np.diag([1,1,-1]));vertical=contains(np.diag([1,-1,1]))
    return dict(c6_z=c6,c2_x=contains(np.diag([1,-1,-1])),horizontal_mirror=horizontal,
                vertical_mirror=vertical,d6h_z=c6 and horizontal and vertical)


def prepare(spec=None,particle_file=None):
    if (spec is None)==(particle_file is None):raise ValueError('Choose particle parameters or particle-file')
    facets=builtin(spec) if spec is not None else read_particle(particle_file)
    # MBS aggregates consist of closed convex components. An overlapping set
    # of internal faces is not a valid single-material BEM boundary.
    if spec is not None and int(spec[0]) in (3,12,999):
        count=13 if int(spec[0])==3 else 8
        parts=[np.vstack(facets[i:i+count]) for i in range(0,len(facets),count)]
        scale=max(np.linalg.norm(np.ptp(np.vstack(parts),axis=0)),1e-100)
        hulls=[ConvexHull(p/scale).equations for p in parts]
        for i in range(len(parts)):
            for j in range(i):
                planes=np.vstack((hulls[i],hulls[j]))
                result=linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],np.ones(len(planes))],
                    b_ub=-planes[:,3],bounds=[(None,None)]*4,method='highs')
                if not result.success:raise ValueError('Could not verify aggregate separation')
                if result.x[3]>=-1e-10:
                    raise ValueError('Aggregate components overlap/touch: a union boundary is required for BEM')
    vertices,triangles,report=triangulate(facets)
    group=PointGroup.detect(vertices,triangles)
    report.update(name=NAMES[int(float(spec[0]))] if spec is not None else 'imported_particle',
                  particle=list(map(float,spec)) if spec is not None else None,
                  symmetry=group.summary(),size_convention='ka = k * equal-volume radius; input D is across hexagon corners')
    report['solver_symmetries']=solver_symmetries(report)
    return vertices,triangles,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--particle','-p',nargs='+',type=float)
    choice.add_argument('--particle-file','--pf',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    vertices,triangles,report=prepare(args.particle,args.particle_file)
    args.out.parent.mkdir(parents=True,exist_ok=True);write_obj(args.out,vertices,triangles)
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='symmetry'},indent=2))


if __name__=='__main__':main()
