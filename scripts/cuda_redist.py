"""Install NVIDIA's checksummed toolkit redistributions without a display driver."""
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,re,shutil,tarfile,tempfile,time,urllib.request

BASE='https://developer.download.nvidia.com/compute/cuda/redist/'
ARCHIVE='https://developer.nvidia.com/cuda-toolkit-archive'
COMPONENTS=['cuda_nvcc','cuda_crt','cuda_cudart','libnvvm','cccl','libnvjitlink','libnvptxcompiler','cuda_culibos','cuda_cuobjdump','cuda_profiler_api','libcufft','libcublas','libcusparse']

def fetch(url):
    with urllib.request.urlopen(url,timeout=60) as response:return response.read()

def latest_release():
    page=fetch(ARCHIVE).decode()
    current=re.search(r'Latest Release(.{0,3000}?)Archived Releases',page,re.S|re.I)
    release=re.search(r'CUDA Toolkit\s+(\d+\.\d+\.\d+)',current[1]) if current else None
    if not release:raise RuntimeError('Cannot resolve the latest stable release from NVIDIA; supply --cuda-version explicitly')
    return release[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def download(entry,cache):
    rel=PurePosixPath(entry['relative_path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid NVIDIA redistribution path')
    path=cache/rel.name
    if path.exists() and path.stat().st_size==int(entry['size']) and sha(path)==entry['sha256']:return path
    partial=path.with_suffix(path.suffix+'.partial')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(BASE+str(rel),timeout=90) as response,partial.open('wb') as out:shutil.copyfileobj(response,out,4*1024*1024)
            if partial.stat().st_size!=int(entry['size']) or sha(partial)!=entry['sha256']:raise RuntimeError('NVIDIA archive size/SHA256 mismatch: '+rel.name)
            partial.replace(path);return path
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt==3:raise
            time.sleep(2*(attempt+1))

def extract(path,destination):
    with tempfile.TemporaryDirectory(dir=destination.parent,prefix='.cuda-extract-') as temp:
        staging=Path(temp)
        with tarfile.open(path,'r:*') as archive:
            members=archive.getmembers()
            for member in members:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or member.isdev() or member.isfifo():raise ValueError('Unsafe NVIDIA archive entry')
                if member.issym() or member.islnk():
                    target=PurePosixPath(member.linkname)
                    if target.is_absolute() or '..' in target.parts:raise ValueError('Unsafe NVIDIA archive link')
            # NVIDIA tarballs are trusted only after SHA256 validation above.
            if hasattr(tarfile,'data_filter'):archive.extractall(staging,filter='data')
            else:archive.extractall(staging)  # All member names/links were validated above.
        roots=list(staging.iterdir())
        if len(roots)!=1 or not roots[0].is_dir():raise RuntimeError('Unexpected NVIDIA archive layout')
        for license in roots[0].glob('*'):
            if license.is_file() and any(k in license.name.lower() for k in ['license','eula']):
                where=destination/'licenses'/path.name;where.mkdir(parents=True,exist_ok=True);shutil.copy2(license,where/license.name)
        shutil.copytree(roots[0],destination,dirs_exist_ok=True,symlinks=True)

def install(version,root,cache):
    if version=='latest':version=latest_release()
    if not re.fullmatch(r'\d+\.\d+\.\d+',version):raise ValueError('CUDA version must be X.Y.Z or latest')
    root=Path(root).expanduser().resolve()/version;cache=Path(cache).expanduser().resolve()/version
    root.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(fetch(BASE+'redistrib_'+version+'.json'))
    if metadata.get('release_label')!=version:raise RuntimeError('Unexpected NVIDIA release manifest')
    entries=[]
    for name in COMPONENTS:
        alias='cuda_cccl' if name=='cccl' and 'cccl' not in metadata else name
        if alias not in metadata:
            if name in ['cuda_profiler_api','cuda_culibos']:continue
            raise RuntimeError('Missing NVIDIA component: '+name)
        entry=dict(metadata[alias]['linux-x86_64'],component=alias,version=metadata[alias]['version']);entries.append(entry)
    with ThreadPoolExecutor(max_workers=3) as pool:paths=list(pool.map(lambda e:download(e,cache),entries))
    for entry,path in zip(entries,paths):
        marker=root/('.installed-'+entry['component']+'.json')
        if marker.exists() and json.loads(marker.read_text()).get('sha256')==entry['sha256']:continue
        print('Installing CUDA component:',entry['component'],flush=True);extract(path,root)
        marker.write_text(json.dumps(entry,indent=2)+'\n')
    # The redistributions use lib/, whereas conventional Linux toolkits expose lib64/.
    if (root/'lib').is_dir() and not (root/'lib64').exists():(root/'lib64').symlink_to('lib',target_is_directory=True)
    for required in ['bin/nvcc','include/cuda_runtime.h','include/crt/host_config.h','nvvm/libdevice/libdevice.10.bc']:
        if not (root/required).exists():raise RuntimeError('Incomplete CUDA toolkit: '+required)
    report=dict(version=version,manifest_url=BASE+'redistrib_'+version+'.json',components=entries,driver_installed=False)
    (root/'redistribution-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return root

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--version',default='latest');p.add_argument('--root',type=Path,default=Path.home()/'.local/share/cuda');p.add_argument('--cache',type=Path,default=Path.home()/'.cache/cuda-redist');a=p.parse_args();print(install(a.version,a.root,a.cache))
