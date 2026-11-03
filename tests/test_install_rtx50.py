import importlib.util,io,json,subprocess,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import cuda_redist as cuda
import install_rtx50 as setup

class InstallerTests(unittest.TestCase):
    def test_latest_stable_excludes_newer_preview(self):
        page=b'<h2>Latest Release</h2><a>CUDA Toolkit 13.4.3</a><h2>Archived Releases</h2><a>CUDA Toolkit 13.5.0 Developer Preview</a>'
        with patch.object(cuda,'fetch',return_value=page):self.assertEqual(cuda.latest_release(),'13.4.3')

    def test_changed_archive_format_requires_explicit_version(self):
        with patch.object(cuda,'fetch',return_value=b'Unexpected page'):
            with self.assertRaises(RuntimeError):cuda.latest_release()

    def test_wrong_archive_hash_never_installed(self):
        entry=dict(relative_path='cuda/test.tar.xz',size='3',sha256='0'*64)
        with tempfile.TemporaryDirectory() as temp,patch.object(cuda.urllib.request,'urlopen',side_effect=lambda *a,**k:io.BytesIO(b'bad')),patch.object(cuda.time,'sleep'):
            root=Path(temp)
            with self.assertRaises(RuntimeError):cuda.download(entry,root)
            self.assertFalse(list(root.iterdir()))

    def test_archive_traversal_and_external_symlink_rejected(self):
        for unsafe in ['path','link']:
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);archive=root/'bad.tar';target=root/'target';target.mkdir()
                with tarfile.open(archive,'w') as out:
                    info=tarfile.TarInfo('../escaped' if unsafe=='path' else 'root/link')
                    if unsafe=='link':info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd'
                    out.addfile(info)
                with self.assertRaises(ValueError):cuda.extract(archive,target)
                self.assertFalse((root/'escaped').exists())

    def test_sm120_profiles_keep_precise_math(self):
        plans=setup.commands('mbs-fast',Path('/cuda'),'/g++',2)
        self.assertEqual([name for name,cmd in plans],['cpu','gpu-fp64','gpu-fp32'])
        for name,cmd in plans[1:]:
            self.assertIn('GPU_ARCH=120',cmd);self.assertIn('GPU_FAST_MATH=0',cmd);self.assertIn('GPU_PHASE_TRIG=precise',cmd);self.assertIn('GPU_LTO=1',cmd)
        for name,cmd in setup.commands('bem-cpp',Path('/cuda'),'/g++',2):
            self.assertIn('ARCH=-arch=sm_120',cmd);self.assertIn('CUDA_CPP_STD=c++17',cmd)
        with self.assertRaises(ValueError):setup.commands('mbs-fast',Path('/cuda'),'/g++',2,arch=86)

    def test_snapshot_excludes_existing_builds_and_python_packages(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);subprocess.run(['git','init','-q',str(root)],check=True)
            for name in ['src/main.cpp','.venv/lib/big.py','build/result.o','bin/solver','scripts/new_installer.py']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('sample')
            paths=[str(p) for p in setup.source_files(root)]
            self.assertEqual(paths,['scripts/new_installer.py','src/main.cpp'])

    def test_wrapper_quotes_paths_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'with spaces'/'wrapper'
            setup.wrapper(p,['/bin/printf','%s\n'],{'LD_LIBRARY_PATH':'/path with spaces'})
            result=subprocess.run([str(p),'$(false)','argument with spaces'],capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout,'$(false)\nargument with spaces\n')

if __name__=='__main__':unittest.main()
