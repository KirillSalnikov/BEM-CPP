"""Real C++/Python socket exchange, wrong geometry, and changed assets."""
import json
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compressed_fft_service import MAGIC,check_handshake,receive_into,prepare_manifest,validate_manifest

META=dict(system_dofs=9,ref=6,ka=40.,ri=[1.3116,0.],aspect=1.6495721976846451,quadrature=7,duffy=6)


class ExternalFFTTests(unittest.TestCase):
    def test_cpp_transport_repeated_inplace_and_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary);binary=path/'check';sockpath=path/'fft.sock'
            subprocess.run(['g++','-O2','-std=c++11','-I'+str(ROOT/'src'),str(ROOT/'tests/external_fft_action_check.cpp'),'-o',str(binary)],check=True)
            failures=[]
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
                listener.bind(str(sockpath));listener.listen(1)
                def service():
                    try:
                        with listener.accept()[0] as connection:
                            connection.settimeout(10)
                            header=bytearray(80);receive_into(connection,header);check_handshake(header,META)
                            connection.sendall(struct.pack('=QQ',MAGIC,9))
                            for seq in range(1,4):
                                header=bytearray(16);receive_into(connection,header)
                                self.assertEqual(struct.unpack('=QQ',header),(1,seq))
                                value=np.empty(9,complex);receive_into(connection,value);value*=3+2j
                                connection.sendall(struct.pack('=QQ',seq,0));connection.sendall(value.tobytes())
                            header=bytearray(16);receive_into(connection,header)
                            self.assertEqual(struct.unpack('=QQ',header),(0,4))
                    except BaseException as error:failures.append(error)
                thread=threading.Thread(target=service,daemon=True);thread.start()
                result=subprocess.run([str(binary),str(sockpath)],capture_output=True,text=True,timeout=15)
                thread.join(12)
                self.assertFalse(thread.is_alive());self.assertFalse(failures,failures)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_geometry_and_asset_mutations_are_rejected(self):
        values=[1,9,6,40.,1.3116,0.,META['aspect'],7,6]
        check_handshake(struct.pack('=Q9d',MAGIC,*values),META)
        for index in (1,2,3,4,5,6,7,8):
            wrong=values.copy();wrong[index]+=.01
            with self.assertRaises(ValueError):check_handshake(struct.pack('=Q9d',MAGIC,*wrong),META)
        with tempfile.TemporaryDirectory() as temporary:
            case=Path(temporary)
            (case/'geometry.json').write_text(json.dumps(META))
            (case/'streamed_preparation.json').write_text(json.dumps(dict(status='complete',records=[])))
            for name in ('bound_operator.pkl','near_rows.bin','near_cols.bin','near_values.bin','mass_rows.bin','mass_cols.bin','mass_values.bin'):
                (case/name).write_bytes(b'test')
            (case/'service_assets.json').write_text(json.dumps(prepare_manifest(case)))
            validate_manifest(case)
            (case/'near_values.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):validate_manifest(case)


if __name__=='__main__':unittest.main()
