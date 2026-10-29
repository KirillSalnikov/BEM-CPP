#!/usr/bin/env python3
"""CPU-only protocol fixture: a known C6/C2 linear map, not a BEM solver."""
import json
import os
from pathlib import Path
import socket
import struct
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tmatrix_incident import regular_fields
from tmatrix_reference import modes, transverse_harmonics, tangent_frame
from tmatrix_symmetry import c2x_plan


def receive(connection, array):
    view = memoryview(array).cast('B')
    while view:
        n = connection.recv_into(view)
        if n == 0:
            raise EOFError
        view = view[n:]


def main():
    directions = np.fromfile(sys.argv[7], dtype=float).reshape(-1, 3)
    band = round(np.sqrt((len(directions)-10)/2))
    order = (band-4)//2
    modal = modes(order)
    ka = float(sys.argv[4])
    rng = np.random.default_rng(678)
    positions = rng.normal(size=(max(48, len(modal)), 3))
    positions /= np.linalg.norm(positions, axis=1)[:, None]
    positions *= rng.uniform(.7, 1.2, (len(positions), 1))
    basis = np.stack([regular_fields(positions, mode, ka).ravel() for mode in modal], axis=1)
    scales = np.max(np.abs(basis), axis=0)
    inverse = np.linalg.pinv(basis/scales)
    q = len(modal)
    kernel = (rng.normal(size=(q,q))+1j*rng.normal(size=(q,q)))*.02
    ms = np.array([m for p,l,m in modal])
    kernel[(ms[:,None]-ms[None,:])%6 != 0] = 0
    mirror=os.environ.get('BEM_TMATRIX_D6H_MESH')=='1'
    if mirror:
        parity=np.array([(p+l+m)%2 for p,l,m in modal])
        kernel[parity[:,None]!=parity[None,:]]=0
    permutation, signs, _, _ = c2x_plan(order)
    kernel = .5*(kernel+signs[:,None]*kernel[np.ix_(permutation,permutation)]*signs[None,:])
    phase = np.array([1j**(l-p) for p,l,m in modal])
    outgoing = transverse_harmonics(directions, order)
    _, _, et, ep = tangent_frame(directions)
    fields = np.empty((2, len(positions), 6), complex)
    audit_degrees = {min(order, 5), max(1, order//2)}
    injection = os.environ.get('BEM_TEST_FAKE_CROSS_TALK', '')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(sys.argv[1]); server.listen(1)
        with server.accept()[0] as connection:
            def message(value):
                data = json.dumps(value).encode()
                connection.sendall(struct.pack('=Q',len(data))+data)
            message({'rhs_points':len(positions),'system_dofs':len(positions)*6,
                     'solver_backend':'CPU_TEST_FIXTURE_NOT_BEM','prism_dihedral_mesh':mirror,
                     'symmetry_warm_start':os.environ.get('BEM_TEST_FAKE_WARM_SUPPORT')=='1',
                     'krylov_recycle_capacity':int(os.environ.get('BEM_TMATRIX_RECYCLE_RANK','0')),
                     'recycle_signature':os.environ.get('BEM_TMATRIX_RECYCLE_SIGNATURE'),
                     'recycle_path':os.environ.get('BEM_TMATRIX_RECYCLE_PATH')})
            connection.sendall(positions.astype(float).tobytes())
            while True:
                operation = np.empty(1, np.int32); receive(connection,operation)
                if operation[0] == 0:
                    break
                if operation[0] in (1,3):
                    receive(connection, fields)
                    coefficients = inverse@fields.reshape(2,-1).T
                    outgoing_coefficients = kernel@(coefficients/scales[:,None]/(4*np.pi*phase[:,None]))
                    for slot in range(2):
                        active = np.flatnonzero(np.abs(coefficients[:,slot]) > 1e-7*np.max(np.abs(coefficients[:,slot])))
                        if len(active) > 1 and injection and (injection=='audit' or modal[active[0]][1] not in audit_degrees):
                            source = active[0]
                            if injection=='mirror':
                                if len({modal[i][0] for i in active})<2:
                                    continue
                                target=next(i for i in range(q) if ms[i]%6==ms[source]%6 and parity[i]!=parity[source])
                            else:
                                target = next(i for i in range(q) if ms[i]%6 != ms[source]%6)
                            outgoing_coefficients[target,slot] += .1*coefficients[source,slot]/scales[source]
                    transverse = outgoing@outgoing_coefficients
                elif operation[0] == 2:
                    plane = np.empty(9); receive(connection,plane)
                    incident = transverse_harmonics(plane[6:].reshape(1,3),order)
                    transverse = outgoing@kernel@incident.conj().T
                else:
                    raise ValueError('unknown test operation')
                transverse = transverse.reshape(len(directions),2,2)
                far = (et[:,:,None]*transverse[:,0,None,:]+ep[:,:,None]*transverse[:,1,None,:]).transpose(2,0,1)
                message({'strict_residual_x':1e-12,'strict_residual_y':1e-12,'iterations':1,
                         'received_operation':int(operation[0]),
                         'solve_s':0.,'total_worker_s':0.,'farfield_s':0.})
                connection.sendall(np.ascontiguousarray(far).tobytes())


if __name__ == '__main__':
    main()
