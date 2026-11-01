"""Explicit amplitude conversion for native coherent MBS and BEM currents.

MBS exports the physical object-centered outgoing-wave amplitude in um.
The BEM repository uses the opposite global radiation sign; see the
-i*k/(4*pi) prefactor in src/muller_dense.cpp:muller_nodal_farfield and the
documented convention in scripts/tmatrix_reference.py. This sign cancels
in Mueller matrices but must be included before reconstructing currents.
Neither a fitted complex phase nor an intensity-derived normalization is
allowed here. Callers must also match the incident/outgoing vector axes.
"""
import numpy as np


def native_jones_to_bem(jones, ka, wavelength):
    value=np.asarray(jones,dtype=complex)
    if value.ndim<2 or value.shape[-2:]!=(2,2) or not np.isfinite(value).all():
        raise ValueError('finite complex Jones matrices required')
    if not np.isfinite(ka) or not np.isfinite(wavelength) or ka<=0 or wavelength<=0:
        raise ValueError('positive finite ka and wavelength required')
    radius=ka*wavelength/(2*np.pi)
    return -value/radius
