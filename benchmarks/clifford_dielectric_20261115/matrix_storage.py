"""Disk-backed T assembly with bounded conversion buffers, no full-size copies."""
import mmap
import numpy as np


def release_pages(array):
    if isinstance(array, np.memmap):
        array.flush()
        if hasattr(array._mmap, 'madvise'):
            array._mmap.madvise(mmap.MADV_DONTNEED)


def convert_projected(projected, phase, ka, destination, columns=32):
    """Keep both historical conventions, converting only a bounded column tile."""
    if columns < 1 or projected.shape != (len(phase), len(phase)):
        raise ValueError('matrix dimensions or conversion tile')
    t = np.lib.format.open_memmap(destination, mode='w+', dtype=np.complex128,
                                  shape=projected.shape, fortran_order=True)
    for begin in range(0, len(phase), columns):
        stop = min(len(phase), begin+columns)
        np.multiply(projected[:, begin:stop], (1j*ka*phase)[:, None], out=t[:, begin:stop])
        projected[:, begin:stop] *= (4*np.pi*phase[begin:stop])[None, :]
        release_pages(t); release_pages(projected)
    return t


def norm_squared(array, columns=32):
    result = 0.
    for begin in range(0, array.shape[1], columns):
        tile = np.asarray(array[:, begin:begin+columns])
        result += float(np.einsum('ij,ij->', tile.conj(), tile).real)
    release_pages(array)
    return result
