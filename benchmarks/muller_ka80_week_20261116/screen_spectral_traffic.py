"""Compute optimal SVD-tail schedules from the saved balanced ka80 factors."""
from pathlib import Path
import json
import numpy as np

base = Path(__file__).resolve().parent
tolerances = [1e-5, 1e-4, 5e-4, 1e-3, 1e-2, 3e-2]
results = {}
for index in (1, 2, 4, 5):
    folder = base/'fft_ka80'/f'cross_{index}_recompressed'
    left, right = [np.load(folder/f'{side}.npy', mmap_mode='r') for side in ('left','right')]
    # Both factors contain sqrt(sigma) in orthogonal columns/rows.
    use_left = left.shape[1] <= right.shape[2]
    data = left if use_left else right
    spectra = []
    for panel in data:
        v = panel if use_left else panel.T
        energy = np.einsum('ij,ij->j', v.real, v.real, dtype='f8') + np.einsum('ij,ij->j', v.imag, v.imag, dtype='f8')
        spectra.append(energy)
    sigma = np.array(spectra)
    tails = np.concatenate((np.cumsum(sigma[:, ::-1]**2,axis=1)[:,::-1],np.zeros((len(sigma),1))),axis=1)
    rows = []
    for tol in tolerances:
        # The same absolute operator bound for every FFT frequency.
        rank = (sigma > tol*sigma.max()).sum(axis=1)
        # Contiguous batches of 8 with a common rank rounded upward.
        padded = np.concatenate([np.full(len(chunk),min(sigma.shape[1],8*((int(chunk.max())+7)//8))) for chunk in np.array_split(rank,np.arange(8,len(rank),8))])
        discard = tails[np.arange(len(sigma)),padded]
        rows.append(dict(tol=tol,traffic_fraction=float(padded.mean()/sigma.shape[1]),relative_frobenius=float(np.sqrt(discard.sum()/tails[:,0].sum())),batch_ranks=padded[::8].tolist()))
    results[index]=dict(rows=rows,bytes=2*(left.size+right.size)*2)
    print(index,[(r['tol'],round(r['traffic_fraction'],3)) for r in rows],flush=True)
(base/'spectral_traffic_screen.json').write_text(json.dumps(results,indent=2))
