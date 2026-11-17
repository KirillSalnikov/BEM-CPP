"""Experimental reusable planar-face radiation transform; FINUFFT >= 2.4.

Preserves the supplied quadrature strengths, including signed weights.
This evaluates the existing radiation sum; it does not approximate currents,
solve BEM, or average incident orientations. External wavenumber must be real.
"""
import numpy as np
import finufft


class FaceRadiationPlan:
    def __init__(self, points, normals, directions, k, channels, eps=1e-10,
                 threads=4, max_faces=128):
        points = np.asarray(points, dtype=np.float64)
        normals = np.asarray(normals, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape
                or directions.ndim != 2 or directions.shape[1] != 3
                or not len(points) or not len(directions)
                or not np.isfinite(points).all() or not np.isfinite(normals).all()
                or not np.isfinite(directions).all()
                or not np.isfinite(k) or np.imag(k) != 0
                or not isinstance(channels, int) or channels < 1
                or not (0 < eps < 1) or threads < 1):
            raise ValueError("finite 3D points/normals/directions, real k, positive channels/threads required")
        if not np.allclose(np.linalg.norm(normals, axis=1), 1, rtol=0, atol=1e-10):
            raise ValueError("unit surface normals required")
        if not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=0, atol=1e-10):
            raise ValueError("unit observation directions required")
        self.shape = (channels, len(points))
        self.output_shape = (channels, len(directions))
        self.faces = []
        # Equal normal plus plane distance, not normal alone: supports separate
        # parallel faces, rotations, translations, arbitrary planar meshes.
        scale = max(float(np.max(np.ptp(points, axis=0))), 1e-100)
        centered = points - points.mean(axis=0)
        keys = np.column_stack((normals, np.sum(centered*normals, axis=1)/scale))
        _, groups = np.unique(np.round(keys, 9), axis=0, return_inverse=True)
        if groups.max()+1 > max_faces:
            raise ValueError("too many planar patches; use direct radiation for curved surfaces")
        for group in range(groups.max()+1):
            ids = np.flatnonzero(groups == group)
            x = points[ids]
            origin = x.mean(axis=0)
            normal = normals[ids[0]]
            local = x-origin
            # Do not silently flatten curved panels or merge distinct planes.
            if np.max(np.abs(local@normal)) > 2e-11*scale:
                raise ValueError("nonplanar patch")
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            u = np.cross(normal, axis); u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            plan = finufft.Plan(3, 2, n_trans=channels, eps=eps, isign=-1,
                                dtype="complex128", nthreads=threads)
            plan.setpts(np.ascontiguousarray(local@u), np.ascontiguousarray(local@v),
                        s=np.ascontiguousarray(float(np.real(k))*(directions@u)),
                        t=np.ascontiguousarray(float(np.real(k))*(directions@v)))
            phase = np.exp(-1j*float(np.real(k))*(directions@origin))
            self.faces.append((ids, phase, plan))

    def execute(self, weighted_currents):
        strengths = np.asarray(weighted_currents, dtype=np.complex128)
        if strengths.shape != self.shape or not np.isfinite(strengths).all():
            raise ValueError("finite weighted currents with shape (channels, points) required")
        out = np.zeros(self.output_shape, dtype=np.complex128)
        for ids, phase, plan in self.faces:
            out += plan.execute(np.ascontiguousarray(strengths[:, ids]))*phase
        return out


def electric_farfield(transformed, directions, k):
    """(..., 6, angles) Jx,Jy,Jz,Mx,My,Mz -> (..., angles, 3)."""
    values = np.asarray(transformed)
    directions = np.asarray(directions)
    if values.shape[-2:] != (6, len(directions)):
        raise ValueError("six Cartesian current components required")
    j = np.moveaxis(values[..., :3, :], -2, -1)
    m = np.moveaxis(values[..., 3:, :], -2, -1)
    transverse = j - directions*np.sum(directions*j, axis=-1)[..., None]
    return (-1j*k/(4*np.pi))*(transverse - np.cross(directions, m))
