"""One lab/body convention for fixed-incidence scattering (angles in degrees)."""
import numpy as np


def rotation(euler):
    """Active body-to-lab rotation Rz(alpha) Ry(beta) Rz(gamma)."""
    a, b, g = np.radians(euler)
    def rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
    c, s = np.cos(b), np.sin(b)
    return rz(a) @ np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]]) @ rz(g)


def frame(euler, theta_degrees):
    """Rows: lab basis in body coordinates; incidence +z, observation xz."""
    r = rotation(euler)
    theta = np.radians(theta_degrees)
    directions = np.column_stack((np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
    et = np.column_stack((np.cos(theta), np.zeros(len(theta)), -np.sin(theta)))
    ep = np.tile([0., 1., 0.], (len(theta), 1))
    return r, directions @ r, et @ r, ep @ r


def mueller(fields, et, ep, ka):
    """Electric far fields [polarization, angle, xyz] -> dimensionless Mueller."""
    j = np.stack((np.einsum('pnc,nc->np', fields, et),
                  np.einsum('pnc,nc->np', fields, ep)), axis=1) * ka
    sigma = np.array([np.eye(2), [[1, 0], [0, -1]], [[0, 1], [1, 0]],
                      [[0, -1j], [1j, 0]]], complex)
    return .5 * np.einsum('iab,nbc,jcd,nad->nij', sigma, j, sigma, j.conj()).real


def save_mueller(directory, theta, values):
    from pathlib import Path
    directory = Path(directory)
    if values.shape != (len(theta), 4, 4) or not np.isfinite(values).all():
        raise ValueError('Invalid Mueller output')
    np.savez(directory / 'mueller.npz', theta_degrees=theta, mueller=values)
    np.savetxt(directory / 'mueller.csv', np.column_stack((theta, values.reshape(-1, 16))),
               delimiter=',', header='theta_deg,' + ','.join(
                   f'M{i}{j}' for i in range(1, 5) for j in range(1, 5)), comments='')
