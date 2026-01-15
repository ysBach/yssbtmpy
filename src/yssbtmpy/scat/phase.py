import numpy as np
from astropy import units as u
import numba as nb
from scipy.interpolate import CloughTocher2DInterpolator

from ..constants import D2R
from ..util import to_val

__all__ = ["hapke_k_theta_phase", "iau_hg_model", "iau_hg_mag"]

_HAPKE_K_VALS = np.array([
    [1.00, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
    [1.00, 0.997, 0.991, 0.984, 0.974, 0.961, 0.943],
    [1.00, 0.994, 0.981, 0.965, 0.944, 0.918, 0.881],
    [1.00, 0.991, 0.970, 0.943, 0.909, 0.866, 0.809],
    [1.00, 0.988, 0.957, 0.914, 0.861, 0.797, 0.715],
    [1.00, 0.986, 0.947, 0.892, 0.825, 0.744, 0.644],
    [1.00, 0.984, 0.938, 0.871, 0.789, 0.692, 0.577],
    [1.00, 0.982, 0.926, 0.846, 0.748, 0.635, 0.509],
    [1.00, 0.979, 0.911, 0.814, 0.698, 0.570, 0.438],
    [1.00, 0.974, 0.891, 0.772, 0.637, 0.499, 0.366],
    [1.00, 0.968, 0.864, 0.719, 0.566, 0.423, 0.296],
    [1.00, 0.959, 0.827, 0.654, 0.487, 0.346, 0.231],
    [1.00, 0.946, 0.777, 0.575, 0.403, 0.273, 0.175],
    [1.00, 0.926, 0.708, 0.484, 0.320, 0.208, 0.130],
    [1.00, 0.894, 0.617, 0.386, 0.243, 0.153, 0.094],
    [1.00, 0.840, 0.503, 0.290, 0.175, 0.107, 0.064],
    [1.00, 0.747, 0.374, 0.201, 0.117, 0.070, 0.041],
    [1.00, 0.590, 0.244, 0.123, 0.069, 0.040, 0.023],
    [1.00, 0.366, 0.127, 0.060, 0.032, 0.018, 0.010],
    [1.00, 0.128, 0.037, 0.016, 0.0085, 0.0047, 0.0026],
    [1.00, 0, 0, 0, 0, 0, 0]
])
_HAPKE_K_ALPHA = np.repeat(np.array([
    0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
    100, 110, 120, 130, 140, 150, 160, 170, 180
]), 7)
_HAPKE_K_THBAR = np.tile(np.array([0, 10, 20, 30, 40, 50, 60]), 21)

hapke_k_theta_phase = CloughTocher2DInterpolator(
    list(zip(_HAPKE_K_THBAR, _HAPKE_K_ALPHA)),
    _HAPKE_K_VALS.ravel(),
)


_D2R = np.pi / 180.0



# numba makes it ~3x faster than the pure numpy version.
@nb.njit(fastmath=True, cache=False)
def iau_hg_model(phase_ang__deg, gpar=0.15):
    """The IAU HG phase function model in intensity (1 at alpha=0).

    Parameters
    ----------
    phase_ang__deg : array_like
        The phase angle (Sun-Target-Observer angle) in degrees.

    gpar : float, optional
        The slope parameter ($G$) in the IAU H, G modeling. See Notes.
        By default ``0.15``.

    Notes
    -----
    Semi-empirical model of the phase function of the Moon, asteroids, and
    other (airless) solar system objects. The phase function defined at
    $(0^\circ \le \alpha \le 120^\circ)$ for the phase angle $\alpha$. It
    is given by the following equation:

    .. math::
        \Phi_\mathrm{HG}(\alpha, G) = G \Phi_{HG1}(\alpha) + (1-G) \Phi_{HG2}(\alpha)

    where

    .. math::
        \Phi_{HG i}(\alpha) = W \left ( 1-\frac{C_i \sin \alpha}{0.119+1.341 \sin \alpha-0.754 \sin ^2 \alpha} \right )
        + (1 - W) \times \exp \left \{ -A_i \left [ \tan \frac{\alpha}{2} \right ]^{B_i} \right \}

    and

    .. math::
        W(\alpha) = \exp \left \{ -90.56 \tan^2 \frac{\alpha}{2} \right \}

    The parameters $A_i$, $B_i$, and $C_i$ are given by:

    .. math::
        A_1, A_2 &= 3.332, 1.862 \sep
        B_1, B_2 = 0.631, 1.218 \sep
        C_1, C_2 = 0.986, 0.238

    Reference: Bowell et al. 1989
    https://ui.adsabs.harvard.edu/abs/1989aste.conf..524B/abstract
    """
    n = phase_ang__deg.shape[0]
    intensity = np.empty(n, dtype=np.float64)
    # onemgpar = 1.0 - gpar
    phi1 = np.empty(n, dtype=np.float64)
    phi2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        # convert degrees to radians
        ar = np.abs(phase_ang__deg[i]) * _D2R

        # intermediate trig and weighting terms
        sa = np.sin(ar)
        fa = sa / (0.119 + 1.341 * sa - 0.754 * sa * sa)
        tah = np.tan(ar * 0.5)
        w = np.exp(-90.56 * tah * tah)

        # smooth (s) and linear (l) components
        phi1_s = 1.0 - 0.986 * fa
        phi2_s = 1.0 - 0.238 * fa
        phi1_l = np.exp(-3.332 * np.power(tah, 0.631))
        phi2_l = np.exp(-1.862 * np.power(tah, 1.218))

        # mix them
        # intensity[i] = gpar[i] * (w * phi1_s + (1.0 - w) * phi1_l) + onemgpar[i] * (
        #     w * phi2_s + (1.0 - w) * phi2_l
        # )
        phi1[i] = w * phi1_s + (1.0 - w) * phi1_l
        phi2[i] = w * phi2_s + (1.0 - w) * phi2_l

    intensity = gpar * phi1 + (1.0 - gpar) * phi2
    return intensity


@nb.njit(fastmath=True, cache=False)
def iau_hg_mag(hmag, phase_ang__deg, gpar=0.15, robs=1, rhel=1):
    """The IAU HG phase function model in magnitudes scale.

    Parameters
    ----------
    hmag : float
        The absolute magnitude of the object.

    phase_ang__deg : array_like
        The phase angle (Sun-Target-Observer angle) in degrees.

    gpar : float, optional
        The slope parameter ($G$) in the IAU H, G modeling. See Notes.
        By default ``0.15``.

    robs, rhel : float, optional
        The observer and heliocentric distance in au. By default 1 au.

    Returns
    -------
    mag : ndarray
        The apparent magnitude of the object at the given phase angle.
    """
    return (
        hmag
        + 5 * np.log10(robs * rhel)
        - 2.5 * np.log10(iau_hg_model(phase_ang__deg, gpar))
    )
