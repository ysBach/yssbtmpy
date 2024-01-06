import numpy as np
from numba import njit
from scipy.interpolate import CloughTocher2DInterpolator
from ..constants import D2R
from ..util import change_to_quantity


__all__ = ["hapke_k_theta_phase", "iau_hg_linear", "iau_hg"]

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


@njit
def _hgphi12_nb(alpha__deg):
    sin_a = np.sin(alpha__deg*D2R)
    f_a = sin_a/(0.119+1.341*sin_a-0.754*sin_a*sin_a)
    tan_a_half = np.tan(alpha__deg*D2R*0.5)
    w = np.exp(-90.56*tan_a_half*tan_a_half)
    phi1_s = 1 - 0.986*f_a
    phi2_s = 1 - 0.238*f_a
    phi1_l = np.exp(-3.332*tan_a_half**0.631)
    phi2_l = np.exp(-1.862*tan_a_half**1.218)
    return (w*phi1_s + (1-w)*phi1_l, w*phi2_s + (1-w)*phi2_l)


def iau_hg_linear(alphas, gpar=0.15):
    """The IAU HG phase function model in intensity (1 at alpha=0)
    """
    alphas = change_to_quantity(alphas, 'deg', to_value=True)
    hgphi1, hgphi2 = _hgphi12_nb(np.array(alphas))
    return (1-gpar)*hgphi1 + gpar*hgphi2


def iau_hg(alphas, hmag=10, gpar=0.15, r_hel=1., r_obs=1.):
    """The IAU HG phase function model in magnitude
    """
    inten = iau_hg_linear(alphas, gpar)
    return np.array(hmag - 2.5*np.log10(inten) + 5*np.log10(r_hel*r_obs))


# def hapke_rough_corr_factor(thetabar=0):
#     """
#     thetabar: degr
#     """
