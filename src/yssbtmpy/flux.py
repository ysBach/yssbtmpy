"""Flux calculation utilities (reflected light)."""

from __future__ import annotations

import numpy as np
from astropy import units as u

from .constants import AU, FLAMU
from .scat.phase import iau_hg_model
from .scat.solar import SOLAR_SPEC
from .util import F_OR_ARR, to_val

__all__ = ["calc_flux_refl"]


def calc_flux_refl(
    phase_ang__deg, diam_eff__km, p_vis, solar_spec=SOLAR_SPEC,
    slope_par=0.15, r_hel__au=1, r_obs__au=1,
    wlen_min=0, wlen_max=1000, refl: F_OR_ARR = None, phase_factor=None
):
    """Calculate reflected flux.

    Parameters
    ----------
    phase_ang__deg : float, Quantity
        The phase angle in degrees.

    diam_eff__km : float, Quantity
        The effective diameter in km.

    p_vis : float, Quantity
        The geometric albedo in the V-band.

    solar_spec : ndarray, optional.
        The solar spectrum. Default is `tm.SOLAR_SPEC`. Otherwise, it must be
        a 2D array with the first column as the wavelength in microns and the
        second column as the flux in W/m^2/um (FLAM unit) measured at the
        distance of 1 au.

    slope_par : float, optional.
        The slope parameter (G) in the IAU H-G magnitude system.

    r_hel__au, r_obs__au : float, optional.
        The heliocentric and observer distance in au.

    wlen_min, wlen_max : float, Quantity, optional.
        The wavelength in microns (if `float`) to be used. The calculation
        will be done for wavelengths of ``wlen_min < tm.SOLAR_SPEC[:, 0] <
        wlen_max``. Default is 0 and 1000.

    refl : float, ndarray, or callable, optional.
        The reflectance, normalized to 1 at V-band, in linear scale. If not
        given, it is set to 1 (flat spectrum). If given as a callable, it
        should accept wavelength (in microns).

    phase_factor : float or ndarray, optional.
        The phase function value. If not given, computed from the IAU H-G
        model using `slope_par`.

    Returns
    -------
    wlen_refl : Quantity
        The wavelengths in microns.
    flux_refl : Quantity or ndarray
        The reflected flux in W/m^2/um.

    Notes
    -----
    At the moment, this functionality is very limited.

    >>> _r = YOUR_REFLECTANCE
    >>> _w = YOUR_WAVELENGTH  # in microns, same size as _r
    >>> refl = UnivariateSpline(_w, _l, k=3, s=0, ext="const")(tm.SOLAR_SPEC[:, 0])
    """
    wlen_min = to_val(wlen_min, u.um)
    wlen_max = to_val(wlen_max, u.um)
    wlen__um = solar_spec[:, 0]
    if wlen_min is not None and wlen_max is not None:
        wlen_mask = (wlen__um >= wlen_min) & (wlen__um <= wlen_max)
        wlen__um = wlen__um[wlen_mask]
        flux__flam = solar_spec[wlen_mask, 1]
    else:
        flux__flam = solar_spec[:, 1]

    wlen_refl = wlen__um*u.um

    if refl is None:
        refl = 1
    elif isinstance(refl, (int, float, np.ndarray)):
        refl = np.atleast_1d(refl)
    else:  # assume functional
        refl = refl(wlen__um)

    if phase_factor is None:
        phase_factor = iau_hg_model(
            phase_ang__deg=np.atleast_1d(phase_ang__deg), gpar=slope_par
        )
    flux_refl = (flux__flam/(r_hel__au)**2
                 * refl*p_vis
                 * (diam_eff__km*500)**2  # *500 is for *1000/2
                 / (r_obs__au*AU)**2
                 * phase_factor
                 ) * FLAMU
    # solar_spec already is for r_hel = 1au, so for r_hel, it should use u.au.
    # See, e.g., Eq. 19 of Myhrvold 2018 Icarus, 303, 91.
    return wlen_refl, flux_refl
