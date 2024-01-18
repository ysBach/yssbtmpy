import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid

from .constants import CC, HH, KB, PI, SIGMA_SB
from .util import to_val

__all__ = ["B_lambda", "flam2jy", "jy2flam", "flam2ab", "planck_avg"]


def B_lambda(wlen, temperature, normalized=False):
    """ Calculates black body radiance [W/m2/um/sr]
    Parameters
    ----------
    wlen : float or `~Quantity`, `~numpy.ndarray` of such.
        The wlengths. In um unit if not Quantity.

    temperature : float or `~Quantity`
        The temperature. In Kelvin unit if not Quantity. For specific purpose,
        you can give it in an ndarray format, but not recommended.

    normalized : bool
        If `True`, the radiance is normalized by the Stefan-Boltzmann relation
        (``int(B_lambda dlambda) = SIGMA_SB * T**4 / PI``), so the returned
        object has the unit of [1/um]. Otherwise, the radiance is not
        normalized.
        Default is `False`.

    Return
    ------
    radiance : ndarray
        The black body radiance [energy/time/area/length/sr] = [W/m2/um/sr] or
        normalized version of it [1/um].
    """
    wl = to_val(wlen, u.um)*1.e-6
    temp = to_val(temperature, u.K)
    coeff = 2*HH*CC**2 / wl**5
    denom = np.exp(HH*CC/(wl*KB*temp)) - 1
    radiance = coeff / denom
    if normalized:
        return radiance * PI / (SIGMA_SB*temp**4) * 1.e+6  # [1/um]
    return radiance * 1.e-6  # [W/m2/**um**/sr]


def flam2jy(flam, wlen):
    """ Convert flux density from [W/m2/um] to [Jy].
    Parameters
    ----------
    flam : array-like
        The flux density in [W/m2/um].

    wlen : array-like
        The wlength in um.

    1Jy = 10-26 W⋅m-2⋅Hz-1
    """
    return 3.33564095e+11 * flam * wlen**2


def jy2flam(jy, wlen):
    """ Convert flux density from [Jy] to [W/m2/um].
    Parameters
    ----------
    jy : array-like
        The flux density in [Jy].

    wlen : array-like
        The wlength in um.

    1Jy = 10-26 W⋅m-2⋅Hz-1
    """
    return 299792458.0/wlen**2 * jy


def flam2ab(flam, wlen, reference_jy=3631):
    """ Convert flux density from [W/m2/um] to [AB mag].
    Parameters
    ----------
    flam : array-like
        The flux density in [W/m2/um].

    wlen : array-like
        The wlength in um.
    """
    return -2.5*np.log10(flam2jy(flam, wlen)/reference_jy)


def planck_avg(wlen, val, temp, use_sb=True):
    """Average by weighting of Planck function (int(B_lambda*val)/int(B_lambda))

    Parameters
    ----------
    wlen, val : array-like
        The wlength (um if float) and value of a thing (frequently used
        example: emissivity or reflectance over wlength) to be averaged.

    temp : float or `~Quantity`
        The temperature of the Planck function. In Kelvin unit if not Quantity. For specific purpose,
        you can give it in an ndarray format, but not recommended.

    use_sb : bool
        If `True`, the Stefan-Boltzmann relation is used for the denominator
        (``integral(B_lambda) = SIGMA_SB * T**4 / PI``). Otherwise, numerical
        integration is used (``integral(B_lambda) = trapezoid(x=wlen,
        y=B_lambda(wlen, temp))``).

    Notes
    -----
    Average is

    ..math::
        \frac{\int B_\lambda(T) val(\lambda) d\lambda}{\int B_\lambda(T) d\lambda}
    """
    temp = to_val(temp, u.K)
    wlen = to_val(wlen, u.um)
    if isinstance(val, u.Quantity):
        val = val.value
    numer = trapezoid(B_lambda(wlen, temp)*val, wlen)
    if use_sb:
        denom = SIGMA_SB*temp**4/PI
    else:
        denom = trapezoid(B_lambda(wlen, temp), wlen)
    return numer/denom
