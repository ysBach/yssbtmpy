import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid

from .constants import CC, HH, KB, PI, SIGMA_SB, FLAMU
from .util import to_val

__all__ = [
    "B_lambda", "B_lambda_wien", "B_lambda_rj",
    "flam2jy", "jy2flam", "flam2ab",
    "jy2ab", "ab2jy",
    "jy2phot", "planck_avg"
]


def B_lambda(wlen, temperature, normalized=False):
    """ Calculates black body radiance [W/m2/um/sr]

    Parameters
    ----------
    wlen : float or `~Quantity`, `~numpy.ndarray` of such.
        The wavelengths. In um unit if not Quantity.

    temperature : float or `~Quantity`
        The temperature. In Kelvin unit if not Quantity. For specific purpose,
        you can give it in an ndarray format, but not recommended.

    normalized : bool
        If `True`, the radiance is normalized by the Stefan-Boltzmann relation
        (``int(B_lambda dlambda) = SIGMA_SB * T**4 / PI``), so the returned
        object has the unit of [1/um]. Otherwise, the radiance is not
        normalized.
        Default is `False`.

    Returns
    -------
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


def B_lambda_wien(wlen, temperature, normalized=False):
    """ Wien's approximation to the Planck function [W/m2/um/sr].

    .. math::

        B_\\lambda^W = \\frac{2hc^2}{\\lambda^5} e^{-hc/(\\lambda k T)}

    This is the high-frequency (short-wavelength) limit of the full Planck
    function.  The fractional error relative to the full Planck function is

    .. math::

        \\frac{B_\\lambda^W}{B_\\lambda} = 1 - e^{-x},
        \\quad x \\equiv \\frac{hc}{\\lambda k T}

    so the approximation is accurate to < 1 % when x > 4.6
    (i.e. lambda*T < 3140 um*K, e.g. lambda < 15 um at 200 K).

    Parameters
    ----------
    wlen : float or `~Quantity`, `~numpy.ndarray` of such.
        Wavelength(s) in um (if not Quantity).

    temperature : float or `~Quantity`
        Temperature in Kelvin (if not Quantity). May be an ndarray.

    normalized : bool
        Same semantics as `B_lambda`.  The denominator used for normalization
        is ``SIGMA_SB * T**4 / PI`` (full-Planck total), so the normalized
        Wien curve does **not** integrate exactly to 1.0.
        Default is `False`.

    Returns
    -------
    radiance : ndarray
        [W/m2/um/sr] or [1/um] if normalized.
    """
    wl = to_val(wlen, u.um) * 1.e-6
    temp = to_val(temperature, u.K)
    coeff = 2 * HH * CC**2 / wl**5
    radiance = coeff * np.exp(-HH * CC / (wl * KB * temp))
    if normalized:
        return radiance * PI / (SIGMA_SB * temp**4) * 1.e+6  # [1/um]
    return radiance * 1.e-6  # [W/m2/um/sr]


def B_lambda_rj(wlen, temperature):
    """ Rayleigh-Jeans approximation to the Planck function [W/m2/um/sr].

    .. math::

        B_\\lambda^{RJ} = \\frac{2 c k T}{\\lambda^4}

    This is the low-frequency (long-wavelength) limit of the full Planck
    function.  The fractional ratio relative to the full Planck function is

    .. math::

        \\frac{B_\\lambda^{RJ}}{B_\\lambda} = \\frac{x}{e^x - 1},
        \\quad x \\equiv \\frac{hc}{\\lambda k T}

    so the approximation is accurate to < 1 % when x < 0.14
    (i.e. lambda*T > 10^5 um*K, e.g. lambda > 500 um at 200 K).

    .. note::
        The integral of ``B_lambda_rj`` over all wavelengths diverges
        (ultraviolet catastrophe), so ``normalized`` is not supported.

    Parameters
    ----------
    wlen : float or `~Quantity`, `~numpy.ndarray` of such.
        Wavelength(s) in um (if not Quantity).

    temperature : float or `~Quantity`
        Temperature in Kelvin (if not Quantity). May be an ndarray.

    Returns
    -------
    radiance : ndarray
        [W/m2/um/sr]
    """
    wl = to_val(wlen, u.um) * 1.e-6
    temp = to_val(temperature, u.K)
    return (2 * CC * KB * temp / wl**4) * 1.e-6  # [W/m2/um/sr]


def flam2jy(flam_si, wlen):
    """ Convert flux density from [FLAM_SI = W/m2/um = 10 erg/s/cm2/AA] to [Jy].

    Parameters
    ----------
    flam_si : array-like
        The flux density in [W/m2/um].

    wlen : array-like
        The wavelength in um.

    Notes
    -----
    1Jy = 10-26 W⋅m-2⋅Hz-1
    """
    return 3.33564095e+11 * to_val(flam_si, FLAMU) * to_val(wlen, u.um)**2


def jy2flam(jy, wlen):
    """ Convert flux density from [Jy] to [FLAM_SI = W/m2/um = 10 erg/s/cm2/AA].

    Parameters
    ----------
    jy : array-like
        The flux density in [Jy].

    wlen : array-like
        The wavelength in um.

    Notes
    -----
    1Jy = 10-26 W⋅m-2⋅Hz-1
    """
    return 299792458.0/((to_val(wlen, u.um)*1.e-6)**2) * to_val(jy, u.Jy)*1.e-32


def flam2ab(flam, wlen):
    """ Convert flux density from [W/m2/um] to [AB mag].

    Parameters
    ----------
    flam : array-like
        The flux density in [W/m2/um].

    wlen : array-like
        The wavelength in um.
    """
    return jy2ab(flam2jy(to_val(flam, FLAMU), to_val(wlen, u.um)))


def jy2ab(jy):
    """ Convert flux density from [Jy] to [AB mag].

    Parameters
    ----------
    jy : array-like
        The flux density in [Jy].
    """
    return -2.5*np.log10(to_val(jy, u.Jy)/3631)


def ab2jy(ab):
    """ Convert flux density from [AB mag] to [Jy].

    Parameters
    ----------
    ab : array-like
        The flux density in [AB mag].
    """
    return 3631*10**(-0.4*to_val(ab, u.mag))


def jy2phot(jy):
    """ Convert flux density from [Jy] to [photon/s/m2].

    Parameters
    ----------
    jy : array-like
        The flux density in [Jy].
    """
    # jy/(hc/wlen) * c/wlen = jy/h
    return to_val(jy, u.Jy)/HH*1.e-26


def ab2phot(ab):
    """ Convert flux density from [AB mag] to [photon/s/m2/um].

    Parameters
    ----------
    ab : array-like
        The flux density in [AB mag].
    """
    return jy2phot(ab2jy(to_val(ab, u.mag)))


def planck_avg(wlen, val, temp, use_sb=False):
    """Average by weighting of Planck function (int(B_lambda*val)/int(B_lambda))

    Parameters
    ----------
    wlen, val : array-like
        The wavelength (um if float) and value of a thing (frequently used
        example: emissivity or reflectance over wavelength) to be averaged.

    temp : float or `~Quantity`
        The temperature of the Planck function. In Kelvin unit if not Quantity.

    use_sb : bool
        If `True`, the Stefan-Boltzmann relation is used for the denominator
        (``integral(B_lambda) = SIGMA_SB * T**4 / PI``). Otherwise, numerical
        integration is used (``integral(B_lambda) = trapezoid(x=wlen,
        y=B_lambda(wlen, temp))``).

    Notes
    -----
    Average is

    .. math::
        \\frac{\\int B_\\lambda(T) val(\\lambda) d\\lambda}{\\int B_\\lambda(T) d\\lambda}
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
