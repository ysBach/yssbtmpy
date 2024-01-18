import numpy as np
from astropy import units as u
from .constants import CC, HH, KB, SIGMA_SB
from .util import to_val


__all__ = ["B_lambda", "b_lambda", "flam2jy", "jy2flam", "flam2ab"]


def B_lambda(wavelen, temperature):
    """ Calculates black body radiance [energy/time/area/length/sr] = [W/m2/m/sr]
    Parameters
    ----------
    wavelen : float or `~Quantity`, `~numpy.ndarray` of such.
        The wavelengths. In meter unit if not Quantity.

    temperature : float or `~Quantity`
        The temperature. In Kelvin unit if not Quantity. For specific purpose,
        you can give it in an ndarray format, but not recommended.

    Return
    ------
    radiance : ndarray
        The black body radiance [energy/time/area/length/sr] = [W/m2/m/sr].
    """
    wl = to_val(wavelen, u.m)
    temp = to_val(temperature, u.K)
    coeff = 2*HH*CC**2 / wl**5
    denom = np.exp(HH*CC/(wl*KB*temp)) - 1
    radiance = coeff / denom
    return radiance


def b_lambda(wavelen, temperature):
    """ Calcualtes the small b function [1/wavelen].

    Parameters
    ----------
    wavelen : float or `~Quantity`, `~numpy.ndarray` of such.
        The wavelengths. In meter unit if not Quantity.

    temperature : float or `~Quantity`
        The temperature. In Kelvin unit if not Quantity. For specific purpose,
        you can give it in an ndarray format, but not recommended.

    Return
    ------
    radiance : ndarray
        The small b function [1/wavelen].
    """
    wl = to_val(wavelen, u.m)
    temp = to_val(temperature, u.K)
    norm = SIGMA_SB * temp**4
    norm_radiance = np.pi * B_lambda(wavelen=wl, temperature=temp) / norm
    return norm_radiance


def flam2jy(flam, wlen):
    """ Convert flux density from [W/m2/um] to [Jy].
    Parameters
    ----------
    flam : array-like
        The flux density in [W/m2/um].

    wlen : array-like
        The wavelength in um.

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
        The wavelength in um.

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
        The wavelength in um.
    """
    return -2.5*np.log10(flam2jy(flam, wlen)/reference_jy)


def planck_avg(wlen, val, temp):
    """Average by weighting of Planck function(B_lambda)

    Parameters
    ----------
    wlen, val : array-like
        The wavelength and value of a thing (frequently used example:
        emissivity or reflectance over wavelength) to be averaged.

    temp : float or `~Quantity`
        The temperature of the Planck function. In Kelvin unit if not Quantity. For specific purpose,
        you can give it in an ndarray format, but not recommended.

    Notes
    -----
    Average is

    ..math::
        \frac{\int B_\lambda(T) val(\lambda) d\lambda}{\int B_\lambda(T) d\lambda}
    """
    temp = to_val(temp, u.K)
    norm = np.pi/SIGMA_SB/temp**4