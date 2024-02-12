from copy import deepcopy
from typing import Union, Any, List

import numba as nb
import numpy as np
from scipy.linalg import inv
from astropy import units as u
from astropy.units import Quantity
from numba import njit

from .constants import CC, D2R, HH, KB, PI, R2D

__all__ = [
    "F_OR_Q", "F_OR_ARR", "F_OR_Q_OR_ARR",
    # "convrt_quant",
    "to_quantity", "to_val", "add_hdr", "parse_obj",
    "lonlat2cart", "sph2cart", "cart2sph",
    "calc_aspect_ang",
    "mat_ec2fs", "mat_bf2ss",
    "calc_mu_vals",
    "newton_iter_tpm", "calc_uarr_tpm",
    "calc_varr_orbit"
]


F_OR_Q = Union[u.Quantity, float]
F_OR_ARR = Union[np.ndarray, float]
F_OR_Q_OR_ARR = Union[u.Quantity, float, np.ndarray]

# def convert_fluxlambda2jy(fluxlambda, wlen):
#     """ Converts W/m^2/m to Jy.
#     fluxlambda : 1-D array
#         The flux density (F_lambda) in W/m^2/m.
#     wlen : 1-D array
#         The wavelength in m.
#     """
#     freq = CC/wlen
#     fluxlambda*wlen


def to_quantity(
    x: F_OR_Q_OR_ARR,
    desired: str | u.Unit = u.dimensionless_unscaled,
    to_value: bool = False
) -> F_OR_Q_OR_ARR:
    """ Treat quantity-like objects

    Parameters
    ----------
    x : float, quantity, array
        The input to be converted or changed to a Quantity, etc. If a Quantity
        is given, `x` is changed to the `desired`, i.e., ``x.to(desired)``.

    desired : str or astropy Unit
        The desired unit for `x`. `None` or ``''`` are identical to give
        ``u.dimensionless_unscaled``.
        Default is ``u.dimensionless_unscaled``.

    to_value : bool, optional.
        Whether to return as scalar value. If `True`, just the value(s) of the
        ``desired`` unit will be returned after conversion.
        Default is `False`.

    Return
    ------
    ux: Quantity

    Notes
    -----
    If Quantity, transform to ``desired``. If ``desired = None``, return it as
    is. If not Quantity, multiply the ``desired``. ``desired = None``, return
    ``x`` with dimensionless unscaled unit.
    """
    if isinstance(x, (float, int)):
        return x*1 if to_value else x*u.Unit(desired)

    try:
        return Quantity(x, desired).value if to_value else Quantity(x, desired)
    except AttributeError:
        if not to_value:
            if isinstance(desired, str):
                desired = u.Unit(desired)
            try:
                ux = x*desired
            except TypeError:
                ux = deepcopy(x)
        else:
            ux = deepcopy(x)
    except TypeError:
        ux = deepcopy(x)
    except u.UnitConversionError:
        raise ValueError(
            "If you use astropy.Quantity, you should use unit convertible to `desired`. \n"
            + f'Now it is in "{x.unit}", unconvertible with "{desired}": {x}.'
        )

    return ux


def to_val(x: F_OR_Q_OR_ARR,
           desired: str | u.Unit = u.dimensionless_unscaled) -> F_OR_ARR:
    """Change input into the desired unit.
    """
    return to_quantity(x, desired=desired, to_value=True)

# def to_quantity(
#     x: F_OR_Q_OR_ARR,
#     desired: str | u.Unit = '',
#     to_value: bool = False
# ) -> F_OR_Q_OR_ARR:
#     """ Change the non-Quantity object to astropy Quantity.

#     Parameters
#     ----------
#     x : object changable to astropy Quantity
#         The input to be changed to a Quantity. If a Quantity is given, ``x`` is
#         changed to the ``desired``, i.e., ``x.to(desired)``.

#     desired : str or astropy Unit
#         The desired unit for ``x``.

#     to_value : bool, optional.
#         Whether to return as scalar value. If `True`, just the value(s) of the
#         ``desired`` unit will be returned after conversion.

#     Return
#     ------
#     ux: Quantity

#     Notes
#     -----
#     If Quantity, transform to ``desired``. If ``desired = None``, return it as
#     is. If not Quantity, multiply the ``desired``. ``desired = None``, return
#     ``x`` with dimensionless unscaled unit.
#     """
#     def _copy(xx):
#         try:
#             xcopy = xx.copy()
#         except AttributeError:
#             import copy
#             xcopy = copy.deepcopy(xx)
#         return xcopy

#     try:
#         ux = x.to(desired)
#         if to_value:
#             ux = ux.value
#     except AttributeError:
#         if not to_value:
#             if isinstance(desired, str):
#                 desired = u.Unit(desired)
#             try:
#                 ux = x*desired
#             except TypeError:
#                 ux = _copy(x)
#         else:
#             ux = _copy(x)
#     except TypeError:
#         ux = _copy(x)
#     except u.UnitConversionError:
#         raise ValueError(
#             "If you use astropy.Quantity, you should use unit convertible to `desired`. \n"
#             + f'Now it is in "{x.unit}", unconvertible with "{desired}".'
#         )

#     return ux


def add_hdr(
        header,
        key: str,
        val: Any,
        desired_unit: str | u.Unit = '',
        comment: str = None
):
    _val = to_val(val, desired=desired_unit)
    header[key] = (_val, comment)
    return header


def parse_obj(objfile: str):
    """ Parses the .obj file.

    Parameters
    ----------
    objfile : path-like
        The path to the file.

    Return
    ------
    a dict object containing the raw str, vertices, facets, normals, and areas.
    """
    objstr = np.loadtxt(objfile, dtype=bytes).astype(str)
    vertices = objstr[objstr[:, 0] == 'v'][:, 1:].astype(float)
    facets = objstr[objstr[:, 0] == 'f'][:, 1:].astype(int)

    # Normals include direction + area information
    facet_normals_ast = []
    facet_areas = []

    # I don't think we need to speed up this for loop too much since it takes
    # only ~ 1 s even for 20000 facet case.
    for facet in facets:
        verts = vertices[facet - 1]  # Python is 0-indexing!!!
        vec10 = verts[1] - verts[0]
        vec20 = verts[2] - verts[0]

        area = np.linalg.norm(np.cross(vec10, vec20)) / 2  # Triangular
        facet_com_ast = np.sum(verts, axis=0) / 3

        facet_normals_ast.append(facet_com_ast)
        facet_areas.append(area)

    facet_normals_ast = np.array(facet_normals_ast)
    facet_areas = np.array(facet_areas)

    return dict(objstr=objstr, vertices=vertices, facets=facets,
                normals=facet_normals_ast, areas=facet_areas)


def lonlat2cart(lon: F_OR_Q, lat: F_OR_Q, degree: bool = True, r: F_OR_Q = 1) -> np.ndarray:
    """ Converts the lon/lat coordinate to Cartesian coordinate.

    Parameters
    ----------
    lon, lat : float or ~astropy.Quantity
        The longitude and latitude. If float, the unit is understood from
        ``degree``. Note that the latitude here is not the usual "theta"
        (``theta = 90 - lat``).

    degree : bool, optional
        Whether the input ``lon, lat`` are degrees (Default) or radian (if
        ``degree=False``).

    r : float, optional.
        The radial distance from the origin. Defaults to ``1``, i.e., the unit
        vector will be returned.

    Return
    ------
    a: 1-d array
        The calculated ``(x, y, z)`` array.
    """
    targ_unit = u.deg if degree else u.rad

    lon = to_quantity(lon, targ_unit)
    lat = to_quantity(lat, targ_unit)
    theta = 90*u.deg - lat
    return sph2cart(theta=theta, phi=lon, r=r)


def sph2cart(theta: F_OR_Q, phi: F_OR_Q, degree: bool = True, r: F_OR_Q = 1) -> np.ndarray:
    """ Converts the spherical coordinate to Cartesian coordinate.

    Parameters
    ----------
    theta, phi : float or ~astropy.Quantity
        The theta and phi of the ``(r, theta, phi)`` notation. If float, the
        unit is understood from ``degree``.

    degree : bool, optional
        Whether the input ``theta, phi`` are degrees (Default) or radian (if
        ``degree=False``).

    r : float, or `~astropy.Quantity`, optional.
        The radial distance from the origin. Defaults to ``1``, i.e., the unit
        vector will be returned.

    Return
    ------
    a: 1-d array
        The calculated ``(x, y, z)`` array.
    """
    targ_unit = u.deg if degree else u.rad

    th = to_quantity(theta, targ_unit)
    ph = to_quantity(phi, targ_unit)

    sin_th = (np.sin(th)).value
    cos_th = (np.cos(th)).value
    sin_ph = (np.sin(ph)).value
    cos_ph = (np.cos(ph)).value

    x = r * sin_th * cos_ph
    y = r * sin_th * sin_ph
    z = r * cos_th
    a = np.array([x, y, z])
    return a


def cart2sph(
        x: F_OR_Q, y: F_OR_Q, z: F_OR_Q,
        from_0: bool = True, degree: bool = True, to_lonlat: bool = False
) -> np.ndarray:
    """ Converts the Cartesian coordinate to lon/lat coordinate
    Parameters
    ----------
    x, y, z : float
        The Cartesian (x, y, z) coordinate.

    degree : bool, optional.
        If `False`, the returned theta and phi will be in radian. If
        `True`(default), those will be in degrees unit.

    from_0: bool, optional
        If `True` (Default), the ``phi`` (or ``lon``) will be in ``0`` to
        ``PI`` radian range. If `False`, i.e., if ``phi`` (or ``lon``) starts
        from ``-PI``, it will be in ``-PI`` to ``+PI`` range.

    Return
    ------
    a: 1-d array
        The ``(r, theta, phi)`` or ``(r, lon=phi, lat=90deg - theta)`` array.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    factor = R2D if degree else 1.

    theta = factor*np.arccos(z/r)
    phi = factor*np.arctan2(y, x)  # -180 to +180 deg
    if from_0:
        phi = phi % (factor*2*PI)

    if to_lonlat:
        lat = factor*PI/2 - theta
        a = np.array([r, phi, lat])
    else:
        a = np.array([r, theta, phi])

    return a


def calc_aspect_ang(
        spin_vec: np.ndarray,
        r_hel_vec: np.ndarray,
        r_obs_vec: np.ndarray = None,
        phase_ang: F_OR_Q = None,
) -> List[np.ndarray]:
    """Calculate the aspect angles (sun & observer) and delta-longitude.

    Parameters
    ----------
    spin_vec : 1-d array
        The spin vector in the ecliptic coordinate xyz (s.t. norm = 1.0).

    r_hel_vec : 1-d array
        The heliocentric vector to the body in the ecliptic coordinate.
        Actually, if one wants to quickly calculate the aspect angle for the
        observer, you can give an array of r_obs_vec here and leave others
        `None`.

    r_obs_vec : 1-d array, optional
        The observer-centric vector to the body in the ecliptic coordinate.

    phase_ang : float or ~astropy.Quantity, optional.
        The phase angle of the body. It does not necessarily have the sign (see
        Notes). Used for calculating delta-longitude of the sub-Solar and
        sub-observer's points, so it is used only if `r_obs_vec` is given.

    Return
    ------
    aspect_ang : float or ~astropy.Quantity
        The aspect angle between the spin vector and the heliocentric vector.

    aspect_ang_obs : float or ~astropy.Quantity
        The aspect angle between the spin vector and the observer-centric
        vector. Only returned if `r_obs_vec` is given.

    dlon_sol_obs : float or ~astropy.Quantity
        The delta-longitude between the sub-Solar and sub-observer's points.
        Only returned if `r_obs_vec` and `phase_ang` are given.


    Notes
    -----
    Using the sign(alpha) convention from DelboM 2004 PhDT p.144,

    .. math::
      sign([(-r_obs) x (-r_sun)] \cdot spin) = sign(alpha) = sign(dphi)

    where dphi is the longitude difference between sub-solar and sub-observer
    points. In words, alpha and dphi are positive if we are looking at the
    morning side. In the calculation, `phase_ang`(alpha) is used only for the
    calculation of dphi. The sign of alpha is determined within the code and
    used as `sign*np.abs(phase_ang)`.
    """
    # default aspect angle
    r_hel_hat = r_hel_vec/np.linalg.norm(r_hel_vec)
    aspect_ang = to_quantity(
        np.rad2deg(np.arccos(np.inner(-1*r_hel_hat, spin_vec))),
        u.deg
    )
    # aux_cos_sun = np.inner(r_hel_hat, self.spin_vec)
    # self.aspect_ang = (180-np.rad2deg(np.arccos(aux_cos_sun)))*u.deg

    if r_obs_vec is None or None in r_obs_vec:  # Return here
        return aspect_ang, None, None

    r_obs_hat = r_obs_vec/np.linalg.norm(r_obs_vec)
    # r_obs_hat = lonlat2cart(obs_ecl_helio.lon,
    #                         obs_ecl_helio.lat)
    aspect_ang_obs = to_quantity(
        np.rad2deg(np.arccos(np.inner(-1*r_obs_hat, spin_vec))),
        u.deg
    )
    # aux_cos_obs = np.inner(r_obs_hat, self.spin_vec)
    # aspect_ang_obs = (180-np.rad2deg(np.arccos(aux_cos_obs)))*u.deg

    if phase_ang is None:  # Return here
        return aspect_ang, aspect_ang_obs, None

    # == If phase_ang is given, proceed delta-longitude calculation:
    sign = np.sign(np.inner(np.cross(r_obs_hat, r_hel_hat), spin_vec))

    # cos(dphi) = [(cos(asp_sun)cos(asp_obs) - cos(alpha))
    # / sin(asp_sun)sin(asp_obs)]
    cc = np.cos(aspect_ang)*np.cos(aspect_ang_obs)
    ss = np.sin(aspect_ang)*np.sin(aspect_ang_obs)
    cos_dphi = ((cc - np.cos(sign*np.abs(phase_ang)))/ss).value
    cos_dphi = cos_dphi - np.sign(cos_dphi)*1.e-10
    dlon_sol_obs = sign*np.rad2deg(np.arccos(cos_dphi))*u.deg
    # self.pos_sub_sol = (self.aspect_ang, 180*u.deg)
    # self.pos_sub_obs = (self.aspect_ang_obs, phi_obs)

    # FIXME: I am a bit uncertain about the dlon calculation...
    # (2023-01-10 22:26:50 (KST: GMT+09:00) ysBach)
    return aspect_ang, aspect_ang_obs, dlon_sol_obs


def mat_ec2fs(r_vec: np.ndarray, spin_vec: np.ndarray) -> np.ndarray:
    """ The conversion matrix to convert ecliptic to frame system.

    Parameters
    ----------
    r_vec : 1-d array
        The Cartesian coordinate of the asteroid in ecliptic coordinate. It can
        be heliocentric or observer-centric (geocentric, planetocentric, etc)

    spin_vec : 1-d array
        The Cartesian coordinate of the spin vector of the asteroid in ecliptic
        coordinate.

    Return
    ------
    m : 3-by-3 matrix
        The matrix that converts ecliptic coordinate to frame system.

    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus, 243, 58.
    If ``a`` is a vector in ecliptic coordinate (in Cartesian (x, y, z)), ``m @
    a`` will give the components of vector ``a`` in frame system, where ``m``
    is the result of this function.

    I found scipy inversion is faster than numpy inversion:
    import numpy as np
    import scipy.linalg as sl
    %timeit np.linalg.inv(m1)
    %timeit sl.inv(m1)
    %timeit sl.inv(m1, overwrite_a=True, check_finite=False)

    # 19 µs +- 34.9 ns per loop (mean +- std. dev. of 7 runs, 100,000 loops each)
    # 4.5 µs +- 28.8 ns per loop (mean +- std. dev. of 7 runs, 100,000 loops each)
    # 3.1 µs +- 11 ns per loop (mean +- std. dev. of 7 runs, 100,000 loops each)
    """
    # Z_fs_ec = spin_vec.copy()
    Y_fs_ec = np.cross(spin_vec, -r_vec)
    X_fs_ec = np.cross(Y_fs_ec, spin_vec)

    # The input rh or spin vector mignt not be unit vectors, so divide by
    # lengths to make a suitable matrix:
    X_fs_ec = X_fs_ec / np.linalg.norm(X_fs_ec)
    Y_fs_ec = Y_fs_ec / np.linalg.norm(Y_fs_ec)
    Z_fs_ec = spin_vec / np.linalg.norm(spin_vec)

    m1 = np.vstack((X_fs_ec, Y_fs_ec, Z_fs_ec)).T
    # m = np.linalg.inv(m1)
    return inv(m1, overwrite_a=True, check_finite=False)


def mat_fs2bf(phase: F_OR_ARR) -> np.ndarray:
    """ The conversion matrix to convert frame system to body-fixed frame.

    Parameters
    ----------
    phase : float
        The rotational phase (2 PI t / P_rot), in the unit of radian.

    Return
    ------
    m : 3-by-3 matrix
        The matrix that converts frame system coordinate to body-fixed frame.

    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus, 243, 58.
    If ``a`` is a vector in frame system (in Cartesian (x, y, z)), ``m @ a``
    will give the components of vector ``a`` in body-fixed frame, where ``m``
    is the result of this function.
    """
    _c = np.cos(phase)
    _s = np.sin(phase)
    return np.array([[-_c, -_s, 0], [_s, -_c, 0], [0, 0, 1]])


def mat_bf2ss(colat: F_OR_Q_OR_ARR) -> np.ndarray:
    """ The conversion matrix to convert body-fixed frame to surface system.

    Parameters
    ----------
    colat__deg : float, Quantity or ndarray
        The co-latitude of the surface (in degrees unit). Co-latitude is the
        angle between the pole (spin) vector and the normal vector of the
        surface of interest.

    Return
    ------
    m : 3-by-3 matrix
        The matrix that converts body-fixed coordinate to surface system.

    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus, 243, 58.
    If ``a`` is vector in body-fixed frame (in Cartesian (x, y, z)), ``m @ a``
    will give the components of vector ``a`` in surface system, where ``m`` is
    the result of this function.
    """
    colat__deg = to_quantity(colat, u.deg, to_value=True)
    _c = np.cos(colat__deg * D2R)
    _s = np.sin(colat__deg * D2R)
    return np.array(((0., 1., 0.), (-_c, 0., _s), (_s, 0., _c)))


def calc_mu_vals(
        r_vec: np.ndarray, spin_vec: np.ndarray,
        phases: F_OR_Q_OR_ARR, colats: F_OR_Q_OR_ARR, r_sun: F_OR_Q = None,
        full: bool = False
) -> np.ndarray:
    """ The conversion matrix to convert body-fixed frame to surface system.

    Parameters
    ----------
    r_vec, spin_vec : 1-D array
        The Cartesian coordinate of the asteroid and spin vector in ecliptic
        coordinate. `r_vec` can be heliocentric or observer-centric
        (geocentric, planetocentric, etc).

    phases : float or array of float or ~astropy.Quantity
        The phase values (in radian unit if floats)

    colats : float or array of float or ~astropy.Quantity
        The co-latitude of the surface (in degrees unit if float). Co-latitude
        is the angle between the pole (spin) vector and the normal vector of
        the surface of interest.

    r_sun : float or ~astropy.Quantity, optional
        The size of the solar disc. If given, partial illumination of the solar
        disc will be calculated. In degrees unit if float.

    Return
    ------
    mu_vals : 2-D array
        The mu values.

    solar_dirs : 3-D array
        The direction to the Sun (``(x, y, z)`` along ``axis=2``).
        Returned only if ``full=True``.

    mat_ec2fs : ndarray
        The conversion matrix to convert ecliptic to frame system, i.e., the
        result of ``mat_ec2fs(r_vec=r_vec, spin_vec=spin_vec)``.
        Returned only if ``full=True``.

    mat_fs2bf_arr : ndarray
        The conversion matrix to convert frame system to body-fixed frame,
        i.e., the result of ``mat_fs2bf(phase=phase)`` for all ``phase in
        phases``.
        Returned only if ``full=True``.

    mat_bf2ss_arr : ndarray
        The conversion matrix to convert body-fixed frame to surface system,
        i.e., the result of ``mat_bf2ss(colat__deg=colat)`` for all ``colat in
        colats``..
        Returned only if ``full=True``.

    Note
    ----
    Adopted from Sect. 2.4. of Davidsson and Rickman (2014), Icarus, 243, 58.
    If ``a`` is vector in body-fixed frame (in Cartesian (x, y, z)), ``m @ a``
    will give the components of vector ``a`` in surface system, where ``m`` is
    the result of this function.
    """
    colats__deg = to_val(colats, u.deg)
    phases__rad = to_val(phases, u.rad)
    r_vec_norm = r_vec/np.linalg.norm(r_vec)
    spin_vec_norm = spin_vec/np.linalg.norm(spin_vec)

    mat_fs2bf_arr = np.array([mat_fs2bf(phase=phase) for phase in phases__rad])
    mat_bf2ss_arr = np.array([mat_bf2ss(colat=colat) for colat in colats__deg])
    _mat_ec2fs = mat_ec2fs(r_vec=r_vec_norm, spin_vec=spin_vec_norm)

    _dirs = np.array([
        _m @ mat_fs2bf_arr @ _mat_ec2fs @ -(r_vec_norm)
        for _m in mat_bf2ss_arr
    ])

    # Z component = cos i_sun for mu_sun case.
    mu_vals = _dirs[:, :, 2].copy() if full else _dirs[:, :, 2]
    if r_sun is not None:
        calc_partial_solar_disc(mu_vals, to_val(r_sun, u.deg))
    else:
        mu_vals[mu_vals < 0] = 0

    if full:
        return mu_vals, _dirs, _mat_ec2fs, mat_fs2bf_arr, mat_bf2ss_arr
    return mu_vals


@njit(parallel=True)
def calc_partial_solar_disc(cos_i, r_sun__deg):
    """ Calculate the partial solar disc illumination.

    Parameters
    ----------
    cos_i : 2-D array
        The cosine of the incidence angle.

    r_sun__deg : float
        The size of the solar disc in degrees.

    Return
    ------
    h_r : 2-D array
        The partial solar disc illumination.
    """
    cos_i_max = np.cos((90 - r_sun__deg)*D2R)
    cos_i_min = np.cos((90 + r_sun__deg)*D2R)
    ycfactor = 2/3*r_sun__deg
    for i in nb.prange(cos_i.shape[0]):
        for j in range(cos_i.shape[1]):
            _cos_i = cos_i[i, j]
            if (_cos_i < cos_i_min):
                cos_i[i, j] = 0
            elif (_cos_i > cos_i_max):
                continue
            else:
                incid = np.arccos(_cos_i)*R2D  # in degree
                h_r = (90 - incid)/r_sun__deg
                _th = np.arccos(-h_r)  # in radian
                _hrsq = 1 - h_r*h_r
                _hrfactor = h_r*np.sqrt(_hrsq)
                y_c = ycfactor*(_hrsq)**(3/2)/(_th + _hrfactor)
                frac = (_th + _hrfactor)/np.pi
                cos_i[i, j] = frac*np.cos((incid - y_c)*D2R)


@njit()
def mat_bf2ss_nb(colat__deg: float) -> np.ndarray:
    """ The conversion matrix to convert body-fixed frame to surface system.

    See mat_bf2ss for the docstring.
    """
    _c = np.cos(colat__deg * np.pi/180)
    _s = np.sin(colat__deg * np.pi/180)
    return np.array(((0., 1., 0.), (-_c, 0., _s), (_s, 0., _c)))


@njit()
def mat_ec2fs_nb(r_vec_norm: np.ndarray, spin_vec_norm: np.ndarray) -> np.ndarray:
    """ The conversion matrix to convert ecliptic to frame system.

    see mat_ec2fs for the docstring.
    """
    # Z_fs_ec = spin_vec.copy()
    Y_fs_ec = np.cross(spin_vec_norm, -r_vec_norm)
    X_fs_ec = np.cross(Y_fs_ec, spin_vec_norm)

    m1 = np.vstack((
        X_fs_ec/np.linalg.norm(X_fs_ec),
        Y_fs_ec/np.linalg.norm(Y_fs_ec),
        spin_vec_norm
    )).T
    return np.ascontiguousarray(np.linalg.inv(m1))


@njit()
def mat_fs2bf_nb(phase__rad: float) -> np.ndarray:
    """ The conversion matrix to convert frame system to body-fixed frame.

    See mat_fs2bf for the docstring.
    """
    _c = np.cos(phase__rad)
    _s = np.sin(phase__rad)
    m = np.array(([-_c, -_s, 0.], [_s, -_c, 0.], [0., 0., 1.]))
    return m


@njit
def calc_mu_val_nb(
    r_vec_norm: np.ndarray,
    spin_vec_norm: np.ndarray,
    phi__rad: float,
    mat_bf2ss: np.ndarray,
):
    """ Calculates mu value
    """
    m1 = mat_ec2fs_nb(r_vec_norm=r_vec_norm, spin_vec_norm=spin_vec_norm)
    m2 = mat_fs2bf_nb(phase__rad=phi__rad)
    solar_dir = (mat_bf2ss @ m2 @ m1 @ -(r_vec_norm))
    # Z component = cos i_sun for mu_sun case:
    mu_val = solar_dir[2]
    return mu_val if mu_val > 0. else 0.


# @njit
# def eph_idx_arr(idarr: np.ndarray, eph_jds: np.ndarray, dt_1: float) -> None:
#     # d_eph_jds = eph_jds - eph_jds[0]
#     idx_of_eph_jds = 0
#     idarr[0] = 0
#     time_i = eph_jds[0]
#     jd2compare = eph_jds[1]
#     for i in range(1, idarr.size - 1):
#         time_i += dt_1
#         if time_i >= jd2compare:
#             idx_of_eph_jds += 1
#             jd2compare = eph_jds[idx_of_eph_jds]
#         idarr[i] = idx_of_eph_jds


@njit(parallel=True, )  # fastmath={"nnan", "ninf", "nsz"}
def calc_varr_orbit(
        varrs_init: np.ndarray,
        phi0s: np.ndarray,
        spin_vec_norm: np.ndarray,
        mat_bf2ss: np.ndarray,
        r_hel_vecs: np.ndarray,
        r_hels: np.ndarray,
        true_anoms__deg: np.ndarray,
        psis__rad: np.ndarray,
        thpar1: float,
        deps: np.ndarray,
        full: bool = False
) -> np.ndarray:
    """
    rot_period__day : float
        The rotation period of the asteroid (in days).
    """
    n_calc = r_hel_vecs.shape[0]  # must be == true_anoms__deg.size
    n_location = phi0s.shape[0]
    varrs_surf = np.zeros((n_location, n_calc))  # 1-D along time for n_location
    varrs_new = varrs_init.copy()  # 1-D along depth at the last iteration
    phi_last = np.empty(n_location)
    # mu_last = np.empty(n_location)
    mu_vals = np.empty((n_location, n_calc))

    for iloc in nb.prange(n_location):
        phi0 = phi0s[iloc]
        mat = mat_bf2ss[iloc, :, :]
        varr_old = varrs_init[iloc].copy()
        varr_new = varrs_init[iloc].copy()
        for i in range(n_calc - 1):
            df__rad = (true_anoms__deg[i+1] - true_anoms__deg[i])*D2R
            phi__rad = (phi0 + psis__rad[i+1] - df__rad)  # %(2*PI)
            # I think %(2*PI) is unnecessary...?            ^^^^^^^
            mu = calc_mu_val_nb(
                r_vec_norm=r_hel_vecs[i, :]/r_hels[i],
                spin_vec_norm=spin_vec_norm,
                phi__rad=phi__rad,
                mat_bf2ss=mat
            )
            update_varr(
                varr_old=varr_old,
                varr_new=varr_new,
                thpar=thpar1,
                dlon__rad=psis__rad[i+1] - psis__rad[i],
                deps=deps,
                mu_sun=mu/r_hels[i]**2
            )
            varrs_surf[iloc, i] = varr_new[0]
            mu_vals[iloc, i] = mu
            varr_old = varr_new.copy()
            # varr_new *= 0

        varrs_new[iloc] = varr_new
        phi_last[iloc] = phi__rad
        # mu_last[iloc] = mu

    return varrs_surf, varrs_new, phi_last, mu_vals


# @njit(parallel=True, fastmath={"nnan", "ninf", "nsz"})
# def calc_varr_orbit(
#         varrs_init: np.ndarray,
#         eph_jds: np.ndarray,
#         r_hels__au: np.ndarray,
#         r_hel_vecs__au: np.ndarray,
#         spin_vec: np.ndarray,
#         true_anoms__deg: np.ndarray, lons__deg: float, lats__deg: float,
#         rot_period__day: float,
#         n_per_rot: int,
#         full: bool = False
# ) -> np.ndarray:
#     """
#     rot_period__day : float
#         The rotation period of the asteroid (in days).
#     """

    # npoints = lons__deg.size
    # dt_1 = rot_period__day/n_per_rot
    # # [day]
    # fullsize = int((eph_jds.max() - eph_jds.min()) / dt_1) + 1
    # eph_size = eph_jds.size
    # # fullsize: the total number of calculations to be done
    # muarrs = np.empty(shape=(npoints, fullsize))
    # varrs_surf = np.empty(shape=(npoints, fullsize))

    # for ipoint in nb.prange(npoints):
    #     varr_old = np.zeros_like(varrs_init[ipoint]) + varrs_init[ipoint]
    #     varr_new = np.zeros_like(varrs_init[ipoint]) + varrs_init[ipoint]
    #     varrs_surf[ipoint, 0] = varr_old[0]
    #     lon = lons__deg[ipoint]
    #     lat = lats__deg[ipoint]
    #     m3 = mat_bf2ss_nb(colat__deg=90-lat)

    #     idx_of_eph_jds = 0
    #     time_i = eph_jds[0]
    #     jd2compare = eph_jds[1]
    #     r_hel = r_hels__au[0]
    #     r_hel_vec = r_hel_vecs__au[0]
    #     true_anom = true_anoms__deg[0]

    #     for i in range(fullsize - 1):

    #     while True:

    #     for i_row in range(eph_size-1):
    #         dt_2 = eph_jds[i_row+1] - eph_jds[i_row]

    #         r_hel_vec0 = r_hel_vecs__au[i_row]
    #         r_hel0 = r_hels__au[i_row]
    #         # query time step
    #         # change rate (linear interpolation)
    #         dr_dt = (r_hel_vecs__au[i_row+1] - r_vec0)/dt_q
    #         df_dt = (true_anoms__deg[i_row+1] - true_anoms__deg[i_row])/dt_q

    #         i_time = 0
    #         psi__rad = lon*D2R  # Initial value
    #         phi0__rad = lon*D2R  # Initial value

    #         while True:
    #             if delta_t >= dt_q:
    #                 break
    #             delta_psi__rad = 2*PI*delta_t/rot_period__day
    #             psi__rad = (psi__rad + delta_psi__rad)%(2*PI)
    #             phi__rad = (phi0__rad + delta_psi__rad - df_dt*delta_t*D2R)%(2*PI)
    #             r_vec = r_vec0 + dr_dt*delta_t
    #             mu_val = calc_mu_val_nb(r_vec, spin_vec, phi__rad, m3)
    #             muarrs[ipoint, i_row*n_per_rot + i_time] = mu_val
    #             i_time += 1
    # return muarrs


@njit
def update_varr(
        varr_old: np.ndarray,
        varr_new: np.ndarray,
        thpar: float,
        dlon__rad: float,
        deps: np.ndarray,
        mu_sun: float
) -> None:
    """Calculates the v value in usual TPM.
    """
    for i_z in range(1, deps.size - 1):
        varr_new[i_z] = (
            varr_old[i_z]
            + dlon__rad/(deps[i_z+1] - deps[i_z])**2
            * (varr_old[i_z - 1] + varr_old[i_z + 1] - 2*varr_old[i_z])
        )
    # Deepest cell
    varr_new[-1] = varr_new[-2]
    # Surface cell
    varr_new[0] = newton_iter_tpm(
        newu0_init=varr_old[0],
        newu1=varr_new[1],
        thpar=thpar,
        dZ=deps[1] - deps[0],
        mu_sun=mu_sun
    )


@njit()
def newton_iter_tpm(
        newu0_init: float,
        newu1: float,
        thdz: float,
        mu_sun: float,
        iter_max: int = 5000,
        atol: float = 1.e-8
) -> float:
    """ Root finding using Newton's method

    Parameters
    ----------
    newu0_init : float
        The first trial to the ``newu[0]`` value, i.e., the ansatz of
        ``newu[0]`` value.

    newu1 : float
        The ``newu[1]`` value (that will have been calculated before this
        function will be called).

    thdz : float
        The theta/dz, where theta is the thermal parameter, and dz is the depth
        slab resolution in the thermal skin depth unit.

    mu_sun : float
        The cosine of the incident angle (zenith angle of the Sun).

    iter_max : int, optional
        The maximum number of iteration to halt the root finding.

    atol : float, optional
        If the absolute difference is smaller than ``atol``, the iteration will
        stop.
    """
    x0 = newu0_init
    for _ in range(iter_max):
        f0 = x0*x0*x0*x0 - mu_sun - thdz * (newu1 - x0)
        slope = 4 * x0*x0*x0 + thdz
        delta = f0 / slope

        # It is good if the iteration ends here:
        if abs(delta) < atol:
            return x0 - delta

        # Reset for next iteration
        x0 = x0 - delta
    return x0


# Tested on 15"MBP2018: speed is by ~10 times faster if parallel is used.
@njit(parallel=True)
def calc_uarr_tpm(
        u_arr: np.ndarray,
        thpar: float,
        dlon: float,
        dZ: float,
        mu_suns: np.ndarray,
        min_iter: int = 50,
        max_iter: int = 5000,
        min_elev__deg: float = 0.,
        permanent_shadow_u: float = 0,
        use_surfmean: bool = False,
        atol: float = 1.e-8
):
    """Calculates the u value in usual TPM.
    Parameters
    ----------
    u_arr : 3d-array
        The u (u is used as the normalized temperature, T/T_EQM) array that
        must have been defined a priori. It must be satisfied that ``u_arr[i,
        j, k]`` is u of ``i``th colatitude, ``j``th time, and ``k``th depth.
        In yssbtmpy.core, the axis 1 (time axis) has length of ``ntime + 1``,
        so the code below will understand ``ntime = u_arr.shape[1] - 1``.

    thpar : float
        The thermal parameter (frequently denoted by Theta).

    dlon, dZ : float
        The longitude and depth bin size in units of radian and thermal skin
        depth. `dlon` is identical to ``dT`` in M. Mueller (2007 thesis)'s
        notation, for instance.

    min_iter, max_iter : int, optional
        The minimum or maxumum number of iteration for the equilibrium
        temperature calculation.

    min_elev__deg : int or float, optional
        The minimum elevation to check whether the latitude is assumed as a
        permanently shadowed region.
        The latitudinal band is assumed to be in a permanent shadow if the sun
        is always below this elevation, and all the temperature on this
        latitude is just set as a constant given by `permanent_shadow_u in the
        unit of ``temp_eqm``.

    permanent_shadow_u : float
        The temperature to be substituted for permanently shadowed regions
        (unit of ``temp_epm``). If `None`, the latitudinal band will just let
        be cooled over the rotations (maybe useful for seasonal variation
        calculations).

    use_surfmean : bool, optional
        If `True`, the deepest temperature will be set as the mean of the
        surface temperature of the previous iteration, to try to guarantee
        the mathematical condition (time average of surface temp = deepest
        temp). Maybe useful for very accurate deep temperature calculation.

    atol : float, optional
        The absolute tolerance for the iteration to stop. (Stops if the T/T_EQM
        < `atol`).
    """
    ncolat, ntimep1, ndepth = u_arr.shape
    ntime = ntimep1 - 1
    dtdz2 = dlon/dZ**2
    thdz = thpar/dZ

    mu_sun_min = np.cos((90 - min_elev__deg)*D2R)
    # For each colatitude, parallel calculation is possible!!!
    # So use numba's prange rather than range:
    for i_lat in nb.prange(ncolat):
        if permanent_shadow_u is not None:
            # Check whether the latitude is under permanent shadow.
            permanent_shadow = True
            for k in range(ntime):
                # If the sun reaches above ``min_elev__deg``, i.e.,
                #   mu_sun = cos(90 - EL_sun) > cos(90 - min_elev__deg)
                # at least once, it's not a permanent shadow:
                if mu_suns[i_lat, k] > mu_sun_min:
                    # If sun rises > min_elev__deg
                    permanent_shadow = False
                    break
            if permanent_shadow:
                for i_t in range(ntimep1):
                    for i_dep in range(ndepth):
                        u_arr[i_lat, i_t, i_dep] = permanent_shadow_u
        else:
            permanent_shadow = False

        if not permanent_shadow:
            discrep = 1.
            if use_surfmean:
                surf_mean = np.mean(u_arr[i_lat, :, 0])

            for i_iter in range(max_iter):
                for i_t in range(ntime):
                    # cells other then surface/deepest ones
                    for i_z in range(1, ndepth - 1):
                        u_arr[i_lat, i_t + 1, i_z] = ftcs(
                            x=u_arr[i_lat, i_t, i_z],
                            xm1=u_arr[i_lat, i_t, i_z - 1],
                            xp1=u_arr[i_lat, i_t, i_z + 1],
                            factor=dtdz2
                        )
                    # Surface cell
                    u_arr[i_lat, i_t + 1, 0] = newton_iter_tpm(
                        newu0_init=u_arr[i_lat, i_t, 0],
                        newu1=u_arr[i_lat, i_t + 1, 1],
                        thdz=thdz,
                        mu_sun=mu_suns[i_lat, i_t]
                    )

                    # Deepest cell
                    if use_surfmean:
                        u_arr[i_lat, i_t + 1, -1] = surf_mean
                    else:
                        u_arr[i_lat, i_t + 1, -1] = u_arr[i_lat, i_t + 1, -2]

                discrep = np.abs(u_arr[i_lat, 0, 0] - u_arr[i_lat, -1, 0])

                if i_iter >= min_iter and discrep < atol:
                    break

                for i in range(ndepth):
                    u_arr[i_lat, 0, i] = u_arr[i_lat, -1, i]

                if use_surfmean:
                    surf_mean = np.mean(u_arr[i_lat, :, 0])

    # TODO: Return i_iter?


@njit()
def ftcs(x, xm1, xp1, factor):
    """ forward time-centered space propagaion.

    Parameters
    ----------
    x, xm1, xp1 : float
        The value at certain index, index minus 1 (`xm1`), and plus 1 (`xp1`).

    factor : float
        The factor multiplied: ``out = x + factor*(xp1 + xm1 - 2x)``.
    """
    return x + factor*(xp1 + xm1 - 2*x)


@njit(parallel=True)
def calc_flux_tpm(fluxarr, wlen, tempsurf, mu_obss, colats, dlat, dlon):
    """ Calculates the fulx at given wlen in W/m^2/m

    Parameters
    ----------
    fluxarr : 1-d array
        The array to be filled with the flux values. Must have the identical
        length to `wlen`.
    wlen : 1-d array
        The wavelength corresponding to `fluxarr`, in um. Both must have the
        identical length.
    tempsurf : 2-d array
        The surface temperature in Kelvin. The value at `tempsurf[i, j]` must
        be corresponding to the `mu_obs[i, j]`.
    mu_obs : 2-d array
        The cosine factor for the emission direction to the observer. The value
        at `tempsurf[i, j]` must be corresponding to the `mu_obs[i, j]`.
    colats : 1-d array
        The co-latitude of the surface (in radians unit). Co-latitude is the
        angle between the pole (spin) vector and the normal vector of the
        surface of interest.
    dlat, dlon : float
        The delta-latitude and -longitude of patches in radian unit.
    """
    dlon_dlat = dlon*dlat
    for k in nb.prange(len(wlen)):
        wl = wlen[k]*1.e-6  # um to m
        factor1 = 2*HH*CC**2/wl**5
        factor2 = (HH*CC)/(KB*wl)
        for i in range(tempsurf.shape[0]):
            area_factor = np.sin(colats[i])*dlon_dlat
            for j in range(tempsurf.shape[1]):
                mu_obs = mu_obss[i, j]
                if mu_obs > 0:
                    radiance = factor1/(np.expm1(factor2/tempsurf[i, j]))
                    fluxarr[k] += radiance*mu_obs*area_factor
