"""Near-Earth Asteroid Thermal Model (NEATM).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from astropy import units as u

from .constants import HH, KB, CC, PI, AU, D2R
from .util import F_OR_Q, F_OR_Q_OR_ARR, to_val

__all__ = ["NeatmBody"]

_PI  = float(PI)
_AU  = float(AU)   # m
_D2R = float(D2R)  # deg -> rad

# Planck radiation constants (pre-computed for the kernel)
_C1 = 2.0 * float(HH) * float(CC) * float(CC)   # 2*h*c^2  [W*m^2/sr]
_C2 = float(HH) * float(CC) / float(KB)          # h*c/k_B  [m*K]


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def _neatm_flux_kernel(
    wlen_m:  np.ndarray,
    T1:      float,
    cos_sun: np.ndarray,
    cos_obs: np.ndarray,
    area:    np.ndarray,
) -> np.ndarray:
    r"""NEATM surface integral, parallelised over wavelengths.

    Computes

    .. math::
        F_{\rm base}(\lambda)
        = \sum_{i:\,\cos\theta_i>0,\;\cos\psi_i>0}
          B_\lambda\!\left(\lambda,\; T_1\,\cos\theta_i^{1/4}\right)
          \cos\psi_i\;\Delta\Omega_i

    where :math:`B_\lambda` is the Planck function in W/m^2/m/sr.

    Parameters
    ----------
    wlen_m : (nw,) float64 array
        Wavelengths in metres.
    T1 : float
        Sub-solar equilibrium temperature [K].
    cos_sun : (nf,) float64 array
        cos(theta) per facet.  All entries must be > 0 (the caller
        pre-filters to the illuminated hemisphere).
    cos_obs : (nf,) float64 array
        cos(psi) per facet.  Only entries > 0 contribute; the rest
        are skipped.
    area : (nf,) float64 array
        Solid-angle element dOmega = sin(theta)*d_theta*d_phi [sr] per facet.

    Returns
    -------
    flux_base : (nw,) float64 array
        Surface-integrated Planck intensity in W/m^2/m.
        Multiply by eps(lambda)*(D/2)^2/r_obs^2*1e-6 to get W/m^2/um.
    """
    nw = wlen_m.shape[0]
    nf = cos_sun.shape[0]
    flux_base = np.zeros(nw)
    for k in prange(nw):
        wl = wlen_m[k]
        c1 = _C1 / (wl * wl * wl * wl * wl)
        c2 = _C2 / wl
        s = 0.0
        for i in range(nf):
            co = cos_obs[i]
            if co > 0.0:
                T = T1 * cos_sun[i] ** 0.25
                s += (c1 / np.expm1(c2 / T)) * co * area[i]
        flux_base[k] = s
    return flux_base


@njit(cache=True, parallel=True)
def _neatm_flux_kernel_batch(
    wlen_m:  np.ndarray,
    T1_arr:  np.ndarray,
    cos_sun: np.ndarray,
    cos_obs: np.ndarray,
    area:    np.ndarray,
) -> np.ndarray:
    r"""NEATM surface integral for a batch of bodies, parallelised over bodies.

    Same physics as :func:`_neatm_flux_kernel`, but vectorised over an array
    of sub-solar temperatures.  Parallelisation is over bodies (outer loop),
    which is efficient when the number of bodies >> number of wavelengths
    (typical in survey-scale fitting).

    Parameters
    ----------
    wlen_m : (nw,) float64 array
        Wavelengths in metres.
    T1_arr : (nb,) float64 array
        Sub-solar equilibrium temperatures [K] for each body.
    cos_sun, cos_obs, area : (nf,) float64 arrays
        Pre-computed geometry (same for all bodies — same phase angle).

    Returns
    -------
    flux_base : (nb, nw) float64 array
        Surface-integrated Planck intensity [W/m^2/m] for each body.
    """
    nb = T1_arr.shape[0]
    nw = wlen_m.shape[0]
    nf = cos_sun.shape[0]
    flux_base = np.zeros((nb, nw))
    for ib in prange(nb):
        T1 = T1_arr[ib]
        for k in range(nw):
            wl = wlen_m[k]
            c1 = _C1 / (wl * wl * wl * wl * wl)
            c2 = _C2 / wl
            s = 0.0
            for i in range(nf):
                co = cos_obs[i]
                if co > 0.0:
                    T = T1 * cos_sun[i] ** 0.25
                    s += (c1 / np.expm1(c2 / T)) * co * area[i]
            flux_base[ib, k] = s
    return flux_base


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sphere_geometry(
    n_theta:   int,
    n_phi:     int,
    alpha_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build facet geometry for a uniform (theta, phi) grid over the
    illuminated hemisphere.

    Coordinate system
    -----------------
    *  z-axis: sub-solar direction  (theta = 0 at sub-solar point).
    *  The surface normal of facet (theta, phi) is
       n_hat = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)).
    *  The observer unit vector is o_hat = (-sin(alpha), 0, cos(alpha)),
       so the sub-observer point (where n_hat = o_hat) is at
       theta = alpha, phi = pi.
    *  phi = 0 lies on the anti-observer meridian.

    The projection formula from spherical trigonometry (Myhrvold 2018):

    .. math::
        \cos\psi = \hat n \cdot \hat o
                 = \cos\theta\cos\alpha - \sin\theta\sin\alpha\cos\phi

    The illumination factor is cos(theta)  (= cos_sun).

    Parameters
    ----------
    n_theta : int
        Number of colatitude bins over [0, pi/2].
    n_phi : int
        Number of longitude bins over [0, 2*pi).
    alpha_rad : float
        Phase angle in radians.

    Returns
    -------
    cos_sun, cos_obs, area : contiguous float64 arrays, length <= n_theta*n_phi
        Pre-filtered to facets satisfying both cos_sun > 0 and cos_obs > 0.
    """
    dtheta = 0.5 * _PI / n_theta   # theta in [0, pi/2]
    dphi   = 2.0  * _PI / n_phi

    # Bin-centred colatitudes and longitudes
    theta = (0.5 + np.arange(n_theta)) * dtheta   # (n_theta,)
    phi   = (0.5 + np.arange(n_phi))   * dphi      # (n_phi,)  centred

    cos_th = np.cos(theta)   # (n_theta,)  all > 0 since theta in (0, pi/2)
    sin_th = np.sin(theta)   # (n_theta,)
    cos_ph = np.cos(phi)     # (n_phi,)

    # cos(psi_ij) = cos(theta_i)*cos(alpha) - sin(theta_i)*sin(alpha)*cos(phi_j)
    cos_obs_2d = (
        cos_th[:, None] * np.cos(alpha_rad)
        - sin_th[:, None] * (np.sin(alpha_rad) * cos_ph[None, :])
    )   # (n_theta, n_phi)

    # solid-angle element sin(theta_i)*d_theta*d_phi  (same for all phi at fixed theta)
    area_1d = sin_th * dtheta * dphi   # (n_theta,)

    # Flatten: repeat each theta-row n_phi times
    cos_sun_flat = np.repeat(cos_th,  n_phi)   # (n_theta*n_phi,)
    cos_obs_flat = cos_obs_2d.ravel()           # (n_theta*n_phi,)
    area_flat    = np.repeat(area_1d, n_phi)   # (n_theta*n_phi,)

    # Discard facets invisible to the observer (cos_obs <= 0)
    mask = cos_obs_flat > 0.0
    return (
        np.ascontiguousarray(cos_sun_flat[mask], dtype=np.float64),
        np.ascontiguousarray(cos_obs_flat[mask], dtype=np.float64),
        np.ascontiguousarray(area_flat[mask],    dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NeatmBody:
    r"""Near-Earth Asteroid Thermal Model (Harris 1998).

    Physics
    -------
    The thermal flux received by a distant observer at wavelength lambda is

    .. math::
        F(\lambda)
        = \frac{\varepsilon(\lambda)\,(D/2)^2}{r_{\rm obs}^2}
          \sum_{i} B_\lambda\!\left(\lambda,\; T_1\,\cos\theta_i^{1/4}\right)
          \cos\psi_i\;\Delta\Omega_i

    summed over surface facets that are both illuminated (cos(theta_i) > 0)
    and visible (cos(psi_i) > 0), where

    *  T_1 = temp_eqm_1au / sqrt(r_hel):  sub-solar equilibrium temperature,
    *  cos(theta):  cosine of the facet's angle from the sub-solar direction,
    *  cos(psi) = cos(theta)*cos(alpha) - sin(theta)*sin(alpha)*cos(phi):
       cosine of the emission angle toward the observer,
    *  dOmega = sin(theta)*d_theta*d_phi:  solid-angle element on the unit
       sphere.

    The ``calc_flux_ther`` result is in W/m^2/um and scales as
    emissivity*D^2/r_obs^2.

    Coordinate convention
    ---------------------
    The sub-solar-centred frame has z = sub-solar direction.
    The observer unit vector is o_hat = (-sin(alpha), 0, cos(alpha)), so the
    sub-observer point (where n_hat = o_hat) lies at theta = alpha, phi = pi.

    Usage
    -----
    >>> body = NeatmBody(temp_eqm_1au=400, r_hel__au=1.0,
    ...                  r_obs__au=1.0, phase_ang__deg=20.0)
    >>> body.set_sphere_grid(n_theta=180, n_phi=360)
    >>> flux = body.calc_flux_ther([10.0, 12.0, 20.0],
    ...                            emissivity=0.9, diam__km=1.0)

    For a custom mesh (e.g. TriangleEllipsoid or Fibonacci lattice) call
    ``set_facets`` instead of ``set_sphere_grid``.

    Parameters
    ----------
    temp_eqm_1au : float or Quantity
        Sub-solar equilibrium temperature at 1 AU [K].
    r_hel__au : float or Quantity
        Heliocentric distance [AU].  Default 1.
    r_obs__au : float or Quantity
        Observer distance [AU].  Default 1.
    phase_ang__deg : float or Quantity
        Phase angle (Sun--target--observer) [deg].  Default 0.

    Attributes
    ----------
    temp_eqm_1au : float
        Sub-solar equilibrium temperature at 1 AU [K].
    temp_eqm : float
        Sub-solar equilibrium temperature at ``r_hel`` [K];
        T_1 = temp_eqm_1au / sqrt(r_hel).
    wlen_ther : ndarray or None
        Wavelength array [um] from the most recent ``calc_flux_ther`` call.
    flux_ther : ndarray or None
        Thermal flux [W/m^2/um] from the most recent ``calc_flux_ther`` call.

    References
    ----------
    Harris AW (1998) Icarus 131 291-301
    """

    def __init__(
        self,
        temp_eqm_1au:   F_OR_Q,
        r_hel__au:      F_OR_Q = 1.0,
        r_obs__au:      F_OR_Q = 1.0,
        phase_ang__deg: F_OR_Q = 0.0,
    ) -> None:
        self.temp_eqm_1au = float(to_val(temp_eqm_1au,   u.K))
        self._r_hel       = float(to_val(r_hel__au,     u.au))
        self._r_obs       = float(to_val(r_obs__au,     u.au))
        self._alpha       = float(to_val(phase_ang__deg, u.deg))

        # T_1 at actual heliocentric distance
        self.temp_eqm = self.temp_eqm_1au / np.sqrt(self._r_hel)

        # Geometry buffers -- filled by set_sphere_grid or set_facets
        self._cos_sun = None
        self._cos_obs = None
        self._area    = None

        # Output buffers
        self.wlen_ther  = None
        self._flux_base = None   # unnormalised integral
        self.flux_ther  = None   # W/m^2/um

    # -- geometry setters -----------------------------------------------------

    def set_sphere_grid(self, n_theta: int = 180, n_phi: int = 360) -> None:
        """Pre-compute spherical-grid geometry for the NEATM integral.

        Covers the illuminated hemisphere (theta in [0, pi/2]) with a uniform
        (theta, phi) grid centred on the sub-solar direction.  Facets invisible
        to the observer (cos(psi) <= 0) are discarded immediately, so the
        numba kernel has no branching on cos_obs.

        Parameters
        ----------
        n_theta : int
            Number of colatitude bins in [0, pi/2].  Default 180.
        n_phi : int
            Number of longitude bins in [0, 2*pi).  Default 360.
        """
        alpha_rad = self._alpha * _D2R
        self._cos_sun, self._cos_obs, self._area = _sphere_geometry(
            n_theta, n_phi, alpha_rad
        )

    def set_facets(
        self,
        normals:       np.ndarray,
        area_elements: np.ndarray,
    ) -> None:
        """Set arbitrary facet geometry (e.g. TriangleEllipsoid or
        Fibonacci lattice).

        Parameters
        ----------
        normals : (nf, 3) float array
            Outward unit normals in the sub-solar frame.
            z_hat = (0, 0, 1) is the sub-solar direction;
            o_hat = (-sin(alpha), 0, cos(alpha)) is toward the observer.
        area_elements : (nf,) float array
            Solid-angle element dOmega [sr] for each facet.

        Notes
        -----
        Facets with cos_sun <= 0 (night side) or cos_obs <= 0 (not visible)
        are silently discarded before storing.
        """
        alpha_rad = self._alpha * _D2R
        obs_dir = np.array([-np.sin(alpha_rad), 0.0, np.cos(alpha_rad)], dtype=np.float64)

        normals_d = np.asarray(normals, dtype=np.float64)
        cos_sun = normals_d[:, 2]        # n_hat dot z_hat  (z_hat = sub-solar)
        cos_obs = normals_d @ obs_dir    # n_hat dot o_hat

        mask = (cos_sun > 0.0) & (cos_obs > 0.0)
        self._cos_sun = np.ascontiguousarray(cos_sun[mask])
        self._cos_obs = np.ascontiguousarray(cos_obs[mask])
        self._area    = np.ascontiguousarray(
            np.asarray(area_elements, dtype=np.float64)[mask]
        )

    # -- geometry reconfigurators ---------------------------------------------

    def set_phase_ang(self, phase_ang__deg: F_OR_Q) -> None:
        """Update phase angle and invalidate geometry cache.

        After calling this, ``set_sphere_grid`` or ``set_facets`` must be
        called again before computing flux.

        Parameters
        ----------
        phase_ang__deg : float or Quantity
            New phase angle [deg].
        """
        self._alpha = float(to_val(phase_ang__deg, u.deg))
        self._cos_sun = None
        self._cos_obs = None
        self._area = None

    # -- flux calculation -----------------------------------------------------

    def _ensure_geometry(self) -> None:
        """Raise if no geometry is set."""
        if self._cos_sun is None:
            raise RuntimeError(
                "No geometry set -- call set_sphere_grid() or set_facets() first."
            )

    def calc_flux_ther(
        self,
        wlen:       F_OR_Q_OR_ARR,
        emissivity: float | np.ndarray = 1.0,
        diam__km:   float              = 1.0,
        temp_eqm:   float | None       = None,
    ) -> np.ndarray:
        r"""Compute the NEATM thermal flux.

        .. math::
            F(\lambda)
            = \varepsilon(\lambda)
              \frac{(D/2)^2}{r_{\rm obs}^2}
              F_{\rm base}(\lambda) \times 10^{-6}

        where :math:`F_{\rm base}` is the surface integral returned by the
        numba kernel (stored in ``self._flux_base``), and the factor
        :math:`10^{-6}` converts W/m^2/m to W/m^2/um.

        Parameters
        ----------
        wlen : array-like or Quantity
            Wavelength(s) in um (if float/ndarray) or as an astropy Quantity.
        emissivity : float or array-like, optional
            Spectral emissivity eps(lambda), dimensionless.  Must broadcast
            with ``wlen``.  Default 1.
        diam__km : float, optional
            Effective diameter in km.  Flux scales as D^2.  Default 1.
        temp_eqm : float, optional
            Override the sub-solar equilibrium temperature [K] for this call.
            If ``None`` (default), uses ``self.temp_eqm``.  Useful for fitting
            temperature without recreating the object.

        Returns
        -------
        flux_ther : ndarray, shape (nw,)
            Thermal flux in W/m^2/um.

        Raises
        ------
        RuntimeError
            If ``set_sphere_grid`` or ``set_facets`` has not been called.
        """
        self._ensure_geometry()

        wlen_um = np.asarray(
            to_val(np.atleast_1d(wlen), u.um), dtype=np.float64
        )
        self.wlen_ther = wlen_um
        wlen_m = wlen_um * 1.0e-6   # um -> m

        T1 = self.temp_eqm if temp_eqm is None else float(temp_eqm)

        self._flux_base = _neatm_flux_kernel(
            wlen_m,
            T1,
            self._cos_sun,
            self._cos_obs,
            self._area,
        )

        # Physical scaling:  eps * R_ast^2 / r_obs^2 * (m -> um)
        radius__m = diam__km * 5.0e2              # km -> m, then /2 for radius
        r_obs__m = self._r_obs * _AU             # AU -> m
        scale   = (
            np.asarray(emissivity, dtype=np.float64)
            * (radius__m / r_obs__m) ** 2
            * 1.0e-6
        )
        self.flux_ther = self._flux_base * scale
        return self.flux_ther

    def calc_flux_ther_batch(
        self,
        wlen:        F_OR_Q_OR_ARR,
        temp_eqms:   np.ndarray,
        diam__kms:   np.ndarray,
        emissivity:  float | np.ndarray = 1.0,
    ) -> np.ndarray:
        r"""Batch-compute NEATM thermal flux for many bodies at the same geometry.

        Efficient for survey-scale fitting: the geometry (phase angle) is
        shared, and the kernel parallelises over bodies rather than wavelengths.

        Parameters
        ----------
        wlen : array-like or Quantity
            Wavelength(s) in um.
        temp_eqms : (nb,) float array
            Sub-solar equilibrium temperature [K] for each body.
        diam__kms : (nb,) float array
            Effective diameter [km] for each body.
        emissivity : float or (nb,) or (nb, nw) array, optional
            Spectral emissivity.  Default 1.

        Returns
        -------
        flux_ther : (nb, nw) float64 array
            Thermal flux [W/m^2/um] for each body and wavelength.
        """
        self._ensure_geometry()

        wlen_um = np.asarray(
            to_val(np.atleast_1d(wlen), u.um), dtype=np.float64
        )
        wlen_m = wlen_um * 1.0e-6

        temp_eqms = np.asarray(temp_eqms, dtype=np.float64)
        diam__kms = np.asarray(diam__kms, dtype=np.float64)
        nb = temp_eqms.shape[0]

        flux_base = _neatm_flux_kernel_batch(
            wlen_m,
            temp_eqms,
            self._cos_sun,
            self._cos_obs,
            self._area,
        )

        # Scale each body:  eps * (D_i/2)^2 / r_obs^2 * 1e-6
        r_obs__m = self._r_obs * _AU
        radii__m = diam__kms * 5.0e2   # (nb,)
        geo_scale = (radii__m / r_obs__m) ** 2 * 1.0e-6   # (nb,)

        eps = np.asarray(emissivity, dtype=np.float64)
        if eps.ndim == 0:
            scale = geo_scale * float(eps)           # (nb,)
        elif eps.ndim == 1 and eps.shape[0] == nb:
            scale = geo_scale * eps                  # (nb,)
        else:
            # (nb, nw) emissivity
            return flux_base * (geo_scale[:, None] * eps)

        return flux_base * scale[:, None]
