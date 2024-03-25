import datetime
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from astroquery.jplsbdb import SBDB
from scipy.interpolate import (RectBivariateSpline, RegularGridInterpolator,
                               interp1d)

from .constants import FLAMU, GG_Q, GG, NOUNIT, PI, R2D, TIU, AU
from .relations import (p2w, solve_Gq, solve_pAG, solve_pDH, solve_rmrho,
                        solve_temp_eqm, solve_thermal_par)
from .scat.phase import iau_hg_model
from .scat.solar import SOLAR_SPEC
from .util import (F_OR_ARR, F_OR_Q, F_OR_Q_OR_ARR, add_hdr, calc_aspect_ang,
                   calc_flux_tpm, calc_mu_vals, calc_uarr_tpm, calc_varr_orbit,
                   lonlat2cart, mat_bf2ss, to_quantity, to_val, calc_flux_neatm)

__all__ = ["NEATMBody", "SmallBody", "OrbitingSmallBody"]

# TODO: Maybe we can inherit the Phys class from sbpy to this, so that
#   the physical data (hmag_vis, Prot, etc) is imported from, e.g., SBDB
#   by default.


_NEATM_FLUX_THER_COEFF = (
    ((1*u.km).to_value(u.m))**2  # diameter in km to m
    / 4  # radius to diameter
    / 1.e+6  # W/m^2/m to W/m^2/um
) # value = 0.25


class NEATMBody():
    def __init__(self, r_hel=1, r_obs=1, alpha=0, temp_eqm_1au=None, skip_quantity=False):
        """ Initializes NEATM object
        Parameters
        ----------
        r_hel, r_obs : float, Quantity
            The heliocentric and observer distance (in au if `float`).

        alpha : float, Quantity
            The phase angle (in degrees if `float`).

        temp_eqm_1au : float, Quantity
            The equilibrium temperature at 1 au (in Kelvin if `float`).

        skip_quantity : bool
            If `True`, the input values are not converted to Quantity.
            The user is responsible to check the unit consistency.
        """
        self.skip_quantity = skip_quantity
        if self.skip_quantity:
            self.r_hel = r_hel
            self.r_obs = r_obs
            self.alpha = alpha
            self.temp_eqm_1au = temp_eqm_1au
            self.temp_eqm = temp_eqm_1au/np.sqrt(r_hel)
        else:
            self.r_hel = to_quantity(r_hel, u.au)
            self.r_obs = to_quantity(r_obs, u.au)
            self.alpha = to_quantity(alpha, u.deg)
            self.temp_eqm_1au = to_quantity(temp_eqm_1au, u.K)
            self.temp_eqm = self.temp_eqm_1au/np.sqrt(self.r_hel.value)

    def calc_flux_ther(self, wlen: float, nlon=360, nlat=90):
        """
        The true flux should be scaled by
            self.flux_ther * (emissivity(lambda)) * (diam_eff/1km)^2
        and the resulting unit is W/m^2/um.
        """
        _wl = np.atleast_1d(wlen)
        if self.skip_quantity:
            self.wlen_ther = _wl
            fluxarr = np.zeros(len(_wl))
            calc_flux_neatm(
                fluxarr=fluxarr,
                wlen=_wl,
                temp_eqm=self.temp_eqm,
                alpha__deg=self.alpha,
                nlon=nlon,
                nlat=nlat,
            )
            _ro__m = self.r_obs * AU
        else:
            self.wlen_ther = to_quantity(_wl, u.um)
            fluxarr = np.zeros(len(_wl))
            calc_flux_neatm(
                fluxarr=fluxarr,
                wlen=self.wlen_ther.to_value(u.um),
                temp_eqm=self.temp_eqm.to_value(u.K),
                alpha__deg=self.alpha.to_value(u.deg),
                nlon=nlon,
                nlat=nlat,
            )
            _ro__m = self.r_obs.value * AU

        self.flux_ther = fluxarr*_NEATM_FLUX_THER_COEFF/(_ro__m*_ro__m)
        if not self.skip_quantity:
            self.flux_ther *= FLAMU



class OrbitingConvexSlope():
    def __init__(self, thermal_par_1au, spin_ecl, r_hel_vecs, r_obs_vecs,
                 slope=0*u.deg, azimuth=0*u.deg):
        """ Initializes orbiting column object
        Parameters
        ----------
        thermal_par_1au

        r_hel_vecs : ndarray
            The heliocentric position vectors of the column in the unit of au.

        """
        self.thpar_1au = to_val(thermal_par_1au)
        self.r_hel_vecs = to_quantity(np.atleast_2d(r_hel_vecs), u.au)
        self.r_obs_vecs = to_quantity(np.atleast_2d(r_obs_vecs), u.au)
        self.spin_ecl = to_quantity(spin_ecl, u.deg)
        self.r_hels = np.linalg.norm(self.r_hel_vecs, axis=1)
        self.thpars = self.thpar_1au*self.r_hels**1.5
        self.slope = to_quantity(slope, u.deg)
        self.azimuth = to_quantity(azimuth, u.deg)

    def propagate(self):
        pass


class SmallBodyMixin():
    """ Set those that do not change by orbit/rotation.
    """

    def _set_aspect_angle(self):
        if self.aspect_ang is None:
            self.aspect_ang, self.aspect_ang_obs, self.dlon_sol_obs = calc_aspect_ang(
                spin_vec=self.spin_vec,
                r_hel_vec=self.r_hel_vec,
                r_obs_vec=self.r_obs_vec,
                phase_ang=self.phase_ang
            )
            """
            r_hel_hat = self.r_hel_vec/self.r_hel
            r_obs_hat = self.r_obs_vec/self.r_obs
            # r_obs_hat = lonlat2cart(self.obs_ecl_helio.lon,
            #                         self.obs_ecl_helio.lat)

            self.aspect_ang = to_quantity(
                np.rad2deg(np.arccos(np.inner(-1*r_hel_hat, self.spin_vec))),
                u.deg
            )
            # aux_cos_sun = np.inner(r_hel_hat, self.spin_vec)
            # self.aspect_ang = (180-np.rad2deg(np.arccos(aux_cos_sun)))*u.deg
            self.aspect_ang_obs = to_quantity(
                np.rad2deg(np.arccos(np.inner(-1*r_obs_hat, self.spin_vec))),
                u.deg
            )
            # aux_cos_obs = np.inner(r_obs_hat, self.spin_vec)
            # aspect_ang_obs = (180-np.rad2deg(np.arccos(aux_cos_obs)))*u.deg

            # Using the sign(alpha) convention from DelboM 2004 PhDT p.144,
            # that is, it is + when observer sees the afternoon/evening side.
            # sign([(-r_obs) x (-r_sun)] \cdot spin) = sign(alpha) = sign(dphi)
            # where dphi is the longitude difference between Sub-solar and
            # Sub-observer points:
            sign = np.sign(np.inner(np.cross(r_obs_hat, r_hel_hat), self.spin_vec))

            # cos(dphi) = [(cos(asp_sun)cos(asp_obs) - cos(alpha))
            # / sin(asp_sun)sin(asp_obs)]
            cc = np.cos(self.aspect_ang)*np.cos(self.aspect_ang_obs)
            ss = np.sin(self.aspect_ang)*np.sin(self.aspect_ang_obs)
            cos_dphi = ((cc - np.cos(self.phase_ang))/ss).value
            cos_dphi -= np.sign(cos_dphi)*1.e-10
            self.dlon_sol_obs = sign*np.rad2deg(np.arccos(cos_dphi))*u.deg
            # self.pos_sub_sol = (self.aspect_ang, 180*u.deg)
            # self.pos_sub_obs = (self.aspect_ang_obs, phi_obs)
            """

    def set_spin(self, spin_ecl_lon: F_OR_Q, spin_ecl_lat: F_OR_Q, rot_period: F_OR_Q):
        """ Set the spin vector
        Parameters
        ----------
        spin_ecl_lon, spin_ecl_lat : float, Quantity
            The ecliptic longitude and latitude of the spin vector (in degrees
            if `float`).

        rot_period : float, Quantity
            The rotational period of the object (in seconds if `float`)
        """
        self.spin_ecl_lon = to_quantity(spin_ecl_lon, u.deg)
        self.spin_ecl_lat = to_quantity(spin_ecl_lat, u.deg)
        self.rot_period = to_quantity(rot_period, u.s)

        try:
            self.spin_vec = lonlat2cart(
                lon=self.spin_ecl_lon, lat=self.spin_ecl_lat, r=1)
        except TypeError:
            self.spin_vec = np.array([None, None, None])

        try:
            rot_omega = 2*PI/(self.rot_period)
            self.rot_omega = rot_omega  # .to('rad/s')
        except TypeError:
            self.rot_omega = None

        try:
            self._set_aspect_angle()
        except TypeError:
            pass

    def set_mass(self, diam_eff=None, mass=None, bulk_mass_den=None):
        """ Set and solve mass-related parameters (at least two of these must be given).
        Parameters
        ----------
        diam_eff : float, Quantity, optional.
            The effective diameter (in meters if `float`). It might be
            overriden by `diam_eff` of `.set_optical()`, so be sure they are
            identical in the script.

        mass : float, Quantity, optional.
            The mass (in kg if `float`).

        bulk_mass_den : float, Quantity, optional.
            The bulk mass density (in kg/m^3 if `float`).
        """
        ps = solve_rmrho(radius=diam_eff/2, mass=mass, mass_den=bulk_mass_den)
        ps["diam_eff"] = 2*ps["radius"]
        for p in ["diam_eff", "mass", "bulk_mass_den", "acc_grav_equator"]:
            if getattr(self, p) is not None:
                try:
                    u.allclose(getattr(self, p), ps[p])
                except AssertionError:
                    warn(f"self.{p} is not None ({getattr(self, p)}), "
                         + f"and will be overridden by {ps[p]}.")

        self.diam_eff = ps["diam_eff"]
        self.radi_eff = self.diam_eff/2
        self.mass = ps["mass"]
        self.bulk_mass_den = ps["mass_den"]
        self.acc_grav_equator = GG_Q*self.mass/(self.radi_eff)**2
        self.v_esc_equator = np.sqrt(2*GG_Q*self.mass/(self.radi_eff)).si

    def set_optical(
            self,
            hmag_vis=None,
            slope_par=None,
            diam_eff=None,
            p_vis=None,
            a_bond=None,
            phase_int=None
    ):
        """ Set and solve optical phase curve related parameters
        Parameters
        ----------
        hmag_vis, slope_par : float, optional.
            The absolute magnitude and slope parameter in the V-band, following
            the IAU H-G magnitude system (Bowell et al. 1989).

        diam_eff : float, Quantity, optional.
            The effective diameter (in meters if `float`). It might be
            overriden by `diam_eff` of `.set_mass()`, so be sure they are
            identical in the script.

        p_vis : float, optional.
            The geometric albedo in the V-band.

        a_bond : float, optional.
            The Bond albedo. If not given, calculated based on the IAU H-G
            magnitude system, assuming ``A_Bond ~ A_V`` (Bowell et al. 1989 and
            Myhrvold (2018) Icarus, 303, 91 Sect 3.2).
.
        phase_int : float, optional.
            The phase integral. If not given, calculated based on the IAU H-G
            magnitude system (Bowell et al. 1989 and Myhrvold (2016) PASP, 128,
            045004 Eq (4)).
        """
        p1 = solve_pAG(p_vis=p_vis, a_bond=a_bond, slope_par=slope_par)
        p2 = solve_pDH(p_vis=p_vis, diam_eff=diam_eff, hmag_vis=hmag_vis)
        p3 = solve_Gq(slope_par=slope_par, phase_int=phase_int)

        try:
            np.testing.assert_allclose(p1["p_vis"], p2["p_vis"])
        except AssertionError:
            raise AssertionError(
                "The p_vis values obtained using the relations of [p_vis, a_bond, "
                + "slope_par] and [p_vis, diam_eff, hmag_vis] are different. "
                + "Please check the input values."
            )

        try:
            np.testing.assert_allclose(p1["slope_par"], p3["slope_par"])
        except AssertionError:
            raise AssertionError(
                "The slope_par values obtained using the relations of [p_vis, a_bond, "
                + "slope_par] and [slope_par, phase_int] are different."
                + " Please check the input values."
            )

        ps = p1.copy()
        ps.update(p2)
        ps.update(p3)

        for p in ["p_vis", "a_bond", "slope_par", "diam_eff", "hmag_vis", "phase_int"]:
            if getattr(self, p) is not None:
                try:
                    u.allclose(getattr(self, p), ps[p])
                except AssertionError:
                    warn(f"self.{p} is not None ({getattr(self, p)}), "
                         + f"and will be overridden by {ps[p]}.")

        self.p_vis = ps["p_vis"]
        self.a_bond = ps["a_bond"]
        self.slope_par = ps["slope_par"]
        self.diam_eff = ps["diam_eff"]
        self.radi_eff = self.diam_eff/2
        self.hmag_vis = ps["hmag_vis"]
        self.phase_int = ps["phase_int"]

    def minimal_set(self, thermal_par: float, aspect_ang: float, temp_eqm: float = 1):
        """ Set minimal model.

        Note
        ----
        When we just want to calculate the temperature on the asteroid, not
        thinking about the viewing geometry, there is no need to consider any
        detail about the spin. The spin direction is absorbed into the aspect
        angle. The spin period is absorbed into thermal paramter. Diameter does
        not affect the temperature, unless we are using the p-D-H relation.
        """
        self.thermal_par = to_quantity(thermal_par, NOUNIT)
        self.aspect_ang = to_quantity(aspect_ang, u.deg)
        self.temp_eqm = to_quantity(temp_eqm, u.K)
        self.temp_eqm__K = self.temp_eqm.to_value(u.K)

        # set fictitious ephemerides
        self.r_hel_vec = np.array([1, 0, 0])
        self.spin_vec = np.array([
            -np.cos(self.aspect_ang).value,
            0,
            np.sin(self.aspect_ang).value
        ])


class SmallBodyConstTPM():
    """ The TPM related thermal parameters are const over time/depth/temp.
    """

    def set_thermal(self, ti: float, emissivity: float, eta_beam: float = 1):
        """ Set thermal model related paramters.
        Parameters
        ----------
        ti : float
            The thermal inertia in SI (tiu = J/m^2/s^0.5/K).

        emissivity : float
            The spectrum-averaged hemispherical emissivity of the surface
            material (assumption is that directional emissivity is direction-
            and wavelength-independent).

        eta_beam : float, optional.
            The beaming parameter (eta) in the simple NEATM or STM, used as a
            first-order correction to the roughness effect in TPM (not useful
            for phase angle larger than ~ 40 degrees). Default is 1.0 (no
            beaming effect).
        """
        ps1 = solve_temp_eqm(temp_eqm=None,
                             a_bond=self.a_bond,
                             eta_beam=eta_beam,
                             r_hel=self.r_hel,
                             emissivity=emissivity,
                             to_value=False)
        self.emissivity = ps1["emissivity"]
        self.eta_beam = ps1["eta_beam"]
        self.temp_eqm = ps1["temp_eqm"]
        self.temp_eqm_1au = np.sqrt(self.r_hel.to_value(u.au))*self.temp_eqm
        self.temp_eqm__K = self.temp_eqm.to_value(u.K)

        ps2 = solve_thermal_par(thermal_par=None,
                                ti=ti,
                                rot_period=self.rot_period,
                                temp_eqm=self.temp_eqm,
                                emissivity=emissivity,
                                to_value=False)
        self.thermal_par = ps2["thermal_par"]
        self.ti = ps2["ti"]

    # Currently due to the spline, nlat must be > 1
    def set_tpm(self, nlon: int = 360, nlat: int = 90, Zmax: float = 10, dZ: float = 0.2,
                lats: np.ndarray = None):
        """ TPM code related parameters.
        The attributes here are all non-dimensional!

        If `lats` is not None, `nlat` will be ignored.

        Notes
        -----
        The number of slabs is defined by ``self.nZ =
        int(np.around(Zmax//dZ))``, so if `Zmax` is not an integer multiple
        of `dZ`, say ``Zmax=10`` and ``dZ=0.15``, there will be
        ``int(np.around(66.66...)) = 67`` slabs.
        """
        self.nlon = nlon
        self.dlon = 2*PI/self.nlon
        if lats is None:
            if nlat < 3:
                warn("Currently nlat < 3 is not supported. Internally I will use nlat = 3.")
                self.nlat = 3
            else:
                self.nlat = nlat
            self.dlat = PI/self.nlat
        else:
            self.lats = np.sort(np.unique(
                np.atleast_1d(to_quantity(lats, u.deg))
            ))[::-1]  # sort in descending order so that colatitude is increasing order
            self.nlat = len(self.lats)
            self.dlat = None
        self.Zmax = Zmax
        self.dZ = dZ
        self.nZ = int(np.around(Zmax/dZ))

        # Check von Neumann stability analysis (See MuellerM PhDT 2007 Sect 3.3.2):
        # For dy/dt = a d^2y/dx^2, stability requires dt <= dx^2/(2a).
        # In TPM, du/dtau = 1*d^2u/dz^2, so dtau/dz^2 <= 0.5.
        # Note that tau = omega*t = t*2pi/Prot = longitude (or rotational phase) in rad
        #           z = x/ls = depth in thermal skin depth (sqrt(k/rho c omega)) unit
        if np.any(self.dlon/(self.dZ)**2 > 0.5):
            raise ValueError(
                "dlon/dZ^2 > 0.5 !! The solution may not converge. Tune such that "
                + f"nlon > 4*PI/dZ^2 ({4*PI/self.dZ**2 = :.2f})."
            )


class SmallBody(SmallBodyMixin, SmallBodyConstTPM):
    """ Spherical Small Body class with constant thermal parameters
    Specified by physical parameters that are the body's characteristic, not
    time-variable ones (e.g., ephemerides).

    Example
    -------
    >>> test = tm.SmallBody()
    >>> test.set_ecl(
    >>>     r_hel=1.5, r_obs=0.5, alpha=0,
    >>>     hel_ecl_lon=0, hel_ecl_lat=0,
    >>>     obs_ecl_lon=0, obs_ecl_lat=0
    >>> )
    >>> test.set_spin(
    >>>     spin_ecl_lon=10, spin_ecl_lat=-90, rot_period=1
    >>> )
    >>> test.aspect_ang, test.aspect_ang_obs
    Test by varying spin vector. In all cases tested below, calculation matched
    with desired values:
        ----------input-----------  ---------desired----------
        spin_ecl_lon  spin_ecl_lat  aspect_ang  aspect_ang_obs
        0             0             180         180
        +- 30, 330    0             150         150
        0             +-30          150         150
        any value     +-90          90          90

    >>> test = tm.SmallBody()
    >>> test.set_ecl(
    >>>     r_hel=1.414, r_obs=1, alpha=45,
    >>>     hel_ecl_lon=45, hel_ecl_lat=0,
    >>>     obs_ecl_lon=90, obs_ecl_lat=0
    >>> )
    >>> test.set_spin(
    >>>     spin_ecl_lon=330, spin_ecl_lat=0, rot_period=1
    >>> )
    >>> test.aspect_ang, test.aspect_ang_obs
    Test by varying spin vector. In all cases tested below, calculation matched
    with desired values:
        ----------input-----------  ---------desired----------
        spin_ecl_lon  spin_ecl_lat  aspect_ang  aspect_ang_obs
        0             0             135         90
        + 30          0             165         120
        - 30, 330     0             105         60
        0             +-30          127.76      90
        any value     +-90          90          90

    """
    # TODO: Better to use setter and getter methods...///
    # TODO: What if a user input values with Quantity?
    #   Removing the units is not what is desired...

    def __init__(self):
        self.id = None
        # self.use_quantity = use_quantity

        # Spin related
        self.spin_ecl_lon = None
        self.spin_ecl_lat = None
        self.spin_vec = np.array([None, None, None])
        self.rot_period = None
        self.rot_omega = None

        # Epemerides related
        self.obs_ecl_lon = None
        self.obs_ecl_lat = None
        self.hel_ecl_lon = None
        self.hel_ecl_lat = None
        self.r_hel = None
        self.r_hel_vec = np.array([None, None, None])
        self.r_obs = None
        self.r_obs_vec = np.array([None, None, None])
        self.phase_ang = None

        self.aspect_ang = None
        self.dlon_sol_obs = None

        # Optical related
        self.hmag_vis = None
        self.slope_par = None
        self.phase_int = None
        self.diam_eff = None
        self.radi_eff = None
        self.p_vis = None
        self.a_bond = None

        # physics related
        self.mass = None
        self.bulk_mass_den = None
        # self.diam_equat = None
        self.acc_grav_equator = None
        self.v_esc_equator = None

        # TPM physics related
        self.ti = None
        self.thermal_par = None
        self.eta_beam = None
        self.emissivity = None
        self.temp_eqm = None
        self.temp_eqm_1au = None

        # TPM code related
        self.nlon = None
        self.nlat = None
        self.lats = None
        self.dlon = None
        self.dlat = None
        self.Zmax = None
        self.nZ = None
        self.dZ = None

        self.r_sun_disc = None
        self.mu_suns = None
        self.tempfull = None
        self.tempsurf = None
        self.tempunit = None

        self.mu_obss = None
        self.wlen_ther = None
        self.wlen_refl = None
        self.flux_ther = None
        self.flux_refl = None

    def set_ecl(self, r_hel, hel_ecl_lon, hel_ecl_lat,
                r_obs, obs_ecl_lon, obs_ecl_lat, alpha,
                ephem_equinox='J2000.0', transform_equinox='J2000.0'):
        """
        Parameters
        ----------
        r_hel, r_obs : float, Quantity
            The heliocentric and observer-centric distance to the object (in au
            if `float`).

        hel_ecl_lon, hel_ecl_lat, obs_ecl_lon, obs_ecl_lat : float, Quantity
            The heliocentric ecliptic and observer-centric ecliptic longitude
            and latitude (in degrees if `float`).

        alpha : float, Quantity
            The phase angle (Sun-target-observer angle) (in degrees if
            `float`). It is not trivial to know `alpha` from ecliptic
            longitudes and latitudes, because the observer is likely not at the
            heliocentric distance of 1 au (i.e., geocenter).

        Note
        ----
        [Not implemented yet]---
        The ``ObsEcLon`` and ``ObsEcLat`` from JPL HORIZONS are in the equinox
        of the observing time, not J2000.0. Although this difference will not
        give large uncertainty, this may be uncomfortable for some application
        purposes. In this case, give the ephemerides date (such as
        ``ephem_equinox=Time(eph['datetime_jd'], format='jd')``) and the
        equinox of ``hEcl-Lon`` and ``hEcl-Lat`` is calculated (as of 2019, it
        is J2000.0, so ``transform_equinox="J2000.0"``).
        """
        self.r_hel = to_quantity(r_hel, u.au)
        self.r_obs = to_quantity(r_obs, u.au)
        self.hel_ecl_lon = to_quantity(hel_ecl_lon, u.deg)
        self.hel_ecl_lat = to_quantity(hel_ecl_lat, u.deg)
        # The obs_ecl values from HORIZONS are observer-centered lambda, beta values.
        self.obs_ecl_lon = to_quantity(obs_ecl_lon, u.deg)
        self.obs_ecl_lat = to_quantity(obs_ecl_lat, u.deg)
        self.phase_ang = to_quantity(alpha, u.deg)

        # helecl_ref = HeliocentricMeanEcliptic(equinox=transform_equinox)
        # obsecl_geo = SkyCoord(
        #     self.obs_ecl_lon,
        #     self.obs_ecl_lat,
        #     self.r_obs,
        #     equinox=ephem_equinox
        # )
        # self.obs_ecl_helio = obsecl_geo.transform_to(helecl_ref)

        try:
            self.r_hel_vec = lonlat2cart(
                lon=self.hel_ecl_lon, lat=self.hel_ecl_lat, r=self.r_hel.value)*u.au
        except (TypeError, AttributeError):
            self.r_hel_vec = np.array([None, None, None])

        try:
            self.r_obs_vec = lonlat2cart(
                lon=self.obs_ecl_lon, lat=self.obs_ecl_lat, r=self.r_obs.value)*u.au
        except (TypeError, AttributeError):
            self.r_obs_vec = np.array([None, None, None])

        try:
            self._set_aspect_angle()
        except (TypeError, AttributeError):
            pass

    def set_musuns(self):
        self.phases = np.linspace(0, 2*PI - self.dlon, self.nlon)*u.rad
        # colats is set s.t. nlat=1 gives colat=90 deg.
        if self.lats is None:
            self.colats = np.linspace(self.dlat/2, PI - self.dlat/2, self.nlat)*u.rad
        else:
            self.colats = (90*u.deg - self.lats).to(u.rad)
        self.tpm_lats = 90 - self.colats.to_value(u.deg)
        self.tpm_lons = self.phases.to_value(u.deg)

        self.mu_suns = calc_mu_vals(
            r_vec=self.r_hel_vec,
            spin_vec=self.spin_vec,
            phases=self.phases,
            colats=self.colats,
            r_sun=self.r_sun_disc.to_value(u.deg) if self.r_sun_disc is not None else None,
            full=False
        )

    def set_muobss(self):
        self.phases = np.linspace(0, 2*PI - self.dlon, self.nlon)*u.rad
        # colats is set s.t. nlat=1 gives colat=90 deg.
        if self.lats is None:
            self.colats = np.linspace(self.dlat/2, PI - self.dlat/2, self.nlat)*u.rad
        else:
            self.colats = (90*u.deg - self.lats).to(u.rad)
        self.tpm_lats = 90 - self.colats.to_value(u.deg)
        self.tpm_lons = self.phases.to_value(u.deg)

        self.mu_obss = calc_mu_vals(
            r_vec=-1*self.r_obs_vec,
            spin_vec=self.spin_vec,
            phases=self.phases - to_quantity(self.phase_ang, u.rad),
            colats=self.colats,
            full=False
        )

    def calc_temp(
            self,
            full: bool = False,
            min_iter: int = 50,
            max_iter: int = 5000,
            u_arr_midnight=None,
            permanent_shadow_u: float = 0,
            in_kelvin: bool = False,
            retain_last_uarr: bool = False,
            atol: float = 1.e-8,
            r_sun_disc: float = None,
            skip_spl: bool = False,
            skip_musun: bool = True,
            use_surfmean: bool = False,
            verbose: bool = False
    ) -> None:
        """ Calculate the temperature using TPM

        Parameters
        ----------
        full : bool, optional.
            If `True`, the temperature beneath the surface is also saved as
            ``self.tempfull`` as well as the surface temperatue as
            ``self.tempsurf``. If `False` (default), only the surface
            tempearatue is saved as ``self.tempsurf``.

        min_iter, max_iter : int, optional
            The minimum or maxumum number of iteration for the equilibrium
            temperature calculation. Default is 50 and 5000, respectively.

        u_arr_midnight : ndarray, optional.
            The initial guess for the temperature (in the unit of T_EQM), at
            time of 0 (midnight), and shape of ``(nlat, ndepth)``. If not
            given, initializes as ``temp_eqm*e^(-depth/skin_depth)``.

        retain_last_uarr : bool, optional.
            If `True`, the last iteration's u_arr (``u_arr[:, -1, :]``, which
            is basically the duplicated midnight value, ``u_arr[:, 0, :]``,
            under perfect thermal equilibrium) is not removed, so that
            ``u_arr.shape = (nlat, ntime + 1, ndepth)``. This is useful for
            seasonal-effect calculation, where ``u_arr[:, -1, :]`` should be
            different from ``u_arr[:, 0, :]``.

        r_sun_disc : float, Quantity, optional.
            The radius of the Sun's disk (in degrees if `float`). If not given,
            the Sun is assumed to be a point source.

        skip_spl : bool, optional.
            If `True`, the spline for the temperature and mu_sun are not
            calculated. This is useful when the user only needs quick
            calculation

        skip_musun : bool, optional.
            If `True`, skips `mu_suns` calculation if possible (i.e., when
            `self.mu_suns` is not `None`). Useful for certain cases. Default is
            `True`.

        use_surfmean : bool, optional
            If `True`, the deepest temperature will be set as the mean of the
            surface temperature of the previous iteration, to try to guarantee
            the mathematical condition (time average of surface temp = deepest
            temp). Maybe useful for very accurate deep temperature calculation.

        atol : float, optional
            The absolute tolerance for the iteration to stop. (Stops if the
            T/T_EQM < `atol`). See `util.calc_uarr_tpm` for more details.
        """
        self.min_iter = min_iter
        self.max_iter = max_iter
        if self.max_iter < self.min_iter:
            raise ValueError(
                f"max_iter must be >= min_iter ({self.max_iter=} and {self.min_iter=}.")

        if verbose and self.thermal_par < 0.1:
            if self.nlon < 1000:
                print(
                    f"nlon ({self.nlon}) is too small for such thermal parameter "
                    + "({self.thermal_par:.3e}); recommended to be >= 1500."
                )

        if r_sun_disc is not None:
            self.r_sun_disc = to_quantity(r_sun_disc, u.deg)

        if not (skip_musun and self.mu_suns is not None):
            self.set_musuns()

        Zarr = np.linspace(0, self.Zmax - self.dZ, self.nZ)

        # For interpolation in lon = 360 deg - dlon to 360 deg:
        self.lons_spl = np.linspace(0, 360 + self.dlon*R2D, self.nlon + 1)
        self.colats_spl = self.colats.to_value(u.deg)

        # For interpolation in lon = 360 deg - dlon to 360 deg:
        # Append lon = 0 to the last -- for interpolation purpose!
        _mu_suns = self.mu_suns.copy()
        _mu_suns = np.append(_mu_suns, np.atleast_2d(_mu_suns[:, 0]).T, axis=1)

        # Make nlon + 1 and then remove this last element later
        u_arr = np.zeros(shape=(self.nlat, self.nlon + 1, self.nZ))

        if self.thermal_par.value < 1.e-8:
            u_arr[:, :, 0] = _mu_suns**(1/4)  # all deeper cells have T=0.
            warn(f"{self.thermal_par=:6.4e} too small: Ignore TPM, run NEATM-like calculation.")
        else:
            if u_arr_midnight is None:
                u_neatm = _mu_suns**(1/4)
                u_neatm_mean = np.mean(u_neatm, axis=1)
                print(f"u_neatm_mean = {u_neatm_mean}")
                for k, _umean in enumerate(u_neatm_mean):
                    u_arr[k, :, :] = _umean  # - 0.5*np.exp(-Zarr/np.sqrt(2))*np.cos(Zarr/np.sqrt(2))
                # u_arr[:, 0, -1] = u_neatm_mean
                # u_arr[:, :, 0] = _mu_suns**(1/4)
                # # ^ first, same as NEATM.
                # for k in range(self.nlat):
                #     u_arr[k, 0, :] = u_arr[k, self.nlon//4*3, 0]*np.exp(-Zarr)
                    # ^ midday temp * exp(-z)
                # u_arr[:, 0, -1] = np.mean(u_arr[:, 0, :], axis=1)  # FIXME: Wrong axis...
                # ^ then, deepest temp = mean (surface)
            else:
                for k in range(self.nlat):
                    u_arr[k, 0, :] = u_arr_midnight[k, :]

            if max_iter > 0:
                self.tpm_niter = calc_uarr_tpm(
                    u_arr,
                    thpar=self.thermal_par.value,
                    dlon=self.dlon,
                    dZ=self.dZ,
                    mu_suns=self.mu_suns,
                    min_iter=self.min_iter,
                    max_iter=self.max_iter,
                    permanent_shadow_u=permanent_shadow_u,
                    use_surfmean=use_surfmean,
                    atol=atol
                )

        if not skip_spl:
            try:
                self.spl_musun = RectBivariateSpline(
                    self.colats_spl, self.lons_spl, _mu_suns, kx=1, ky=1, s=0
                )
            except:  # only one colat
                self.spl_musun = interp1d(self.lons_spl, _mu_suns[0], kind="linear",
                                          bounds_error=False, fill_value="extrapolate")

            try:
                # Because there is one more "phase" value, we make spline here
                # before erasing it in the next line:
                self.spl_uarr = RectBivariateSpline(
                    self.colats_spl, self.lons_spl, u_arr[:, :, 0], kx=1, ky=1, s=0
                )
            except:  # only one colat
                self.spl_uarr = interp1d(self.lons_spl, u_arr[0, :, 0], kind="linear",
                                         bounds_error=False, fill_value="extrapolate")

        # because there is one more "phase" value, erase it:
        if not retain_last_uarr:
            u_arr = u_arr[:, :-1, :]

        self.tempunit = "K" if in_kelvin else "T_EQM"
        self.tpm_Zarr = Zarr

        if full:
            if in_kelvin:
                self.tempfull = u_arr*self.temp_eqm__K
                self.tempsurf = u_arr[:, :, 0]*self.temp_eqm__K
            else:
                self.tempfull = u_arr
                self.tempsurf = u_arr[:, :, 0]
        else:
            if in_kelvin:
                self.tempsurf = u_arr[:, :, 0]*self.temp_eqm__K
            else:
                self.tempsurf = u_arr[:, :, 0]

    def calc_flux_ther(self, wlen: float):
        """ Calculates flux in W/m^2/um
        wlen : float, Quantity
            The wavelength in microns (if `float`).
        """
        if self.mu_obss is None:
            self.set_muobss()

        self.wlen_ther = to_quantity(wlen, u.um)

        if self.tempsurf is None:
            raise ValueError("tempsurf is None. Please run .calc_temp() first.")

        fluxarr = np.zeros(self.wlen_ther.size, dtype=float)
        calc_flux_tpm(
            fluxarr,
            wlen=self.wlen_ther.value,  # in um
            tempsurf=self.tempsurf, mu_obss=self.mu_obss,
            colats=to_val(self.colats, u.rad),
            dlon=self.dlon, dlat=self.dlat
        )
        self.flux_ther = (fluxarr * FLAMU
                          * self.emissivity
                          * (to_val(self.radi_eff, u.m)**2)
                          / (to_val(self.r_obs, u.m)**2)
                          / 1.e+6)  # W/m^2/**um**
        # See, e.g., Eq. 19 of Myhrvold 2018 Icarus, 303, 91.

    def calc_flux_refl(self, wlen_min=0, wlen_max=1000, refl: F_OR_ARR = None):
        """
        wlen_min, wlen_max : float, Quantity, optional.
            The wavelength in microns (if `float`) to be used. The calculation
            will be done for wavelengths of ``wlen_min < tm.SOLAR_SPEC[:, 0] <
            wlen_max``. Default is 0 and 1000.

        refl : float, Quantity, functional optional.
            The reflectance, normalized to 1 at V-band, in linear scale. If not
            given, it is set to 1 (flat spectrum). If given as a function, it
            should be a function that accepts wavelength (in microns).

        Notes
        -----
        At the moment, this functionality is very limited.

        >>> _r = YOUR_REFLECTANCE
        >>> _w = YOUR_WAVELENGTH  # in microns, same size as _r
        >>> refl = UnivariateSpline(_w, _l, k=3, s=0, ext="const")(tm.SOLAR_SPEC[:, 0])
        """
        wlen_min = to_val(wlen_min, u.um)
        wlen_max = to_val(wlen_max, u.um)
        wlen_mask = (SOLAR_SPEC[:, 0] >= wlen_min) & (SOLAR_SPEC[:, 0] <= wlen_max)
        self.wlen_refl = SOLAR_SPEC[wlen_mask, 0]*u.um
        wlen__um = self.wlen_refl.value

        if refl is None:
            refl = np.ones(wlen__um.size)
        elif isinstance(refl, (int, float, np.ndarray)):
            refl = np.atleast_1d(refl)
        else:  # assume functional
            refl = refl(wlen__um)
            # if (refl.size != wlen__um.size):
            #     raise ValueError(
            #         "At the moment, `refl` must be given for all the wavelengths of "
            #         + f"`tm.SOLAR_SPEC`, which is length {SOLAR_SPEC.shape[0]}."
            #         + f" Now {refl.size=}. Fix it by, e.g.,\n"
            #         + " >>> _r = YOUR_REFLECTANCE\n"
            #         + " >>> _w = YOUR_WAVELENGTH  # in microns, same size as _r\n"
            #         + " >>> refl = UnivariateSpline(_w, _l, k=3, s=0, ext='const')"
            #         + "(tm.SOLAR_SPEC[:, 0])"
            #     )

        phase_factor = iau_hg_model(alphas=self.phase_ang.to_value(u.deg),
                                    gpar=self.slope_par.value)
        self.flux_refl = (SOLAR_SPEC[wlen_mask, 1]/(self.r_hel.to_value(u.au))**2
                          * refl*self.p_vis
                          * (self.radi_eff.to_value(u.m))**2
                          / (self.r_obs.to_value(u.m))**2
                          * phase_factor
                          ) * FLAMU
        # SOLAR_SPEC already is for r_hel = 1au, so for r_hel, it should use u.au.
        # See, e.g., Eq. 19 of Myhrvold 2018 Icarus, 303, 91.

    def get_temp_1d(self, colat__deg, lon__deg):
        """ Return 1d array of temperature.

        Parameters
        ----------
        colat__deg, lon__deg : float or Quantity, or array of such
            The colatitude, which is 0 at North and 180 at South, and the phase
            (longitude), which is 0 at midnight and 90 at sunrise in degrees
            unit. Note this is different from low-level cases where the default
            is radian in many cases.

        Notes
        -----
        For performance issue, I didn't put any astropy quantity here. This
        function may be used hundreds of thousands of times for each
        simulation, so 1ms is not a small time delay.
        """
        try:
            temp = self.spl_uarr(colat__deg, lon__deg)
        except AttributeError:
            raise AttributeError(
                "self.spl_uarr is not set. Run .calc_temp() with skip_spl=False."
            )
        return self.temp_eqm__K*temp.flatten()

    def get_temp_2d(self, colat__deg, lon__deg):
        """ Return 1d array of temperature.

        Parameters
        ----------
        colat__deg, lon__deg : float or Quantity, or array of such
            The colatitude, which is 0 at North and 180 at South, and the phase
            (longitude), which is 0 at midnight and 90 at sunrise in degrees
            unit. Note this is different from low-level cases where the default
            is radian in many cases.

        Notes
        -----
        For performance issue, I didn't put any astropy quantity here. This
        function may be used hundreds of thousands of times for each
        simulation, so 1ms is not a small time delay.
        """
        try:
            return self.temp_eqm__K*self.spl_uarr(colat__deg, lon__deg)
        except AttributeError:
            raise AttributeError(
                "self.spl_uarr is not set. Run .calc_temp() with skip_spl=False."
            )

    def get_musun(self, colat__deg, lon__deg):
        """ Return 1d array of temperature.

        Parameters
        ----------
        colat, phi : float or Quantity, or array of such
            The colatitude, which is 0 at North and 180 at South, and the phase
            (longitude), which is 0 at midnight and 90 at sunrise in degrees
            unit. Note this is different from low-level cases where the default
            is radian in many cases.

        Notes
        -----
        If you want 2-d array, just use ``self.spl_musun(colat, phi)``.
        For performance issue, I didn't put any astropy quantity here. This
        function may be used hundreds of thousands of times for each
        simulation, so 1ms is not a small time delay.
        """
        try:
            musun = self.spl_musun(colat__deg, lon__deg)
        except AttributeError:
            raise AttributeError(
                "self.spl_musun is not set. Run .calc_temp() with skip_spl=False."
            )
        musun[musun < 1.e-4] = 0
        # 1.e-4 corresponds to incidence angle of 89.994˚
        return musun.flatten()

    def tohdul(self, output=None, dtype='float32', **kwargs):
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=self.tempsurf.astype(dtype)),
            fits.ImageHDU(data=self.mu_suns.astype(dtype))
        ])
        hdu_0 = hdul[0]
        hdu_T = hdul[1]
        hdu_m = hdul[2]
        # names = ["T_SURF", "MU_SUN"]

        hdu_T.header["EXTNAME"] = ("T_SURF", "Extension Name")
        hdu_T.header["BUNIT"] = (self.tempunit, "Pixel unit")
        hdu_m.header["EXTNAME"] = ("MU_SUN", "Extension Name")
        hdu_m.header["BUNIT"] = ("DIMENSIONLESS", "Pixel unit")

        now = datetime.datetime.utcnow()

        # TODO: Is is better to save these to all the extensions? Or just to
        #   the Primary?
        for i, hdr in enumerate([hdu_0.header, hdu_T.header, hdu_m.header]):
            hdr["DATE-OBS"] = (str(now), "[ISO] UT time this FITS is made")

            # TPM code parameters
            hdr["RES_LON"] = (self.dlon*R2D,
                              "[deg], phase (longitude) resolution (full=360)")
            hdr["NUM_LON"] = (self.nlon,
                              "Number of phase bins (2PI/RES_LON)")
            hdr["RES_LAT"] = (self.dlat*R2D,
                              "[deg], (co-)latitude resolution")
            hdr["NUM_LAT"] = (self.nlat,
                              "Number of (co-)latitude bins")
            hdr["RES_DEP"] = (self.dZ,
                              "[thermal_skin_depth] Depth resolution")
            hdr["MAX_DEP"] = (self.Zmax,
                              "[thermal_skin_depth] Maxumum depth")
            hdr["NUM_DEP"] = (self.nZ,
                              "Number of depth bins")

            # TPM parameters
            add_hdr(hdr, "EPSILON", self.emissivity, NOUNIT,
                    "Assumed constant emissivity at thermal region")
            add_hdr(hdr, "T_EQM", self.temp_eqm, u.K,
                    "[K] Equilibrium subsolar temperature when TI=0")
            add_hdr(hdr, "T_1AU", self.temp_eqm_1au, u.K,
                    "[K] T_EQM at r_hel=1AU")
            add_hdr(hdr, "T_MAXEQM", self.tempsurf.max(), u.K,
                    "[-] Maximum surface temperature in T_EQM unit")
            add_hdr(hdr, "T_MAX", self.temp_eqm__K*self.tempsurf.max(), u.K,
                    "[K] Maximum surface temperature")
            add_hdr(hdr, "TI", self.ti, TIU,
                    "[tiu] Thermal Inertia")
            add_hdr(hdr, "THETAPAR", self.thermal_par, NOUNIT,
                    "Thermal Parameter")

            # Spin-related paramters
            add_hdr(hdr, "SPIN_LON", self.spin_ecl_lon, u.deg,
                    "[deg] Spin vector, ecliptic longitude")
            add_hdr(hdr, "SPIN_LAT", self.spin_ecl_lon, u.deg,
                    "[deg] Spin vector, ecliptic latitude")
            add_hdr(hdr, "SPIN_X", self.spin_vec[0], NOUNIT,
                    "Spin vector, ecliptic X (unit vector)")
            add_hdr(hdr, "SPIN_Y", self.spin_vec[1], NOUNIT,
                    "Spin vector, ecliptic Y (unit vector)")
            add_hdr(hdr, "SPIN_Z", self.spin_vec[2], NOUNIT,
                    "Spin vector, ecliptic Z (unit vector)")
            add_hdr(hdr, "P_ROT", self.rot_period, u.h,
                    "[h] The rotational period")
            add_hdr(hdr, "OMEGAROT", self.rot_omega, 1/u.s,
                    "[rad/s] The rotational angular frequency")
            add_hdr(hdr, "ASP_ANG", self.aspect_ang, u.deg,
                    "[deg] The aspect angle")

            # Ephemerides parameters
            add_hdr(hdr, "R_HEL", self.r_hel, u.au,
                    "[au] Heliocentric distance")
            add_hdr(hdr, "R_HEL_X", self.r_hel_vec[0], u.au,
                    "[au] Sun-target vector, ecliptic X")
            add_hdr(hdr, "R_HEL_Y", self.r_hel_vec[1], u.au,
                    "[au] Sun-target vector, ecliptic Y")
            add_hdr(hdr, "R_HEL_Z", self.r_hel_vec[2], u.au,
                    "[au] Sun-target vector, ecliptic Z")

            add_hdr(hdr, "HECL_LON", self.hel_ecl_lon, u.deg,
                    "[deg] Sun-target vector, ecliptic longitude")
            add_hdr(hdr, "HECL_LAT", self.hel_ecl_lat, u.deg,
                    "[deg] Sun-target vector, ecliptic latitude")

            add_hdr(hdr, "R_OBS", self.r_obs, u.au,
                    "[au] Geocentric(observer-target) distance")
            add_hdr(hdr, "R_OBS_X", self.r_obs_vec[0], u.au,
                    "[au] Observer-target vector, ecliptic X")
            add_hdr(hdr, "R_OBS_Y", self.r_obs_vec[1], u.au,
                    "[au] Observer-target vector, ecliptic Y")
            add_hdr(hdr, "R_OBS_Z", self.r_obs_vec[2], u.au,
                    "[au] Observer-target vector, ecliptic Z")

            add_hdr(hdr, "OECL_LON", self.obs_ecl_lon, u.deg,
                    "[deg] Observer-target vector, ecliptic longitude")
            add_hdr(hdr, "OECL_LAT", self.obs_ecl_lat, u.deg,
                    "[deg] Observer-target vector, ecliptic latitude")

            # Albedo-size-magnitude
            add_hdr(hdr, "ALB_GEOM", self.p_vis, NOUNIT,
                    "Geometric albedo in V-band")
            add_hdr(hdr, "ALB_BOND", self.a_bond, NOUNIT,
                    "Bond albedo, e.g., (0.286+0.656*SLOPEPAR)*ALB_GEOM")
            add_hdr(hdr, "SLOPEPAR", self.slope_par, NOUNIT,
                    "Slope parameter for IAU HG mag system")
            add_hdr(hdr, "ABS_MAG", self.hmag_vis, NOUNIT,
                    "[mag] Absolute magnitude in V-band")
            add_hdr(hdr, "DIAM_EFF", self.diam_eff, u.km,
                    "[km] Effective diameter (twice RADI_EFF)")
            add_hdr(hdr, "RADI_EFF", self.radi_eff, u.km,
                    "[km] Effective radius (half DIAM_EFF)")

            # WCS: image XY -> Longitude (0, 360)/Latitude (-90, +90)
            hdr["CTYPE1"] = ("LINEAR", "Coordinate unit")
            hdr["CTYPE2"] = ("LINEAR", "Coordinate unit")
            hdr["CNAME1"] = ("Latitude", "[deg] 90 = north pole")
            hdr["CNAME2"] = ("Longitude", "[deg] 180 = noon")
            hdr["CUNIT1"] = ("deg", "Coordinate unit")
            hdr["CUNIT2"] = ("deg", "Coordinate unit")
            hdr["CRPIX1"] = (1, "Pixel coordinate of reference point")
            hdr["CRPIX2"] = (1, "Pixel coordinate of reference point")
            hdr["CRVAL1"] = (self.dlon/2*R2D, "Coordinate value at reference point")
            hdr["CRVAL2"] = (-90 + self.dlat/2*R2D,
                             "Coordinate value at reference point")
            hdr["CD1_1"] = (self.dlon*R2D, "Coordinate transformation matrix element")
            hdr["CD1_2"] = (0, "Coordinate transformation matrix element")
            hdr["CD2_1"] = (0, "Coordinate transformation matrix element")
            hdr["CD2_2"] = (self.dlat*R2D, "Coordinate transformation matrix element")

            # TODO: put LON-TMAX, LAT-TMAX, LON-SS, LAT-SS, LON-SO, LAT-SO
        if output is not None:
            print(type(hdul))
            print(type(hdul[0]))
            hdul.writeto(output, **kwargs)
        return hdul

    @classmethod
    def from_eph(cls, eph_row):
        sb = cls()
        sb.set_ecl(r_hel=eph_row["r"], r_obs=eph_row["delta"], alpha=eph_row["alpha"],
                   hel_ecl_lat=eph_row["EclLat"], hel_ecl_lon=eph_row["EclLon"],
                   obs_ecl_lat=eph_row["ObsEclLat"], obs_ecl_lon=eph_row["ObsEclLon"])
        return sb


@dataclass
class OrbitingSmallBody:
    id: str | dict[str, float | str]
    spin_ecl_lon: F_OR_Q = 0
    spin_ecl_lat: F_OR_Q = 90
    location: str | dict[str, float | str] = None
    id_type: str = None
    epochs: dict | np.ndarray = None
    query_step: str = "6h"
    verbose: int = 1
    # use_quantity = True
    # ---  albedo related (should be solved)
    a_bond: F_OR_Q = None
    p_vis: F_OR_Q = None
    slope_par: F_OR_Q = 0.15
    classical: bool = False
    # ---  scaling temperature related (should be solved)
    temp_1au: F_OR_Q = None
    eta_beam: F_OR_Q = 1.
    emissivity: F_OR_Q = 1.0
    # ---  thermal_par (theta) related (should be solved)
    thermal_par_1au: F_OR_Q = None
    ti: F_OR_Q = None
    rot_period: F_OR_Q = None
    # ---  other (TPM related)
    lonlats: np.ndarray = np.array(((0, 0), (180, 0), (0, 45), (0, -45)))*u.deg
    deps: np.ndarray = field(default_factory=np.arange(0, 10.1, 0.2))
    n_per_rot: float = 360
    n_per_rot_aph: float = 360
    ignore_stability: bool = False
    # ---
    eph_interp_kind: str = "cubic"
    orb_period: F_OR_Q = None

    """
    Parameters
    ----------
    id : str or dict, required
        Name, number, or designation of target object. Uses the same codes as
        JPL Horizons. Arbitrary topocentric coordinates can be added in a dict.
        The dict has to be of the form {``'lon'``: longitude in deg (East
        positive, West negative), ``'lat'``: latitude in deg (North positive,
        South negative), ``'elevation'``: elevation in km above the reference
        ellipsoid, [``'body'``: Horizons body ID of the central body; optional;
        if this value is not provided it is assumed that this location is on
        Earth]}.

    location : str or dict, optional
        Observer's location for ephemerides queries or center body name for
        orbital element or vector queries. Uses the same codes as JPL Horizons.
        If no location is provided, Earth's center is used for ephemerides
        queries and the Sun's center for elements and vectors queries.
        Arbitrary topocentric coordinates for ephemerides queries can be
        provided in the format of a dictionary. The dictionary has to be of the
        form {``'lon'``: longitude in deg (East positive, West negative),
        ``'lat'``: latitude in deg (North positive, South negative),
        ``'elevation'``: elevation in km above the reference ellipsoid,
        [``'body'``: Horizons body ID of the central body; optional; if this
        value is not provided it is assumed that this location is on Earth]}.

    id_type : str, optional
        Controls Horizons's object selection for ``id``
        [HORIZONSDOC_SELECTION]_ .  Options: ``'designation'`` (small body
        designation), ``'name'`` (asteroid or comet name), ``'asteroid_name'``,
        ``'comet_name'``, ``'smallbody'`` (asteroid and comet search), or
        ``None`` (first search search planets, natural satellites, spacecraft,
        and special cases, and if no matches, then search small bodies).

    epochs : dict or np.ndarray
        Epochs to query. If None, automatically determine epochs to query: (1)
        query JPL SBDB and get the orbital period and the recent time of
        perihelion, (2) start time is the aphelion time (perihelion -
        orb_period / 2), (3) the auto-determined epochs is ``{"start":
        aphelion, "stop": aphelion+P_orb+1day, "step": query_step}``.

        ..warning::
            Currently, only ``epochs=None`` will work as expected.

        Either a list of epochs in JD or MJD format or a dictionary defining a
        range of times and dates; the range dictionary has to be of the form
        {``'start'``: 'YYYY-MM-DD [HH:MM:SS]', ``'stop'``: 'YYYY-MM-DD
        [HH:MM:SS]', ``'step'``: 'n[y|d|m|s]'}. Epoch timescales depend on the
        type of query performed: UTC for ephemerides queries, TDB for element
        and vector queries. If no epochs are provided, the current time is
        used.

    query_step : str, optional
        The step size for the range of epochs to be used, if `epochs` is
        `None`.

    verbose : int, optional
        Verbosity level [0-3].

    thermal_par : float or astropy.units.Quantity, optional
        The thermal parameter (theta) of the body. Note that the definition of
        it for the orbiting body is different from the one for the non-orbiting
        body. For the orbiting body, it is defined based on the
        ``self.temp_1au``, the equilibrium temperature at 1 au
        (``temp_eqm(r_hel=1)``), not at semi-major axis, or any other specific
        heliocentric distance.

    # TODO: Maybe better to use perihelion longitudes for consistency with shape models?
    lonlats : ndarray (N, 2)
        The specific point on the body, i.e., longitude and latitude on the
        body to be traced over the orbit (assumed in deg unit if plain
        ndarray). Longitude 0 and 180 (deg) are the midnight and noon positions
        at the aphelion, respectively. Default: (0, 0), (180, 0), (0, 45), (0,
        -45). (i.e., equator midnight, equator noon, mid-north midnight,
        mid-south midnight)

    deps : ndarray
        The depth grids (assumed in the unit of thermal skin depth,
        ``sqrt(kappa/rho*c*omega)``). Default: 0-10 (0.2 separation, total
        50 steps).

    nt : float
        The number of phase steps (per rotation) to calculate temperature.
        Default: 360. Recommended to be larger as thermal parameter gets smaller
        (>= 1500 at thermal parameter Theta < 0.1: DelboM 2004 PhDT).

    ignore_stability : bool
        Whether to ignore the stability check of the solution. If `False`
        (default), the code will raise an error if ``(2pi/nt)/d_dep_max^2 >
        0.5``. ``d_dep`` is the step size of `deps`.
    """

    # FIXME: let epoch!=None work properly (because they will not have `self.obj.sbdb`)
    def __post_init__(self):
        # === Initial settings
        # ---  Spin related
        self.spin_ecl_lon = to_quantity(self.spin_ecl_lon, u.deg)
        self.spin_ecl_lat = to_quantity(self.spin_ecl_lat, u.deg)
        self.spin_vec = lonlat2cart(lon=self.spin_ecl_lon, lat=self.spin_ecl_lat, r=1)

        # ---  solve albedo
        _pag = solve_pAG(
            p_vis=self.p_vis, a_bond=self.a_bond, slope_par=self.slope_par,
            classical=self.classical, to_value=False)
        self.p_vis = _pag["p_vis"]
        self.a_bond = _pag["a_bond"]
        self.slope_par = _pag["slope_par"]

        # ---  solve scaling (characteristic) temperature
        # self.sma = to_quantity(self.obj.sbdb["orbit"]["a"], u.au)
        # TODO: will the scaling using temp_eqm at 1au guarantee the numerical
        # stability at, e.g., very small or very large r_hels (when T/T_1au =
        # 10 // 0.1 level)? I think so, but... 2023-01-15 17:24:33 (KST: GMT+09:00) ysBach
        _teq = solve_temp_eqm(
            temp_eqm=self.temp_1au, a_bond=self.a_bond, eta_beam=self.eta_beam,
            r_hel=1, emissivity=self.emissivity, to_value=False
        )
        self.temp_1au = _teq["temp_eqm"]  # NOT temp_eqm, but temp_1au
        self.eta_beam = _teq["eta_beam"]
        self.emissivity = _teq["emissivity"]
        if self.a_bond != _teq["a_bond"]:
            raise ValueError(
                "a_bond is inconsistent between solve_pAG and solve_temp_eqm. "
                + "Check input parameters.")

        # ---  solve thermal_par
        _thp = solve_thermal_par(
            thermal_par=self.thermal_par_1au, ti=self.ti, rot_period=self.rot_period,
            temp_eqm=self.temp_1au, emissivity=self.emissivity, to_value=False
        )
        self.thermal_par_1au = _thp["thermal_par"]
        self.ti = _thp["ti"]
        self.rot_period = _thp["rot_period"]
        if self.temp_1au != _thp["temp_eqm"]:
            raise ValueError(
                "temp_eqm is inconsistent between solve_temp_eqm and solve_thermal_par. "
                + "Check input parameters.")
        if self.emissivity != _thp["emissivity"]:
            raise ValueError(
                "emissivity is inconsistent between solve_temp_eqm and solve_thermal_par. "
                + "Check input parameters.")

        # ---  set miscellanea
        self.rot_omega = p2w(self.rot_period)
        self.lonlats = to_quantity(np.atleast_2d(self.lonlats), u.deg)
        self.npoints = self.lonlats.shape[0]
        self.deps = to_quantity(np.atleast_1d(self.deps), NOUNIT).value
        self.n_per_rot = int(self.n_per_rot)

        # See MuellerM PhDT 2007 Sect 3.3.2:
        if not self.ignore_stability:
            _dt = 2*PI/self.n_per_rot
            _dz = np.ediff1d(self.deps).max()
            if (_dt/_dz**2) > 0.5:
                raise ValueError("(2pi/nt)/d_dep_max^2 > 0.5 (unstable)."
                                 + "To force-run, set ignore_stability=True.")

        # TODO: Check if nt is enough for the given Theta parameter. Since
        # Theta is defined by 1-au temperature, it is not straightforward at
        # the moment...

    def query(self):
        self.sbdb = None
        self.aspect_angs = []
        self.aspect_angs_obs = []
        self.dlons_sol_obs = []

        # === 1. Query from JPL HORIZONS
        if self.epochs is None:  # automatically determine epochs to query
            if self.verbose > 0:
                print("Querying JPL SBDB to auto-set epochs...", end=" ")
            self.sbdb = SBDB.query(self.id, full_precision=True)
            to = self.sbdb["orbit"]["elements"]["per"].si.value  # in sec
            self.time_peri = Time(self.sbdb["orbit"]["epoch"].value, format="jd")
            tp = self.time_peri.jd  # in JD [days]
            ta = tp - to/2/86400  # aphelion time in JD
            self.epochs = dict(
                start=Time(ta, format="jd").iso,
                stop=Time(ta+to/86400 + 1, format="jd").iso,  # add 1 more day
                step=self.query_step
            )
            if self.verbose > 1:
                print(
                    "DONE.\n"
                    f"\t  Orbital period  : {to:.2f} s = {to/86400:.4f} d \n"
                    f"\tRecent perihelion : {Time(tp, format='jd').iso} (JD {tp})\n"
                    f"\t Auto-set epochs  : {self.epochs}"
                )
            self.orb_period = (to*u.s).to(u.day)  # in days
            self.jd_aph = ta
            self.jd_per = tp
            try:
                if self.query_step[-1] == "d":
                    self.query_step = float(self.query_step[:-1])*u.day
                elif self.query_step[-1] == "h":
                    self.query_step = float(self.query_step[:-1])*u.hour
                elif self.query_step[-1] == "m":
                    self.query_step = float(self.query_step[:-1])*u.minute
                elif self.query_step[-1] == "s":
                    self.query_step = float(self.query_step[:-1])*u.second
                elif self.query_step[-1] == "y":
                    self.query_step = float(self.query_step[:-1])*u.year
                else:
                    raise ValueError("query_step not understood. "
                                     + "Currently only '<int>[d|h|m|s|y]' are supported.")
            except ValueError:
                raise ValueError("query_step not understood. "
                                 + "Currently only '<int>[d|h|m|s|y]' are supported.")
        else:
            raise ValueError("Not implemented yet.")

        obj = Horizons(
            id=self.id, location=self.location, epochs=self.epochs, id_type=self.id_type)
        if self.verbose > 0:
            print("Querying to JPL HORIZONS (Tune `query_step` if it takes too long)\n...",
                  end=" ")

        # Calculate spin obliquity
        self.eph = obj.ephemerides().to_pandas()
        obj_vec = Horizons(
            id=self.id, location=self.location, epochs=self.time_peri, id_type=self.id_type)
        _vecs = obj_vec.vectors()
        self.spin_obliquity = np.rad2deg(np.arccos(np.dot(
            np.cross([_vecs["x"][0], _vecs["y"][0], _vecs["z"][0]],
                     [_vecs["vx"][0], _vecs["vy"][0], _vecs["vz"][0]]),
            self.spin_vec
        )))*u.deg

        if self.verbose > 0:
            print("DONE.")

        # === 2. Calculate aspect angle
        if self.verbose > 0:
            print("Calculating angles and interpolations\n... ", end="")

        self.eph_r_hel_vecs = lonlat2cart(
            self.eph["EclLon"].to_numpy(),
            self.eph["EclLat"].to_numpy(),
            r=self.eph.r.to_numpy(),
            degree=True
        ).T*u.au
        self.eph_r_obs_vecs = lonlat2cart(
            self.eph["ObsEclLon"].to_numpy(),
            self.eph["ObsEclLat"].to_numpy(),
            r=self.eph.delta.to_numpy(),
            degree=True
        ).T*u.au

        # TODO: no need to save aspect_angs, etc into self. Just use splines
        for _xyz, _xyz_o, _alp in zip(self.eph_r_hel_vecs, self.eph_r_obs_vecs, self.eph["alpha"]):
            _asp, _asp_obs, _dlon = calc_aspect_ang(spin_vec=self.spin_vec, r_hel_vec=_xyz,
                                                    r_obs_vec=_xyz_o, phase_ang=_alp*u.deg)
            self.aspect_angs.append(_asp.value)
            self.aspect_angs_obs.append(_asp_obs.value)
            self.dlons_sol_obs.append(_dlon.value)
        self.aspect_angs = np.array(self.aspect_angs)*u.deg
        self.aspect_angs_obs = np.array(self.aspect_angs_obs)*u.deg
        self.dlons_sol_obs = np.array(self.dlons_sol_obs)*u.deg
        self.phase_angs = self.eph["alpha"].to_numpy()*u.deg*np.sign(self.dlons_sol_obs)

        # === 3. Interpolations of ephemerides
        self.eph_jds = self.eph["datetime_jd"].to_numpy()

        def _spl_interp(x):
            return interp1d(self.eph_jds, x, kind=self.eph_interp_kind,
                            fill_value="extrapolate")

        self._spl_r_hel_x = _spl_interp(self.eph_r_hel_vecs[:, 0])
        self._spl_r_hel_y = _spl_interp(self.eph_r_hel_vecs[:, 1])
        self._spl_r_hel_z = _spl_interp(self.eph_r_hel_vecs[:, 2])
        self._spl_r_obs_x = _spl_interp(self.eph_r_obs_vecs[:, 0])
        self._spl_r_obs_y = _spl_interp(self.eph_r_obs_vecs[:, 1])
        self._spl_r_obs_z = _spl_interp(self.eph_r_obs_vecs[:, 2])
        self._spl_aspect_ang = _spl_interp(self.aspect_angs)
        self._spl_aspect_ang_obs = _spl_interp(self.aspect_angs_obs)
        self._spl_phase_ang = _spl_interp(self.eph["alpha"].to_numpy())
        self._spl_r_hel = _spl_interp(self.eph["r"].to_numpy())
        self._spl_r_obs = _spl_interp(self.eph["delta"].to_numpy())

        self.eph_true_anom = self.eph["true_anom"].to_numpy()
        self.eph_true_anom_aph = (self.eph_true_anom + 180) % 360 - 180
        _len = len(self.eph_true_anom)//10
        self.eph_true_anom_aph[:_len] = (self.eph_true_anom[:_len] + 360) % 360 - 360
        # This is needed to make e.g., 179.99 deg -> -0.01 deg
        self.eph_true_anom_aph[-_len:] = (self.eph_true_anom[-_len:] - 360) % 360
        # This is needed to make e.g., 180.01 deg -> 180.01 deg

        self._spl_true_anom = _spl_interp(self.eph_true_anom)
        self._spl_true_anom_aph = _spl_interp(self.eph_true_anom_aph)
        # The time for calculating self._spl_xxx(jd) is only ~ 10-100 us order.

        if self.verbose > 0:
            print("DONE.")

    def r_hel_vec(self, jd, xyz=None, to_value=False):
        """Interpolate heliocentric position vector."""
        if xyz is None:
            return np.array([
                self._spl_r_hel_x(jd), self._spl_r_hel_y(jd), self._spl_r_hel_z(jd)
            ]).T*(1 if to_value else u.au)
        elif xyz == "x":
            return self._spl_r_hel_x(jd)*(1 if to_value else u.au)
        elif xyz == "y":
            return self._spl_r_hel_y(jd)*(1 if to_value else u.au)
        elif xyz == "z":
            return self._spl_r_hel_z(jd)*(1 if to_value else u.au)
        else:
            raise ValueError("xyz must be 'x', 'y', 'z' or None.")

    def r_obs_vec(self, jd, xyz=None, to_value=False):
        """Interpolate observer-centric position vector."""
        if xyz is None:
            return np.array([
                self._spl_r_obs_x(jd), self._spl_r_obs_y(jd), self._spl_r_obs_z(jd)
            ]).T*(1 if to_value else u.au)
        elif xyz == "x":
            return self._spl_r_obs_x(jd)*(1 if to_value else u.au)
        elif xyz == "y":
            return self._spl_r_obs_y(jd)*(1 if to_value else u.au)
        elif xyz == "z":
            return self._spl_r_obs_z(jd)*(1 if to_value else u.au)
        else:
            raise ValueError("xyz must be 'x', 'y', 'z' or None.")

    def r_hel(self, jd, to_value=False):
        return self._spl_r_hel(jd)*(1 if to_value else u.au)

    def r_obs(self, jd, to_value=False):
        return self._spl_r_obs(jd)*(1 if to_value else u.au)

    def aspect_ang(self, jd, to_value=False):
        return self._spl_aspect_ang(jd)*(1 if to_value else u.deg)

    def aspect_ang_obs(self, jd, to_value=False):
        return self._spl_aspect_ang_obs(jd)*(1 if to_value else u.deg)

    def true_anom(self, jd, to_value=False):
        return self._spl_true_anom(jd)*(1 if to_value else u.deg)

    def true_anom_aph(self, jd, to_value=False):
        """True anomaly, but set to 0 at aphelion, so easier to interpolate.
        """
        return self._spl_true_anom_aph(jd)*(1 if to_value else u.deg)

    def calc_temp(self):
        """Calculate the temperature profile."""
        # === Calculate the non-seasonal temperature profile at the aphelion
        if self.verbose > 0:
            print("aph")

        def _set_sb(sb):
            sb.set_spin(
                spin_ecl_lon=self.spin_ecl_lon,
                spin_ecl_lat=self.spin_ecl_lat,
                rot_period=self.rot_period
            )
            sb.set_thermal(
                ti=self.ti,
                emissivity=self.emissivity,
                eta_beam=self.eta_beam
            )
            sb.set_tpm(
                nlon=self.n_per_rot_aph,
                lats=self.lonlats[:, 1],
                Zmax=self.deps.max(),
                dZ=np.ediff1d(self.deps).min()
            )

        self.sb_aph = SmallBody()
        self.sb_aph.a_bond = self.a_bond
        self.sb_aph.set_ecl(
            r_hel=np.linalg.norm(self.r_hel(self.jd_aph)),  # aphelion
            hel_ecl_lat=self.eph["EclLat"][0],
            hel_ecl_lon=self.eph["EclLon"][0],
            r_obs=None,  # actually a dummy value...
            obs_ecl_lon=None,
            obs_ecl_lat=None,
            alpha=None
        )
        _set_sb(self.sb_aph)
        self.sb_aph.calc_temp(full=True, in_kelvin=True, permanent_shadow_u=0.1)
        # Note: in KELVIN!               ^^^^^^^^^^^^^^

        if self.verbose > 0:
            print("interpolating...")

        self.spl_temp_aph = RegularGridInterpolator(
            (self.sb_aph.tpm_lats, self.sb_aph.tpm_lons, self.sb_aph.tpm_Zarr),
            self.sb_aph.tempfull,
            method="linear",
            bounds_error=False,
            fill_value=None  # extrapolate
        )
        if self.verbose > 0:
            print("aph done")

        # TODO: del self.sb_aph ?

        # === seasonal calculation
        self.varrs_aph = np.array(
            [self.spl_temp_aph((lat, lon, self.deps))/self.temp_1au.value
             for lon, lat in self.lonlats]
        )

        # self.varrs_aph = []
        # for _i, (lon, lat) in enumerate(self.lonlats):
        #     assert np.isclose(lat.value, self.sb_aph.tpm_lats[_i])
        #     lon = lon.to_value(u.deg)
        #     lonidx = int(lon//(R2D*self.sb_aph.dlon))
        #     temp = self.sb_aph.tempfull[_i, lonidx, :]
        #     # If need to interpolate along longitude
        #     if lon % self.sb_aph.dlon > self.sb_aph.dlon/10:
        #         _t2 = self.sb_aph.tempfull[_i, (lonidx + 1) % self.sb_aph.tpm_nlon, :]
        #         temp = temp + (_t2 - temp)/self.sb_aph.dlon*(lon - lonidx*self.sb_aph.dlon)
        #     # If NO NEED to interpolate along depth
        #     if (np.ediff1d(self.sb_aph.tpm_Zarr)).max() - self.sb_aph.dZ/20 < 0:
        #         self.varrs_aph.append(temp/self.temp_1au.value)
        #         continue
        #     # If need to interpolate along depth
        #     _spl = interp1d(
        #         self.sb_aph.tpm_Zarr,
        #         temp,
        #         kind="linear",
        #         bounds_error=False,
        #         fill_value=None  # extrapolate
        #     )
        #     self.varrs_aph.append(_spl(self.deps)/self.temp_1au.value)
        # self.varrs_aph = np.array(self.varrs_aph)

        _mat_bf2ss = np.array(
            [mat_bf2ss(colat__deg=90 - lat) for lat in to_val(self.lonlats[:, 1], u.deg)]
        )
        self.dt_1 = self.rot_period.to_value(u.day)/self.n_per_rot
        calc_jds = np.arange(self.eph_jds[0], self.eph_jds[-1] + self.dt_1, self.dt_1)
        self.logged = {"jd": calc_jds[::self.n_per_rot*10],
                       "varr": [self.varrs_aph.copy()], }
        self.varrs_surf = np.zeros((self.lonlats.shape[0], calc_jds.size))
        self.mu_vals = np.zeros((self.lonlats.shape[0], calc_jds.size))

        nrot_per_calc = 50
        calc_size = self.n_per_rot*nrot_per_calc
        dpsi = 2*np.pi/self.n_per_rot
        psis__rad = np.arange(0, 2*np.pi*nrot_per_calc + dpsi, dpsi)
        _varrs = self.varrs_aph.copy()
        _phi0 = self.lonlats[:, 0].to_value(u.rad)

        for i_calc in range(len(calc_jds)//calc_size+1):
            i1 = max(i_calc*calc_size - 1, 0)  # from i_calc == 1, I need the previous value
            i2 = (i_calc+1)*calc_size
            calc_jds_i = calc_jds[i1:i2]
            r_hel_vecs = self.r_hel_vec(calc_jds_i, to_value=True)
            self.varrs_surf[:, i1:i2], _varrs, _phi0, self.mu_vals[:, i1:i2] = calc_varr_orbit(
                varrs_init=_varrs,
                phi0s=_phi0,
                spin_vec_norm=self.spin_vec,
                mat_bf2ss=_mat_bf2ss,
                r_hel_vecs=r_hel_vecs,
                r_hels=np.linalg.norm(r_hel_vecs, axis=1),
                true_anoms__deg=self.true_anom_aph(calc_jds_i, to_value=True),
                psis__rad=psis__rad,
                thpar1=self.thermal_par_1au,
                deps=self.deps
            )
            # print(f"{i_calc}, {self.r_hel(calc_jds_i[0]):.2f}, {self.true_anom_aph(calc_jds_i[0]):.2f}, {_phi0}, {self.mu_vals[:, i1]}")
            self.logged["varr"].append(_varrs.copy())
        self.varrs_surf = self.varrs_surf[:, :-1]
        self.mu_vals = self.mu_vals[:, :-1]
        for k, v in self.logged.items():
            self.logged[k] = np.array(v)

        # r_hel_vecs_norm = np.linalg.norm(self.eph_r_hel_vecs, axis=1)
        # nrot = (eph_jds[-1] - ephem_jds[0])/rot_period__day + 1
        # muarrs = np.empty((self.lonlats.shape[0], self.n_per_rot, ))

        # self.muarrs = calc_varr_orbit(
        #     self.varrs_aph,
        #     self.eph_jds,
        #     r_vecs__au=self.eph_r_hel_vecs,
        #     spin_vec=self.spin_vec,
        #     true_anoms__deg=self.eph_true_anom_aph,
        #     lons__deg=self.lonlats[:, 0],
        #     lats__deg=self.lonlats[:, 1],
        #     rot_period__day=self.rot_period.to(u.day).value,
        #     n_per_rot=self.n_per_rot,
        # )

        # self.temp[jd] = np.array([
        #     self.varrs_aph[i]*self.spl_temp_aph((lat, lon, self.deps))
        #     for i, (lat, lon) in enumerate(self.lonlats)
        # ])

        # # Make nlon + 1 and then remove this last element later
        # phases = np.arange(0, 2*np.pi, self.dphase.to(u.rad).value)

        # u_arr = np.zeros(
        #     shape=(self.tpm.lats.size, self.tpm.lons.size + 1, self.tpm.deps.size))

        # for k in range(self.nlat):
        #     u_arr[k, 0, :] = np.exp(-self.tpm.deps)

        # self.mu_suns = calc_mu_vals(
        #     r_vec=self.r_hel_vecs[0],
        #     spin_vec=self.spin_vec,
        #     phases=phases,
        #     colats=colats,
        #     full=False
        # )

        # time_0 = starting_time_of_simulation

        # for time in self.eph.eph_time:
        #     xyz_hel = self.interp_r_hel(time)  # [au]
        #     for lat in self.lats:
        #         pass
