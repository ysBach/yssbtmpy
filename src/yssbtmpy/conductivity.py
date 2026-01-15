"""
Conductivity modeling

TODO: How to implement low/high bounds of, e.g., k_gb13?
    As ME22, Monte Carlo is required...
"""
from dataclasses import dataclass, field
from astropy import units as u
from astropy.units import Quantity
from .constants import SIGMA_SB_Q, NOUNIT, PI, TIU, HCU, MDU, TCU
from .util import F_OR_Q_OR_ARR, F_OR_Q, to_quantity
from scipy.optimize import root_scalar
import numpy as np

__all__ = ["Material", "solve_r_grain_ME22", "k_gb13", "k_solid_gb13", "k_rad_gb13"]


def power_fun(x: F_OR_Q, coeffs: dict = None) -> F_OR_Q:
    """ Similar to np.polyval, but includes arbitrary power.

    Parameters
    ----------
    x : float
        The value of the variable.

    coeffs : dict
        The coefficients of the dependence. The keys are the exponents of `x`
        and the values are the coefficients. For example, for `a*x**2 + b*x +
        c/x`, then `coeffs` should be `coeffs = {2: a, 1: b, -1: c}`.
    """
    if isinstance(x, Quantity):
        x = x.value
    return x if coeffs is None else sum([coeffs[exp]*x**exp for exp in coeffs])


def _mat_pm(name, cen, sd, unit, coeffs=None, fun=True):
    """ Convenience function for Materials class
    """
    unit = 1 if unit is NOUNIT else unit
    if fun:
        return {
            f"{name}_fun": lambda _: power_fun(cen, coeffs)*unit,
            f"{name}_fun_lo": lambda _: power_fun(cen - sd, coeffs)*unit,
            f"{name}_fun_hi": lambda _: power_fun(cen + sd, coeffs)*unit
        }
    return {
        f"{name}": cen*unit,
        f"{name}_lo": (cen-sd)*unit,
        f"{name}_hi": (cen+sd)*unit
    }


def _mat_range(name, lo, hi, unit=NOUNIT, coeffs=None, fun=True):
    """ Convenience function for Materials class
    """
    unit = 1 if unit is NOUNIT else unit
    if fun:
        return {
            f"{name}_fun": lambda _: power_fun(np.mean((lo, hi)), coeffs)*unit,
            f"{name}_fun_lo": lambda _: power_fun(lo, coeffs)*unit,
            f"{name}_fun_hi": lambda _: power_fun(hi, coeffs)*unit
        }
    return {
        f"{name}": np.mean((lo, hi))*unit,
        f"{name}_lo": lo*unit,
        f"{name}_hi": hi*unit
    }


@dataclass
class Material:
    spec: str = None
    model: str = "ME22"
    k_grain_fun: callable = field(default=None, repr=False)
    k_grain_fun_lo: callable = field(default=None, repr=False)
    k_grain_fun_hi: callable = field(default=None, repr=False)
    rho_grain_fun: callable = field(default=None, repr=False)
    rho_grain_fun_lo: callable = field(default=None, repr=False)
    rho_grain_fun_hi: callable = field(default=None, repr=False)
    cs_fun: callable = field(default=None, repr=False)
    cs_fun_lo: callable = field(default=None, repr=False)
    cs_fun_hi: callable = field(default=None, repr=False)
    emissivity_fun: callable = field(default=None, repr=False)
    ymod: F_OR_Q = None
    ymod_lo: F_OR_Q = None
    ymod_hi: F_OR_Q = None
    prat: F_OR_Q = None
    prat_lo: F_OR_Q = None
    prat_hi: F_OR_Q = None
    f1: float = 0.0518
    f2: float = 5.26
    chi: float = 0.41
    e1: float = 1.34
    xi: float = 0.4
    gamma_coeffs: dict = {1: 6.67e-5}

    """
    all _fun arguments should take a temperature in Kelvin. Many of them are just
    a constant function regardless of the input temperature.
    """

    @property
    def f1(self):
        return self._f1

    @f1.setter
    def f1(self, value):
        self._f1 = float(value)

    @property
    def f2(self):
        return self._f2

    @f2.setter
    def f2(self, value):
        self._f2 = float(value)

    @property
    def chi(self):
        return self._chi

    @chi.setter
    def chi(self, value):
        self._chi = float(value)

    @property
    def e1(self):
        return self._e1

    @e1.setter
    def e1(self, value):
        self._e1 = float(value)

    @property
    def xi(self):
        return self._xi

    @chi.setter
    def xi(self, value):
        self._xi = float(value)

    @property
    def gamma_coeffs(self):
        return self._gamma_coeffs

    @gamma_coeffs.setter
    def gamma_coeffs(self, value):
        self._gamma_coeffs = value

    def __copy__(self):
        return Material(**self.__dict__)

    @classmethod
    def from_ME22_GB13(cls, spec: str):
        """ Initialize a `Material` instance from the ME22 database.

        Parameters
        ----------
        spec : str
            The species name. One of 'S', 'V', 'E', 'M', 'Met', 'P', 'C', 'Ch',
            'B', 'K', 'simple'. Case-insensitive.

        Returns
        -------
        Material
            The initialized Material object.
        """
        _consts = dict(f1=0.0518, f2=5.26, chi=0.41, e1=1.34, xi=0.4,
                       gamma_coeffs={1: 6.67e-5}
                       )
        if spec.lower() == "s":
            return cls(
                spec="S",
                **_mat_pm("k_grain", 4.05, 0.35, TCU),
                **_mat_range("rho_grain", 3180, 3710, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 28.8, 2.4, u.GPa, fun=False),
                **_mat_pm("prat", 0.23, 0.04, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "v":
            return cls(
                spec="V",
                **_mat_pm("k_grain", 4.05, 0.35, TCU),
                **_mat_range("rho_grain", 3180, 3440, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 28.8, 2.4, u.GPa, fun=False),
                **_mat_pm("prat", 0.23, 0.04, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "e":
            return cls(
                spec="E",
                k_grain_fun=lambda temp: power_fun(temp, {0: 4.28, -1: 258})*TCU,
                k_grain_fun_lo=lambda temp: 0.9*power_fun(temp, {0: 4.28, -1: 258})*HCU,
                k_grain_fun_hi=lambda temp: 1.1*power_fun(temp, {0: 4.28, -1: 258})*HCU,
                **_mat_pm("rho_grain", 3150, 20, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 28.8, 2.4, u.GPa, fun=False),
                **_mat_pm("prat", 0.23, 0.04, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "m":
            return cls(
                spec="M",
                k_grain_fun=lambda temp: power_fun(temp, {0: 4.76, -1: 287})*TCU,
                k_grain_fun_lo=lambda temp: 0.9*power_fun(temp, {0: 4.76, -1: 287})*TCU,
                k_grain_fun_hi=lambda temp: 1.1*power_fun(temp, {0: 4.76, -1: 287})*TCU,
                **_mat_pm("rho_grain", 3635, 35, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 28.8, 2.4, u.GPa, fun=False),
                **_mat_pm("prat", 0.23, 0.04, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "met":
            return cls(
                spec="Met",
                k_grain_fun=lambda temp: power_fun(temp, {0: 12.4, 1: 0.05})*TCU,
                k_grain_fun_lo=lambda temp: 0.9*power_fun(temp, {0: 12.4, 1: 0.05})*TCU,
                k_grain_fun_hi=lambda temp: 1.1*power_fun(temp, {0: 12.4, 1: 0.05})*TCU,
                **_mat_pm("rho_grain", 7500, 200, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0042, 1: 2.77})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0042, 1: 2.77})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0042, 1: 2.77})*HCU,
                emissivity_fun=lambda _: 0.66,
                **_mat_range("ymod", 169, 209, u.GPa, fun=False),
                **_mat_range("prat", 0.27, 0.37, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "p":
            return cls(
                spec="P",
                **_mat_pm("k_grain", 1.5, 0.5, TCU),
                **_mat_pm("rho_grain", 2420, 40, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 28.8, 2.4, u.GPa, fun=False),
                **_mat_pm("prat", 0.23, 0.04, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "c":
            return cls(
                spec="C",
                **_mat_pm("k_grain", 4.05, 0.35, TCU),
                **_mat_pm("rho_grain", 3520, 130, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 18.9, 3.7, u.GPa, fun=False),
                **_mat_pm("prat", 0.14, 0.06, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "ch":
            return cls(
                spec="Ch",
                **_mat_pm("k_grain", 1.5, 0.5, TCU),
                **_mat_pm("rho_grain", 2940, 40, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0036, 1: 3.84})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0036, 1: 3.84})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0036, 1: 3.84})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 18.9, 3.7, u.GPa, fun=False),
                **_mat_pm("prat", 0.14, 0.06, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "b":
            return cls(
                spec="B",
                **_mat_pm("k_grain", 1.5, 0.5, TCU),
                **_mat_pm("rho_grain", 2940, 40, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0036, 1: 3.84})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0036, 1: 3.84})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0036, 1: 3.84})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 18.9, 3.7, u.GPa, fun=False),
                **_mat_pm("prat", 0.14, 0.06, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "k":
            return cls(
                spec="K",
                **_mat_pm("k_grain", 4.05, 0.35, TCU),
                **_mat_pm("rho_grain", 3520, 60, MDU),
                cs_fun=lambda temp: power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_lo=lambda temp: 0.9*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                cs_fun_hi=lambda temp: 1.1*power_fun(temp, {2: -0.0033, 1: 3.39})*HCU,
                emissivity_fun=lambda _: 0.9,
                **_mat_pm("ymod", 18.9, 3.7, u.GPa, fun=False),
                **_mat_pm("prat", 0.14, 0.06, NOUNIT, fun=False),
                **_consts
            )
        elif spec.lower() == "simple":
            return cls(
                spec="simple",
                **_mat_pm("k_grain", 2.18, 0., TCU),
                **_mat_pm("rho_grain", 3700, 0, MDU),
                cs_fun=lambda temp: power_fun(temp, {0: 500})*HCU,
                emissivity_fun=lambda _: 1,
                **_mat_pm("ymod", 78, 0, u.GPa, fun=False),
                **_mat_pm("prat", 0.25, 0.0, NOUNIT, fun=False),
                **_consts
            )
        else:
            raise ValueError(f"Unknown material spec: {spec}")

    def jkr_contact(self, r_grain, temp):
        return jkr_contact(r=r_grain, temp=temp, prat=self.prat, ymod=self.ymod,
                           gamma_coeffs=self.gamma_coeffs)

    def k_grain(self, temp):
        return (self.k_grain_fun(temp),
                self.k_grain_fun_lo(temp), self.k_grain_fun_hi(temp))

    def rho_grain(self, temp):
        return (self.rho_grain_fun(temp),
                self.rho_grain_fun_lo(temp), self.rho_grain_fun_hi(temp))

    def cs_grain(self, temp):
        return (self.cs_fun(temp),
                self.cs_fun_lo(temp), self.cs_fun_hi(temp))

    def emissivity(self, temp):
        return self.emissivity_fun(temp)

    def k_ti(self, ti, temp, porosity):
        ti = to_quantity(ti, TIU)
        temp = to_quantity(temp, u.K)
        return (ti**2/(self.rho_grain_fun(temp)*(1-porosity)*self.cs_fun(temp))).to(TCU)

    def k_eff(
        self,
        r_grain: F_OR_Q_OR_ARR,
        temp: F_OR_Q_OR_ARR,
        porosity: F_OR_Q_OR_ARR = 0
    ) -> F_OR_Q_OR_ARR:
        r_grain = to_quantity(r_grain, u.um)
        temp = to_quantity(temp, u.K)
        r_c = self.jkr_contact(r_grain=r_grain, temp=temp)
        k_grain = self.k_grain(temp)[0]
        magic_factor = self.f1*np.exp(self.f2*(1 - porosity))*self.chi
        k_solid = (k_grain*(r_c/r_grain)*magic_factor).to(TCU)

        k2 = 8*self.emissivity(temp)*SIGMA_SB_Q*self.e1*porosity/(1 - porosity)*r_grain
        k_rad = (k2*temp**3).to(u.W/u.m/u.K)
        # print(f"{r_c=}\n{magic_factor=}")
        # print(f"{r_grain=}\n{k_solid=}\n{k_rad=}\n{k2.si=}\n{temp=}\n{porosity=}\n")
        return (k_solid + k_rad, k_solid, k_rad)

    def solve_r_grain(self, temp, ti, porosity=0):
        """ Finds the r_grain solution using GB13 algorithm
        solve_r_grain(ti=200, temp=200, porosity=0.5) takes
        6.8 ms ± 269 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        on MBP 14" [2021, macOS 13.1, M1Pro(6P+2E/G16c/N16c/32G)]

        returned:
                r_lo  r_hi
        poro0 a[0, 0] a[0, 1]
        poro1 a[1, 0] a[1, 1]
        poro2 a[2, 0] a[2, 1]
        ....
        """
        temps = np.atleast_1d(temp)
        tis = np.atleast_1d(ti)
        poros = np.atleast_1d(porosity)

        if (temps.size > 1) + (tis.size > 1) + (poros.size > 1) > 1:
            raise ValueError("Only one of temp, ti, porosity can be an array")

        def _calc_size(temp, ti, porosity):
            def _diff(x):  # return scalar
                _k_eff = self.k_eff(x, temp=temp, porosity=porosity)[0]
                _k_ti = self.k_ti(ti=ti, temp=temp, porosity=porosity)
                return _k_eff.value - _k_ti.value

            sol_lo = root_scalar(_diff, x0=1.e-3, x1=2.e-3)  # first two guesses = 1 & 2 nm
            sol_hi = root_scalar(_diff, x0=2.e+6, x1=1.e+6)  # first two guesses = 2 & 1 m
            r_grain_lo = sol_lo.root*u.um if sol_lo.converged else None
            r_grain_hi = sol_hi.root*u.um if sol_hi.converged else None
            return r_grain_lo, r_grain_hi

        r_lo = []
        r_hi = []

        for temp in temps:
            for ti in tis:
                for porosity in poros:
                    r_grain_lo, r_grain_hi = _calc_size(temp, ti, porosity)
                    r_lo.append(r_grain_lo)
                    r_hi.append(r_grain_hi)

        return r_lo, r_hi


def solve_r_grain_ME22(
    material: Material,
    ti: F_OR_Q,
    temp: F_OR_Q,
    porosity: F_OR_Q = 0,
    f1: float = None,
    f2: float = None,
    chi: float = None,
    e1: float = None,
    xi: float = None,
    gamma_coeffs: dict = None,
    k_grain_fun: callable = None,
    rho_grain_fun: callable = None,
    cs_fun: callable = None,
    emissivity_fun: callable = None,
):
    """Convenience function to do Material.solve_r_grain in one-line
    """
    from copy import deepcopy
    material = deepcopy(material)

    if f1 is not None:
        material.f1 = f1

    if f2 is not None:
        material.f2 = f2

    if chi is not None:
        material.chi = chi

    if e1 is not None:
        material.e1 = e1

    if xi is not None:
        material.xi = xi

    if gamma_coeffs is not None:
        material.gamma_coeffs = gamma_coeffs

    if k_grain_fun is not None:
        material.k_grain_fun = k_grain_fun

    if rho_grain_fun is not None:
        material.rho_grain_fun = rho_grain_fun

    if cs_fun is not None:
        material.cs_fun = cs_fun

    if emissivity_fun is not None:
        material.emissivity_fun = emissivity_fun

    return material, material.solve_r_grain(temp=temp, ti=ti, porosity=porosity)


def jkr_contact(
    r: F_OR_Q_OR_ARR,
    temp: F_OR_Q,
    prat: F_OR_Q,
    ymod: F_OR_Q,
    gamma_coeffs: dict = {1: 6.67e-5}
) -> F_OR_Q_OR_ARR:
    """ The contact radius from Johnson-Kendall-Roberts theory.

    Parameters
    ----------
    r : float
        The radius of the sphere. Assumed to be in micrometer if `float`.

    temp : float or Quantity
        The temperature of the sphere.

    prat, ymod : float or Quantity
        The Poisson's ratio [dimensionless] and Young's modulus [GPa] of the
        sphere. `young` is assumed to be in GPa if `float`.

    gamma_coeffs : dict
        The coefficients of the temperature dependence for the specific surface
        energy of the sphere (in the unit of J/m^2). The keys are the exponents
        of the temperature and the values are the coefficients. For example, if
        the temperature dependence is `a*T**2 + b*T + c/T`, then `coeffs`
        should be `coeffs = {2: a, 1: b, -1: c}`. Default is `coeffs = {1:
        6.67e-5}`.

    Returns
    -------
    r_contact : Quantity
        The contact radius of the sphere in micrometer.
    """
    r = to_quantity(r, u.um)
    temp = to_quantity(temp, u.K)
    prat = to_quantity(prat, NOUNIT)
    ymod = to_quantity(ymod, u.GPa)
    spsurferg = power_fun(temp, gamma_coeffs)*u.J/u.m**2
    # print(f"{spsurferg=}")
    return (((9*PI/4)*((1-prat**2)/ymod)*spsurferg*r**2)**(1/3)).to(u.um)


def k_solid_gb13(
    r_grain: F_OR_Q_OR_ARR,
    cond_grain: F_OR_Q,
    temp: F_OR_Q,
    porosity: float,
    prat: F_OR_Q,
    ymod: F_OR_Q,
    gamma_coeffs: dict = {1: 6.67e-5},
    f1: float = 0.0518,
    f2: float = 5.26,
    chi: float = 0.41
) -> F_OR_Q_OR_ARR:
    """ The solid thermal conductivity using GB13 (Gundlach & Blum 2013).

    Parameters
    ----------
    r_grain, cond_grain : float or Quantity
        The radius and thermal conductivity of the sphere. Assumed as
        micrometer and W/m/K if `float`.

    temp : float or Quantity
        The temperature of the sphere. Assumed to be in Kelvin if `float`.

    porosity : float
        The porosity of the regolith [dimensionless].

    prat, ymod : float or Quantity
        The Poisson's ratio [dimensionless] and Young's modulus [GPa] of the
        sphere. `young` is assumed to be in GPa if `float`.

    gamma_coeffs : function
        The coefficients of the temperature dependence for the specific surface
        energy of the sphere (in the unit of J/m^2). The keys are the exponents
        of the temperature and the values are the coefficients. For example, if
        the temperature dependence is `a*T**2 + b*T + c/T`, then `coeffs`
        should be `coeffs = {2: a, 1: b, -1: c}`. Default is `coeffs = {1:
        6.67e-5}`.

    f1, f2, chi : float
        The model parameters (`f1` and `f2` calibrated by SC, FCC, and BCC
        (Chem & Tien 1973) and `xi` is a magic number found to make this model
        work for Apollo 11/12 samples (Gundlach & Blum 2013)).
        From GB13, f1 = 0.0518 +- 0.0345, f2 = 5.26 +- 0.94, and chi = 0.41 +-
        0.02 (stat) +- 0.10 (syst).
    """
    r_grain = to_quantity(r_grain, u.um)
    r_c = jkr_contact(r=r_grain, temp=temp, prat=prat, ymod=ymod, gamma_coeffs=gamma_coeffs)
    cond_grain = to_quantity(cond_grain, u.W/u.m/u.K)
    magic_factor = f1*np.exp(f2*(1 - porosity))*chi
    return (magic_factor*cond_grain*(r_c/r_grain)).to(u.W/u.m/u.K)


def k_rad_gb13(
    r_grain: F_OR_Q_OR_ARR,
    temp: F_OR_Q,
    porosity: float,
    emissivity: float,
    e1: float = 1.34,
) -> F_OR_Q_OR_ARR:
    """ The radiative thermal radiation using GB13 (Gundlach & Blum 2013).

    Parameters
    ----------
    r_grain : float or Quantity
        The radius of the sphere. Assumed as micrometer if `float`.

    temp : float or Quantity
        The temperature of the sphere. Assumed to be in Kelvin if `float`.

    porosity : float
        The porosity of the regolith [dimensionless].

    emissivity : float
        The emissivity of the sphere [dimensionless].

    e1 : float
        The model parameter (mean-free path fitting by Gundlach and Blum (2012)
        Fig. 12), e1 = 1.34 +- 0.01. Interpolation only near porosity = 0.65 to
        0.85 (See Skorov et al. 2011).
    """
    r_grain = to_quantity(r_grain, u.um)
    temp = to_quantity(temp, u.K)
    k2 = 8*emissivity*SIGMA_SB_Q*e1*(porosity)/(1 - porosity)*r_grain
    return (k2*temp**3).to(u.W/u.m/u.K)


def coord_num(
    porosity: float,
    f_coeffs: dict = {1: 0.07318, 2: 2.193, 3: -3.357, 4: 3.914}
) -> float:
    """ The coordination number model (Sakatani et al. 2017).

    Parameters
    ----------
    porosity : float
        The porosity of the regolith [dimensionless].

    f_coeffs : dict
        The coefficients of the porosity dependence for the coordination
        number. The keys are the exponents of the porosity and the values are
        the coefficients. For example, if the porosity dependence is
        `a*porosity**2 + b*porosity + c/porosity`, then `coeffs` should be
        `coeffs = {2: a, 1: b, -1: c}`.
        Default is `coeffs = {1: 0.07318, 2: 2.193, 3: -3.357, 4: 3.914}`.
    """
    f = power_fun(porosity, f_coeffs)
    return 2.8112*(1 - porosity)**(-1/3) / (f**2 + f**4)


def k_solid_s17(
    r_grain: F_OR_Q_OR_ARR,
    cond_grain: F_OR_Q,
    temp: F_OR_Q,
    porosity: float,
    prat: F_OR_Q,
    ymod: F_OR_Q,
    gamma_coeffs: dict = {1: 6.67e-5},
    xi: float = 0.4
):
    """ The solid thermal conductivity using S17 (Sakatani et al. 2017).

    Parameters
    ----------
    r_grain, cond_grain : float or Quantity
        The radius and thermal conductivity of the sphere. Assumed as
        micrometer and W/m/K if `float`.

    temp : float or Quantity
        The temperature of the sphere. Assumed to be in Kelvin if `float`.

    porosity : float
        The porosity of the regolith [dimensionless].

    prat, ymod : float or Quantity
        The Poisson's ratio [dimensionless] and Young's modulus [GPa] of the
        sphere. `young` is assumed to be in GPa if `float`.

    gamma_coeffs : function
        The coefficients of the temperature dependence for the specific surface
        energy of the sphere (in the unit of J/m^2). The keys are the exponents
        of the temperature and the values are the coefficients. For example, if
        the temperature dependence is `a*T**2 + b*T + c/T`, then `coeffs`
        should be `coeffs = {2: a, 1: b, -1: c}`. Default is `coeffs = {1:
        6.67e-5}`.

    xi : float
        The model parameter (1 when perfectly smooth spheres), 0.4 is used by
        MacLennan and Emery (2022).

    Returns
    -------
    k_solid : Quantity
        The solid thermal conductivity in W/m/K.
    """
    r_c = jkr_contact(r=r_grain, temp=temp, prat=prat, ymod=ymod, gamma_coeffs=gamma_coeffs)
    coo = coord_num(porosity)
    return (cond_grain*(r_c/r_grain)*4*(1-porosity)*coo*xi/PI**2).to(u.W/u.m/u.K)


def k_gb13(
    r_grain: F_OR_Q_OR_ARR,
    cond_grain: F_OR_Q,
    temp: F_OR_Q,
    porosity: float,
    prat: F_OR_Q,
    ymod: F_OR_Q,
    emissivity: float,
    gamma_coeffs: dict = {1: 6.67e-5},
    f1: float = 0.0518,
    f2: float = 5.26,
    chi: float = 0.41,
    e1: float = 1.34,
) -> F_OR_Q_OR_ARR:
    """ The total thermal conductivity using GB13 (Gundlach & Blum 2013).

    Parameters
    ----------
    r_grain, cond_grain : float or Quantity
        The radius and thermal conductivity of the sphere. Assumed as
        micrometer and W/m/K if `float`.

    temp : float or Quantity
        The temperature of the sphere. Assumed to be in Kelvin if `float`.

    porosity : float
        The porosity of the regolith [dimensionless].

    prat, ymod : float or Quantity
        The Poisson's ratio [dimensionless] and Young's modulus [GPa] of the
        sphere. `young` is assumed to be in GPa if `float`.

    emissivity : float
        The emissivity of the sphere [dimensionless].

    gamma_coeffs : function
        The coefficients of the temperature dependence for the specific surface
        energy of the sphere (in the unit of J/m^2). The keys are the exponents
        of the temperature and the values are the coefficients. For example, if
        the temperature dependence is `a*T**2 + b*T + c/T`, then `coeffs`
        should be `coeffs = {2: a, 1: b, -1: c}`. Default is `coeffs = {1:
        6.67e-5}`.

    f1, f2, chi : float
        The model parameters (`f1` and `f2` calibrated by SC, FCC, and BCC
        (Chem & Tien 1973) and `xi` is a magic number found to make this model
        work for Apollo 11/12 samples (Gundlach & Blum 2013)).
        From GB13, f1 = 0.0518 +- 0.0345, f2 = 5.26 +- 0.94, and chi = 0.41 +-
        0.02 (stat) +- 0.10 (syst).

    e1 : float
        The model parameter (mean-free path fitting by Gundlach and Blum 2013).

    Returns
    -------
    k_total : Quantity
        The total thermal conductivity in W/m/K.
    """
    return (k_solid_gb13(r_grain=r_grain, cond_grain=cond_grain, temp=temp,
                         porosity=porosity, prat=prat, ymod=ymod,
                         gamma_coeffs=gamma_coeffs, f1=f1, f2=f2, chi=chi)
            + k_rad_gb13(r_grain=r_grain, temp=temp, porosity=porosity,
                         emissivity=emissivity, e1=e1))
