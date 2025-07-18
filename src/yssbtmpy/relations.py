from warnings import warn

import numpy as np
from astropy import units as u

from .constants import NOUNIT, PI, S1AU, SIGMA_SB, TIU
from .util import F_OR_ARR, F_OR_Q, F_OR_Q_OR_ARR, to_quantity

# TODO: maybe make it to accept terminal command lines.

__all__ = ["rot_omega2h", "h2rot_omega",
           "solve_phid", "solve_Gq", "solve_pAG", "solve_pDH", "solve_temp_eqm",
           "solve_thermal_par", "solve_rmrho", "solve_pw",
           "G2q", "q2G", "pG2A", "pA2G", "AG2p", "pD2H", "pH2D", "DH2p",
           "T_eqm", "Thetapar", "rm2rho", "rrho2m", "mrho2r", "p2w", "w2p",
           "phi2d", "d2phi"]


def _count_not_None(*args) -> int:
    cnt = 0
    for arg in args:
        if arg is not None:
            cnt += 1
    return cnt


def _setup_pars(pardict: dict, unitdict: dict, to_value: bool) -> dict:
    newpars = {}
    for k, v in pardict.items():
        newpar = to_quantity(v, desired=unitdict[k], to_value=to_value)
        newpars[k] = newpar
    return newpars


def _i_should_solve(pardict: dict) -> bool:
    N_pars = len(pardict)
    pars = list(pardict.values())
    N_not_None = _count_not_None(*pars)

    if N_not_None == N_pars:
        warn(f"All of {N_pars} parameters are given: nothing to solve.")
        return False

    elif N_not_None < N_pars - 1:
        raise ValueError(
            f"Only {N_not_None} out of {N_pars - 1} parameters are given:\n{pardict}")

    return True


# def _take_values(*args):
#     vals = []
#     for arg in args:
#         if isinstance(arg, u.Quantity):
#             vals.append(arg.value)
#         else:
#             vals.append(arg)
#     return vals


def rot_omega2h(radps: F_OR_ARR = None, degps: F_OR_ARR = None) -> F_OR_ARR:
    if _count_not_None(radps, degps) != 1:
        raise ValueError("One and only one of [radps, degps] must be given.")

    if radps is not None:
        return 2*PI/(radps*3600)

    elif degps is not None:
        return 360/(degps*3600)


def h2rot_omega(hour: F_OR_ARR) -> F_OR_ARR:
    return 2*PI/(hour*3600)


def solve_phid(
        phi: F_OR_ARR = None,
        d: F_OR_ARR = None,
        to_value: bool = False
) -> F_OR_ARR:
    """ Solves for Krumbien phi-scale grain size from micrometer
    """
    pdict = dict(phi=phi, d=d)
    udict = dict(phi=NOUNIT, d=u.um)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)
    if not solve:
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    phi = ps["phi"]
    d = ps["d"]

    # 3. Solve
    if phi is None:
        ps["phi"] = -np.log2(d/1000)
    elif d is None:
        ps["d"] = 2**(-phi)*1000  # in um, not mm

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)

    return ps


def solve_Gq(
        slope_par: F_OR_Q_OR_ARR = None,
        phase_int: F_OR_Q_OR_ARR = None,
        classical: bool = False,
        to_value: bool = False
) -> dict:
    """ Calculates the phase integral of IAU HG system.

    Notes
    -----
    In the Appendix of Bowell et al. (1989) in Asteroids II (Binzel ed.)
    pp.524-556, the phase integral is given as
    .. math::
        q = 0.290 + 0.684 G \quad (0 \le G \le 1)
    in Eq (A7). But as first pointed out in Myhrvold (2016), PASP, 128,
    045004(d5004), this is not true, but it should be
    .. math::
        q = 0.286 + 0.656 G
    and I (Y. P. Bach) also confirmed it. Thus, it is more sensical to use
    this new one than the classically used (previous) one.

    Parameters
    ----------
    slope_par : float or array-like
        The slope parameter (G parameter) in the IAU HG system.

    phase_int : float or array-like
        The phase integral.

    classical : bool, optional.
        Whether to use the classical (erroneous) formula which have been used
        for decades.
        Default is `False`.

    to_value : bool, optional.
        Whether to return in float (value) or `~astropy.Quantity`.
        Defalut is `True`

    Return
    ------
    slope_par, phase_int : float or array-like
        The the slope parameter and phase integral.
    """
    pdict = dict(slope_par=slope_par, phase_int=phase_int)
    udict = dict(slope_par=NOUNIT, phase_int=NOUNIT)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)

    if not solve:  # return as is
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    g = ps["slope_par"]
    q = ps["phase_int"]

    # 3. Solve
    const1, const2 = 0.286, 0.656
    if classical:
        const1, const2 = 0.290, 0.684

    if q is None:
        ps["phase_int"] = const1 + const2*g

    elif g is None:
        ps["slope_par"] = (q - const1)/const2

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)
    return ps


def solve_pAG(
        p_vis: F_OR_Q_OR_ARR = None,
        a_bond: F_OR_Q_OR_ARR = None,
        slope_par: F_OR_Q_OR_ARR = None,
        classical: bool = False,
        to_value: bool = False
) -> dict:
    """ Solves the albedo relations from IAU H, G magnitude system.
    Note
    ----
    In the Appendix of Bowell et al. (1989) in Asteroids II (Binzel ed.)
    pp.524-556, the phase integral is given as
    .. math::
        q = 0.290 + 0.684 G \quad (0 \le G \le 1)
    in Eq (A7). But as first pointed out in Myhrvold (2016), PASP, 128,
    045004(d5004), this is not true, but it should be
    .. math::
        q = 0.286 + 0.656 G
    and I (Y. P. Bach) also confirmed it. Thus, it is more sensical to use this
    new one than the classically used (previous) one.

    Parameters
    ----------
    p_vis, a_bond: float or array-like
        The geometric albedo in Johnson-Cousins V-band and the Bond albedo.

    slope_par : float or array-like
        The slope parameter (G parameter) in the IAU HG system.

    classical : bool, optional.
        Whether to use the classical (erroneous) formula which have been used
        for decades.
        Default is `False`.

    to_value : bool, optional.
        Whether to return in float (value) or `~astropy.Quantity`.
        Defalut is `True`

    Return
    ------
    a_bond or p_vis or slope_par: float or array-like
        The Bond albedo or the geometric albedo or the slope parameter.
    """
    pdict = dict(p_vis=p_vis, a_bond=a_bond, slope_par=slope_par)
    udict = dict(p_vis=NOUNIT, a_bond=NOUNIT, slope_par=NOUNIT)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)

    if not solve:  # return as is
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    p = ps["p_vis"]
    a = ps["a_bond"]
    g = ps["slope_par"]

    # 3. Solve
    if p is None:
        _ = solve_Gq(slope_par=g, phase_int=None, classical=classical)
        q = _["phase_int"]
        ps["p_vis"] = a/q

    elif a is None:
        _ = solve_Gq(slope_par=g, phase_int=None, classical=classical)
        q = _["phase_int"]
        ps["a_bond"] = p*q

    elif g is None:
        q = a/p
        _ = solve_Gq(slope_par=None, phase_int=q, classical=classical)
        ps["slope_par"] = _["slope_par"]

    # there should be no case which exits the above if clause.

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)
    return ps


def solve_pDH(
        p_vis: F_OR_Q_OR_ARR = None,
        diam_eff: F_OR_Q_OR_ARR = None,
        hmag_vis: F_OR_Q_OR_ARR = None,
        d0: F_OR_Q = 1329,
        to_value: bool = False
) -> dict:
    """ Get one of geometric albdeo, size, or absolute magnitude.
    Note
    ----
    The equation
    .. math::
        D = \frac{D_0}{\sqrt{p_\mathrm{V}}} \times 10^{-H_\mathrm{V}/5}
    is used. Here, :math:`D_0 = 2 \mathrm{au} \times 10^{-V_\odot/5}`, where
    :math:`V_\odot` is the apparent magnitude of the Sun in Johnson-Cousins
    V-band filter (-26.76). The equation is derived such that the visual
    magnitude of a circular Lambertian plate of diameter D and geometric albedo
    p_vis at perfect opposition to be the same as hmag_vis. This diameter is
    called the effective diameter. Shape-modelers sometimes confuse this
    diameter with the volume-equivalent diameter, but they are not necessarily
    identical (although the difference must not be significant).

    It is first used in Fowler and Chillemi (1992), IRAS Minor Planet Surv.,
    ed. E. F. Tedesco et al. (Phillips Laboratory Tech. Rep. No.PL-TR-92-2049),
    17. There it is defined that :math:`D_0 = 1329 \mathrm{km}`, while the
    modern measurements would give 1330.3 km, but the uncertainty in solar
    magnitude will tune it by 0.1% level, thus 1329 km is still in the error
    range, and thus there is no much need to update the value "1329". But to
    leave a flexibility, I set d0 an input paremeter here.

    Parameters
    ----------
    p_vis, diam_eff, hmag_vis : float, ~Quantity, or array-like of such
        Geometric albdeo, effective diameter in km, and the visual absolute
        magnitude. Two and only two of these three must be given. if not
        Quantity, they must be dimensionless, km, and magnitude, respectively.

    d0 : float, Quantity, optional.
        The effective diameter when geometric albedo is unity and the absolute
        magnitude is zero (in km or `Quantity` convertable to it).

    to_value : bool, optional.
        Whether to return in float (value) or `~astropy.Quantity`.
        Defalut is `True`

    Return
    ------
    a : float or array-like
        One of geometric albdeo, effective diameter in km, or the visual
        absolute magnitude, depending on the input parmeters.
    """

    pdict = dict(p_vis=p_vis, diam_eff=diam_eff, hmag_vis=hmag_vis)
    udict = dict(p_vis=NOUNIT, diam_eff=u.km, hmag_vis=NOUNIT)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)

    if not solve:  # return as is
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    p = ps["p_vis"]
    d = ps["diam_eff"]
    h = ps["hmag_vis"]
    d0 = to_quantity(d0, desired=u.km, to_value=True)

    # 3. Solve
    if p is None:
        ps["p_vis"] = (d0/d)**2*10**(-2*h/5)

    elif d is None:
        ps["diam_eff"] = d0/np.sqrt(p_vis)*10**(-1*h/5)

    elif hmag_vis is None:
        ps["hmag_vis"] = 5*np.log10(d0/d*1/np.sqrt(p))

    # there should be no case which exits the above if clause.

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)
    return ps


def solve_temp_eqm(
        temp_eqm: F_OR_Q_OR_ARR = None,
        a_bond: F_OR_Q_OR_ARR = None,
        eta_beam: F_OR_Q_OR_ARR = None,
        r_hel: F_OR_Q_OR_ARR = None,
        emissivity: F_OR_Q_OR_ARR = None,
        to_value: bool = False
) -> dict:
    """ Solve the equilibrium temperature formula
    Note
    ----
    Identical to the subsolar temperature in STM/NEATM, which assume
    instanteneous thermal equilibrium (i.e., null thermal inertia). But this is
    not the subsolar temperature in TPM.
    If ``rh__au = 1``, temp_eqm here is identical to T_1.

    Parameters
    ----------
    temp_eqm : float or array-like
        The equilibrium temperature (in K if float).

    a_bond : float or array-like
        The Bond albedo.

    eta_beam : float or array-like
        The beaming parameter. It is normally fixed to 1 in TPM.

    r_hel : float or array-like
        The heliocentric distance of the body (in the unit of au if float).

    emissivity : float or array-like
        The (spectrum averaged) emissivity. It is conventionally fixed to 0.900
        in thermal modeling.

    to_value : bool, optional.
        Whether to return in float (value) or `~astropy.Quantity`.
        Defalut is `True`
    """
    pdict = dict(temp_eqm=temp_eqm, a_bond=a_bond,
                 eta_beam=eta_beam, r_hel=r_hel, emissivity=emissivity)
    udict = dict(temp_eqm=u.K, a_bond=NOUNIT, eta_beam=NOUNIT,
                 r_hel=u.au, emissivity=NOUNIT)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)

    if not solve:  # return as is
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    t = ps["temp_eqm"]
    a = ps["a_bond"]
    beam = ps["eta_beam"]
    rh__au = ps["r_hel"]  # in au unit, not in meter!!
    e = ps["emissivity"]

    # 3. Solve
    if rh__au is None:
        rh__au2 = (1 - a)*S1AU/(SIGMA_SB*beam*e*t**4)
        ps["r_hel"] = np.sqrt(rh__au2)

    elif t is None:
        t4 = ((1 - a)*S1AU/(beam*SIGMA_SB*e*rh__au*rh__au))
        ps["temp_eqm"] = t4**(1/4)

    elif a is None:
        ps["a_bond"] = (1 - t**4*e*SIGMA_SB*beam*rh__au*rh__au/S1AU)

    elif beam is None:
        ps["eta_beam"] = (1 - a)*S1AU/(SIGMA_SB*e*rh__au*rh__au*t**4)

    elif e is None:
        ps["emissivity"] = (1 - a)*S1AU/(SIGMA_SB*beam*rh__au*rh__au*t**4)

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)
    return ps


def solve_thermal_par(
        thermal_par: F_OR_Q_OR_ARR = None,
        ti: F_OR_Q_OR_ARR = None,
        rot_period: F_OR_Q_OR_ARR = None,
        temp_eqm: F_OR_Q_OR_ARR = None,
        emissivity: F_OR_Q_OR_ARR = None,
        to_value: bool = False
) -> dict:
    """ Solves the thermal parameter (Theta) equation

    Parameters
    ----------
    thermal_par : float or array-like or Quantity
        The thermal parameter.

    ti : float or array-like or Quantity
        The thermal inertia in SI unit (in tiu if not Quantity).

    rot_period : float or array-like or Quantity
        The rotational period (in seconds if not Quantity).

    temp_eqm : float or array-like or Quantity
        The subsolar equilibrium temperature (see ~solve_temp_eqm).

    emissivity : float or array-like or Quantity
        The (spectrum averaged) emissivity. It is conventionally fixed to 0.900
        in thermal modeling.

    to_value : bool, optional.
        Whether to return in float (value) or `~astropy.Quantity`.
        Defalut is `True`
    """
    pdict = dict(thermal_par=thermal_par, ti=ti, rot_period=rot_period,
                 temp_eqm=temp_eqm, emissivity=emissivity)
    udict = dict(thermal_par=NOUNIT, ti=TIU, rot_period=u.s,
                 temp_eqm=u.K, emissivity=NOUNIT)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)

    if not solve:  # return as is
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    th = ps["thermal_par"]
    ti = ps["ti"]
    prot = ps["rot_period"]
    temp = ps["temp_eqm"]  # in au unit, not in meter!!
    e = ps["emissivity"]

    # 3. Solve
    if prot is None:
        wrot = (e*SIGMA_SB*th*temp**3/ti)**2  # in [rad/s]
        ps["rot_period"] = (2*PI/wrot)

    else:
        wrot = (2*PI/prot)  # in [rad/s]

        if th is None:
            ps["thermal_par"] = ti*np.sqrt(wrot)/(e*SIGMA_SB*temp**3)

        elif ti is None:
            ps["ti"] = th*e*SIGMA_SB*temp**3/np.sqrt(wrot)

        elif temp is None:
            t3 = ti*np.sqrt(wrot)/(e*SIGMA_SB*th)
            ps["temp_eqm"] = t3**(1/3)

        elif e is None:
            ps["emissivity"] = ti*np.sqrt(wrot)/(th*SIGMA_SB*temp**3)

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)
    return ps


def solve_rmrho(
        radius: F_OR_Q_OR_ARR = None,
        mass: F_OR_Q_OR_ARR = None,
        mass_den: F_OR_Q_OR_ARR = None,
        to_value: bool = False
) -> dict:
    """ Solves the radius, mass, density relation for a homogeneous spehre.
    Parameters
    ----------
    radius, mass, mass_den : float, Quantity, or array-like of such, optional
        The radius, mass, and the mass density of the homogeneous sphere to be
        solved. All in SI unit if not Quantity.
    """
    pdict = dict(radius=radius, mass=mass, mass_den=mass_den)
    udict = dict(radius=u.m, mass=u.kg, mass_den=u.kg/u.m**3)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)

    if not solve:  # return as is
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    r = ps["radius"]
    m = ps["mass"]
    rho = ps["mass_den"]

    # 3. Solve
    if r is None:
        ps["radius"] = (3*m/(4*PI*rho))**(1/3)
    elif m is None:
        ps["mass"] = (4*PI/3)*rho*r**3
    elif rho is None:
        ps["mass_den"] = 3*m/(4*PI*r**3)

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)
    return ps


def solve_pw(
        rot_period: F_OR_Q_OR_ARR = None,
        rot_omega: F_OR_Q_OR_ARR = None,
        to_value: bool = False
) -> dict:
    """ Solves the rotational period and angular speed equation.

    Parameters
    ----------
    rot_period, rot_omega : float, Quantity, or array-like of such, optional.
        The rotational period and rotational angular speed to be solved. If not
        quantity, they are interpreted as SI units (seconds and rad/s).
    """
    pdict = dict(rot_period=rot_period, rot_omega=rot_omega)
    udict = dict(rot_period=u.s, rot_omega=1/u.s)

    # 1. Check: (a) all are given or (b) more than one missing
    solve = _i_should_solve(pardict=pdict)

    if not solve:  # return as is
        return _setup_pars(pardict=pdict, unitdict=udict, to_value=to_value)

    # 2. Convert all to plain float
    ps = _setup_pars(pardict=pdict, unitdict=udict, to_value=True)
    prot = ps["rot_period"]
    wrot = ps["rot_omega"]

    # 3. Solve
    if wrot is None:
        ps["rot_omega"] = 2*PI/prot
    elif prot is None:
        ps["rot_period"] = 2*PI/wrot

    # 4. Convert to astropy.Quantity if needed.
    ps = _setup_pars(pardict=ps, unitdict=udict, to_value=to_value)
    return ps


def G2q(
        slope_par: F_OR_Q_OR_ARR = 0.15,
        classical: bool = False,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_qG``.
    """
    return solve_Gq(slope_par=slope_par,
                    phase_int=None,
                    classical=classical,
                    to_value=to_value)["phase_int"]


def q2G(
        phase_int: F_OR_Q_OR_ARR = 0.35,
        classical: bool = False,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_qG``.
    """
    return solve_Gq(slope_par=None,
                    phase_int=phase_int,
                    classical=classical,
                    to_value=to_value)["slope_par"]


def pG2A(
        p_vis: F_OR_Q_OR_ARR,
        slope_par: F_OR_Q_OR_ARR = 0.15,
        classical: bool = False,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pAG``.
    """
    return solve_pAG(p_vis=p_vis,
                     a_bond=None,
                     slope_par=slope_par,
                     classical=classical,
                     to_value=to_value)["a_bond"]


def pA2G(
        p_vis: F_OR_Q_OR_ARR,
        a_bond=0.1,
        classical: bool = False,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pAG``.
    """
    return solve_pAG(p_vis=p_vis,
                     a_bond=a_bond,
                     slope_par=None,
                     classical=classical,
                     to_value=to_value)["slope_par"]


def AG2p(
        a_bond: F_OR_Q_OR_ARR,
        slope_par: F_OR_Q_OR_ARR = 0.15,
        classical: bool = False,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pAG``.
    """
    return solve_pAG(p_vis=None,
                     a_bond=a_bond,
                     slope_par=slope_par,
                     classical=classical,
                     to_value=to_value)["p_vis"]


def pD2H(
        p_vis: F_OR_Q_OR_ARR,
        diam_eff: F_OR_Q_OR_ARR = 1,
        d0: F_OR_Q_OR_ARR = 1329,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pDH``.
    """
    return solve_pDH(p_vis=p_vis,
                     diam_eff=diam_eff,
                     hmag_vis=None,
                     d0=d0,
                     to_value=to_value)["hmag_vis"]


def pH2D(
        p_vis: F_OR_Q_OR_ARR,
        hmag_vis: F_OR_Q_OR_ARR,
        d0=1329,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pDH``.
    """
    return solve_pDH(p_vis=p_vis,
                     diam_eff=None,
                     hmag_vis=hmag_vis,
                     d0=d0,
                     to_value=to_value)["diam_eff"]


def DH2p(
        diam_eff: F_OR_Q_OR_ARR,
        hmag_vis: F_OR_Q_OR_ARR,
        d0: F_OR_Q_OR_ARR = 1329,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pDH``.
    """
    return solve_pDH(p_vis=None,
                     diam_eff=diam_eff,
                     hmag_vis=hmag_vis,
                     d0=d0,
                     to_value=to_value)["p_vis"]


def T_eqm(
        a_bond: F_OR_Q_OR_ARR,
        eta_beam: F_OR_Q_OR_ARR = 1.0,
        r_hel: F_OR_Q_OR_ARR = 1.0,
        emissivity: F_OR_Q_OR_ARR = 1.,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_Teqm``.
    """
    return solve_temp_eqm(a_bond=a_bond,
                          eta_beam=eta_beam,
                          r_hel=r_hel,
                          emissivity=emissivity,
                          to_value=to_value)["temp_eqm"]


def Thetapar(
        ti: F_OR_Q_OR_ARR,
        temp_eqm: F_OR_Q_OR_ARR,
        rot_omega: F_OR_Q_OR_ARR = None,
        rot_period: F_OR_Q_OR_ARR = None,
        emissivity: F_OR_Q_OR_ARR = 1.
) -> float | u.Quantity:
    """ Convenience function of ``solve_Theta``.
    """
    if _count_not_None(rot_omega, rot_period) != 1:
        raise ValueError(
            "One and only one of [rot_omega, rot_period] must be given.")

    rot_omega = to_quantity(rot_omega, desired=1/u.s, to_value=True)
    ti__tiu = to_quantity(ti, desired=TIU, to_value=True)
    emissivity = to_quantity(emissivity, desired=NOUNIT, to_value=True)
    rot_period__h = to_quantity(rot_period, desired=u.s, to_value=True)
    temp_eqm__k = to_quantity(temp_eqm, desired=u.K, to_value=True)

    if rot_omega is None:
        rot_omega = 2*PI/(rot_period)

    ps = solve_thermal_par(thermal_par=None,
                           ti=ti__tiu,
                           rot_period=rot_period__h,
                           temp_eqm=temp_eqm__k,
                           emissivity=emissivity)
    return ps["thermal_par"]


def rm2rho(
        radius: F_OR_Q_OR_ARR,
        mass: F_OR_Q_OR_ARR,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_rmrho``.
    """
    return solve_rmrho(radius=radius,
                       mass=mass,
                       mass_den=None,
                       to_value=to_value)["mass_den"]


def rrho2m(
        radius: F_OR_Q_OR_ARR,
        mass_den: F_OR_Q_OR_ARR,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_rmrho``.
    """
    return solve_rmrho(radius=radius,
                       mass=None,
                       mass_den=mass_den,
                       to_value=to_value)["mass"]


def mrho2r(
        mass: F_OR_Q_OR_ARR,
        mass_den: F_OR_Q_OR_ARR,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_rmrho``.
    """
    return solve_rmrho(radius=None,
                       mass=mass,
                       mass_den=mass_den,
                       to_value=to_value)["radius"]


def p2w(
        rot_period: F_OR_Q_OR_ARR,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pw``.
    """
    return solve_pw(rot_period=rot_period,
                    rot_omega=None,
                    to_value=to_value)["rot_omega"]


def w2p(
        rot_omega: F_OR_Q_OR_ARR,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pw``.
    """
    return solve_pw(rot_period=None,
                    rot_omega=rot_omega,
                    to_value=to_value)["rot_period"]


def phi2d(
        phi: F_OR_Q_OR_ARR,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pw``.
    """
    return solve_phid(phi=phi,
                      d=None,
                      to_value=to_value)["d"]


def d2phi(
        d: F_OR_Q_OR_ARR,
        to_value: bool = False
) -> float | u.Quantity:
    """ Convenience function of ``solve_pw``.
    """
    return solve_phid(phi=None,
                      d=d,
                      to_value=to_value)["phi"]
