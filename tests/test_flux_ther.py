import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline

import yssbtmpy as tm

InteractiveShell.ast_node_interactivity = 'last_expr'

# We need to do it in a separate cell. See:
# https://github.com/jupyter/notebook/issues/3385
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'latin modern math', 'font.size': 12, 'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True, 'axes.formatter.limits': (-4, 4),
    'axes.grid': True, 'grid.color': 'gray', 'grid.linewidth': 0.5,
    'xtick.top': True, 'ytick.right': True,
    'xtick.direction': 'inout', 'ytick.direction': 'inout',
    'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,  # default 2.0
    'xtick.major.size': 4.0, 'ytick.major.size': 4.0,  # default 3.5
    'xtick.minor.visible': True, 'ytick.minor.visible': True
})


def test_flux_ther():
    # Compare with Moeyens et al. 2020 Icar, 341, 113575, Fig. 1
    # Note: my calculation is slightly different from theirs, likely becuase (i)
    # they used NEATM, not TPM, and (ii) I used dlon&dlat, while I guess they used
    # sphere obj file (needs check)

    wlen = [9.1, 13, 22.6, 30]*u.um
    sb = tm.SmallBody()
    sb.set_ecl(r_hel=3, r_obs=2, hel_ecl_lon=0, hel_ecl_lat=0, obs_ecl_lon=0, obs_ecl_lat=0, alpha=0)
    sb.set_spin(spin_ecl_lon=0, spin_ecl_lat=90, rot_period=3*3600)
    sb.set_optical(diam_eff=1*u.km, p_vis=0.2, slope_par=0.15)
    sb.set_thermal(ti=0, emissivity=0.7)
    sb.set_tpm(nlon=360, nlat=90, Zmax=10, dZ=0.2)
    sb.calc_temp(in_kelvin=True)
    sb.calc_flux_ther(wlen)

    u.allclose(
        sb.flux_ther,
        np.array([9.51722899e-18, 1.30671540e-17, 7.10093836e-18, 3.73525690e-18])*u.W/u.m**2/u.um
    )


def test_neatm():
    wlen_mask = (0.4 <= tm.SOLAR_SPEC[:, 0]) & (tm.SOLAR_SPEC[:, 0] <= 20)
    WLEN = tm.SOLAR_SPEC[wlen_mask, 0]*u.um

    # Table 1 from Lebofsky paper (taking only data above 4 microns)
    fnu_max = np.array([7.13, 40, 289, 317, 388, 424, 457, 489, 470])  # last was erroneously 480.
    # fnu_min = np.array([2.16, 11.8, 97, 103, 128, 141, 154, 168, 164])
    # fnu_avg = (fnu_max + fnu_min) / 2
    wlen_um = np.array([3.6, 4.9, 8.45, 8.76, 10.4, 11.6, 12.6, 19.0, 22.0])

    # Convert to sensible units
    # flam_avg = tm.jy2flam(fnu_avg, wlen_um) # fnu_avg / (wlen_um**2 * 1 / (tm.CC / 10**8) * 10**12)
    # flam_min = tm.jy2flam(fnu_min, wlen_um) # fnu_min / (wlen_um**2 * 1 / (tm.CC / 10**8) * 10**12)
    flam_max = tm.jy2flam(fnu_max, wlen_um)  # fnu_max / (wlen_um**2 * 1 / (tm.CC / 10**8) * 10**12)
    flam_err = flam_max * np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05])

    # _ecl, _optical, _thermal below are NEATM parameters for for Eros
    #   See
    #   https://nbviewer.org/github/moeyensj/atm_notebooks/blob/master/paper1/validation/example_1991EE&Eros.ipynb (Moeyens et al. 2020)
    #   Original ref:
    #   http://adsabs.harvard.edu/abs/1979Icar...40..297L & http://adsabs.harvard.edu/abs/1998Icar..131..291H
    # SPIN is useless because anyway ti=0.

    sb = tm.SmallBody()
    sb.set_ecl(r_hel=1.134, r_obs=0.153, hel_ecl_lon=0, hel_ecl_lat=0, obs_ecl_lon=0, obs_ecl_lat=0, alpha=9.9)
    sb.set_optical(slope_par=0.15, p_vis=0.2, diam_eff=23.6*u.km, hmag_vis=tm.pD2H(0.2, 23.6*u.km))
    sb.set_spin(spin_ecl_lon=0, spin_ecl_lat=90, rot_period=6*3600)
    sb.set_thermal(ti=0.1, emissivity=0.9, eta_beam=1.05)
    sb.set_tpm(nlon=360, nlat=90, Zmax=10, dZ=0.2)
    sb.calc_temp(in_kelvin=True)
    sb.calc_flux_ther(wlen_um)
    sb.calc_flux_refl(refl=1., wlen_min=WLEN.min(), wlen_max=WLEN.max())
    spl_refl = UnivariateSpline(WLEN.value, sb.flux_refl, k=3, s=0)
    flux = sb.flux_ther.value + spl_refl(wlen_um)

    np.testing.assert_allclose(
        flux,
        np.array([1.432e-12, 5.228e-12, 1.237e-11, 1.235e-11, 1.134e-11,
                  1.013e-11, 9.058e-12, 3.960e-12, 2.709e-12]),
        rtol=0.002
    )


def test_eros_figure():
    wlen_mask = (0.4 <= tm.SOLAR_SPEC[:, 0]) & (tm.SOLAR_SPEC[:, 0] <= 30)
    WLEN = tm.SOLAR_SPEC[wlen_mask, 0]*u.um

    # Table 1 from Lebofsky paper (taking only data above 4 microns)
    fnu_max = np.array([7.13, 40, 289, 317, 388, 424, 457, 489, 470])  # last was erroneously 480.
    fnu_min = np.array([2.16, 11.8, 97, 103, 128, 141, 154, 168, 164])
    # fnu_avg = (fnu_max + fnu_min) / 2
    wlen_um = np.array([3.6, 4.9, 8.45, 8.76, 10.4, 11.6, 12.6, 19.0, 22.0])

    # Convert to flam_SI
    flam_max = tm.jy2flam(fnu_max, wlen_um)  # fnu_max / (wlen_um**2 * 1 / (tm.CC / 10**8) * 10**12)
    flam_min = tm.jy2flam(fnu_min, wlen_um)  # fnu_min / (wlen_um**2 * 1 / (tm.CC / 10**8) * 10**12)
    # flam_avg = tm.jy2flam(fnu_avg, wlen_um) # fnu_avg / (wlen_um**2 * 1 / (tm.CC / 10**8) * 10**12)
    flam_err = flam_max * np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05])

    fig, axs = plt.subplots(1, 1, figsize=(8, 5), layout="constrained",
                            gridspec_kw=None, sharex=False, sharey=False)
    ax = axs

    # _ecl, _optical, _thermal below are NEATM parameters for for Eros
    # ecliptic coords from HORIZONS for 1975-01-17 UT Catalina (see Lebofsky et al. 1979)
    #   See
    # https://nbviewer.org/github/moeyensj/atm_notebooks/blob/master/paper1/validation/example_1991EE&Eros.ipynb (Moeyens et al. 2020)
    #   Original ref:
    # http://adsabs.harvard.edu/abs/1979Icar...40..297L & http://adsabs.harvard.edu/abs/1998Icar..131..291H
    # SPIN is useless because anyway ti=0.
    sb = tm.SmallBody()
    sb.set_ecl(
        r_hel=1.134, r_obs=0.153,
        hel_ecl_lon=117, hel_ecl_lat=1.45,
        obs_ecl_lon=113, obs_ecl_lat=10.7,
        alpha=9.9
    )
    sb.set_optical(slope_par=0.15, p_vis=0.2, diam_eff=23.6*u.km, hmag_vis=tm.pD2H(0.2, 23.6*u.km))
    sb.set_spin(spin_ecl_lon=0, spin_ecl_lat=90, rot_period=1)
    sb.set_thermal(ti=0.1, emissivity=0.9, eta_beam=1.05)
    sb.set_tpm(nlon=360, nlat=90, Zmax=10, dZ=0.2)
    print(sb.thermal_par)
    sb.calc_temp(in_kelvin=True)
    sb.calc_flux_ther(WLEN)
    sb.calc_flux_refl(refl=1., wlen_min=WLEN.min(), wlen_max=WLEN.max())

    ax.plot(WLEN.to_value(u.um), sb.flux_refl, "b:", label="refl (NEATM)")
    ax.plot(WLEN.to_value(u.um), sb.flux_ther, "k:", label="ther (NEATM)")
    ax.plot(WLEN.to_value(u.um), sb.flux_refl+sb.flux_ther, "r-", label="NEATM (Harris 1998)")

    # Now, using TPM parameters by Hinkle et al. 2022 Icar. 382, 114939.
    #   ti=100 because we have no roughness model (see lowest χ2 models in their Figs. 4).
    spin = SkyCoord(11.37*u.deg, 17.22*u.deg, 1*u.kpc, frame='icrs')
    spin = spin.transform_to('heliocentrictrueecliptic')  # from SBDB

    sb_tpm = tm.SmallBody()
    sb_tpm.set_ecl(
        r_hel=1.134, r_obs=0.153,
        hel_ecl_lon=117, hel_ecl_lat=1.45,
        obs_ecl_lon=113, obs_ecl_lat=10.7,
        alpha=9.9
    )
    sb_tpm.set_optical(slope_par=0.46, p_vis=0.25, diam_eff=15.6*u.km)
    # ^ slope Gpar from SBDB. diam from H=11.16.
    sb_tpm.set_spin(
        spin_ecl_lon=spin.lon.to_value(u.deg), spin_ecl_lat=spin.lat.to_value(u.deg),
        rot_period=5.27*u.h
    )
    sb_tpm.set_thermal(ti=100, emissivity=0.9, eta_beam=1.0)
    sb_tpm.set_tpm(nlon=360, nlat=90, Zmax=10, dZ=0.2)
    print(sb_tpm.thermal_par)
    sb_tpm.calc_temp(in_kelvin=True)
    sb_tpm.calc_flux_ther(WLEN)
    sb_tpm.calc_flux_refl(refl=1., wlen_min=WLEN.min(), wlen_max=WLEN.max())
    ax.plot(WLEN.to_value(u.um), sb_tpm.flux_refl+sb_tpm.flux_ther, "b-", label="TPM (Hinkle 2022)")

    ax.errorbar(wlen_um, flam_max, yerr=flam_err,
                fmt='o', c="r", ms=2, capsize=1, elinewidth=1, label="flux_max")
    ax.plot(wlen_um, flam_min, "k.", label="flux_min")

    ax.legend(loc="upper left", framealpha=0.95)
    ax.set(
        xscale='log', yscale='log',
        xlabel='Wavelength (µm)', ylabel='Flux (W/m2/µm)',
        title="Eros. Also see Moeyens et al. (2020) Icarus 341, 113622 Fig. 5",
        ylim=(4e-13, 2e-11),
        xlim=(0.8, 31),
    )

    plt.savefig("tests/eros_neatm_tpm.png", dpi=300, bbox_inches="tight")
    plt.close()
