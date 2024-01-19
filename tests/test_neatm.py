import numpy as np
from astropy import units as u
from scipy.interpolate import UnivariateSpline

import yssbtmpy as tm

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
#   See https://nbviewer.org/github/moeyensj/atm_notebooks/blob/master/paper1/validation/example_1991EE&Eros.ipynb (Moeyens et al. 2020)
#   Original ref: http://adsabs.harvard.edu/abs/1979Icar...40..297L & http://adsabs.harvard.edu/abs/1998Icar..131..291H
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
