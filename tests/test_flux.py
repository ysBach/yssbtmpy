import numpy as np
from astropy import units as u

from yssbtmpy import SmallBody

# Compare with Moeyens et al. 2020 Icar, 341, 113575, Fig. 1
# Note: my calculation is slightly different from theirs, likely becuase (i)
# they used NEATM, not TPM, and (ii) I used dlon&dlat, while I guess they used
# sphere obj file (needs check)

wlen = [9.1, 13, 22.6, 30]*u.um
sb = SmallBody()
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