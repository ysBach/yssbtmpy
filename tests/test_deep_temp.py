import numpy as np
from astropy import units as u

import yssbtmpy as tm


def test_deep_temp():
    sb = tm.SmallBody()
    sb.set_ecl(r_hel=1.1, hel_ecl_lon=0, hel_ecl_lat=0, r_obs=1, obs_ecl_lon=0, obs_ecl_lat=0, alpha=0)
    sb.set_spin(spin_ecl_lon=0, spin_ecl_lat=90, rot_period=6*u.h)
    sb.set_optical(a_bond=0.1, diam_eff=1*u.km, slope_par=0.15, p_vis=tm.AG2p(0.1, 0.15))
    sb.set_thermal(ti=300, emissivity=1)
    sb.set_tpm(nlon=360*5, nlat=91)
    sb.calc_temp(full=True, permanent_shadow_u=0, min_iter=50, max_iter=100000)

    # The fluctuation of the deepest temperature should not be large.
    # Here, it tests it is < 0.002T_eqm.
    np.testing.assert_allclose(np.ptp(sb.tempfull[:, :, -1], axis=1), 0, atol=0.002)

    # For periodic insolation, time-average of the surface temperature should be
    # equal to the deepest temperature.
    # Here, the deepest temperature is taken as the mean of deepest slab over the
    # time, because it actually fluctuates up to
    #   np.max(np.ptp(sb.tempfull[:, :, -1], axis=1)) ~ 0.002 of T_eqm
    # at rhel = 0.1. (at 1.1, it is <1.e-8)
    np.testing.assert_allclose(
        np.mean(sb.tempsurf, axis=1) - np.mean(sb.tempfull[:, :, -1], axis=1),
        0,
        atol=1.e-6
    )