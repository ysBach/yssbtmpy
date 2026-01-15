import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

import yssbtmpy as tm

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

wlen_mask = (0.4 <= tm.SOLAR_SPEC[:, 0]) & (tm.SOLAR_SPEC[:, 0] <= 6)
WLEN = tm.SOLAR_SPEC[wlen_mask, 0]*u.um


phys = dict(
    S=dict(p_vis=0.25, emis=0.845, refl=lambda x: 1),
    C=dict(p_vis=0.06, emis=0.97, refl=lambda x: 1)
)


fig, axs = plt.subplots(1, 1, figsize=(8, 6), sharex=False, sharey=False, gridspec_kw=None)

# sbs = [make_sb(0.05, 200), make_sb(0.05, 800), make_sb(0.2, 200)]

_RH = 2
for ti, color in zip([0., 600], "rb"):
    for name, _ls in zip("SC", ['-', '--']):
        _phy = phys[name]
        sb = tm.SmallBody()
        sb.set_ecl(r_hel=_RH, r_obs=np.sqrt(_RH**2-1), hel_ecl_lon=0, hel_ecl_lat=0, obs_ecl_lon=0, obs_ecl_lat=0, phase_ang=np.arcsin(1/_RH)*u.rad)
        sb.set_optical(slope_par=0.15, p_vis=_phy["p_vis"], diam_eff=10*u.km)
        sb.set_spin(spin_ecl_lon=0, spin_ecl_lat=90, rot_period=1)
        sb.set_thermal(ti=ti, emissivity=_phy["emis"], eta_beam=1)
        sb.set_tpm(nlon=360, nlat=90, Zmax=10, dZ=0.2)
        # print(sb.thermal_par)
        sb.calc_temp(in_kelvin=True)
        sb.calc_flux_ther(WLEN)
        sb.calc_flux_refl(refl=1., wlen_min=WLEN.min(), wlen_max=WLEN.max())

        emis_lambda = 1 - sb.phase_int*_phy["refl"](WLEN)
        axs.plot(WLEN, tm.flam2ab(emis_lambda*(sb.flux_ther/sb.emissivity) + sb.flux_refl, WLEN),
                 ls=_ls, color=color,
                 label=fr"$p_\mathrm{{V}} = {_phy['p_vis']}$ ({name}-class), $Γ={ti:.0f}$ [tiu], $\bar{{ε}}={_phy['emis']:.3f}$")
        # axs.plot(WLEN, sb.flux_ther + sb.flux_refl, ls=_ls)

    print(sb.diam_eff, sb.radi_eff)
# axs.set_aspect('auto')
axs.legend(loc="lower left")
axs.set(
    # yscale='log',
    # xscale='log',
    xlim=(0.7, 5),
    ylim=[21, 14],
    # ylim=(1.e-22, 1.e-15),
    xlabel="Wavelength [µm]",
    ylabel="AB magnitude",
    # ylabel=r"$F_\lambda \mathrm{[W/m^2/µm]}$",
    title=f"See Ivezic+2022 Icar 371, 114696, Fig. 1.\n$D = {sb.diam_eff.to_value(u.km)}$ km @ $r_\mathrm{{hel}} = {sb.r_hel.value:.1f}$ au, $r_\mathrm{{obs}} = {sb.r_obs.value:.1f}$ au, $α = {sb.phase_ang.value:.0f}°$"
)
plt.tight_layout()
plt.savefig("tests/spherex_tpm.png")
plt.close(fig)
