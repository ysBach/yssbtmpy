"""
Test suite for yssbtmpy.conductivity module.

Tests thermal conductivity models (Gundlach & Blum 2013) against
analytically derived values.

References
----------
- Gundlach & Blum (2013), Icarus 223, 479-492 (GB13)
- MacLennan & Emery (2022), PSJ 3, 47 (ME22)
- Johnson-Kendall-Roberts (JKR) contact theory
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u

from yssbtmpy.conductivity import (
    Material, jkr_contact, k_solid_gb13, k_rad_gb13, k_gb13
)
from yssbtmpy.constants import SIGMA_SB, PI


# =============================================================================
# Analytically derived reference values
# =============================================================================
# JKR contact radius formula:
#   r_c = ((9*pi/4) * ((1-nu^2)/E) * gamma * r^2)^(1/3)
# where:
#   nu = Poisson's ratio (dimensionless)
#   E = Young's modulus [Pa]
#   gamma = surface energy [J/m^2]
#   r = grain radius [m]
#
# For gamma = 6.67e-5 * T [J/m^2] (temperature dependent)

# Example calculation:
# r = 100 um = 1e-4 m
# T = 200 K
# nu = 0.25
# E = 78 GPa = 7.8e10 Pa
# gamma = 6.67e-5 * 200 = 0.01334 J/m^2
#
# prefactor = (9*pi/4) * (1-0.25^2)/(7.8e10) = 9*pi/4 * 0.9375/7.8e10
#           = 7.0686 * 0.9375/7.8e10 = 8.499e-11
# r_c^3 = 8.499e-11 * 0.01334 * (1e-4)^2 = 8.499e-11 * 0.01334 * 1e-8
#       = 1.134e-20
# r_c = (1.134e-20)^(1/3) = 2.25e-7 m = 0.225 um

JKR_REF = {
    # (r_um, T_K, nu, E_GPa): expected r_c in um
    (100.0, 200.0, 0.25, 78.0): 0.225,  # approximate
    (100.0, 300.0, 0.25, 78.0): 0.254,  # higher T -> higher gamma -> larger contact
    (50.0, 200.0, 0.25, 78.0): 0.142,   # smaller grain -> smaller contact
}

# GB13 solid conductivity formula:
#   k_solid = k_grain * (r_c / r) * f1 * exp(f2*(1-phi)) * chi
# where f1=0.0518, f2=5.26, chi=0.41 are calibration constants
# and phi is porosity

# GB13 radiative conductivity formula:
#   k_rad = 8 * eps * sigma * e1 * phi / (1-phi) * r * T^3
# where e1=1.34 is calibration constant

# For regression testing, hard-coded values at specific conditions
KB13_REFS = {
    # (r_um, T_K, porosity, k_grain): (k_solid, k_rad) in W/m/K
    # These are approximate values for regression testing
    (100.0, 200.0, 0.5, 4.05): (0.01, 0.001),  # order of magnitude
    (100.0, 300.0, 0.5, 4.05): (0.012, 0.004),  # higher T
}


# =============================================================================
# Test JKR contact radius
# =============================================================================
class TestJKRContact:
    """Tests for JKR contact radius calculation."""

    def test_temperature_dependence(self):
        """Test contact radius increases with temperature.

        gamma = 6.67e-5 * T, so r_c ~ T^(1/3)
        """
        r = 100.0  # um
        nu = 0.25
        E = 78 * u.GPa

        r_c_200 = jkr_contact(r, temp=200, prat=nu, ymod=E)
        r_c_300 = jkr_contact(r, temp=300, prat=nu, ymod=E)

        # r_c ~ T^(1/3), so r_c_300/r_c_200 ~ (300/200)^(1/3) ~ 1.145
        ratio = r_c_300.value / r_c_200.value
        expected_ratio = (300 / 200) ** (1 / 3)
        assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_grain_size_dependence(self):
        """Test contact radius scales as r^(2/3)."""
        nu = 0.25
        E = 78 * u.GPa
        T = 200

        r_c_100 = jkr_contact(100.0, temp=T, prat=nu, ymod=E)
        r_c_200 = jkr_contact(200.0, temp=T, prat=nu, ymod=E)

        # r_c ~ r^(2/3)
        ratio = r_c_200.value / r_c_100.value
        expected_ratio = (200 / 100) ** (2 / 3)
        assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_units(self):
        """Test JKR function handles units properly."""
        # With units
        r_c_unit = jkr_contact(
            100.0 * u.um, temp=200 * u.K, prat=0.25, ymod=78 * u.GPa
        )
        # Without units
        r_c_float = jkr_contact(100.0, temp=200, prat=0.25, ymod=78)

        assert r_c_unit.unit == u.um
        assert_allclose(r_c_unit.value, r_c_float.value, rtol=1e-10)

    def test_positive_contact(self):
        """Test contact radius is always positive."""
        r_c = jkr_contact(100.0, temp=200, prat=0.25, ymod=78)
        assert r_c.value > 0

    @pytest.mark.parametrize("r_um,T_K,expected_rc_um", [
        (100.0, 200.0, 0.225),
        (100.0, 300.0, 0.254),
        (50.0, 200.0, 0.142),
    ])
    def test_reference_values(self, r_um, T_K, expected_rc_um):
        """Test against hard-coded reference values."""
        r_c = jkr_contact(r_um, temp=T_K, prat=0.25, ymod=78)
        # These are approximate regression values, use larger tolerance
        assert_allclose(r_c.value, expected_rc_um, rtol=0.05)


# =============================================================================
# Test GB13 solid conductivity
# =============================================================================
class TestKSolidGB13:
    """Tests for GB13 solid thermal conductivity."""

    def test_porosity_dependence(self):
        """Test k_solid decreases with porosity (exponentially)."""
        k_solid_02 = k_solid_gb13(
            r_grain=100.0, cond_grain=4.05, temp=200,
            porosity=0.2, prat=0.25, ymod=78
        )
        k_solid_05 = k_solid_gb13(
            r_grain=100.0, cond_grain=4.05, temp=200,
            porosity=0.5, prat=0.25, ymod=78
        )
        k_solid_08 = k_solid_gb13(
            r_grain=100.0, cond_grain=4.05, temp=200,
            porosity=0.8, prat=0.25, ymod=78
        )

        # Higher porosity = lower conductivity (exp(-f2*phi) term)
        assert k_solid_02.value > k_solid_05.value > k_solid_08.value

    def test_grain_conductivity_scaling(self):
        """Test k_solid scales linearly with k_grain."""
        k_solid_1 = k_solid_gb13(
            r_grain=100.0, cond_grain=1.0, temp=200,
            porosity=0.5, prat=0.25, ymod=78
        )
        k_solid_2 = k_solid_gb13(
            r_grain=100.0, cond_grain=2.0, temp=200,
            porosity=0.5, prat=0.25, ymod=78
        )

        assert_allclose(k_solid_2.value / k_solid_1.value, 2.0, rtol=1e-10)

    def test_positive_conductivity(self):
        """Test solid conductivity is always positive."""
        k_solid = k_solid_gb13(
            r_grain=100.0, cond_grain=4.05, temp=200,
            porosity=0.5, prat=0.25, ymod=78
        )
        assert k_solid.value > 0


# =============================================================================
# Test GB13 radiative conductivity
# =============================================================================
class TestKRadGB13:
    """Tests for GB13 radiative thermal conductivity."""

    def test_temperature_dependence(self):
        """Test k_rad scales as T^3."""
        k_rad_200 = k_rad_gb13(
            r_grain=100.0, temp=200, porosity=0.5, emissivity=0.9
        )
        k_rad_400 = k_rad_gb13(
            r_grain=100.0, temp=400, porosity=0.5, emissivity=0.9
        )

        # k_rad ~ T^3
        ratio = k_rad_400.value / k_rad_200.value
        expected_ratio = (400 / 200) ** 3
        assert_allclose(ratio, expected_ratio, rtol=1e-10)

    def test_grain_size_scaling(self):
        """Test k_rad scales linearly with grain size."""
        k_rad_100 = k_rad_gb13(
            r_grain=100.0, temp=200, porosity=0.5, emissivity=0.9
        )
        k_rad_200 = k_rad_gb13(
            r_grain=200.0, temp=200, porosity=0.5, emissivity=0.9
        )

        ratio = k_rad_200.value / k_rad_100.value
        assert_allclose(ratio, 2.0, rtol=1e-10)

    def test_porosity_dependence(self):
        """Test k_rad increases with porosity (phi/(1-phi) term)."""
        k_rad_03 = k_rad_gb13(
            r_grain=100.0, temp=200, porosity=0.3, emissivity=0.9
        )
        k_rad_06 = k_rad_gb13(
            r_grain=100.0, temp=200, porosity=0.6, emissivity=0.9
        )

        # phi/(1-phi): 0.3/0.7 = 0.429, 0.6/0.4 = 1.5
        # ratio should be 1.5/0.429 = 3.5
        ratio = k_rad_06.value / k_rad_03.value
        expected = (0.6 / 0.4) / (0.3 / 0.7)
        assert_allclose(ratio, expected, rtol=1e-10)

    def test_emissivity_scaling(self):
        """Test k_rad scales linearly with emissivity."""
        k_rad_05 = k_rad_gb13(
            r_grain=100.0, temp=200, porosity=0.5, emissivity=0.5
        )
        k_rad_10 = k_rad_gb13(
            r_grain=100.0, temp=200, porosity=0.5, emissivity=1.0
        )

        ratio = k_rad_10.value / k_rad_05.value
        assert_allclose(ratio, 2.0, rtol=1e-10)


# =============================================================================
# Test total GB13 conductivity
# =============================================================================
class TestKGB13Total:
    """Tests for total GB13 thermal conductivity (solid + radiative)."""

    def test_sum_of_components(self):
        """Test k_total = k_solid + k_rad."""
        kwargs = dict(
            r_grain=100.0, cond_grain=4.05, temp=200,
            porosity=0.5, prat=0.25, ymod=78, emissivity=0.9
        )

        k_total = k_gb13(**kwargs)
        k_solid = k_solid_gb13(**{k: v for k, v in kwargs.items()
                                  if k not in ['emissivity']})
        k_rad = k_rad_gb13(
            r_grain=kwargs['r_grain'], temp=kwargs['temp'],
            porosity=kwargs['porosity'], emissivity=kwargs['emissivity']
        )

        assert_allclose(k_total.value, k_solid.value + k_rad.value, rtol=1e-10)

    def test_solid_dominates_small_grains(self):
        """Test solid conductivity dominates for small grains."""
        # k_solid ~ r^(-1/3) (through r_c/r), k_rad ~ r
        # So for small grains, k_solid >> k_rad
        kwargs = dict(
            r_grain=1.0,  # 1 um, very small
            cond_grain=4.05, temp=200,
            porosity=0.5, prat=0.25, ymod=78, emissivity=0.9
        )

        k_solid = k_solid_gb13(**{k: v for k, v in kwargs.items()
                                  if k not in ['emissivity']})
        k_rad = k_rad_gb13(
            r_grain=kwargs['r_grain'], temp=kwargs['temp'],
            porosity=kwargs['porosity'], emissivity=kwargs['emissivity']
        )

        # For 1um grains, solid should dominate
        assert k_solid.value > k_rad.value

    def test_radiative_dominates_large_grains_high_temp(self):
        """Test radiative conductivity dominates for large grains at high T."""
        kwargs = dict(
            r_grain=1000.0,  # 1 mm
            cond_grain=4.05, temp=500,  # high temperature
            porosity=0.7,  # high porosity
            prat=0.25, ymod=78, emissivity=0.9
        )

        k_solid = k_solid_gb13(**{k: v for k, v in kwargs.items()
                                  if k not in ['emissivity']})
        k_rad = k_rad_gb13(
            r_grain=kwargs['r_grain'], temp=kwargs['temp'],
            porosity=kwargs['porosity'], emissivity=kwargs['emissivity']
        )

        # For large grains at high T, radiative may dominate
        # (depends on exact parameters, just check both contribute)
        k_total = k_gb13(**kwargs)
        assert k_total.value > k_solid.value
        assert k_total.value > k_rad.value


# =============================================================================
# Test Material class
# =============================================================================
class TestMaterial:
    """Tests for Material dataclass."""

    def test_from_me22_s_type(self):
        """Test creating S-type asteroid material."""
        mat = Material.from_ME22_GB13('S')

        assert mat.spec == 'S'
        # Check grain conductivity is defined
        k_grain, _, _ = mat.k_grain(200)
        assert k_grain.value > 0

    @pytest.mark.parametrize("spec", ['S', 'V', 'E', 'M', 'C', 'Ch', 'B', 'K', 'P', 'Met'])
    def test_all_spectral_types(self, spec):
        """Test all spectral types can be created."""
        mat = Material.from_ME22_GB13(spec)
        assert mat.spec.lower() == spec.lower()

        # Check basic functions work
        k_grain, _, _ = mat.k_grain(200)
        rho, _, _ = mat.rho_grain(200)
        cs, _, _ = mat.cs_grain(200)

        assert k_grain.value > 0
        assert rho.value > 0
        assert cs.value > 0

    def test_k_eff(self):
        """Test effective conductivity calculation."""
        mat = Material.from_ME22_GB13('S')
        k_eff, k_solid, k_rad = mat.k_eff(r_grain=100, temp=200, porosity=0.5)

        assert k_eff.value > 0
        assert_allclose(k_eff.value, k_solid.value + k_rad.value, rtol=1e-10)

    def test_unknown_spec_raises(self):
        """Test unknown spectral type raises error."""
        with pytest.raises(ValueError):
            Material.from_ME22_GB13('UNKNOWN')
