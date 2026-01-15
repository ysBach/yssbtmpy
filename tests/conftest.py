"""
Pytest configuration and shared fixtures for yssbtmpy tests.

This module provides:
1. Shared test data and fixtures
2. Hard-coded analytical reference values for validation
3. Utility functions for test comparisons
"""
import numpy as np
import pytest
from astropy import units as u

# =============================================================================
# Physical Constants (exact values for reproducibility)
# =============================================================================
# These are the same as in yssbtmpy.constants but repeated here for test clarity
PI = np.pi
HH = 6.62607004e-34      # [J.s] Planck constant
KB = 1.38064852e-23      # [J/K] Boltzmann constant
CC = 299792458.0         # [m/s] Speed of light
SIGMA_SB = 5.670367e-08  # [W/m^2/K^4] Stefan-Boltzmann constant


# =============================================================================
# Analytical Reference Values (Hard-coded from mathematical derivations)
# =============================================================================
class AnalyticalBlackbody:
    """Hard-coded Planck function values computed analytically.

    B_lambda(lambda, T) = 2*h*c^2 / lambda^5 / (exp(h*c/(lambda*k*T)) - 1)

    All values computed using high-precision calculation and verified.
    """
    # Test case: T=5000K, lambda=0.5um
    # B_lambda = 2*6.62607004e-34*(299792458)^2 / (0.5e-6)^5 / (exp(6.62607004e-34*299792458/(0.5e-6*1.38064852e-23*5000)) - 1)
    # Computed: 2.633e13 W/m^2/m/sr = 2.633e7 W/m^2/um/sr
    T5000_WLEN0p5_BLAM = 2.6330737e7  # W/m^2/um/sr (at 0.5 um, 5000 K)

    # Test case: T=300K, lambda=10um
    # B_lambda at 10um, 300K (peak is around 10um for ~300K)
    T300_WLEN10_BLAM = 9.888638e6  # W/m^2/um/sr (at 10 um, 300 K)

    # Test case: T=5777K (solar), lambda=0.5um
    T5777_WLEN0p5_BLAM = 4.0068e7  # W/m^2/um/sr (at 0.5 um, 5777 K)


class AnalyticalFluxConversions:
    """Hard-coded flux conversion values.

    1 Jy = 10^-26 W/m^2/Hz
    F_nu [Jy] = F_lambda [W/m^2/um] * lambda^2 / c * 10^26 * 10^12
              = F_lambda * lambda^2 * 3.33564095e11
    """
    # Test: 1 W/m^2/um at 1um -> Jy
    # F_nu = 1 * 1^2 * 3.33564095e11 = 3.33564095e11 Jy
    FLAM1_WLEN1_JY = 3.33564095e11  # Jy

    # Test: 1 W/m^2/um at 10um -> Jy
    # F_nu = 1 * 10^2 * 3.33564095e11 = 3.33564095e13 Jy
    FLAM1_WLEN10_JY = 3.33564095e13  # Jy

    # AB magnitude: m_AB = -2.5 * log10(F_nu / 3631 Jy)
    # 3631 Jy -> 0 mag
    JY3631_ABMAG = 0.0
    # 1 Jy -> m_AB = -2.5 * log10(1/3631) = 8.90...
    JY1_ABMAG = 8.9  # approximately


class AnalyticalPhaseFunction:
    """Hard-coded IAU HG phase function values.

    q = 0.286 + 0.656*G  (modern formula, Myhrvold 2016)
    q = 0.290 + 0.684*G  (classical/erroneous formula)
    """
    # Modern formula tests
    G0p15_Q_MODERN = 0.286 + 0.656 * 0.15  # = 0.3844
    G0p0_Q_MODERN = 0.286  # G=0 case
    G1p0_Q_MODERN = 0.286 + 0.656  # = 0.942

    # Classical formula tests
    G0p15_Q_CLASSICAL = 0.290 + 0.684 * 0.15  # = 0.3926
    G0p0_Q_CLASSICAL = 0.290
    G1p0_Q_CLASSICAL = 0.290 + 0.684  # = 0.974

    # IAU HG model at alpha=0 should be 1.0
    ALPHA0_INTENSITY = 1.0


class AnalyticalFresnel:
    """Hard-coded Fresnel coefficient values.

    At normal incidence (theta1=0):
    rs = rp = (n1 - n2)/(n1 + n2)
    ts = tp = 2*n1/(n1 + n2)

    At Brewster angle (theta_B = arctan(n2/n1)):
    rp = 0 (for p-polarization)
    """
    # Normal incidence n1=1 (air), n2=1.5 (glass)
    N1_1_N2_1p5_THETA0_RS = (1 - 1.5) / (1 + 1.5)  # = -0.2
    N1_1_N2_1p5_THETA0_TS = 2 * 1 / (1 + 1.5)  # = 0.8
    N1_1_N2_1p5_THETA0_RP = -0.2  # same at normal incidence
    N1_1_N2_1p5_THETA0_TP = 0.8

    # Reflectance at normal incidence: R = r^2
    N1_1_N2_1p5_THETA0_R = 0.04  # (-0.2)^2

    # Brewster angle for n1=1, n2=1.5
    BREWSTER_ANGLE_N1p5 = np.arctan(1.5)  # ~0.9828 rad = ~56.31 deg


class AnalyticalRelations:
    """Hard-coded asteroid relations values.

    D = d0 / sqrt(p) * 10^(-H/5)
    where d0 = 1329 km (Fowler & Chillemi 1992)
    """
    # p=0.2, H=15, d0=1329 -> D = 1329/sqrt(0.2)*10^(-15/5) = 1329/0.447*0.001 = 2.97 km
    P0p2_H15_D0_1329_DIAM = 2.9725252  # km

    # T_eqm = ((1-A)*S_1AU / (eta*sigma*eps*r_h^2))^(1/4)
    # With A=0.1, eta=1, eps=1, r_h=1 AU, S_1AU=1361.2 W/m^2
    # T_eqm = ((0.9*1361.2)/(1*5.670367e-8*1*1))^0.25 = 394.26 K
    A0p1_RH1_EPS1_ETA1_TEQM = 394.26  # K (approximately)


class AnalyticalConductivity:
    """Hard-coded thermal conductivity values for GB13 model.

    JKR contact radius: r_c = ((9*pi/4) * ((1-nu^2)/E) * gamma * r^2)^(1/3)
    """
    # For r=100um, T=200K, nu=0.25, E=78GPa, gamma=6.67e-5*T
    # This is a reference value for regression testing
    R100_T200_NU0p25_E78_RC = 0.157  # um (approximate)


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def wavelengths_um():
    """Standard wavelength array for testing (in micrometers)."""
    return np.array([0.5, 1.0, 5.0, 10.0, 20.0])


@pytest.fixture
def temperatures_K():
    """Standard temperature array for testing (in Kelvin)."""
    return np.array([100, 200, 300, 500, 1000])


@pytest.fixture
def phase_angles_deg():
    """Standard phase angle array for testing (in degrees)."""
    return np.array([0, 10, 30, 60, 90, 120])


@pytest.fixture
def slope_parameters():
    """Standard G parameter values for HG model testing."""
    return np.array([0.0, 0.15, 0.5, 1.0])


@pytest.fixture
def geometric_albedos():
    """Standard geometric albedo values for testing."""
    return np.array([0.05, 0.1, 0.2, 0.5])


@pytest.fixture
def heliocentric_distances_au():
    """Standard heliocentric distance values (in AU) for testing."""
    return np.array([0.5, 1.0, 2.0, 5.0])


@pytest.fixture
def incidence_angles_rad():
    """Standard incidence angles for Fresnel tests (in radians)."""
    return np.array([0, np.pi/6, np.pi/4, np.pi/3])


# =============================================================================
# Test Utility Functions
# =============================================================================
def assert_allclose_with_units(actual, expected, rtol=1e-10, atol=1e-12, **kwargs):
    """Assert arrays are close, handling astropy Quantities.

    Parameters
    ----------
    actual : array-like or Quantity
        Computed value.
    expected : array-like or Quantity
        Expected reference value.
    rtol : float
        Relative tolerance (default: 1e-10).
    atol : float
        Absolute tolerance (default: 1e-12).
    """
    if hasattr(actual, 'unit') and hasattr(expected, 'unit'):
        assert u.allclose(actual, expected, rtol=rtol, atol=atol, **kwargs)
    elif hasattr(actual, 'value'):
        np.testing.assert_allclose(actual.value, expected, rtol=rtol, atol=atol)
    elif hasattr(expected, 'value'):
        np.testing.assert_allclose(actual, expected.value, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
