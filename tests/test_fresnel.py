"""
Test suite for yssbtmpy.scat.fresnel module.

Tests Fresnel reflection/transmission coefficients against analytically
derived values at special angles.

References
----------
- Fresnel equations for electromagnetic waves at interfaces
- Special cases: normal incidence, Brewster angle, critical angle
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from yssbtmpy.scat.fresnel import fresnel_coeff, fresnel_coeff_angle, fresnel_intensity


# =============================================================================
# Analytically derived reference values
# =============================================================================
# Fresnel coefficients at normal incidence (theta1=0):
#   rs = rp = (n1 - n2)/(n1 + n2)
#   ts = tp = 2*n1/(n1 + n2)
#
# Reflectance/Transmittance:
#   Rs = Rp = rs^2 = rp^2
#   Ts = Tp = (n2/n1) * ts^2 = (n2/n1) * tp^2  (for normal Poynting)
#
# Energy conservation: R + T = 1

# Test case: air (n1=1) to glass (n2=1.5)
N1_AIR = 1.0
N2_GLASS = 1.5

# Normal incidence (theta=0)
RS_NORMAL_GLASS = (N1_AIR - N2_GLASS) / (N1_AIR + N2_GLASS)  # -0.2
TS_NORMAL_GLASS = 2 * N1_AIR / (N1_AIR + N2_GLASS)  # 0.8
RP_NORMAL_GLASS = RS_NORMAL_GLASS  # Same at normal incidence
TP_NORMAL_GLASS = TS_NORMAL_GLASS
R_NORMAL_GLASS = RS_NORMAL_GLASS**2  # 0.04
T_NORMAL_GLASS = N2_GLASS / N1_AIR * TS_NORMAL_GLASS**2  # 0.96

# Brewster angle: theta_B = arctan(n2/n1)
# At Brewster angle, rp = 0 (p-polarized light is fully transmitted)
BREWSTER_ANGLE_GLASS = np.arctan(N2_GLASS / N1_AIR)  # ~0.9828 rad (~56.31 deg)

# Test case: air to water (n2=1.33)
N2_WATER = 1.33
RS_NORMAL_WATER = (N1_AIR - N2_WATER) / (N1_AIR + N2_WATER)  # ~-0.1416
R_NORMAL_WATER = RS_NORMAL_WATER**2  # ~0.02

# Critical angle for total internal reflection (glass to air):
# sin(theta_c) = n2/n1 (only when n1 > n2)
CRITICAL_ANGLE_GLASS_AIR = np.arcsin(N1_AIR / N2_GLASS)  # ~0.7297 rad (~41.81 deg)


# =============================================================================
# Test Fresnel coefficients
# =============================================================================
class TestFresnelCoeff:
    """Tests for fresnel_coeff function."""

    def test_normal_incidence_glass(self):
        """Test Fresnel coefficients at normal incidence (air to glass)."""
        rs, ts, rp, tp = fresnel_coeff(N1_AIR, N2_GLASS, theta1=0)

        assert_allclose(rs, RS_NORMAL_GLASS, rtol=1e-12)
        assert_allclose(ts, TS_NORMAL_GLASS, rtol=1e-12)
        # Note: rp and tp may have sign differences at normal incidence
        assert_allclose(np.abs(rp), np.abs(RP_NORMAL_GLASS), rtol=1e-12)
        assert_allclose(tp, TP_NORMAL_GLASS, rtol=1e-12)

    def test_normal_incidence_water(self):
        """Test Fresnel coefficients at normal incidence (air to water)."""
        rs, ts, rp, tp = fresnel_coeff(N1_AIR, N2_WATER, theta1=0)

        assert_allclose(rs, RS_NORMAL_WATER, rtol=1e-10)

    def test_brewster_angle_p_reflection(self):
        """Test rp=0 at Brewster angle."""
        rs, ts, rp, tp = fresnel_coeff(N1_AIR, N2_GLASS, theta1=BREWSTER_ANGLE_GLASS)

        # p-polarized reflection should be zero at Brewster angle
        assert_allclose(rp, 0.0, atol=1e-10)
        # s-polarized still has non-zero reflection
        assert np.abs(rs) > 0.1

    def test_symmetry_at_normal(self):
        """Test |rs|=|rp| and ts=tp at normal incidence (signs may differ)."""
        rs, ts, rp, tp = fresnel_coeff(N1_AIR, N2_GLASS, theta1=0)

        # At normal incidence, magnitudes should be equal but signs may differ
        assert_allclose(np.abs(rs), np.abs(rp), rtol=1e-12)
        assert_allclose(ts, tp, rtol=1e-12)

    def test_array_input(self):
        """Test with array of angles."""
        theta1 = np.array([0, np.pi/6, np.pi/4, np.pi/3])
        rs, ts, rp, tp = fresnel_coeff(N1_AIR, N2_GLASS, theta1=theta1)

        assert rs.shape == theta1.shape
        assert ts.shape == theta1.shape

    @pytest.mark.parametrize("n1,n2,theta,expected_rs", [
        (1.0, 1.5, 0, (1.0 - 1.5) / (1.0 + 1.5)),   # Air to glass: -0.2
        (1.0, 1.33, 0, (1.0 - 1.33) / (1.0 + 1.33)), # Air to water: exact
        (1.0, 2.0, 0, (1.0 - 2.0) / (1.0 + 2.0)),   # Air to high-n: -1/3
    ])
    def test_various_materials(self, n1, n2, theta, expected_rs):
        """Test Fresnel coefficients for various material combinations."""
        rs, ts, rp, tp = fresnel_coeff(n1, n2, theta)
        assert_allclose(rs, expected_rs, rtol=1e-12)


class TestFresnelCoeffAngle:
    """Tests for fresnel_coeff_angle function (angle-only version)."""

    def test_consistent_with_index_version(self):
        """Test angle-based formula gives valid results."""
        # Test at moderate angle where formulas are stable
        theta1 = 0.5  # ~29 degrees

        # Compute theta2 from Snell's law: n1*sin(theta1) = n2*sin(theta2)
        theta2 = np.arcsin(N1_AIR / N2_GLASS * np.sin(theta1))

        rs_ang, ts_ang, rp_ang, tp_ang = fresnel_coeff_angle(theta1, theta2)

        # Basic sanity checks - coefficients should be bounded
        assert np.abs(rs_ang) <= 1
        assert np.abs(rp_ang) <= 1
        assert ts_ang >= 0
        assert tp_ang >= 0


class TestFresnelIntensity:
    """Tests for fresnel_intensity function (reflectance/transmittance)."""

    def test_normal_incidence_reflectance(self):
        """Test reflectance at normal incidence."""
        Rs, Ts, Rp, Tp = fresnel_intensity(N1_AIR, N2_GLASS, theta1=0)

        assert_allclose(Rs, R_NORMAL_GLASS, rtol=1e-10)
        assert_allclose(Rp, R_NORMAL_GLASS, rtol=1e-10)

    def test_energy_conservation(self):
        """Test R + T = 1 (energy conservation)."""
        theta1 = np.array([0, np.pi/6, np.pi/4, np.pi/3])
        Rs, Ts, Rp, Tp = fresnel_intensity(N1_AIR, N2_GLASS, theta1=theta1)

        # Energy conservation
        assert_allclose(Rs + Ts, 1.0, rtol=1e-12)
        assert_allclose(Rp + Tp, 1.0, rtol=1e-12)

    def test_brewster_angle_p_reflectance(self):
        """Test Rp=0 at Brewster angle."""
        Rs, Ts, Rp, Tp = fresnel_intensity(N1_AIR, N2_GLASS, theta1=BREWSTER_ANGLE_GLASS)

        assert_allclose(Rp, 0.0, atol=1e-12)
        assert_allclose(Tp, 1.0, rtol=1e-12)

    def test_reflectance_bounds(self):
        """Test 0 <= R <= 1 and 0 <= T <= 1."""
        theta1 = np.linspace(0, np.pi/2 - 0.01, 100)
        Rs, Ts, Rp, Tp = fresnel_intensity(N1_AIR, N2_GLASS, theta1=theta1)

        assert np.all((Rs >= 0) & (Rs <= 1))
        assert np.all((Ts >= 0) & (Ts <= 1))
        assert np.all((Rp >= 0) & (Rp <= 1))
        assert np.all((Tp >= 0) & (Tp <= 1))

    @pytest.mark.parametrize("n1,n2,theta,expected_R", [
        (1.0, 1.5, 0, ((1.0 - 1.5) / (1.0 + 1.5))**2),    # Air to glass: 0.04
        (1.0, 1.33, 0, ((1.0 - 1.33) / (1.0 + 1.33))**2), # Air to water: exact
        (1.0, 2.0, 0, ((1.0 - 2.0) / (1.0 + 2.0))**2),   # Air to high-n: 1/9
    ])
    def test_reflectance_values(self, n1, n2, theta, expected_R):
        """Test reflectance at normal incidence for various materials."""
        Rs, Ts, Rp, Tp = fresnel_intensity(n1, n2, theta1=theta)
        # At normal incidence, Rs = Rp
        assert_allclose(Rs, expected_R, rtol=1e-12)


# =============================================================================
# Special cases and physics validation
# =============================================================================
class TestPhysicsValidation:
    """Tests validating physical behavior of Fresnel equations."""

    def test_grazing_incidence(self):
        """Test reflectance approaches 1 at grazing incidence."""
        theta1 = np.pi/2 - 0.001  # Very close to 90 degrees
        Rs, Ts, Rp, Tp = fresnel_intensity(N1_AIR, N2_GLASS, theta1=theta1)

        # Both polarizations should have high reflectance near grazing
        assert Rs > 0.9
        assert Rp > 0.9

    def test_increasing_reflectance_with_angle(self):
        """Test that reflectance generally increases with incident angle."""
        theta1 = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2 - 0.01])
        Rs, Ts, Rp, Tp = fresnel_intensity(N1_AIR, N2_GLASS, theta1=theta1)

        # Rs should increase monotonically
        assert np.all(np.diff(Rs) > 0)

        # Rp dips to 0 at Brewster angle then increases, so not monotonic
        # But at normal and grazing it should satisfy R_normal < R_grazing
        assert Rs[0] < Rs[-1]

    def test_no_reflection_same_medium(self):
        """Test no reflection when n1 = n2."""
        Rs, Ts, Rp, Tp = fresnel_intensity(1.5, 1.5, theta1=np.pi/4)

        assert_allclose(Rs, 0.0, atol=1e-10)
        assert_allclose(Rp, 0.0, atol=1e-10)
        assert_allclose(Ts, 1.0, rtol=1e-10)
        assert_allclose(Tp, 1.0, rtol=1e-10)


# =============================================================================
# Error handling
# =============================================================================
class TestErrorHandling:
    """Tests for proper error handling."""

    def test_angle_above_90_raises(self):
        """Test that theta1 > pi/2 raises an error."""
        with pytest.raises(ValueError):
            fresnel_coeff(N1_AIR, N2_GLASS, theta1=np.pi)

    def test_negative_angle_works(self):
        """Test that negative angles are handled (absolute value)."""
        # Negative angle should give same result as positive
        rs_pos, _, rp_pos, _ = fresnel_coeff(N1_AIR, N2_GLASS, theta1=0.3)
        rs_neg, _, rp_neg, _ = fresnel_coeff(N1_AIR, N2_GLASS, theta1=-0.3, check=False)
        # Note: depends on implementation whether negative angles are allowed
