"""
Test suite for yssbtmpy.util module.

Tests coordinate transformations and utility functions against
analytically derived values.

References
----------
- Standard spherical coordinate conventions
- Rotation matrices for coordinate system transformations
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from astropy import units as u

from yssbtmpy.util import (
    to_quantity, to_val, lonlat2cart, sph2cart, cart2sph,
    calc_aspect_ang, mat_ec2fs, mat_bf2ss
)
from yssbtmpy.constants import PI, D2R


# =============================================================================
# Analytically derived reference values
# =============================================================================
# Spherical to Cartesian:
#   x = r * sin(theta) * cos(phi)
#   y = r * sin(theta) * sin(phi)
#   z = r * cos(theta)
# where theta = colatitude (0 at pole), phi = longitude

# Special cases:
# theta=0, phi=0: (0, 0, 1) - north pole
# theta=90, phi=0: (1, 0, 0) - on equator at phi=0
# theta=90, phi=90: (0, 1, 0) - on equator at phi=90
# theta=180, phi=0: (0, 0, -1) - south pole

SPH2CART_REFS = {
    # (theta_deg, phi_deg, r): expected (x, y, z)
    (0, 0, 1): (0, 0, 1),        # North pole
    (90, 0, 1): (1, 0, 0),       # Equator, phi=0
    (90, 90, 1): (0, 1, 0),      # Equator, phi=90
    (180, 0, 1): (0, 0, -1),     # South pole
    (45, 0, 1): (np.sqrt(2)/2, 0, np.sqrt(2)/2),  # 45 deg colatitude
    (90, 45, 1): (np.sqrt(2)/2, np.sqrt(2)/2, 0),  # Equator, phi=45
}

# lon/lat to Cartesian:
# lat = 90 - theta (so lat=90 is north pole, lat=0 is equator)
LONLAT2CART_REFS = {
    # (lon_deg, lat_deg, r): expected (x, y, z)
    (0, 90, 1): (0, 0, 1),       # North pole
    (0, 0, 1): (1, 0, 0),        # Equator, lon=0
    (90, 0, 1): (0, 1, 0),       # Equator, lon=90
    (0, -90, 1): (0, 0, -1),     # South pole
}


# =============================================================================
# Test to_quantity and to_val
# =============================================================================
class TestQuantityConversion:
    """Tests for quantity conversion utilities."""

    def test_to_quantity_float(self):
        """Test float input gets unit assigned."""
        result = to_quantity(10.0, u.km)
        assert result.unit == u.km
        assert result.value == 10.0

    def test_to_quantity_with_unit(self):
        """Test Quantity input gets converted."""
        result = to_quantity(1.0 * u.m, u.km)
        assert result.unit == u.km
        assert_allclose(result.value, 0.001, rtol=1e-12)

    def test_to_quantity_to_value(self):
        """Test to_value=True returns float."""
        result = to_quantity(1.0 * u.km, u.m, to_value=True)
        assert not hasattr(result, 'unit')
        assert_allclose(result, 1000.0, rtol=1e-12)

    def test_to_val_shortcut(self):
        """Test to_val is equivalent to to_quantity with to_value=True."""
        result1 = to_quantity(1.0 * u.km, u.m, to_value=True)
        result2 = to_val(1.0 * u.km, u.m)
        assert_allclose(result1, result2, rtol=1e-12)

    def test_array_input(self):
        """Test array input handling."""
        arr = np.array([1.0, 2.0, 3.0])
        result = to_quantity(arr, u.km)
        assert result.shape == arr.shape

    def test_incompatible_units_raises(self):
        """Test incompatible units raise error."""
        with pytest.raises(ValueError):
            to_quantity(1.0 * u.kg, u.m)


# =============================================================================
# Test spherical-Cartesian conversions
# =============================================================================
class TestSph2Cart:
    """Tests for spherical to Cartesian conversion."""

    @pytest.mark.parametrize("theta,phi,expected", [
        (0, 0, (0, 0, 1)),
        (90, 0, (1, 0, 0)),
        (90, 90, (0, 1, 0)),
        (180, 0, (0, 0, -1)),
    ])
    def test_cardinal_directions(self, theta, phi, expected):
        """Test cardinal direction conversions."""
        result = sph2cart(theta, phi, degree=True, r=1)
        assert_allclose(result, expected, atol=1e-10)

    def test_45_degree_colatitude(self):
        """Test 45 degree colatitude."""
        result = sph2cart(45, 0, degree=True, r=1)
        expected = (np.sqrt(2) / 2, 0, np.sqrt(2) / 2)
        assert_allclose(result, expected, rtol=1e-10)

    def test_radius_scaling(self):
        """Test radius scales the result."""
        result_r1 = sph2cart(45, 45, degree=True, r=1)
        result_r2 = sph2cart(45, 45, degree=True, r=2)
        assert_allclose(result_r2, 2 * result_r1, rtol=1e-10)

    def test_unit_vector_norm(self):
        """Test unit vector has norm 1."""
        theta = np.random.uniform(0, 180)
        phi = np.random.uniform(0, 360)
        result = sph2cart(theta, phi, degree=True, r=1)
        assert_allclose(np.linalg.norm(result), 1.0, rtol=1e-10)

    def test_radian_input(self):
        """Test radian input mode."""
        result_deg = sph2cart(90, 90, degree=True, r=1)
        result_rad = sph2cart(np.pi / 2, np.pi / 2, degree=False, r=1)
        assert_allclose(result_deg, result_rad, rtol=1e-10)


class TestLonLat2Cart:
    """Tests for longitude/latitude to Cartesian conversion."""

    @pytest.mark.parametrize("lon,lat,expected", [
        (0, 90, (0, 0, 1)),       # North pole
        (0, 0, (1, 0, 0)),        # Equator, prime meridian
        (90, 0, (0, 1, 0)),       # Equator, 90 deg east
        (0, -90, (0, 0, -1)),     # South pole
    ])
    def test_geographic_points(self, lon, lat, expected):
        """Test geographic coordinate conversions."""
        result = lonlat2cart(lon, lat, degree=True, r=1)
        assert_allclose(result, expected, atol=1e-10)

    def test_equator_sweep(self):
        """Test points on equator."""
        for lon in [0, 45, 90, 135, 180, 270]:
            result = lonlat2cart(lon, 0, degree=True, r=1)
            # z should be 0 on equator
            assert_allclose(result[2], 0, atol=1e-10)
            # norm should be 1
            assert_allclose(np.linalg.norm(result), 1, rtol=1e-10)


class TestCart2Sph:
    """Tests for Cartesian to spherical conversion."""

    def test_inverse_of_sph2cart(self):
        """Test cart2sph is inverse of sph2cart."""
        # Test non-pole points to avoid singularities
        for theta_in in [30, 60, 90, 120, 150]:
            for phi_in in [45, 90, 180, 270]:
                xyz = sph2cart(theta_in, phi_in, degree=True, r=1)
                r, theta, phi = cart2sph(xyz[0], xyz[1], xyz[2], degree=True)

                assert_allclose(r, 1.0, rtol=1e-12)
                assert_allclose(theta, theta_in, atol=1e-10)
                # phi can wrap around at boundaries
                assert_allclose(phi % 360, phi_in % 360, atol=1e-10)

    def test_cardinal_vectors(self):
        """Test Cartesian unit vectors."""
        # x-axis
        r, theta, phi = cart2sph(1, 0, 0, degree=True)
        assert_allclose(theta, 90, atol=1e-10)
        assert_allclose(phi, 0, atol=1e-10)

        # y-axis
        r, theta, phi = cart2sph(0, 1, 0, degree=True)
        assert_allclose(theta, 90, atol=1e-10)
        assert_allclose(phi, 90, atol=1e-10)

        # z-axis
        r, theta, phi = cart2sph(0, 0, 1, degree=True)
        assert_allclose(theta, 0, atol=1e-10)


# =============================================================================
# Test aspect angle calculation
# =============================================================================
class TestCalcAspectAng:
    """Tests for aspect angle calculation."""

    def test_parallel_vectors(self):
        """Test aspect angle of parallel vectors."""
        spin = np.array([0, 0, 1])  # pointing +z
        r_hel = np.array([0, 0, -1])  # pointing -z (toward spin)

        aspect, _, _ = calc_aspect_ang(spin, r_hel)

        # Aspect angle is angle between spin and -r_hel
        # -r_hel = +z, spin = +z, so angle = 0
        assert_allclose(aspect.value, 0, atol=1e-10)

    def test_antiparallel_vectors(self):
        """Test aspect angle of antiparallel vectors."""
        spin = np.array([0, 0, 1])  # pointing +z
        r_hel = np.array([0, 0, 1])  # pointing +z

        aspect, _, _ = calc_aspect_ang(spin, r_hel)

        # -r_hel = -z, spin = +z, so angle = 180
        assert_allclose(aspect.value, 180, atol=1e-10)

    def test_perpendicular_vectors(self):
        """Test aspect angle of perpendicular vectors."""
        spin = np.array([0, 0, 1])  # pointing +z
        r_hel = np.array([1, 0, 0])  # pointing +x

        aspect, _, _ = calc_aspect_ang(spin, r_hel)

        # -r_hel = -x, spin = +z, angle = 90
        assert_allclose(aspect.value, 90, atol=1e-10)


# =============================================================================
# Test rotation matrices
# =============================================================================
class TestRotationMatrices:
    """Tests for coordinate transformation matrices."""

    def test_mat_bf2ss_identity_at_equator(self):
        """Test body-fixed to surface system at equator (colat=90)."""
        mat = mat_bf2ss(90)  # equator

        # At equator, the surface normal points radially outward
        # Check matrix properties
        assert mat.shape == (3, 3)
        # Should be orthogonal
        assert_allclose(np.dot(mat, mat.T), np.eye(3), atol=1e-10)

    def test_mat_bf2ss_pole(self):
        """Test body-fixed to surface system at pole (colat=0)."""
        mat = mat_bf2ss(0)  # north pole

        assert mat.shape == (3, 3)
        # Should be orthogonal
        assert_allclose(np.dot(mat, mat.T), np.eye(3), atol=1e-10)

    def test_mat_ec2fs_orthogonality(self):
        """Test ecliptic to frame system matrix is orthogonal."""
        r_vec = np.array([1.0, 0.0, 0.0])
        spin_vec = np.array([0.0, 0.0, 1.0])

        mat = mat_ec2fs(r_vec, spin_vec)

        assert mat.shape == (3, 3)
        # Should be orthogonal (M @ M.T = I)
        assert_allclose(np.dot(mat, mat.T), np.eye(3), atol=1e-10)


# =============================================================================
# Regression tests with hard-coded values
# =============================================================================
class TestRegressionValues:
    """Regression tests with hard-coded reference values."""

    def test_sph2cart_45_45(self):
        """Test specific point: theta=45, phi=45 degrees."""
        result = sph2cart(45, 45, degree=True, r=1)

        # Analytical: x = sin(45)*cos(45) = 0.5
        #             y = sin(45)*sin(45) = 0.5
        #             z = cos(45) = sqrt(2)/2 ~ 0.707
        expected = (0.5, 0.5, np.sqrt(2) / 2)
        assert_allclose(result, expected, rtol=1e-10)

    def test_lonlat2cart_45_45(self):
        """Test specific point: lon=45, lat=45 degrees."""
        result = lonlat2cart(45, 45, degree=True, r=1)

        # theta = 90 - 45 = 45
        # x = sin(45)*cos(45) = 0.5
        # y = sin(45)*sin(45) = 0.5
        # z = cos(45) = sqrt(2)/2
        expected = (0.5, 0.5, np.sqrt(2) / 2)
        assert_allclose(result, expected, rtol=1e-10)
