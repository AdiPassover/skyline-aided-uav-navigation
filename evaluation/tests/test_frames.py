"""Tests for canonical coordinate and heading conventions (DEC-004)."""

import math

import pytest

from naveval.frames import (
    compass_to_enu_math_deg,
    enu_math_to_compass_deg,
    geodetic_to_enu,
    heading_to_unit_vector,
    interpolate_heading_deg,
    ned_to_enu,
    normalize_heading_deg,
    shortest_angle_diff_deg,
    unit_vector_to_heading_deg,
)


class TestGeodeticToEnu:
    def test_origin_maps_to_zero(self):
        east, north, up = geodetic_to_enu(32.1093, 34.8555, 42.0, 32.1093, 34.8555, 42.0)
        assert east == pytest.approx(0.0, abs=1e-9)
        assert north == pytest.approx(0.0, abs=1e-9)
        assert up == pytest.approx(0.0, abs=1e-9)

    def test_one_degree_latitude_is_known_distance(self):
        # 1 degree of latitude at the WGS84 semi-major axis is 2*pi*R/360.
        east, north, up = geodetic_to_enu(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        expected_north = 2 * math.pi * 6378137.0 / 360.0
        assert north == pytest.approx(expected_north, rel=1e-9)
        assert east == pytest.approx(0.0, abs=1e-9)

    def test_longitude_scales_by_cos_latitude(self):
        # At 60 deg latitude, cos(60)=0.5, so one degree of longitude covers half the
        # ground distance it would at the equator.
        east_at_equator, _, _ = geodetic_to_enu(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        east_at_60, _, _ = geodetic_to_enu(60.0, 1.0, 0.0, 60.0, 0.0, 0.0)
        assert east_at_60 == pytest.approx(east_at_equator * 0.5, rel=1e-6)

    def test_altitude_is_direct_difference(self):
        _, _, up = geodetic_to_enu(0.0, 0.0, 142.5, 0.0, 0.0, 100.0)
        assert up == pytest.approx(42.5, abs=1e-9)


class TestNedToEnu:
    def test_axis_relabelling_and_sign(self):
        # NED (north=5, east=3, down=-2) -> ENU (east=3, north=5, up=2).
        east, north, up = ned_to_enu(north_m=5.0, east_m=3.0, down_m=-2.0)
        assert (east, north, up) == (3.0, 5.0, 2.0)

    def test_down_positive_becomes_up_negative(self):
        east, north, up = ned_to_enu(north_m=0.0, east_m=0.0, down_m=10.0)
        assert up == -10.0

    def test_handedness_not_swapped(self):
        # A NED north-only motion must remain a pure-north ENU motion, not east.
        east, north, up = ned_to_enu(north_m=7.0, east_m=0.0, down_m=0.0)
        assert east == 0.0
        assert north == 7.0


class TestHeadingConventions:
    def test_north_is_zero(self):
        east, north = heading_to_unit_vector(0.0)
        assert east == pytest.approx(0.0, abs=1e-9)
        assert north == pytest.approx(1.0, abs=1e-9)

    def test_east_is_ninety(self):
        east, north = heading_to_unit_vector(90.0)
        assert east == pytest.approx(1.0, abs=1e-9)
        assert north == pytest.approx(0.0, abs=1e-9)

    def test_south_is_180(self):
        east, north = heading_to_unit_vector(180.0)
        assert east == pytest.approx(0.0, abs=1e-9)
        assert north == pytest.approx(-1.0, abs=1e-9)

    def test_west_is_270(self):
        east, north = heading_to_unit_vector(270.0)
        assert east == pytest.approx(-1.0, abs=1e-9)
        assert north == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("heading", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 359.9])
    def test_round_trip_through_unit_vector(self, heading):
        east, north = heading_to_unit_vector(heading)
        recovered = unit_vector_to_heading_deg(east, north)
        assert recovered == pytest.approx(heading, abs=1e-6)

    def test_compass_to_enu_math_is_self_inverse(self):
        for h in [0.0, 30.0, 90.0, 200.0, 359.0]:
            once = compass_to_enu_math_deg(h)
            twice = enu_math_to_compass_deg(once)
            assert twice == pytest.approx(normalize_heading_deg(h), abs=1e-9)

    def test_compass_north_is_enu_math_ninety(self):
        # North (compass 0) is +Y, which is 90 deg in ENU-math (CCW from East/+X).
        assert compass_to_enu_math_deg(0.0) == pytest.approx(90.0, abs=1e-9)

    def test_compass_east_is_enu_math_zero(self):
        assert compass_to_enu_math_deg(90.0) == pytest.approx(0.0, abs=1e-9)


class TestNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0.0, 0.0),
            (359.999, 359.999),
            (360.0, 0.0),
            (360.5, 0.5),
            (720.0, 0.0),
            (-0.5, 359.5),
            (-360.0, 0.0),
            (-720.5, 359.5),
        ],
    )
    def test_normalize_range(self, raw, expected):
        assert normalize_heading_deg(raw) == pytest.approx(expected, abs=1e-9)

    def test_normalize_never_returns_360(self):
        assert normalize_heading_deg(360.0) < 360.0
        assert normalize_heading_deg(-1e-9) < 360.0

    def test_normalize_never_returns_360_at_float_rounding_boundary(self):
        # Regression: for an input whose true value mod 360 is a tiny negative number
        # very close to zero, Python's `%` can round the result UP to exactly 360.0
        # rather than just under it, because 360.0's representable precision (~5.7e-14)
        # is coarser than the residual. -1e-9 (the case above) is not small enough to
        # trigger this; -1e-14 and smaller are. Found via alignment.py's rotation_deg
        # computation hitting this exact case during development.
        for tiny_negative in (-1e-13, -1e-14, -1e-15, -1e-16, -1e-300):
            result = normalize_heading_deg(tiny_negative)
            assert result < 360.0, f"normalize_heading_deg({tiny_negative}) returned {result}"


class TestShortestAngleDiff:
    def test_simple_case(self):
        assert shortest_angle_diff_deg(30.0, 10.0) == pytest.approx(20.0, abs=1e-9)

    def test_wrap_forward(self):
        # 1 deg vs 359 deg: true difference is 2 deg, not 358.
        assert shortest_angle_diff_deg(1.0, 359.0) == pytest.approx(2.0, abs=1e-9)

    def test_wrap_backward(self):
        assert shortest_angle_diff_deg(359.0, 1.0) == pytest.approx(-2.0, abs=1e-9)

    def test_antipodal_is_180(self):
        diff = shortest_angle_diff_deg(180.0, 0.0)
        assert abs(diff) == pytest.approx(180.0, abs=1e-9)

    def test_zero_difference(self):
        assert shortest_angle_diff_deg(200.0, 200.0) == pytest.approx(0.0, abs=1e-9)

    def test_result_always_in_range(self):
        for a in range(0, 360, 17):
            for b in range(0, 360, 23):
                diff = shortest_angle_diff_deg(float(a), float(b))
                assert -180.0 < diff <= 180.0


class TestHeadingInterpolation:
    def test_interpolate_no_wrap(self):
        assert interpolate_heading_deg(10.0, 30.0, 0.5) == pytest.approx(20.0, abs=1e-9)

    def test_interpolate_at_endpoints(self):
        assert interpolate_heading_deg(10.0, 30.0, 0.0) == pytest.approx(10.0, abs=1e-9)
        assert interpolate_heading_deg(10.0, 30.0, 1.0) == pytest.approx(30.0, abs=1e-9)

    def test_interpolate_across_wrap_takes_shortest_arc(self):
        # 350 -> 10 (through 0/360): the shortest arc is 20 deg wide, so the midpoint
        # is 0 deg -- NOT 180 deg, which is what naive linear interpolation would give.
        mid = interpolate_heading_deg(350.0, 10.0, 0.5)
        assert mid == pytest.approx(0.0, abs=1e-6)

    def test_interpolate_across_wrap_other_direction(self):
        mid = interpolate_heading_deg(10.0, 350.0, 0.5)
        assert mid == pytest.approx(0.0, abs=1e-6)

    def test_interpolate_result_always_normalized(self):
        result = interpolate_heading_deg(350.0, 10.0, 0.75)
        assert 0.0 <= result < 360.0
