"""Known-answer tests for the UE→project conversion seam (Part A, "camera orientation verification").

The brief's instruction is the point of this file: *do not merely trust the field names*. Every test
here fixes a pose whose answer is known before the code runs — a North-facing identity orientation, a
metre that is a hundred centimetres, a quaternion that is a rotation of a known angle about a known
axis — and checks the conversion against it rather than against another part of the same code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hsreloc.simret.conventions import (AXES_X_NORTH_Y_EAST, ConventionError, SimConventions,
                                        cm_to_m, identify_quaternion_convention, normalise_0_360,
                                        quaternion_angle_deg, quaternion_conjugate,
                                        quaternion_norm_error, ue_attitude_to_enu,
                                        ue_position_to_enu, ue_rotator_to_quaternion,
                                        zyx_right_handed_quaternion)

C = SimConventions()


# -- unit conversion ---------------------------------------------------------------------------

def test_cm_to_m_is_a_division_by_one_hundred():
    assert cm_to_m(100.0) == 1.0
    assert cm_to_m(-2550.0) == -25.5
    assert cm_to_m(0.0) == 0.0


def test_position_conversion_is_the_declared_axis_swap():
    # UE +X = North, +Y = East, +Z = Up (PROT-001 §2.2). 12 345 cm North, 6 700 cm East, 8 000 cm up.
    east, north, up = ue_position_to_enu(12_345.0, 6_700.0, 8_000.0, C)
    assert (east, north, up) == (67.0, 123.45, 80.0)


def test_altitude_datum_is_subtracted_before_conversion():
    conv = SimConventions(altitude_datum_ue_cm=5_000.0)
    _, _, up = ue_position_to_enu(0.0, 0.0, 11_000.0, conv)
    assert up == pytest.approx(60.0)


def test_axis_swap_preserves_right_handedness():
    """East x North = Up. If the mapping were an even permutation this would come out negative."""
    def enu(x, y, z):
        return np.array(ue_position_to_enu(x, y, z, C))
    east = enu(0, 100, 0)          # +1 m East
    north = enu(100, 0, 0)         # +1 m North
    up = enu(0, 0, 100)            # +1 m Up
    assert np.allclose(np.cross(east, north), up)


def test_unsupported_axis_declaration_is_refused_with_the_reason():
    with pytest.raises(ConventionError, match="reflection, not a relabelling"):
        SimConventions(ue_world_axes="x_east_y_north").validate()


# -- attitude ----------------------------------------------------------------------------------

def test_north_facing_identity_orientation():
    """A world-locked-North, level camera: UE yaw 0 → compass 0, pitch 0, roll 0."""
    yaw, pitch, roll = ue_attitude_to_enu(0.0, 0.0, 0.0, C)
    assert (yaw, pitch, roll) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("ue_yaw,expected", [(0.0, 0.0), (90.0, 90.0), (180.0, 180.0),
                                             (-90.0, 270.0), (359.9, 359.9), (360.0, 0.0),
                                             (450.0, 90.0)])
def test_yaw_is_compass_clockwise_from_north(ue_yaw, expected):
    """UE yaw 0 faces +X = North and increases toward +Y = East, i.e. clockwise. No offset, no flip."""
    assert ue_attitude_to_enu(ue_yaw, 0.0, 0.0, C)[0] == pytest.approx(expected)
    assert 0.0 <= normalise_0_360(ue_yaw) < 360.0


def test_pitch_and_roll_pass_through_under_the_declared_signs():
    _, pitch, roll = ue_attitude_to_enu(0.0, 5.0, -3.0, C)
    assert (pitch, roll) == (5.0, -3.0)
    assert C.pitch_sign == "positive_up" and C.roll_sign == "positive_clockwise_image"
    assert C.attitude_verified is False          # declared, NOT verified — no R0-S render exists


# -- quaternions -------------------------------------------------------------------------------

def test_identity_rotator_is_the_identity_quaternion():
    assert np.allclose(ue_rotator_to_quaternion(0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(zyx_right_handed_quaternion(0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 0.0])


def test_pure_yaw_is_a_rotation_about_z_of_the_same_angle():
    q = ue_rotator_to_quaternion(90.0, 0.0, 0.0)
    assert np.allclose(q, [math.cos(math.radians(45)), 0.0, 0.0, math.sin(math.radians(45))])
    assert quaternion_angle_deg(q, [1, 0, 0, 0]) == pytest.approx(90.0, abs=1e-9)


@pytest.mark.parametrize("angle", [1.0, 5.0, 30.0, 90.0])
def test_quaternion_angle_matches_the_rotator_angle_on_each_single_axis(angle):
    for yaw, pitch, roll in ((angle, 0, 0), (0, angle, 0), (0, 0, angle)):
        q = ue_rotator_to_quaternion(yaw, pitch, roll)
        assert quaternion_angle_deg(q, [1, 0, 0, 0]) == pytest.approx(angle, abs=1e-9)


def test_quaternions_are_generated_normalised_and_norm_error_detects_when_they_are_not():
    q = ue_rotator_to_quaternion(37.0, -6.0, 11.0)
    assert quaternion_norm_error(q) == pytest.approx(0.0, abs=1e-12)
    assert quaternion_norm_error(np.array(q) * 1.05) == pytest.approx(0.05, abs=1e-9)
    assert not math.isfinite(quaternion_norm_error([np.nan, 0, 0, 0]))


def test_sign_flip_is_the_same_rotation():
    q = ue_rotator_to_quaternion(23.0, 4.0, -7.0)
    assert quaternion_angle_deg(q, -np.asarray(q)) == pytest.approx(0.0, abs=1e-9)


def test_convention_identification_finds_the_generating_convention():
    poses = [(10.0, 3.0, -2.0), (-45.0, 12.0, 6.0), (170.0, -8.0, 15.0)]
    samples = [(y, p, r, ue_rotator_to_quaternion(y, p, r)) for y, p, r in poses]
    out = identify_quaternion_convention(samples)
    assert out["identified"] == "ue_frotator"
    assert out["discriminating"] is True
    assert out["ambiguous"] is False
    assert out["residual_deg"]["ue_frotator"] == pytest.approx(0.0, abs=1e-9)


def test_convention_identification_finds_a_conjugated_export():
    poses = [(10.0, 3.0, -2.0), (-45.0, 12.0, 6.0)]
    samples = [(y, p, r, quaternion_conjugate(ue_rotator_to_quaternion(y, p, r))) for y, p, r in poses]
    out = identify_quaternion_convention(samples)
    assert out["identified"] == "ue_frotator_conjugate"


def test_convention_identification_distinguishes_ue_from_textbook_zyx():
    poses = [(30.0, 20.0, 10.0)]
    samples = [(y, p, r, zyx_right_handed_quaternion(y, p, r)) for y, p, r in poses]
    out = identify_quaternion_convention(samples)
    assert out["identified"] == "zyx_right_handed"
    assert out["residual_deg"]["ue_frotator"] > 1.0        # the two really are different rotations


def test_identity_only_attitudes_cannot_discriminate_and_say_so():
    """The normal state of a level, north-locked Stage-0/1 run: every convention fits, so none is
    established. The adapter must report that rather than claim a verification."""
    samples = [(0.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0, 0.0]))] * 5
    out = identify_quaternion_convention(samples)
    assert out["ambiguous"] is True
    assert out["discriminating"] is False
    assert len(out["fits"]) == 4


def test_no_convention_fits_a_corrupted_quaternion():
    samples = [(10.0, 3.0, -2.0, np.array([0.0, 1.0, 0.0, 0.0]))]
    out = identify_quaternion_convention(samples)
    assert out["fits"] == []
    assert out["identified"] is None


def test_conventions_round_trip_and_reject_unknown_keys():
    conv = SimConventions(altitude_datum_ue_cm=100.0, verified_by="R0-S1")
    d = conv.as_dict()
    assert d["frame_convention"] == "ENU"
    assert d["heading_convention"] == "compass_cw_from_north"
    assert d["attitude_verified"] is True
    assert d["ue_world_axes"] == AXES_X_NORTH_Y_EAST
    assert SimConventions.from_dict({"altitude_datum_ue_cm": 5.0}).altitude_datum_ue_cm == 5.0
    with pytest.raises(ConventionError, match="unknown convention keys"):
        SimConventions.from_dict({"nonsense": 1})
