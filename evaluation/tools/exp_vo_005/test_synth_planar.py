"""Known-answer tests for the EXP-VO-005 renderer's geometry (LIT-VO-003 §2): height ratio = image
scale, translation in current-frame GSD, rotation about the principal point, and that the
sequences stay inside the texture. Run alone::

    cd evaluation; python -m pytest tools/exp_vo_005 -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import synth_planar as sp  # noqa: E402


def frame_to_frame(H1, H2):
    """Image-1 -> image-2 homography from two ground->image homographies."""
    Hf = H2 @ np.linalg.inv(H1)
    return Hf / Hf[2, 2]


def test_pure_altitude_change_is_exact_height_ratio_scale_about_principal_point():
    H1 = sp.ground_to_image(0, 0, 80.0, 0, 0)
    H2 = sp.ground_to_image(0, 0, 100.0, 0, 0)
    F = frame_to_frame(H1, H2)
    assert math.isclose(F[0, 0], 0.8, rel_tol=1e-12) and math.isclose(F[1, 1], 0.8, rel_tol=1e-12)
    assert abs(F[2, 0]) < 1e-15 and abs(F[2, 1]) < 1e-15
    # principal point is the fixed point
    p = F @ np.array([sp.CX, sp.CY, 1.0]); p /= p[2]
    assert np.allclose(p[:2], [sp.CX, sp.CY])


def test_pure_north_translation_maps_to_image_up_by_f_t_over_h():
    H1 = sp.ground_to_image(0, 0, 80.0, 0, 0)
    H2 = sp.ground_to_image(0, 3.0, 80.0, 0, 0)     # 3 m north
    F = frame_to_frame(H1, H2)
    assert np.allclose(F[:2, :2], np.eye(2))
    # moving north, the ground content moves DOWN the image (image y down = backward)
    assert math.isclose(F[0, 2], 0.0, abs_tol=1e-9)
    assert math.isclose(F[1, 2], sp.F_PX * 3.0 / 80.0, rel_tol=1e-12)


def test_east_translation_with_heading_90_is_forward():
    H1 = sp.ground_to_image(0, 0, 80.0, 90, 0)
    H2 = sp.ground_to_image(3.0, 0, 80.0, 90, 0)
    F = frame_to_frame(H1, H2)
    assert math.isclose(F[1, 2], sp.F_PX * 3.0 / 80.0, rel_tol=1e-12)
    assert math.isclose(F[0, 2], 0.0, abs_tol=1e-9)


def test_pure_yaw_is_rotation_about_the_principal_point():
    H1 = sp.ground_to_image(0, 0, 80.0, 0, 0)
    H2 = sp.ground_to_image(0, 0, 80.0, 30, 0)
    F = frame_to_frame(H1, H2)
    p = F @ np.array([sp.CX, sp.CY, 1.0]); p /= p[2]
    assert np.allclose(p[:2], [sp.CX, sp.CY])
    ang = math.degrees(math.atan2(F[1, 0], F[0, 0]))
    assert math.isclose(abs(ang), 30.0, abs_tol=1e-9)
    assert math.isclose(abs(np.linalg.det(F[:2, :2])), 1.0, rel_tol=1e-12)


def test_roll_produces_perspective_of_order_theta_over_f_and_centre_scale_cos_1p5():
    H1 = sp.ground_to_image(0, 0, 80.0, 0, 0)
    H2 = sp.ground_to_image(0, 0, 80.0, 0, 10.0)
    F = frame_to_frame(H1, H2)
    assert abs(F[2, 0]) > 1e-4                       # roll about the forward axis tilts across x
    # Evaluate at the nadir-image point that lands on the ROLLED image's principal point (the
    # ground point under the rolled boresight), which is where LIT-VO-003 §4's formula applies.
    p = np.linalg.inv(F) @ np.array([sp.CX, sp.CY, 1.0]); p /= p[2]
    q = F @ p; w = q[2]; u, v = q[0] / w, q[1] / w
    J = np.array([[F[0, 0] - u * F[2, 0], F[0, 1] - u * F[2, 1]], [F[1, 0] - v * F[2, 0], F[1, 1] - v * F[2, 1]]]) / w
    # F maps nadir image -> rolled image, so its centre Jacobian is the ground->image scale ratio
    # s = cos^1.5(theta) (LIT-VO-003 §4); the footprint magnification m = 1/s of the rolled frame
    # is therefore 1/cos^1.5(theta) and its anisotropy 1/cos(theta).
    s = math.sqrt(abs(np.linalg.det(J)))
    assert math.isclose(s, math.cos(math.radians(10)) ** 1.5, rel_tol=0.02), s
    sv = np.linalg.svd(J, compute_uv=False)
    assert math.isclose(sv[0] / sv[1], 1 / math.cos(math.radians(10)), rel_tol=0.01)


def test_sequences_stay_inside_texture():
    half = sp.TEX_SIZE / 2 * sp.TEXEL_M
    for sid, spec in sp.SEQUENCES.items():
        tr = sp.trajectory(spec)
        reach = np.max(np.abs(np.stack([tr["east"], tr["north"]]))) + 0.75 * sp.footprint_extent_m(tr["up"].max())
        assert reach < half, (sid, reach, half)


def test_render_known_answer_pixel():
    tex = np.zeros((512, 512), np.uint8)
    tex[250:262, 250:262] = 255                       # a bright square at the texture centre = ground origin
    sp_TEX = sp.TEX_SIZE
    try:
        sp.TEX_SIZE = 512
        Hg2i = sp.ground_to_image(0, 0, 80.0, 0, 0)
        img = sp.render(tex, Hg2i)
    finally:
        sp.TEX_SIZE = sp_TEX
    ys, xs = np.where(img > 128)
    assert abs(xs.mean() - sp.CX) < 2 and abs(ys.mean() - sp.CY) < 2
