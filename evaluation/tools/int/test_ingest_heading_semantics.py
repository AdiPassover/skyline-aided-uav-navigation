"""The body-yaw / nadir-camera-heading distinction, pinned on the INT ingest's own checks.

Every 2026-09-07 UE5 recording logs a bit-identical ``heading_deg`` while the airframe visibly
turns (``yaw.mode = follow_movement``). The heading INT consumes is defined as the **nadir camera's
image-up azimuth**, and the camera is ``world_stabilized_nadir_independent_yaw``, so a constant
heading is the correct export *if and only if* the camera's world orientation really is constant.
``ingest_ue_run_int.verify_heading_aware`` decides that from the raw quaternions: C4' derives the
image-up axis from ``cam_ue_q*`` and C5 requires the logged ``heading_deg`` to equal its azimuth at
every frame, while the body quaternion is only reported (``body_yaw_deg_range``,
``camera_body_yaw_is_fixed_offset``) and never used as a heading.

These tests build the three cases the audit distinguishes, on a synthetic record that satisfies
every other adapter check, and assert the ingest's verdicts:

* **A** — body rotates, camera world-fixed, ``heading_deg`` constant → accepted, and the record says
  the camera and the body are *not* a fixed offset apart (so body yaw could never stand in);
* **B** — the camera rotates in the world but ``heading_deg`` stays constant → refused (an exporter
  that froze the heading column would be caught here);
* **C** — ``heading_deg`` follows the *airframe* while the camera stays fixed (the ``EXP-VO-013`` R6
  error) → refused.

Run from the repository root: ``python -m pytest evaluation/tools/int -q``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ingest_ue_run_int as ingest  # noqa: E402

vo = ingest.vo

N_FRAMES = 12
GT_HZ = 60
FRAME_EVERY = 6            # 10 Hz frames on the 60 Hz telemetry grid
H0_AGL_M = 70.0


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _q_yaw(psi_deg: float):
    h = math.radians(psi_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


# The nadir camera frame of the recordings: optical +X -> world down, image-up +Z -> world North.
NADIR_NORTH = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


def _record(camera_yaw_deg, body_yaw_deg, heading_deg, tmp_path: Path):
    """A minimal frames/groundtruth/settings triple that passes C1-C3 and C6-C12 by construction."""
    n_gt = (N_FRAMES - 1) * FRAME_EVERY + 1
    t_gt = np.arange(n_gt) / GT_HZ
    # A gentle curve so the body has a reason to turn; camera 0.3 m ahead of the body.
    east = 2.0 * t_gt
    north = 0.5 * t_gt ** 2
    up = 100.0 + 0.1 * t_gt
    agl = H0_AGL_M + 0.1 * t_gt * 0.5
    terr = up - agl
    baro = up - up[0]
    gt = [{"sim_time_s": float(t_gt[i]), "east_m": float(east[i]), "north_m": float(north[i]),
           "up_m": float(up[i]), "true_agl_m": float(agl[i]), "baro_relative_alt_m": float(baro[i])}
          for i in range(n_gt)]
    frames = []
    images_dir = tmp_path / "vo" / "images"
    images_dir.mkdir(parents=True)
    for k in range(N_FRAMES):
        i = k * FRAME_EVERY
        qc = _qmul(_q_yaw(camera_yaw_deg[k]), NADIR_NORTH)
        qb = _q_yaw(body_yaw_deg[k])
        name = f"vo/images/frame_{k:06d}.png"
        (tmp_path / name).write_bytes(b"")
        frames.append({
            "frame_id": k, "sim_time_s": float(t_gt[i]), "image_path": name,
            "cam_ue_x_cm": float(north[i] * 100), "cam_ue_y_cm": float(east[i] * 100), "cam_ue_z_cm": float(up[i] * 100),
            "cam_ue_qw": qc[0], "cam_ue_qx": qc[1], "cam_ue_qy": qc[2], "cam_ue_qz": qc[3],
            "body_ue_x_cm": float(north[i] * 100 - 30.0), "body_ue_y_cm": float(east[i] * 100), "body_ue_z_cm": float(up[i] * 100),
            "body_ue_qw": qb[0], "body_ue_qx": qb[1], "body_ue_qy": qb[2], "body_ue_qz": qb[3],
            "east_m": float(east[i]), "north_m": float(north[i]), "up_m": float(up[i]),
            "heading_deg": float(heading_deg[k]), "camera_tilt_deg": 0.0,
            "baro_relative_alt_m": float(baro[i]), "true_agl_m": float(agl[i]),
            "terrain_elevation_m": float(terr[i]), "ground_hit": 1,
        })
    settings = {
        "world_frame": {"enu_mapping": vo.EXPECTED_ENU_MAPPING, "enu_position_source": vo.EXPECTED_POSITION_SOURCE,
                        "ue_units": "cm", "enu_handedness": "right"},
        "quaternion_convention": {"order": vo.EXPECTED_QUAT_ORDER},
        "height": {"h0_agl_m": H0_AGL_M},
    }
    return settings, frames, gt, images_dir


BODY_SWEEP = np.linspace(-170.0, 170.0, N_FRAMES)      # the airframe turns through 340 degrees
CAMERA_FIXED = np.zeros(N_FRAMES)                       # the nadir camera holds image-up = North
HEADING_CONSTANT = np.zeros(N_FRAMES)


def test_case_a_body_rotates_camera_world_fixed_constant_heading_is_correct(tmp_path):
    settings, frames, gt, images = _record(CAMERA_FIXED, BODY_SWEEP, HEADING_CONSTANT, tmp_path)
    out = ingest.verify_heading_aware(settings, frames, gt, images)
    assert out["camera_quaternion_unique_rows"] == 1
    assert out["heading_vs_camera_quaternion_max_deg"] < vo.TOL_DEG
    assert out["camera_heading_deg_range"] == [0.0, 0.0]
    lo, hi = out["body_yaw_deg_range"]
    assert hi - lo > 300.0, "the body must actually have turned for this case to mean anything"
    assert out["camera_body_yaw_is_fixed_offset"] is False, \
        "body yaw and camera heading are unrelated here; no mounting offset could map one onto the other"


def test_case_b_camera_rotates_but_heading_column_is_frozen_is_refused(tmp_path):
    camera_sweep = np.linspace(0.0, 90.0, N_FRAMES)
    settings, frames, gt, images = _record(camera_sweep, BODY_SWEEP, HEADING_CONSTANT, tmp_path)
    with pytest.raises(vo.ConventionError, match="heading_deg is not the camera image-up azimuth"):
        ingest.verify_heading_aware(settings, frames, gt, images)


def test_case_c_heading_following_the_airframe_while_the_camera_holds_is_refused(tmp_path):
    settings, frames, gt, images = _record(CAMERA_FIXED, BODY_SWEEP, BODY_SWEEP % 360.0, tmp_path)
    with pytest.raises(vo.ConventionError, match="AIRFRAME yaw"):
        ingest.verify_heading_aware(settings, frames, gt, images)


def test_a_camera_that_really_turns_is_accepted_when_heading_follows_it(tmp_path):
    # The converse of case B: a varying heading_deg is fine exactly when the camera quaternion
    # says the image-up axis varied the same way. This is the recording the lane still lacks.
    camera_sweep = np.linspace(10.0, 80.0, N_FRAMES)
    settings, frames, gt, images = _record(camera_sweep, BODY_SWEEP, camera_sweep, tmp_path)
    out = ingest.verify_heading_aware(settings, frames, gt, images)
    assert out["camera_quaternion_unique_rows"] == N_FRAMES
    assert out["heading_vs_camera_quaternion_max_deg"] < vo.TOL_DEG
    assert out["camera_heading_deg_range"] == pytest.approx([10.0, 80.0])
