"""Known-answer tests for `EXP-VO-013`'s tooling, and the gates it declared in advance.

The ones that matter most are the negative-result guards. This experiment's headline B1 finding is
that a channel is empty, and the first version of the probe produced exactly that finding from a
broken reader — so the tests here pin the *reader*, not just the conclusion.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp_vo_012"))
sys.path.insert(0, str(HERE.parents[1]))

import altitude_sources as als                                           # noqa: E402
import metric_readout as mr                                              # noqa: E402


def _load(name: str, path: Path):
    """Import by path under a UNIQUE module name.

    `exp_vo_007`, `exp_vo_012` and `exp_vo_013` each ship an `analyse.py`, and a plain
    `import analyse` inside one pytest session resolves to whichever was imported first — so this
    file was silently testing another experiment's module until the name was made unique.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


an = _load("exp_vo_013_analyse", HERE / "analyse.py")


# ---------------------------------------------------------------------------- B1: classification


def test_an_inert_height_above_takeoff_is_class_e_not_class_b():
    """The whole B1 result. A channel that never leaves a metre-wide band across a flight that
    reached 80 m is UNUSABLE — not "a barometer reading nearly zero".

    The values here are the ones `AMtown01` actually carries: a 0.23 m pre-take-off offset, then
    micrometre noise. An exact-zero test would call this *populated* and classify it B, which is how
    a dead channel gets mistaken for a live one.
    """
    letter, why = als.classify("/dji_osdk_ros/height_above_takeoff",
                               [0.232, 1e-6, -1.8e-6, 0.0, 2e-7])
    assert letter == "E"
    assert "NEVER POPULATED" in why


def test_a_genuinely_populated_height_above_takeoff_is_class_b_not_class_a():
    """Even when it works it is a FUSED relative altitude. Calling it barometric would be an
    unsupported claim about DJI's internals, which this work forbids itself."""
    letter, why = als.classify("/dji_osdk_ros/height_above_takeoff", [0.0, 40.0, 80.0])
    assert letter == "B"
    assert "FUSED" in why


def test_rtk_altitude_is_never_labelled_barometer():
    letter, why = als.classify("/dji_osdk_ros/rtk_position", [1074.0, 1154.0])
    assert letter == "C"
    assert "NOT a barometer" in why


def test_lidar_is_the_agl_reference():
    assert als.classify("/livox/lidar", [80.0, 40.0])[0] == "D"


def test_local_position_is_classified_from_measurement_not_from_its_name():
    letter, why = als.classify("/dji_osdk_ros/local_position", [1060.0, 1140.0])
    assert letter == "C"
    assert "gps_position" in why


# ---------------------------------------------------------------------------- B1: the decoders


def _cdr(body: bytes) -> bytes:
    """A little-endian CDR encapsulation header plus a body."""
    return b"\x00\x01\x00\x00" + body


def _header(sec: int = 7, nsec: int = 0, frame_id: str = "b") -> bytes:
    """std_msgs/Header in CDR: int32 sec, uint32 nsec, then a length-prefixed string whose length
    includes the trailing NUL, which is how ROS 2 serialises it."""
    fid = frame_id.encode() + b"\x00"
    return struct.pack("<iI", sec, nsec) + struct.pack("<I", len(fid)) + fid


def _align8(body: bytes) -> bytes:
    """Pad so the next float64 is 8-aligned.

    XCDR1 aligns each primitive to its own size **relative to the start of the body**, i.e. after
    the 4-byte encapsulation header — so the condition is `len(body) % 8 == 0`, not
    `(4 + len(body)) % 8 == 0`. Aligning against the absolute offset instead reads eight bytes
    straddling two fields and returns a denormal, which is what this helper did on its first try.
    """
    return body + b"\x00" * ((-len(body)) % 8)


def test_decode_by_schema_reads_point_stamped_z_and_not_x():
    body = _align8(_header()) + struct.pack("<ddd", 11.0, 22.0, 33.0)
    v, what = als.decode_by_schema("geometry_msgs/msg/PointStamped", _cdr(body))
    assert v == pytest.approx(33.0)
    assert "z" in what


def test_decode_by_schema_reads_float32():
    v, _ = als.decode_by_schema("std_msgs/msg/Float32", _cdr(struct.pack("<f", 1.25)))
    assert v == pytest.approx(1.25)


def test_decode_by_schema_returns_none_for_a_type_with_no_height():
    """`None` means "this type carries no height", which is a different statement from "the value is
    zero" and must never be collapsed into it."""
    v, what = als.decode_by_schema("std_msgs/msg/UInt8", _cdr(b"\x32"))
    assert v is None
    assert "not a height" in what


def test_parse_schemas_reads_the_declared_type_names():
    """An MCAP Schema record: opcode 3, uint64 length, then id / name / encoding / data.

    MCAP strings are `<uint32 length><bytes>` with NO trailing NUL — a different encoding from the
    CDR strings inside a message payload, which is exactly the sort of detail that makes decoding
    by guess unsafe.
    """
    name = b"geometry_msgs/msg/PointStamped"
    enc = b"ros2msg"
    body = (struct.pack("<H", 9) + struct.pack("<I", len(name)) + name
            + struct.pack("<I", len(enc)) + enc + struct.pack("<I", 0))
    buf = bytes([3]) + struct.pack("<Q", len(body)) + body
    assert als.parse_schemas(buf) == {9: "geometry_msgs/msg/PointStamped"}


# ---------------------------------------------------------------------------- B2: window search


def _profile(t, up, cum):
    return {"_series": {"t": list(t), "up": list(up), "cum": list(cum)}}


def test_a_climb_in_place_is_rejected_by_the_path_criterion():
    """The measured shape of every MARS-LVIG flight: a big altitude ratio over almost no path.

    This is the criterion that decided the experiment's design, so it is pinned: a window must not
    qualify on altitude ratio alone.
    """
    t = np.arange(0, 200, 1.0)
    up = np.clip(t * 1.0, 20, 130)          # 20 -> 130 m, ratio 6.5
    cum = t * 0.3                            # 60 m of path in total
    w = als.best_altitude_windows(_profile(t, up, cum), min_path_m=200.0, min_ratio=1.25,
                                  min_duration_s=60.0, min_altitude_m=15.0)
    assert w == []


def test_a_climb_with_real_horizontal_travel_is_accepted():
    t = np.arange(0, 400, 1.0)
    up = np.clip(20 + t * 0.3, 20, 130)
    cum = t * 4.0                            # 1.6 km of path
    w = als.best_altitude_windows(_profile(t, up, cum), min_path_m=200.0, min_ratio=1.25,
                                  min_duration_s=60.0, min_altitude_m=15.0)
    assert w and w[0]["alt_ratio"] > 1.25 and w[0]["path_m"] >= 200.0


def test_windows_below_the_altitude_floor_are_rejected():
    """Below a few metres the ratio explodes for arithmetic reasons, and the imagery is a different
    problem rather than a harder instance of the same one."""
    t = np.arange(0, 400, 1.0)
    up = np.clip(0.5 + t * 0.3, 0.5, 130)
    cum = t * 4.0
    w = als.best_altitude_windows(_profile(t, up, cum), min_path_m=200.0, min_ratio=1.25,
                                  min_duration_s=60.0, min_altitude_m=15.0)
    assert all(x["alt_min_m"] >= 15.0 for x in w)


# ---------------------------------------------------------------------------- the closed form


def test_closed_form_is_path_weighted_not_time_weighted():
    """Half the path at ratio 2, half at ratio 1, with the ratio-2 half taking 9x longer in samples.

    A sample-weighted mean gives ~1.9; the path-weighted mean is 1.5. `EXP-VO-012` H9's
    pre-registered prose used the wrong one and under-predicted by a third, which is why this is a
    test rather than a comment.
    """
    n_slow, n_fast = 90, 10
    gt = np.zeros((n_slow + n_fast, 2))
    gt[:n_slow, 0] = np.linspace(0, 100, n_slow)          # 100 m over 90 samples
    gt[n_slow:, 0] = np.linspace(100, 200, n_fast)        # 100 m over 10 samples
    h_true = np.ones(n_slow + n_fast)
    h_used = np.concatenate([np.full(n_slow, 2.0), np.ones(n_fast)])
    out = an.closed_form_prediction(h_used, h_true, gt)
    assert out["path_weighted_mean_h_used_over_h_true"] == pytest.approx(1.5, abs=0.03)
    assert out["unweighted_mean_ratio"] == pytest.approx(1.9, abs=0.03)


def test_closed_form_is_exactly_one_when_the_height_is_right():
    gt = np.stack([np.linspace(0, 100, 50), np.zeros(50)], axis=1)
    h = np.linspace(40, 90, 50)
    out = an.closed_form_prediction(h, h, gt)
    assert out["path_weighted_mean_h_used_over_h_true"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------- gates G3, G5, G6


def _increments(n=200, seed=3):
    rng = np.random.default_rng(seed)
    return mr.Increments(
        frame_index=np.arange(n),
        events=["init"] + ["none"] * (n - 1),
        dq=np.concatenate([np.zeros((1, 2)), rng.normal(0, 5, (n - 1, 2))]),
        dtheta=np.concatenate([[0.0], rng.normal(0, 1e-3, n - 1)]),
        inc_log_scale=rng.normal(0, 1e-4, n),
        timestamps_s=np.arange(n) * 0.1,
    )


def test_g5_the_arms_differ_only_in_the_height_array():
    """The same increments under a doubled height must give an exactly doubled trajectory, and an
    unchanged yaw — no other term may depend on the height."""
    inc = _increments()
    n = len(inc)
    a = mr.integrate_metric(inc, np.full(n, 80.0), 700.0, arm="a")
    b = mr.integrate_metric(inc, np.full(n, 160.0), 700.0, arm="b")
    assert np.allclose(b.east_m, 2.0 * a.east_m, rtol=0, atol=1e-12)
    assert np.allclose(b.north_m, 2.0 * a.north_m, rtol=0, atol=1e-12)
    assert np.array_equal(b.yaw_deg, a.yaw_deg)


def test_g6_the_visual_scale_channel_has_zero_authority():
    """`DEC-VO-007` D4, enforced rather than intended: corrupting the visual-scale series must not
    move a single output bit."""
    inc = _increments()
    n = len(inc)
    h = np.linspace(40, 90, n)
    base = mr.integrate_metric(inc, h, 700.0)
    inc.inc_log_scale[:] = 1e6
    other = mr.integrate_metric(inc, h, 700.0)
    assert np.array_equal(base.east_m, other.east_m)
    assert np.array_equal(base.north_m, other.north_m)
    assert np.array_equal(base.yaw_deg, other.yaw_deg)


def test_g3_h0_comes_from_the_lidar_reading_not_from_ground_truth():
    """`h₀` must be a sensor reading at the reference frame. Shifting ground-truth altitude must not
    move it; changing the LiDAR reading must."""
    frame_t = np.linspace(0, 10, 21)
    gt_t = np.linspace(0, 10, 11)
    gt_up = np.full(11, 80.0)
    lt = np.linspace(0, 10, 6)
    lh = np.array([70.0, 60.0, 50.0, 55.0, 65.0, 75.0])
    _h1, p1 = an.arm_heights(frame_t, gt_t, gt_up, lt, lh)
    assert p1["h0_m"] == pytest.approx(70.0)
    _h2, p2 = an.arm_heights(frame_t, gt_t, gt_up + 500.0, lt, lh)
    assert p2["h0_m"] == pytest.approx(70.0)
    _h3, p3 = an.arm_heights(frame_t, gt_t, gt_up, lt, lh * 2)
    assert p3["h0_m"] == pytest.approx(140.0)


def test_external_arm_is_the_takeoff_relative_difference_not_the_absolute_altitude():
    """`h₀ + Δh`, never `h₀ + h`. Getting this wrong would add ~1,074 m of geoid to every frame."""
    frame_t = np.linspace(0, 10, 21)
    gt_t = np.linspace(0, 10, 11)
    gt_up = np.linspace(80.0, 100.0, 11)
    lt = np.linspace(0, 10, 6)
    lh = np.full(6, 70.0)
    h, p = an.arm_heights(frame_t, gt_t, gt_up, lt, lh)
    assert h["external"][0] == pytest.approx(70.0)         # equals h0 at the reference frame
    assert h["external"][-1] == pytest.approx(90.0)        # h0 + 20 m of climb
    assert p["dh_external_ptp_m"] == pytest.approx(20.0)


def test_fixed_arm_is_constant_and_equals_h0():
    frame_t = np.linspace(0, 10, 21)
    h, p = an.arm_heights(frame_t, np.linspace(0, 10, 11), np.linspace(80, 100, 11),
                          np.linspace(0, 10, 6), np.full(6, 70.0))
    assert np.all(h["fixed"] == p["h0_m"])


def test_lidar_profile_loads_both_shapes_and_sorts():
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "a.json").write_text(json.dumps({"_series": {"t": [3.0, 1.0, 2.0],
                                                      "agl": [30.0, 10.0, 20.0]}}))
    t, h = an.load_lidar_profile(d / "a.json")
    assert list(t) == [1.0, 2.0, 3.0]
    assert list(h) == [10.0, 20.0, 30.0]

    (d / "b.json").write_text(json.dumps({"t": [2.0, 1.0], "height_m": [20.0, 10.0]}))
    t, h = an.load_lidar_profile(d / "b.json")
    assert list(t) == [1.0, 2.0] and list(h) == [10.0, 20.0]
