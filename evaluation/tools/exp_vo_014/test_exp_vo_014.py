"""Regression tests for `EXP-VO-014`'s UE ingest adapter and its metric arms.

Two kinds of test here, and the second kind is the one that matters.

**Known-answer tests** on a tiny synthetic UE run built in `_fixture()`: does the adapter carry the
numbers through unchanged, and does it compute the one thing it does compute (the take-off ground
datum subtraction) correctly?

**Negative tests**: does it *refuse* when a convention changes? A dataset adapter that silently
accepts a flipped axis, a body-yaw heading column or a moved datum is worse than no adapter, because
every downstream number stays plausible. Each `test_rejects_*` corrupts exactly one thing in an
otherwise valid fixture and asserts the build fails. `EXP-VO-013` R6 is the reason `test_rejects_
body_yaw_as_heading` exists: conflating camera yaw with airframe yaw cost ~66 degrees from the first
metre on real data, and nothing downstream noticed.

    cd evaluation && python -m pytest tools/exp_vo_014 -q
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp_vo_012"))

import ingest_ue_run as ing                                              # noqa: E402
import metric_readout as mro                                             # noqa: E402
import metric_metrics as mm                                              # noqa: E402

# The real capture's constant nadir camera quaternion: +X optical -> world down, +Y -> East,
# +Z -> North. Reproduced here so the fixture exercises the same axis check the real data does.
CAM_Q = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)

FRAME_COLS = ["frame_id", "sim_time_s", "image_path",
              "cam_ue_x_cm", "cam_ue_y_cm", "cam_ue_z_cm",
              "cam_ue_qw", "cam_ue_qx", "cam_ue_qy", "cam_ue_qz",
              "body_ue_x_cm", "body_ue_y_cm", "body_ue_z_cm",
              "body_ue_qw", "body_ue_qx", "body_ue_qy", "body_ue_qz",
              "east_m", "north_m", "up_m", "heading_deg", "camera_tilt_deg",
              "baro_relative_alt_m", "true_agl_m", "terrain_elevation_m", "ground_hit"]


def _settings(h0: float) -> dict:
    return {
        "run_id": "Run_TEST", "level": "test_level", "engine_version": "5.4.4-test",
        "world_frame": {"type": "local_non_georeferenced", "ue_units": "cm",
                        "ue_handedness": "left", "north_axis": "+X", "east_axis": "+Y",
                        "up_axis": "+Z", "enu_handedness": "right",
                        "enu_position_source": "nadir_camera_center",
                        "enu_mapping": dict(ing.EXPECTED_ENU_MAPPING),
                        "vertical_datum": "UE world Z = 0", "georeference": None},
        "quaternion_convention": {"order": "wxyz", "rotation_sense": "component_local_to_UE_world",
                                  "expressed_in": "UE_world", "normalised": True},
        "path": {"path_id": "PathT", "cruise_speed_mps": 7.5},
        "camera_intrinsics": {"fx": 512.0, "fy": 512.0, "cx": 512.0, "cy": 512.0,
                              "k1": 0.0, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0,
                              "width": 1024, "height": 1024, "source": "test"},
        "capture": {"pose_instant": "same callback", "image_decimation_ticks": 2,
                    "simulation_rate_hz": 20.0, "image_rate_hz": 10.0},
        "height": {"h0_agl_m": h0, "h0_source": "test oracle",
                   "gsd0_m_per_px": h0 / 512.0,
                   "baro_relative_alt_definition": "body Z minus reference-frame body Z; positive up",
                   "agl_definition": "camera centre to first blocking surface along the principal ray"},
        "vo_rendering_baseline": {"motion_blur": "off"},
    }


def _fixture(tmp: Path, n: int = 12, climb: bool = False, terrain_slope: float = 0.0,
             mutate=None) -> Path:
    """A minimal but fully conforming UE run folder. `mutate(rows, settings)` corrupts one thing."""
    src = tmp / "Run_TEST"
    (src / "vo" / "images").mkdir(parents=True)
    terr0, z0 = 10.0, 60.0
    h0 = z0 - terr0
    settings = _settings(h0)

    rows = []
    for k in range(n):
        t = 0.1 * k
        north, east = 2.0 * k, 0.5 * k                     # UE x = North, UE y = East
        terr = terr0 + terrain_slope * north
        z = z0 + (1.5 * k if climb else 0.0)
        byaw = math.radians(7.0 * k)                       # airframe yaws; camera does not
        rows.append({
            "frame_id": k, "sim_time_s": "%.9f" % t, "image_path": f"vo/images/frame_{k:06d}.png",
            "cam_ue_x_cm": north * 100, "cam_ue_y_cm": east * 100, "cam_ue_z_cm": z * 100,
            "cam_ue_qw": CAM_Q[0], "cam_ue_qx": CAM_Q[1], "cam_ue_qy": CAM_Q[2], "cam_ue_qz": CAM_Q[3],
            "body_ue_x_cm": north * 100, "body_ue_y_cm": east * 100, "body_ue_z_cm": z * 100,
            "body_ue_qw": math.cos(byaw / 2), "body_ue_qx": 0.0, "body_ue_qy": 0.0,
            "body_ue_qz": math.sin(byaw / 2),
            "east_m": east, "north_m": north, "up_m": z,
            "heading_deg": 0.0, "camera_tilt_deg": 0.0,
            "baro_relative_alt_m": z - z0, "true_agl_m": z - terr,
            "terrain_elevation_m": terr, "ground_hit": 1})
        (src / "vo" / "images" / f"frame_{k:06d}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(8))

    if mutate is not None:
        mutate(rows, settings)

    (src / "settings.json").write_text(json.dumps(settings, indent=2))
    for name in ("frames.csv", "groundtruth.csv"):
        with (src / "vo" / name).open("w", newline="") as f:
            cols = FRAME_COLS if name == "frames.csv" else [c for c in FRAME_COLS if c != "image_path"]
            cols = cols if name == "frames.csv" else ["sample_id"] + cols[1:]
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "sample_id": r["frame_id"]} if name != "frames.csv" else r)
    return src


def _build(tmp: Path, **kw):
    src = _fixture(tmp, **kw)
    return ing.ingest(src, "test-ds", "constant_height", tmp / "datasets", link_images=False), tmp


# ---------------------------------------------------------------- known-answer: the adapter

def test_ingest_writes_the_contract_layout(tmp_path):
    prov, tmp = _build(tmp_path)
    ds = tmp / "datasets" / "test-ds"
    for name in ("dataset.json", "frames.csv", "groundtruth.csv", "terrain.csv",
                 "ue_body.csv", "ingest_provenance.json"):
        assert (ds / name).exists(), name
    d = json.loads((ds / "dataset.json").read_text())
    assert d["source_type"] == "ue5_simulator"
    assert d["evidence_tier"] == "T2"
    assert d["evidence_caveat"]                      # required non-null by this experiment's policy
    assert d["frame_convention"] == "ENU"
    assert d["heading_convention"] == "compass_cw_from_north"
    assert d["clock_offset_s"] == 0.0
    assert prov["n_image_frames"] == 12


def test_frames_csv_is_contract_shaped(tmp_path):
    _, tmp = _build(tmp_path)
    rows = list(csv.DictReader((tmp / "datasets" / "test-ds" / "frames.csv").open(newline="")))
    assert [r["frame_index"] for r in rows] == [str(i) for i in range(12)]
    assert all(r["image_path"] == f"images/frame_{i:06d}.png" for i, r in enumerate(rows))
    t = [float(r["timestamp_s"]) for r in rows]
    assert all(b > a for a, b in zip(t, t[1:]))


def test_up_m_is_height_above_the_takeoff_ground_plane(tmp_path):
    """The one quantity the adapter computes. up_m[0] must equal h0, as in the bvo-* datasets."""
    _, tmp = _build(tmp_path, climb=True)
    ds = tmp / "datasets" / "test-ds"
    h0 = json.loads((ds / "dataset.json").read_text())["metadata"]["h0_agl_m"]
    gt = list(csv.DictReader((ds / "groundtruth.csv").open(newline="")))
    assert float(gt[0]["up_m"]) == pytest.approx(h0, abs=1e-9)
    ter = list(csv.DictReader((ds / "terrain.csv").open(newline="")))
    assert float(ter[0]["up_m"]) == pytest.approx(h0, abs=1e-9)
    assert float(ter[0]["terrain_m"]) == pytest.approx(0.0, abs=1e-9)
    # up_m - terrain_m == agl_m at every frame, by construction of the datums
    for r in ter:
        assert float(r["up_m"]) - float(r["terrain_m"]) == pytest.approx(float(r["agl_m"]), abs=1e-6)


def test_baro_arm_equals_up_m_and_diverges_from_agl_over_terrain(tmp_path):
    """h0 + baro == up_m always; == agl only where terrain is flat. The datum limitation, pinned."""
    for slope, flat in ((0.0, True), (0.05, False)):
        tmp = tmp_path / f"s{slope}"
        tmp.mkdir()
        _build(tmp, climb=True, terrain_slope=slope)
        ds = tmp / "datasets" / "test-ds"
        h0 = json.loads((ds / "dataset.json").read_text())["metadata"]["h0_agl_m"]
        ter = list(csv.DictReader((ds / "terrain.csv").open(newline="")))
        baro = np.array([float(r["baro_relative_m"]) for r in ter])
        agl = np.array([float(r["agl_m"]) for r in ter])
        up = np.array([float(r["up_m"]) for r in ter])
        assert np.abs((h0 + baro) - up).max() < 1e-6
        if flat:
            assert np.abs((h0 + baro) - agl).max() < 1e-6
        else:
            assert np.abs((h0 + baro) - agl).max() > 1.0


def test_groundtruth_heading_is_the_camera_not_the_airframe(tmp_path):
    """The EXP-VO-013 R6 trap. Body yaw sweeps; the ground-truth heading column must not."""
    _, tmp = _build(tmp_path)
    ds = tmp / "datasets" / "test-ds"
    gt = list(csv.DictReader((ds / "groundtruth.csv").open(newline="")))
    head = np.array([float(r["heading_deg"]) for r in gt])
    body = np.array([float(r["body_yaw_deg"]) for r in
                     csv.DictReader((ds / "ue_body.csv").open(newline=""))])
    assert head.std() < 1e-9                      # camera is world-stabilised
    assert body.std() > 10.0                      # airframe is not
    assert json.loads((ds / "dataset.json").read_text())["heading_quality"]["notes"].startswith(
        "THE CAMERA'S heading")


def test_ingest_is_deterministic(tmp_path):
    """Same input bytes -> same output bytes, so a rebuild is verifiable rather than trusted."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    _build(a); _build(b)
    for name in ("frames.csv", "groundtruth.csv", "terrain.csv", "ue_body.csv", "dataset.json"):
        assert (a / "datasets" / "test-ds" / name).read_bytes() == \
               (b / "datasets" / "test-ds" / name).read_bytes(), name


def test_provenance_records_source_digests_and_checks(tmp_path):
    prov, _ = _build(tmp_path)
    assert set(prov["source_digests"]) == {"settings.json", "vo/frames.csv", "vo/groundtruth.csv"}
    assert all(len(v) == 64 for v in prov["source_digests"].values())
    v = prov["verification"]
    assert v["camera_quaternion_unique_rows"] == 1
    assert v["camera_body_yaw_is_fixed_offset"] is False
    assert v["height_identity_residual_m"]["baro_datum_identity"] < ing.TOL_M


# ---------------------------------------------------------------- negative: conventions

def _expect_reject(tmp_path, mutate, match):
    src = _fixture(tmp_path, mutate=mutate)
    with pytest.raises(ing.ConventionError, match=match):
        ing.ingest(src, "test-ds", "constant_height", tmp_path / "datasets", link_images=False)


def test_rejects_swapped_east_north(tmp_path):
    def m(rows, s):
        for r in rows:
            r["east_m"], r["north_m"] = r["north_m"], r["east_m"]
    _expect_reject(tmp_path, m, "ENU columns disagree")


def test_rejects_centimetres_left_unconverted(tmp_path):
    def m(rows, s):
        for r in rows:
            r["east_m"] *= 100.0
    _expect_reject(tmp_path, m, "ENU columns disagree")


def test_rejects_changed_enu_mapping_declaration(tmp_path):
    def m(rows, s):
        s["world_frame"]["enu_mapping"]["east_m"] = "ue_x_cm / 100"
    _expect_reject(tmp_path, m, "enu_mapping")


def test_rejects_body_centred_position_source(tmp_path):
    def m(rows, s):
        s["world_frame"]["enu_position_source"] = "body_origin"
    _expect_reject(tmp_path, m, "enu_position_source")


def test_rejects_body_yaw_as_heading(tmp_path):
    """EXP-VO-013 R6, mechanised: a heading column carrying airframe yaw must not build."""
    def m(rows, s):
        for r in rows:
            bw, bz = r["body_ue_qw"], r["body_ue_qz"]
            r["heading_deg"] = math.degrees(2 * math.atan2(bz, bw)) % 360.0
    _expect_reject(tmp_path, m, "AIRFRAME yaw")


def test_rejects_tilted_camera(tmp_path):
    def m(rows, s):
        rows[5]["camera_tilt_deg"] = 3.0
    _expect_reject(tmp_path, m, "camera_tilt_deg")


def test_rejects_non_nadir_camera_orientation(tmp_path):
    def m(rows, s):
        for r in rows:                          # identity quaternion: optical axis is +X = North
            r["cam_ue_qw"], r["cam_ue_qx"], r["cam_ue_qy"], r["cam_ue_qz"] = 1.0, 0.0, 0.0, 0.0
            r["heading_deg"] = 90.0
    _expect_reject(tmp_path, m, "not the assumed nadir")


def test_rejects_broken_height_datum(tmp_path):
    def m(rows, s):
        for r in rows:
            r["true_agl_m"] += 3.0              # agl no longer equals up - terrain
    _expect_reject(tmp_path, m, "height datum identities")


def test_rejects_nonzero_baro_at_reference_frame(tmp_path):
    def m(rows, s):
        for r in rows:
            r["baro_relative_alt_m"] += 2.0
    _expect_reject(tmp_path, m, "height datum identities|baro_relative_alt_m\\[0\\]")


def test_rejects_missing_ground_hit(tmp_path):
    def m(rows, s):
        rows[4]["ground_hit"] = 0
    _expect_reject(tmp_path, m, "ground_hit")


def test_rejects_non_monotonic_timestamps(tmp_path):
    def m(rows, s):
        rows[6]["sim_time_s"] = rows[5]["sim_time_s"]
    _expect_reject(tmp_path, m, "strictly increasing")


def test_rejects_missing_image(tmp_path):
    src = _fixture(tmp_path)
    (src / "vo" / "images" / "frame_000007.png").unlink()
    with pytest.raises(ing.ConventionError, match="absent"):
        ing.ingest(src, "test-ds", "constant_height", tmp_path / "datasets", link_images=False)


# ---------------------------------------------------------------- the arms, known-answer

def _straight_increments(n: int, dx_px: float):
    """A synthetic RIGID increment stream: constant `dx_px` to image right, no rotation."""
    inc = mro.Increments(frame_index=np.arange(n), events=["ok"] * n,
                         dq=np.zeros((n, 2)), dtheta=np.zeros(n),
                         inc_log_scale=np.zeros(n), timestamps_s=np.arange(n) * 0.1)
    inc.dq[1:, 0] = dx_px
    return inc


def test_arm_scale_error_is_the_path_weighted_mean_of_h_ratio(tmp_path):
    """The closed form `EXP-VO-014` H2 tests, verified on a known answer.

    A wrong height enters as the mean of `h_used/h_true` **weighted by each step's TRUE ground
    distance**, evaluated at the step's reference frame. The weighting is not decoration: with an
    unweighted mean this same fixture predicts 0.7607 where the answer is 0.7510, a 1.3 % error, and
    the two coincide only when the true height is constant. `analyse.py` uses the weighted form.
    """
    n, f = 200, 512.0
    inc = _straight_increments(n, 10.0)
    h_true = 40.0 + 20.0 * np.sin(np.linspace(0, 3.0, n))
    h_fixed = np.full(n, h_true[0])
    truth = mro.integrate_metric(inc, h_true, f, arm="oracle")
    fixed = mro.integrate_metric(inc, h_fixed, f, arm="fixed")

    d = np.linalg.norm(np.diff(truth.xy(), axis=0), axis=1)      # true ground distance of each step
    predicted = float(np.sum(d * (h_fixed[:-1] / h_true[:-1])) / np.sum(d))
    measured = mm.path_length(fixed.xy(), 1) / mm.path_length(truth.xy(), 1)
    assert measured == pytest.approx(predicted, rel=1e-12)

    unweighted = float(np.mean(h_fixed[:-1] / h_true[:-1]))
    assert abs(unweighted - measured) > 1e-3                      # the weighting genuinely matters


def test_identical_height_arrays_give_bit_identical_tracks(tmp_path):
    """H1's mechanism: where baro is identically zero, the BARO arm IS the FIXED arm."""
    n, f, h0 = 60, 512.0, 43.818340019
    inc = _straight_increments(n, 7.0)
    baro = np.zeros(n)
    a = mro.integrate_metric(inc, np.full(n, h0), f, arm="fixed")
    b = mro.integrate_metric(inc, mro.h_agl_from_baro(h0, baro), f, arm="baro")
    assert np.array_equal(a.east_m, b.east_m)
    assert np.array_equal(a.north_m, b.north_m)
    assert np.array_equal(a.h_used_m, b.h_used_m)


def test_visual_scale_has_no_authority(tmp_path):
    """DEC-VO-007 D4: replacing the visual series with garbage must change nothing."""
    n, f = 50, 512.0
    inc = _straight_increments(n, 4.0)
    h = np.linspace(40.0, 90.0, n)
    a = mro.integrate_metric(inc, h, f, arm="baro")
    inc.inc_log_scale[:] = np.random.default_rng(0).normal(size=n) * 5.0
    b = mro.integrate_metric(inc, h, f, arm="baro")
    assert np.array_equal(a.east_m, b.east_m) and np.array_equal(a.north_m, b.north_m)


def test_se2_alignment_never_fits_scale(tmp_path):
    """The primary family must be blind to nothing and must fit no scale."""
    rng = np.random.default_rng(1)
    gt = np.cumsum(rng.normal(size=(300, 2)), axis=0)
    al = mm.align_se2(gt * 1.5, gt)
    assert al.scale == 1.0 and al.n_fitted == 3
    sim = mm.align_sim2(gt * 1.5, gt)
    assert sim.n_fitted == 4 and sim.scale == pytest.approx(1 / 1.5, rel=1e-9)
    ref = mm.reference_initialised(gt, gt, 0.0)
    assert ref.n_fitted == 0 and ref.scale == 1.0
