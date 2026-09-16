"""The simulator run adapter: what it accepts, what it refuses, and what it preserves (Part A).

Every fixture is a synthetic run in the delivered export format with constructed poses, so each
expected ENU coordinate, compass heading and GT curve is known before the adapter runs. The refusal
tests matter as much as the happy path: the brief's rule is *do not silently repair invalid runs*, and
a validator that cannot be shown to reject a bad run is not a validator.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from hsreloc.observation import read_session
from hsreloc.simret import adapter
from hsreloc.simret.conventions import SimConventions, zyx_right_handed_quaternion
from tests.fixtures import sim_run


@pytest.fixture
def good_run(tmp_path):
    return sim_run.build_run(tmp_path / "Run_A", sim_run.line_north(6, spacing_m=10.0, east_m=0.0))


# -- the happy path ----------------------------------------------------------------------------

def test_a_well_formed_run_validates_clean(good_run):
    run = adapter.read_run(good_run["run_dir"])
    report = adapter.validate_run(run, SimConventions())
    assert report.ok, report.errors
    assert report.checks["heading_lock"]["ok"]
    assert report.checks["camera_calibration"]["ok"]
    assert report.checks["quaternion_norm"]["ok"]


def test_the_nested_observations_layout_of_the_first_real_batch_is_accepted(tmp_path):
    """The 2026-09-02 export batch puts observations.csv under skyline/, not at the run root."""
    built = sim_run.build_run(tmp_path / "Run_N", sim_run.line_north(3), nested_observations=True)
    run = adapter.read_run(built["run_dir"])
    assert run.observations_file == "skyline/observations.csv"
    assert len(run.rows) == 3
    report = adapter.validate_run(run, SimConventions())
    assert report.ok, report.errors
    summary = adapter.ingest_run(built["run_dir"], tmp_path / "obs", session_id="simN")
    assert summary["n_observations"] == 3
    meta, _ = read_session(tmp_path / "obs" / "simN")
    assert meta["simulator_run"]["observations_file"] == "skyline/observations.csv"


def test_ingest_writes_a_spec_006_session_with_the_expected_poses(good_run, tmp_path):
    summary = adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA")
    assert summary["n_observations"] == 6
    assert summary["n_database_holes"] == 0
    meta, observations = read_session(tmp_path / "obs" / "simA")

    assert meta["frame_convention"] == "ENU"
    assert meta["heading_convention"] == "compass_cw_from_north"
    assert meta["evidence_tier"] == "T2"
    assert meta["source_type"] == "ue5_simulator"
    assert meta["evidence_caveat"]
    assert meta["camera"]["heading_mode"] == "world_locked_north"
    assert meta["scene"]["level_id"] == "hills_with_forest"
    assert meta["condition"] == {"time_of_day": "DAY", "hour": 12.0, "clouds": "CLEAR",
                                 "time_speed": 0.0}

    # The fixture put frame i at north = 10 i m, east = 0, up = 60, facing North, level.
    for i, obs in enumerate(observations):
        assert obs.pos_north_m == pytest.approx(10.0 * i)
        assert obs.pos_east_m == pytest.approx(0.0)
        assert obs.up_m == pytest.approx(60.0)
        assert obs.yaw_deg == pytest.approx(0.0)
        assert obs.pitch_deg == pytest.approx(0.0)
        assert obs.roll_deg == pytest.approx(0.0)
        assert obs.gt_source == "sim_exact"
        assert obs.gt_pos_sigma_m == 0.0
        assert obs.fov_deg == pytest.approx(90.0)
        assert obs.frame_index == i


def test_raw_ue_values_are_preserved_for_re_conversion(good_run, tmp_path):
    adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA")
    _, observations = read_session(tmp_path / "obs" / "simA")
    obs = observations[3]
    assert float(obs.extra["sim_ue_x_cm"]) == pytest.approx(3000.0)
    assert float(obs.extra["sim_ue_z_cm"]) == pytest.approx(6000.0)
    assert obs.extra["sim_ue_yaw_deg"] is not None
    assert float(obs.extra["sim_ue_quat_w"]) == pytest.approx(1.0)
    assert obs.extra["sim_observation_id_raw"] == "sky_000004"
    assert obs.extra["sim_run_id"] == "Run_20260831_172540"
    # PROT-001 §3.3: a convention error must be fixable by re-conversion, so the raw values are enough
    # to redo the conversion without the run directory.
    assert float(obs.extra["sim_ue_y_cm"]) / 100.0 == pytest.approx(obs.pos_east_m)


def test_condition_and_nadir_metadata_ride_along(good_run, tmp_path):
    adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA")
    _, observations = read_session(tmp_path / "obs" / "simA")
    obs = observations[0]
    assert obs.extra["condition_level"] == "hills_with_forest"
    assert obs.extra["condition_time_of_day"] == "DAY"
    assert obs.extra["condition_clouds"] == "CLEAR"
    assert float(obs.extra["condition_hour"]) == 12.0
    assert obs.extra["sim_nadir_ue_pitch_deg"] == "-90.0"       # preserved, never depended on


def test_gt_curves_match_the_constructed_masks(good_run, tmp_path):
    adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA")
    session = tmp_path / "obs" / "simA"
    _, observations = read_session(session)
    for obs in observations:
        raw_id = obs.extra["sim_observation_id_raw"]
        expected = good_run["expected"][raw_id]["boundary"]
        stored = {}
        with (session / "skylines_oracle" / f"{obs.observation_id}.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stored[int(row["col"])] = float(row["row"])
        got = np.array([stored[c] for c in range(len(expected))])
        assert np.allclose(got, expected)
        assert obs.oracle_provenance == "oracle:sim_exact"
        assert (session / "sim" / f"{obs.observation_id}_skyline_gt.csv").exists()


def test_a_column_without_sky_becomes_a_hole_not_a_fabricated_curve(tmp_path):
    def rows_with_tower(width, height, phase=0.0, offset=0.0):
        rows = sim_run.default_rows(width, height, phase, offset)
        rows[width // 2] = 0                       # a tower filling the column to the top of frame
        return rows

    run = sim_run.build_run(tmp_path / "Run_T", sim_run.line_north(2), row_fn=rows_with_tower)
    summary = adapter.ingest_run(run["run_dir"], tmp_path / "obs", session_id="simT")
    assert summary["n_seam_curves"] == 0
    assert summary["n_database_holes"] == 2
    assert not (tmp_path / "obs" / "simT" / "skylines_oracle").exists()
    _, observations = read_session(tmp_path / "obs" / "simT")
    assert observations[0].oracle_skyline_path is None
    assert observations[0].extra["sim_gt_status"] == "insufficient_sky"
    # the full-fidelity GT curve is still written, invalid column and all
    assert (tmp_path / "obs" / "simT" / "sim" / f"{observations[0].observation_id}_skyline_gt.csv").exists()


def test_image_mode_reference_leaves_frames_in_the_run_directory(good_run, tmp_path):
    adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA",
                       options={"image_mode": "reference"})
    session = tmp_path / "obs" / "simA"
    assert not any((session / "images").glob("*.png"))
    _, observations = read_session(session)
    assert observations[0].extra["sim_image_source"].startswith("skyline/images/")


def test_session_json_carries_the_conventions_and_the_validation_report(good_run, tmp_path):
    adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA",
                       conventions=SimConventions(altitude_datum_ue_cm=0.0))
    meta = json.loads((tmp_path / "obs" / "simA" / "session.json").read_text(encoding="utf-8"))
    assert meta["conventions"]["ue_world_axes"] == "x_north_y_east"
    assert meta["conventions"]["attitude_verified"] is False
    assert meta["validation"]["ok"] is True
    assert any("pitch and roll signs are DECLARED but not verified" in w
               for w in meta["validation"]["warnings"])
    assert meta["simulator_run"]["raw_settings"]["skyline"]["capture_distance_m"] == 100.0


def test_ingest_refuses_to_overwrite_an_existing_session(good_run, tmp_path):
    adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA")
    with pytest.raises(adapter.SimRunError, match="refusing to overwrite"):
        adapter.ingest_run(good_run["run_dir"], tmp_path / "obs", session_id="simA")


# -- refusals ----------------------------------------------------------------------------------

def test_missing_settings_file_is_not_a_run(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(adapter.SimRunError, match="not a simulator run"):
        adapter.read_run(tmp_path / "empty")


def test_missing_camera_key_is_refused(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_B", sim_run.line_north(2))
    settings = json.loads((run["run_dir"] / "settings.json").read_text(encoding="utf-8"))
    del settings["north_camera"]["heading_mode"]
    (run["run_dir"] / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert not report.ok
    assert any("heading_mode" in e for e in report.errors)


def test_missing_observation_column_is_refused(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_C", sim_run.line_north(2))
    path = run["run_dir"] / "observations.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    drop = header.index("north_ue_quat_w")
    path.write_text("\n".join(",".join(c for i, c in enumerate(line.split(",")) if i != drop)
                             for line in lines) + "\n", encoding="utf-8")
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("north_ue_quat_w" in e for e in report.errors)


def test_missing_image_and_missing_mask_are_both_refused(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_D", sim_run.line_north(3))
    (run["run_dir"] / "skyline" / "images" / "sky_000002.png").unlink()
    (run["run_dir"] / "skyline" / "sim" / "sky_000003_sky.png").unlink()
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("RGB image(s) do not exist" in e for e in report.errors)
    assert any("simulator mask(s) do not exist" in e for e in report.errors)
    with pytest.raises(adapter.SimRunError, match="nothing was written"):
        adapter.ingest_run(run["run_dir"], tmp_path / "obs")
    assert not (tmp_path / "obs").exists()


def test_mask_dimensions_must_match_their_rgb_frame(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_E", sim_run.line_north(2))
    sim_run.write_png(run["run_dir"] / "skyline" / "sim" / "sky_000001_sky.png",
                      np.zeros((10, 10), dtype=np.uint8))
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("mask is 10x10" in e for e in report.errors)


def test_image_resolution_must_match_settings(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_F", sim_run.line_north(2))
    sim_run.write_png(run["run_dir"] / "skyline" / "images" / "sky_000001.png",
                      np.zeros((8, 8, 3), dtype=np.uint8))
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("do not match the resolution in settings.json" in e for e in report.errors)


def test_inconsistent_camera_calibration_is_refused(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_G", sim_run.line_north(2),
                            settings_override={"north_camera": None})
    settings = json.loads((run["run_dir"] / "settings.json").read_text(encoding="utf-8"))
    settings["north_camera"] = {**sim_run.settings()["north_camera"]}
    settings["north_camera"]["intrinsics_px"]["fx"] = 999.0        # not what 90 deg HFOV implies
    settings["north_camera"]["width_px"] = run["width"]
    settings["north_camera"]["height_px"] = run["height"]
    settings["north_camera"]["intrinsics_px"]["cx"] = run["width"] / 2
    settings["north_camera"]["intrinsics_px"]["cy"] = run["height"] / 2
    (run["run_dir"] / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("horizontal_fov_deg" in e and "implies" in e for e in report.errors)


def test_a_non_perspective_or_distorted_camera_is_refused(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_H", sim_run.line_north(2),
                            settings_override={"north_camera": {"projection": "fisheye",
                                                                "distortion": "kannala_brandt"}})
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("not 'perspective'" in e for e in report.errors)
    assert any("distortion-free camera" in e for e in report.errors)


def test_unnormalised_quaternions_are_refused(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_I", sim_run.line_north(2))
    path = run["run_dir"] / "observations.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["north_ue_quat_w"] = "1.4"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("off unit length" in e for e in report.errors)


def test_a_quaternion_in_an_unknown_convention_is_refused_with_every_residual(tmp_path):
    """The quaternion is deliberately not a rotation of the logged Euler angles under ANY candidate."""
    def bogus(yaw, pitch, roll):
        return np.array([0.0, 1.0, 0.0, 0.0])

    poses = [{"x_cm": 0.0, "y_cm": 0.0, "z_cm": 6000.0, "yaw": 0.0, "pitch": 7.0, "roll": -4.0}]
    run = sim_run.build_run(tmp_path / "Run_J", poses, quaternion_fn=bogus)
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("NONE of the candidate conventions" in e for e in report.errors)
    assert set(report.checks["quaternion_convention"]["detail"]["residual_deg"]) == {
        "ue_frotator", "ue_frotator_conjugate", "zyx_right_handed", "zyx_right_handed_conjugate"}


def test_a_declared_convention_that_the_data_contradicts_is_refused(tmp_path):
    poses = [{"yaw": 30.0, "pitch": 20.0, "roll": 10.0, "z_cm": 6000.0}]
    run = sim_run.build_run(tmp_path / "Run_K", poses, quaternion_fn=zyx_right_handed_quaternion,
                            heading_mode="body_fixed")
    report = adapter.validate_run(adapter.read_run(run["run_dir"]),
                                  SimConventions(quaternion_convention="ue_frotator"))
    assert any("declares quaternion convention 'ue_frotator'" in e for e in report.errors)


def test_identity_attitudes_warn_that_the_convention_is_not_established(good_run):
    report = adapter.validate_run(adapter.read_run(good_run["run_dir"]), SimConventions())
    assert report.ok
    assert any("cannot establish the export's quaternion convention" in w for w in report.warnings)
    assert report.checks["quaternion_convention"]["detail"]["discriminating"] is False


def test_a_world_locked_north_camera_that_is_not_facing_north_is_refused(tmp_path):
    """The empirical test of the axis declaration. A constant yaw of 90 under +X = North means the
    camera is East-facing, or the project does not declare what PROT-001 §2.2 asks for."""
    run = sim_run.build_run(tmp_path / "Run_L", sim_run.line_north(3, yaw=90.0))
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("not pointing North" in e and "nearest cardinal 90" in e for e in report.errors)


def test_a_varying_yaw_under_a_north_lock_is_refused(tmp_path):
    poses = sim_run.line_north(3)
    poses[1]["yaw"] = 5.0
    run = sim_run.build_run(tmp_path / "Run_M", poses)
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("compass yaw varies by" in e for e in report.errors)


def test_body_fixed_heading_mode_warns_rather_than_refusing(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_N", sim_run.line_north(3, yaw=42.0),
                            heading_mode="body_fixed")
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert report.ok
    assert any("known-heading isolation" in w for w in report.warnings)


def test_a_non_binary_mask_stops_the_ingest(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_O", sim_run.line_north(2))
    grey = np.full((run["height"], run["width"]), 128, dtype=np.uint8)
    grey[:10] = 255
    sim_run.write_png(run["run_dir"] / "skyline" / "sim" / "sky_000001_sky.png", grey)
    with pytest.raises(adapter.SimRunError, match="not binary enough"):
        adapter.ingest_run(run["run_dir"], tmp_path / "obs")


def test_duplicate_observation_ids_are_refused(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_P", sim_run.line_north(2))
    path = run["run_dir"] / "observations.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[1]["observation_id"] = rows[0]["observation_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    report = adapter.validate_run(adapter.read_run(run["run_dir"]), SimConventions())
    assert any("duplicated" in e for e in report.errors)


def test_png_size_reads_the_header_without_decoding(tmp_path):
    sim_run.write_png(tmp_path / "a.png", np.zeros((7, 13), dtype=np.uint8))
    assert adapter.png_size(tmp_path / "a.png") == (13, 7)
    (tmp_path / "b.png").write_bytes(b"not a png at all, really not")
    with pytest.raises(adapter.SimRunError, match="not a PNG"):
        adapter.png_size(tmp_path / "b.png")
