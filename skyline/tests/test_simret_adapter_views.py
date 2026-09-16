"""The adapter's ``view`` option (2026-09-06 dual-direction batch): North stays the default and is
byte-for-byte the old behaviour; West is read from its own columns, camera block, image and mask.
"""

from __future__ import annotations

import numpy as np
import pytest

from hsreloc.observation import read_session
from hsreloc.simret import adapter
from hsreloc.simret.conventions import SimConventions
from tests.fixtures import sim_run


@pytest.fixture
def dual_run(tmp_path):
    return sim_run.build_run(tmp_path / "Run_D", sim_run.line_north(4, spacing_m=10.0),
                             views=("north", "west"), nested_observations=True)


def test_north_is_the_default_and_its_required_columns_are_unchanged():
    assert adapter.DEFAULT_VIEW == "north"
    assert adapter.required_columns() == adapter.REQUIRED_OBS_COLUMNS
    assert adapter.required_columns("west")[2:5] == ("west_ue_x_cm", "west_ue_y_cm", "west_ue_z_cm")
    assert adapter.required_columns("west")[-2:] == ("west_image_path", "west_sim_sky_mask_path")


def test_an_unknown_view_is_refused(dual_run):
    with pytest.raises(adapter.SimRunError, match="unknown skyline view"):
        adapter.read_run(dual_run["run_dir"], view="south")


def test_both_views_of_a_dual_run_validate_clean(dual_run):
    for view in ("north", "west"):
        run = adapter.read_run(dual_run["run_dir"], view=view)
        assert run.view == view
        assert run.camera["heading_mode"] == f"world_locked_{view}"
        report = adapter.validate_run(run, SimConventions())
        assert report.ok, (view, report.errors)
        assert report.checks["heading_lock"]["detail"]["view"] == view
        assert report.checks["heading_lock"]["detail"]["expected_compass_deg"] == (0.0 if view == "north" else 270.0)


def test_a_run_without_west_columns_still_ingests_north_and_refuses_west(tmp_path):
    built = sim_run.build_run(tmp_path / "Run_S", sim_run.line_north(3))
    assert adapter.validate_run(adapter.read_run(built["run_dir"]), SimConventions()).ok
    report = adapter.validate_run(adapter.read_run(built["run_dir"], view="west"), SimConventions())
    assert not report.ok
    assert any("west_camera" in e for e in report.errors)
    assert any("west-camera columns" in e for e in report.errors)


def test_the_west_view_ingests_its_own_pose_image_and_mask(dual_run, tmp_path):
    sn = adapter.ingest_run(dual_run["run_dir"], tmp_path / "obs", session_id="D-north",
                            options={"view": "north"})
    sw = adapter.ingest_run(dual_run["run_dir"], tmp_path / "obs", session_id="D-west",
                            options={"view": "west"})
    assert sn["view"] == "north" and sw["view"] == "west"
    meta_n, obs_n = read_session(tmp_path / "obs" / "D-north")
    meta_w, obs_w = read_session(tmp_path / "obs" / "D-west")
    assert meta_n["skyline_view"] == "north" and meta_w["skyline_view"] == "west"
    assert meta_w["camera"]["heading_mode"] == "world_locked_west"
    assert meta_w["camera"]["world_direction_deg"] == 270.0
    assert len(obs_n) == len(obs_w) == 4
    for a, b in zip(obs_n, obs_w):
        # the same physical point and instant, a camera a quarter turn to the left
        assert (a.pos_east_m, a.pos_north_m, a.up_m, a.timestamp_s) == (b.pos_east_m, b.pos_north_m, b.up_m, b.timestamp_s)
        assert a.yaw_deg == pytest.approx(0.0)
        assert b.yaw_deg == pytest.approx(270.0)
        assert a.extra["sim_view"] == "north" and b.extra["sim_view"] == "west"
        assert b.extra["sim_image_source"].endswith("_west.png")
        assert b.extra["sim_sky_mask_path"].endswith("_west_sky.png")
        assert a.extra["sim_image_source"] == f"skyline/images/{a.extra['sim_observation_id_raw']}.png"
    # the West GT curve is the West mask's boundary, not the North one's
    raw = obs_w[0].extra["sim_observation_id_raw"]
    expected = dual_run["expected"][raw]
    import csv
    stored = {}
    with (tmp_path / "obs" / "D-west" / "skylines_oracle" / f"{obs_w[0].observation_id}.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stored[int(row["col"])] = float(row["row"])
    boundary = np.array([stored[c] for c in range(len(expected["west_boundary"]))])
    assert np.allclose(boundary, expected["west_boundary"])
    assert not np.allclose(boundary, expected["boundary"])


def test_a_west_camera_that_is_not_facing_west_is_refused(tmp_path):
    built = sim_run.build_run(tmp_path / "Run_W", sim_run.line_north(3), views=("north", "west"),
                              west_yaw_deg=0.0)
    report = adapter.validate_run(adapter.read_run(built["run_dir"], view="west"), SimConventions())
    assert any("not pointing West" in e and "nearest cardinal 0" in e for e in report.errors)
    # and the North view of the same run is untouched by the West defect
    assert adapter.validate_run(adapter.read_run(built["run_dir"]), SimConventions()).ok
