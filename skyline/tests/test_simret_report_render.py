"""Measurement and figure layers, and the config-driven CLI (Parts I and the entry points).

The assertions that matter most here are the **refusals**. Both layers must be incapable of producing
an artifact with no data behind it: an empty deformation table and a plot with no series are the two
things that would look like results while being nothing at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hsreloc.matchers import (BoundedLagNccMatcher, ConstrainedDtwMatcher, FrozenNccMatcher,
                              ShiftScaleNccMatcher)
from hsreloc.observation import read_session
from hsreloc.simret import adapter, cli, geometry, render, report, sets, sources
from tests.fixtures import sim_run

SPACING = 40.0


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim_bench")
    obs_root = root / "observations_sim"
    ref = sim_run.build_run(root / "Run_ref", sim_run.line_north(12, spacing_m=10.0), run_id="Run_ref")
    sim_run.build_run(root / "Run_lat", sim_run.line_north(12, spacing_m=10.0, east_m=10.0),
                      run_id="Run_lat", time_of_day="DUSK", clouds="VERY_CLOUDY")
    adapter.ingest_run(ref["run_dir"], obs_root, session_id="ref_day")
    adapter.ingest_run(root / "Run_lat", obs_root, session_id="lat_dusk")
    sessions = {sid: read_session(obs_root / sid)[1] for sid in ("ref_day", "lat_dusk")}
    tasks = geometry.build_tasks(sessions, {"reference_session": "ref_day",
                                            "query_sessions": ["lat_dusk"],
                                            "spacings_m": [SPACING]})
    geometry.write_tasks(tasks, root / "tasks")
    tasks["manifest"] = geometry.load_tasks(root / "tasks")["manifest"]
    smap = sources.session_map(obs_root)
    obs_by_id = {o.observation_id: o for s in sessions.values() for o in s}
    return {"root": root, "obs_root": obs_root, "tasks": tasks, "sessions": sessions,
            "obs_by_id": obs_by_id, "smap": smap, "w": ref["width"], "h": ref["height"],
            "source": sources.SimExactSource(obs_root, smap, ref["width"], ref["height"])}


# -- curve disagreement ---------------------------------------------------------------------------

def test_curve_disagreement_separates_a_constant_offset_from_a_shape_change():
    a = np.linspace(10.0, 30.0, 64)
    assert report.curve_disagreement(a, a + 4.0, 100)["offset_px"] == pytest.approx(4.0)
    assert report.curve_disagreement(a, a + 4.0, 100)["median_abs_offset_removed_px"] == pytest.approx(0.0)
    noisy = a + 4.0 + np.where(np.arange(64) % 2, 1.0, -1.0)
    out = report.curve_disagreement(a, noisy, 100)
    assert out["offset_px"] == pytest.approx(4.0)
    assert out["median_abs_offset_removed_px"] == pytest.approx(1.0)


def test_curve_disagreement_refuses_mismatched_shapes():
    with pytest.raises(report.ReportError, match="curve shapes differ"):
        report.curve_disagreement(np.zeros(4), np.zeros(5), 10)


# -- deformation vs translation -------------------------------------------------------------------

def test_deformation_table_carries_pose_offsets_curves_and_matcher_parameters(bench):
    out = report.deformation_vs_translation(
        bench["tasks"], bench["obs_by_id"], bench["source"], bench["h"],
        matchers={"c0_frozen_ncc": FrozenNccMatcher(),
                  "c1_bounded_lag_ncc": BoundedLagNccMatcher(max_lag_samples=8, min_overlap_frac=0.7)})
    rows = out["rows"]
    assert len(rows) == len(bench["tasks"]["queries"])
    r = rows[0]
    assert r["lateral_m"] == pytest.approx(10.0)          # the query line is 10 m East
    assert r["yaw_diff_deg"] == pytest.approx(0.0)
    for key in ("curve_median_abs_px", "curve_offset_px", "curve_median_abs_offset_removed_px",
                "c0_frozen_ncc_score", "c1_bounded_lag_ncc_shift", "c1_bounded_lag_ncc_accepted"):
        assert key in r
    assert out["summary"]                                  # grouped by translation bin, with its own n
    assert out["source"]["provenance"] == "oracle:sim_exact"


def test_the_measurement_layer_refuses_an_empty_task_set(bench):
    with pytest.raises(report.ReportError, match="refusing to emit an empty table"):
        report.deformation_vs_translation({"queries": []}, {}, bench["source"], bench["h"])


def test_summarise_by_keeps_each_group_s_own_n():
    rows = [{"bin": "a", "v": 1.0}, {"bin": "a", "v": 3.0}, {"bin": "b", "v": 10.0}]
    out = report.summarise_by(rows, "bin", ["v"])
    assert out["a"]["n"] == 2 and out["b"]["n"] == 1
    assert out["a"]["v"]["median"] == pytest.approx(2.0)


# -- extraction accuracy --------------------------------------------------------------------------

def test_extraction_accuracy_is_measured_against_ground_truth_and_counts_refusals(bench, tmp_path):
    class Shifted:
        """A stand-in extractor that is exactly 3 px low, and silent on one frame."""
        def __init__(self, inner, silent):
            self.inner, self.silent = inner, silent

        def get(self, oid):
            from hsreloc.retrieval.skyline_curve import CurveError, make_curve
            if oid == self.silent:
                raise CurveError(f"{oid}: refused")
            c = self.inner.get(oid)
            return make_curve(oid, c.row_per_col + 3.0, c.image_width_px, c.image_height_px,
                              "automatic:poc_robust_dp", "test")

        def describe(self):
            return {"kind": "test", "provenance": "automatic:poc_robust_dp"}

    ids = [o.observation_id for o in bench["sessions"]["ref_day"]]
    pred = Shifted(bench["source"], ids[0])
    conditions = {o: "DAY_CLEAR" for o in ids}
    out = report.extraction_accuracy(bench["source"], pred, ids, bench["h"], conditions)
    assert out["n_evaluated"] == len(ids) - 1
    assert out["n_refused"] == 1
    assert out["refusal_rate"] == pytest.approx(1 / len(ids))
    assert out["rows"][0]["median_abs_px"] == pytest.approx(3.0)
    assert out["by_condition"]["DAY_CLEAR"]["n"] == len(ids) - 1
    assert "never mixed into a retrieval number" in out["note"]
    path = report.write_table(out, tmp_path / "tables", "extraction_accuracy")
    assert path.exists() and (tmp_path / "tables" / "extraction_accuracy_summary.json").exists()


# -- variant comparison ---------------------------------------------------------------------------

def test_variant_comparison_requires_the_frozen_baseline(bench):
    with pytest.raises(report.ReportError, match="frozen C0 baseline must be one of"):
        report.variant_comparison(bench["tasks"], bench["source"],
                                  {"c1": BoundedLagNccMatcher(max_lag_samples=4)})


def test_variant_comparison_scores_every_rung_on_identical_pairs(bench):
    out = report.variant_comparison(bench["tasks"], bench["source"], {
        "c0_frozen_ncc": FrozenNccMatcher(),
        "c1_bounded_lag_ncc": BoundedLagNccMatcher(max_lag_samples=8, min_overlap_frac=0.7),
        "c3_shift_scale_ncc": ShiftScaleNccMatcher(max_lag_samples=8,
                                                   scales=ShiftScaleNccMatcher.scale_grid(0.04, 5),
                                                   min_overlap_frac=0.7),
        "c4_banded_dtw_ncc": ConstrainedDtwMatcher(band=8, max_mean_warp=6.0)})
    assert len(out["rows"]) == len(bench["tasks"]["queries"])
    assert set(out["matchers"]) == {"c0_frozen_ncc", "c1_bounded_lag_ncc", "c3_shift_scale_ncc",
                                    "c4_banded_dtw_ncc"}
    # every variant contains C0's alignment, so none of them can score below it on the same pair
    for r in out["rows"]:
        assert r["c1_bounded_lag_ncc_score"] >= r["c0_frozen_ncc_score"] - 1e-12
        assert r["c3_shift_scale_ncc_score"] >= r["c0_frozen_ncc_score"] - 1e-12


def test_write_table_refuses_to_write_nothing(tmp_path):
    with pytest.raises(report.ReportError):
        report.write_table({"rows": []}, tmp_path, "empty")


# -- renders --------------------------------------------------------------------------------------

def test_the_pair_panel_draws_every_source_and_the_transform_parameters(bench, tmp_path):
    ref = bench["sessions"]["ref_day"][0]
    qry = bench["sessions"]["lat_dusk"][0]
    gt_ref = bench["source"].get(ref.observation_id).row_per_col
    gt_qry = bench["source"].get(qry.observation_id).row_per_col
    out = render.render_pair(
        tmp_path / "pair.jpg",
        query={"observation_id": qry.observation_id,
               "image_path": bench["obs_root"] / "lat_dusk" / qry.image_path,
               "curves": {"oracle:sim_exact": gt_qry,
                          "automatic:segformer_b0_ade20k": gt_qry + 2.0,
                          "automatic:poc_robust_dp": gt_qry - 1.5}},
        reference={"observation_id": ref.observation_id,
                   "image_path": bench["obs_root"] / "ref_day" / ref.image_path,
                   "curves": {"oracle:sim_exact": gt_ref,
                              "automatic:segformer_b0_ade20k": gt_ref + 2.0,
                              "automatic:poc_robust_dp": gt_ref - 1.5}},
        offset=geometry.viewpoint_offset(qry, ref).as_dict(),
        scores={"c0_frozen_ncc": {"score": 0.42, "shift": 0.0, "scale": 1.0, "warp_magnitude": 0.0,
                                  "accepted": True},
                "c1_bounded_lag_ncc": {"score": 0.71, "shift": -3.0, "scale": 1.0,
                                       "warp_magnitude": 0.0, "accepted": True}},
        correct=True)
    assert out.exists() and out.stat().st_size > 5000


def test_the_pair_panel_refuses_a_query_with_no_curves(tmp_path):
    with pytest.raises(render.RenderError, match="no data for pair panel"):
        render.render_pair(tmp_path / "x.jpg", {"observation_id": "q", "curves": {}},
                           {"observation_id": "r", "curves": {}}, {}, {}, False)


def test_the_trend_plots_draw_from_real_rows(bench, tmp_path):
    table = report.deformation_vs_translation(
        bench["tasks"], bench["obs_by_id"], bench["source"], bench["h"],
        matchers={"c0_frozen_ncc": FrozenNccMatcher(),
                  "c1_bounded_lag_ncc": BoundedLagNccMatcher(max_lag_samples=8, min_overlap_frac=0.7)})
    rows = table["rows"]
    assert render.plot_deformation_vs_translation(tmp_path / "deform.png", rows).exists()
    assert render.plot_lateral_vs_longitudinal(tmp_path / "map.png", rows,
                                               "curve_median_abs_px", "median |Δrow| (px)").exists()
    assert render.plot_variant_comparison(tmp_path / "variants.png", rows,
                                          ["c0_frozen_ncc", "c1_bounded_lag_ncc"]).exists()
    assert render.plot_metric_vs_axis(tmp_path / "axis.png", rows, "translation_bin",
                                      "curve_median_abs_px", "deformation", "px",
                                      series_field="stage").exists()


def test_every_plot_refuses_empty_input(tmp_path):
    for fn, args in ((render.plot_deformation_vs_translation, (tmp_path / "a.png", [])),
                     (render.plot_lateral_vs_longitudinal, (tmp_path / "b.png", [], "v")),
                     (render.plot_variant_comparison, (tmp_path / "c.png", [], ["c0_frozen_ncc"])),
                     (render.plot_metric_vs_axis, (tmp_path / "d.png", [], "x", "y", "t", "l"))):
        with pytest.raises(render.RenderError):
            fn(*args)


def test_a_variant_comparison_plot_without_the_baseline_is_refused(bench, tmp_path):
    with pytest.raises(render.RenderError, match="frozen C0 baseline must appear"):
        render.plot_variant_comparison(tmp_path / "x.png", [{"translation_bin": "0-1", "c1_score": 1.0}],
                                       ["c1"])


# -- the CLI --------------------------------------------------------------------------------------

def test_the_cli_runs_validate_ingest_groups_tasks_and_sets(tmp_path):
    run = sim_run.build_run(tmp_path / "Run_cli", sim_run.line_north(8, spacing_m=10.0),
                            run_id="Run_cli")
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    ingest_cfg = {"run_dir": str(run["run_dir"]), "store_root": str(tmp_path / "obs"),
                  "session_id": "cli_session", "conventions": {"altitude_datum_ue_cm": 0.0},
                  "options": {"image_mode": "copy"}, "sessions": ["cli_session"],
                  "silver_frame_list": str(tmp_path / "silver_frames.csv")}
    (cfg_dir / "ingest.json").write_text(json.dumps(ingest_cfg), encoding="utf-8")
    assert cli.main(["validate", "--config", str(cfg_dir / "ingest.json")]) == 0
    assert cli.main(["ingest", "--config", str(cfg_dir / "ingest.json")]) == 0
    assert cli.main(["silver-list", "--config", str(cfg_dir / "ingest.json")]) == 0
    frames = (tmp_path / "silver_frames.csv").read_text(encoding="utf-8").splitlines()
    assert frames[0].startswith("session_id,observation_id,image_path")
    assert len(frames) == 9

    anchors = cfg_dir / "anchors.csv"
    anchors.write_text("anchor_id,east_m,north_m,up_m,heading_deg,radius_m\nA,0,0,60,0,3\n",
                       encoding="utf-8")
    groups_cfg = {"store_root": str(tmp_path / "obs"), "anchors_csv": "anchors.csv",
                  "out_dir": str(tmp_path / "index")}
    (cfg_dir / "groups.json").write_text(json.dumps(groups_cfg), encoding="utf-8")
    assert cli.main(["groups", "--config", str(cfg_dir / "groups.json")]) == 0
    assert (tmp_path / "index" / "experiment_index.csv").exists()

    exp_cfg = {"task_name": "e0_plumbing", "store_root": str(tmp_path / "obs"),
               "tasks_dir": str(tmp_path / "tasks"), "datasets_root": str(tmp_path / "datasets"),
               "refdb_root": str(tmp_path / "refdb"),
               "tasks": {"reference_session": "cli_session", "query_sessions": ["cli_session"],
                         "spacings_m": [30.0]},
               "sources": {"sim_exact": {}}}
    (cfg_dir / "exp.json").write_text(json.dumps(exp_cfg), encoding="utf-8")
    assert cli.main(["tasks", "--config", str(cfg_dir / "exp.json")]) == 0
    assert cli.main(["sets", "--config", str(cfg_dir / "exp.json")]) == 0
    qs = tmp_path / "datasets" / sets.query_set_id("e0_plumbing", 30.0, "sim_exact")
    assert (qs / "dataset.json").exists() and (qs / "skyline_queries.csv").exists()
    rs = tmp_path / "refdb" / sets.reference_set_id("e0_plumbing", 30.0, "sim_exact")
    assert json.loads((rs / "manifest.json").read_text(encoding="utf-8"))[
        "reference_source"]["extraction_mode"] == "oracle:sim_exact"


def test_run_and_evaluate_exist_only_behind_the_preregistration_guard():
    """Research R3 (2026-09-02): with real data present, the guard moved from 'no command exists'
    to the placeret-proven mechanism — the command exists and refuses without a committed,
    matching pre-registration (exercised in test_simret_pipeline)."""
    assert {"prereg", "run", "evaluate"} <= set(cli.COMMANDS)
    assert {"validate", "ingest", "extract-dp", "silver-list", "groups", "tasks", "sets",
            "exp-extraction", "exp-consistency", "exp-deformation", "exp-variant",
            "report"} <= set(cli.COMMANDS)


# -- condition consistency (Experiment B; research R7) --------------------------------------------

class _CurveDict:
    """A minimal curve source: observation_id -> row array (or a CurveError)."""

    provenance = "oracle:sim_exact"

    def __init__(self, curves, width, height):
        self.curves, self.w, self.h = curves, width, height

    def get(self, oid):
        from hsreloc.retrieval.skyline_curve import CurveError, make_curve
        if oid not in self.curves:
            raise CurveError(f"{oid}: constructed refusal")
        return make_curve(oid, np.asarray(self.curves[oid], dtype=float), self.w, self.h,
                          self.provenance, f"test:{oid}")

    def describe(self):
        return {"kind": "test_dict", "provenance": self.provenance}


def _index_row(oid, pg, tod, clouds, anchor="aX"):
    return {"observation_id": oid, "condition_pose_group": pg, "anchor_id": anchor,
            "condition_time_of_day": tod, "condition_clouds": clouds}


def test_condition_consistency_pairs_within_pose_groups_with_known_answers():
    w, h = 32, 100
    base = np.linspace(30.0, 50.0, w)
    curves = {"day": base, "dawn": base + 4.0,                     # pure offset: removed entirely
              "dusk": base + np.where(np.arange(w) % 2, 3.0, -3.0)}  # pure shape change
    src = _CurveDict(curves, w, h)
    rows = [_index_row("day", "pg1", "DAY", "CLEAR"), _index_row("dawn", "pg1", "DAWN", "CLEAR"),
            _index_row("dusk", "pg1", "DUSK", "CLEAR")]
    out = report.condition_consistency(rows, {"sim_exact": src}, h, catastrophic_frac=0.05)
    assert out["n_pose_groups"] == 1 and out["n_pairs_per_source"]["sim_exact"] == 3
    day_dawn = next(r for r in out["rows"] if {r["observation_a"], r["observation_b"]} == {"day", "dawn"})
    assert day_dawn["pair"] == "DAWN+CLEAR vs DAY+CLEAR"
    # pair members are ordered by observation id ("dawn" < "day"), so offset = b - a = day - dawn
    assert (day_dawn["observation_a"], day_dawn["observation_b"]) == ("dawn", "day")
    assert day_dawn["curve_offset_px"] == pytest.approx(-4.0)
    assert day_dawn["curve_median_abs_offset_removed_px"] == pytest.approx(0.0)
    assert day_dawn["catastrophic"] is False
    day_dusk = next(r for r in out["rows"] if {r["observation_a"], r["observation_b"]} == {"day", "dusk"})
    assert day_dusk["curve_median_abs_offset_removed_px"] == pytest.approx(3.0)
    assert day_dusk["catastrophic"] is False                       # 3 px < 5 % of 100
    strict = report.condition_consistency(rows, {"sim_exact": src}, h, catastrophic_frac=0.02)
    assert strict["catastrophic_counts"]["sim_exact"] == 2         # both dusk pairs exceed 2 px


def test_condition_consistency_keeps_duplicates_and_records_refusals():
    w, h = 16, 60
    base = np.full(w, 20.0)
    src = _CurveDict({"a": base, "b": base}, w, h)
    rows = [_index_row("a", "pg", "DAY", "CLEAR"), _index_row("b", "pg", "DAY", "CLEAR"),
            _index_row("c", "pg", "DUSK", "CLEAR")]
    out = report.condition_consistency(rows, {"sim_exact": src}, h)
    dup = next(r for r in out["rows"] if {r["observation_a"], r["observation_b"]} == {"a", "b"})
    assert dup["pair"] == "same-condition rerender"
    refused = [r for r in out["rows"] if r["status"] != "ok"]
    assert len(refused) == 2 and all("c" in (r["observation_a"], r["observation_b"]) for r in refused)
    assert len(out["refusals"]) == 2


def test_condition_consistency_refuses_groups_of_one():
    rows = [_index_row("solo", "pg_solo", "DAY", "CLEAR")]
    with pytest.raises(report.ReportError, match="more than one member"):
        report.condition_consistency(rows, {}, 100)


# -- rank-based variant retrieval (research R9) ---------------------------------------------------

def test_variant_retrieval_ranks_the_whole_database_with_known_answers(bench):
    out = report.variant_retrieval(bench["tasks"], bench["source"],
                                   {"c0_frozen_ncc": FrozenNccMatcher()})
    assert out["n_queries"] == len(bench["tasks"]["queries"])
    r = out["rows"][0]
    n_refs = r["n_references_scored"]
    assert n_refs == bench["tasks"]["grids"][f"{SPACING:g}"]["n_references"]
    assert 1 <= r["c0_frozen_ncc_rank"] <= n_refs
    assert r["c0_frozen_ncc_top1"] == (r["c0_frozen_ncc_rank"] == 1)
    if r["c0_frozen_ncc_top1"]:
        assert r["c0_frozen_ncc_margin"] == pytest.approx(
            r["c0_frozen_ncc_correct_score"] - r["c0_frozen_ncc_best_incorrect_score"])
    assert out["top1_counts"]["c0_frozen_ncc"] == sum(1 for x in out["rows"]
                                                      if x["c0_frozen_ncc_top1"])


def test_variant_retrieval_requires_the_frozen_baseline(bench):
    with pytest.raises(report.ReportError, match="C0 baseline"):
        report.variant_retrieval(bench["tasks"], bench["source"],
                                 {"c1": BoundedLagNccMatcher(max_lag_samples=4)})
