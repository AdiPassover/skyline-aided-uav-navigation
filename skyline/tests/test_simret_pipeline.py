"""The simulator chain end to end on synthetic data: ingest → groups → tasks → sets → matcher → evaluator.

This is the **E0 plumbing proof**. It runs the real modules — the real adapter, the real GT
conversion, the real pose geometry, the real set writers, the **unchanged** ``hsreloc.retrieval``
matcher and the **unchanged** ``naveval`` evaluator — over a constructed simulator run, so that when a
real run arrives the only new thing in the path is the data.

It deliberately makes **no retrieval-quality claim**. The corpus is constructed; its curves were
chosen to be well-behaved, and the assertions are about the chain (records parse, provenance is
right, condition axes arrive at the evaluator, the three sources produce identical sets) and never
about how well anything matched.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from hsreloc.observation import read_session
from hsreloc.retrieval import runconfig as rc
from hsreloc.retrieval.run import execute
from hsreloc.simret import adapter, geometry, groups, sets, sources
from tests.fixtures import sim_run

SPACING = 40.0


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Two ingested sessions: a North-bound reference line and a laterally offset query line."""
    root = tmp_path_factory.mktemp("sim_store")
    obs_root = root / "observations_sim"
    ref = sim_run.build_run(root / "Run_ref", sim_run.line_north(12, spacing_m=10.0, east_m=0.0),
                            run_id="Run_ref")
    qry = sim_run.build_run(root / "Run_lat10", sim_run.line_north(12, spacing_m=10.0, east_m=10.0),
                            run_id="Run_lat10", time_of_day="DUSK", hour=18.0, clouds="VERY_CLOUDY")
    adapter.ingest_run(ref["run_dir"], obs_root, session_id="ref_day_clear")
    adapter.ingest_run(qry["run_dir"], obs_root, session_id="lat10_dusk_cloudy")
    return {"root": root, "obs_root": obs_root, "width": ref["width"], "height": ref["height"]}


@pytest.fixture(scope="module")
def sessions(store):
    return {sid: read_session(store["obs_root"] / sid)[1]
            for sid in ("ref_day_clear", "lat10_dusk_cloudy")}


@pytest.fixture(scope="module")
def tasks(store, sessions, tmp_path_factory):
    built = geometry.build_tasks(sessions, {
        "reference_session": "ref_day_clear",
        "query_sessions": ["lat10_dusk_cloudy"],
        "spacings_m": [SPACING],
    })
    out = tmp_path_factory.mktemp("sim_tasks")
    geometry.write_tasks(built, out)
    built["manifest"] = geometry.load_tasks(out)["manifest"]
    return built


# -- Part D: the experiment sidecar --------------------------------------------------------------

def test_the_index_derives_geometry_and_takes_identity_from_the_sidecar(store, tmp_path):
    anchors = tmp_path / "anchors.csv"
    anchors.write_text(
        "anchor_id,east_m,north_m,up_m,heading_deg,radius_m,notes\n"
        "ridge_a,0,0,60,0,3,the reference anchor\n"
        "ridge_a_e10,10,0,60,0,3,10 m East of ridge_a\n", encoding="utf-8")
    index = groups.build_index(store["obs_root"], anchors_csv=anchors)
    by_id = {r["observation_id"]: r for r in index["rows"]}

    ref0 = by_id["ref_day_clear__sky_000001"]
    assert ref0["anchor_id"] == "ridge_a"
    assert ref0["condition_anchor_translation_m"] == pytest.approx(0.0)
    assert ref0["condition_time_of_day"] == "DAY"
    assert ref0["condition_clouds"] == "CLEAR"

    lat0 = by_id["lat10_dusk_cloudy__sky_000001"]
    assert lat0["anchor_id"] == "ridge_a_e10"
    assert lat0["condition_time_of_day"] == "DUSK"
    # 10 m East of a North-facing anchor is pure lateral displacement, no along-track component.
    assert lat0["condition_anchor_lateral_m"] == pytest.approx(0.0, abs=1e-9)
    assert lat0["condition_anchor_along_m"] == pytest.approx(0.0, abs=1e-9)

    # Frames beyond the anchors' radii are counted, never silently dropped.
    assert index["manifest"]["n_unassigned"] == len(index["rows"]) - 2
    assert index["manifest"]["assignment_rule"].endswith("NEVER assign a group.")


def test_lateral_and_longitudinal_split_is_relative_to_the_anchor_heading(store, tmp_path):
    anchors = tmp_path / "anchors.csv"
    anchors.write_text("anchor_id,east_m,north_m,up_m,heading_deg,radius_m\n"
                       "north_facing,0,0,60,0,200\n", encoding="utf-8")
    index = groups.build_index(store["obs_root"], session_ids=["lat10_dusk_cloudy"],
                               anchors_csv=anchors)
    row = next(r for r in index["rows"] if r["north_m"] == pytest.approx(30.0))
    assert row["condition_anchor_lateral_m"] == pytest.approx(10.0)   # East of a North-facing anchor
    assert row["condition_anchor_along_m"] == pytest.approx(30.0)     # 30 m up the line
    assert row["condition_translation_bin"] is not None


def test_same_pose_frames_share_a_pose_group_across_conditions(store, tmp_path):
    """E2's grouping is objective: two frames group because their poses agree, not their names."""
    index = groups.build_index(store["obs_root"])
    keys = {r["observation_id"]: r["condition_pose_group"] for r in index["rows"]}
    # ref and lat10 lines are 10 m apart, so no pose is shared between them.
    assert len(set(keys.values())) == len(keys)
    ref = next(o for o in read_session(store["obs_root"] / "ref_day_clear")[1])
    assert groups.pose_group_key(ref) == keys[ref.observation_id]


def test_explicit_assignments_win_over_proximity_and_unknown_anchors_are_refused(store, tmp_path):
    anchors = tmp_path / "anchors.csv"
    anchors.write_text("anchor_id,east_m,north_m,radius_m\nA,0,0,3\n", encoding="utf-8")
    assign = tmp_path / "assignments.csv"
    assign.write_text("observation_id,anchor_id,role,condition_group,notes\n"
                      "lat10_dusk_cloudy__sky_000005,A,reference,dusk_set,hand-picked\n",
                      encoding="utf-8")
    index = groups.build_index(store["obs_root"], anchors_csv=anchors, assignments_csv=assign)
    row = next(r for r in index["rows"] if r["observation_id"] == "lat10_dusk_cloudy__sky_000005")
    assert row["anchor_id"] == "A" and row["role"] == "reference"
    assert row["condition_group"] == "dusk_set" and row["notes"] == "hand-picked"

    bad = tmp_path / "bad.csv"
    bad.write_text("observation_id,anchor_id\nlat10_dusk_cloudy__sky_000001,NOPE\n", encoding="utf-8")
    with pytest.raises(groups.GroupError, match="anchors.csv does not declare"):
        groups.build_index(store["obs_root"], anchors_csv=anchors, assignments_csv=bad)


def test_index_round_trips_and_selects(store, tmp_path):
    index = groups.build_index(store["obs_root"])
    manifest = groups.write_index(index, tmp_path / "idx")
    assert manifest["content_digest"]
    back = groups.load_index(tmp_path / "idx")
    assert len(back["rows"]) == len(index["rows"])
    dusk = groups.select(back, condition_time_of_day="DUSK")
    assert len(dusk) == 12 and all(r["session_id"] == "lat10_dusk_cloudy" for r in dusk)


def test_a_malformed_sidecar_is_refused_by_line_number(tmp_path):
    p = tmp_path / "anchors.csv"
    p.write_text("anchor_id,east_m,north_m\nA,0,0\nA,5,5\n", encoding="utf-8")
    with pytest.raises(groups.GroupError, match="duplicate anchor_id"):
        groups.read_anchors(p)
    p.write_text("anchor_id,east_m\nA,0\n", encoding="utf-8")
    with pytest.raises(groups.GroupError, match="missing required column"):
        groups.read_anchors(p)


# -- Part C: the three interchangeable sources ---------------------------------------------------

def test_sim_exact_source_returns_ground_truth_curves(store, sessions):
    smap = sources.session_map(store["obs_root"])
    src = sources.SimExactSource(store["obs_root"], smap, store["width"], store["height"])
    curve = src.get("ref_day_clear__sky_000001")
    assert curve.provenance == "oracle:sim_exact"
    assert curve.row_per_col.size == store["width"]
    assert src.describe()["gt"] is True


def test_dp_source_reads_what_the_frozen_extractor_wrote(store):
    summary = sources.extract_dp_session(store["obs_root"], "ref_day_clear")
    assert summary["method"] == "poc_robust_dp"
    assert summary["n_images"] == 12
    smap = sources.session_map(store["obs_root"], ["ref_day_clear"])
    src = sources.SessionDPSource(store["obs_root"], smap, store["width"], store["height"])
    if summary["stored"]:
        curve = src.get(summary["stored"][0])
        assert curve.provenance == "automatic:poc_robust_dp"
        assert curve.row_per_col.size == store["width"]
    else:                                     # every frame refused: still the contracted behaviour
        from hsreloc.retrieval.skyline_curve import CurveError
        with pytest.raises(CurveError):
            src.get("ref_day_clear__sky_000001")


def test_segformer_source_is_keyed_by_session_and_carries_its_pinned_identifiers(store, tmp_path):
    """The ECL source keys label maps by condition; the simulator source keys them by session. That
    is the only difference — revision, class index and conversion are the frozen ones."""
    mask_root = tmp_path / "silver"
    (mask_root / "ref_day_clear" / "s512").mkdir(parents=True)
    _, observations = read_session(store["obs_root"] / "ref_day_clear")
    for obs in observations:
        label = np.full((store["height"], store["width"]), 7, dtype=np.uint8)
        label[:store["height"] // 3, :] = 2                # ADE20K class 2 == sky
        np.savez_compressed(mask_root / "ref_day_clear" / "s512" / f"{obs.observation_id}.npz",
                            label=label)
    (mask_root / "inference_manifest.json").write_text(json.dumps({
        "model": {"repo_id": "nvidia/segformer-b0-finetuned-ade-512-512",
                  "revision": "489d5cd8", "sky_class_index": 2},
        "keying": "session", "output_digest": "deadbeef"}), encoding="utf-8")

    smap = sources.session_map(store["obs_root"], ["ref_day_clear"])
    src = sources.SessionSilverSource(mask_root, smap, store["width"], store["height"],
                                      expected_model_revision="489d5cd8")
    curve = src.get("ref_day_clear__sky_000001")
    assert curve.provenance == "automatic:segformer_b0_ade20k"
    assert np.allclose(curve.row_per_col, store["height"] // 3 - 0.5)
    d = src.describe()
    assert d["silver_label"] is True and d["model_revision"] == "489d5cd8"
    assert d["convert_version"] == "1.0.0"
    assert "never ground truth" in d["notes"]

    with pytest.raises(sources.SourceError, match="expected 'other'"):
        sources.SessionSilverSource(mask_root, smap, store["width"], store["height"],
                                    expected_model_revision="other")


def test_a_refused_query_curve_becomes_a_sentinel_and_a_reference_stays_a_hole(store, tmp_path):
    from hsreloc.retrieval.skyline_curve import CurveError
    empty = tmp_path / "silver_empty"
    empty.mkdir()
    (empty / "inference_manifest.json").write_text(json.dumps(
        {"model": {"revision": "489d5cd8", "sky_class_index": 2}}), encoding="utf-8")
    smap = sources.session_map(store["obs_root"], ["ref_day_clear"])
    raising = sources.SessionSilverSource(empty, smap, store["width"], store["height"])
    with pytest.raises(CurveError, match="no label map"):
        raising.get("ref_day_clear__sky_000001")
    sentinel = sources.SessionSilverSource(empty, smap, store["width"], store["height"],
                                           refusal_mode="sentinel")
    curve = sentinel.get("ref_day_clear__sky_000001")
    assert np.allclose(curve.row_per_col, 0.0)          # flat -> the matcher's degenerate rule fires
    assert curve.source_ref.startswith("sentinel:")


# -- tasks and sets ------------------------------------------------------------------------------

def test_the_task_builder_produces_a_grid_and_pose_defined_correctness(tasks):
    grid = tasks["grids"][f"{SPACING:g}"]
    assert grid["n_references"] == 3            # 110 m of line at 40 m spacing
    assert grid["tau_pos_m"] == SPACING / 2
    q = tasks["queries"][0]
    assert q["condition_stage"] in geometry.STAGES
    assert q["condition_lateral_m"] == pytest.approx(10.0)   # the query line is 10 m East
    assert q["condition_yaw_diff_deg"] == pytest.approx(0.0)


def test_query_and_reference_sets_are_written_in_the_frozen_contracts(store, sessions, tasks, tmp_path):
    obs_by_id = {o.observation_id: o for s in sessions.values() for o in s}
    meta = json.loads((store["obs_root"] / "ref_day_clear" / "session.json").read_text(encoding="utf-8"))
    out = sets.write_query_set(tasks, "e3_lateral", SPACING, "sim_exact", tmp_path / "datasets",
                               obs_by_id, store["width"], store["height"], session_meta=meta)
    ds = json.loads((out / "dataset.json").read_text(encoding="utf-8"))
    assert ds["source_type"] == "ue5_simulator"
    assert ds["evidence_tier"] == "T2" and ds["evidence_caveat"]
    assert ds["local_frame_origin"]["is_synthetic"] is True
    assert ds["position_quality"]["class"] == "simulator_exact"
    assert ds["heading_quality"]["class"] == "simulator_exact"
    with (out / "skyline_queries.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["condition_stage"] == "stage1"
    assert rows[0]["condition_time_of_day"] == "DUSK"
    assert rows[0]["oracle_provenance"] == "sim_exact"
    assert float(rows[0]["compass_prior_deg"]) == pytest.approx(0.0)
    with (out / "groundtruth.csv").open(encoding="utf-8") as f:
        gt = list(csv.DictReader(f))
    assert gt[0]["fix_quality"] == "simulator_exact"
    assert gt[0]["heading_deg"] == "0"           # heading is known and written, unlike ECL

    smap = sources.session_map(store["obs_root"])
    src = sources.SimExactSource(store["obs_root"], smap, store["width"], store["height"])
    manifest = sets.write_reference_set(tasks, "e3_lateral", SPACING, "sim_exact", src,
                                        tmp_path / "refdb")
    assert manifest["reference_source"]["n_references"] == 3
    assert manifest["reference_source"]["n_holes"] == 0
    assert manifest["reference_source"]["extraction_mode"] == "oracle:sim_exact"
    assert manifest["local_frame_origin"] == sets.SYNTHETIC_ORIGIN


def test_the_three_sources_differ_only_in_the_provenance_column(store, sessions, tasks, tmp_path):
    obs_by_id = {o.observation_id: o for s in sessions.values() for o in s}
    for key in ("sim_exact", "segformer", "dp"):
        sets.write_query_set(tasks, "e3_lateral", SPACING, key, tmp_path / "datasets",
                             obs_by_id, store["width"], store["height"])
    sets.assert_source_materialisations_identical(tmp_path / "datasets", "e3_lateral", SPACING,
                                                  ["sim_exact", "segformer", "dp"])


# -- the unchanged matcher and the unchanged evaluator -------------------------------------------

def test_e0_the_whole_chain_runs_and_the_record_parses(store, sessions, tasks, tmp_path):
    obs_by_id = {o.observation_id: o for s in sessions.values() for o in s}
    meta = json.loads((store["obs_root"] / "ref_day_clear" / "session.json").read_text(encoding="utf-8"))
    qs_dir = sets.write_query_set(tasks, "e3_lateral", SPACING, "sim_exact", tmp_path / "datasets",
                                  obs_by_id, store["width"], store["height"], session_meta=meta)
    smap = sources.session_map(store["obs_root"])
    ref_source = sources.SimExactSource(store["obs_root"], smap, store["width"], store["height"])
    sets.write_reference_set(tasks, "e3_lateral", SPACING, "sim_exact", ref_source, tmp_path / "refdb")

    config = rc.parse_config({
        "schema_version": "1.1.0", "run_id": "sim-e0-plumbing",
        "inputs": {
            "reference_set": str(tmp_path / "refdb" / sets.reference_set_id("e3_lateral", SPACING, "sim_exact")),
            "query_set": str(qs_dir),
            "query_curves": {"kind": "observation_store", "root": str(store["obs_root"]),
                             "expect_provenance": "oracle:sim_exact"},
        },
        "geodetic_origin": {k: sets.SYNTHETIC_ORIGIN[k] for k in ("lat_deg", "lon_deg", "alt_m")},
        "profile": {"n_samples": 256, "normalize_mean": True, "detrend": False,
                    "units": "image_fraction"},
        "match": {"baseline": "ncc", "recall_k": 5, "lag_search": False},
        "acceptance": {"confidence_k": 5.0, "reject_threshold": 0.5, "viable_sigma": 3.0,
                       "ambiguity_margin_sigma": 1.0, "degenerate_profile_std": 1e-06},
        "output": {"root": str(tmp_path / "runs")},
        "evidence": {"tier": "T2", "caveat": sets.CAVEAT},
        "environment": {"is_target_hardware": False},
    }, base_dir=tmp_path)

    query_source = sources.SimExactSource(store["obs_root"], smap, store["width"], store["height"],
                                          refusal_mode="sentinel")
    summary = execute(config, curve_source=query_source)
    record_dir = Path(summary["record_dir"]) if "record_dir" in summary else tmp_path / "runs" / "sim-e0-plumbing"
    manifest = json.loads((record_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["relocalizer_config"]["curve_source"]["provenance"] == "oracle:sim_exact"
    assert manifest["reference_source"]["evidence_tier"] == "T2"
    assert manifest["reference_source"]["evidence_caveat"]
    assert manifest["completed"] is True
    assert manifest["environment"]["is_target_hardware"] is False
    with (record_dir / "queries.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(tasks["queries"])
    assert all(r["outcome"] for r in rows)

    # The record is readable by the unchanged evaluator's own reader.
    from naveval.skyline_record import load_skyline_record
    assert load_skyline_record(record_dir).manifest.run_id == "sim-e0-plumbing"


def test_the_evaluator_slices_by_the_simulator_condition_axes(store, sessions, tasks, tmp_path):
    """The unchanged ``naveval`` must find the condition columns the geometry emitted — that is what
    makes "performance versus translation" a query against the evaluator rather than new code."""
    from naveval.skyline_record import load_skyline_query_set

    obs_by_id = {o.observation_id: o for s in sessions.values() for o in s}
    qs_dir = sets.write_query_set(tasks, "e3_lateral", SPACING, "sim_exact", tmp_path / "datasets",
                                  obs_by_id, store["width"], store["height"])
    loaded = load_skyline_query_set(qs_dir)
    factors = json.loads((qs_dir / "dataset.json").read_text(encoding="utf-8"))["skyline"]["condition_factors"]
    for axis in ("condition_translation_m", "condition_lateral_m", "condition_along_m",
                 "condition_abs_yaw_diff_deg", "condition_stage", "condition_translation_bin"):
        assert axis in factors
    first = loaded.queries[0]
    assert first.conditions["condition_stage"] == "stage1"
    assert float(first.conditions["condition_lateral_m"]) == pytest.approx(10.0)
    assert first.conditions["condition_time_of_day"] == "DUSK"
    assert loaded.dataset.evidence_tier == "T2"
    # Heading and attitude survive into the evaluator's per-query ground truth, unlike ECL's.
    assert first.heading_deg == pytest.approx(0.0)
    assert first.compass_prior_deg == pytest.approx(0.0)


# -- the pre-registered run mechanism (research R3): prereg -> run -> evaluate --------------------

@pytest.fixture()
def pilot_dir(tmp_path):
    """A miniature pilot: one raw batch, a full config, everything driven through the CLI."""
    from hsreloc.simret import cli

    raw = tmp_path / "raw"
    sim_run.build_run(raw / "Run_ref", sim_run.line_north(12, spacing_m=10.0), run_id="Run_ref")
    sim_run.build_run(raw / "Run_lat", sim_run.line_north(6, spacing_m=10.0, east_m=10.0),
                      run_id="Run_lat", time_of_day="DUSK", hour=18.0, clouds="VERY_CLOUDY",
                      nested_observations=True)
    cfg_dir = tmp_path / "configs"
    (cfg_dir / "frozen").mkdir(parents=True)
    cfg = {
        "raw_root": str(raw), "store_root": str(tmp_path / "obs"),
        "task_name": "mini",
        "tasks": {"reference_observation_ids": ["Run_ref__sky_000001", "Run_ref__sky_000007"],
                   "query_sessions": ["Run_lat"], "deform_query_sessions": ["Run_lat"],
                   "spacings_m": [40.0]},
        "tasks_dir": str(tmp_path / "tasks_pilot"),
        "deform_tasks_dir": str(tmp_path / "tasks_deform"),
        "datasets_root": str(tmp_path / "datasets"), "refdb_root": str(tmp_path / "refdb"),
        "runs_root": str(tmp_path / "runs"), "evaluations_root": str(tmp_path / "evals"),
        "evaluation_configs_root": str(tmp_path / "eval_cfgs"),
        "sources": {"sim_exact": {}},
        "matcher": {"profile": {"n_samples": 256, "normalize_mean": True, "detrend": False,
                                 "units": "image_fraction"},
                     "acceptance": {"confidence_k": 5.0, "reject_threshold": 0.5,
                                    "viable_sigma": 3.0, "ambiguity_margin_sigma": 1.0,
                                    "degenerate_profile_std": 1e-06},
                     "recall_k": 2, "lag_search": False},
        "baselines": ["ncc"],
        "geodetic_origin": {"lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "is_synthetic": True},
        "run_id_pattern": "mini-s{spacing}-{baseline}-{source}",
        "preregistration_ref": "frozen/mini.prereg",
    }
    cfg_path = cfg_dir / "mini.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert cli.main(["ingest", "--config", str(cfg_path)]) == 0
    assert cli.main(["tasks", "--config", str(cfg_path)]) == 0
    assert cli.main(["sets", "--config", str(cfg_path)]) == 0
    return {"cfg": cfg, "cfg_path": cfg_path, "tmp": tmp_path}


def test_run_refuses_without_a_preregistration_then_runs_and_keeps_the_record(pilot_dir):
    from hsreloc.simret import cli
    from hsreloc.simret.prereg import PreregError

    cfg_path = pilot_dir["cfg_path"]
    with pytest.raises(PreregError, match="COMMIT it before"):
        cli.main(["run", "--config", str(cfg_path)])

    assert cli.main(["prereg", "--config", str(cfg_path)]) == 0
    pre_path = cfg_path.parent / "frozen" / "mini.prereg"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    assert pre["task"]["reference_observation_ids"] == ["Run_ref__sky_000001", "Run_ref__sky_000007"]
    assert pre["sources"]["sim_exact"]["gt"] is True
    assert pre["chance_baseline"]["n_references"] == 6      # the declared pilot constant
    assert "case_4_geometry_dominated" in pre["interpretation_matrix"]

    assert cli.main(["run", "--config", str(cfg_path)]) == 0
    record_dir = pilot_dir["tmp"] / "runs" / "mini-s40-ncc-sim_exact"
    manifest = json.loads((record_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] and manifest["processed_count"] == manifest["query_count"]

    # A second run keeps the complete record rather than redoing or overwriting it.
    assert cli.main(["run", "--config", str(cfg_path)]) == 0

    # A changed experiment is refused: the config no longer matches the frozen pre-registration.
    tampered = dict(pilot_dir["cfg"])
    tampered["matcher"] = json.loads(json.dumps(tampered["matcher"]))
    tampered["matcher"]["acceptance"]["reject_threshold"] = 0.4
    tampered_path = cfg_path.parent / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PreregError, match="matcher block"):
        cli.main(["run", "--config", str(tampered_path)])


def test_evaluate_scores_every_record_with_the_unmodified_evaluator(pilot_dir):
    from hsreloc.simret import cli

    cfg_path = pilot_dir["cfg_path"]
    assert cli.main(["prereg", "--config", str(cfg_path)]) == 0
    assert cli.main(["run", "--config", str(cfg_path)]) == 0
    assert cli.main(["evaluate", "--config", str(cfg_path)]) == 0
    out = pilot_dir["tmp"] / "evals" / "sim-retrieval-mini-s40-ncc-sim_exact"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert "retrieval" in metrics and "topological_success" in metrics["retrieval"]
    # the generated evaluator config pinned the task's own tolerances
    gen = json.loads((pilot_dir["tmp"] / "eval_cfgs" / "eval-mini-s40-ncc-sim_exact.json")
                     .read_text(encoding="utf-8"))
    assert gen["operational_tolerance_m"] == 20.0 and gen["near_tolerance_m"] == 40.0
    assert gen["tier_pooling"] == "forbidden"
