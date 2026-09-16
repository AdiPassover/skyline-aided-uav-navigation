"""Metric-correctness tests for the skyline evaluator (spec 004).

Mostly in-memory: construct QueryResult + QueryGroundTruth objects with known values and
assert the frozen metric definitions. Pure metric functions read only `query_set.queries`,
so a lightweight SkylineQuerySet with `dataset=None` suffices. One end-to-end test runs the
CLI over the committed mock fixture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from naveval import skyline
from naveval.evaluate_skyline import _load_config, run_skyline_evaluation
from naveval.skyline_record import (
    QueryGroundTruth,
    QueryResult,
    SkylineQuerySet,
    SkylineResultRecord,
    SkylineRunManifest,
)

FIXTURES = Path(__file__).parent / "fixtures" / "skyline"


def _result(qid, outcome="SUCCESS", east=None, north=None, heading=None, source=None,
            prior=None, conf=None, matched_e=None, matched_n=None, pt=1_000_000, candidates=()):
    return QueryResult(
        query_id=qid, timestamp_s=float(qid), outcome=outcome, est_east_m=east, est_north_m=north,
        est_heading_deg=heading, heading_source=source, compass_prior_deg=prior, confidence=conf,
        best_score=None, second_best_score=None, score_margin=None, n_viable_candidates=None,
        alignment_residual=None, matched_east_m=matched_e, matched_north_m=matched_n,
        process_time_ns=pt, candidates=candidates,
    )


def _gt(qi, east=None, north=None, heading=None, roll=None, pitch=None, prior=None, in_cov=True):
    return QueryGroundTruth(query_index=qi, east_m=east, north_m=north, heading_deg=heading,
                            roll_deg=roll, pitch_deg=pitch, compass_prior_deg=prior,
                            in_coverage=in_cov, conditions={})


def _bundle(pairs):
    """pairs: list of (QueryResult, QueryGroundTruth). Returns (record, query_set)."""
    results = [r for r, _ in pairs]
    queries = {int(r.query_id): gt for r, gt in pairs}
    manifest = SkylineRunManifest(
        schema_version="1.0.0", run_id="r", query_set_id="qs", query_set_revision="1",
        relocalizer_id="", relocalizer_version="", relocalizer_config={},
        reference_db={"total_bytes": 51200, "n_references": 100, "covered_area_km2": 2.0,
                      "sample_spacing_m": 20.0, "camera_height_m": 2.0},
        compass_prior={}, environment={"is_target_hardware": False, "peak_rss_mb": 64},
        run_timestamp="t", query_count=len(results), processed_count=len(results), completed=True,
    )
    return SkylineResultRecord(manifest=manifest, results=results), \
        SkylineQuerySet(dataset=None, camera={}, compass_prior_spec={}, queries=queries)


# --- E1 position ---

def test_enu_position_error_known_coords():
    r = _result("0", east=103.0, north=204.0, matched_e=100.0, matched_n=200.0)
    gt = _gt(0, east=100.0, north=200.0)
    assert skyline.position_error_m(r, gt) == pytest.approx(5.0)  # 3-4-5


def test_position_error_none_without_position_or_gt():
    assert skyline.position_error_m(_result("0", outcome="REJECTED"), _gt(0, 1, 2)) is None
    assert skyline.position_error_m(_result("0", east=1, north=2), _gt(0)) is None


# --- E1/heading: circular wraparound + heading-source ---

def test_heading_wraparound_359_vs_1_is_two_degrees():
    assert skyline.heading_error_deg(359.0, 1.0) == pytest.approx(2.0)
    assert skyline.heading_error_deg(1.0, 359.0) == pytest.approx(2.0)


def test_heading_prior_vs_result_improvement():
    # prior 20 deg off, skyline result 2 deg off -> improvement 18
    r = _result("0", east=1, north=1, matched_e=1, matched_n=1, heading=88.0,
                source="skyline_estimated", prior=70.0, conf=0.9)
    gt = _gt(0, east=1, north=1, heading=90.0)
    rec, qs = _bundle([(r, gt)])
    h = skyline.heading_metrics(rec, qs)
    assert h["result_stats"]["median"] == pytest.approx(2.0)
    assert h["prior_stats"]["median"] == pytest.approx(20.0)
    assert h["improvement_stats"]["median"] == pytest.approx(18.0)


def test_missing_heading_ground_truth_yields_no_heading_stats():
    r = _result("0", east=1, north=1, matched_e=1, matched_n=1, heading=90.0,
                source="compass_prior", prior=90.0, conf=0.9)
    gt = _gt(0, east=1, north=1, heading=None)  # no heading GT
    rec, qs = _bundle([(r, gt)])
    h = skyline.heading_metrics(rec, qs)
    assert h["result_stats"] is None and h["prior_stats"] is None


def test_absent_prior_excluded_from_prior_stats():
    r = _result("0", east=1, north=1, matched_e=1, matched_n=1, heading=91.0,
                source="skyline_estimated", prior=None, conf=0.9)  # no compass prior
    gt = _gt(0, east=1, north=1, heading=90.0)
    rec, qs = _bundle([(r, gt)])
    h = skyline.heading_metrics(rec, qs)
    assert h["result_stats"]["median"] == pytest.approx(1.0)
    assert h["prior_stats"] is None  # prior absent -> not counted


# --- E1 region-in-candidates ---

def test_region_in_candidates_top1():
    pairs = [
        (_result("0", east=101, north=201, matched_e=100, matched_n=200, conf=0.9), _gt(0, 100, 200)),  # hit
        (_result("1", east=900, north=900, matched_e=900, matched_n=900, conf=0.9), _gt(1, 0, 0)),      # miss
    ]
    rec, qs = _bundle(pairs)
    region = skyline.region_in_candidates(rec, qs, tolerance_m=50.0, k=1)
    assert region["top1_rate"] == pytest.approx(0.5)
    assert region["recall_at_k_rate"] is None  # no shortlist exposed


# --- E2 reliability: rejected vs false-success, confident false ---

def test_rejected_not_counted_as_false_success():
    pairs = [
        (_result("0", outcome="REJECTED", conf=0.2), _gt(0, 0, 0)),
        (_result("1", east=999, north=999, matched_e=999, matched_n=999, conf=0.9), _gt(1, 0, 0)),  # SUCCESS wrong
    ]
    rec, qs = _bundle(pairs)
    rel = skyline.reliability_metrics(rec, qs, tolerance_m=50.0, reject_threshold=0.5)
    assert rel["rejection_rate"] == pytest.approx(0.5)
    assert rel["false_relocalization_rate"] == pytest.approx(0.5)  # only the SUCCESS
    assert rel["confident_false_relocalization_rate"] == pytest.approx(0.5)  # conf 0.9 >= 0.5


def test_low_confidence_false_success_not_confident():
    r = _result("0", east=999, north=999, matched_e=999, matched_n=999, conf=0.3)  # wrong, low conf
    rec, qs = _bundle([(r, _gt(0, 0, 0))])
    rel = skyline.reliability_metrics(rec, qs, tolerance_m=50.0, reject_threshold=0.5)
    assert rel["false_relocalization_rate"] == pytest.approx(1.0)
    assert rel["confident_false_relocalization_rate"] == pytest.approx(0.0)


def test_success_on_out_of_coverage_is_false_relocalization():
    r = _result("0", east=10, north=10, matched_e=10, matched_n=10, conf=0.9)
    gt = _gt(0, east=10, north=10, in_cov=False)  # position matches but out of coverage
    rec, qs = _bundle([(r, gt)])
    rel = skyline.reliability_metrics(rec, qs, tolerance_m=50.0, reject_threshold=0.5)
    assert rel["false_relocalization_rate"] == pytest.approx(1.0)
    assert rel["confident_false_relocalization_rate"] == pytest.approx(1.0)


def test_correct_out_of_coverage_report_is_not_false():
    r = _result("0", outcome="OUT_OF_COVERAGE")
    gt = _gt(0, east=10, north=10, in_cov=False)
    rec, qs = _bundle([(r, gt)])
    rel = skyline.reliability_metrics(rec, qs, tolerance_m=50.0, reject_threshold=0.5)
    assert rel["out_of_coverage_rate"] == pytest.approx(1.0)
    assert rel["false_relocalization_rate"] == pytest.approx(0.0)


# --- E2 calibration gating ---

def test_calibration_omitted_when_thin():
    r = _result("0", east=1, north=1, matched_e=1, matched_n=1, conf=0.9)
    rec, qs = _bundle([(r, _gt(0, 1, 1))])
    assert skyline.confidence_calibration(rec, qs, 50.0, n_bins=3, min_samples=10) is None


def test_calibration_present_with_enough_data():
    pairs = []
    for i in range(12):
        wrong = i % 2 == 0
        e = 999 if wrong else 1
        pairs.append((_result(str(i), east=e, north=1, matched_e=e, matched_n=1, conf=0.9),
                      _gt(i, 1, 1)))
    rec, qs = _bundle(pairs)
    calib = skyline.confidence_calibration(rec, qs, 50.0, n_bins=3, min_samples=10)
    assert calib is not None and calib["n"] == 12


# --- E4 latency + RAM ---

def test_latency_and_resource_flags():
    r = _result("0", outcome="REJECTED", conf=0.2, pt=5_000_000)
    rec, qs = _bundle([(r, _gt(0))])
    lat = skyline.latency_metrics(rec)
    assert lat["latency"]["median_ms"] == pytest.approx(5.0)
    assert lat["is_target_hardware"] is False
    assert lat["peak_rss_mb"] == 64


def test_missing_peak_rss_is_none():
    r = _result("0", outcome="REJECTED", conf=0.2)
    rec, qs = _bundle([(r, _gt(0))])
    rec.manifest.environment.pop("peak_rss_mb")
    assert skyline.latency_metrics(rec)["peak_rss_mb"] is None


# --- E5 footprint + sample spacing ---

def test_database_footprint_and_sample_spacing():
    r = _result("0", outcome="REJECTED", conf=0.2)
    rec, qs = _bundle([(r, _gt(0))])
    fp = skyline.database_footprint(rec)
    assert fp["bytes_per_reference"] == pytest.approx(512.0)
    assert fp["bytes_per_km2"] == pytest.approx(25600.0)  # 51200 / 2.0
    assert fp["sample_spacing_m"] == 20.0


# --- E3 slices ---

def test_condition_slices_prior_and_roll():
    pairs = [
        (_result("0", east=1, north=1, matched_e=1, matched_n=1, prior=90, conf=0.9), _gt(0, 1, 1, heading=90, roll=2, prior=90)),
        (_result("1", east=1, north=1, matched_e=1, matched_n=1, prior=120, conf=0.9), _gt(1, 1, 1, heading=90, roll=20, prior=120)),
    ]
    rec, qs = _bundle(pairs)
    slices = skyline.condition_slices(rec, qs, 50.0, {"roll_bins": [0, 5, 90], "prior_error_bins": [0, 5, 180]})
    assert "prior_error_deg" in slices and "roll_magnitude_deg" in slices
    assert "coverage" in slices


# --- end-to-end over committed mock fixture ---

def test_end_to_end_metrics_json(tmp_path):
    cfg = _load_config(FIXTURES / "eval-mock.json")
    out = run_skyline_evaluation(cfg, tmp_path / "out")
    names = {e["name"]: e for e in out["metrics"]}
    assert names["success_rate"]["value"] == pytest.approx(0.6)
    assert names["false_relocalization_rate"]["value"] == pytest.approx(0.2)
    assert names["confident_false_relocalization_rate"]["value"] == pytest.approx(0.2)
    assert names["position_error_median"]["value"] == pytest.approx(7.0710678, abs=1e-4)
    assert out["evidence_tier"] == "T1"
    assert out["runtime"]["disclaimer"] is not None  # is_target_hardware False
    # determinism: re-run reproduces identical metrics.json
    out2 = run_skyline_evaluation(cfg, tmp_path / "out2")
    assert json.dumps(out, sort_keys=True) == json.dumps(out2, sort_keys=True)
