"""Spec 006 — additive evaluator metrics + the eight synthetic known-answer fixtures.

The evaluator is validated **only** on deterministic, hand-constructed result records (no matcher,
no dataset), following the in-memory `_bundle` pattern of `test_skyline_metrics.py`. When the
perfect / random / aliased / near-positive / out-of-database / confident-wrong / oracle-vs-automatic
/ condition-tier fixtures all evaluate as asserted, the matcher-agnostic evaluator is trusted
(spec 006 FR-F1–FR-F8; the scientific boundary of DEC-013).

Metric grounding: VPR-Bench protocol (LIT-013) — AUC-PR primary, ROC for new-place rejection,
score-separation (d', ROC-AUC) for perceptual aliasing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from naveval import skyline
from naveval.errors import ContractViolationError
from naveval.evaluate_skyline import _load_config, run_skyline_evaluation
from naveval.skyline_record import (
    QueryGroundTruth,
    QueryResult,
    SkylineQuerySet,
    SkylineResultRecord,
    SkylineRunManifest,
)


def _result(qid, outcome="SUCCESS", east=None, north=None, conf=None, best=None,
            matched_e=None, matched_n=None, pt=1_000_000, candidates=()):
    return QueryResult(
        query_id=str(qid), timestamp_s=float(qid), outcome=outcome, est_east_m=east, est_north_m=north,
        est_heading_deg=None, heading_source=None, compass_prior_deg=None, confidence=conf,
        best_score=best, second_best_score=None, score_margin=None, n_viable_candidates=None,
        alignment_residual=None, matched_east_m=matched_e, matched_north_m=matched_n,
        process_time_ns=pt, candidates=tuple(candidates),
    )


def _gt(qi, east=None, north=None, in_cov=True, conditions=None):
    return QueryGroundTruth(query_index=qi, east_m=east, north_m=north, heading_deg=None,
                            roll_deg=None, pitch_deg=None, compass_prior_deg=None,
                            in_coverage=in_cov, conditions=conditions or {})


def _bundle(pairs, extraction_mode=None):
    results = [r for r, _ in pairs]
    queries = {int(r.query_id): gt for r, gt in pairs}
    reference_source = {
        "kind": "historical_imagery", "n_references": 100, "total_bytes": 51200,
        "covered_area_km2": 2.0, "reference_spacing_m": 100.0,
    }
    if extraction_mode:
        reference_source["extraction_mode"] = extraction_mode
    manifest = SkylineRunManifest(
        schema_version="1.0.0", run_id="r", query_set_id="qs", query_set_revision="1",
        relocalizer_id="", relocalizer_version="", relocalizer_config={},
        reference_db={}, compass_prior={}, environment={"is_target_hardware": False, "peak_rss_mb": 32},
        run_timestamp="t", query_count=len(results), processed_count=len(results), completed=True,
        reference_source=reference_source,
    )
    return SkylineResultRecord(manifest=manifest, results=results), \
        SkylineQuerySet(dataset=None, camera={}, compass_prior_spec={}, queries=queries)


# ===========================================================================
# Known-input metric oracles (hand-computed truth for each helper)
# ===========================================================================

def test_roc_auc_perfect_and_half():
    assert skyline._roc_auc([3, 4], [1, 2]) == pytest.approx(1.0)   # all pos > all neg
    assert skyline._roc_auc([1, 4], [2, 3]) == pytest.approx(0.5)   # 2 of 4 pairs
    assert skyline._roc_auc([1, 2], [1, 2]) == pytest.approx(0.5)   # ties count 0.5
    assert skyline._roc_auc([], [1]) is None


def test_d_prime_known_value_and_zero_variance():
    # pos=[2,3] neg=[0,1]: means 2.5 vs 0.5, var 0.25 each, denom=0.5 -> d'=4.0
    assert skyline._d_prime([2, 3], [0, 1]) == pytest.approx(4.0)
    # perfect separation, zero within-class variance -> None (rely on ROC-AUC)
    assert skyline._d_prime([2, 2], [0, 0]) is None


def test_auc_pr_perfect_ranking():
    # two positives ranked above one negative -> AP = 1.0
    assert skyline._auc_pr([(0.9, 1), (0.8, 1), (0.1, 0)]) == pytest.approx(1.0)
    # no positives -> None
    assert skyline._auc_pr([(0.5, 0), (0.4, 0)]) is None


def test_topological_success_boundary_at_tolerance():
    # exactly tolerance away counts as a hit (<=)
    r = _result(0, east=130.0, north=0.0, matched_e=130.0, matched_n=0.0, conf=0.9)
    rec, qs = _bundle([(r, _gt(0, east=0.0, north=0.0))])
    topo = skyline.topological_success(rec, qs, tolerance_m=130.0, k=1)
    assert topo["success_rate"] == pytest.approx(1.0)
    topo2 = skyline.topological_success(rec, qs, tolerance_m=129.0, k=1)
    assert topo2["success_rate"] == pytest.approx(0.0)


# ===========================================================================
# FR-F1..FR-F8 — the eight synthetic known-answer fixtures
# ===========================================================================

def _cands_true_first(gt_e, gt_n, true_score, wrong_score, far=1000.0):
    """A shortlist: the correct reference (at GT) scored `true_score`, one wrong reference far away
    scored `wrong_score`."""
    return [(gt_e, gt_n, true_score), (gt_e + far, gt_n + far, wrong_score)]


def test_FRF1_perfect_ranking():
    pairs = []
    for i in range(6):
        e, n = float(i * 10), 0.0
        pairs.append((_result(i, east=e, north=n, matched_e=e, matched_n=n, conf=0.95, best=0.95,
                              candidates=_cands_true_first(e, n, 0.95, 0.10)),
                      _gt(i, east=e, north=n)))
    rec, qs = _bundle(pairs)
    assert skyline.topological_success(rec, qs, 5.0)["success_rate"] == pytest.approx(1.0)
    sep = skyline.score_separation(rec, qs, 5.0, near_m=50.0)
    assert sep["roc_auc"] == pytest.approx(1.0)
    assert skyline.pr_roc(rec, qs, 5.0)["auc_pr"] == pytest.approx(1.0)
    rel = skyline.reliability_metrics(rec, qs, 5.0, 0.5)
    assert rel["confident_false_relocalization_rate"] == pytest.approx(0.0)


def test_FRF2_random_ranking_is_chance():
    # Interleave true/wrong scores so positives and negatives are indistinguishable -> AUC ~ 0.5.
    pairs = []
    scores = [(0.6, 0.4), (0.4, 0.6), (0.55, 0.45), (0.45, 0.55)]
    for i, (ts, ws) in enumerate(scores):
        e, n = float(i * 10), 0.0
        pairs.append((_result(i, east=e + 900, north=n + 900, matched_e=e + 900, matched_n=n + 900,
                              conf=ts, best=ts, candidates=_cands_true_first(e, n, ts, ws)),
                      _gt(i, east=e, north=n)))
    rec, qs = _bundle(pairs)
    auc = skyline.score_separation(rec, qs, 5.0, near_m=50.0)["roc_auc"]
    assert 0.35 <= auc <= 0.65  # near chance
    assert skyline.topological_success(rec, qs, 5.0)["success_rate"] == pytest.approx(0.0)


def test_FRF3_aliased_hidden_by_recall_but_caught_by_separation():
    # Correct reference is present but wrong references score just as high (heavy overlap):
    # recall@1 low, and score-separation reveals the aliasing (AUC well below 1).
    pairs = []
    for i in range(6):
        e, n = float(i * 10), 0.0
        # wrong candidate scores slightly HIGHER than the true one -> top-1 misses
        pairs.append((_result(i, east=e + 1000, north=n + 1000, matched_e=e + 1000, matched_n=n + 1000,
                              conf=0.80, best=0.80, candidates=[(e, n, 0.80), (e + 1000, n + 1000, 0.82)]),
                      _gt(i, east=e, north=n)))
    rec, qs = _bundle(pairs)
    assert skyline.topological_success(rec, qs, 5.0)["success_rate"] == pytest.approx(0.0)
    sep = skyline.score_separation(rec, qs, 5.0, near_m=50.0)
    assert sep["roc_auc"] is not None and sep["roc_auc"] < 0.7   # aliasing visible in separation
    assert sep["n_positive"] > 0 and sep["n_negative"] > 0


def test_FRF4_near_positive_ignore_band():
    # A candidate in the ignore band (tol < d <= near_m) must be counted as neither positive nor
    # negative, so it does not inflate the negative distribution.
    e, n = 0.0, 0.0
    cands = [(e, n, 0.9), (e + 30.0, n, 0.85), (e + 1000.0, n, 0.1)]  # true, near (30 m), distant
    r = _result(0, east=e, north=n, matched_e=e, matched_n=n, conf=0.9, best=0.9, candidates=cands)
    rec, qs = _bundle([(r, _gt(0, east=e, north=n))])
    sep = skyline.score_separation(rec, qs, tolerance_m=5.0, near_m=50.0)
    assert sep["n_positive"] == 1 and sep["n_near"] == 1 and sep["n_negative"] == 1
    alias = skyline.aliasing_by_distance(rec, qs, 5.0, 50.0)
    assert alias["nearby_non_match"]["n"] == 1 and alias["distant_non_match"]["n"] == 1


def test_FRF5_out_of_database():
    # One out-of-DB query correctly reported OUT_OF_COVERAGE; one confidently (wrongly) localized.
    pairs = [
        (_result(0, outcome="OUT_OF_COVERAGE"), _gt(0, east=10, north=10, in_cov=False)),
        (_result(1, east=10, north=10, matched_e=10, matched_n=10, conf=0.9, best=0.9),
         _gt(1, east=10, north=10, in_cov=False)),  # confident fix on a place not in the DB
    ]
    rec, qs = _bundle(pairs)
    rel = skyline.reliability_metrics(rec, qs, 50.0, 0.5)
    assert rel["out_of_coverage_rate"] == pytest.approx(0.5)
    assert rel["false_relocalization_rate"] == pytest.approx(0.5)          # the confident one
    assert rel["confident_false_relocalization_rate"] == pytest.approx(0.5)


def test_FRF6_confident_wrong_vs_rejected():
    pairs = [
        (_result(0, east=999, north=999, matched_e=999, matched_n=999, conf=0.9, best=0.9), _gt(0, 0, 0)),  # confident wrong
        (_result(1, outcome="REJECTED", conf=0.2), _gt(1, 0, 0)),                                            # rejected (not a fix)
    ]
    rec, qs = _bundle(pairs)
    rel = skyline.reliability_metrics(rec, qs, 50.0, 0.5)
    assert rel["confident_false_relocalization_rate"] == pytest.approx(0.5)   # only the confident wrong
    assert rel["rejection_rate"] == pytest.approx(0.5)                        # counted separately


def test_FRF7_oracle_vs_automatic_identical_metrics_distinct_provenance():
    def build(mode):
        pairs = [(_result(0, east=1, north=1, matched_e=1, matched_n=1, conf=0.9, best=0.9,
                          candidates=_cands_true_first(1, 1, 0.9, 0.1)), _gt(0, east=1, north=1))]
        return _bundle(pairs, extraction_mode=mode)

    rec_o, qs_o = build("oracle:manual")
    rec_a, qs_a = build("automatic:dp")
    # provenance is a label, not a metric input -> identical metrics ...
    assert (skyline.topological_success(rec_o, qs_o, 5.0)["success_rate"]
            == skyline.topological_success(rec_a, qs_a, 5.0)["success_rate"])
    # ... but the two are reported as DISTINCT rows via extraction_mode (FR-P5 discipline)
    assert rec_o.manifest.reference_source["extraction_mode"] == "oracle:manual"
    assert rec_a.manifest.reference_source["extraction_mode"] == "automatic:dp"
    assert rec_o.manifest.reference_source["extraction_mode"] != rec_a.manifest.reference_source["extraction_mode"]


def test_FRF8_condition_and_generic_axis_slices():
    # Queries tagged with a displacement axis value; the generic slicer bins by it and skips an
    # absent axis cleanly.
    pairs = [
        (_result(0, east=1, north=1, matched_e=1, matched_n=1, conf=0.9),
         _gt(0, east=1, north=1, conditions={"condition_displacement_m": "10"})),
        (_result(1, east=1, north=1, matched_e=1, matched_n=1, conf=0.9),
         _gt(1, east=1, north=1, conditions={"condition_displacement_m": "80"})),
    ]
    rec, qs = _bundle(pairs)
    axes = skyline.generic_axis_slices(rec, qs, 50.0, {
        "displacement": {"field": "condition_displacement_m", "bins": [0, 50, 200]},
        "prior_radius": {"field": "condition_prior_radius_m", "bins": [0, 100]},  # absent -> skipped
    })
    assert "displacement" in axes and "prior_radius" not in axes   # absent axis skipped cleanly
    assert len(axes["displacement"]) == 2                          # two bins


def test_distribution_stats_uses_median_not_mean_only():
    stats = skyline.distribution_stats([1.0, 2.0, 3.0, 100.0])
    assert "median" in stats and "p90" in stats and "p95" in stats  # never mean alone


# ===========================================================================
# End-to-end CLI: the additive blocks are emitted, deterministic, and additive
# ===========================================================================

_SKYLINE_FIXTURES = Path(__file__).parent / "fixtures" / "skyline"


def test_cli_emits_006_blocks_and_is_deterministic(tmp_path):
    cfg = _load_config(_SKYLINE_FIXTURES / "eval-mock.json")
    out = run_skyline_evaluation(cfg, tmp_path / "o")
    # spec 006 additive blocks present
    assert "topological_success" in out["retrieval"] and "pr_roc" in out["retrieval"]
    assert "score_separation" in out["aliasing"]
    assert "generic_axis_slices" in out
    assert out["tier_pooling"] == "forbidden"
    # existing 004 metrics unchanged (additive-only within this feature)
    names = {e["name"]: e for e in out["metrics"]}
    assert names["success_rate"]["value"] == pytest.approx(0.6)
    assert names["confident_false_relocalization_rate"]["value"] == pytest.approx(0.2)
    # deterministic
    out2 = run_skyline_evaluation(cfg, tmp_path / "o2")
    assert json.dumps(out, sort_keys=True) == json.dumps(out2, sort_keys=True)


def test_cli_rejects_tier_pooling(tmp_path):
    cfg = _load_config(_SKYLINE_FIXTURES / "eval-mock.json")
    cfg["tier_pooling"] = "allow"
    with pytest.raises(ContractViolationError):
        run_skyline_evaluation(cfg, tmp_path / "o")
