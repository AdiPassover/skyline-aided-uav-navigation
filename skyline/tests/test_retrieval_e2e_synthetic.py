"""Stage 1 end to end: constructed answers, and scoring through the real spec-006 evaluator.

Tasks T027 and T032. Spec User Story 1; SC-001..SC-004, SC-006.

Every case here has an answer that was **constructed, not measured** -- which query belongs to which
place is a fact of how the fixture was built. That is what makes this a proof of the chain rather
than an experiment: if the unambiguous query does not return its own place, the plumbing is wrong,
and no argument about representations is needed to know it.

The evaluator used is ``naveval.evaluate_skyline``, unmodified and imported directly. Nothing here
re-implements a metric.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from hsreloc.retrieval.run import execute
from hsreloc.retrieval.runconfig import parse_config
from tests import synth_corpus

from naveval.evaluate_skyline import _load_config, run_skyline_evaluation
from naveval.skyline_record import load_skyline_record

EXPECTED_OUTCOME = {
    "unambiguous": "SUCCESS",
    "aliased": "AMBIGUOUS",
    "out_of_coverage": "REJECTED",
    "degenerate": "EXTRACTION_FAILURE",
}


def _run(sky_run_config_dict, sky_corpus, tmp_path, baseline="ncc", run_id=None):
    raw = copy.deepcopy(sky_run_config_dict)
    raw["match"]["baseline"] = baseline
    raw["run_id"] = run_id or f"stage1-{baseline}"
    raw["output"] = {"root": str(tmp_path / "skyline_runs")}
    config = parse_config(raw, base_dir=tmp_path)
    summary = execute(config)
    return summary, load_skyline_record(summary["record_dir"])


def _by_observation(record, query_set):
    return {obs: record.results[synth_corpus.query_index_of(query_set, obs)]
            for obs in ("q_unambiguous_2", "q_unambiguous_5", "q_aliased",
                        "q_out_of_coverage", "q_degenerate")}


# --- the five constructed cases ------------------------------------------------------------------

@pytest.mark.parametrize("baseline", ["ncc", "l1", "l2"])
def test_every_constructed_case_resolves_to_its_known_answer(sky_run_config_dict, sky_corpus,
                                                             tmp_path, baseline):
    summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, baseline)
    answers = sky_corpus["answers"]
    for observation_id, answer in answers.items():
        idx = synth_corpus.query_index_of(sky_corpus["query_set"], observation_id)
        result = record.results[idx]
        assert result.outcome == EXPECTED_OUTCOME[answer["case"]], (
            f"{observation_id} ({answer['case']}) under {baseline}: got {result.outcome}")


def test_unambiguous_success_lands_on_the_constructed_place(sky_run_config_dict, sky_corpus,
                                                            tmp_path):
    _summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc")
    for observation_id, answer in sky_corpus["answers"].items():
        if answer["case"] != "unambiguous":
            continue
        idx = synth_corpus.query_index_of(sky_corpus["query_set"], observation_id)
        result = record.results[idx]
        assert result.est_east_m == pytest.approx(answer["east_m"], abs=1e-6)
        assert result.est_north_m == pytest.approx(answer["north_m"], abs=1e-6)
        assert result.matched_east_m == result.est_east_m


def test_success_converts_to_the_constructed_geodetic_answer(sky_run_config_dict, sky_corpus,
                                                             tmp_path):
    """The absolute-fix half of the chain, checked against a geodetic answer that was constructed."""
    from hsreloc.retrieval.fix import enu_to_geodetic

    _summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc")
    origin = sky_corpus["origin"]
    observation_id = "q_unambiguous_4"
    answer = sky_corpus["answers"][observation_id]
    idx = synth_corpus.query_index_of(sky_corpus["query_set"], observation_id)
    result = record.results[idx]

    got = enu_to_geodetic(result.est_east_m, result.est_north_m, 0.0, origin)
    want = enu_to_geodetic(answer["east_m"], answer["north_m"], 0.0, origin)
    assert got[0] == pytest.approx(want[0], abs=1e-12)
    assert got[1] == pytest.approx(want[1], abs=1e-12)


@pytest.mark.parametrize("case", ["aliased", "out_of_coverage", "degenerate"])
def test_every_refusal_carries_no_consumable_position(sky_run_config_dict, sky_corpus, tmp_path,
                                                      case):
    _summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc")
    observation_id = next(k for k, v in sky_corpus["answers"].items() if v["case"] == case)
    idx = synth_corpus.query_index_of(sky_corpus["query_set"], observation_id)
    result = record.results[idx]
    assert result.est_east_m is None and result.est_north_m is None


def test_no_query_is_ever_reported_out_of_coverage(sky_run_config_dict, sky_corpus, tmp_path):
    """Emitting it would require reading in_coverage, which is ground truth (research R7)."""
    _summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc")
    assert all(r.outcome != "OUT_OF_COVERAGE" for r in record.results)


def test_heading_is_never_emitted_end_to_end(sky_run_config_dict, sky_corpus, tmp_path):
    _summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc")
    assert all(r.est_heading_deg is None and r.heading_source is None for r in record.results)


def test_shortlist_is_emitted_for_recall_at_k(sky_run_config_dict, sky_corpus, tmp_path):
    _summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc")
    non_degenerate = [r for r in record.results if r.outcome != "EXTRACTION_FAILURE"]
    assert non_degenerate and all(len(r.candidates) == 5 for r in non_degenerate)


# --- the mock: matcher-agnosticism, proved from the producer side --------------------------------

def test_mock_ranking_produces_a_record_the_evaluator_reads_identically(sky_run_config_dict,
                                                                       sky_corpus, tmp_path):
    """SC-003: a ranking from no skyline matching at all must evaluate the same way in shape."""
    _s1, real_record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc", run_id="cmp-ncc")
    _s2, mock_record = _run(sky_run_config_dict, sky_corpus, tmp_path, "mock", run_id="cmp-mock")

    assert len(real_record.results) == len(mock_record.results)
    real_fields = {f for f in vars(real_record.results[0])}
    mock_fields = {f for f in vars(mock_record.results[0])}
    assert real_fields == mock_fields
    assert mock_record.manifest.reference_source == real_record.manifest.reference_source
    assert mock_record.manifest.relocalizer_config["baseline"] == "mock"


def test_mock_ranking_is_independent_of_curve_content(sky_run_config_dict, sky_corpus, tmp_path):
    _s, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "mock")
    idx = synth_corpus.query_index_of(sky_corpus["query_set"], "q_degenerate")
    # Even the degenerate query is refused on its *profile*, before any scoring -- so the mock's
    # indifference to content shows up as the other queries scoring without regard to their place.
    assert record.results[idx].outcome == "EXTRACTION_FAILURE"


# --- T032: scoring through the unmodified spec-006 evaluator --------------------------------------

def _evaluate(record_dir: Path, query_set: Path, tmp_path: Path, name: str) -> dict:
    config_path = tmp_path / f"eval-{name}.json"
    config_path.write_text(json.dumps({
        "schema_version": "1.0.0",
        "evaluation_id": f"sky-stage1-{name}",
        "run_record": str(record_dir),
        "query_set": str(query_set),
        "operational_tolerance_m": 500,
        "near_tolerance_m": 1500,
        "recall_k": 5,
        "confidence_reject_threshold": 0.5,
        "tier_pooling": "forbidden",
    }), encoding="utf-8")
    config = _load_config(config_path)
    return run_skyline_evaluation(config, tmp_path / f"out-{name}")


def test_the_unmodified_evaluator_scores_the_record(sky_run_config_dict, sky_corpus, tmp_path):
    summary, _record = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc")
    metrics = _evaluate(summary["record_dir"], sky_corpus["query_set"], tmp_path, "ncc")
    assert metrics["evidence_tier"] == "T1"
    assert (tmp_path / "out-ncc" / "metrics.json").exists()
    assert metrics["metrics"]


#: Metrics whose presence depends on how many successes a run produced, not on its schema.
#: A distribution tail needs enough samples to have a p90; with one success there is none to report.
_SUPPORT_DEPENDENT_PREFIXES = ("position_error_", "heading_error_", "confidence_calibration_")


def test_mock_and_real_records_evaluate_through_the_same_schema(sky_run_config_dict, sky_corpus,
                                                                tmp_path):
    """SC-003, the producer-side proof that the evaluator is indifferent to *how* a ranking was made.

    The two runs do **not** produce an identical metric list, and should not: a random ranking gets
    far fewer successes, so its distribution tails have no support and the evaluator correctly omits
    them. What must hold -- and what actually demonstrates matcher-agnosticism -- is that both records
    are read by the same code path without a contract violation, carry the same tier and source block,
    and differ only in metrics whose *support* differs. A schema difference would show up as a name
    the mock produces that the real run does not, and there must be none.
    """
    s_real, _ = _run(sky_run_config_dict, sky_corpus, tmp_path, "ncc", run_id="ev-ncc")
    s_mock, _ = _run(sky_run_config_dict, sky_corpus, tmp_path, "mock", run_id="ev-mock")
    real = _evaluate(s_real["record_dir"], sky_corpus["query_set"], tmp_path, "real")
    mock = _evaluate(s_mock["record_dir"], sky_corpus["query_set"], tmp_path, "mock")

    real_names = {m["name"] for m in real["metrics"]}
    mock_names = {m["name"] for m in mock["metrics"]}

    assert not (mock_names - real_names), (
        f"the mock produced metric names the real matcher did not: {sorted(mock_names - real_names)} "
        f"-- that would be a schema difference, not a support difference")
    for name in real_names - mock_names:
        assert name.startswith(_SUPPORT_DEPENDENT_PREFIXES), (
            f"{name!r} differs between a real and a mock ranking for a reason other than support")
    assert len(real_names & mock_names) >= 10
    assert real["evidence_tier"] == mock["evidence_tier"] == "T1"


def test_evaluated_tier_is_t1_and_never_claims_real_evidence(sky_run_config_dict, sky_corpus,
                                                             tmp_path):
    summary, record = _run(sky_run_config_dict, sky_corpus, tmp_path, "l2")
    metrics = _evaluate(summary["record_dir"], sky_corpus["query_set"], tmp_path, "l2")
    assert metrics["evidence_tier"] == "T1"
    assert record.manifest.reference_source["evidence_tier"] == "T1"
    assert "not real data" in record.manifest.reference_source["evidence_caveat"].lower()
