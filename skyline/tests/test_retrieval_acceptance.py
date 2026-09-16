"""Acceptance and confidence (T012) -- and the standing proof that neither can see correctness.

Spec FR-016, FR-017. Research **R7**.

The most important test in this file is ``test_acceptance_cannot_see_correctness``: it inspects the
signatures of every public function here and asserts that none of them accepts ground truth, a
position error, or an "is this right" flag. That is a structural guarantee rather than a promise --
if someone later threads correctness into this module to make a number look better, this test fails.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hsreloc.retrieval import acceptance as acc

CFG = acc.AcceptanceConfig()


def _population(best, others):
    return np.array([best] + list(others), dtype=np.float64)


# --- the confidence map ------------------------------------------------------------------------

def test_confidence_is_monotonic_in_separation():
    values = [acc.confidence_from_z(z, CFG.confidence_k) for z in (0.0, 1.0, 3.0, 5.0, 20.0, 200.0)]
    assert values == sorted(values)


def test_confidence_half_point_is_exactly_five_sigma():
    """conf >= 0.5 iff z >= 5 sigma -- so the repo's existing 0.5 threshold keeps its meaning."""
    assert acc.confidence_from_z(5.0, 5.0) == pytest.approx(0.5)
    assert acc.confidence_from_z(4.999, 5.0) < 0.5
    assert acc.confidence_from_z(5.001, 5.0) > 0.5


def test_confidence_is_bounded_and_never_saturates():
    """The EXP-009 degenerate case: the pre-fix map saturated at 1.0 and collapsed every margin."""
    assert acc.confidence_from_z(1e9, 5.0) < 1.0
    assert acc.confidence_from_z(-100.0, 5.0) == 0.0
    assert 0.0 <= acc.confidence_from_z(7.0, 5.0) < 1.0


# --- robust spread -----------------------------------------------------------------------------

def test_robust_sigma_uses_mad_when_available():
    null = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert acc.robust_sigma(null, float(np.median(null))) > 0.0


def test_robust_sigma_falls_back_when_mad_is_zero():
    """A majority-tied population has MAD 0; the one-sided fallback still finds real spread."""
    null = np.array([1.0, 1.0, 1.0, 1.0, 9.0])
    assert acc.robust_sigma(null, 1.0) > 0.0


def test_robust_sigma_is_zero_only_when_there_is_genuinely_no_spread():
    assert acc.robust_sigma(np.array([2.0, 2.0, 2.0]), 2.0) == 0.0


# --- the outcome table -------------------------------------------------------------------------

def test_degenerate_query_is_an_extraction_failure_with_no_position():
    d = acc.decide(_population(0.9, [0.1] * 5), CFG, query_profile_degenerate=True)
    assert d.outcome == acc.EXTRACTION_FAILURE
    assert d.confidence is None and d.best_score is None


def test_no_spread_in_the_population_is_a_rejection_not_an_extraction_failure():
    """A perfectly good query against mutually indistinguishable references is a REFUSAL.

    Calling it EXTRACTION_FAILURE would blame the query for a reference-set property, and the
    evaluator counts those outcomes separately -- so the misattribution would show up in the results.
    """
    d = acc.decide(_population(1.0, [1.0] * 5), CFG)
    assert d.outcome == acc.REJECTED
    assert d.confidence == 0.0


def test_clear_separation_is_a_success():
    d = acc.decide(_population(1.0, [0.10, 0.11, 0.09, 0.12, 0.08]), CFG)
    assert d.outcome == acc.SUCCESS
    assert d.confidence >= CFG.reject_threshold
    assert d.n_viable_candidates == 1


def test_near_tie_at_the_top_is_ambiguous_not_a_confident_fix():
    d = acc.decide(_population(1.000, [0.999, 0.10, 0.11, 0.09, 0.12]), CFG)
    assert d.outcome == acc.AMBIGUOUS
    assert d.n_viable_candidates >= 2
    assert d.confidence is not None


def test_weak_separation_is_rejected():
    d = acc.decide(_population(0.30, [0.28, 0.26, 0.24, 0.22, 0.20]), CFG)
    assert d.outcome == acc.REJECTED
    assert d.confidence < CFG.reject_threshold


def test_a_single_reference_cannot_demonstrate_separation():
    d = acc.decide(_population(0.99, []), CFG)
    assert d.outcome == acc.REJECTED


def test_empty_population_is_rejected():
    d = acc.decide(np.array([]), CFG)
    assert d.outcome == acc.REJECTED


def test_out_of_coverage_is_never_produced():
    """Emitting it would require reading in_coverage, which is ground truth (research R7)."""
    populations = [
        _population(1.0, [0.1] * 5),
        _population(0.2, [0.19, 0.18, 0.17]),
        _population(1.0, [1.0] * 4),
    ]
    outcomes = {acc.decide(p, CFG).outcome for p in populations}
    outcomes.add(acc.decide(populations[0], CFG, query_profile_degenerate=True).outcome)
    assert "OUT_OF_COVERAGE" not in outcomes


# --- the structural guarantee -------------------------------------------------------------------

def test_acceptance_cannot_see_correctness():
    """No public entry point here may take ground truth or a correctness signal (FR-017)."""
    forbidden = {"gt", "ground_truth", "truth", "correct", "is_correct", "in_coverage",
                 "position_error", "error_m", "label", "answer", "expected", "target"}
    for name, fn in vars(acc).items():
        if name.startswith("_") or not callable(fn) or not hasattr(fn, "__module__"):
            continue
        if getattr(fn, "__module__", None) != acc.__name__:
            continue
        params = set(inspect.signature(fn).parameters)
        leaked = params & forbidden
        assert not leaked, f"{name}() accepts correctness-bearing parameters: {sorted(leaked)}"


def test_frozen_constants_are_what_the_plan_pre_registered():
    """These are approved and frozen; a change makes a new run, never a retro-edit."""
    assert acc.DEFAULT_CONFIDENCE_K == 5.0
    assert acc.DEFAULT_REJECT_THRESHOLD == 0.5
    assert acc.DEFAULT_VIABLE_SIGMA == 3.0
    assert acc.DEFAULT_AMBIGUITY_MARGIN_SIGMA == 1.0
    assert acc.DEFAULT_DEGENERATE_PROFILE_STD == 1e-6
