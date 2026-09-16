"""``hsreloc.simret.dualview`` — the dual-direction fusion rules on constructed outcomes.

Every expected value is known before the rule runs: outcomes are hand-written dicts in the shape
``recog.retrieval_outcome`` produces, references sit at declared positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from hsreloc.simret import dualview as dv
from hsreloc.simret import recog

REF_XY = {"A": (0.0, 0.0), "A2": (30.0, 0.0), "B": (600.0, 0.0), "C": (0.0, 900.0)}


def outcome(top1, s1=0.97, frame_margin=0.3, region_margin=0.3, in_top_k=True, region_top1=None):
    return {"top1_id": top1, "top1_score": s1, "frame_margin": frame_margin,
            "region_level_margin": region_margin, "region_in_top_k": in_top_k,
            "region_top1": (top1 in ("A", "A2")) if region_top1 is None else region_top1}


class TestGate:
    def test_frame_and_region_levels_read_their_own_margin(self):
        o = outcome("A", s1=0.95, frame_margin=0.02, region_margin=0.25)
        assert not dv.view_passes(o, dv.Gate(0.9, 0.15, "frame"))
        assert dv.view_passes(o, dv.Gate(0.9, 0.15, "region"))

    def test_non_finite_or_missing_score_never_passes(self):
        assert not dv.view_passes({"top1_score": float("-inf"), "frame_margin": 1.0}, dv.Gate())
        assert not dv.view_passes({"top1_score": None, "frame_margin": 1.0}, dv.Gate())
        assert not dv.view_passes(outcome("A", frame_margin=None), dv.Gate())

    def test_unknown_gate_level_is_refused(self):
        with pytest.raises(dv.DualViewError):
            dv.Gate(0.9, 0.1, "frames")


class TestStrictAgreement:
    def test_both_pass_and_agree_on_adjacent_references_is_accepted_and_true(self):
        r = dv.strict_agreement(outcome("A"), outcome("A2"), REF_XY, (5.0, 0.0), dv.Gate(), dv.Gate(),
                                tau_agree=125.0, tau_region=125.0)
        assert r["agree"] and not r["same_reference"]
        assert r["accepted"] and not r["false_region"] and not r["accepted_false"]
        assert r["candidate_id"] == "A" and r["candidate_distance_m"] == pytest.approx(5.0)
        assert r["dual_region_top1"] and r["dual_region_in_top_k"]
        assert r["min_score"] == pytest.approx(0.97)

    def test_disagreeing_views_abstain_even_when_both_are_confident(self):
        r = dv.strict_agreement(outcome("A", s1=0.99, frame_margin=0.5), outcome("B", s1=0.99, frame_margin=0.5),
                                REF_XY, (5.0, 0.0), dv.Gate(), dv.Gate(), 125.0, 125.0)
        assert not r["agree"] and not r["accepted"]
        assert r["candidate_id"] is None and r["false_region"] and not r["accepted_false"]
        assert not r["dual_region_top1"]

    def test_agreement_on_a_wrong_region_is_an_accepted_false(self):
        r = dv.strict_agreement(outcome("B", region_top1=False), outcome("B", region_top1=False), REF_XY,
                                (5.0, 0.0), dv.Gate(), dv.Gate(), 125.0, 125.0)
        assert r["accepted"] and r["false_region"] and r["accepted_false"]

    def test_one_failing_gate_blocks_acceptance_but_not_the_ungated_top1(self):
        r = dv.strict_agreement(outcome("A", frame_margin=0.05), outcome("A2"), REF_XY, (5.0, 0.0),
                                dv.Gate(), dv.Gate(), 125.0, 125.0)
        assert not r["pass_north"] and r["pass_west"]
        assert not r["accepted"] and r["dual_region_top1"]

    def test_tau_agree_decides_what_counts_as_the_same_region(self):
        r = dv.strict_agreement(outcome("A"), outcome("A2"), REF_XY, (0.0, 0.0), dv.Gate(), dv.Gate(),
                                tau_agree=20.0, tau_region=125.0)
        assert not r["agree"] and r["top1_separation_m"] == pytest.approx(30.0)

    def test_a_missing_top1_never_agrees(self):
        r = dv.strict_agreement({"top1_id": None, "top1_score": float("-inf"), "frame_margin": None,
                                 "region_level_margin": None}, outcome("A"), REF_XY, (0.0, 0.0),
                                dv.Gate(), dv.Gate(), 125.0, 125.0)
        assert not r["agree"] and not r["accepted"] and r["min_score"] is None


class TestFusedScores:
    def test_weakest_view_is_the_elementwise_minimum(self):
        f = dv.fuse_scores([0.9, 0.2, 0.7], [0.5, 0.8, 0.7], "weakest_view")
        assert np.allclose(f, [0.5, 0.2, 0.7])

    def test_mean_score_is_the_mean_and_minus_inf_poisons_a_reference(self):
        f = dv.fuse_scores([0.9, -np.inf, 0.7], [0.5, 0.8, 0.7], "mean_score")
        assert f[0] == pytest.approx(0.7) and f[2] == pytest.approx(0.7)
        assert np.isneginf(f[1])
        g = dv.fuse_scores([0.9, -np.inf], [0.5, 0.8], "weakest_view")
        assert np.isneginf(g[1])

    def test_shape_mismatch_and_unknown_rule_are_refused(self):
        with pytest.raises(dv.DualViewError):
            dv.fuse_scores([0.1, 0.2], [0.1], "weakest_view")
        with pytest.raises(dv.DualViewError):
            dv.fuse_scores([0.1], [0.1], "product")

    def test_fused_vector_feeds_retrieval_outcome_unchanged(self):
        """The fused ranking is scored by the same bookkeeping as a single view — a false region
        under weakest_view is exactly a false region under retrieval_outcome."""
        ids = ["A", "A2", "B"]
        xy = np.array([[0.0, 0.0], [30.0, 0.0], [600.0, 0.0]])
        s_n = np.array([0.95, 0.90, 0.99])          # North is fooled by B
        s_w = np.array([0.96, 0.91, 0.40])          # West is not
        fused = dv.fuse_scores(s_n, s_w, "weakest_view")
        out = recog.retrieval_outcome((2.0, 0.0), ids, xy, fused, np.zeros(3), tau_region=125.0, k=2,
                                      acceptance=recog.Acceptance(0.9, 0.15))
        assert out["top1_id"] == "A" and out["region_top1"] and not out["false_region"]
        single = recog.retrieval_outcome((2.0, 0.0), ids, xy, s_n, np.zeros(3), tau_region=125.0, k=2,
                                         acceptance=recog.Acceptance(0.9, 0.15))
        assert single["top1_id"] == "B" and single["false_region"]


class TestTemporal:
    def test_window_needs_k_consecutive_passing_and_agreeing_queries(self):
        seq = [outcome("A"), outcome("A2"), outcome("A"), outcome("B"), outcome("B")]
        q = [(0, 0)] * 5
        rows = dv.temporal_confirmation(seq, q, 2, dv.Gate(), REF_XY, 125.0, 125.0)
        assert [r["accepted"] for r in rows] == [False, True, True, False, True]
        assert [r["window_complete"] for r in rows] == [False, True, True, True, True]
        assert rows[4]["accepted_false"]                       # B agreed twice, but B is 600 m away
        assert not rows[3]["accepted"] and rows[3]["window_agrees"] is False

    def test_k_equals_one_is_the_single_view_gate(self):
        seq = [outcome("A"), outcome("A", frame_margin=0.01)]
        rows = dv.temporal_confirmation(seq, [(0, 0), (0, 0)], 1, dv.Gate(), REF_XY, 125.0, 125.0)
        assert [r["accepted"] for r in rows] == [True, False]

    def test_a_failing_query_inside_the_window_blocks_it(self):
        seq = [outcome("A"), outcome("A", s1=0.5), outcome("A")]
        rows = dv.temporal_confirmation(seq, [(0, 0)] * 3, 3, dv.Gate(), REF_XY, 125.0, 125.0)
        assert not rows[2]["accepted"] and rows[2]["n_pass_in_window"] == 2

    def test_segments_are_independent_by_construction(self):
        """Two segments are two calls: the second segment's first query has no window, so a
        teleport can never be bridged by the caller's grouping."""
        a = dv.temporal_confirmation([outcome("A")], [(0, 0)], 2, dv.Gate(), REF_XY, 125.0, 125.0)
        b = dv.temporal_confirmation([outcome("A")], [(0, 0)], 2, dv.Gate(), REF_XY, 125.0, 125.0)
        assert not a[0]["window_complete"] and not b[0]["window_complete"]

    def test_k_below_one_is_refused(self):
        with pytest.raises(dv.DualViewError):
            dv.temporal_confirmation([], [], 0, dv.Gate(), REF_XY, 125.0, 125.0)


class TestSweepAndSelection:
    @staticmethod
    def _rows():
        # scores/margins chosen so the grid separates cleanly: high thresholds lose coverage,
        # low thresholds admit the one false row
        return [
            {"s": 0.99, "m": 0.30, "false": False},
            {"s": 0.95, "m": 0.20, "false": False},
            {"s": 0.92, "m": 0.10, "false": False},
            {"s": 0.98, "m": 0.08, "false": True},
        ]

    @staticmethod
    def _eval(r, g):
        acc = r["s"] >= g.theta_s and r["m"] >= g.theta_m
        return acc, r["false"]

    def test_sweep_counts_accepted_and_false_per_point(self):
        sw = dv.sweep(self._rows(), [0.9, 0.95], [0.05, 0.15], self._eval)
        by = {(r["theta_s"], r["theta_m"]): r for r in sw}
        assert by[(0.9, 0.05)]["n_accepted"] == 4 and by[(0.9, 0.05)]["n_accepted_false"] == 1
        assert by[(0.9, 0.15)]["n_accepted"] == 2 and by[(0.9, 0.15)]["n_accepted_false"] == 0
        assert by[(0.95, 0.15)]["n_accepted"] == 2
        assert by[(0.9, 0.05)]["p_false_given_accepted"] == pytest.approx(0.25)

    def test_selection_prefers_zero_false_then_coverage_then_the_conservative_point(self):
        sw = dv.sweep(self._rows(), [0.9, 0.95], [0.05, 0.15], self._eval)
        best = dv.select_gate(sw)
        assert best["false_free_point_exists"]
        assert best["n_accepted_false"] == 0
        # (0.9, 0.15) and (0.95, 0.15) both accept 2 with 0 false; the tie goes to the larger theta_s
        assert (best["theta_s"], best["theta_m"]) == (0.95, 0.15)

    def test_hard_negative_sweeps_veto_a_point(self):
        sw = dv.sweep(self._rows(), [0.9], [0.05, 0.15], self._eval)
        hard = [{"theta_s": 0.9, "theta_m": 0.15, "n_accepted_false": 3}]
        best = dv.select_gate(sw, hard)
        assert (best["theta_s"], best["theta_m"]) == (0.9, 0.05)
        assert best["total_accepted_false_dev"] == 1 and not best["false_free_point_exists"]
