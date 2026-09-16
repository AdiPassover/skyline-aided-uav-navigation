"""Known-answer tests for the retrieval-summary layer (Experiments 3/5 joins and slices)."""

import pytest

from hsreloc.simret import extreport


def _row(rank=1, attainable=True, accepted=False, top1=None, dist=10.0, correct="r0", cand="r0"):
    top1_correct = (cand == correct) if top1 is None else top1
    return {"attainable": attainable, "rank_of_correct": rank, "accepted": accepted,
            "top1_correct": top1_correct, "top1_reference_id": cand,
            "nearest_reference_id": correct, "nearest_reference_distance_m": dist,
            "outcome": "SUCCESS" if accepted else "REJECTED",
            "margin_correct_minus_best_incorrect": 0.1}


class TestRetrievalSummary:
    def test_recalls_over_attainable_only(self):
        rows = [_row(rank=1), _row(rank=2, cand="r1"), _row(rank=None, cand="r1"),
                _row(rank=None, attainable=False)]
        s = extreport._retrieval_summary(rows, n_references=10)
        assert s["n_attainable"] == 3 and s["n_unattainable"] == 1
        assert s["recall_at_1"] == pytest.approx(1 / 3)
        assert s["recall_at_3"] == pytest.approx(2 / 3)
        assert s["chance_recall_at_1"] == pytest.approx(0.1)

    def test_confident_false_strict_counts_unattainable_accepts(self):
        rows = [_row(rank=1, accepted=True),                        # accepted correct
                _row(rank=2, accepted=True, cand="r1"),             # accepted wrong
                _row(rank=None, attainable=False, accepted=True)]   # accepted unattainable
        s = extreport._retrieval_summary(rows, 10)
        assert s["n_confident_false_strict"] == 2
        assert s["accepted_precision"] == pytest.approx(0.5)

    def test_aliases_name_the_winning_wrong_reference(self):
        rows = [_row(rank=2, cand="r7"), _row(rank=2, cand="r7"), _row(rank=1)]
        s = extreport._retrieval_summary(rows, 10)
        assert s["top_aliases"] == {"r0 -> r7": 2}


class TestSliceAndRadius:
    def test_slice_recall(self):
        rows = [dict(_row(rank=1), phase="outbound"), dict(_row(rank=None, cand="r1"),
                                                           phase="outbound"),
                dict(_row(rank=1), phase="inbound")]
        s = extreport._slice_recall(rows, "phase")
        assert s["outbound"]["recall_at_1"] == pytest.approx(0.5)
        assert s["inbound"]["recall_at_1"] == pytest.approx(1.0)

    def test_radius_needs_support(self):
        rows = [_row(rank=1, dist=5.0) for _ in range(10)]
        r = extreport._radius(rows, "x", 0.8, min_n=30)
        assert r["radius_m"] is None and "no bin" in r["note"]

    def test_radius_cumulative_rule(self):
        rows = ([_row(rank=1, dist=5.0) for _ in range(30)]            # perfect inside 7.5
                + [_row(rank=None, cand="r1", dist=50.0) for _ in range(30)])  # all wrong at 50
        r80 = extreport._radius(rows, "x", 0.8, min_n=30)
        assert r80["radius_m"] == 7.5 and r80["recall_at_1"] == 1.0

    def test_wilson_low_monotone_in_n(self):
        assert extreport._wilson_low(0.9, 10) < extreport._wilson_low(0.9, 100) < 0.9
