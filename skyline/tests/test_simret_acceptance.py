"""Known-answer tests for the one authorized acceptance variant (research R8)."""

import pytest

from hsreloc.simret import acceptance


def _row(best=0.9, second=0.6, attainable=True, top1=True, accepted=False):
    return {"best_score": best, "second_best_score": second, "score_margin": best - second,
            "attainable": attainable, "top1_correct": top1, "accepted": accepted}


class TestDecide:
    def test_thresholds_are_conjunctive(self):
        assert acceptance.decide(_row(0.9, 0.6), 0.8, 0.2)
        assert not acceptance.decide(_row(0.7, 0.4), 0.8, 0.2)       # score too low
        assert not acceptance.decide(_row(0.9, 0.85), 0.8, 0.2)      # margin too small

    def test_missing_scores_reject(self):
        assert not acceptance.decide({"best_score": None, "score_margin": None}, 0.5, 0.1)


class TestCalibrate:
    GRID = {"theta_s": [0.5, 0.8], "theta_m": [0.1, 0.3]}

    def test_zero_confident_false_branch_prefers_coverage_then_safety(self):
        rows = [
            _row(0.95, 0.5, top1=True),                # margin .45: accepted by every cell
            _row(0.85, 0.7, top1=True),                # margin .15: only theta_m=0.1 cells
            _row(0.60, 0.55, top1=False),              # wrong, margin .05: accepted by no cell
        ]
        res = acceptance.calibrate(rows, self.GRID)
        assert res["rule_path"].startswith("zero-confident-false")
        # theta_m=0.1 cells accept 2 correct; ties between theta_s 0.5/0.8 -> larger theta_s
        assert res["chosen"]["theta_m"] == 0.1
        assert res["chosen"]["theta_s"] == 0.8
        assert res["chosen"]["n_accepted_correct"] == 2
        assert res["chosen"]["n_confident_false_strict"] == 0

    def test_accepted_unattainable_is_strict_confident_false(self):
        rows = [_row(0.99, 0.1, attainable=False, top1=False)]
        cell = acceptance.evaluate_cell(rows, 0.5, 0.1)
        assert cell["n_confident_false_strict"] == 1

    def test_fallback_branch_when_no_zero_cell(self):
        rows = [_row(0.99, 0.1, top1=False)]           # a confident wrong every cell accepts...
        res = acceptance.calibrate(rows, {"theta_s": [0.5], "theta_m": [0.1]})
        assert res["rule_path"].startswith("min-confident-false")

    def test_empty_rows_refused(self):
        with pytest.raises(acceptance.AcceptanceError):
            acceptance.calibrate([], self.GRID)


class TestSideBySide:
    def test_both_rules_counted_on_the_same_rows(self):
        rows = [
            _row(0.95, 0.5, top1=True, accepted=True),    # both accept, correct
            _row(0.60, 0.55, top1=False, accepted=True),  # frozen accepts (confident-false);
                                                          # variant rejects (margin .05)
            _row(0.90, 0.5, top1=True, accepted=False),   # frozen rejects; variant accepts
        ]
        res = acceptance.side_by_side(rows, {"theta_s": 0.8, "theta_m": 0.2})
        assert res["frozen_rule"]["n_accepted"] == 2
        assert res["frozen_rule"]["n_confident_false_strict"] == 1
        assert res["variant"]["n_accepted"] == 2
        assert res["variant"]["n_confident_false_strict"] == 0
        assert res["variant"]["accepted_precision"] == 1.0
