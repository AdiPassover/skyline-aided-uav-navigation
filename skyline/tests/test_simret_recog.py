"""EXP-SKY-010 mechanism tests (T1): the batched frozen-C1 ranking is bit-equal to the frozen
matcher, the region bookkeeping is right, and the pre-registered bin/radius/spacing rules do what
the record says."""

import math

import numpy as np
import pytest

from hsreloc.matchers.variants import BoundedLagNccMatcher
from hsreloc.simret import recog


def _walk(seed, n=256):
    rng = np.random.default_rng(seed)
    p = np.cumsum(rng.standard_normal(n)) * 0.01
    return p - p.mean()


class TestBankEqualsFrozenC1:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("shift", [-40, -32, -7, 0, 3, 19, 32, 45])
    @pytest.mark.parametrize("chunk", [1, 2, 48])
    def test_winner_exact_and_score_to_1e12(self, seed, shift, chunk):
        base = _walk(seed, 400)
        q = base[100:356].copy()
        refs = [np.roll(base, shift)[100:356].copy(), _walk(seed + 10), _walk(seed + 20)]
        bank = recog.ReferenceBank(["r0", "r1", "r2"], refs, max_lag=32, min_overlap_frac=0.6,
                                   chunk_references=chunk)
        scores, lags = bank.score(q)
        frozen = BoundedLagNccMatcher(max_lag_samples=32, min_overlap_frac=0.6)
        for k, r in enumerate(refs):
            res = frozen.match(q, r)
            assert lags[k] == res.shift, (k, lags[k], res.shift)              # winner: exact
            assert scores[k] == pytest.approx(res.score, abs=1e-12), (k, scores[k], res.score)

    def test_tie_break_least_abs_lag_then_negative(self):
        lags = np.arange(-3, 4, dtype=float)
        scores = np.array([[0.5, 0.9, 0.9, 0.1, 0.9, 0.9, 0.5],
                           [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                           [-np.inf] * 7])
        best, lag = recog.winner_per_reference(scores, lags)
        assert best[0] == 0.9 and lag[0] == -1.0          # ties at ±1, ±2 -> least |lag|, negative
        assert best[1] == 0.2 and lag[1] == 0.0
        assert best[2] == -np.inf and math.isnan(lag[2])

    def test_rejects_wrong_length(self):
        bank = recog.ReferenceBank(["a"], [_walk(0)])
        with pytest.raises(recog.RecogError):
            bank.score(_walk(1, 200))


class TestRetrievalOutcome:
    IDS = ["near", "same_region", "far_a", "far_b"]
    XY = np.array([[10.0, 0.0], [60.0, 0.0], [400.0, 0.0], [800.0, 0.0]])

    def test_ranks_margins_and_acceptance(self):
        scores = np.array([0.95, 0.70, 0.60, 0.20])
        lags = np.array([1.0, 2.0, 3.0, 4.0])
        o = recog.retrieval_outcome((0.0, 0.0), self.IDS, self.XY, scores, lags, tau_region=125.0, k=2,
                                    extra_taus=(50.0,))
        assert o["nearest_id"] == "near" and o["d_near_m"] == 10.0
        assert o["exact_rank"] == 1 and o["region_rank"] == 1 and o["region_top1"]
        assert o["top1_id"] == "near" and o["top1_distance_m"] == 10.0
        assert o["frame_margin"] == pytest.approx(0.25) and o["accepted_frame"]
        assert o["best_in_region_score"] == 0.95 and o["best_out_region_score"] == 0.60
        assert o["region_margin"] == pytest.approx(0.35) and not o["ambiguous"]
        assert o["region_level_margin"] == pytest.approx(0.35) and o["accepted_region"]
        assert not o["false_region"] and o["attainable"]
        assert o["top2_id"] == "same_region" and "top3_id" not in o          # k = 2
        # sensitivity at 50 m: same_region is now outside; nothing changes for the top-1
        assert o["attainable_tau50"] and o["region_rank_tau50"] == 1
        assert o["best_out_region_score_tau50"] == 0.70

    def test_false_region_and_ambiguity(self):
        scores = np.array([0.80, 0.30, 0.85, 0.20])
        lags = np.zeros(4)
        o = recog.retrieval_outcome((0.0, 0.0), self.IDS, self.XY, scores, lags, tau_region=125.0)
        assert o["top1_id"] == "far_a" and o["false_region"] and not o["region_top1"]
        assert o["region_rank"] == 2 and o["exact_rank"] == 2 and o["region_in_top_k"]
        assert o["region_margin"] == pytest.approx(-0.05) and o["ambiguous"]
        assert not o["accepted_frame"]                     # margin 0.05 < 0.15

    def test_frame_margin_refuses_adjacent_twin_but_region_level_accepts(self):
        scores = np.array([0.96, 0.95, 0.40, 0.20])
        lags = np.zeros(4)
        o = recog.retrieval_outcome((0.0, 0.0), self.IDS, self.XY, scores, lags, tau_region=125.0)
        assert not o["accepted_frame"] and o["frame_margin"] == pytest.approx(0.01)
        assert o["accepted_region"] and o["region_level_margin"] == pytest.approx(0.56)

    def test_holdout_mask_changes_nearest_and_attainability(self):
        scores = np.array([0.95, 0.70, 0.60, 0.20])
        lags = np.zeros(4)
        keep = np.array([False, True, True, True])
        o = recog.retrieval_outcome((0.0, 0.0), self.IDS, self.XY, scores, lags, keep=keep, tau_region=125.0)
        assert o["nearest_id"] == "same_region" and o["d_near_m"] == 60.0 and o["n_references"] == 3
        assert o["region_top1"] and o["exact_top1"]
        keep2 = np.array([False, False, True, True])
        o2 = recog.retrieval_outcome((0.0, 0.0), self.IDS, self.XY, scores, lags, keep=keep2, tau_region=125.0)
        assert not o2["attainable"] and o2["false_region"] and o2["region_rank"] is None
        assert o2["best_in_region_score"] is None and o2["region_margin"] is None and not o2["ambiguous"]

    def test_tie_breaks_by_reference_id(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        o = recog.retrieval_outcome((0.0, 0.0), self.IDS, self.XY, scores, np.zeros(4))
        assert [o[f"top{j}_id"] for j in (1, 2, 3, 4)] == sorted(self.IDS)

    def test_no_admissible_alignment(self):
        scores = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        o = recog.retrieval_outcome((0.0, 0.0), self.IDS, self.XY, scores, np.full(4, np.nan))
        assert o["outcome"] == "NO_ADMISSIBLE_ALIGNMENT" and not o["accepted_frame"] and not o["accepted_region"]


def _row(d, site, region_top1=True, in_k=True, accepted=False, false_region=None, score=0.95):
    fr = (not region_top1) if false_region is None else false_region
    return {"d_near_m": d, "site": site, "attainable": d <= 125.0, "exact_top1": region_top1,
            "exact_in_top_k": in_k, "region_top1": region_top1, "region_in_top_k": in_k,
            "top1_score": score, "best_in_region_score": score, "region_margin": 0.3,
            "false_region": fr, "accepted_frame": accepted, "accepted_region": accepted,
            "ambiguous": False}


EDGES = [0.0, 5.0, 10.0, 20.0, 30.0]
SUPPORT = {"min_queries": 4, "min_sites": 2}
RULE = {"region_top1_min": 0.9, "accepted_false_region_max": 0.02, "region_recall_k_min": 0.9}


class TestBinsAndRadius:
    def test_bin_assignment(self):
        assert recog.assign_bin(0.0, EDGES) == "0-5"
        assert recog.assign_bin(5.0, EDGES) == "5-10"
        assert recog.assign_bin(29.9, EDGES) == "20-30"
        assert recog.assign_bin(31.0, EDGES) == "30+"

    def test_radius_stops_at_failing_bin(self):
        rows = []
        for d in (1, 2, 3, 4):
            rows += [_row(d, "s1"), _row(d, "s2")]                       # 0-5: 8 rows, 2 sites, all correct
        for d in (6, 7, 8, 9):
            rows += [_row(d, "s1"), _row(d, "s2")]                       # 5-10: all correct
        rows += [_row(12, "s1"), _row(13, "s2"), _row(14, "s1", region_top1=False, in_k=True),
                 _row(15, "s2", region_top1=False, in_k=True)]           # 10-20: top1 0.5, R@k 1.0
        rows += [_row(22, "s1"), _row(23, "s2"), _row(24, "s1"), _row(25, "s2")]
        table = recog.bin_table(rows, EDGES, SUPPORT, k=5)
        assert [e["bin"] for e in table] == ["0-5", "5-10", "10-20", "20-30", "30+"]
        assert table[2]["region_top1"] == 0.5 and table[2]["region_recall_k"] == 1.0
        assert table[2]["n_matcher_failed_with_nearby_reference"] == 2
        r = recog.radius_from_bins(table, RULE)
        assert r["conservative_m"] == 10.0 and r["conservative_stop"]["bin"] == "10-20"
        assert r["recoverable_m"] == 30.0 and r["recoverable_stop"]["bin"] == "30+"

    def test_radius_stops_at_unsupported_bin_without_extrapolating(self):
        rows = []
        for d in (1, 2, 3, 4):
            rows += [_row(d, "s1"), _row(d, "s2")]
        rows += [_row(7, "s1"), _row(8, "s1")]                           # 5-10: 2 rows, 1 site -> unsupported
        for d in (12, 13, 14, 15):
            rows += [_row(d, "s1"), _row(d, "s2")]                       # 10-20 fine but unreachable
        table = recog.bin_table(rows, EDGES, SUPPORT)
        assert not table[1]["supported"]
        r = recog.radius_from_bins(table, RULE)
        assert r["conservative_m"] == 5.0 and "unsupported" in r["conservative_stop"]["reason"]
        assert r["recoverable_m"] == 5.0

    def test_accepted_false_region_blocks_conservative_only(self):
        rows = []
        for d in (1, 2, 3, 4):
            rows += [_row(d, "s1"), _row(d, "s2")]
        rows += [_row(6, "s1"), _row(7, "s2"), _row(8, "s1"), _row(9, "s2")] * 5
        # one accepted false region among 21 rows in 5-10 -> 4.8 % > 2 %, but top-1 still >= 0.9
        rows += [_row(9.5, "s1", region_top1=False, in_k=True, accepted=True)]
        table = recog.bin_table(rows, EDGES, SUPPORT)
        e = table[1]
        assert e["n"] == 21 and e["n_accepted_false_region"] == 1
        assert e["p_false_given_accepted"] == 1.0
        assert e["region_top1"] >= 0.95
        r = recog.radius_from_bins(table, RULE)
        assert r["conservative_m"] == 5.0 and r["recoverable_m"] == 10.0

    def test_first_bin_failing_gives_no_radius(self):
        rows = [_row(d, s, region_top1=False, in_k=False) for d in (1, 2, 3) for s in ("s1", "s2")]
        table = recog.bin_table(rows, EDGES, SUPPORT)
        r = recog.radius_from_bins(table, RULE)
        assert r["conservative_m"] is None and r["recoverable_m"] is None

    def test_site_aggregation(self):
        rows = [_row(1, "s1"), _row(2, "s1"), _row(3, "s2", region_top1=False), _row(4, "s2")]
        e = recog.bin_table(rows, EDGES, {"min_queries": 1, "min_sites": 1})[0]
        assert e["n_sites"] == 2 and e["site_region_top1_min"] == 0.5 and e["site_region_top1_median"] == 0.75


class TestSpacing:
    def test_geometry(self):
        s = recog.spacing_implication(30.0, current_spacing_m=10.0)
        assert s["theoretical_max_spacing_m"] == 60.0 and s["conservative_spacing_m"] == 30.0
        assert s["assessment"].startswith("denser")
        assert recog.spacing_implication(15.0)["assessment"].startswith("reasonable")
        assert recog.spacing_implication(8.0)["assessment"].startswith("sparse")
        assert recog.spacing_implication(4.0)["assessment"].startswith("too sparse")
        assert recog.spacing_implication(None)["conservative_spacing_m"] is None

    def test_wilson(self):
        assert recog.wilson_low(1.0, 30) == pytest.approx(0.8863, abs=1e-3)
        assert recog.wilson_low(None, 5) is None
