"""Deterministic checks for the claim-closure analysis tools (2026-09).

    cd evaluation; python -m pytest tools/claim_closure -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "exp_conf_001"))

import conf_benchmark as cb                                                     # noqa: E402
from p4_analysis import average_precision, rankdata, spearman                   # noqa: E402


def test_fast_ap_equals_reference_with_ties():
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(50, 400))
        score = np.round(rng.normal(size=n), 1)            # coarse rounding forces ties
        label = rng.random(n) < 0.05
        if label.sum() == 0:
            label[0] = True
        ref, _, _ = average_precision(score, label)
        assert abs(cb.fast_ap(score, label) - ref) < 1e-12


def test_avg_rank_and_spearman_equal_reference():
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(20, 300))
        x = np.round(rng.normal(size=n), 1)
        y = x + rng.normal(size=n)
        assert np.allclose(cb.avg_rank(x), rankdata(x))
        r_ref, _ = spearman(x, y)
        assert abs(cb.fast_spearman(x, y) - r_ref) < 1e-12


def test_auroc_known_answers():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([False, False, True, True])
    assert abs(cb.auroc(s, y) - 0.75) < 1e-12
    assert abs(cb.auroc(np.array([1.0, 2.0, 3.0]), np.array([False, False, True])) - 1.0) < 1e-12


def test_naive_policies_differ_from_frozen_only_in_declared_keys():
    repo = HERE.parents[2]
    frozen = json.loads((repo / "evaluation/eval_configs/int/exp-int-002/reloc-c0-primary-retry20.json").read_text(encoding="utf-8"))
    allowed = {"region_rule", "region_gap_references", "fusion_rule", "temporal_confirmation", "margin_threshold",
               "ambiguity_region_radius_m", "dual_agreement_radius_m", "temporal_region_radius_m"}
    for name, fusion in (("reloc-naive-1v-top1.json", "north_only"), ("reloc-naive-2v-top1.json", "weakest_view")):
        p = json.loads((repo / "evaluation/eval_configs/int/claim-closure-2026-09" / name).read_text(encoding="utf-8"))
        diff = {k for k in set(frozen) | set(p) if frozen.get(k, "<absent>") != p.get(k, "<absent>")}
        assert diff <= allowed, diff
        assert p["fusion_rule"] == fusion and p["temporal_confirmation"] == "disabled"
        assert p["margin_threshold"] == 0.0 and p["region_rule"] == "id_gap" and p["region_gap_references"] == 0
        assert p["match_threshold"] == frozen["match_threshold"] == 0.9


def test_zero_count_upper_bound():
    from int_comparison_arms import upper95_zero
    assert abs(upper95_zero(17) - (1 - 0.05 ** (1 / 17))) < 1e-15
    assert upper95_zero(0) is None
