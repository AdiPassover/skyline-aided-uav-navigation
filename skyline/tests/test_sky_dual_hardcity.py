"""``EXP-SKY-012`` study-script rules (``scripts/sky_dual_hardcity.py``): the declared
North-primary / West-veto predicate, the flat-row round trip used to re-evaluate the frozen
``EXP-SKY-011`` populations read-only, and the declared A–D classification. T1 (synthetic rows)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from hsreloc.simret import dualview as dv                    # noqa: E402
from sky_dual_report import CHANNELS, KEEP_FIELDS, flat      # noqa: E402
from sky_relpose_study import _fmt                          # noqa: E402
import sky_dual_hardcity as study                           # noqa: E402

GATE = dv.Gate(0.9, 0.15, "frame")


def _row(n_score, n_margin, w_score, w_margin, agree, n_false=True):
    def view(score, margin, tid, false):
        return {"top1_score": score, "frame_margin": margin, "region_level_margin": margin,
                "false_region": false, "region_top1": not false, "top1_id": tid, "top1_distance_m": 500.0}
    return {"sa_agree": agree, "_out": {"north": view(n_score, n_margin, "rN", n_false),
                                        "west": view(w_score, w_margin, "rW", True)}}


def test_veto_rejects_when_north_fails_its_gate():
    assert not study.veto_accepted(_row(0.85, 0.30, 0.99, 0.30, agree=True), GATE)
    assert not study.veto_accepted(_row(0.99, 0.05, 0.99, 0.30, agree=True), GATE)


def test_veto_west_refusal_leaves_north_standing():
    # the documented weakness: a West that fails its own gate cannot contradict anything
    assert study.veto_accepted(_row(0.95, 0.20, 0.92, 0.01, agree=False), GATE)


def test_veto_west_contradiction_rejects():
    assert not study.veto_accepted(_row(0.95, 0.20, 0.95, 0.20, agree=False), GATE)


def test_veto_west_agreement_accepts():
    assert study.veto_accepted(_row(0.95, 0.20, 0.95, 0.20, agree=True), GATE)


def test_veto_never_accepts_more_than_north_alone():
    for w_score, w_margin, agree in ((0.99, 0.3, True), (0.99, 0.3, False), (0.5, 0.0, False)):
        for n_score, n_margin in ((0.95, 0.2), (0.85, 0.2), (0.95, 0.1)):
            r = _row(n_score, n_margin, w_score, w_margin, agree)
            assert study.veto_accepted(r, GATE) <= dv.view_passes(r["_out"]["north"], GATE)


def _full_row():
    out = {}
    for i, c in enumerate(CHANNELS):
        o = {}
        for j, k in enumerate(KEEP_FIELDS):
            if k in ("region_top1", "region_in_top_k", "exact_top1", "exact_in_top_k", "false_region", "ambiguous",
                     "accepted_frame", "accepted_region"):
                o[k] = bool((i + j) % 2)
            elif k in ("top1_id", "outcome"):
                o[k] = f"{c}-{k}"
            else:
                o[k] = 0.123456 + i + j * 0.01
        out[c] = o
    return {"level": "city", "split": "final", "source": "segformer", "memory": "dense", "holdout_m": 10.0,
            "query_key": "Run/sky_000001", "segment_id": "Run/seg00", "segment_kind": "star_swipe", "site": "HardSwipe@a16",
            "hard_tag": True, "phase": "center", "leg_axis": "lateral", "leg_direction": "east", "vertical_up_m": 0.0,
            "total_displacement_m": 12.0, "index_in_segment": 7, "east_m": 862.5, "north_m": -37.25, "up_m": 131.19,
            "primary": False, "horizontal": True, "n_references": 385, "d_near_m": 43.7, "nearest_key": "Run/sky_000002",
            "attainable": True, "gate_level": "region", "sa_agree": False, "sa_agree_tau2": False,
            "sa_same_reference": False, "sa_top1_separation_m": 1170.5, "sa_candidate_id": None,
            "sa_candidate_distance_m": None, "sa_false_region": True, "sa_dual_region_top1": False,
            "sa_dual_region_in_top_k": False, "sa_min_score": 0.9, "sa_accepted_frozen": False,
            "sa_accepted_false_frozen": False, "_out": out}


def test_unflatten_inverts_the_csv_flattening():
    row = _full_row()
    as_csv = {k: _fmt(v) for k, v in flat(row).items()}      # what _write_csv puts on disk
    back = study._unflatten(as_csv)
    for k in ("level", "memory", "query_key", "segment_kind", "leg_axis", "sa_candidate_id"):
        assert back[k] == row[k]
    for k in ("hard_tag", "primary", "horizontal", "attainable", "sa_agree", "sa_false_region"):
        assert back[k] is row[k]
    assert back["holdout_m"] == row["holdout_m"] and back["index_in_segment"] == row["index_in_segment"]
    for c in CHANNELS:
        for k in KEEP_FIELDS:
            a, b = back["_out"][c][k], row["_out"][c][k]
            if isinstance(b, bool):
                assert a is b
            elif isinstance(b, str):
                assert a == b
            else:
                assert a == pytest.approx(b, abs=1e-5)


def _metrics(strict_false=0, min_false=0, north_frozen_false=(82, 0), north_dev_false=(3, 0), min_cov=0.38, n=87):
    pops = {}
    for s_i, s in enumerate(("segformer", "sim_exact")):
        for name in study.HARD_POPS:
            for rule in ("strict", "min", "north", "west", "mean", "veto"):
                for tag in ("dev", "frozen"):
                    f = {"strict": strict_false, "min": min_false,
                         "north": (north_frozen_false if tag == "frozen" else north_dev_false)[s_i]}.get(rule, 0)
                    f = f if (rule != "strict" and rule != "min") or name == "hard_sparse_h" else 0
                    pops[f"{s}/{name}/{rule}/{tag}"] = {"n": n, "n_accepted_false": f, "accepted_false_rate": f / n}
        pops[f"{s}/hard_dense0_h_attainable/strict/dev"] = {"n": n, "n_accepted_false": 0, "coverage": 0.0}
        pops[f"{s}/hard_dense0_h_attainable/min/dev"] = {"n": n, "n_accepted_false": 0, "coverage": min_cov}
    return {"sources": ["segformer", "sim_exact"], "populations": pops,
            "combined_hard_negatives": {"strict": {}, "min": {}}}


def test_classification_D_needs_zero_false_and_city_coverage():
    assert study.classify(_metrics())["label"] == "D"
    assert study.classify(_metrics(min_cov=0.10))["label"] == "C"


def test_classification_B_and_A_thresholds():
    # 2 strict false of 4 x 87 = 348 hard queries is <= 2 % and <= half of North's 82 -> B
    assert study.classify(_metrics(strict_false=2))["label"] == "B"
    # 10 strict false is > 2 % of 348 -> A
    assert study.classify(_metrics(strict_false=10))["label"] == "A"


def test_classification_weakest_view_false_blocks_C():
    assert study.classify(_metrics(min_false=1))["label"] in ("A", "B")
