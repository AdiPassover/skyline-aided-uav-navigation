"""Known-answer tests for `EXP-VO-010`'s drivers.

The load-bearing one is the heading decomposition. `EXP-VO-009` R2 found that the metric this lane
has quoted for four experiments is ~95 % a constant frame offset; `EXP-VO-010` H2 is decided on the
*corrected* quantity, so that quantity has to be trustworthy before any flight number is read.
Phase 5 therefore requires it validated two independent ways, and both are here:

1. against a **second, independently written implementation** — a numerical minimisation over the
   offset, rather than the closed-form circular mean the module uses;
2. against **known-answer synthetic series** whose offset and spread are constructed, including ones
   that straddle the 0/360 branch cut, where a naive arithmetic mean is wrong by up to 180 degrees.

Run from inside `evaluation/`: ``python -m pytest tools/exp_vo_010 -q``
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hm = _load("exp_vo_010_heading_metrics", "heading_metrics.py")


# --------------------------------------------------------------- second implementation

def independent_decompose(aligned_yaw_deg, gt_heading_deg, grid=720001):
    """A deliberately different route to the same two numbers.

    Instead of a closed-form circular mean it sweeps the offset over a fine grid and takes the one
    minimising the wrapped sum of squares. Slower and cruder, and that is the point: it shares no
    code path with the module under test beyond the wrap, so a sign slip or a mean/median confusion
    in either shows up as a disagreement.
    """
    err = np.asarray(aligned_yaw_deg, float) - np.asarray(gt_heading_deg, float)
    candidates = np.linspace(-180.0, 180.0, grid)
    # (n, ) errors against (g, ) candidates -> pick the candidate with the least SSE.
    best, best_cost = None, np.inf
    for c in candidates[::200]:            # coarse pass
        cost = float((((err - c + 180.0) % 360.0 - 180.0) ** 2).sum())
        if cost < best_cost:
            best, best_cost = c, cost
    for c in np.linspace(best - 0.5, best + 0.5, 20001):    # fine pass
        cost = float((((err - c + 180.0) % 360.0 - 180.0) ** 2).sum())
        if cost < best_cost:
            best, best_cost = c, cost
    resid = ((err - best + 180.0) % 360.0 - 180.0)
    return float(best), float(np.sqrt((resid ** 2).mean()))


# --------------------------------------------------------------- circular primitives

def test_wrap180_is_a_true_wrap():
    assert hm.wrap180(0.0) == 0.0
    assert hm.wrap180(180.0) == 180.0
    assert hm.wrap180(-180.0) == 180.0
    assert hm.wrap180(190.0) == pytest.approx(-170.0)
    assert hm.wrap180(-190.0) == pytest.approx(170.0)
    assert hm.wrap180(359.0) == pytest.approx(-1.0)
    assert np.allclose(hm.wrap180([720.0, -720.0, 361.0]), [0.0, 0.0, 1.0])


def test_circular_mean_survives_the_branch_cut():
    """The case that makes an arithmetic mean useless: values straddling 0/360."""
    headings = np.array([359.0, 1.0, 0.0, 358.0, 2.0])
    assert hm.circular_mean_deg(headings) == pytest.approx(0.0, abs=1e-9)
    assert np.mean(headings) == pytest.approx(144.0)      # what the naive answer would have been


# --------------------------------------------------------------- known answers

@pytest.mark.parametrize("offset", [0.0, 7.5, -63.8, 179.0, -179.0, 267.84 - 360.0])
@pytest.mark.parametrize("spread", [0.0, 1.0, 9.4])
def test_known_answer_offset_and_spread(offset, spread):
    """Construct a series with a known constant offset and known tracking noise, recover both."""
    rng = np.random.default_rng(7)
    n = 4000
    gt = rng.uniform(0, 360, n)
    noise = rng.normal(0.0, spread, n) if spread > 0 else np.zeros(n)
    aligned = (gt + offset + noise) % 360.0

    d = hm.decompose_heading(aligned, gt)
    assert hm.wrap180(d["constant_offset_deg"] - offset) == pytest.approx(0.0, abs=0.25)
    # The recovered spread is the sample RMS of the injected noise, not its nominal sigma.
    assert d["aligned_heading_rms_deg"] == pytest.approx(
        float(np.sqrt((hm.wrap180(noise - noise.mean()) ** 2).mean())), abs=0.05)


@pytest.mark.parametrize("offset,spread", [(63.8, 7.2), (-71.75, 9.43), (42.6, 9.3)])
def test_agrees_with_an_independent_implementation(offset, spread):
    """The real flights' measured (offset, spread) pairs, recovered two ways."""
    rng = np.random.default_rng(11)
    n = 3000
    gt = rng.uniform(0, 360, n)
    aligned = (gt + offset + rng.normal(0.0, spread, n)) % 360.0

    mine = hm.decompose_heading(aligned, gt)
    theirs_offset, theirs_rms = independent_decompose(aligned, gt)

    assert hm.wrap180(mine["constant_offset_deg"] - theirs_offset) == pytest.approx(0.0, abs=0.02)
    assert mine["aligned_heading_rms_deg"] == pytest.approx(theirs_rms, rel=2e-3)


def test_raw_rmse_is_dominated_by_the_offset_when_the_offset_is_large():
    """The EXP-VO-009 R2 finding, as an executable statement rather than a claim in prose."""
    rng = np.random.default_rng(3)
    n = 6000
    gt = rng.uniform(0, 360, n)
    aligned = (gt + 63.81 + rng.normal(0.0, 7.20, n)) % 360.0

    d = hm.decompose_heading(aligned, gt)
    assert d["raw_yaw_rmse_deg"] == pytest.approx(64.2, abs=0.6)
    assert d["aligned_heading_rms_deg"] == pytest.approx(7.2, abs=0.3)
    assert d["offset_fraction_of_raw"] > 0.98, \
        "the raw metric should be ~99 % offset here, which is why it must not be read as drift"


def test_zero_offset_leaves_the_raw_metric_meaning_what_it_appears_to():
    """The decomposition must not manufacture a correction where none is needed."""
    rng = np.random.default_rng(5)
    n = 2000
    gt = rng.uniform(0, 360, n)
    aligned = (gt + rng.normal(0.0, 4.0, n)) % 360.0
    d = hm.decompose_heading(aligned, gt)
    assert abs(d["constant_offset_deg"]) < 0.3
    assert d["raw_yaw_rmse_deg"] == pytest.approx(d["aligned_heading_rms_deg"], rel=0.01)


# --------------------------------------------------------------- per-frame statistics

def test_per_frame_stats_separate_a_walk_from_a_drift():
    rng = np.random.default_rng(13)
    n = 6000
    walk = hm.per_frame_rotation_stats(rng.normal(0.0, 0.2, n), np.zeros(n))
    assert walk["excess_over_random_walk"] < 3.0
    assert abs(walk["per_frame_t_of_mean"]) < 3.0

    drift = hm.per_frame_rotation_stats(np.full(n, 0.01) + rng.normal(0.0, 0.2, n), np.zeros(n))
    assert drift["excess_over_random_walk"] > 3.0
    assert drift["per_frame_t_of_mean"] > 3.0
    assert drift["cumulative_drift_deg"] == pytest.approx(n * drift["per_frame_mean_deg"], rel=1e-9)


def test_per_frame_error_is_wrapped():
    """Increments near +/-180 must not report a 360 degree error."""
    est = np.array([179.9, -179.9])
    gt = np.array([-179.9, 179.9])
    s = hm.per_frame_rotation_stats(est, gt)
    assert s["max_abs_error_deg"] < 1.0, "wrapped difference expected, not ~360"
