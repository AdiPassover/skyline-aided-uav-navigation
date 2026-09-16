"""Known-answer tests for the two closure figures' underlying computations.

`closure_figures.py` draws pictures, which no test can check. What a test *can* check is the three
things that would make either picture say something false:

1. **The accumulated-anisotropy definition.** `σ₁/σ₂` of `J(G_k⁻¹)` at the image centre — inverse
   direction, evaluated at the centre. Both choices are load-bearing: the forward direction gives a
   different number for the homography, and any other evaluation point gives a different number
   again. A known-answer transform pins the formula, and the committed `EXP-VO-004` R4 medians pin
   the choice.
2. **The per-frame row filter.** Dropping `init` and `restart` is `EXP-VO-011`'s pre-declared
   filter; without it three arms' means disagree with the committed budget in the third digit.
3. **Circular safety of the heading split.** The offsets are 62–77°, so a series can straddle the
   wrap. A naive arithmetic mean is wrong by up to 180° there.

Tests that need committed run records skip cleanly when the records are absent, so the suite still
runs in a checkout that has not fetched them.

Run from inside `evaluation/`: ``python -m pytest tools/vo_closure -q``
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cf = _load("vo_closure_closure_figures", "closure_figures.py")


# --------------------------------------------------------------------- known-answer geometry

def test_accumulated_anisotropy_of_a_known_stretch():
    """A pure axis-aligned stretch by (a, b) has anisotropy max(a,b)/min(a,b), exactly.

    Built as `G` (the forward L -> C_k map) so the helper's inversion is exercised: the inverse of
    diag(a, b) is diag(1/a, 1/b), whose singular-value ratio is the same. The test would therefore
    still pass if the direction were flipped — which is deliberate, because it isolates the
    *formula* here and leaves the *direction* to the record-reproduction test below.
    """
    a, b = 3.0, 1.25
    g = np.array([[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, 1.0]])
    j = cf.rc._jacobian(np.linalg.inv(g), cf.CX, cf.CY)
    sv = np.linalg.svd(j, compute_uv=False)
    assert sv[0] / sv[1] == pytest.approx(a / b, rel=1e-12)


def test_rotation_alone_is_isotropic():
    """A rotation is not a deformation: composing thousands of them must not inflate anisotropy.

    This is the property that makes panel (d) meaningful — everything it shows above 1.0 is genuine
    stretch, not the flight's 91-496 degrees of accumulated turn leaking into the metric.
    """
    for deg in (0.0, 7.5, 91.0, 179.0, 359.0):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        g = np.array([[c, -s, 11.0], [s, c, -4.0], [0.0, 0.0, 1.0]])
        j = cf.rc._jacobian(np.linalg.inv(g), cf.CX, cf.CY)
        sv = np.linalg.svd(j, compute_uv=False)
        assert sv[0] / sv[1] == pytest.approx(1.0, abs=1e-12)


def test_uniform_scale_alone_is_isotropic_but_moves_the_scale_channel():
    """Panels (c) and (d) must separate: a pure zoom is all scale and no anisotropy."""
    m = 1.4
    g = np.array([[m, 0.0, 0.0], [0.0, m, 0.0], [0.0, 0.0, 1.0]])
    j = cf.rc._jacobian(np.linalg.inv(g), cf.CX, cf.CY)
    sv = np.linalg.svd(j, compute_uv=False)
    assert sv[0] / sv[1] == pytest.approx(1.0, abs=1e-12)
    assert math.sqrt(abs(np.linalg.det(j))) == pytest.approx(1.0 / m, rel=1e-12)


def test_tilt_ceiling_formula():
    """`1/cos θ` is what a rigid camera tilt can explain (`LIT-VO-003` §4)."""
    for deg, expect in ((0.0, 1.0), (15.524794054782385, 1.0378668643791706)):
        assert 1.0 / math.cos(math.radians(deg)) == pytest.approx(expect, rel=1e-12)


# --------------------------------------------------------------------- circular safety

def test_offset_split_is_circular_safe_across_the_branch_cut():
    """A series centred on 179° with a few degrees of spread must not report an offset near 0.

    The arithmetic mean of {179, -179} is 0; the circular mean is 180. The real arms sit at 62-77°,
    far from the cut, but the split must be correct by construction rather than by luck.
    """
    rng = np.random.default_rng(7)
    truth_offset, spread = 179.0, 3.0
    err = cf.wrap180(truth_offset + rng.normal(0.0, spread, 20000))
    got = cf.circular_mean_deg(err)
    assert abs(float(cf.wrap180(got - truth_offset))) < 0.1
    assert cf.circular_rms_about_deg(err, got) == pytest.approx(spread, rel=0.05)


def test_offset_split_recovers_a_constructed_offset_and_spread():
    rng = np.random.default_rng(11)
    for truth_offset in (-74.4, 0.0, 62.1):
        err = cf.wrap180(truth_offset + rng.normal(0.0, 7.0, 50000))
        got = cf.circular_mean_deg(err)
        assert abs(float(cf.wrap180(got - truth_offset))) < 0.15
        assert cf.circular_rms_about_deg(err, got) == pytest.approx(7.0, rel=0.03)
        # A pure-offset series must make the raw RMSE almost entirely the offset — the claim F8
        # exists to make.
        raw = float(np.sqrt((err ** 2).mean()))
        if abs(truth_offset) > 1.0:
            assert abs(truth_offset) / raw > 0.99


def test_raw_rmse_equals_tracking_rms_when_there_is_no_offset():
    """The decomposition must be a no-op on a well-datumed run, not a free reduction."""
    rng = np.random.default_rng(3)
    err = rng.normal(0.0, 5.0, 40000)
    got = cf.circular_mean_deg(err)
    assert cf.circular_rms_about_deg(err, got) == pytest.approx(
        float(np.sqrt((err ** 2).mean())), rel=1e-3)


# --------------------------------------------------------------------- committed-record agreement

def _have(rel: str) -> bool:
    return (REPO / rel / "logical_transform.csv").exists()


@pytest.mark.parametrize("window,model,expect_median", [
    ("hkairport01-a", "homography", 1.39),
    ("hkairport01-a", "affine", 1.35),
    ("hkairport01-b", "homography", 3.39),
    ("hkairport01-b", "affine", 2.19),
])
def test_accumulated_anisotropy_reproduces_exp_vo_004_r4(window, model, expect_median):
    """`EXP-VO-004` R4 published these four medians. Panel (d) must land on them.

    This is what pins the *direction* and the *evaluation point*: the forward Jacobian gives 1.80
    where the record says 1.39, so a flip here fails loudly instead of drawing a plausible-looking
    wrong curve.
    """
    rel = f"runs/{window}-{model}-rigid-v1"
    if not _have(rel):
        pytest.skip(f"{rel} not present in this checkout")
    a = cf.accumulated_anisotropy(rel)
    assert float(np.median(a)) == pytest.approx(expect_median, abs=0.005)


@pytest.mark.parametrize("window,model", [
    ("hkairport01-a", "homography"), ("hkairport01-a", "affine"),
    ("hkairport01-b", "homography"), ("hkairport01-b", "affine"),
    ("amtown01-c", "homography"), ("amtown01-c", "affine"),
])
def test_per_frame_filter_matches_exp_vo_011(window, model):
    """The row filter and the per-frame mean must agree with the committed budget exactly."""
    import json
    budget = REPO / "evaluations/exp-vo-011/scale_budget.json"
    rel = f"runs/{window}-{model}-rigid-v1"
    if not budget.exists() or not _have(rel):
        pytest.skip("committed EXP-VO-011 budget or run record not present in this checkout")
    ref = {(a["window"], a["model"]): a for a in json.loads(budget.read_text())["arms"]}
    if (window, model) not in ref:
        pytest.skip(f"{window}/{model} absent from scale_budget.json")
    mine = cf.sidecar_columns(rel)["inc_log_scale"]
    theirs = ref[(window, model)]["inc_log_scale"]
    assert mine.size == theirs["n"]
    assert float(mine.mean()) == pytest.approx(theirs["mean"], rel=1e-12)
    assert cf.mad_sigma(mine) == pytest.approx(theirs["mad_sigma"], rel=1e-12)


def test_dropping_the_filter_would_change_the_answer():
    """Guards the filter itself: if it were a no-op, keeping it would be cargo-culting.

    `hkairport01-a` homography has one restart, so the unfiltered mean must differ from the
    committed one — otherwise this module is passing the test above for the wrong reason.
    """
    import csv
    import json
    rel = "runs/hkairport01-a-homography-rigid-v1"
    budget = REPO / "evaluations/exp-vo-011/scale_budget.json"
    if not budget.exists() or not _have(rel):
        pytest.skip("committed artifacts not present in this checkout")
    with (REPO / rel / "logical_transform.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    unfiltered = np.array([float(r["inc_log_scale"]) for r in rows])[1:]     # drop `init` only
    ref = {(a["window"], a["model"]): a for a in json.loads(budget.read_text())["arms"]}
    committed = ref[("hkairport01-a", "homography")]["inc_log_scale"]
    assert unfiltered.size == committed["n"] + 1
    assert float(unfiltered.mean()) != pytest.approx(committed["mean"], rel=1e-6)
