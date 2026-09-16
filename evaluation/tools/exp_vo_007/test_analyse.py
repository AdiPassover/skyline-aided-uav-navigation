"""`EXP-VO-007`'s analysis must produce numbers comparable with `EXP-VO-006`'s, digit for digit.

The whole experiment is a comparison between two sequences. If `analyse.evaluate` differed from
`exp_vo_006/rigid_readout.evaluate` in any detail -- the alignment, the synchronisation, which
frames are kept, how the path length is measured -- then "AMtown01 is 2.1x worse than HKairport01"
would be partly an artefact of the analysis code and there would be no way to see it from the
output.

So the check is a known-answer one against the committed record: run this record's own `evaluate`
over the six committed `hkairport01-b` run records and require it to reproduce `EXP-VO-006` R1's
published normalised ATE for every readout and both motion models.

Needs only the tracked descriptors (`frames.csv`, `groundtruth.csv`, run records) -- no imagery, so
it passes in a fresh worktree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import analyse                                                           # noqa: E402
from naveval.dataset import load_dataset                                 # noqa: E402

REPO = Path(__file__).resolve().parents[3]

# EXP-VO-006 R1 (hkairport01-b), normalised ATE in per cent, as published.
COMMITTED = {
    "hkairport01-b-run-v1": 2.649,              # legacy homography
    "hkairport01-b-affine-v1": 1.099,           # legacy affine
    "hkairport01-b-homography-logical-v1": 7.056,
    "hkairport01-b-affine-logical-v1": 4.402,
    "hkairport01-b-homography-rigid-v1": 1.148,
    "hkairport01-b-affine-rigid-v1": 0.988,
}


@pytest.fixture(scope="module")
def dataset():
    root = REPO / "datasets" / "hkairport01-b"
    if not (root / "groundtruth.csv").exists():
        pytest.skip("hkairport01-b descriptors absent from this checkout")
    return load_dataset(root)


@pytest.mark.parametrize("run_id,expected", sorted(COMMITTED.items()))
def test_reproduces_exp_vo_006_normalised_ate(dataset, run_id, expected):
    run = REPO / "runs" / run_id
    if not (run / "frames.csv").exists():
        pytest.skip(f"{run_id} absent from this checkout")
    got = 100.0 * analyse.evaluate(run, dataset)["ate_rmse_normalised"]
    assert got == pytest.approx(expected, abs=0.002), (
        f"{run_id}: EXP-VO-007's analysis gives {got:.4f} %, EXP-VO-006 R1 published "
        f"{expected:.3f} % -- the two records' numbers are not comparable")


def test_partial_correlation_removes_a_pure_common_driver():
    """H4's statistic must not fire on two series that share only their growth with distance.

    Constructed so the raw rank correlation is ~1 and the true partial correlation is ~0: both
    quantities are strictly increasing functions of the control alone plus independent noise.
    """
    import numpy as np
    rng = np.random.default_rng(7)
    ctrl = np.arange(2000, dtype=float)
    a = ctrl + rng.normal(0, 50, ctrl.size)
    b = ctrl + rng.normal(0, 50, ctrl.size)
    raw = np.corrcoef(analyse._rank(a), analyse._rank(b))[0, 1]
    partial = analyse.partial_spearman(a, b, ctrl)
    assert raw > 0.9, raw
    assert abs(partial) < 0.2, partial
