"""Known-answer tests for `EXP-VO-009`'s drivers.

These check the *tools*, not the VO. They exist because two of the numbers this experiment turns on
are produced here rather than by the estimator — the closed-form variance ratio of `LIT-VO-005`
eq. (8), and the random-walk-versus-systematic split of the accumulated heading error — and both are
easy to get subtly wrong in a way no downstream figure would reveal.

Run from inside `evaluation/`: ``python -m pytest tools/exp_vo_009 -q``
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


vt = _load("exp_vo_009_variance_theory", "variance_theory.py")
cr = _load("exp_vo_009_check_reproduction", "check_reproduction.py")


# --------------------------------------------------------------------------- the Umeyama fit

def _apply(theta, scale, tx, ty, p):
    c, s = math.cos(theta), math.sin(theta)
    R = scale * np.array([[c, -s], [s, c]])
    return p @ R.T + np.array([tx, ty])


@pytest.mark.parametrize("theta,scale", [(0.0, 1.0), (0.37, 1.0), (0.0, 1.21),
                                         (-2.9, 0.63), (3.05, 1.004)])
def test_similarity_fit_recovers_an_exact_transform(theta, scale):
    """The Python fit must agree with the Java `GenerateSimilarity2D` on exact input.

    Both implement `LIT-VO-005` eq. (4); if they ever disagree, the tier-1 Monte Carlo and the
    flight arms would be measuring two different estimators.
    """
    rng = np.random.default_rng(3)
    p = rng.normal(0, 100, (12, 2))
    q = _apply(theta, scale, 31.0, -12.0, p)
    th_hat, s_hat = vt.fit_similarity(p, q)
    assert math.isclose(math.sin(th_hat), math.sin(theta), abs_tol=1e-12)
    assert math.isclose(math.cos(th_hat), math.cos(theta), abs_tol=1e-12)
    assert math.isclose(s_hat, scale, rel_tol=1e-12)


def test_polar_rotation_matches_the_java_closed_form():
    """`atan2(a21 - a12, a11 + a22)` — the same expression `RigidMotionDecomposition` uses."""
    for theta in (-3.0, -0.4, 0.0, 0.4, 3.0):
        c, s = math.cos(theta), math.sin(theta)
        J = 1.37 * np.array([[c, -s], [s, c]])
        assert math.isclose(vt.polar_rotation(J), theta, abs_tol=1e-12)


# --------------------------------------------------------------------------- LIT-VO-005 eq. (8)

def test_variance_ratio_is_one_for_an_isotropic_sample():
    """The equality case of eq. (8): κ = 1 means the constraint buys exactly nothing."""
    # Four points on a circle: the centred second-moment matrix is a multiple of the identity.
    p = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    k = vt.kappa(p)
    assert math.isclose(k, 1.0, rel_tol=1e-9)
    assert math.isclose((2 + k + 1 / k) / 4, 1.0, rel_tol=1e-9)


def test_variance_ratio_matches_monte_carlo_per_sample():
    """Gate 1 in miniature: fix the points, vary only the noise, compare against eq. (8)."""
    rows = vt.run("per-sample", (6,), [(1.0, 1.0)], sigma=0.02, s_true=1.0,
                  theta_true=0.05, trials=20000, seed=5)
    r = rows[0]
    assert r["kappa"] > 1.0                       # a random 6-point sample is not isotropic
    assert abs(r["variance_ratio_observed"] - r["variance_ratio_predicted"]) \
        / r["variance_ratio_predicted"] < 0.05
    # ...and the constraint must never make the rotation *more* variable on the same sample.
    assert r["sd_similarity_rad"] <= r["sd_affine_rad"] * 1.01


# --------------------------------------------------------------------------- the heading split

def _summarise(err_deg: np.ndarray) -> dict:
    """Drive `rotation_across_models.summarise` with a known per-frame error series."""
    ram = _load("exp_vo_009_rotation_across_models", "rotation_across_models.py")
    inc = np.concatenate([[0.0], np.radians(err_deg)])
    return ram.summarise(inc, np.zeros_like(inc))


def test_pure_random_walk_scores_about_one_excess():
    """Zero-mean independent noise: the accumulation should sit near its random-walk prediction."""
    rng = np.random.default_rng(11)
    stats = _summarise(rng.normal(0.0, 0.2, 6000))
    assert stats["excess_over_random_walk"] < 3.0     # a walk, not a drift
    assert abs(stats["t_of_mean"]) < 3.0
    assert abs(stats["lag1_autocorrelation"]) < 0.05


def test_a_pure_bias_is_separated_from_the_walk():
    """A constant offset accumulates linearly, far beyond sd*sqrt(n)."""
    n = 6000
    stats = _summarise(np.full(n, 0.01) + np.random.default_rng(12).normal(0, 0.2, n))
    assert stats["excess_over_random_walk"] > 3.0
    assert stats["t_of_mean"] > 3.0
    assert math.isclose(stats["accumulated_error_deg"], stats["mean_error_deg"] * n, rel_tol=1e-9)


# --------------------------------------------------------------------------- reproduction gate

def test_reproduction_check_detects_a_single_changed_cell(tmp_path):
    header = "frame_index,est_x,est_y,est_yaw_deg,success,event,reference_id\n"
    body = ["0,0.0,0.0,0.0,true,init,0\n", "1,1.5,-2.5,0.25,true,none,0\n"]
    for name in ("ref", "same", "diff"):
        (tmp_path / name).mkdir()
    (tmp_path / "ref" / "frames.csv").write_text(header + "".join(body))
    (tmp_path / "same" / "frames.csv").write_text(header + "".join(body))
    (tmp_path / "diff" / "frames.csv").write_text(
        header + body[0] + body[1].replace("1.5", "1.5000000001"))

    ref = cr.load(tmp_path / "ref")
    assert ref == cr.load(tmp_path / "same")
    assert ref != cr.load(tmp_path / "diff"), \
        "a change in the 10th decimal must count as a mismatch; the gate is bit-identity, " \
        "not approximate agreement"
