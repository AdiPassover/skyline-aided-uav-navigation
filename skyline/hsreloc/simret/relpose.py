"""Relative horizontal position from a matched, north-facing skyline pair (EXP-SKY-009).

Research question (owner brief, 2026-09-05): given two matched, north-aligned skyline observations
and known camera-height information, what relative horizontal pose information — ``Δx``/``Δy``
between a stored reference and a later query — can reliably be recovered? This module holds the
**geometry helpers and estimators** the study script drives; it decides nothing on its own and it
touches no frozen module. ``hsreloc/retrieval/`` and ``hsreloc/matchers/`` are imported, never edited:
the global lag is still the frozen C1 matcher's, and the windowed lag search below is validated
bit-equal to it on a full-width window (``tests/test_simret_relpose.py``).

Geometry, first order, north-facing camera (heading is *not* a variable here)
------------------------------------------------------------------------------
A skyline point at horizontal range ``D`` and bearing ``α`` (radians east of north, which for a
world-locked-North camera is also its viewing azimuth) seen from the reference camera is seen from a
query camera displaced by ``(ΔE, ΔN)`` at bearing ``α + Δα`` with, to first order in ``t / D``::

    Δα(α) = (−ΔE · cos α + ΔN · sin α) / D(α)                                   (1)

so a lateral (east) move shifts the whole skyline left by an amount that depends on range, and a
longitudinal (north) move stretches it about the optical axis — the LIT-SKY-005 shift/scale pair.
The frozen profile indexes by **image fraction**, and the C1 lag ``δ`` means ``q[i] ≈ r[i + δ]``:
what the reference saw at sample ``i + δ`` the query sees at sample ``i``, i.e. that feature moved by
``α(i) − α(i + δ)`` in azimuth (``α(i) = atan((x_i − cx) / fx)``, the exact pinhole map, not the
small-angle rate). Equation (1) is linear in ``(ΔE, ΔN)`` **only once every ``D`` is known**: with one
global lag there is one equation and three unknowns, which is the observability problem the study
measures rather than assumes away.

Height enters through the vertical: a query ``Δz`` above the reference sees the boundary at range
``D`` lower in the frame by ``Δrow = fy · Δz / D`` (rows increase downward; exact for the height
term, ignoring the second-order effect of the horizontal move on elevation). So a known height
difference **measures range per column** — and therefore converts a lag into metres — but only when
``|Δz|`` is large enough for ``Δrow`` to be resolved, which is exactly the regime level flight avoids.

Sign conventions, stated once: ``ΔE, ΔN, Δz = query − reference`` in the ENU frame of
``hsreloc.simret.conventions``; a positive C1 lag is what an eastward query produces; ``Δrow > 0``
is what a higher query produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hsreloc.matchers import BoundedLagNccMatcher
from hsreloc.retrieval.profile import ProfileConfig, normalize

RELPOSE_VERSION = "1.0.0"

#: Frozen C1-32 bound (DEC-SKY-007) and the frozen overlap floor.
DEFAULT_MAX_LAG_SAMPLES = 32
DEFAULT_MIN_OVERLAP_FRAC = 0.6
#: Windows the profile is cut into for the local-lag representation (declared, not tuned).
DEFAULT_N_WINDOWS = 8
#: Ranges outside this band are not skyline ranges this camera could have resolved.
RANGE_BOUNDS_M = (25.0, 30000.0)


class RelPoseError(Exception):
    """A helper was asked for something the data cannot support."""


# --------------------------------------------------------------------------------------------------
# camera / profile coordinate mapping
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfileCamera:
    """What converts a profile sample into a viewing azimuth: the pinhole model plus the resampling.

    ``height_profile`` resamples ``W`` columns onto ``n`` samples with ``linspace(0, 1, n)``, so sample
    ``i`` sits at column ``i · (W − 1) / (n − 1)``; nothing here assumes 512 or 256.
    """

    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float
    n_samples: int = 256

    @classmethod
    def from_session_meta(cls, meta: dict, n_samples: int = 256) -> "ProfileCamera":
        cam = meta.get("camera") or {}
        intr = cam.get("intrinsics") or {}
        res = cam.get("resolution_px") or [None, None]
        missing = [k for k in ("fx", "fy", "cx", "cy") if intr.get(k) is None]
        if missing or res[0] is None:
            raise RelPoseError(f"session carries no usable intrinsics (missing {missing or 'resolution_px'})")
        return cls(int(res[0]), int(res[1]), float(intr["fx"]), float(intr["fy"]),
                   float(intr["cx"]), float(intr["cy"]), int(n_samples))

    @property
    def columns_per_sample(self) -> float:
        return (self.width_px - 1) / float(self.n_samples - 1)

    def sample_to_column(self, sample) -> np.ndarray:
        return np.asarray(sample, dtype=np.float64) * self.columns_per_sample

    def sample_to_azimuth_rad(self, sample) -> np.ndarray:
        """Viewing azimuth of a (possibly fractional) profile sample, radians, positive to the right (east)."""
        return np.arctan((self.sample_to_column(sample) - self.cx) / self.fx)

    def lag_to_azimuth_shift_rad(self, sample, lag) -> np.ndarray:
        """Azimuth motion of the feature a lag ``δ`` aligns at sample ``i``: ``α(i) − α(i + δ)``.

        Exact pinhole map: ``dx/dα = fx · sec²α``, so the same pixel lag is worth *fewer* degrees near
        the frame edge than on axis — the reason ``lag_samples_for_degrees`` calls itself conservative.
        """
        s = np.asarray(sample, dtype=np.float64)
        return self.sample_to_azimuth_rad(s) - self.sample_to_azimuth_rad(s + np.asarray(lag, dtype=np.float64))

    def as_dict(self) -> dict:
        return {"width_px": self.width_px, "height_px": self.height_px, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy, "n_samples": self.n_samples,
                "columns_per_sample": self.columns_per_sample}


def predicted_azimuth_shift_rad(delta_east_m: float, delta_north_m: float, azimuth_rad, range_m) -> np.ndarray:
    """Equation (1): first-order azimuth motion of a skyline point for a query displaced by ``(ΔE, ΔN)``."""
    a = np.asarray(azimuth_rad, dtype=np.float64)
    D = np.asarray(range_m, dtype=np.float64)
    if np.any(D <= 0):
        raise RelPoseError("range must be positive")
    return (-float(delta_east_m) * np.cos(a) + float(delta_north_m) * np.sin(a)) / D


# --------------------------------------------------------------------------------------------------
# lag search: full-width (== frozen C1) and windowed
# --------------------------------------------------------------------------------------------------

def _shift_matrix(r: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """``S[l, i] = r[i + lag_l]`` where that index exists, NaN elsewhere."""
    n = r.size
    idx = np.arange(n)[None, :] + lags[:, None].astype(int)
    valid = (idx >= 0) & (idx < n)
    S = np.full(idx.shape, np.nan)
    S[valid] = r[np.clip(idx, 0, n - 1)][valid]
    return S


def _masked_ncc(Q: np.ndarray, S: np.ndarray, floor: int) -> np.ndarray:
    """NCC of ``Q`` (n,) against each row of ``S`` (L, n) over the non-NaN overlap, each window
    re-centred on its own overlap — the frozen ``profile_distance`` primitive per row."""
    valid = ~np.isnan(S)
    cnt = valid.sum(axis=1)
    Sm = np.where(valid, S, 0.0)
    Qm = np.where(valid, Q[None, :], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_s = Sm.sum(axis=1) / cnt
        mean_q = Qm.sum(axis=1) / cnt
        a = np.where(valid, Qm - mean_q[:, None], 0.0)
        b = np.where(valid, Sm - mean_s[:, None], 0.0)
        denom = np.sqrt((a * a).sum(axis=1)) * np.sqrt((b * b).sum(axis=1)) + 1e-12
        score = (a * b).sum(axis=1) / denom
    score[cnt < floor] = -np.inf
    return score


@dataclass(frozen=True)
class LagSearch:
    """One bounded-lag NCC search over one window, with the score-versus-lag curve retained.

    The frozen matcher keeps only the winning lag and score; the curve is what the peak-shape
    diagnostics (``secondary_margin``, ``curvature``) are read from — information C1 computes and
    discards.
    """

    lag: float
    score: float
    score_at_zero: Optional[float]
    overlap: int
    saturated: bool
    secondary_margin: float
    curvature: float
    n_searched: int
    lags: np.ndarray = field(repr=False, compare=False)
    scores: np.ndarray = field(repr=False, compare=False)


def _search(scores: np.ndarray, lags: np.ndarray, max_lag: int, secondary_exclusion: int = 4) -> LagSearch:
    finite = np.isfinite(scores)
    if not finite.any():
        return LagSearch(lag=math.nan, score=-math.inf, score_at_zero=None, overlap=0, saturated=False,
                         secondary_margin=math.nan, curvature=math.nan, n_searched=0, lags=lags, scores=scores)
    best = scores.max()
    # the frozen tie-break: least |lag|, then the negative one
    cands = np.flatnonzero(scores == best)
    j = min(cands, key=lambda k: (abs(lags[k]), lags[k]))
    lag = float(lags[j])
    far = finite & (np.abs(lags - lag) >= secondary_exclusion)
    secondary = float(best - scores[far].max()) if far.any() else math.nan
    nb = [scores[k] for k in (j - 1, j + 1) if 0 <= k < scores.size and np.isfinite(scores[k])]
    curvature = float(best - np.mean(nb)) if nb else math.nan
    zero = np.flatnonzero(lags == 0)
    at_zero = float(scores[zero[0]]) if zero.size and np.isfinite(scores[zero[0]]) else None
    return LagSearch(lag=lag, score=float(best), score_at_zero=at_zero, overlap=0,
                     saturated=bool(abs(lag) >= max_lag), secondary_margin=secondary,
                     curvature=curvature, n_searched=int(finite.sum()), lags=lags, scores=scores)


def bounded_lag_search(query: np.ndarray, reference: np.ndarray, max_lag: int = DEFAULT_MAX_LAG_SAMPLES,
                       min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC) -> LagSearch:
    """Vectorised full-width bounded-lag NCC. Bit-equal winner/score to ``BoundedLagNccMatcher`` at
    integer lags (asserted by test); additionally keeps the whole score-vs-lag curve."""
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if q.shape != r.shape or q.ndim != 1:
        raise RelPoseError(f"profiles must be 1-D and equal length, got {q.shape} vs {r.shape}")
    n = q.size
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.float64)
    S = _shift_matrix(r, lags)
    floor = max(2, int(math.ceil(min_overlap_frac * n)))
    scores = _masked_ncc(q, S, floor)
    res = _search(scores, lags, max_lag)
    if np.isfinite(res.score):
        overlap = int((~np.isnan(S[int(res.lag) + max_lag])).sum())
        res = LagSearch(**{**res.__dict__, "overlap": overlap})
    return res


def window_bounds(n: int, n_windows: int) -> list:
    """``[(start, stop), ...]`` — equal windows over ``n`` samples, last one absorbing the remainder."""
    if n_windows < 1 or n_windows > n:
        raise RelPoseError(f"n_windows must be in [1, {n}], got {n_windows}")
    edges = np.linspace(0, n, n_windows + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def windowed_lag_search(query: np.ndarray, reference: np.ndarray, n_windows: int = DEFAULT_N_WINDOWS,
                        max_lag: int = DEFAULT_MAX_LAG_SAMPLES,
                        min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                        min_window_std: float = 1e-4) -> list:
    """Local lag per window: the query window against the reference sampled at ``i + δ``.

    Each window is searched independently over the same bound, re-centred on its own overlap. A
    window whose query content is flat (std below ``min_window_std`` in profile units) is reported
    with ``score = nan``: correlating a constant against anything is not a measurement.
    Returns one dict per window with its centre sample, centre azimuth left to the caller.
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if q.shape != r.shape or q.ndim != 1:
        raise RelPoseError(f"profiles must be 1-D and equal length, got {q.shape} vs {r.shape}")
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.float64)
    S = _shift_matrix(r, lags)
    out = []
    for k, (a, b) in enumerate(window_bounds(q.size, n_windows)):
        w = b - a
        floor = max(2, int(math.ceil(min_overlap_frac * w)))
        qs = q[a:b]
        centre = (a + b - 1) / 2.0
        if float(np.std(qs)) < min_window_std:
            out.append({"window": k, "start": a, "stop": b, "centre_sample": centre, "lag": math.nan,
                        "score": math.nan, "secondary_margin": math.nan, "curvature": math.nan,
                        "saturated": False, "flat": True, "query_std": float(np.std(qs))})
            continue
        scores = _masked_ncc(qs, S[:, a:b], floor)
        res = _search(scores, lags, max_lag)
        out.append({"window": k, "start": a, "stop": b, "centre_sample": centre, "lag": res.lag,
                    "score": res.score if np.isfinite(res.score) else math.nan,
                    "secondary_margin": res.secondary_margin, "curvature": res.curvature,
                    "saturated": res.saturated, "flat": False, "query_std": float(np.std(qs))})
    return out


def frozen_c1(query: np.ndarray, reference: np.ndarray, max_lag: int = DEFAULT_MAX_LAG_SAMPLES):
    """The actual frozen matcher object, for the study's primary lag column (no reimplementation)."""
    return BoundedLagNccMatcher(max_lag_samples=max_lag, min_overlap_frac=DEFAULT_MIN_OVERLAP_FRAC).match(
        query, reference)


def profile_of(curve, config: ProfileConfig | None = None) -> np.ndarray:
    """The frozen profile stage, unchanged (256 samples, mean removed, image-fraction units)."""
    return normalize(curve, config or ProfileConfig())


# --------------------------------------------------------------------------------------------------
# height as a constraint: range from vertical parallax
# --------------------------------------------------------------------------------------------------

def range_from_vertical_parallax(row_ref, row_query, delta_z_m: float, fy: float,
                                 min_abs_dz_m: float = 20.0, bounds_m=RANGE_BOUNDS_M) -> np.ndarray:
    """Per-column horizontal range from a known height difference: ``D = fy · Δz / Δrow``.

    ``Δrow = row_query − row_ref`` (rows increase downward; a higher query sees the boundary lower).
    NaN where ``|Δz|`` is below the declared minimum (the parallax would be unresolved at pixel
    precision), where ``Δrow`` has the wrong sign or is zero, or where the implied range leaves the
    declared band. This is the whole of what "known height" contributes to horizontal geometry:
    a range, not a position.
    """
    if abs(float(delta_z_m)) < float(min_abs_dz_m):
        return np.full(np.asarray(row_ref).shape, np.nan)
    d_row = np.asarray(row_query, dtype=np.float64) - np.asarray(row_ref, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        D = float(fy) * float(delta_z_m) / d_row
    lo, hi = bounds_m
    D = np.where(np.isfinite(D) & (D >= lo) & (D <= hi), D, np.nan)
    return D


def window_ranges(range_per_column: np.ndarray, camera: ProfileCamera, n_windows: int,
                  min_valid_frac: float = 0.3) -> np.ndarray:
    """Median range per profile window from a per-column range map (NaN = no usable range)."""
    R = np.asarray(range_per_column, dtype=np.float64)
    out = np.full(n_windows, np.nan)
    cps = camera.columns_per_sample
    for k, (a, b) in enumerate(window_bounds(camera.n_samples, n_windows)):
        c0, c1 = int(round(a * cps)), int(round((b - 1) * cps)) + 1
        seg = R[c0:c1]
        ok = np.isfinite(seg)
        if ok.mean() >= min_valid_frac:
            out[k] = float(np.median(seg[ok]))
    return out


# --------------------------------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Estimate:
    delta_east_m: Optional[float]
    delta_north_m: Optional[float]
    valid: bool
    refusal: Optional[str] = None
    residual_rad: Optional[float] = None
    n_equations: int = 0
    condition: Optional[float] = None

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}dE": self.delta_east_m, f"{prefix}dN": self.delta_north_m, f"{prefix}valid": self.valid,
                f"{prefix}refusal": self.refusal, f"{prefix}residual_rad": self.residual_rad,
                f"{prefix}n_eq": self.n_equations, f"{prefix}cond": self.condition}


REFUSED = Estimate(None, None, False)


def estimate_global_lag(lag: float, camera: ProfileCamera, range_m: float,
                        saturated: bool = False) -> Estimate:
    """Model A: one global lag + one scalar range → ``ΔE`` only (``ΔN`` is unobservable from it).

    The lag is read at the profile centre through the exact azimuth map, then equation (1) at
    ``α = 0`` gives ``ΔE = −Δα · D``.
    """
    if lag is None or not np.isfinite(lag):
        return Estimate(None, None, False, "no_lag")
    if saturated:
        return Estimate(None, None, False, "lag_saturated")
    if not (range_m and np.isfinite(range_m) and range_m > 0):
        return Estimate(None, None, False, "no_range")
    centre = (camera.n_samples - 1) / 2.0
    d_alpha = float(camera.lag_to_azimuth_shift_rad(centre, lag))
    return Estimate(-d_alpha * float(range_m), 0.0, True, None, None, 1, None)


def solve_translation(azimuth_rad, azimuth_shift_rad, range_m, weights=None, min_equations: int = 3,
                      max_residual_rad: Optional[float] = None, max_condition: float = 50.0) -> Estimate:
    """Model B/C: least-squares ``(ΔE, ΔN)`` from windowed azimuth shifts with known per-window range.

    Row ``k`` of the system is ``Δα_k · D_k = −ΔE cos α_k + ΔN sin α_k``. Windows with a NaN shift or
    range drop out; fewer than ``min_equations`` survivors, an ill-conditioned system (the ``ΔN``
    column vanishes for a narrow field of view), or a residual above ``max_residual_rad`` refuse.
    """
    a = np.asarray(azimuth_rad, dtype=np.float64)
    s = np.asarray(azimuth_shift_rad, dtype=np.float64)
    D = np.asarray(range_m, dtype=np.float64)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(s) & np.isfinite(D) & (D > 0) & np.isfinite(w) & (w > 0)
    if ok.sum() < min_equations:
        return Estimate(None, None, False, f"too_few_windows({int(ok.sum())})", None, int(ok.sum()))
    A = np.stack([-np.cos(a[ok]), np.sin(a[ok])], axis=1) * np.sqrt(w[ok])[:, None]
    y = (s[ok] * D[ok]) * np.sqrt(w[ok])
    sv = np.linalg.svd(A, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else math.inf
    if cond > max_condition:
        return Estimate(None, None, False, "ill_conditioned", None, int(ok.sum()), cond)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = A @ x - y
    # residual back in radians per equation (undo the range scaling)
    resid_rad = float(np.sqrt(np.mean((resid / (D[ok] * np.sqrt(w[ok]))) ** 2)))
    if max_residual_rad is not None and resid_rad > max_residual_rad:
        return Estimate(None, None, False, "residual_too_large", resid_rad, int(ok.sum()), cond)
    return Estimate(float(x[0]), float(x[1]), True, None, resid_rad, int(ok.sum()), cond)


def estimate_from_windows(windows: list, camera: ProfileCamera, range_per_window, min_score: float = 0.0,
                          exclude_saturated: bool = True, **solver_kwargs) -> Estimate:
    """Model B/C wrapper: windowed lags → azimuth shifts at each window centre → ``solve_translation``."""
    az, sh, D, w = [], [], [], []
    R = np.asarray(range_per_window, dtype=np.float64)
    if R.ndim == 0:
        R = np.full(len(windows), float(R))
    for k, win in enumerate(windows):
        lag, score = win.get("lag"), win.get("score")
        if lag is None or score is None or not (np.isfinite(lag) and np.isfinite(score)):
            continue
        if score < min_score or (exclude_saturated and win.get("saturated")):
            continue
        az.append(float(camera.sample_to_azimuth_rad(win["centre_sample"])))
        sh.append(float(camera.lag_to_azimuth_shift_rad(win["centre_sample"], lag)))
        D.append(float(R[k]) if k < R.size else math.nan)
        w.append(1.0)
    return solve_translation(az, sh, D, w, **solver_kwargs)


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------

def error_stats(err: np.ndarray, catastrophic_m: float = 37.5) -> dict:
    """The brief's minimum: median / mean / p90 / p95 / max / catastrophic fraction, never mean alone."""
    e = np.asarray(err, dtype=np.float64)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "median": float(np.median(e)), "mean": float(np.mean(e)),
            "p90": float(np.percentile(e, 90)), "p95": float(np.percentile(e, 95)), "max": float(e.max()),
            "frac_over_catastrophic": float((e > catastrophic_m).mean()), "catastrophic_m": catastrophic_m}


def snap_comparison(e_snap: np.ndarray, e_refined: np.ndarray, e_before_grid) -> dict:
    """The INT safety logic on a pair set.

    ``e_snap`` is the error of ``p_reloc = p_ref`` (= the true separation); ``e_refined`` that of
    ``p_ref + Δp̂``. Both are compared with a VO error ``e_before`` — not measured on this data, so
    reported over a declared grid: a correction is *harmful* when it leaves the vehicle further from
    truth than it was, ``Δe = e_after − e_before > 0``.
    """
    s = np.asarray(e_snap, dtype=np.float64)
    r = np.asarray(e_refined, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    out = {"n": int(s.size), "e_snap": error_stats(s), "e_refined": error_stats(r),
           "improvement_m": error_stats(s - r) if s.size else {"n": 0},
           "frac_refined_better": float((r < s).mean()) if s.size else None,
           "frac_refined_worse_by_10m": float((r > s + 10.0).mean()) if s.size else None,
           "harmful_rate_vs_e_before": {}}
    for eb in e_before_grid:
        out["harmful_rate_vs_e_before"][f"{float(eb):g}"] = {
            "snap": float((s > eb).mean()) if s.size else None,
            "refined": float((r > eb).mean()) if s.size else None,
            "delta_e_snap_median": float(np.median(s - eb)) if s.size else None,
            "delta_e_refined_median": float(np.median(r - eb)) if s.size else None}
    return out


# --------------------------------------------------------------------------------------------------
# the proposed SKY-side interface for INT (EXP-SKY-009 §Interface) — PROPOSED, NOT INTEGRATED
# --------------------------------------------------------------------------------------------------

#: Provenance strings for the per-window range the estimate used. INT must be able to see which one
#: it is: the study's evidence is an oracle map (what a DEM-backed database carries) and a weaker
#: extractor-built map; a scalar prior is measured as insufficient.
RANGE_SOURCES = ("reference_range_map", "pair_height_difference", "scalar_prior")


@dataclass(frozen=True)
class SkylineRelativePositionEstimate:
    """One query→reference relative horizontal displacement, with the variables that define its validity.

    Deliberately **no confidence scalar**. The domain of validity measured in ``EXP-SKY-009`` is
    defined by observable quantities, so those are what is exposed: the frozen matcher's own outputs
    (``c1_score``, ``c1_lag_samples``, ``c1_lag_saturated``), the windowed-solve diagnostics
    (``n_windows_used``, ``min_window_score``, ``fit_residual_rad`` — the least-squares residual in
    radians, the single best predictor of the error in the study), and where the range came from.
    ``valid`` is the conjunction of the declared gate; ``refusal`` names the first failed condition.
    ``delta_east_m``/``delta_north_m`` are ``query − reference`` in the ENU frame of
    ``hsreloc.simret.conventions``; heading is **not** estimated and is not a field.
    """

    reference_id: str
    query_id: str
    delta_east_m: Optional[float]
    delta_north_m: Optional[float]
    valid: bool
    refusal: Optional[str]
    c1_score: float
    c1_lag_samples: float
    c1_lag_saturated: bool
    n_windows_used: int
    min_window_score: Optional[float]
    fit_residual_rad: Optional[float]
    range_source: Optional[str]
    range_median_m: Optional[float]
    frame: str = "ENU query-minus-reference; heading not estimated"

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_GATE = {"min_c1_score": 0.9, "min_window_score": 0.7, "max_residual_rad": 0.01, "min_windows": 3}


def relative_position_estimate(reference_id: str, query_id: str, query_profile: np.ndarray,
                               reference_profile: np.ndarray, camera: ProfileCamera,
                               range_per_window, range_source: str,
                               gate: dict | None = None, n_windows: int = DEFAULT_N_WINDOWS,
                               max_lag: int = DEFAULT_MAX_LAG_SAMPLES) -> SkylineRelativePositionEstimate:
    """Model C end to end on two frozen profiles: frozen C1 → windowed lags → range-scaled solve → gate.

    ``range_per_window`` is what the caller knows about range (a reference-side map reduced to
    windows, a pair's height-difference range, or a scalar prior); ``range_source`` names which. The
    function never invents a range: with none, every window drops out and the estimate refuses.
    """
    if range_source not in RANGE_SOURCES:
        raise RelPoseError(f"range_source must be one of {RANGE_SOURCES}, got {range_source!r}")
    g = {**DEFAULT_GATE, **(gate or {})}
    c1 = frozen_c1(query_profile, reference_profile, max_lag)
    saturated = bool(np.isfinite(c1.score) and abs(c1.shift) >= max_lag)
    R = np.asarray(range_per_window, dtype=np.float64)
    if R.ndim == 0:
        R = np.full(n_windows, float(R))
    wins = windowed_lag_search(query_profile, reference_profile, n_windows, max_lag)
    used = [w for k, w in enumerate(wins) if np.isfinite(w["lag"]) and np.isfinite(w["score"])
            and w["score"] >= g["min_window_score"] and not w["saturated"]
            and k < R.size and np.isfinite(R[k]) and R[k] > 0]
    min_ws = min((w["score"] for w in used), default=None)
    est = estimate_from_windows(wins, camera, R, min_score=g["min_window_score"],
                                min_equations=int(g["min_windows"]), max_residual_rad=float(g["max_residual_rad"]))
    refusal = None
    if not np.isfinite(c1.score):
        refusal = "no_alignment"
    elif c1.score < g["min_c1_score"]:
        refusal = "c1_score_below_gate"
    elif saturated:
        refusal = "c1_lag_saturated"
    elif not est.valid:
        refusal = est.refusal
    finite = R[np.isfinite(R)]
    return SkylineRelativePositionEstimate(
        reference_id=reference_id, query_id=query_id,
        delta_east_m=est.delta_east_m if refusal is None else None,
        delta_north_m=est.delta_north_m if refusal is None else None,
        valid=refusal is None, refusal=refusal,
        c1_score=float(c1.score), c1_lag_samples=float(c1.shift), c1_lag_saturated=saturated,
        n_windows_used=int(est.n_equations if est.n_equations else len(used)),
        min_window_score=None if min_ws is None else float(min_ws),
        fit_residual_rad=est.residual_rad, range_source=range_source,
        range_median_m=float(np.median(finite)) if finite.size else None)
