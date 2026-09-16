"""EXP-SKY-013 — how much positional resolution does bounded-lag freedom cost?

``prototype``. Additive: this module imports the frozen matcher, the bit-equal vectorised search of
:mod:`hsreloc.simret.relpose`, the batched bank of :mod:`hsreloc.simret.recog` and the dual-view
rules of :mod:`hsreloc.simret.dualview`, and **edits none of them**. It adds exactly two things the
earlier studies did not need:

* :class:`NestedBoundBank` — score a query against a whole database **once**, at the frozen bound
  32, and read off every nested bound (C0, C1-4, C1-8, C1-16, C1-32) from the same per-lag scores.
* the positional-resolution bookkeeping INT needs: *how far away is the reference the matcher
  picked*, versus *how far away is the nearest one it could have picked*.

Why the nested trick is exact, not an approximation
---------------------------------------------------
The lag grid of bound ``L'`` is the contiguous sub-slice ``[L-L' : L+L'+1]`` of the bound-``L``
grid, and the per-lag score of a given (query, reference, lag) triple does not depend on which
bound is being searched — it is one call of the frozen ``_masked_ncc`` on one shifted row. So the
bound-``L'`` result is ``winner_per_reference`` applied to that slice, using the *unchanged*
function. The tie-break survives slicing: its key is ``|lag| * 2 * size + (lag + size)``, whose
first term steps by ``2 * size = 4L'+2`` between adjacent ``|lag|`` while the second term spans only
``2L'``, so ordering by ``|lag|`` then by sign is preserved at every size.

The practical consequence is the one the experiment needs: the five arms see **numerically
identical** per-lag scores, so any difference between them is attributable to the alignment freedom
alone and to nothing else. :func:`verify_bounds` re-checks that against the real matcher objects —
winning lag exactly, score to ``1e-12`` — and the study script calls it periodically so a
divergence aborts the run rather than being reported.

Conventions follow ``PROT-SKY-001``: positions are ENU metres, heights ENU up in metres, ranks are
1-based, ties break by ``(-score, reference_id)``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from hsreloc.matchers import BoundedLagNccMatcher, FrozenNccMatcher
from hsreloc.simret.recog import RecogError, winner_per_reference
from hsreloc.simret.relpose import (DEFAULT_MAX_LAG_SAMPLES, DEFAULT_MIN_OVERLAP_FRAC, _masked_ncc,
                                    _shift_matrix)

LAGSTUDY_VERSION = "1.0.0"

#: The pre-registered arms. 0 is C0: ``BoundedLagNccMatcher(max_lag_samples=0)`` is bit-identical to
#: ``FrozenNccMatcher`` in score (tests/test_matchers.py:235), so C0 is run once, not twice.
DEFAULT_BOUNDS = (0, 4, 8, 16, 32)


class LagStudyError(Exception):
    """Inputs that cannot be evaluated as asked."""


def bound_label(bound: int) -> str:
    """``0 -> 'C0'`` (the frozen NCC), any other bound -> ``'C1-<bound>'``."""
    return "C0" if int(bound) == 0 else f"C1-{int(bound)}"


def degrees_per_sample(fov_deg: float, n_samples: int) -> float:
    """Centre-rate azimuth per profile sample — the same small-angle convention as
    ``matchers.lag_samples_for_degrees``, which under-counts off axis and is conservative there."""
    if fov_deg <= 0 or n_samples <= 0:
        raise LagStudyError("fov_deg and n_samples must be positive")
    return float(fov_deg) / float(n_samples)


def nested_slices(bounds: Sequence[int], max_lag: int) -> dict:
    """``{bound: slice}`` into the ``[-max_lag, max_lag]`` grid. Every bound must fit inside it."""
    out = {}
    for b in bounds:
        b = int(b)
        if b < 0 or b > int(max_lag):
            raise LagStudyError(f"bound {b} is outside the scored grid [-{max_lag}, {max_lag}]")
        out[b] = slice(int(max_lag) - b, int(max_lag) + b + 1)
    return out


class NestedBoundBank:
    """A reference database scored once at ``max_lag``, read off at every nested bound.

    ``score(query)`` returns ``{bound: (scores, lags)}``, each entry equal to
    ``BoundedLagNccMatcher(max_lag_samples=bound, min_overlap_frac=...).match()`` — the winning lag
    exactly and the score to 1e-12 (float summation order only). ``score_full`` returns the whole
    ``(R, L)`` per-lag score matrix, which is what the lag-versus-distance analysis reads.
    """

    def __init__(self, reference_ids: Sequence[str], profiles: Sequence[np.ndarray],
                 bounds: Sequence[int] = DEFAULT_BOUNDS,
                 max_lag: int = DEFAULT_MAX_LAG_SAMPLES,
                 min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                 chunk_references: int = 48):
        if len(reference_ids) != len(profiles):
            raise LagStudyError("reference_ids and profiles differ in length")
        if not profiles:
            raise LagStudyError("an empty reference bank cannot rank anything")
        self.reference_ids = list(reference_ids)
        self.max_lag = int(max_lag)
        self.bounds = tuple(int(b) for b in bounds)
        self.slices = nested_slices(self.bounds, self.max_lag)
        self.lags = np.arange(-self.max_lag, self.max_lag + 1, dtype=np.float64)
        n = int(np.asarray(profiles[0]).size)
        self.n_samples = n
        self.min_overlap_frac = float(min_overlap_frac)
        self.floor = max(2, int(math.ceil(self.min_overlap_frac * n)))
        mats = []
        for p in profiles:
            p = np.asarray(p, dtype=np.float64)
            if p.ndim != 1 or p.size != n:
                raise LagStudyError(f"every profile must be 1-D of length {n}, got {p.shape}")
            mats.append(_shift_matrix(p, self.lags))
        self.n_references = len(profiles)
        self.chunk = max(1, int(chunk_references))
        self.S_chunks = [np.concatenate(mats[i:i + self.chunk], axis=0)
                         for i in range(0, self.n_references, self.chunk)]

    def score_full(self, query: np.ndarray) -> np.ndarray:
        """The ``(R, L)`` per-lag score matrix for one query, ``-inf`` where overlap < floor."""
        q = np.asarray(query, dtype=np.float64)
        if q.ndim != 1 or q.size != self.n_samples:
            raise LagStudyError(f"query must be 1-D of length {self.n_samples}, got {q.shape}")
        L = self.lags.size
        return np.concatenate([_masked_ncc(q, S, self.floor).reshape(-1, L) for S in self.S_chunks])

    def score(self, query: np.ndarray) -> dict:
        """``{bound: (scores (R,), lags (R,))}`` — every arm from the one pass."""
        full = self.score_full(query)
        out = {}
        for b, sl in self.slices.items():
            out[b] = winner_per_reference(full[:, sl], self.lags[sl])
        return out


def verify_bounds(bank: NestedBoundBank, query: np.ndarray, profiles: Sequence[np.ndarray],
                  indices: Optional[Sequence[int]] = None, tol: float = 1e-12) -> dict:
    """Cross-check the nested read-off against the real matcher objects.

    For each checked reference and each bound: the winning lag must agree **exactly** and the score
    to ``tol``. Bound 0 is additionally checked against :class:`FrozenNccMatcher`, the C0 object.
    Raises :class:`LagStudyError` on any disagreement — a divergence must stop the run, not appear
    in a table.
    """
    idx = list(range(min(len(profiles), 3))) if indices is None else [int(i) for i in indices]
    got = bank.score(query)
    worst_score, n_lag_mismatch, n_checked = 0.0, 0, 0
    for b in bank.bounds:
        m = BoundedLagNccMatcher(max_lag_samples=b, min_overlap_frac=bank.min_overlap_frac)
        s_arr, l_arr = got[b]
        for i in idx:
            ref = np.asarray(profiles[i], dtype=np.float64)
            r = m.match(np.asarray(query, dtype=np.float64), ref)
            d = abs(float(s_arr[i]) - r.score)
            worst_score = max(worst_score, d)
            n_checked += 1
            if d > tol:
                raise LagStudyError(
                    f"{bound_label(b)} score disagrees with BoundedLagNccMatcher at reference "
                    f"{bank.reference_ids[i]}: |{s_arr[i]!r} - {r.score!r}| = {d:g} > {tol:g}")
            if float(l_arr[i]) != float(r.shift):
                n_lag_mismatch += 1
                raise LagStudyError(
                    f"{bound_label(b)} winning lag disagrees at reference {bank.reference_ids[i]}: "
                    f"{l_arr[i]!r} vs {r.shift!r}")
            if b == 0:
                c0 = FrozenNccMatcher().match(np.asarray(query, dtype=np.float64), ref)
                d0 = abs(float(s_arr[i]) - c0.score)
                worst_score = max(worst_score, d0)
                if d0 > tol:
                    raise LagStudyError(
                        f"C0 disagrees with FrozenNccMatcher at reference {bank.reference_ids[i]}: "
                        f"{d0:g} > {tol:g}")
    return {"n_checked": n_checked, "max_abs_score_delta": worst_score,
            "n_lag_mismatch": n_lag_mismatch, "tol": tol}


# --------------------------------------------------------------------------------------------------
# positional resolution — the quantity INT actually re-anchors on
# --------------------------------------------------------------------------------------------------

def positional_fields(query_xy, query_h: Optional[float], ref_xy: np.ndarray,
                      ref_h: Optional[np.ndarray], outcome: dict, ref_index: dict,
                      catastrophic_m: float) -> dict:
    """The INT-facing half of a retrieval outcome, added beside ``recog.retrieval_outcome``.

    ``outcome`` is that function's return value; ``ref_index`` maps reference id -> row of
    ``ref_xy``. Nothing here re-ranks or re-scores — it only reads the selection back out in the
    units of the downstream pose action.
    """
    out = {}
    t1, near = outcome.get("top1_id"), outcome.get("nearest_id")
    d1, dn = outcome.get("top1_distance_m"), outcome.get("d_near_m")
    s1, sn = outcome.get("top1_score"), outcome.get("nearest_score")
    out["selected_distance_m"] = d1
    out["nearest_distance_m"] = dn
    #: how much further away the matcher's pick is than the best available pick
    out["selection_excess_m"] = (d1 - dn) if (d1 is not None and dn is not None) else None
    #: what the appearance score buys for that excess: >0 means the wrong-position reference
    #: genuinely looked better, which is the mechanism INT is exposed to
    out["top1_minus_nearest_score"] = (s1 - sn) if (s1 is not None and sn is not None
                                                    and np.isfinite(s1) and np.isfinite(sn)) else None
    out["nearest_rank"] = outcome.get("exact_rank")
    out["nearest_is_top1"] = outcome.get("exact_top1")
    out["catastrophic"] = bool(d1 is not None and d1 > catastrophic_m)
    if query_h is not None and ref_h is not None:
        ref_h = np.asarray(ref_h, dtype=np.float64)
        for key, rid in (("selected", t1), ("nearest", near)):
            i = ref_index.get(rid)
            out[f"{key}_dh_m"] = (float(query_h) - float(ref_h[i])) if i is not None else None
    return out


def eligibility(n_refs: int, self_index: Optional[int] = None,
                exclude_indices: Sequence[int] = (), keep: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean keep-mask over the *saved* score matrix — every memory in this study is a mask over
    one scoring pass, exactly as in EXP-SKY-010/011, so all arms see identical numbers."""
    m = np.ones(int(n_refs), dtype=bool) if keep is None else np.asarray(keep, dtype=bool).copy()
    if self_index is not None:
        m[int(self_index)] = False
    for i in exclude_indices:
        m[int(i)] = False
    return m


def recency_mask(query_order: int, reference_orders: np.ndarray, k: int) -> np.ndarray:
    """INT-style recent-reference exclusion, emulated as a **declared sweep**: a reference is
    ineligible when it was captured within ``k`` positions along the trajectory of the query.

    INT's own policy is not visible from the SKY lane (no ``DISCRETE_REFERENCE`` /
    ``recent_reference`` / ``position_radius`` symbol exists in this worktree), so this is declared,
    never presented as a reproduction of INT's rule. ``k = 0`` removes only the query itself.
    """
    if k < 0:
        raise LagStudyError("k must be >= 0")
    orders = np.asarray(reference_orders)
    return np.abs(orders - int(query_order)) > int(k)


def quantiles(values: Sequence[float], qs: Sequence[float] = (50, 90, 95, 99)) -> dict:
    """Percentiles of a distance distribution, ``None`` when empty — the report never prints a mean
    in place of a distribution (§6 of the reopening brief)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {f"p{int(q)}": None for q in qs} | {"n": 0, "mean": None, "max": None}
    out = {f"p{int(q)}": float(np.percentile(v, q)) for q in qs}
    out.update({"n": int(v.size), "mean": float(v.mean()), "max": float(v.max())})
    return out


def binned(values: Sequence[float], by: Sequence[float], edges: Sequence[float],
           qs: Sequence[float] = (10, 25, 50, 75, 90)) -> list:
    """Distribution of ``values`` inside bins of ``by`` — the score-versus-distance table."""
    v = np.asarray(values, dtype=np.float64)
    b = np.asarray(by, dtype=np.float64)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (b >= lo) & (b < hi) & np.isfinite(v)
        sel = v[m]
        row = {"lo_m": float(lo), "hi_m": float(hi), "n": int(sel.size)}
        row.update({f"p{int(q)}": (float(np.percentile(sel, q)) if sel.size else None) for q in qs})
        row["max"] = float(sel.max()) if sel.size else None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------
# altitude — what the profile stage throws away
# --------------------------------------------------------------------------------------------------

def mean_elevation(curve_rows: np.ndarray, image_height_px: int) -> float:
    """The scalar the frozen profile stage discards.

    ``profile.normalize`` computes ``1 - row/H`` and then subtracts the mean, so the mean elevation
    never reaches the matcher. It is the component a pure altitude change shifts, which is why
    EXP-SKY-013 H6 tests it as an **auxiliary scalar** rather than by removing the mean subtraction
    (removing it would also let illumination and extraction bias walk straight into the score).
    """
    rows = np.asarray(curve_rows, dtype=np.float64)
    if rows.ndim != 1 or rows.size == 0:
        raise LagStudyError("curve_rows must be a non-empty 1-D array of image rows")
    if image_height_px <= 0:
        raise LagStudyError("image_height_px must be positive")
    return float(np.mean(1.0 - rows / float(image_height_px)))


def elevation_offset_deg(mean_q: float, mean_r: float, vfov_deg: float) -> float:
    """Mean-elevation difference expressed as an angle, via the same centre-rate convention used
    for lag. A first-order reading only: off-axis the true rate is ``sec^2`` larger."""
    return float((mean_q - mean_r) * float(vfov_deg))


def implied_range_m(delta_h_m: float, delta_elev_deg: float) -> Optional[float]:
    """The range a mean-elevation shift would imply for a pure altitude change, ``D ~ dh / dtheta``.

    Present so the altitude analysis can state its own refutation quantitatively: if the implied
    range is not stable across a site, then mean elevation is not a clean altitude channel and
    **altitude alone is insufficient** — the terrain-range term is required, which is the deferred
    EXP-SKY-009 path and is explicitly out of scope for a deployable matcher.
    """
    if delta_elev_deg is None or abs(delta_elev_deg) < 1e-9:
        return None
    return float(delta_h_m / math.radians(delta_elev_deg))
