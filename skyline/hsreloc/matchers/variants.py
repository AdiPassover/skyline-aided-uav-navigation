"""C0 frozen NCC, C1 bounded-lag NCC, C3 bounded shift + horizontal scale (LIT-SKY-005).

The conservative end of the ladder. Each variant adds exactly one degree of freedom to the alignment
and nothing else; none of them is tuned, and none of their bounds may be chosen from a retrieval
number (Principle V — the bound comes from heading uncertainty and the camera's own geometry, and
Stage 2 measures whether it was *sufficient*, which is a different question from what value maximises
recall).

C0 — the frozen baseline
    ``hsreloc.retrieval.baselines.ncc`` called directly. Not reimplemented, not re-derived: the same
    function object the ``EXP-SKY-006`` runs used. A test asserts bit equality.

C1 — bounded lag
    ``max_{|δ| ≤ δ_max} NCC(q(x), r(x + δ))``. Under a known heading, a residual yaw error is the one
    distortion that is *uniform* across the profile (``LIT-SKY-005`` §1), so a bounded shift is the
    smallest extension that can absorb it. ``δ_max`` is declared in samples or in degrees — the degree
    form is the honest one, because it converts through the camera's own FOV and therefore says what
    heading uncertainty is being tolerated rather than a bare number of pixels.

C3 — bounded shift and horizontal scale
    ``x' = s·x + δ``. First-order, along-track translation towards distant structure scales the
    profile horizontally by ``1 + t/D`` while lateral translation shifts it (``LIT-SKY-005`` §1); in
    the **far field** the depth-dependent warp collapses to exactly this pair. Whether the simulator's
    skyline is in that regime is measurable from the per-column range, and this variant is the one to
    reach for only if it is.

Ordering within the ladder is deliberate: C1 before C3, because C3 contains C1 (``s = 1``) and a
freedom that is never needed is a freedom that can only add false matches.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from hsreloc.matchers.base import (DEFAULT_MIN_OVERLAP_FRAC, BaseMatcher, MatcherError, MatchResult)

C0 = "c0_frozen_ncc"
C1 = "c1_bounded_lag_ncc"
C3 = "c3_shift_scale_ncc"


class FrozenNccMatcher(BaseMatcher):
    """C0. The ``EXP-SKY-006`` scorer, wearing the family's interface and nothing more."""

    variant = C0

    def __init__(self) -> None:
        super().__init__(min_overlap_frac=1.0)

    def match(self, query, reference) -> MatchResult:
        q, r = self._check(query, reference)
        return MatchResult(variant=self.variant, score=self._ncc(q, r), shift=0.0, scale=1.0,
                           overlap=int(q.size), overlap_frac=1.0, warp_magnitude=0.0,
                           diagnostics={"search_space": 1})

    def describe(self) -> dict:
        return {**super().describe(), "search_space": "none — pointwise, no lag, no scale, no warp",
                "equivalence": "delegates to hsreloc.retrieval.baselines.ncc; byte-equivalent to the "
                               "EXP-SKY-006 baseline"}


def lag_samples_for_degrees(max_lag_deg: float, fov_deg: float, n_samples: int) -> int:
    """Convert a heading-uncertainty budget into a lag bound in profile samples.

    Small-angle, uniform-column approximation: over an ``fov_deg`` field sampled into ``n_samples``
    columns, one sample spans ``fov_deg / n_samples`` degrees near the optical axis. It is deliberately
    the *centre* rate — ``dx/dα = f·sec²α`` grows towards the edges, so this under-counts samples away
    from the axis and the bound is conservative there. The exact mapping is what the C2 angular
    representation exists to provide; this is the pixel-domain approximation and says so.
    """
    if fov_deg <= 0 or n_samples < 2:
        raise MatcherError(f"fov_deg must be positive and n_samples >= 2, got {fov_deg}, {n_samples}")
    return int(math.ceil(abs(float(max_lag_deg)) * float(n_samples) / float(fov_deg)))


class BoundedLagNccMatcher(BaseMatcher):
    """C1. Integer or sub-sample horizontal shift within a declared bound.

    ``max_lag_samples`` is the bound; ``subsample_step`` below 1.0 enables sub-sample lags by linear
    interpolation of the reference (reported as a fractional ``shift``). The search is exhaustive over
    the bound — there is no coarse-to-fine heuristic, because at these sizes it is cheap and a
    heuristic would be a second free variable.
    """

    variant = C1

    def __init__(self, max_lag_samples: int = 0, min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                 subsample_step: float = 1.0, max_lag_deg=None, fov_deg=None, n_samples=None) -> None:
        super().__init__(min_overlap_frac)
        if max_lag_deg is not None:
            if fov_deg is None or n_samples is None:
                raise MatcherError("max_lag_deg needs fov_deg and n_samples to convert into samples")
            max_lag_samples = lag_samples_for_degrees(max_lag_deg, fov_deg, n_samples)
        if int(max_lag_samples) < 0:
            raise MatcherError(f"max_lag_samples must be >= 0, got {max_lag_samples}")
        if not 0.0 < float(subsample_step) <= 1.0:
            raise MatcherError(f"subsample_step must be in (0, 1], got {subsample_step}")
        self.max_lag_samples = int(max_lag_samples)
        self.subsample_step = float(subsample_step)
        self.max_lag_deg = None if max_lag_deg is None else float(max_lag_deg)
        self.fov_deg = None if fov_deg is None else float(fov_deg)

    def lags(self) -> np.ndarray:
        if self.subsample_step >= 1.0:
            return np.arange(-self.max_lag_samples, self.max_lag_samples + 1, dtype=np.float64)
        n = int(round(self.max_lag_samples / self.subsample_step))
        return np.round(np.arange(-n, n + 1) * self.subsample_step, 9)

    def match(self, query, reference) -> MatchResult:
        q, r = self._check(query, reference)
        n = q.size
        floor = max(2, int(math.ceil(self.min_overlap_frac * n)))
        idx = np.arange(n, dtype=np.float64)
        candidates, searched = [], 0
        for lag in self.lags():
            lo = int(math.ceil(max(0.0, -lag)))
            hi = int(math.floor(min(n - 1, n - 1 - lag)))
            overlap = hi - lo + 1
            if overlap < floor:
                continue
            searched += 1
            qs = q[lo:hi + 1]
            if float(lag).is_integer():
                rs = r[lo + int(lag):hi + int(lag) + 1]
            else:
                rs = np.interp(idx[lo:hi + 1] + lag, idx, r)
            candidates.append(MatchResult(
                variant=self.variant, score=self._ncc(qs, rs), shift=float(lag), scale=1.0,
                overlap=int(overlap), overlap_frac=overlap / n, warp_magnitude=0.0,
                diagnostics={"n_alignments_searched": 0}))
        if not candidates:
            return self._refused(f"no lag in [-{self.max_lag_samples}, {self.max_lag_samples}] keeps "
                                 f"overlap >= {self.min_overlap_frac:.2f}", searched)
        best = self._best(candidates)
        return replace(best, diagnostics={
            "n_alignments_searched": searched, "max_lag_samples": self.max_lag_samples,
            "max_lag_deg": self.max_lag_deg,
            "score_at_zero_lag": next((c.score for c in candidates if c.shift == 0.0), None)})

    def describe(self) -> dict:
        return {**super().describe(), "max_lag_samples": self.max_lag_samples,
                "max_lag_deg": self.max_lag_deg, "fov_deg": self.fov_deg,
                "subsample_step": self.subsample_step,
                "search_space": "horizontal shift only; scale fixed at 1, no warp",
                "bound_provenance": ("declared from heading uncertainty and camera FOV — NEVER chosen "
                                     "from retrieval correctness (LIT-SKY-005 C1; PROT-SKY-001 §3.2)")}


class ShiftScaleNccMatcher(BaseMatcher):
    """C3. ``x' = s·x + δ`` over declared, bounded grids of shift and scale.

    The reference is resampled by explicit linear interpolation at the mapped positions; samples that
    map outside the reference are outside the overlap and are not compared. Both winning parameters
    and the overlap are reported, so a result can always be read as "it needed this much transform".
    """

    variant = C3

    def __init__(self, max_lag_samples: int = 0, scales=(1.0,),
                 min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC,
                 subsample_step: float = 1.0) -> None:
        super().__init__(min_overlap_frac)
        if int(max_lag_samples) < 0:
            raise MatcherError(f"max_lag_samples must be >= 0, got {max_lag_samples}")
        scales = [float(s) for s in scales]
        if not scales or any(s <= 0 for s in scales):
            raise MatcherError(f"scales must be a non-empty list of positive numbers, got {scales}")
        if not 0.0 < float(subsample_step) <= 1.0:
            raise MatcherError(f"subsample_step must be in (0, 1], got {subsample_step}")
        self.max_lag_samples = int(max_lag_samples)
        self.scales = scales
        self.subsample_step = float(subsample_step)

    @staticmethod
    def scale_grid(max_scale_deviation: float, n_steps: int) -> list:
        """A symmetric, log-spaced scale grid — declared, not searched adaptively."""
        if max_scale_deviation < 0 or n_steps < 1:
            raise MatcherError("max_scale_deviation must be >= 0 and n_steps >= 1")
        if max_scale_deviation == 0 or n_steps == 1:
            return [1.0]
        hi = math.log(1.0 + float(max_scale_deviation))
        return [float(math.exp(v)) for v in np.linspace(-hi, hi, int(n_steps))]

    def lags(self) -> np.ndarray:
        if self.subsample_step >= 1.0:
            return np.arange(-self.max_lag_samples, self.max_lag_samples + 1, dtype=np.float64)
        n = int(round(self.max_lag_samples / self.subsample_step))
        return np.round(np.arange(-n, n + 1) * self.subsample_step, 9)

    def match(self, query, reference) -> MatchResult:
        q, r = self._check(query, reference)
        n = q.size
        floor = max(2, int(math.ceil(self.min_overlap_frac * n)))
        idx = np.arange(n, dtype=np.float64)
        centre = (n - 1) / 2.0                      # scale about the profile centre, not column 0
        candidates, searched = [], 0
        for scale in self.scales:
            mapped = centre + scale * (idx - centre)
            for lag in self.lags():
                pos = mapped + lag
                inside = (pos >= 0.0) & (pos <= n - 1)
                overlap = int(inside.sum())
                if overlap < floor:
                    continue
                searched += 1
                rs = np.interp(pos[inside], idx, r)
                candidates.append(MatchResult(
                    variant=self.variant, score=self._ncc(q[inside], rs), shift=float(lag),
                    scale=float(scale), overlap=overlap, overlap_frac=overlap / n,
                    warp_magnitude=float(np.mean(np.abs(pos[inside] - idx[inside]))),
                    diagnostics={}))
        if not candidates:
            return self._refused(
                f"no (shift, scale) in the declared grid keeps overlap >= {self.min_overlap_frac:.2f}",
                searched)
        best = self._best(candidates)
        return replace(best, diagnostics={
            "n_alignments_searched": searched, "max_lag_samples": self.max_lag_samples,
            "n_scales": len(self.scales), "scale_min": min(self.scales), "scale_max": max(self.scales),
            "score_at_identity": next((c.score for c in candidates
                                       if c.shift == 0.0 and c.scale == 1.0), None)})

    def describe(self) -> dict:
        return {**super().describe(), "max_lag_samples": self.max_lag_samples,
                "scales": self.scales, "subsample_step": self.subsample_step,
                "scale_anchor": "profile centre",
                "search_space": "horizontal shift x scale; no local warp",
                "bound_provenance": ("declared from the far-field first-order model (LIT-SKY-005 §1: "
                                     "scale ~ 1 + t/D); NEVER chosen from retrieval correctness")}
