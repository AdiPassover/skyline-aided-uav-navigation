"""C4 — constrained local warp (banded DTW), designed and isolated, not promoted (LIT-SKY-005 C7).

This is the *last* rung of the ladder and it is here to be ready, not to be used. ``LIT-SKY-005``
places it behind bounded lag, angular units, lag+scale and multiscale for a specific reason: a warp
that can stretch parts of the profile independently can also make two unrelated skylines agree, and on
a small reference database that shows up as confident false matches rather than as improved recall.

**The gate is empirical and pre-declared**: run it on real data only if the simulator's *ground-truth*
curve deformation shows that a shift and a scale are not enough — i.e. only if the measured
GT-vs-GT disagreement at a given translation cannot be removed by C3's two parameters. Until then it
is synthetic-test-only, and nothing in ``simret`` selects it by default.

Every requirement the brief placed on a DTW candidate is enforced here rather than documented:

* **left-to-right order is preserved** — the step set is ``(i-1, j-1)``, ``(i-1, j)``, ``(i, j-1)``,
  all monotone, so an alignment can never cross itself;
* **a narrow Sakoe–Chiba band** — ``|i - j| ≤ band``, declared, with the band being the whole point:
  an unbanded DTW is not a candidate here at all;
* **excessive warping is penalised** — each non-diagonal step costs ``step_penalty``, so a path buys
  flexibility rather than getting it free;
* **total warp magnitude is exposed** — mean ``|i - j|`` along the path, plus the non-diagonal step
  count and the path length;
* **pathological alignments are rejected** — a path whose mean warp exceeds ``max_mean_warp`` returns
  ``accepted=False`` with ``score = -inf``, so it loses instead of winning oddly;
* **the score stays on the family's scale** — the reference is resampled onto the query's index
  through the path, and the pair is then scored with the same frozen NCC primitive as C0.
"""

from __future__ import annotations

import math

import numpy as np

from hsreloc.matchers.base import BaseMatcher, MatcherError, MatchResult

C4 = "c4_banded_dtw_ncc"

DEFAULT_BAND = 8
DEFAULT_STEP_PENALTY = 0.05
DEFAULT_MAX_MEAN_WARP = 4.0


def banded_dtw(query: np.ndarray, reference: np.ndarray, band: int,
               step_penalty: float = DEFAULT_STEP_PENALTY) -> dict:
    """Monotone banded DTW. Returns the path, its cost, and warp statistics.

    Cost is ``|q_i - r_j|`` plus ``step_penalty`` on every non-diagonal step. Cells outside the band
    are unreachable, which is what bounds the warp *by construction* rather than by penalty alone.
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    n, m = q.size, r.size
    band = int(band)
    if band < 0:
        raise MatcherError(f"band must be >= 0, got {band}")
    if band < abs(n - m):
        raise MatcherError(f"band {band} cannot span the length difference |{n} - {m}|")

    inf = math.inf
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    # 0 = diagonal, 1 = from (i-1, j), 2 = from (i, j-1)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            local = abs(q[i - 1] - r[j - 1])
            diag = cost[i - 1, j - 1]
            up = cost[i - 1, j] + step_penalty
            left = cost[i, j - 1] + step_penalty
            best, arg = diag, 0
            if up < best:
                best, arg = up, 1
            if left < best:
                best, arg = left, 2
            cost[i, j] = local + best
            back[i, j] = arg
    total = cost[n, m]
    if not math.isfinite(total):
        raise MatcherError(f"no monotone path fits inside band {band} for lengths {n}, {m}")

    path, i, j, n_non_diagonal = [], n, m, 0
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = int(back[i, j])
        if move == 0:
            i, j = i - 1, j - 1
        elif move == 1:
            i -= 1
            n_non_diagonal += 1
        else:
            j -= 1
            n_non_diagonal += 1
    path.reverse()
    pairs = np.array(path, dtype=np.int64)
    offsets = np.abs(pairs[:, 0] - pairs[:, 1])
    return {"cost": float(total), "path": pairs,
            "mean_warp": float(offsets.mean()), "max_warp": int(offsets.max()),
            "path_length": int(pairs.shape[0]),
            "n_non_diagonal": int(n_non_diagonal),
            "normalised_cost": float(total / max(1, pairs.shape[0]))}


def align_by_path(reference: np.ndarray, path: np.ndarray, n_query: int) -> np.ndarray:
    """Reference resampled onto the query's index: the mean of every reference sample matched to ``i``."""
    r = np.asarray(reference, dtype=np.float64)
    out = np.zeros(n_query, dtype=np.float64)
    count = np.zeros(n_query, dtype=np.float64)
    for i, j in path:
        out[i] += r[j]
        count[i] += 1.0
    if np.any(count == 0):                       # unreachable for a monotone full path, guarded anyway
        raise MatcherError("DTW path does not cover every query sample")
    return out / count


class ConstrainedDtwMatcher(BaseMatcher):
    """C4. Banded, penalised, warp-reporting, and refusing pathological alignments."""

    variant = C4

    def __init__(self, band: int = DEFAULT_BAND, step_penalty: float = DEFAULT_STEP_PENALTY,
                 max_mean_warp: float = DEFAULT_MAX_MEAN_WARP, min_overlap_frac: float = 1.0) -> None:
        super().__init__(min_overlap_frac)
        if int(band) < 0:
            raise MatcherError(f"band must be >= 0, got {band}")
        if float(step_penalty) < 0:
            raise MatcherError(f"step_penalty must be >= 0, got {step_penalty}")
        self.band = int(band)
        self.step_penalty = float(step_penalty)
        self.max_mean_warp = float(max_mean_warp)

    def match(self, query, reference) -> MatchResult:
        q, r = self._check(query, reference)
        dtw = banded_dtw(q, r, self.band, self.step_penalty)
        aligned = align_by_path(r, dtw["path"], q.size)
        if dtw["mean_warp"] > self.max_mean_warp:
            return MatchResult(
                variant=self.variant, score=-math.inf, shift=0.0, scale=1.0, overlap=int(q.size),
                overlap_frac=1.0, warp_magnitude=dtw["mean_warp"], accepted=False,
                diagnostics={"reason": f"mean warp {dtw['mean_warp']:.3f} samples exceeds the declared "
                                       f"maximum {self.max_mean_warp}", **_diag(dtw)})
        # A DTW path has no single shift, but its mean signed offset is the closest honest summary.
        pairs = dtw["path"]
        mean_signed = float(np.mean(pairs[:, 1] - pairs[:, 0]))
        return MatchResult(variant=self.variant, score=self._ncc(q, aligned), shift=mean_signed,
                           scale=1.0, overlap=int(q.size), overlap_frac=1.0,
                           warp_magnitude=dtw["mean_warp"], accepted=True, diagnostics=_diag(dtw))

    def describe(self) -> dict:
        return {**super().describe(), "band": self.band, "step_penalty": self.step_penalty,
                "max_mean_warp": self.max_mean_warp,
                "step_set": "(i-1,j-1), (i-1,j), (i,j-1) — monotone, left-to-right order preserved",
                "search_space": "banded monotone local warp",
                "status": ("DESIGNED AND SYNTHETIC-TESTED ONLY. Not to be run on real data unless the "
                           "simulator's GT curve deformation shows shift+scale (C3) is insufficient "
                           "(LIT-SKY-005 C7; PROT-SKY-001 §5 pre-registration)")}


def _diag(dtw: dict) -> dict:
    return {"dtw_cost": dtw["cost"], "dtw_normalised_cost": dtw["normalised_cost"],
            "mean_warp_samples": dtw["mean_warp"], "max_warp_samples": dtw["max_warp"],
            "path_length": dtw["path_length"], "n_non_diagonal_steps": dtw["n_non_diagonal"]}
