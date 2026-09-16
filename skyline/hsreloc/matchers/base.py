"""The common interface for candidate profile matchers (Part F; LIT-SKY-005).

Why this package exists — and what it is *not*
----------------------------------------------
``EXP-SKY-006`` ended in Outcome C: mean-removed pointwise NCC is too rigid to recognise a place from
a viewpoint a few metres away at street level. ``LIT-SKY-005`` derived why, and ranked the smallest
extensions that could plausibly help under a **known heading**. This package implements that ladder so
that a later experiment can introduce **one transformation at a time**.

It does **not** replace anything. ``hsreloc/retrieval/`` — the frozen chain that produced every
existing SKY result — is imported here and never edited; ``C0`` *is* that chain's scorer, called
directly rather than reimplemented, so "the baseline" cannot drift away from the baseline. A variant
becomes a result only through its own pre-registration, and never by being switched on inside a run
that was registered for the baseline.

One score scale across the ladder
---------------------------------
Every variant is an **alignment** followed by the *same* NCC primitive. C1 chooses a shift, C3 a shift
and a scale, C4 a banded monotone warp; each then scores the aligned pair with
``hsreloc.retrieval.baselines.ncc``. The consequence is that C0…C4 scores are directly comparable and
that any difference between them is attributable to the alignment freedom alone, which is the only
variable the experiment is allowed to change.

Overlap discipline
------------------
Alignment creates non-overlapping samples, and a similarity maximised over ever-smaller windows
converges to 1.0 on noise. Every variant therefore:

* compares only the overlapping samples, re-normalised on that window;
* refuses any alignment whose overlap is below ``min_overlap_frac`` — a refused alignment is not
  scored, so a tiny-overlap match cannot win;
* returns ``accepted=False`` with ``score = -inf`` if *no* alignment in the search space clears the
  floor, so the candidate loses rather than being silently promoted.

Determinism
-----------
Ties are broken by ``(-score, |shift|, shift, |log scale|, scale)``: the least-transformed alignment
wins, and equal-magnitude opposite shifts resolve to the negative one. Two runs of the same
configuration return the same winning parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from hsreloc.retrieval.baselines import ncc as frozen_ncc

MATCHERS_VERSION = "1.0.0"

#: Default floor on the fraction of query samples an alignment must still compare.
DEFAULT_MIN_OVERLAP_FRAC = 0.6

NO_ALIGNMENT_SCORE = -math.inf


class MatcherError(Exception):
    """A matcher variant is misconfigured, or was asked for something it cannot do."""


@dataclass(frozen=True)
class MatchResult:
    """One query–reference comparison, with every transform parameter reported.

    ``score`` is on the NCC scale in ``[-1, 1]`` (or ``-inf`` when no alignment cleared the overlap
    floor). ``shift`` is in profile samples, positive meaning the reference is sampled further right;
    ``scale`` is the horizontal scale applied to the query's axis; ``warp_magnitude`` is the mean
    absolute departure from the identity alignment, in samples — 0 for C0/C1 by construction.
    """

    variant: str
    score: float
    shift: float = 0.0
    scale: float = 1.0
    overlap: int = 0
    overlap_frac: float = 1.0
    warp_magnitude: float = 0.0
    accepted: bool = True
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"variant": self.variant, "score": self.score, "shift": self.shift, "scale": self.scale,
                "overlap": self.overlap, "overlap_frac": self.overlap_frac,
                "warp_magnitude": self.warp_magnitude, "accepted": self.accepted,
                **{f"diag_{k}": v for k, v in self.diagnostics.items()}}


@runtime_checkable
class ProfileMatcher(Protocol):
    """What every variant provides. ``score`` is the drop-in ``scorer(q, r) -> float`` the unchanged
    ``hsreloc.retrieval.rank.rank_candidates`` already accepts, so a variant needs no ranker change."""

    variant: str

    def match(self, query: np.ndarray, reference: np.ndarray) -> MatchResult: ...

    def score(self, query: np.ndarray, reference: np.ndarray) -> float: ...

    def describe(self) -> dict: ...


class BaseMatcher:
    """Shared scoring, overlap accounting and tie-breaking."""

    variant = "base"

    def __init__(self, min_overlap_frac: float = DEFAULT_MIN_OVERLAP_FRAC) -> None:
        if not 0.0 < float(min_overlap_frac) <= 1.0:
            raise MatcherError(f"min_overlap_frac must be in (0, 1], got {min_overlap_frac}")
        self.min_overlap_frac = float(min_overlap_frac)

    # -- scoring -----------------------------------------------------------------------------
    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        """The frozen primitive, on whatever window the alignment produced.

        ``profile_distance`` removes each window's own mean before correlating, so a partial window is
        compared on its own terms rather than inheriting the full profile's offset. With a full window
        and a zero-mean profile this is bit-for-bit the C0 score.
        """
        return frozen_ncc(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))

    def _refused(self, reason: str, searched: int) -> MatchResult:
        return MatchResult(variant=self.variant, score=NO_ALIGNMENT_SCORE, overlap=0, overlap_frac=0.0,
                           accepted=False, diagnostics={"reason": reason, "n_alignments_searched": searched})

    @staticmethod
    def _check(query, reference) -> tuple:
        q = np.asarray(query, dtype=np.float64)
        r = np.asarray(reference, dtype=np.float64)
        if q.ndim != 1 or r.ndim != 1:
            raise MatcherError(f"profiles must be 1-D, got {q.shape} and {r.shape}")
        if q.size != r.size:
            raise MatcherError(
                f"profiles must be the same length ({q.size} vs {r.size}); resampling to a common "
                f"n_samples is the profile stage's job, not the matcher's")
        if q.size < 2:
            raise MatcherError("profiles must have at least 2 samples")
        return q, r

    @staticmethod
    def _best(candidates: list):
        """Least-transformed alignment among the highest-scoring ones (documented tie-break)."""
        return min(candidates, key=lambda c: (-c.score, abs(c.shift), c.shift,
                                              abs(math.log(c.scale)) if c.scale > 0 else math.inf,
                                              c.scale, c.warp_magnitude))

    # -- interface ---------------------------------------------------------------------------
    def match(self, query, reference) -> MatchResult:      # pragma: no cover - abstract
        raise NotImplementedError

    def score(self, query, reference) -> float:
        return self.match(query, reference).score

    def __call__(self, query, reference) -> float:
        return self.score(query, reference)

    def describe(self) -> dict:
        return {"variant": self.variant, "matchers_version": MATCHERS_VERSION,
                "min_overlap_frac": self.min_overlap_frac,
                "scoring_primitive": "hsreloc.retrieval.baselines.ncc (frozen, unmodified)"}
