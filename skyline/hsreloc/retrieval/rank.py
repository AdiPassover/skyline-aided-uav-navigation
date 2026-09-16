"""Score a query against every reference and order the candidates.

No shortlisting heuristic, no index, no early termination: 16 references is a linear scan, and an
approximate index here would add a second free variable to an experiment whose entire purpose is to
isolate the representation.

**No lag/shift search** (research **R10**). A sliding correlation would implicitly estimate a heading
difference this dataset declares unsupported, and maximising a similarity over lags inflates scores
exactly for self-similar signals -- which is the failure mode being measured. ``lag_search`` exists in
the run config, defaults to false, and an enabled run is reportable only as an explicitly labelled
secondary comparison.

Ties break deterministically by ``reference_id`` so a run is reproducible (SC-009) and so a
pathological all-equal score population produces a stable, inspectable ordering rather than whatever
the sort happened to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankedCandidate:
    reference_id: str
    east_m: float
    north_m: float
    score: float


def rank_candidates(query_profile: np.ndarray, references: list, scorer) -> list:
    """Score ``query_profile`` against every reference; return candidates best-first.

    ``references`` is a list of ``refset.Reference``; ``scorer`` is a callable
    ``(query, reference) -> float`` where higher is better.
    """
    scored = [
        RankedCandidate(
            reference_id=ref.reference_id,
            east_m=ref.east_m,
            north_m=ref.north_m,
            score=float(scorer(query_profile, ref.profile)),
        )
        for ref in references
    ]
    return sort_candidates(scored)


def rank_by_identity(query_id: str, references: list, mock_scorer) -> list:
    """The mock path: score from ``(query_id, reference_id)`` alone, never from curve content."""
    scored = [
        RankedCandidate(
            reference_id=ref.reference_id,
            east_m=ref.east_m,
            north_m=ref.north_m,
            score=float(mock_scorer.score_pair(query_id, ref.reference_id)),
        )
        for ref in references
    ]
    return sort_candidates(scored)


def sort_candidates(candidates: list) -> list:
    """Best score first; ties broken by ``reference_id`` so the order is deterministic."""
    return sorted(candidates, key=lambda c: (-c.score, c.reference_id))


def shortlist(candidates: list, recall_k: int) -> list:
    """The top ``recall_k`` candidates, or all of them when the reference set is smaller."""
    if recall_k < 1:
        raise ValueError(f"recall_k must be >= 1, got {recall_k}")
    return candidates[:recall_k]
