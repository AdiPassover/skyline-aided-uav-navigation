"""Confidence and the accept/refuse decision -- derived from score-distribution shape alone.

Research: ``research.md`` **R7**. Spec FR-015..FR-018.

**The binding rule (FR-017): nothing here may see correctness.** No function in this module takes
ground truth, a position error, or an "is this right" signal, and none may ever be added. Confidence
answers *"does one reference stand out from the others?"*, which is a property of the score
population; it does not answer *"is the winner correct?"*, which is the evaluator's question. Keeping
those apart is what stops a threshold from being quietly tuned until the numbers look good.

The derivation is reused verbatim from ``EXP-009``'s T048 work, including the failure it already
found and fixed: standardising over *all* scores was degenerate when most references scored zero
(median = MAD = 0, so every confidence saturated at 1.0 and every margin collapsed). The fix -- a
null taken over the **competing** scores plus a bounded, non-saturating map -- is a
distribution-shape correction, not an outcome-driven one.

    null    = every score except the best
    med     = median(null)
    sigma   = 1.4826 * MAD(null),  falling back to (P84 - med), then to std(null)
    z_i     = (score_i - med) / sigma
    conf    = max(z_best, 0) / (max(z_best, 0) + k)          k = 5, fixed a priori

``k = 5`` is chosen so ``conf >= 0.5`` exactly when ``z_best >= 5 sigma``, which preserves the meaning
the repository's existing 0.5 reject threshold already carries. ``conf`` is bounded in ``[0, 1)`` and
approaches 1 only asymptotically, so it cannot saturate the way the pre-fix version did.

Outcome, evaluated in this order
--------------------------------
=========================================  ====================  ========
condition                                  outcome               position
=========================================  ====================  ========
query profile is degenerate                ``EXTRACTION_FAILURE``  none
score population has no spread             ``REJECTED``            none
``conf < reject_threshold``                ``REJECTED``            none
``n_viable >= 2`` and margin ``< 1 sigma``  ``AMBIGUOUS``           none
otherwise                                  ``SUCCESS``             matched reference
=========================================  ====================  ========

*Implementation note, 2026-08-23.* ``plan.md``'s table folded "score population has no spread" into
``EXTRACTION_FAILURE`` alongside a degenerate query. That is wrong in a way worth fixing rather than
reproducing: a flat *query curve* is an input-quality failure, but references that are mutually
indistinguishable are a *reference-set* property with a perfectly good query. Labelling the second as
an extraction failure would misattribute it in the evaluator's own outcome counts -- precisely the
kind of misattribution this feature exists to prevent. It is a refusal, so it is ``REJECTED``. No
threshold changed.

**``OUT_OF_COVERAGE`` is never produced, deliberately.** Emitting it requires evidence that the query
lies outside the database's coverage, and the only such evidence available here is ``in_coverage`` --
which is ground truth. Reading it would be leakage. An out-of-database query must therefore surface
as a low-confidence ``REJECTED``, and *whether it does* is exactly the acceptance result Stage 2
measures. The outcome stays in the record vocabulary for a future variant carrying a genuine position
prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SUCCESS = "SUCCESS"
REJECTED = "REJECTED"
AMBIGUOUS = "AMBIGUOUS"
EXTRACTION_FAILURE = "EXTRACTION_FAILURE"

#: Frozen a-priori constants (research R7). Changing one makes a *new* run with its own
#: pre-registration; it never retro-edits an existing one.
DEFAULT_CONFIDENCE_K = 5.0
DEFAULT_REJECT_THRESHOLD = 0.5
DEFAULT_VIABLE_SIGMA = 3.0
DEFAULT_AMBIGUITY_MARGIN_SIGMA = 1.0
DEFAULT_DEGENERATE_PROFILE_STD = 1e-6

_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class AcceptanceConfig:
    confidence_k: float = DEFAULT_CONFIDENCE_K
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD
    viable_sigma: float = DEFAULT_VIABLE_SIGMA
    ambiguity_margin_sigma: float = DEFAULT_AMBIGUITY_MARGIN_SIGMA
    degenerate_profile_std: float = DEFAULT_DEGENERATE_PROFILE_STD

    def as_dict(self) -> dict:
        return {
            "confidence_k": self.confidence_k,
            "reject_threshold": self.reject_threshold,
            "viable_sigma": self.viable_sigma,
            "ambiguity_margin_sigma": self.ambiguity_margin_sigma,
            "degenerate_profile_std": self.degenerate_profile_std,
        }


@dataclass(frozen=True)
class AcceptanceDecision:
    outcome: str
    confidence: Optional[float]
    best_score: Optional[float]
    second_best_score: Optional[float]
    score_margin: Optional[float]
    n_viable_candidates: Optional[int]
    sigma: Optional[float]
    z_best: Optional[float]

    @property
    def is_success(self) -> bool:
        return self.outcome == SUCCESS


def robust_sigma(null_scores: np.ndarray, med: float) -> float:
    """Robust spread of the competing scores, with the two documented fallbacks.

    Returns 0.0 when the population genuinely has no spread -- the caller turns that into a refusal
    rather than dividing by it.
    """
    null = np.asarray(null_scores, dtype=np.float64)
    if null.size == 0:
        return 0.0
    mad = float(np.median(np.abs(null - med)))
    sigma = _MAD_TO_SIGMA * mad
    if sigma > 0.0:
        return sigma
    sigma = float(np.percentile(null, 84.0) - med)          # fallback 1: one-sided spread
    if sigma > 0.0:
        return sigma
    sigma = float(np.std(null))                             # fallback 2: plain std
    return sigma if sigma > 0.0 else 0.0


def confidence_from_z(z_best: float, k: float) -> float:
    """Bounded, non-saturating map from separation to confidence. ``conf >= 0.5`` iff ``z >= k``."""
    z = max(float(z_best), 0.0)
    return float(z / (z + float(k)))


def decide(scores: np.ndarray, config: AcceptanceConfig,
           query_profile_degenerate: bool = False) -> AcceptanceDecision:
    """Turn a ranked score population into a confidence and one outcome.

    ``scores`` must be ordered best-first (``rank.sort_candidates``). It carries **no** identity and
    **no** ground truth -- by construction this function cannot consult correctness.
    """
    if query_profile_degenerate:
        return AcceptanceDecision(EXTRACTION_FAILURE, None, None, None, None, None, None, None)

    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return AcceptanceDecision(REJECTED, 0.0, None, None, None, 0, None, None)

    best = float(s[0])
    second = float(s[1]) if s.size > 1 else None
    null = s[1:]

    if null.size == 0:
        # A single reference: there is no population to stand out from, so no separation can be
        # demonstrated. Refuse rather than award confidence for being the only candidate.
        return AcceptanceDecision(REJECTED, 0.0, best, None, None, 0, None, None)

    med = float(np.median(null))
    sigma = robust_sigma(null, med)
    if sigma <= 0.0:
        # Every competing reference scores alike: the score population cannot discriminate. Not an
        # extraction failure -- the query may be perfectly good (see the module docstring).
        margin = None if second is None else best - second
        return AcceptanceDecision(REJECTED, 0.0, best, second, margin, 0, 0.0, None)

    z = (s - med) / sigma
    z_best = float(z[0])
    conf = confidence_from_z(z_best, config.confidence_k)
    n_viable = int(np.count_nonzero(z >= config.viable_sigma))
    margin = None if second is None else best - second

    if conf < config.reject_threshold:
        outcome = REJECTED
    elif (n_viable >= 2 and margin is not None
          and margin < config.ambiguity_margin_sigma * sigma):
        outcome = AMBIGUOUS
    else:
        outcome = SUCCESS

    return AcceptanceDecision(
        outcome=outcome,
        confidence=conf,
        best_score=best,
        second_best_score=second,
        score_margin=margin,
        n_viable_candidates=n_viable,
        sigma=sigma,
        z_best=z_best,
    )
