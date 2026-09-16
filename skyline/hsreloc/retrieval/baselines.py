"""The scorers: NCC, normalized L1, normalized L2, and a mock.

Research: ``research.md`` **R6**/**R9**. Only simple, published, literature-grounded 1-D skyline
baselines (``LIT-013`` Bucket A) -- no learned descriptor, no descriptor invented here (spec FR-007,
FR-008).

Every scorer is **higher-is-better** on its own raw scale. The evaluator never assumes a range: it
uses the raw values for score separation and aliasing, so ``ncc``'s ``[-1, 1]`` and ``l1``'s
unbounded negatives are equally acceptable and are never rescaled to look alike.

Why three and not one
---------------------
They **bracket the amplitude question** ``LIT-009`` identified as a determinant of what a horizon
descriptor can express. ``profile.normalize`` retains vertical amplitude (it removes only the mean);
the scorer then decides:

* ``ncc`` divides by the norm and therefore **discards** amplitude -- two skylines with the same
  shape but different vertical extent score identically;
* ``l1`` and ``l2`` compare sample-by-sample and therefore **retain** it.

Running all three over identical profiles measures that effect directly rather than assuming it. That
is the entire reason the baseline set has three members, and it is why none of them may be dropped
for being "redundant".

``mock`` produces a ranking from a seeded permutation, independent of every curve. It exists to prove
matcher-agnosticism from the producer side: if the evaluator scores a record whose ranking came from
no skyline matching at all, identically in shape to a real one, then no matcher-specific assumption
leaked into the record contract. It is refused on any non-T1 run (``runconfig``), so it can never
touch real evidence.
"""

from __future__ import annotations

import hashlib

import numpy as np

from skyline.descriptors import profile_distance

MOCK = "mock"


class BaselineError(Exception):
    """An unknown or misconfigured scorer."""


def ncc(query: np.ndarray, reference: np.ndarray) -> float:
    """Normalized cross-correlation, in ``[-1, 1]``; 1.0 is identical shape.

    Wraps the published primitive already in the repository (``skyline.descriptors.
    profile_distance`` returns ``1 - NCC``) rather than reimplementing it (FR-009). Amplitude is
    discarded by the norm division -- see the module docstring.
    """
    return float(1.0 - profile_distance(np.asarray(query), np.asarray(reference)))


def l1(query: np.ndarray, reference: np.ndarray) -> float:
    """Negated mean absolute difference; 0.0 is identical, more negative is worse.

    Negated so that, like every scorer here, higher is better. Normalized by sample count so the
    scale does not depend on ``n_samples``. Amplitude is retained.
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    return float(-np.mean(np.abs(q - r)))


def l2(query: np.ndarray, reference: np.ndarray) -> float:
    """Negated RMS difference; 0.0 is identical, more negative is worse. Amplitude is retained."""
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    return float(-np.sqrt(np.mean((q - r) ** 2)))


class MockScorer:
    """A deterministic ranking that ignores the curves entirely (Stage 1 only).

    The score depends on the *identity* of the pair and the run seed, never on curve content, so a
    record it produces exercises the whole emission and evaluation path while carrying no skyline
    information whatsoever.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def __call__(self, query: np.ndarray, reference: np.ndarray) -> float:  # pragma: no cover - unused
        raise BaselineError(
            "the mock scorer ranks by identity, not by curve content; call score_pair() instead"
        )

    def score_pair(self, query_id: str, reference_id: str) -> float:
        h = hashlib.sha256(f"{self.seed}|{query_id}|{reference_id}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / float(1 << 64)


_SCORERS = {"ncc": ncc, "l1": l1, "l2": l2}

#: Scorers that may run against real evidence. ``mock`` is deliberately absent.
REAL_BASELINES = tuple(sorted(_SCORERS))
ALL_BASELINES = REAL_BASELINES + (MOCK,)


def get_scorer(baseline: str):
    """Return the scoring callable for ``baseline``. ``mock`` is handled separately by the runner."""
    if baseline == MOCK:
        raise BaselineError(
            "the mock baseline does not score curves; the runner must use MockScorer.score_pair"
        )
    try:
        return _SCORERS[baseline]
    except KeyError:
        raise BaselineError(
            f"unknown baseline {baseline!r}; expected one of {ALL_BASELINES}"
        ) from None
