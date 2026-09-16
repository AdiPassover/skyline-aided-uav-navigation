"""hsreloc.retrieval -- the skyline matcher stage (SKY lane, **research-grade**).

A query skyline curve in; a ranked shortlist of georeferenced reference skylines out; an acceptance
decision; and, where accepted, the matched reference's coordinate as a **coarse georeferenced fix**.
The result is emitted as the frozen spec-006 generic result record and scored by the **unmodified**
spec-006 evaluator (``naveval``) -- this package computes no metrics of its own.

Maturity label (Principle VII / spec FR-028): **research-grade**. Nothing here has been measured on
target UAV hardware; no target platform is specified repo-wide, so every emitted record carries
``environment.is_target_hardware = false`` and no "lightweight" or "real-time" claim is available.

What this package deliberately does NOT contain
-----------------------------------------------
* **Automatic skyline extraction.** Curves arrive through a ``CurveSource``; nothing here opens an
  image. A future automatic extractor becomes another ``CurveSource`` and this package does not
  change (``contracts/skyline-curve-source.md``).
* **Ground truth.** The matcher never sees ``in_coverage``, ``groundtruth.csv``, or any GT position
  or heading. Those are the evaluator's, and the separation is what makes a result attributable.
* **Metrics.** ``naveval.skyline`` owns those, unmodified.
* Sequence matching, VO-prior-constrained retrieval, learned descriptors, a production index -- all
  deferred to later specs.

Reuse boundary (``DEC-012``): importing ``hsreloc`` puts ``evaluation/`` (-> ``naveval``) and
``skyline/`` (-> ``skyline``) on ``sys.path``, so ``skyline.descriptors`` and
``naveval.frames`` resolve here without nesting or vendoring.

Two stages, and the boundary between them is a hard checkpoint
--------------------------------------------------------------
**Stage 1** proves this whole chain on synthetic and mock data with every real artifact absent (T1).
**Stage 2** is the first run against the real spec-007 Nordland oracle data (T3) and requires explicit
author authorization. Tier is never pooled across the two.
"""

from __future__ import annotations

__all__ = [
    "acceptance",
    "baselines",
    "fix",
    "profile",
    "rank",
    "record",
    "refset",
    "run",
    "runconfig",
    "skyline_curve",
    "sources",
]
