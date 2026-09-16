"""hsreloc.extraction — automatic skyline extraction: benchmark GT, methods, evaluation, audit.

Decisions: ``DEC-SKY-002``
(benchmark adoption), ``DEC-SKY-003`` (POC method choice). Surveys: ``LIT-SKY-001`` (benchmarks),
``LIT-SKY-002`` (method families).

This package is a **producer** for the matcher-facing seam of ``DEC-SKY-001``: it turns images into
skyline curves and can store them as ``skylines_auto/<observation_id>.csv`` for the
``AutomaticCurveSource`` defined here. The matcher (``hsreloc.retrieval``) is a **consumer** and is
not modified by anything in this package — that boundary is the point.

Hard rules inherited from the feature spec:

- extractor development is judged against **independent external ground truth** only; downstream
  retrieval metrics never appear in the development loop (spec FR-009);
- an extractor that cannot decide emits an explicit failure or invalid columns, never a fabricated
  boundary (FR-013);
- the Nordland oracle audit (``hsreloc.extraction.audit``) is strictly **read-only** over the
  spec-007 store and is diagnostic, never statistical (FR-018..021).

Importing ``hsreloc`` bootstraps ``sys.path`` for the reused roots (``naveval``, ``skyline``); this
subpackage relies on that, adding nothing to the path itself.
"""
