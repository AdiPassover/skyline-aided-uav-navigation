"""ECL skyline place retrieval bench (SKY lane, feature 20260825-215212; EXP-SKY-006).

Pose-only reference/query design over the ECL index (``frame``, ``split``, ``sets``), two frozen
skyline sources behind the unchanged ``CurveSource`` seam (``sources``), a runner that drives the
**unchanged** ``hsreloc.retrieval`` matcher with the source as the only variable (``runner``,
``prereg``), a reporting layer pinned to the **unchanged** spec-006 evaluator (``report``), and
post-hoc diagnostics and renders (``diagnostics``, ``render``).

Binding restrictions: ``DEC-SKY-006`` R1–R8 (cross-traversal primary in two scenes; grids at
10 / 25 / 50 SfM m; declared synthetic frame; SfM-metre scale unverified; no UAV claim), and the
feature's STOP list (no extractor tuning, no learned descriptor, no thesis claim). ``hsreloc/retrieval``
is a dependency, never an edit target.
"""
